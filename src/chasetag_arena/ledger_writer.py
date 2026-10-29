"""The writer for the research ledger. Each repository that writes to the store carries a copy.

The schema is ``migrations/0001_ledger.sql``, beside this file. The module uses the standard
library only, so a copy needs nothing else. Keep the copies identical. Each store records the
SHA-256 of the migration it was built from, and ``Ledger.open`` refuses a store whose recorded
hash differs from the migration beside this file.

What the writer keeps:

- A result is added once and never changed. A correction is a new row. The only UPDATEs are the
  ones the schema allows: filling in the migration hash, erasing a participant, ending a segment,
  and finishing a run.
- A file is stored under the SHA-256 of its uncompressed bytes, and is never overwritten.
  Compressible formats are gzipped; archives and video are stored as they are.
- Each write runs in one ``BEGIN IMMEDIATE`` transaction. If it fails, none of its rows are kept.
  A file written by a failed transaction stays on disk, unreferenced, and is safe to delete.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import secrets
import socket
import sqlite3
import subprocess
import zlib
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIGRATION = Path(__file__).resolve().parent / "migrations" / "0001_ledger.sql"
SCHEMA_VERSION = 1
REPOS = ("groundtag", "chasetag-arena", "rl-experiment")
GZIP_LEVEL = 6
UNCOMPRESSED_FORMATS = frozenset({"mp4", "zip", "tgz"})


class LedgerError(Exception):
    """A write that the writer refuses before it reaches the database."""


class SchemaMismatch(LedgerError):
    """The store was built from a different migration than the one beside this writer."""


def utc_now() -> str:
    """The current time in the form the schema accepts, such as 2026-10-28T17:04:29Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    """JSON with sorted keys and no spaces, so equal values have equal bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _insert(conn: sqlite3.Connection, table: str, **values: Any) -> int:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cursor = conn.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values())
    )
    return int(cursor.lastrowid or 0)


def _event(conn: sqlite3.Connection, artifact_id: int, event: str, note: str = "") -> None:
    _insert(conn, "artifact_event", artifact_id=artifact_id, event=event, at=utc_now(), note=note)


def _last_event(conn: sqlite3.Connection, artifact_id: int) -> str | None:
    row = conn.execute(
        "SELECT event FROM artifact_event WHERE artifact_id = ? ORDER BY id DESC LIMIT 1",
        (artifact_id,),
    ).fetchone()
    return None if row is None else str(row["event"])


def _decode(payload: bytes, compressed: bool) -> bytes:
    return gzip.decompress(payload) if compressed else payload


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write a file under its final name only once it is complete. An existing file is kept."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    with temp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _git(checkout: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(checkout), *args], check=True, capture_output=True
    ).stdout


def _running_segment(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row:
    """The latest segment of a run that is still running. Anything else is refused."""
    run = conn.execute("SELECT status FROM training_run WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise LedgerError(f"there is no training run {run_id}")
    if run["status"] != "running":
        raise LedgerError(f"training run {run_id} is {run['status']}; it takes no new results")
    row = conn.execute(
        "SELECT id, segment_no FROM run_segment WHERE run_id = ? ORDER BY segment_no DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return row


class Ledger:
    """An open connection to one store. Many processes may share a store, one connection each."""

    def __init__(self, root: Path, conn: sqlite3.Connection) -> None:
        self.root = root
        self.conn = conn

    @classmethod
    def open(cls, root: str | Path, migration: Path = MIGRATION) -> Ledger:
        """Open the store at ``root``. A new store gets the migration; an old one is checked."""
        root = Path(root).expanduser()
        (root / "objects").mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(root / "ledger.sqlite", isolation_level=None, timeout=30)
        conn.row_factory = sqlite3.Row
        for pragma in (
            "foreign_keys = ON",
            "recursive_triggers = ON",
            "busy_timeout = 30000",
            "journal_mode = WAL",
            "synchronous = NORMAL",
        ):
            conn.execute(f"PRAGMA {pragma}")
        ledger = cls(root, conn)
        try:
            ledger._migrate(migration)
        except BaseException:
            conn.close()
            raise
        return ledger

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Ledger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transactions

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run the enclosed writes as one ``BEGIN IMMEDIATE`` transaction. Writes do not nest."""
        if self.conn.in_transaction:
            raise LedgerError("a transaction is already open; writes do not nest")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except BaseException:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def _migrate(self, migration: Path) -> None:
        digest = sha256_hex(migration.read_bytes())
        if not self._has_table("schema_migrations"):
            self._apply(migration)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT sha256 FROM schema_migrations WHERE version = ?", (SCHEMA_VERSION,)
            ).fetchone()
            if row is None:
                raise SchemaMismatch("the store has no record of migration 1")
            if row["sha256"] is None:
                # A migration cannot record its own hash, so the first open after it writes it.
                conn.execute(
                    "UPDATE schema_migrations SET sha256 = ? WHERE version = ?",
                    (digest, SCHEMA_VERSION),
                )
            elif row["sha256"] != digest:
                raise SchemaMismatch(
                    f"the store was built from migration {row['sha256'][:12]}, "
                    f"this writer has {digest[:12]}"
                )

    def _has_table(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    def _apply(self, migration: Path) -> None:
        """Run the migration. It commits its own transaction; a failure rolls that back."""
        try:
            self.conn.executescript(migration.read_text(encoding="utf-8"))
        except sqlite3.Error:
            if self.conn.in_transaction:
                self.conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ files

    def object_path(self, digest: str, format: str, compressed: bool) -> Path:
        """Where a file lives: objects/<first two hex digits>/<sha256>.<format>[.gz]."""
        name = f"{digest}.{format}" + (".gz" if compressed else "")
        return self.root / "objects" / digest[:2] / name

    def store_object(
        self,
        data: bytes,
        *,
        kind: str,
        format: str,
        code_version_id: int | None = None,
        schema_version: int | None = None,
        compress: bool | None = None,
    ) -> int:
        """Store a file and return its artifact id.

        A file is identified by the SHA-256 of its uncompressed bytes. Storing bytes that are
        already stored returns the existing artifact, provided its kind and format match.
        Compression is on by default, except for the formats in ``UNCOMPRESSED_FORMATS``.
        """
        with self.transaction() as conn:
            return self._store(
                conn,
                data,
                kind=kind,
                format=format,
                code_version_id=code_version_id,
                schema_version=schema_version,
                compress=compress,
            )

    def _store(
        self,
        conn: sqlite3.Connection,
        data: bytes,
        *,
        kind: str,
        format: str,
        code_version_id: int | None = None,
        schema_version: int | None = None,
        compress: bool | None = None,
    ) -> int:
        digest = sha256_hex(data)
        existing = conn.execute(
            "SELECT id, sha256, kind, format, compression FROM artifact WHERE sha256 = ?",
            (digest,),
        ).fetchone()
        if existing is not None:
            return self._reuse(conn, existing, data, kind=kind, format=format)
        if compress is None:
            compress = format not in UNCOMPRESSED_FORMATS
        payload = gzip.compress(data, compresslevel=GZIP_LEVEL, mtime=0) if compress else data
        _write_atomic(self.object_path(digest, format, compress), payload)
        artifact_id = _insert(
            conn,
            "artifact",
            sha256=digest,
            kind=kind,
            format=format,
            bytes=len(data),
            schema_version=schema_version,
            code_version_id=code_version_id,
            created_at=utc_now(),
            compression="gzip" if compress else "none",
            stored_bytes=len(payload) if compress else None,
        )
        _event(conn, artifact_id, "stored")
        return artifact_id

    def _reuse(
        self, conn: sqlite3.Connection, row: sqlite3.Row, data: bytes, *, kind: str, format: str
    ) -> int:
        artifact_id = int(row["id"])
        if row["kind"] != kind or row["format"] != format:
            raise LedgerError(
                f"these bytes are artifact {artifact_id} as {row['kind']}/{row['format']}, "
                f"not {kind}/{format}"
            )
        if _last_event(conn, artifact_id) == "erased":
            raise LedgerError(f"artifact {artifact_id} was erased; its content is not stored again")
        gzipped = row["compression"] == "gzip"
        payload = gzip.compress(data, compresslevel=GZIP_LEVEL, mtime=0) if gzipped else data
        # A file that was deleted from disk is written again from the bytes just given.
        _write_atomic(self.object_path(row["sha256"], format, gzipped), payload)
        return artifact_id

    def read_object(self, artifact_id: int) -> bytes:
        """Return the uncompressed bytes of a stored file, checked against its hash."""
        row = _artifact(self.conn, artifact_id)
        if _last_event(self.conn, artifact_id) == "erased":
            raise LedgerError(f"artifact {artifact_id} was erased")
        try:
            data = _decode(self._path_of(row).read_bytes(), row["compression"] == "gzip")
        except (OSError, EOFError, zlib.error) as err:
            raise LedgerError(
                f"artifact {artifact_id} cannot be read; run verify_artifact"
            ) from err
        if sha256_hex(data) != row["sha256"]:
            raise LedgerError(
                f"artifact {artifact_id} does not match its hash; run verify_artifact"
            )
        return data

    def verify_artifact(self, artifact_id: int) -> str:
        """Check a stored file against its hash, record the outcome, and return it.

        The outcome is ``verified``, ``missing`` or ``hash_mismatch``. An erased artifact is
        reported as ``erased`` and nothing is recorded for it.
        """
        with self.transaction() as conn:
            row = _artifact(conn, artifact_id)
            if _last_event(conn, artifact_id) == "erased":
                return "erased"
            outcome = self._check(row)
            _event(conn, artifact_id, outcome)
            return outcome

    def _check(self, row: sqlite3.Row) -> str:
        path = self._path_of(row)
        if not path.exists():
            return "missing"
        try:
            data = _decode(path.read_bytes(), row["compression"] == "gzip")
        except (OSError, EOFError, zlib.error):
            return "hash_mismatch"
        return "verified" if sha256_hex(data) == row["sha256"] else "hash_mismatch"

    def _path_of(self, row: sqlite3.Row) -> Path:
        return self.object_path(row["sha256"], row["format"], row["compression"] == "gzip")

    # ------------------------------------------------------------------ provenance

    def code_version(self, repo: str, checkout: str | Path) -> int:
        """Record a checkout's commit and its uncommitted state, and return the version id.

        A clean checkout is identified by its commit. A dirty one is identified by a fingerprint
        as well: the diff of tracked files, the status list, and the content of every untracked
        file. Any change to those gives a new version.
        """
        checkout = Path(checkout)
        commit = _git(checkout, "rev-parse", "HEAD").decode().strip()
        status = _git(checkout, "status", "--porcelain", "-z")
        if not status:
            return self.ensure_code_version(repo, commit, dirty=False)
        parts = [_git(checkout, "diff", "HEAD", "--binary"), status]
        untracked = _git(checkout, "ls-files", "--others", "--exclude-standard", "-z")
        for name in filter(None, untracked.split(b"\0")):
            parts.append(name + b"\0" + (checkout / name.decode()).read_bytes())
        fingerprint = sha256_hex(b"\0".join(parts))
        return self.ensure_code_version(repo, commit, dirty=True, diff_sha256=fingerprint)

    def ensure_code_version(
        self, repo: str, commit_sha: str, *, dirty: bool, diff_sha256: str = ""
    ) -> int:
        """Return the id of a code version, adding it if it is new."""
        if repo not in REPOS:
            raise LedgerError(f"unknown repository {repo!r}")
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM code_version WHERE repo = ? AND commit_sha = ? AND diff_sha256 = ?",
                (repo, commit_sha, diff_sha256),
            ).fetchone()
            if row is not None:
                return int(row["id"])
            return _insert(
                conn,
                "code_version",
                repo=repo,
                commit_sha=commit_sha,
                dirty=int(dirty),
                diff_sha256=diff_sha256,
                recorded_at=utc_now(),
            )

    def ensure_rules_version(
        self,
        name: str,
        environment: str,
        *,
        episode_s: float,
        control_dt: float,
        decision_substeps: int,
        tag_radius_m: float,
        fall_height_m: float,
        start_separation_m: float,
        arena_half_m: float,
        max_speed_mps: float,
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Return the id of a rules set, adding it if its values are new.

        Rules are identified by their values, so the same values give the same row whatever
        they are called. New values under a name that is already taken are refused.
        """
        fields = {
            "episode_s": float(episode_s),
            "control_dt": float(control_dt),
            "decision_substeps": int(decision_substeps),
            "tag_radius_m": float(tag_radius_m),
            "fall_height_m": float(fall_height_m),
            "start_separation_m": float(start_separation_m),
            "arena_half_m": float(arena_half_m),
            "max_speed_mps": float(max_speed_mps),
        }
        extra = dict(extra or {})
        digest = sha256_hex(_canonical({"environment": environment, **fields, "extra": extra}))
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT id FROM rules_version WHERE content_sha256 = ?", (digest,)
            ).fetchone()
            if row is not None:
                return int(row["id"])
            taken = conn.execute("SELECT 1 FROM rules_version WHERE name = ?", (name,)).fetchone()
            if taken is not None:
                raise LedgerError(f"rules name {name!r} is taken by other rules; choose a name")
            return _insert(
                conn,
                "rules_version",
                name=name,
                environment=environment,
                **fields,
                extra_json=_canonical(extra).decode(),
                content_sha256=digest,
                created_at=utc_now(),
            )

    # ------------------------------------------------------------------ people and consent

    def ensure_consent_version(
        self, label: str, purpose: str, text: bytes, effective_at: str
    ) -> int:
        """Return the id of a consent version, adding it with its text if the label is new.

        A label keeps the text it was first given. The same label with other text is refused.
        """
        digest = sha256_hex(text)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT cv.id, a.sha256 FROM consent_version cv "
                "JOIN artifact a ON a.id = cv.text_artifact_id WHERE cv.label = ?",
                (label,),
            ).fetchone()
            if row is not None:
                if row["sha256"] != digest:
                    raise LedgerError(f"consent version {label!r} has other text; use a new label")
                return int(row["id"])
            text_id = self._store(conn, text, kind="consent_text", format="md")
            return _insert(
                conn,
                "consent_version",
                label=label,
                text_artifact_id=text_id,
                purpose=purpose,
                effective_at=effective_at,
            )

    def new_participant(self, display_name: str | None = None) -> tuple[int, str]:
        """Create a participant with a random pseudonym. Returns (participant id, pseudonym)."""
        pseudonym = secrets.token_urlsafe(24)
        with self.transaction() as conn:
            participant_id = _insert(
                conn,
                "participant",
                pseudonym=pseudonym,
                display_name=display_name,
                created_at=utc_now(),
            )
        return participant_id, pseudonym

    def record_consent(
        self, participant_id: int, consent_version_id: int, event: str, at: str | None = None
    ) -> int:
        """Record a grant or a withdrawal. The schema refuses one that makes no sense.

        A refusal raises ``sqlite3.IntegrityError``.
        """
        with self.transaction() as conn:
            return _insert(
                conn,
                "consent_event",
                participant_id=participant_id,
                consent_version_id=consent_version_id,
                event=event,
                at=at or utc_now(),
            )

    def request_erasure(self, participant_id: int, at: str | None = None) -> int:
        with self.transaction() as conn:
            return _insert(
                conn,
                "erasure_event",
                participant_id=participant_id,
                event="requested",
                at=at or utc_now(),
            )

    def complete_erasure(self, participant_id: int, at: str | None = None) -> list[int]:
        """Replace a participant's identity and delete the files that are theirs alone.

        Returns the artifact ids whose files were deleted. A file that another participant also
        refers to is kept. Each file is deleted after the database commits, so a failure in
        between leaves a file that ``purge_erased_files`` removes later.
        """
        when = at or utc_now()
        with self.transaction() as conn:
            requested = conn.execute(
                "SELECT 1 FROM erasure_event WHERE participant_id = ? AND event = 'requested'",
                (participant_id,),
            ).fetchone()
            if requested is None:
                raise LedgerError("an erasure is completed only after it is requested")
            erased = [
                artifact_id
                for artifact_id in _linked_artifacts(conn, participant_id)
                if not _referenced_by_others(conn, artifact_id, participant_id)
                and _last_event(conn, artifact_id) != "erased"
            ]
            conn.execute(
                "UPDATE participant SET pseudonym = ?, display_name = NULL, erased_at = ? "
                "WHERE id = ?",
                (f"erased-{secrets.token_urlsafe(24)}", when, participant_id),
            )
            _insert(
                conn, "erasure_event", participant_id=participant_id, event="completed", at=when
            )
            for artifact_id in erased:
                _event(conn, artifact_id, "erased", note="participant erasure")
        for artifact_id in erased:
            self._path_of(_artifact(self.conn, artifact_id)).unlink(missing_ok=True)
        return erased

    def purge_erased_files(self) -> int:
        """Delete any file of an erased artifact that is still on disk. Returns how many."""
        rows = self.conn.execute(
            "SELECT DISTINCT artifact_id FROM artifact_event WHERE event = 'erased'"
        ).fetchall()
        removed = 0
        for row in rows:
            path = self._path_of(_artifact(self.conn, int(row["artifact_id"])))
            if path.exists():
                path.unlink()
                removed += 1
        return removed

    # ------------------------------------------------------------------ training runs

    def ensure_metric(self, name: str, unit: str, description: str) -> None:
        """Register a metric. A name keeps the unit and description it was first given."""
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT unit, description FROM metric WHERE name = ?", (name,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO metric (name, unit, description) VALUES (?, ?, ?)",
                    (name, unit, description),
                )
            elif (row["unit"], row["description"]) != (unit, description):
                raise LedgerError(f"metric {name!r} is registered with another unit or text")

    def start_run(
        self,
        kind: str,
        name: str,
        *,
        rules_version_id: int,
        config: bytes,
        config_format: str = "yaml",
        code_version_id: int | None = None,
        seed: int | None = None,
        body_profile_id: int | None = None,
        obs_spec: str | None = None,
        parent_run_id: int | None = None,
        host: str | None = None,
    ) -> int:
        """Start a training run with its first segment, and return the run id.

        A run that does not name its code and its seed is stored with ``provenance_complete``
        set to 0. Its missing fields stay empty rather than being guessed.
        """
        with self.transaction() as conn:
            config_id = self._store(
                conn, config, kind="config", format=config_format, code_version_id=code_version_id
            )
            started = utc_now()
            run_id = _insert(
                conn,
                "training_run",
                kind=kind,
                name=name,
                code_version_id=code_version_id,
                provenance_complete=int(code_version_id is not None and seed is not None),
                rules_version_id=rules_version_id,
                body_profile_id=body_profile_id,
                config_artifact_id=config_id,
                seed=seed,
                obs_spec=obs_spec,
                status="running",
                parent_run_id=parent_run_id,
                host=host or socket.gethostname(),
                started_at=started,
            )
            _insert(
                conn,
                "run_segment",
                run_id=run_id,
                segment_no=1,
                resumed_from_checkpoint_id=None,
                started_at=started,
            )
            return run_id

    def resume_segment(self, run_id: int, checkpoint_id: int, at: str | None = None) -> int:
        """Start a new segment of a running run, resumed from one of its own checkpoints."""
        when = at or utc_now()
        with self.transaction() as conn:
            last = _running_segment(conn, run_id)
            conn.execute(
                "UPDATE run_segment SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (when, last["id"]),
            )
            return _insert(
                conn,
                "run_segment",
                run_id=run_id,
                segment_no=int(last["segment_no"]) + 1,
                resumed_from_checkpoint_id=checkpoint_id,
                started_at=when,
            )

    def record_checkpoint(
        self,
        run_id: int,
        step: int,
        data: bytes,
        *,
        format: str = "npz",
        role: str = "periodic",
        code_version_id: int | None = None,
    ) -> int:
        """Store a checkpoint of the current segment and return its id."""
        with self.transaction() as conn:
            segment = _running_segment(conn, run_id)
            artifact_id = self._store(
                conn, data, kind="checkpoint", format=format, code_version_id=code_version_id
            )
            return _insert(
                conn,
                "checkpoint",
                run_id=run_id,
                segment_id=int(segment["id"]),
                step=step,
                role=role,
                artifact_id=artifact_id,
                created_at=utc_now(),
            )

    def record_metrics(self, run_id: int, rows: Iterable[tuple[str, int, float]]) -> int:
        """Record (metric name, step, value) rows for the current segment. Returns the count.

        A value that is NaN or infinite is not stored: an absent row is how a missing value is
        recorded. The metric must already be registered with ``ensure_metric``.
        """
        with self.transaction() as conn:
            segment = _running_segment(conn, run_id)
            written = 0
            for metric_name, step, value in rows:
                if not math.isfinite(value):
                    continue
                conn.execute(
                    "INSERT INTO run_metric (run_id, segment_id, metric_name, step, value) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (run_id, int(segment["id"]), metric_name, step, float(value)),
                )
                written += 1
            return written

    def finish_run(self, run_id: int, status: str, at: str | None = None) -> None:
        """Mark a running run completed, stopped or failed, and end its last segment."""
        if status not in ("completed", "stopped", "failed"):
            raise LedgerError(f"a run cannot finish as {status!r}")
        when = at or utc_now()
        with self.transaction() as conn:
            _running_segment(conn, run_id)
            conn.execute(
                "UPDATE run_segment SET ended_at = ? WHERE run_id = ? AND ended_at IS NULL",
                (when, run_id),
            )
            conn.execute(
                "UPDATE training_run SET status = ?, finished_at = ? WHERE id = ?",
                (status, when, run_id),
            )


def _artifact(conn: sqlite3.Connection, artifact_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM artifact WHERE id = ?", (artifact_id,)).fetchone()
    if row is None:
        raise LedgerError(f"there is no artifact {artifact_id}")
    return row


def _linked_artifacts(conn: sqlite3.Connection, participant_id: int) -> set[int]:
    """The files a participant's own submissions and play rounds point at."""
    rows = conn.execute(
        "SELECT artifact_id AS aid FROM submission WHERE participant_id = ? "
        "UNION "
        "SELECT r.trajectory_artifact_id AS aid FROM round r "
        "JOIN play_session p ON p.id = r.play_session_id "
        "WHERE p.participant_id = ? AND r.trajectory_artifact_id IS NOT NULL",
        (participant_id, participant_id),
    ).fetchall()
    return {int(row["aid"]) for row in rows}


def _referenced_by_others(conn: sqlite3.Connection, artifact_id: int, participant_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM submission WHERE artifact_id = ? AND participant_id IS NOT ? "
        "UNION ALL "
        "SELECT 1 FROM round r JOIN play_session p ON p.id = r.play_session_id "
        "WHERE r.trajectory_artifact_id = ? AND p.participant_id IS NOT ?",
        (artifact_id, participant_id, artifact_id, participant_id),
    ).fetchone()
    return row is not None
