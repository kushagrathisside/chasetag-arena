"""Record a logged play session in the research ledger, and check the play log a player submits.

A logged session has one participant at the keyboard seat, seat 0, and a bot on seat 1. Its rounds
are written as the game loop plays them: the session, each round, the two seats of each round, and
one trajectory file per round. A round is written in one transaction with its seats, so a failed
round leaves nothing behind.

The play log is the session's manifest: the session and its rounds, as the store holds them. The
player submits it, and ``submit_play_log`` accepts it only when it matches the store exactly and
every trajectory it names verifies.

The ledger writer is ``ledger_writer.py``, a copy of the one in groundtag. Its migration is not
in this repository, so the caller passes the migration's path.

Each call that writes opens its own connection. The game thread writes rounds while the server's
thread may end the session, and a SQLite connection must not be shared between threads.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from jax import Array

from chasetag_arena import ledger_writer as lw
from chasetag_arena.arcade.game_loop import BODY_LABEL
from chasetag_arena.core.families.base import BodyParams
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.spec.rules import OFFICIAL
from chasetag_arena.core.state import FALL, TAG, TIMEOUT, GameState

PURPOSE = "research_logs"
RULES_NAME = "official-2d"
PLAY_LOG_FORMAT = "chasetag-play-log"
PLAY_LOG_VERSION = 1
VALIDATOR_VERSION = "play-log-v1"
ENDED_BY = {TAG: "tag", FALL: "fall", TIMEOUT: "timeout"}
REPO = "chasetag-arena"
REPO_ROOT = Path(__file__).resolve().parents[2]


def open_store(root: str | Path, migration: Path) -> lw.Ledger:
    """Open the store at ``root``, creating it from ``migration`` if it is new."""
    return lw.Ledger.open(root, migration)


def check_seats(selectors: Sequence[str]) -> None:
    """A logged session has the keyboard on seat 0 and a bot on seat 1. Others are refused."""
    if len(selectors) != 2 or selectors[0] != "keyboard" or selectors[1] in ("keyboard", "planner"):
        raise lw.LedgerError("a logged session needs the keyboard on seat 0 and a bot on seat 1")


def participant_for(ledger: lw.Ledger, pseudonym: str) -> int:
    """The participant id for a pseudonym. An erased participant cannot play a logged session."""
    row = ledger.conn.execute(
        "SELECT id, erased_at FROM participant WHERE pseudonym = ?", (pseudonym,)
    ).fetchone()
    if row is None:
        raise lw.LedgerError("no participant has that pseudonym")
    if row["erased_at"] is not None:
        raise lw.LedgerError("that participant has been erased")
    return int(row["id"])


def _canonical(value: Any) -> bytes:
    """JSON with sorted keys and no spaces, so equal values have equal bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _row(conn: Any, table: str, **values: Any) -> int:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cursor = conn.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values())
    )
    return int(cursor.lastrowid or 0)


def rules_name(episode_s: float) -> str:
    """The official 20 s rules keep their name. Any other round length names itself."""
    if episode_s == OFFICIAL.episode_s:
        return RULES_NAME
    return f"arena-2d-{int(episode_s)}s"


def rules_id(ledger: lw.Ledger, episode_s: float) -> int:
    """The id of the 2-D rules with this round length. A new length is a new rules row."""
    rules = replace(OFFICIAL, episode_s=episode_s)
    return ledger.ensure_rules_version(
        rules_name(episode_s),
        "arena-2d",
        episode_s=rules.episode_s,
        control_dt=rules.control_dt,
        decision_substeps=rules.decision_substeps,
        tag_radius_m=rules.tag_radius_m,
        fall_height_m=rules.fall_height_m,
        start_separation_m=rules.start_separation_m,
        arena_half_m=rules.arena_half_m,
        max_speed_mps=rules.max_speed_mps,
    )


def body_profile_id(ledger: lw.Ledger, params: BodyParams | None = None) -> int:
    """The id of the default body. Its parameters are stored as a file, and its hash is kept.

    The name and version identify the body. If the default parameters change, the version must
    change too, so a changed body under an old name is refused rather than recorded as the same.
    """
    params = default_params() if params is None else params
    theta = {
        f.name: np.asarray(getattr(params.theta, f.name)).tolist() for f in fields(params.theta)
    }
    content = _canonical(theta)
    digest = lw.sha256_hex(content)
    with ledger.transaction() as conn:
        row = conn.execute(
            "SELECT id, content_hash FROM body_profile WHERE name = ? AND version = 1",
            (BODY_LABEL,),
        ).fetchone()
        if row is not None:
            if row["content_hash"] != digest:
                raise lw.LedgerError(f"body {BODY_LABEL!r} v1 holds other parameters")
            return int(row["id"])
        artifact = ledger._store(conn, content, kind="body_profile", format="json")
        return _row(
            conn,
            "body_profile",
            family="K",
            name=BODY_LABEL,
            version=1,
            tier="default",
            source="preset",
            content_hash=digest,
            params_artifact_id=artifact,
            produced_by_run_id=None,
            created_at=lw.utc_now(),
        )


def _controller(conn: Any, selector: str, participant_id: int, code_version_id: int) -> int:
    """The controller row for a seat: the participant's keyboard, or a bot in this code."""
    if selector == "keyboard":
        row = conn.execute(
            "SELECT id FROM controller WHERE kind = 'keyboard' AND participant_id = ?",
            (participant_id,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
        return _row(conn, "controller", kind="keyboard", participant_id=participant_id)
    row = conn.execute(
        "SELECT id FROM controller WHERE kind = 'bot' AND bot_name = ? AND code_version_id = ? "
        "AND walker_policy_id IS NULL",
        (selector, code_version_id),
    ).fetchone()
    if row is not None:
        return int(row["id"])
    return _row(conn, "controller", kind="bot", bot_name=selector, code_version_id=code_version_id)


def _controller_name(row: Any) -> str:
    return "keyboard" if row["kind"] == "keyboard" else str(row["bot_name"])


def _frame(state: GameState, actions: Array | None = None) -> dict[str, Any]:
    """One decision's record: both bodies' pelvis positions and headings, and the falls so far.

    Positions and headings are rounded to six decimals, which is finer than any decision.
    """
    frame: dict[str, Any] = {
        "step": int(np.asarray(state.step)),
        "pos": np.asarray(state.bodies.pos, dtype=float).round(6).tolist(),
        "heading": np.asarray(state.bodies.heading, dtype=float).round(6).tolist(),
        "fallen": [bool(x) for x in np.asarray(state.bodies.fallen)],
    }
    if actions is not None:
        frame["actions"] = np.asarray(actions, dtype=float).round(6).tolist()
    return frame


@dataclass
class _OpenRound:
    round_no: int
    controllers: tuple[int, int]
    frames: list[dict[str, Any]]


class Recording:
    """Writes one logged play session, round by round, as the game loop plays it.

    Construct it before the game loop, and pass it to ``GameLoop(recorder=...)``. The session is
    created here, so each round can refer to it. A session is refused unless its participant has
    research consent in force, and unless its seats are the keyboard on seat 0 and a bot on seat 1.
    Planner seats are refused, since a planner's policy is not recorded yet.
    """

    def __init__(
        self,
        store: str | Path,
        migration: Path,
        *,
        participant_id: int,
        seed: int,
        selectors: Sequence[str],
        episode_s: float,
        code_checkout: Path,
    ) -> None:
        check_seats(selectors)
        if not 0 <= seed <= 4294967295:
            raise lw.LedgerError(f"seed {seed} is outside the 32-bit range")
        self._store = Path(store)
        self._migration = migration
        self._participant = participant_id
        self._open: _OpenRound | None = None
        self.rounds_recorded = 0
        with self._connect() as ledger:
            self._require_consent(ledger, participant_id)
            self._code = ledger.code_version(REPO, code_checkout)
            self._rules = rules_id(ledger, episode_s)
            self._body = body_profile_id(ledger)
            self.session_id = self._start_session(ledger, seed)

    @contextmanager
    def _connect(self) -> Iterator[lw.Ledger]:
        ledger = lw.Ledger.open(self._store, self._migration)
        try:
            yield ledger
        finally:
            ledger.close()

    def _require_consent(self, ledger: lw.Ledger, participant_id: int) -> None:
        row = ledger.conn.execute(
            "SELECT ce.event FROM consent_event ce "
            "JOIN consent_version cv ON cv.id = ce.consent_version_id "
            "JOIN participant p ON p.id = ce.participant_id "
            "WHERE ce.participant_id = ? AND cv.purpose = ? AND p.erased_at IS NULL "
            "ORDER BY ce.id DESC LIMIT 1",
            (participant_id, PURPOSE),
        ).fetchone()
        if row is None or row["event"] != "granted":
            raise lw.LedgerError("this participant has no research consent in force")

    def _start_session(self, ledger: lw.Ledger, seed: int) -> int:
        with ledger.transaction() as conn:
            return _row(
                conn,
                "play_session",
                participant_id=self._participant,
                code_version_id=self._code,
                rules_version_id=self._rules,
                body_profile_id=self._body,
                seed=seed,
                logged=1,
                started_at=lw.utc_now(),
            )

    def finish(self) -> None:
        """Mark the session ended. Its rounds are already written."""
        with self._connect() as ledger, ledger.transaction() as conn:
            conn.execute(
                "UPDATE play_session SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (lw.utc_now(), self.session_id),
            )

    # --- the round observer the game loop calls ---------------------------------------

    def round_started(self, round_no: int, selectors: tuple[str, str], state: GameState) -> None:
        check_seats(selectors)
        with self._connect() as ledger, ledger.transaction() as conn:
            controllers = tuple(
                _controller(conn, name, self._participant, self._code) for name in selectors
            )
        self._open = _OpenRound(round_no, (controllers[0], controllers[1]), [_frame(state)])

    def decision(self, round_no: int, state: GameState, actions: Array) -> None:
        self._require_open(round_no).frames.append(_frame(state, actions))

    def round_ended(self, round_no: int, state: GameState, scores: Array, ended_by: int) -> None:
        open_round = self._require_open(round_no)
        step = int(np.asarray(state.step))
        payload = _canonical(
            {
                "round_no": round_no,
                "chaser_seat": int(np.asarray(state.chaser)),
                "ended_by": ENDED_BY[int(ended_by)],
                "frames": open_round.frames,
            }
        )
        fell = [int(bool(x)) for x in np.asarray(state.bodies.fallen)]
        with self._connect() as ledger, ledger.transaction() as conn:
            trajectory = ledger._store(conn, payload, kind="round_trajectory", format="json")
            round_id = _row(
                conn,
                "round",
                origin="arena",
                play_session_id=self.session_id,
                round_index=round_no,
                provenance_complete=1,
                seed=None,
                body_profile_id=self._body,
                rules_version_id=self._rules,
                code_version_id=self._code,
                chaser_seat=int(np.asarray(state.chaser)),
                ended_by=ENDED_BY[int(ended_by)],
                t_event_s=step * OFFICIAL.control_dt,
                decisions=-(-step // OFFICIAL.decision_substeps),
                trajectory_artifact_id=trajectory,
                created_at=lw.utc_now(),
            )
            for seat in (0, 1):
                _row(
                    conn,
                    "round_seat",
                    round_id=round_id,
                    seat=seat,
                    controller_id=open_round.controllers[seat],
                    score=float(scores[seat]),
                    fell=fell[seat],
                )
        self._open = None
        self.rounds_recorded += 1

    def _require_open(self, round_no: int) -> _OpenRound:
        if self._open is None or self._open.round_no != round_no:
            raise lw.LedgerError(f"round {round_no} was not started in this session")
        return self._open


# --- the play log ---------------------------------------------------------------------


def _round_record(conn: Any, row: Any) -> dict[str, Any]:
    seats = conn.execute(
        "SELECT rs.seat, rs.score, rs.fell, c.kind, c.bot_name FROM round_seat rs "
        "JOIN controller c ON c.id = rs.controller_id WHERE rs.round_id = ? ORDER BY rs.seat",
        (row["id"],),
    ).fetchall()
    trajectory = conn.execute(
        "SELECT sha256 FROM artifact WHERE id = ?", (row["trajectory_artifact_id"],)
    ).fetchone()
    return {
        "round_index": int(row["round_index"]),
        "ended_by": str(row["ended_by"]),
        "t_event_s": float(row["t_event_s"]),
        "decisions": int(row["decisions"]),
        "chaser_seat": int(row["chaser_seat"]),
        "seats": [
            {
                "seat": int(s["seat"]),
                "controller": _controller_name(s),
                "score": float(s["score"]),
                "fell": int(s["fell"]),
            }
            for s in seats
        ],
        "trajectory_sha256": str(trajectory["sha256"]),
    }


def manifest(ledger: lw.Ledger, session_id: int) -> dict[str, Any]:
    """The session and its rounds, as the store holds them. Raises if it is unknown or running."""
    conn = ledger.conn
    session = conn.execute(
        "SELECT id, seed, started_at, ended_at FROM play_session WHERE id = ?", (session_id,)
    ).fetchone()
    if session is None:
        raise lw.LedgerError(f"there is no play session {session_id}")
    if session["ended_at"] is None:
        raise lw.LedgerError(f"play session {session_id} has not finished")
    rows = conn.execute(
        "SELECT * FROM round WHERE play_session_id = ? ORDER BY round_index", (session_id,)
    ).fetchall()
    return {
        "format": PLAY_LOG_FORMAT,
        "version": PLAY_LOG_VERSION,
        "session": {
            "id": int(session["id"]),
            "seed": int(session["seed"]),
            "started_at": str(session["started_at"]),
            "ended_at": str(session["ended_at"]),
        },
        "rounds": [_round_record(conn, row) for row in rows],
    }


def build_manifest(ledger: lw.Ledger, session_id: int) -> bytes:
    """The play log for a finished session, as the bytes the player submits."""
    return _canonical(manifest(ledger, session_id))


@dataclass(frozen=True)
class Submission:
    id: int
    status: str
    rejection_reason: str | None


def _problem(ledger: lw.Ledger, artifact_id: int, participant_id: int) -> str | None:
    """Why an uploaded play log is refused, or None when it matches the store."""
    try:
        document = json.loads(ledger.read_object(artifact_id))
    except (ValueError, lw.LedgerError) as err:
        return f"the file cannot be read: {err}"
    if not isinstance(document, dict) or document.get("format") != PLAY_LOG_FORMAT:
        return "the file is not a play log"
    if document.get("version") != PLAY_LOG_VERSION:
        return "the play log has another version"
    try:
        session_id = int(document["session"]["id"])
        expected = manifest(ledger, session_id)
    except (KeyError, TypeError, ValueError, lw.LedgerError) as err:
        return f"the play log does not name a finished session: {err}"
    owner = ledger.conn.execute(
        "SELECT participant_id FROM play_session WHERE id = ?", (session_id,)
    ).fetchone()
    if owner["participant_id"] != participant_id:
        return "the session belongs to another participant"
    if document != expected:
        return "the play log does not match the store"
    trajectories = ledger.conn.execute(
        "SELECT trajectory_artifact_id FROM round WHERE play_session_id = ?", (session_id,)
    ).fetchall()
    for row in trajectories:
        if ledger.verify_artifact(int(row["trajectory_artifact_id"])) != "verified":
            return "a trajectory is missing or changed"
    return None


def submit_play_log(
    ledger: lw.Ledger, *, participant_id: int, consent_version_id: int, data: bytes
) -> Submission:
    """Store an uploaded play log, check it against the store, and record the decision."""
    with ledger.transaction() as conn:
        artifact = ledger._store(conn, data, kind="submission_upload", format="json")
        submission_id = _row(
            conn,
            "submission",
            kind="play_log",
            artifact_id=artifact,
            participant_id=participant_id,
            consent_version_id=consent_version_id,
            status="received",
            received_at=lw.utc_now(),
        )
    reason = _problem(ledger, artifact, participant_id)
    now = lw.utc_now()
    with ledger.transaction() as conn:
        if reason is None:
            conn.execute(
                "UPDATE submission SET status = 'accepted', validator_version = ?, decided_at = ? "
                "WHERE id = ?",
                (VALIDATOR_VERSION, now, submission_id),
            )
            return Submission(submission_id, "accepted", None)
        conn.execute(
            "UPDATE submission SET status = 'rejected', validator_version = ?, "
            "rejection_reason = ?, decided_at = ? WHERE id = ?",
            (VALIDATOR_VERSION, reason, now, submission_id),
        )
    return Submission(submission_id, "rejected", reason)


# --- participants -----------------------------------------------------------------------


def enrol(
    ledger: lw.Ledger,
    *,
    consent_label: str,
    consent_text: bytes,
    effective_at: str | None = None,
) -> tuple[int, str, int]:
    """Create a participant who has granted research consent. Returns (participant id,
    pseudonym, consent version id)."""
    when = effective_at or lw.utc_now()
    consent = ledger.ensure_consent_version(consent_label, PURPOSE, consent_text, when)
    participant, pseudonym = ledger.new_participant()
    ledger.record_consent(participant, consent, "granted", when)
    return participant, pseudonym, consent
