"""Opt-in contribution of a player's own games: consent, recording, review, export, withdrawal.

Nothing is recorded unless the arcade is started with ``--contribute`` and the player accepts the
consent text in the browser. Recorded games stay in a folder on the player's own computer. The
player can review them, stop recording, export a package to share, or withdraw, which deletes
everything this module wrote.

A game is recorded only when exactly one seat is the keyboard, so a second person at the same
keyboard is never recorded. The keyboard seat is labelled ``human``. The file has the same layout as
the bot episodes dataset: a header line, then one JSON object per game, with 13 numbers per row.

    step, x0, y0, x1, y1, height0, height1, heading0, heading1, raw0x, raw0y, raw1x, raw1y

The recorder never raises into the game loop. A failed write stops the recording and is reported
through ``status``.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import secrets
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
from jax import Array

from chasetag_arena import __version__
from chasetag_arena.core.state import FALL, TAG, TIMEOUT, GameState

COLUMNS: Final[tuple[str, ...]] = (
    "step",
    "x0",
    "y0",
    "x1",
    "y1",
    "height0",
    "height1",
    "heading0",
    "heading1",
    "raw0x",
    "raw0y",
    "raw1x",
    "raw1y",
)
ENDED_NAMES: Final[dict[int, str]] = {TAG: "tag", FALL: "fall", TIMEOUT: "timeout"}
FORMAT: Final[str] = "chasetag-human-games"
FORMAT_VERSION: Final[int] = 1
GAMES_FILE: Final[str] = "games.jsonl"
CONSENT_FILE: Final[str] = "consent.json"
EXPORT_DIR: Final[str] = "exports"
DEFAULT_DIR: Final[Path] = Path.home() / ".local" / "share" / "chasetag-arena" / "contributions"
GUIDE_URL: Final[str] = (
    "https://huggingface.co/spaces/chasetag-arena/README/blob/main/CONTRIBUTING.md"
)

# The status of the consent text. Packages made under a draft are marked as such, and are not
# accepted by the project. Change it to "final" only after the text has been reviewed.
CONSENT_STATUS: Final[str] = "draft"
CONSENT_TEXT: Final[str] = """\
What this is. If you agree, this program records the games you play with the keyboard and saves \
them in a folder on your own computer. Nothing leaves your computer unless you choose to share it.

What is recorded. For each round you play: where both bodies are and which way they face, and the \
movement command each seat chose, for every 0.1 seconds; who controlled each seat (you are \
recorded only as "human"); the round length; how the round ended; and the scores.

What is not recorded. Your name, email address, account, location, IP address or device, or \
anything you type. Rounds with two keyboard players are not recorded.

A random contribution ID. The files carry a random ID that is made on your computer and is not \
based on anything about you. It lets you, or the maintainers, find your games if you later ask for \
them to be removed.

Sharing. You can look at everything that was recorded before you decide. If you export and share \
it, it is published on Hugging Face under the CC BY-NC 4.0 license, so anyone may use it for \
noncommercial purposes. Published data can be copied by others, so removing it later cannot undo \
copies that already exist. The way you move can be distinctive to you.

Your choices. You can stop recording at any time. You can withdraw, which deletes everything this \
program recorded and exported on this computer. If you have already shared games, you can ask the \
maintainers to remove them by quoting your contribution ID, but copies made by others cannot be \
recalled.

Who can agree. You must be 18 or older.
"""
CONSENT_VERSION: Final[str] = hashlib.sha256(CONSENT_TEXT.encode("utf-8")).hexdigest()[:12]

EXPORT_README: Final[str] = """\
This package holds games recorded with ChaseTag Arena, with the player's consent.

episodes.jsonl   the games, in the layout of the ChaseTag Arena bot episodes dataset
receipt.json     what the package contains, and the consent text it was recorded under

To contribute the package, follow the contribution guide:
{guide}

Packages recorded under a draft consent text are not accepted. To ask for games to be removed,
quote the contribution ID in receipt.json.
"""


class ContributionError(ValueError):
    """The request cannot be done now, for example an export before any game was recorded."""


def _today() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")


def _round4(values: Any) -> list[float]:
    return [float(v) for v in np.round(np.asarray(values, dtype=float), 4).reshape(-1)]


def _row(state: GameState, actions: Array | None) -> list[float]:
    """One frame: the step, both pelvis positions, heights and headings, and the raw commands."""
    b = state.bodies
    command = [0.0] * 4 if actions is None else _round4(actions)
    return [
        float(int(np.asarray(state.step))),
        *_round4(b.pos),
        *_round4(b.height),
        *_round4(b.heading),
        *command,
    ]


def _label(selector: str) -> str:
    return "human" if selector == "keyboard" else selector


class Contribution:
    """The consent state and the local record of one player. Also the game loop's round observer."""

    # The game loop leaves the seat choices alone for this recorder, since it only watches.
    locks_seats = False
    is_research_ledger = False

    def __init__(self, root: str | Path = DEFAULT_DIR) -> None:
        self._root = Path(root)
        self._lock = threading.Lock()
        self._limit: Callable[[], int] = lambda: 0
        self._recording = False
        self._error: str | None = None
        self._open: dict[str, Any] | None = None
        self._games = 0
        self._decisions = 0
        self._count_existing()

    # --- files -------------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def _games_path(self) -> Path:
        return self._root / GAMES_FILE

    @property
    def _consent_path(self) -> Path:
        return self._root / CONSENT_FILE

    @property
    def _exports(self) -> Path:
        return self._root / EXPORT_DIR

    def _count_existing(self) -> None:
        """Pick up games recorded in an earlier run. Recording itself stays off until consent."""
        self._games = self._decisions = 0
        if not self._games_path.is_file():
            return
        with self._games_path.open(encoding="utf-8") as f:
            next(f, None)
            for line in f:
                game = json.loads(line)
                self._games += 1
                self._decisions += int(game["decisions"])

    def _read_consent(self) -> dict[str, Any] | None:
        if not self._consent_path.is_file():
            return None
        return json.loads(self._consent_path.read_text(encoding="utf-8"))

    def bind(self, limit_s: Callable[[], int]) -> None:
        """Tell the recorder how to read the round limit, once the game loop exists."""
        self._limit = limit_s

    # --- what the page asks ------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            consent = self._read_consent()
            return {
                "enabled": True,
                "consent": {
                    "status": CONSENT_STATUS,
                    "version": CONSENT_VERSION,
                    "text": CONSENT_TEXT,
                },
                "recording": self._recording,
                "contribution_id": None if consent is None else consent["contribution_id"],
                "games": self._games,
                "decisions": self._decisions,
                "error": self._error,
                "folder": str(self._root),
            }

    def grant(self, version: str) -> dict[str, Any]:
        """Start recording, once the player has seen the current consent text."""
        if version != CONSENT_VERSION:
            raise ContributionError("the consent text has changed; please read it again")
        with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            consent = self._read_consent() or {
                "format": FORMAT,
                "contribution_id": secrets.token_hex(8),
                "history": [],
            }
            consent["history"].append(
                {
                    "version": CONSENT_VERSION,
                    "status": CONSENT_STATUS,
                    "text_sha256": hashlib.sha256(CONSENT_TEXT.encode("utf-8")).hexdigest(),
                    "granted_on": _today(),
                }
            )
            self._consent_path.write_text(json.dumps(consent, indent=1) + "\n", encoding="utf-8")
            if not self._games_path.is_file():
                header = {
                    "format": FORMAT,
                    "format_version": FORMAT_VERSION,
                    "columns": list(COLUMNS),
                    "contribution_id": consent["contribution_id"],
                }
                self._games_path.write_text(json.dumps(header) + "\n", encoding="utf-8")
            self._recording = True
            self._error = None
            self._open = None  # the round in progress is not recorded; the next one is
        return self.status()

    def stop(self) -> dict[str, Any]:
        """Stop recording and keep what was recorded. A round in progress is dropped."""
        with self._lock:
            self._recording = False
            self._open = None
        return self.status()

    def withdraw(self) -> dict[str, Any]:
        """Stop recording and delete everything this module wrote: games, consent and exports."""
        with self._lock:
            self._recording = False
            self._open = None
            for path in (self._games_path, self._consent_path):
                path.unlink(missing_ok=True)
            if self._exports.is_dir():
                for package in self._exports.glob("chasetag-contribution-*.zip"):
                    package.unlink(missing_ok=True)
                # If something else is in there, it is not ours to delete.
                with contextlib.suppress(OSError):
                    self._exports.rmdir()
            with contextlib.suppress(OSError):
                self._root.rmdir()
            self._games = self._decisions = 0
            self._error = None
        return self.status()

    def review_text(self) -> bytes:
        """The recorded games exactly as they would be shared."""
        with self._lock:
            if not self._games_path.is_file():
                raise ContributionError("nothing has been recorded yet")
            return self._games_path.read_bytes()

    def export(self) -> dict[str, Any]:
        """Write a package of the recorded games, ready to share, and describe it."""
        with self._lock:
            consent = self._read_consent()
            if consent is None or self._games == 0:
                raise ContributionError("record at least one game before exporting")
            episodes = self._games_path.read_bytes()
            receipt = {
                "format": FORMAT,
                "format_version": FORMAT_VERSION,
                "contribution_id": consent["contribution_id"],
                "consent": consent["history"],
                "consent_status": CONSENT_STATUS,
                "games": self._games,
                "decisions": self._decisions,
                "episodes_sha256": hashlib.sha256(episodes).hexdigest(),
                "exported_on": _today(),
                "software": {"chasetag_arena": __version__},
            }
            self._exports.mkdir(parents=True, exist_ok=True)
            stamp = _today().replace("-", "")
            name = f"chasetag-contribution-{consent['contribution_id']}-{stamp}.zip"
            path = self._exports / name
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("episodes.jsonl", episodes)
                archive.writestr("receipt.json", json.dumps(receipt, indent=1) + "\n")
                archive.writestr("README.txt", EXPORT_README.format(guide=GUIDE_URL))
            return {
                "file": name,
                "path": str(path),
                "games": self._games,
                "decisions": self._decisions,
                "consent_status": CONSENT_STATUS,
                "bytes": path.stat().st_size,
            }

    def package_bytes(self) -> tuple[str, bytes]:
        """The most recent exported package, as (file name, contents)."""
        with self._lock:
            packages = sorted(self._exports.glob("chasetag-contribution-*.zip"))
            if not packages:
                raise ContributionError("export a package first")
            latest = max(packages, key=lambda p: p.stat().st_mtime)
            return latest.name, latest.read_bytes()

    # --- the round observer the game loop calls ----------------------------------------

    def round_started(self, round_no: int, selectors: tuple[str, str], state: GameState) -> None:
        with self._lock:
            self._open = None
            if not self._recording or list(selectors).count("keyboard") != 1:
                return
            self._open = {
                "round_no": round_no,
                "selectors": selectors,
                "limit": int(self._limit()),
                "rows": [_row(state, None)],
            }

    def decision(self, round_no: int, state: GameState, actions: Array) -> None:
        with self._lock:
            if self._open is not None and self._open["round_no"] == round_no:
                self._open["rows"].append(_row(state, actions))

    def round_ended(self, round_no: int, state: GameState, scores: Array, ended_by: int) -> None:
        with self._lock:
            current, self._open = self._open, None
            if current is None or current["round_no"] != round_no or not self._recording:
                return
            frames = current["rows"]
            game = {
                "episode": self._games,
                "round_limit_s": current["limit"],
                "seat0_bot": _label(current["selectors"][0]),
                "seat1_bot": _label(current["selectors"][1]),
                "chaser": int(np.asarray(state.chaser)),
                "ended_by": ENDED_NAMES[int(ended_by)],
                "decisions": len(frames) - 1,
                "scores": [round(float(scores[0]), 6), round(float(scores[1]), 6)],
                "frames": frames,
            }
            try:
                line = json.dumps(game, separators=(",", ":"), allow_nan=False) + "\n"
                with self._games_path.open("a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            except (OSError, ValueError) as err:
                self._recording = False
                self._error = f"{type(err).__name__}: {err}"
                return
            self._games += 1
            self._decisions += game["decisions"]
