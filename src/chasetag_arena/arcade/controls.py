"""Controls from the browser: parsing, and the held keyboard command.

Discrete controls (pause, step, restart, speed) are queued and applied in order by the game
thread. A selector change and a time-limit change are held until the next round starts, so
neither can alter a round in progress. The keyboard command is a held velocity that expires, so
a lost key release cannot keep a body moving.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from chasetag_arena.agents.bots import BOTS
from chasetag_arena.core.spec.rules import OFFICIAL

# A held keyboard command older than this is treated as released.
KEYBOARD_STALE_S: Final[float] = 0.25
# The largest keyboard velocity component, m/s. Diagonal keys stay within the speed limit.
_MAX_KEY_COMPONENT: Final[float] = OFFICIAL.max_speed_mps
_MAX_SPEED_MULTIPLIER: Final[float] = 8.0
SELECTORS: Final[tuple[str, ...]] = ("keyboard", "planner", *BOTS)
# The longest a round may run, in game seconds. 20 s is the official rule set the earlier results
# were recorded under. One minute is the default, and 3, 5 or 10 minutes are the longer choices.
TIME_LIMITS_S: Final[tuple[int, ...]] = (20, 60, 180, 300, 600)
DEFAULT_TIME_LIMIT_S: Final[int] = 60


@dataclass(frozen=True)
class SelectorControl:
    """Choose who plays a seat. Applied at the next round start."""

    seat: int
    value: str


@dataclass(frozen=True)
class KeyboardControl:
    """The held command for a seat, world frame, m/s. Expires after ``KEYBOARD_STALE_S``."""

    seat: int
    cmd: tuple[float, float]


@dataclass(frozen=True)
class PauseControl:
    """Stop making decisions until a play or a step."""


@dataclass(frozen=True)
class PlayControl:
    """Resume decisions."""


@dataclass(frozen=True)
class StepControl:
    """Make exactly one decision while paused."""


@dataclass(frozen=True)
class RestartControl:
    """Start a new round and clear the match tally."""


@dataclass(frozen=True)
class SpeedControl:
    """Game-time speed multiplier: decisions come this many times faster in wall time."""

    value: float


@dataclass(frozen=True)
class TimeLimitControl:
    """The longest a round may run, in game seconds. Applied at the next round start."""

    value: int


type Control = (
    SelectorControl
    | KeyboardControl
    | PauseControl
    | PlayControl
    | StepControl
    | RestartControl
    | SpeedControl
    | TimeLimitControl
)

_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "selector": ("seat", "value"),
    "keyboard": ("seat", "cmd"),
    "pause": (),
    "play": (),
    "step": (),
    "restart": (),
    "speed": ("value",),
    "limit": ("value",),
}


def _seat(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {value!r}")
    return value


def _number(value: object, name: str) -> float:
    # bool is an int subclass, so it is excluded explicitly.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def parse_control(obj: object) -> Control:
    """Validate one JSON control object and return its typed form.

    Unknown kinds and unknown fields raise ``ValueError``. The contract is strict on purpose,
    so a typo in the client shows up as an error and not as a silent no-op.
    """
    if not isinstance(obj, Mapping):
        raise ValueError("control must be a JSON object")
    kind = obj.get("kind")
    if not isinstance(kind, str) or kind not in _FIELDS:
        raise ValueError(f"unknown control kind: {kind!r}")
    extra = set(obj) - {"kind", *_FIELDS[kind]}
    if extra:
        raise ValueError(f"unexpected fields for {kind!r}: {sorted(extra)}")

    if kind == "selector":
        value = obj.get("value")
        if value not in SELECTORS:
            raise ValueError(f"selector must be one of {SELECTORS}, got {value!r}")
        return SelectorControl(seat=_seat(obj.get("seat")), value=str(value))

    if kind == "keyboard":
        cmd = obj.get("cmd")
        if not isinstance(cmd, list | tuple) or len(cmd) != 2:
            raise ValueError("keyboard cmd must be a [vx, vy] pair")
        vx = _number(cmd[0], "cmd vx")
        vy = _number(cmd[1], "cmd vy")
        if abs(vx) > _MAX_KEY_COMPONENT or abs(vy) > _MAX_KEY_COMPONENT:
            raise ValueError(f"keyboard cmd components must be within ±{_MAX_KEY_COMPONENT}")
        return KeyboardControl(seat=_seat(obj.get("seat")), cmd=(vx, vy))

    if kind == "speed":
        value = _number(obj.get("value"), "speed")
        if not 0.0 < value <= _MAX_SPEED_MULTIPLIER:
            raise ValueError(f"speed must be in (0, {_MAX_SPEED_MULTIPLIER}], got {value}")
        return SpeedControl(value=value)

    if kind == "limit":
        value = obj.get("value")
        if isinstance(value, bool) or value not in TIME_LIMITS_S:
            raise ValueError(f"time limit must be one of {TIME_LIMITS_S} seconds, got {value!r}")
        return TimeLimitControl(value=int(value))

    if kind == "pause":
        return PauseControl()
    if kind == "play":
        return PlayControl()
    if kind == "step":
        return StepControl()
    return RestartControl()


class KeyboardState:
    """The held keyboard command for each seat.

    The two seats live in one tuple. Each update builds a new tuple and assigns it in a single
    step, so a reader never sees one seat updated and the other not.
    """

    _ZERO: Final[tuple[float, float, float]] = (0.0, 0.0, -math.inf)

    def __init__(self) -> None:
        self._seats: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            self._ZERO,
            self._ZERO,
        )

    def update(self, seat: int, cmd: tuple[float, float], t_received: float) -> None:
        """Hold ``cmd`` for ``seat`` as of ``t_received``. Replaces the seat's entry."""
        entries = list(self._seats)
        entries[seat] = (cmd[0], cmd[1], t_received)
        self._seats = (entries[0], entries[1])

    def current(self, seat: int, now: float) -> tuple[float, float]:
        """The command for ``seat`` at ``now``, or zero once the entry is stale."""
        vx, vy, t_received = self._seats[seat]
        if now - t_received > KEYBOARD_STALE_S:
            return (0.0, 0.0)
        return (vx, vy)
