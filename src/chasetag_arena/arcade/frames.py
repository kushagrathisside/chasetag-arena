"""The frame the browser draws: one JSON-ready snapshot per decision.

Frames are full snapshots, so a client that misses one loses nothing once the next arrives.
That is why the server may drop the oldest frame from a slow client's queue.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

import numpy as np

from chasetag_arena.core.spec.rules import OFFICIAL
from chasetag_arena.core.state import FALL, RUNNING, TAG, TIMEOUT, GameState

Phase = Literal["running", "ended", "intermission"]
PHASES: Final[tuple[str, ...]] = ("running", "ended", "intermission")

# Core outcome codes to the frame's ``ended_by`` value.
_ENDED_BY: Final[dict[int, str | None]] = {
    RUNNING: None,
    TAG: "tag",
    FALL: "fall",
    TIMEOUT: "timeout",
}

_MATCH_KEYS: Final[frozenset[str]] = frozenset({"seat0", "seat1", "tags", "falls", "timeouts"})
_SELECTOR_KEYS: Final[frozenset[str]] = frozenset({"seat0", "seat1"})
_TAU: Final[float] = 2.0 * math.pi
_HEADING_EPS: Final[float] = 1e-4  # one unit in the fourth decimal, the heading resolution


def wrap_heading(angle: float) -> float:
    """Wrap an angle into (-pi, pi], then round it to 4 decimals.

    The rounding can push a value across the seam (3.14159 rounds to 3.1416, which is past
    pi). It is moved one unit inward so the stated interval holds exactly.
    """
    if not math.isfinite(angle):
        raise ValueError(f"heading is not finite: {angle}")
    wrapped = math.remainder(angle, _TAU)  # in [-pi, pi]
    if wrapped <= -math.pi:
        wrapped += _TAU  # -pi is part of the interval as pi
    value = round(wrapped, 4)
    if value > math.pi:
        value -= _HEADING_EPS
    elif value <= -math.pi:
        value += _HEADING_EPS
    return value


def _finite_round(value: float, digits: int, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {number}")
    return round(number, digits)


def _match(match: Mapping[str, Any]) -> dict[str, float | int]:
    if set(match) != _MATCH_KEYS:
        raise ValueError(f"match must have keys {sorted(_MATCH_KEYS)}, got {sorted(match)}")
    return {
        "seat0": _finite_round(match["seat0"], 3, "match.seat0"),
        "seat1": _finite_round(match["seat1"], 3, "match.seat1"),
        "tags": int(match["tags"]),
        "falls": int(match["falls"]),
        "timeouts": int(match["timeouts"]),
    }


def _selectors(selectors: Mapping[str, str], body_label: str) -> dict[str, str]:
    if set(selectors) != _SELECTOR_KEYS:
        raise ValueError(f"selectors must have keys {sorted(_SELECTOR_KEYS)}")
    return {"seat0": selectors["seat0"], "seat1": selectors["seat1"], "body": body_label}


def state_to_frame(
    state: GameState,
    scores: Sequence[float] | np.ndarray,
    *,
    seq: int,
    t_server: float,
    round_no: int,
    phase: Phase,
    ended_by: int,
    selectors: Mapping[str, str],
    body_label: str,
    match: Mapping[str, Any],
    round_score: Sequence[float] | np.ndarray,
    paused: bool,
    speed: float,
) -> dict[str, Any]:
    """Build the frame for ``state``.

    ``scores`` is the per-seat official score that ``game.step`` returned for this decision.
    It is zero except on the decision that ends the round, and it is shown only when
    ``phase == "ended"``. ``round_score`` is the score the frame shows in every other phase,
    which lets intermission keep the score of the round that just ended.

    ``ended_by`` takes a core outcome code (``RUNNING``, ``TAG``, ``FALL``, ``TIMEOUT``), not
    a string, so the caller can pass the code of a finished round during intermission.

    ``paused`` and ``speed`` are the server's current pause and speed. A viewer shows them as
    they are, so it never has to guess them from the frames it has received.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    if ended_by not in _ENDED_BY:
        raise ValueError(f"ended_by must be a core outcome code, got {ended_by!r}")
    if not isinstance(paused, bool):
        raise ValueError(f"paused must be a bool, got {paused!r}")
    if not (math.isfinite(speed) and speed > 0.0):
        raise ValueError(f"speed must be a positive finite number, got {speed!r}")

    pos = np.asarray(state.bodies.pos, dtype=np.float64)
    height = np.asarray(state.bodies.height, dtype=np.float64)
    heading = np.asarray(state.bodies.heading, dtype=np.float64)
    fallen = np.asarray(state.bodies.fallen, dtype=bool)
    step = int(np.asarray(state.step))
    chaser = int(np.asarray(state.chaser))

    shown = scores if phase == "ended" else round_score
    shown_list = [float(v) for v in np.asarray(shown, dtype=np.float64).reshape(-1)]
    if len(shown_list) != 2:
        raise ValueError("round score must have one value per seat")

    bodies = [
        {
            "seat": seat,
            "x": _finite_round(pos[seat, 0], 3, "x"),
            "y": _finite_round(pos[seat, 1], 3, "y"),
            "heading": wrap_heading(float(heading[seat])),
            "height": _finite_round(height[seat], 3, "height"),
            "fallen": bool(fallen[seat]),
        }
        for seat in (0, 1)
    ]

    return {
        "seq": int(seq),
        "t_server": _finite_round(t_server, 3, "t_server"),
        "round": int(round_no),
        "phase": phase,
        # Exact: the core records the control step of the tag or the fall itself.
        "t_event": _finite_round(step * OFFICIAL.control_dt, 3, "t_event"),
        "ended_by": _ENDED_BY[ended_by],
        "bodies": bodies,
        "chaser": chaser,
        "round_score": [_finite_round(v, 3, "round_score") for v in shown_list],
        "match": _match(match),
        "selectors": _selectors(selectors, body_label),
        "paused": paused,
        "speed": float(speed),
    }


def encode_frame(frame: Mapping[str, Any]) -> str:
    """Serialise a frame as compact JSON. NaN or infinity raises ``ValueError``."""
    return json.dumps(frame, allow_nan=False, separators=(",", ":"))
