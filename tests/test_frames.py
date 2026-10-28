"""The frame builder meets the contract: keys, types, heading wrap, event time and NaN handling."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from chasetag_arena.arcade.frames import encode_frame, state_to_frame, wrap_heading
from chasetag_arena.core import game
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.state import FALL, RUNNING, TAG, TIMEOUT, GameState

PARAMS = default_params()
MATCH = {"seat0": 0.5, "seat1": 0.25, "tags": 3, "falls": 1, "timeouts": 2}
SELECTORS = {"seat0": "keyboard", "seat1": "greedy"}


def _state(step: int = 0, ended_by: int = RUNNING) -> GameState:
    base = game.reset(jax.random.PRNGKey(7), PARAMS)
    return replace(base, step=jnp.int32(step), ended_by=jnp.int32(ended_by))


def _frame(state: GameState, *, phase: str = "running", ended_by: int = RUNNING) -> dict:
    return state_to_frame(
        state,
        [0.0, 0.0],
        seq=5,
        t_server=12.3456,
        round_no=2,
        phase=phase,  # type: ignore[arg-type]
        ended_by=ended_by,
        selectors=SELECTORS,
        body_label="K/default",
        match=MATCH,
        round_score=[0.0, 0.0],
    )


def test_contract_keys_and_types_are_present() -> None:
    frame = _frame(_state())
    assert set(frame) == {
        "seq",
        "t_server",
        "round",
        "phase",
        "t_event",
        "ended_by",
        "bodies",
        "chaser",
        "round_score",
        "match",
        "selectors",
    }
    assert isinstance(frame["seq"], int)
    assert isinstance(frame["round"], int)
    assert frame["phase"] == "running"
    assert frame["ended_by"] is None
    assert frame["chaser"] in (0, 1)
    assert len(frame["bodies"]) == 2
    for seat, body in enumerate(frame["bodies"]):
        assert body["seat"] == seat
        assert set(body) == {"seat", "x", "y", "heading", "height", "fallen"}
        assert isinstance(body["x"], float)
        assert isinstance(body["heading"], float)
        assert isinstance(body["fallen"], bool)
    assert frame["selectors"] == {"seat0": "keyboard", "seat1": "greedy", "body": "K/default"}
    assert set(frame["match"]) == {"seat0", "seat1", "tags", "falls", "timeouts"}


def test_heading_wraps_across_plus_and_minus_pi() -> None:
    assert wrap_heading(math.pi + 0.1) == pytest.approx(-math.pi + 0.1, abs=1e-4)
    assert wrap_heading(-math.pi - 0.1) == pytest.approx(math.pi - 0.1, abs=1e-4)
    assert wrap_heading(3 * math.pi) == pytest.approx(math.pi, abs=1e-4)
    # The interval is (-pi, pi]: the seam itself is pi, and rounding never leaves it.
    for angle in (math.pi, -math.pi, math.pi - 1e-6, -math.pi + 1e-6):
        value = wrap_heading(angle)
        assert -math.pi < value <= math.pi, angle


def test_frame_headings_are_wrapped() -> None:
    state = _state()
    shifted = replace(
        state,
        bodies=replace(state.bodies, heading=state.bodies.heading + 4 * math.pi),
    )
    for body in _frame(shifted)["bodies"]:
        assert -math.pi < body["heading"] <= math.pi


def test_t_event_is_step_times_control_dt() -> None:
    for step in (0, 1, 137, 2000):
        frame = _frame(_state(step=step))
        assert frame["t_event"] == pytest.approx(step * 0.01, abs=1e-9)


@pytest.mark.parametrize(
    ("code", "expected"),
    [(RUNNING, None), (TAG, "tag"), (FALL, "fall"), (TIMEOUT, "timeout")],
)
def test_ended_by_maps_core_codes(code: int, expected: str | None) -> None:
    assert _frame(_state(), ended_by=code)["ended_by"] == expected


def test_ended_phase_shows_the_step_scores_and_intermission_keeps_round_score() -> None:
    state = _state(step=40, ended_by=TAG)
    ended = state_to_frame(
        state,
        [0.8, 0.2],
        seq=1,
        t_server=0.0,
        round_no=1,
        phase="ended",
        ended_by=TAG,
        selectors=SELECTORS,
        body_label="K/default",
        match=MATCH,
        round_score=[0.0, 0.0],
    )
    assert ended["round_score"] == [0.8, 0.2]
    held = state_to_frame(
        state,
        [0.0, 0.0],
        seq=2,
        t_server=0.0,
        round_no=1,
        phase="intermission",
        ended_by=TAG,
        selectors=SELECTORS,
        body_label="K/default",
        match=MATCH,
        round_score=[0.8, 0.2],
    )
    assert held["round_score"] == [0.8, 0.2]


def test_nan_in_the_state_raises() -> None:
    state = _state()
    poisoned = replace(state, bodies=replace(state.bodies, pos=state.bodies.pos * jnp.nan))
    with pytest.raises(ValueError, match="not finite"):
        _frame(poisoned)


def test_encode_frame_rejects_nan_and_emits_compact_json() -> None:
    frame = _frame(_state())
    text = encode_frame(frame)
    assert ": " not in text and ", " not in text
    assert json.loads(text) == json.loads(json.dumps(frame))
    frame["t_server"] = float("nan")
    with pytest.raises(ValueError):
        encode_frame(frame)
