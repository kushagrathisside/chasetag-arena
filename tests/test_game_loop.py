"""The game loop, tested without HTTP. The clock is a fake, so timing does not decide anything."""

from __future__ import annotations

import queue
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chasetag_arena.agents.bots import BOTS
from chasetag_arena.arcade.controls import (
    KeyboardControl,
    PauseControl,
    PlayControl,
    SelectorControl,
    SpeedControl,
    StepControl,
)
from chasetag_arena.arcade.game_loop import GameLoop
from chasetag_arena.core import game
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.spec.features import encode_v1
from chasetag_arena.core.state import RUNNING, GameState
from chasetag_arena.planner import Planner

# An episode is 2000 control steps, which is 200 decisions. This bound never binds.
MAX_DECISIONS = 210


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ended(loop: GameLoop) -> int:
    return int(np.asarray(loop.state.ended_by))


def _drain(frames: queue.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(frames.get_nowait())
        except queue.Empty:
            return out


def _same_state(a: GameState, b: GameState) -> None:
    np.testing.assert_array_equal(np.asarray(a.bodies.pos), np.asarray(b.bodies.pos))
    np.testing.assert_array_equal(np.asarray(a.bodies.heading), np.asarray(b.bodies.heading))
    np.testing.assert_array_equal(np.asarray(a.bodies.height), np.asarray(b.bodies.height))
    assert int(np.asarray(a.step)) == int(np.asarray(b.step))
    assert int(np.asarray(a.ended_by)) == int(np.asarray(b.ended_by))


def test_streamed_bot_versus_bot_decisions_equal_direct_steps() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=3, selectors=("greedy", "intercept"), clock=clock)
    frames = loop.subscribe()
    _drain(frames)
    params = default_params()
    direct = loop.state
    streamed: list[dict[str, Any]] = []

    for _ in range(MAX_DECISIONS):
        assert loop.decide()
        direct, _ = game.step(direct, loop.last_actions, params)
        _same_state(direct, loop.state)
        streamed.extend(_drain(frames))
        if _ended(loop) != RUNNING:
            break
    assert _ended(loop) != RUNNING, "the round did not end within the bound"

    # Every streamed frame's step and positions are the direct run's, rounded for the wire.
    assert streamed[-2]["phase"] == "ended"
    assert streamed[-1]["phase"] == "intermission"
    assert streamed[-2]["t_event"] == pytest.approx(int(np.asarray(direct.step)) * 0.01)
    last_body = streamed[-2]["bodies"][0]
    assert last_body["x"] == round(float(np.asarray(direct.bodies.pos)[0, 0]), 3)


def test_keyboard_command_expires_and_the_seat_holds_zero() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=4, selectors=("keyboard", "idle"), clock=clock)
    loop.submit(KeyboardControl(seat=0, cmd=(1.0, 0.0)))

    clock.now = 0.1
    loop.decide()
    assert np.any(np.asarray(loop.last_actions)[0] != 0.0)

    clock.now = 0.3  # 0.3 s after the command: stale
    loop.decide()
    np.testing.assert_array_equal(np.asarray(loop.last_actions)[0], np.zeros(2))


def test_selector_change_applies_only_at_round_reset() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=1, selectors=("greedy", "greedy"), clock=clock)
    loop.submit(SelectorControl(seat=0, value="idle"))

    loop.decide()
    assert loop.selectors == ("greedy", "greedy")
    while _ended(loop) == RUNNING:
        loop.decide()
        assert loop.selectors == ("greedy", "greedy")

    clock.now = 2.0  # past the intermission
    loop.decide()
    assert loop.selectors == ("idle", "greedy")


def test_discrete_controls_drain_in_order() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=0, clock=clock)

    loop.submit(PauseControl())
    loop.submit(PlayControl())
    loop.decide()
    assert not loop.paused

    loop.submit(PlayControl())
    loop.submit(PauseControl())
    loop.decide()
    assert loop.paused

    loop.submit(SpeedControl(value=2.0))
    loop.submit(SpeedControl(value=4.0))
    loop.decide()
    assert loop.speed == 4.0


def test_paused_loop_steps_once_per_step_control() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=0, selectors=("idle", "idle"), clock=clock)
    loop.submit(PauseControl())
    loop.decide()
    assert loop.decisions == 0

    loop.submit(StepControl())
    assert loop.decide()
    assert loop.decisions == 1
    assert not loop.decide()
    assert loop.decisions == 1


def test_no_subscriber_means_no_decisions_while_running() -> None:
    loop = GameLoop(seed=2, selectors=("idle", "idle"))
    loop.start()
    try:
        time.sleep(0.3)
        assert loop.decisions == 0, "a loop with no viewer must park"

        frames = loop.subscribe()
        deadline = time.monotonic() + 30.0
        while loop.decisions < 3 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert loop.decisions >= 3

        loop.unsubscribe(frames)
        time.sleep(0.2)  # let an in-flight decision finish
        settled = loop.decisions
        time.sleep(0.5)
        assert loop.decisions == settled
    finally:
        loop.stop()


def test_terminal_frame_is_followed_by_intermission_then_a_new_round() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=5, selectors=("greedy", "greedy"), clock=clock)
    frames = loop.subscribe()
    _drain(frames)

    collected: list[dict[str, Any]] = []
    while _ended(loop) == RUNNING:
        loop.decide()
        collected.extend(_drain(frames))

    ended, intermission = collected[-2], collected[-1]
    assert ended["phase"] == "ended"
    assert ended["ended_by"] in ("tag", "fall", "timeout")
    assert intermission["phase"] == "intermission"
    assert intermission["ended_by"] == ended["ended_by"]
    assert intermission["round_score"] == ended["round_score"]
    assert sum(ended["round_score"]) in (pytest.approx(0.0), pytest.approx(1.0))

    assert not loop.decide(), "no decision is made during the intermission"
    clock.now = 1.5
    assert loop.decide()
    new_round = _drain(frames)
    assert new_round[0]["phase"] == "running"
    assert new_round[0]["round"] == ended["round"] + 1


def test_a_slow_subscriber_keeps_only_the_newest_frames() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=6, selectors=("idle", "idle"), clock=clock)
    frames = loop.subscribe()  # never drained while the loop runs
    for _ in range(100):
        loop.decide()
    kept = _drain(frames)
    assert len(kept) == 64
    seqs = [f["seq"] for f in kept]
    assert seqs == sorted(seqs)
    assert seqs[-1] >= 100, "the newest frame must survive; the oldest are dropped"


def test_unknown_selector_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError):
        GameLoop(seed=0, selectors=("greedy", "telepathy"))


def test_same_seed_gives_the_same_first_state() -> None:
    a = GameLoop(seed=9, clock=FakeClock())
    b = GameLoop(seed=9, clock=FakeClock())
    np.testing.assert_array_equal(np.asarray(a.state.key), np.asarray(b.state.key))
    np.testing.assert_array_equal(np.asarray(a.state.bodies.pos), np.asarray(b.state.bodies.pos))
    assert np.asarray(a.state.key).shape == (2,)


def test_submit_selector_is_held_until_round_start() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=8, clock=clock)
    loop.submit(SelectorControl(seat=1, value="wander"))
    assert loop.selectors == ("greedy", "greedy")
    loop.submit(PauseControl())
    loop.decide()
    assert loop.selectors == ("greedy", "greedy")
    loop.submit(PlayControl())
    loop.decide()
    assert loop.selectors == ("greedy", "greedy")


def _constant_planner(raw: tuple[float, float]) -> Planner:
    """A planner that outputs ``raw`` in every state: zero weights, so the output is the bias."""
    hidden = 4
    return Planner(
        name="constant",
        file="constant.npz",
        weights=(np.zeros((hidden, 17)), np.zeros((2, hidden))),
        biases=(np.zeros(hidden), np.asarray(raw, dtype=np.float64)),
        activation="tanh",
        obs_mean=np.zeros(17),
        obs_var=np.ones(17),
    )


def test_the_planner_seat_plays_the_loaded_planner() -> None:
    planner = _constant_planner((0.5, -0.25))
    loop = GameLoop(seed=2, selectors=("planner", "idle"), clock=FakeClock(), planner=planner)
    assert loop.planner is planner
    loop.decide()
    np.testing.assert_allclose(np.asarray(loop.last_actions)[0], [0.5, -0.25], atol=1e-6)
    np.testing.assert_array_equal(np.asarray(loop.last_actions)[1], [0.0, 0.0])


def test_the_planner_seat_plays_the_fallback_bot_when_no_planner_is_loaded() -> None:
    loop = GameLoop(seed=2, selectors=("planner", "idle"), clock=FakeClock())
    assert loop.planner is None
    expected, _ = BOTS["fallback"].act(
        encode_v1(loop.state, 0), jax.random.PRNGKey(0), jnp.zeros(2)
    )
    loop.decide()
    np.testing.assert_allclose(np.asarray(loop.last_actions)[0], np.asarray(expected), atol=1e-6)


def test_pause_and_speed_reach_viewers_without_a_decision() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=3, selectors=("idle", "idle"), clock=clock)
    frames = loop.subscribe()
    _drain(frames)

    loop.submit(PauseControl())
    loop.decide()
    (paused,) = _drain(frames)
    assert paused["paused"] is True
    assert paused["speed"] == 1.0
    assert paused["phase"] == "running"
    assert loop.decisions == 0

    loop.submit(SpeedControl(value=2.0))
    loop.submit(PlayControl())
    loop.decide()
    played, decision = _drain(frames)  # the status frame, then the decision Play allows
    assert played["paused"] is False
    assert played["speed"] == 2.0
    assert played["seq"] > paused["seq"]
    assert decision["seq"] > played["seq"]
    assert decision["paused"] is False


def test_pause_during_intermission_repeats_the_intermission_frame() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=5, selectors=("greedy", "greedy"), clock=clock)
    frames = loop.subscribe()
    _drain(frames)
    while _ended(loop) == RUNNING:
        loop.decide()
    _drain(frames)

    loop.submit(PauseControl())
    loop.decide()
    (repeat,) = _drain(frames)
    assert repeat["phase"] == "intermission"
    assert repeat["paused"] is True
    assert repeat["ended_by"] is not None
    assert repeat["round_score"] == loop.latest_frame()["round_score"]


def test_latest_frame_carries_the_pause_for_a_new_viewer() -> None:
    clock = FakeClock()
    loop = GameLoop(seed=0, selectors=("idle", "idle"), clock=clock)
    loop.submit(PauseControl())
    loop.decide()
    late = loop.subscribe()
    (snapshot,) = _drain(late)
    assert snapshot["paused"] is True
