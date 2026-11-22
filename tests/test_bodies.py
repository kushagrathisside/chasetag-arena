"""The body registry: every body builds, plays a round, and can be chosen by the player."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from chasetag_arena import bodies
from chasetag_arena.arcade.controls import BodyControl, parse_control
from chasetag_arena.arcade.game_loop import GameLoop
from chasetag_arena.bodies import DEFAULT_BODY, BodySpec, body_by_label, body_names, get_body
from chasetag_arena.core import game
from chasetag_arena.core.spec.rules import OFFICIAL
from chasetag_arena.core.state import RUNNING

ALL = body_names()


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_registry_lists_the_default_and_the_three_families() -> None:
    assert DEFAULT_BODY in ALL
    assert {get_body(name).family for name in ALL} == {"K", "SP", "LS"}
    assert len(ALL) == len(set(ALL))


def test_the_default_body_keeps_its_label() -> None:
    # Recorded sessions and the research store identify the default body by this label.
    assert get_body(DEFAULT_BODY).label == "K/default"


@pytest.mark.parametrize("name", ALL)
def test_a_body_has_a_label_and_a_summary(name: str) -> None:
    spec = get_body(name)
    assert spec.label == f"{spec.family}/{spec.name}"
    assert spec.summary.endswith(".")
    assert body_by_label(spec.label) is spec


def test_an_unknown_body_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown body"):
        get_body("nope")
    with pytest.raises(ValueError, match="unknown body label"):
        body_by_label("K/nope")


def test_a_name_can_be_registered_only_once() -> None:
    with pytest.raises(ValueError, match="already registered"):
        bodies.register(BodySpec("dot", "K", "Again.", lambda: get_body("dot").build()))


@pytest.mark.parametrize("name", ["Upper", "has space", "under_score", ""])
def test_a_bad_name_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="body name"):
        bodies.register(BodySpec(name, "K", "Bad.", lambda: get_body("dot").build()))


@pytest.mark.parametrize("name", ALL)
def test_a_body_plays_a_round_with_finite_state(name: str) -> None:
    params = get_body(name).build()
    state = game.reset(jax.random.PRNGKey(0), params, OFFICIAL)
    actions = jnp.array([[0.5, 0.0], [-0.5, 0.2]])
    for _ in range(30):
        state, _ = game.step(state, actions, params, OFFICIAL)
    assert bool(jnp.all(jnp.isfinite(state.bodies.pos)))
    assert bool(jnp.all(jnp.isfinite(state.bodies.vel)))
    assert bool(jnp.all(jnp.abs(state.bodies.pos) <= OFFICIAL.arena_half_m + 1e-6))


def test_bodies_differ_in_how_fast_they_pick_up_speed() -> None:
    # Half a second of full forward command. The dot follows almost at once, and the default
    # walker is still lagging behind its command.
    speeds = {}
    for name in ("dot", "default"):
        params = get_body(name).build()
        state = game.reset(jax.random.PRNGKey(1), params, OFFICIAL)
        forward = jnp.array([[3.0, 0.0], [3.0, 0.0]])
        for _ in range(5):
            state, _ = game.step(state, forward, params, OFFICIAL)
        speeds[name] = float(jnp.linalg.norm(state.bodies.vel[0]))
    assert speeds["dot"] > speeds["default"] + 0.4


# --- choosing a body ---------------------------------------------------------------------


def test_the_control_parses_a_listed_body() -> None:
    assert parse_control({"kind": "body", "value": "dot"}) == BodyControl(value="dot")


@pytest.mark.parametrize("value", ["K/dot", "nope", 3, None, ["dot"]])
def test_the_control_refuses_anything_else(value: object) -> None:
    with pytest.raises(ValueError, match="body must be one of"):
        parse_control({"kind": "body", "value": value})


def test_the_control_refuses_extra_fields() -> None:
    with pytest.raises(ValueError, match="unexpected fields"):
        parse_control({"kind": "body", "value": "dot", "seat": 0})


def test_a_body_change_waits_for_the_next_round() -> None:
    clock = _Clock()
    loop = GameLoop(seed=2, selectors=("idle", "idle"), clock=clock)
    assert loop.body.name == DEFAULT_BODY
    loop.submit(BodyControl(value="spring"))
    assert loop.body.name == DEFAULT_BODY
    while int(loop.state.ended_by) == RUNNING:
        loop.decide()
        clock.now += OFFICIAL.decision_dt
    clock.now += 2.0  # past the intermission
    loop.decide()
    assert loop.body.name == "spring"
    assert loop.state.bodies.aux.shape[-1] == 8  # the spring body's state: base, tilt and rates


def test_the_loop_starts_with_the_body_it_is_given() -> None:
    loop = GameLoop(seed=0, selectors=("idle", "idle"), body="stepping", clock=_Clock())
    assert loop.body.label == "LS/stepping"


def test_an_unknown_body_at_start_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown body"):
        GameLoop(seed=0, body="nope", clock=_Clock())


def test_a_recorded_session_keeps_the_default_body() -> None:
    class Recorder:
        def round_started(self, *args: object) -> None: ...
        def decision(self, *args: object) -> None: ...
        def round_ended(self, *args: object) -> None: ...

    loop = GameLoop(seed=0, selectors=("idle", "idle"), clock=_Clock(), recorder=Recorder())
    assert loop.body_locked
    with pytest.raises(ValueError, match="body is fixed"):
        loop.submit(BodyControl(value="dot"))


def test_the_serve_command_reads_the_body() -> None:
    from chasetag_arena.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["serve"]).body == DEFAULT_BODY
    assert parser.parse_args(["serve", "--body", "dodger"]).body == "dodger"
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--body", "nope"])
