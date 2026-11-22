"""Every body is playable: it does not fall at rest, the bots do not make it fall in most games,
and a chaser can still catch a runner. The numbers are rates over a batch of 20 s games with
the scripted bots, so they hold up to noise and catch a body whose physics has become unplayable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chasetag_arena.bodies import body_names, get_body
from chasetag_arena.league import BOT_NAMES, make_leg, planner_league

GAMES = 64
LIMIT_S = 20
_EMPTY_LEAGUE = planner_league([])[0]


def _play(name: str, chaser_bot: str, runner_bot: str) -> dict[str, float]:
    """Seat 0 chases with ``chaser_bot`` and seat 1 runs with ``runner_bot``."""
    leg = _LEGS[name]
    codes = jnp.array([BOT_NAMES.index(chaser_bot), BOT_NAMES.index(runner_bot)])
    out = leg(codes, jnp.int32(0), _EMPTY_LEAGUE, jax.random.PRNGKey(7), GAMES)
    return {
        "tag": float(out["tag"]) / GAMES,
        "fall": float(np.asarray(out["fallen"]).sum()) / GAMES,  # falling seats per game
        "chaser": float(out["scores"][0]),
        "runner": float(out["scores"][1]),
    }


_LEGS = {name: make_leg(get_body(name).build(), LIMIT_S) for name in body_names()}


@pytest.mark.parametrize("name", body_names())
def test_a_body_at_rest_almost_never_falls(name: str) -> None:
    assert _play(name, "idle", "idle")["fall"] <= 0.10


@pytest.mark.parametrize("name", body_names())
def test_two_greedy_bots_rarely_make_a_body_fall(name: str) -> None:
    assert _play(name, "greedy", "greedy")["fall"] <= 0.25


@pytest.mark.parametrize("name", body_names())
def test_a_chaser_can_catch_a_wandering_runner(name: str) -> None:
    result = _play(name, "intercept", "wander")
    assert result["tag"] >= 0.5
    assert result["chaser"] > result["runner"]


@pytest.mark.parametrize("name", body_names())
def test_neither_side_wins_every_game_between_equal_bots(name: str) -> None:
    result = _play(name, "greedy", "greedy")
    assert 0.1 <= result["chaser"] <= 0.9
    assert 0.1 <= result["runner"] <= 0.9
