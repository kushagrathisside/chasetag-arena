"""The trainer: the dense reward adds up to the official score, and its files work in the game."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chasetag_arena import train
from chasetag_arena.arcade.game_loop import GameLoop, rules_for_limit
from chasetag_arena.core import game
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.state import FALL, RUNNING, TAG
from chasetag_arena.mock_player import ManualClock
from chasetag_arena.planner import load_planner

TINY = train.Config(
    limits=(20,),
    envs=8,
    iters=2,
    rollout=16,
    hidden=(8,),
    critic_hidden=(8,),
    minibatches=2,
    epochs=1,
    eval_every=2,
    eval_games=8,
    checkpoint_every=1,
)


@pytest.mark.parametrize("limit_s", [20, 60])
def test_dense_rewards_add_up_to_the_official_score(limit_s: int) -> None:
    rules = rules_for_limit(limit_s)
    params = default_params()
    n = 64
    states = train.batched_reset(jax.random.split(jax.random.PRNGKey(5), n), params, rules)

    @jax.jit
    def one(states, key):
        # Large random commands, so that some games end in falls as well as tags.
        raw = 2.0 * jax.random.normal(key, (n, 2, 2))
        nxt, _ = train.batched_step(states, raw, params, rules)
        reward = jax.vmap(lambda b, a: train.dense_rewards(b, a, rules))(states, nxt)
        return nxt, reward

    total = jnp.zeros((n, 2))
    key = jax.random.PRNGKey(6)
    for _ in range(rules.decisions_per_episode):
        key, sub = jax.random.split(key)
        states, reward = one(states, sub)
        total = total + reward
    ended = np.asarray(states.ended_by)
    assert np.all(ended != RUNNING)
    assert np.any(ended == TAG)
    assert np.any(ended == FALL)
    official = jax.vmap(lambda s: game.official_scores(s, rules))(states)
    np.testing.assert_allclose(np.asarray(total), np.asarray(official), atol=1e-4)


def test_a_trained_planner_loads_and_plays_in_the_game(tmp_path) -> None:
    agent, norm = train.train(TINY, tmp_path, "tiny", log=lambda s: None)
    planner = load_planner(tmp_path / "planner.npz")
    assert planner.name == "tiny"

    features = np.random.default_rng(0).normal(size=train.INPUT_DIM)
    expected = np.asarray(train.actor(agent, norm, jnp.asarray(features, dtype=jnp.float32)))
    np.testing.assert_allclose(planner.mean_action(features), expected, atol=1e-4)

    loop = GameLoop(seed=0, selectors=("planner", "greedy"), clock=ManualClock(), planner=planner)
    for _ in range(5):
        loop.decide()
    assert loop.decisions >= 5

    lines = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [line["iter"] for line in lines if "eval" not in line] == [1, 2]
    assert any("eval" in line for line in lines)


def test_opponent_shares_follow_the_league_rules() -> None:
    cfg = train.Config(league_slots=2)
    empty = np.array([False, False])
    probs = train.opponent_probs(cfg, None, empty)
    assert probs.shape == (train.N_OPPONENTS + 2,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.isclose(probs[0], cfg.self_play_share)
    assert np.all(probs[train.N_OPPONENTS :] == 0)  # no snapshot is drawn before one exists
    idle = 1 + train.BOT_NAMES.index("idle")
    assert probs[idle] < probs[1]

    scores = {"greedy": 0.9, "intercept": 0.1, "snap0": 0.5, "snap1": 0.2}
    probs = train.opponent_probs(cfg, scores, np.array([True, True]))
    greedy, intercept = 1 + train.BOT_NAMES.index("greedy"), 1 + train.BOT_NAMES.index("intercept")
    assert probs[intercept] > probs[greedy]  # the harder opponent is drawn more often
    assert np.isclose(probs[train.N_OPPONENTS :].sum(), cfg.snapshot_share)
    assert probs[train.N_OPPONENTS + 1] > probs[train.N_OPPONENTS]


def test_the_weakest_snapshot_is_relegated() -> None:
    assert train.snapshot_slot(np.array([True, False, False]), None, 1) == 1
    full = np.array([True, True, True])
    scores = {"snap0": 0.4, "snap1": 0.9, "snap2": 0.6}
    assert train.snapshot_slot(full, scores, 3) == 1
    assert train.snapshot_slot(full, None, 4) == 1


def test_the_league_fills_and_survives_a_resume(tmp_path) -> None:
    cfg = train.Config(**{**TINY.__dict__, "league_slots": 2, "snapshot_every": 1, "iters": 2})
    train.train(cfg, tmp_path, "tiny", log=lambda s: None)
    evals = [
        json.loads(line)["eval"]
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
        if "eval" in json.loads(line)
    ]
    assert {"snap0", "snap1"} <= set(evals[-1]["matrix"]["20"])
    assert np.isclose(sum(evals[-1]["opponent_probs"]), 1.0, atol=1e-3)  # stored to 4 places
    longer = train.Config(**{**cfg.__dict__, "iters": 3})
    messages: list[str] = []
    train.train(longer, tmp_path, "tiny", resume=True, log=messages.append)
    assert any("joins the league" in m for m in messages)


def test_a_warm_start_is_scored_before_training(tmp_path) -> None:
    train.train(TINY, tmp_path / "first", "first", log=lambda s: None)
    messages: list[str] = []
    train.train(
        TINY,
        tmp_path / "second",
        "second",
        init=tmp_path / "first" / "planner.npz",
        log=messages.append,
    )
    assert any("(the warm start)" in m for m in messages)
    first_eval = json.loads((tmp_path / "second" / "metrics.jsonl").read_text().splitlines()[0])
    assert first_eval["iter"] == 0


def test_a_run_resumes_from_its_checkpoint(tmp_path) -> None:
    train.train(TINY, tmp_path, "tiny", log=lambda s: None)
    messages: list[str] = []
    longer = train.Config(**{**TINY.__dict__, "iters": 3})
    train.train(longer, tmp_path, "tiny", resume=True, log=messages.append)
    assert any("resumed" in m and "round 3" in m for m in messages)
    iters = [
        json.loads(line)["iter"]
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
        if "eval" not in json.loads(line)
    ]
    assert iters == [1, 2, 3]
