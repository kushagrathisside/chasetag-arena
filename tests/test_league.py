"""The sports league: legs fix the roles, and the table adds up its fixtures."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chasetag_arena import league, train
from chasetag_arena.core.families.kinematic import default_params


def test_a_leg_plays_the_roles_it_is_given() -> None:
    planners, _ = league.planner_league([])
    leg = league.make_leg(default_params(), 20)
    idle = train.BOT_NAMES.index("idle")
    codes = jnp.array([idle, idle], dtype=jnp.int32)
    out = leg(codes, jnp.int32(0), planners, jax.random.PRNGKey(0), 16)
    falls = int(np.asarray(out["fallen"]).sum())
    assert int(out["tag"]) == 0  # idle never moves, so nobody is tagged
    # Seat 1 runs in every game, so it keeps the point in every game that does not end in a fall.
    np.testing.assert_allclose(float(out["scores"][1]), (16 - falls) / 16, atol=1e-6)
    np.testing.assert_allclose(float(out["scores"][0]), 0.0, atol=1e-6)


def test_a_season_table_adds_up(tmp_path) -> None:
    cfg = train.Config(
        limits=(20,),
        envs=8,
        iters=1,
        rollout=8,
        hidden=(8,),
        critic_hidden=(8,),
        minibatches=2,
        epochs=1,
        eval_every=1,
        eval_games=8,
    )
    train.train(cfg, tmp_path, "tiny", log=lambda s: None)
    planners, planner_teams = league.planner_league([tmp_path / "planner.npz"])
    assert planner_teams[0].formats == (20,)
    teams = [
        league.Team("greedy", train.BOT_NAMES.index("greedy"), (20, 60)),
        league.Team("wander", train.BOT_NAMES.index("wander"), (20, 60)),
        *planner_teams,
    ]
    season = league.play_season(teams, planners, 20, 4, 0, 0.02, log=lambda s: None)
    table = season["table"]
    assert [r["team"] for r in table if r["team"] == "tiny"] == ["tiny"]
    assert all(r["played"] == 4 for r in table)  # two meetings with each of two rivals
    assert len(season["fixtures"]) == 6
    decisive = sum(f["result"] != "draw" for f in season["fixtures"])
    draws = len(season["fixtures"]) - decisive
    assert sum(r["points"] for r in table) == 3 * decisive + 2 * draws
    points = [(r["points"], r["difference"]) for r in table]
    assert points == sorted(points, reverse=True)
    # The planner was trained at 20 s only, so it does not enter the one-day division.
    other = league.play_season(teams, planners, 60, 2, 0, 0.02, log=lambda s: None)
    assert {r["team"] for r in other["table"]} == {"greedy", "wander"}
