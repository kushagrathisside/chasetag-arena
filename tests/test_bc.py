"""Behaviour cloning: the rebuilt features match the game's, and only winning seats are cloned."""

from __future__ import annotations

import jax
import numpy as np

from chasetag_arena import bc, dataset, train
from chasetag_arena.arcade.game_loop import rules_for_limit
from chasetag_arena.core.families.kinematic import default_params


def test_the_first_decision_rebuilds_the_game_features_exactly(tmp_path) -> None:
    # Bodies start at rest, so the velocity estimate is exact at the first decision.
    params = default_params()
    rules = rules_for_limit(20)
    path = tmp_path / dataset.zip_name(20, "greedy", "fallback")
    dataset.write_zip(path, params, 20, "greedy", "fallback", 2, 2, 0, lambda *a: None)
    games = list(bc.read_games(path))
    encode = bc.make_encoder(rules)
    for game in games:
        arrays = bc.game_arrays(game)
        assert arrays is not None
        obs = np.asarray(
            encode(
                arrays["pos"][:1],
                arrays["vel"][:1],
                arrays["height"][:1],
                arrays["heading"][:1],
                np.full(1, game["chaser"], dtype=np.int32),
                arrays["step"][:1],
            )
        )[0]
        # The start layout is fixed and only the chaser varies, so take it from the game.
        start = train.batched_reset(jax.random.split(jax.random.PRNGKey(0), 1), params, rules)
        start = start.__class__(
            bodies=start.bodies,
            chaser=np.full(1, game["chaser"], dtype=np.int32),
            step=start.step,
            ended_by=start.ended_by,
            key=start.key,
        )
        expected = np.asarray(train.seat_observations(start, rules))[0]
        np.testing.assert_allclose(obs, expected, atol=2e-4)


def test_only_winning_moving_seats_are_cloned() -> None:
    def game(ended_by, scores, bots=("greedy", "wander")):
        return {"ended_by": ended_by, "scores": scores, "seat0_bot": bots[0], "seat1_bot": bots[1]}

    assert bc.kept_seats(game("fall", [0.0, 0.0])) == []
    assert bc.kept_seats(game("tag", [0.8, 0.2])) == [0]
    assert bc.kept_seats(game("timeout", [0.0, 1.0])) == [1]
    assert bc.kept_seats(game("tag", [0.5, 0.5])) == [0, 1]
    assert bc.kept_seats(game("timeout", [1.0, 0.0], bots=("idle", "greedy"))) == []


def test_cloning_pairs_each_state_with_the_next_command(tmp_path) -> None:
    params = default_params()
    for seat0, seat1 in [("greedy", "wander")]:
        dataset.write_zip(
            tmp_path / dataset.zip_name(20, seat0, seat1),
            params,
            20,
            seat0,
            seat1,
            4,
            2,
            0,
            lambda *a: None,
        )
    obs, target = bc.load_pairs(tmp_path, (20,), 4, log=lambda s: None)
    assert obs.shape[1] == train.INPUT_DIM
    assert target.shape == (len(obs), 2)
    assert np.all(np.isfinite(obs))
    assert np.all(np.isfinite(target))
