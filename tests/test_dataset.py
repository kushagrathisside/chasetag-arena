"""The bot-episode dataset: round lengths are honoured, and a zip holds what its header says."""

from __future__ import annotations

import json
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from chasetag_arena import dataset
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.state import TIMEOUT
from chasetag_arena.train import BOT_NAMES


def test_a_long_round_runs_past_the_official_twenty_seconds() -> None:
    # Two idle bots never tag, so a game that does not fall must time out at the round's limit.
    run, steps = dataset.make_chunk_runner(default_params(), 60)
    idle = BOT_NAMES.index("idle")
    out = run(jnp.full((4, 2), idle, dtype=jnp.int32), jax.random.PRNGKey(3), 4)
    out = jax.tree_util.tree_map(np.asarray, out)
    assert steps == 600
    timed_out = out["ended_by"] == TIMEOUT
    assert timed_out.any()
    final_step = out["step"][-1]
    assert np.all(final_step[timed_out] == 6000)
    # The runner keeps the whole point at a time out, under the round's own limit.
    runner = 1 - out["chaser"]
    scores = out["scores"][np.arange(4), runner]
    assert np.allclose(scores[timed_out], 1.0)


def test_a_zip_holds_its_header_and_every_game(tmp_path) -> None:
    path = tmp_path / dataset.zip_name(20, "greedy", "wander")
    entry = dataset.write_zip(
        path, default_params(), 20, "greedy", "wander", 3, 2, 0, lambda *a: None
    )
    assert dataset.is_complete(path, 3)
    with zipfile.ZipFile(path) as zf:
        lines = zf.read(dataset.MEMBER).decode().splitlines()
    header = json.loads(lines[0])
    assert header["columns"] == list(dataset.COLUMNS)
    games = [json.loads(line) for line in lines[1:]]
    assert len(games) == 3
    for game in games:
        assert game["decisions"] == len(game["frames"]) - 1
        assert all(len(row) == len(dataset.COLUMNS) for row in game["frames"])
        assert game["frames"][-1][0] <= 2000
    assert entry["decisions"] == sum(g["decisions"] for g in games)


def test_archive_seeds_count_the_excluded_pairing() -> None:
    # Idle against idle is the 19th of the 25 ordered pairs. Leaving it out must not move the seed
    # of any later pairing, so that the same command still makes the same archives.
    assert dataset.seed_index(("greedy", "greedy")) == 0
    assert dataset.seed_index(("idle", "idle")) == 18
    assert dataset.seed_index(("idle", "fallback")) == 19
    assert dataset.seed_index(("fallback", "fallback")) == 24
    seeds = [dataset.seed_index(p) for p in dataset.pairings()]
    assert len(set(seeds)) == len(seeds) == 24
    assert 18 not in seeds


def test_the_manifest_records_each_archives_seed(tmp_path) -> None:
    entries = dataset.build(tmp_path, 2, (20,), 2, 100, log=lambda *a: None)
    assert len(entries) == 24
    by_pair = {(e["seat0_bot"], e["seat1_bot"]): e["seed"] for e in entries}
    assert by_pair[("greedy", "greedy")] == 100
    assert by_pair[("idle", "fallback")] == 119
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert [f["seed"] for f in manifest["files"]] == [e["seed"] for e in entries]
