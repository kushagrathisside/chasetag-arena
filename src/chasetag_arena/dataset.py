"""Scripted-bot episode dataset: every ordered pair of bots, at each round length.

Each zip holds one pairing (the bot in seat 0 and the bot in seat 1) at one round length, with
the same number of games. Inside a zip, the first line is a header that names the columns, and
every later line is one game as a JSON object. A game's frames are its start, then one row for
each decision it made. Each row is a list of 13 numbers:

    step, x0, y0, x1, y1, height0, height1, heading0, heading1, raw0x, raw0y, raw1x, raw1y

Positions are metres in the world frame, headings are radians, and the raw commands are the
planner-style outputs that ``core.spec.commands.to_command`` maps to a velocity (the same space
the planner file uses). ``step`` counts control steps of 0.01 s, so a decision is 10 steps.

Run it from a checkout. It skips zips that already hold the full count, so an interrupted run
can be restarted:

    uv run --no-sync python -m chasetag_arena.dataset --out DATA --episodes 10000
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np

from chasetag_arena.agents.bots import BOTS
from chasetag_arena.arcade.game_loop import rules_for_limit
from chasetag_arena.core import game
from chasetag_arena.core.families.base import BodyParams
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.state import FALL, RUNNING, TAG, TIMEOUT
from chasetag_arena.train import BOT_NAMES, N_BOTS, batched_reset, init_bots, seat_observations

COLUMNS: Final[tuple[str, ...]] = (
    "step",
    "x0",
    "y0",
    "x1",
    "y1",
    "height0",
    "height1",
    "heading0",
    "heading1",
    "raw0x",
    "raw0y",
    "raw1x",
    "raw1y",
)
ROUND_LIMITS_S: Final[tuple[int, ...]] = (20, 60, 180)
ENDED_NAMES: Final[dict[int, str]] = {TAG: "tag", FALL: "fall", TIMEOUT: "timeout"}
MEMBER: Final[str] = "episodes.jsonl"


EXCLUDED_PAIRINGS: Final[frozenset[tuple[str, str]]] = frozenset({("idle", "idle")})
"""Pairings left out of the dataset: two idle bots never move, so every game is the same."""


def pairings() -> list[tuple[str, str]]:
    """Every ordered pair of bots, seat 0's then seat 1's, without the excluded ones: 24 pairs."""
    return [p for p in itertools.product(BOT_NAMES, BOT_NAMES) if p not in EXCLUDED_PAIRINGS]


def seed_index(pair: tuple[str, str]) -> int:
    """A pairing's place among all 25 ordered pairs, excluded ones included.

    The archive's seed is the run's seed plus this index. Counting the excluded pairs keeps every
    archive's seed the same as when the pairing list had no exclusions, so the same command still
    makes the same archives.
    """
    return list(itertools.product(BOT_NAMES, BOT_NAMES)).index(pair)


def zip_name(limit_s: int, seat0: str, seat1: str) -> str:
    return f"limit{limit_s:03d}s__seat0-{seat0}__seat1-{seat1}.zip"


def make_chunk_runner(params: BodyParams, limit_s: int):
    """Play ``n`` games between two fixed bots, and return every decision's state and command."""
    rules = rules_for_limit(limit_s)
    steps = rules.decisions_per_episode

    @partial(jax.jit, static_argnames=("n",))
    def run(codes: jax.Array, key: jax.Array, n: int) -> dict[str, jax.Array]:
        # codes (n, 2): the bot index of each seat, into BOT_NAMES
        k_reset, k_bots, k_steps = jax.random.split(key, 3)
        start = batched_reset(jax.random.split(k_reset, n), params, rules)
        bots = jnp.stack([init_bots(jax.random.fold_in(k_bots, s), n) for s in (0, 1)], axis=1)

        def decide(carry, _):
            states, bots, key = carry
            key, k_act = jax.random.split(key)
            obs = seat_observations(states, rules)
            keys = jax.random.split(k_act, n * 2 * N_BOTS).reshape(n, 2, N_BOTS, 2)
            raws, new_bots = [], []
            for seat in (0, 1):
                rs, ss = [], []
                for j, name in enumerate(BOT_NAMES):
                    r, st = jax.vmap(BOTS[name].act)(
                        obs[:, seat], keys[:, seat, j], bots[:, seat, j]
                    )
                    rs.append(r)
                    ss.append(st)
                raws.append(jnp.stack(rs, axis=1))
                new_bots.append(jnp.stack(ss, axis=1))
            all_raw = jnp.stack(raws, axis=1)  # (n, seat, bot, 2)
            bots = jnp.stack(new_bots, axis=1)  # (n, seat, bot, 2)
            pick = jnp.broadcast_to(codes[:, :, None, None], (n, 2, 1, 2))
            raw = jnp.take_along_axis(all_raw, pick, axis=2)[:, :, 0]  # (n, 2, 2)
            running = states.ended_by == RUNNING
            nxt, _ = jax.vmap(lambda s, a: game.step(s, a, params, rules))(states, raw)
            row = {
                "step": nxt.step,
                "pos": nxt.bodies.pos,
                "height": nxt.bodies.height,
                "heading": nxt.bodies.heading,
                "raw": raw,
                "running": running,
            }
            return (nxt, bots, key), row

        (final, _, _), rows = jax.lax.scan(decide, (start, bots, k_steps), None, length=steps)
        return {
            "start_pos": start.bodies.pos,
            "start_height": start.bodies.height,
            "start_heading": start.bodies.heading,
            "chaser": start.chaser,
            "ended_by": final.ended_by,
            "scores": jax.vmap(lambda s: game.official_scores(s, rules))(final),
            **rows,
        }

    return run, steps


def game_lines(out: dict[str, Any], g: int, seat0: str, seat1: str, limit_s: int, index: int):
    """One game as a JSON line, from the arrays of one chunk."""
    running = out["running"][:, g]
    sel = np.nonzero(running)[0]
    start = np.concatenate(
        [
            [0],
            np.asarray(out["start_pos"][g]).reshape(-1),
            np.asarray(out["start_height"][g]),
            np.asarray(out["start_heading"][g]),
            np.zeros(4),
        ]
    )
    frames = [np.round(start, 4).tolist()]
    steps = np.asarray(out["step"][sel, g])
    pos = np.asarray(out["pos"][sel, g]).reshape(len(sel), 4)
    height = np.asarray(out["height"][sel, g])
    heading = np.asarray(out["heading"][sel, g])
    raw = np.asarray(out["raw"][sel, g]).reshape(len(sel), 4)
    rows = np.concatenate([steps[:, None], pos, height, heading, raw], axis=1)
    frames += np.round(rows, 4).tolist()
    ended = ENDED_NAMES[int(out["ended_by"][g])]
    record = {
        "episode": index,
        "round_limit_s": limit_s,
        "seat0_bot": seat0,
        "seat1_bot": seat1,
        "chaser": int(out["chaser"][g]),
        "ended_by": ended,
        "decisions": len(frames) - 1,
        "scores": np.round(np.asarray(out["scores"][g]), 6).tolist(),
        "frames": frames,
    }
    return json.dumps(record, separators=(",", ":")) + "\n"


def write_zip(
    path: Path,
    params: BodyParams,
    limit_s: int,
    seat0: str,
    seat1: str,
    episodes: int,
    chunk: int,
    seed: int,
    log,
) -> dict[str, Any]:
    """Write one pairing's zip, with ``episodes`` games, and return its manifest entry."""
    runner, _ = make_chunk_runner(params, limit_s)
    codes = jnp.array([BOT_NAMES.index(seat0), BOT_NAMES.index(seat1)], dtype=jnp.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(path.name + ".partial")
    decisions = 0
    start = time.monotonic()
    with (
        zipfile.ZipFile(partial_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf,
        zf.open(MEMBER, "w", force_zip64=True) as member,
    ):
        header = {
            "columns": COLUMNS,
            "round_limit_s": limit_s,
            "seat0_bot": seat0,
            "seat1_bot": seat1,
            "episodes": episodes,
        }
        member.write((json.dumps(header, separators=(",", ":")) + "\n").encode())
        index = 0
        chunk_key = jax.random.PRNGKey(seed)
        for c in range(-(-episodes // chunk)):
            n = min(chunk, episodes - index)
            out = runner(jnp.broadcast_to(codes, (n, 2)), jax.random.fold_in(chunk_key, c), n)
            out = jax.tree_util.tree_map(np.asarray, out)
            for g in range(n):
                member.write(game_lines(out, g, seat0, seat1, limit_s, index).encode())
                index += 1
            decisions += int(out["running"].sum())
            log(f"    {path.name}: {index}/{episodes} games")
    partial_path.replace(path)
    return {
        "file": path.name,
        "round_limit_s": limit_s,
        "seat0_bot": seat0,
        "seat1_bot": seat1,
        "episodes": episodes,
        "decisions": decisions,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "seconds": round(time.monotonic() - start, 1),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_complete(path: Path, episodes: int) -> bool:
    """A zip that opens and holds the header plus the full count of games."""
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as zf, zf.open(MEMBER) as member:
            lines = sum(1 for _ in member)
        return lines == episodes + 1
    except (zipfile.BadZipFile, OSError, KeyError):
        return False


def build(
    out: Path,
    episodes: int,
    limits: tuple[int, ...],
    chunk: int,
    seed: int,
    log=print,
) -> list[dict[str, Any]]:
    """Write every zip, skipping complete ones, and a manifest beside them."""
    out.mkdir(parents=True, exist_ok=True)
    params = default_params()
    entries: list[dict[str, Any]] = []
    for limit_s in limits:
        for seat0, seat1 in pairings():
            path = out / zip_name(limit_s, seat0, seat1)
            file_seed = seed + seed_index((seat0, seat1))
            if is_complete(path, episodes):
                log(f"skip {path.name} (complete)")
            else:
                log(f"write {path.name}")
                write_zip(path, params, limit_s, seat0, seat1, episodes, chunk, file_seed, log)
            entries.append(
                {
                    "file": path.name,
                    "round_limit_s": limit_s,
                    "seat0_bot": seat0,
                    "seat1_bot": seat1,
                    "episodes": episodes,
                    "seed": file_seed,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
            (out / "manifest.json").write_text(
                json.dumps({"columns": COLUMNS, "files": entries}, indent=2) + "\n"
            )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="folder for the zips")
    parser.add_argument("--episodes", type=int, default=10_000, help="games per zip")
    parser.add_argument(
        "--limits",
        default=",".join(str(s) for s in ROUND_LIMITS_S),
        help="round lengths in game seconds, comma separated",
    )
    parser.add_argument("--chunk", type=int, default=500, help="games per device batch")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    limits = tuple(int(x) for x in args.limits.split(",") if x)
    build(args.out, args.episodes, limits, args.chunk, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
