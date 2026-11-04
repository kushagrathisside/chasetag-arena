"""Behaviour cloning from the bot-episode dataset: a warm start for ``chasetag_arena.train``.

The dataset stores each decision's positions, heights, headings and raw commands, but not the
velocities. The planner's features need velocity, so it is estimated from the change in position
over the decision before (0.1 s). Bodies start at rest, so the first decision's velocity is exact.
The estimate is an average over the decision, while the body lags its command, so the rebuilt
features are close to, not equal to, what the game would have seen.

Only the winning seat of each game is cloned: a seat that scored at least as much as the other,
in a game that did not end in a fall. Idle seats are never cloned, since they never move. The
fit compares commands after the tanh, in the space the game applies, because a bot's raw output
can be very large where its command is at the speed limit.

The result is an ordinary planner file. It can be played in the game, scored with the trainer's
evaluation, and passed to ``train --init`` to start PPO from it:

    uv run --no-sync python -m chasetag_arena.bc --data DATA --limits 20 --out bc20.npz
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from chasetag_arena.arcade.game_loop import rules_for_limit
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.spec.features import SPEC_V1
from chasetag_arena.core.spec.rules import Rules
from chasetag_arena.dataset import COLUMNS, MEMBER, pairings, zip_name
from chasetag_arena.train import (
    INPUT_DIM,
    Adam,
    Agent,
    Config,
    Normalizer,
    adam_step,
    evaluate,
    init_mlp,
    make_evaluator,
    mlp,
    normalize,
    save_planner,
)

DECISION_DT: Final[float] = 0.1  # s between rows of a game
_COL = {name: i for i, name in enumerate(COLUMNS)}


class _Bodies(NamedTuple):
    """The body fields the spec v1 encoder reads."""

    pos: jax.Array  # (2, 2)
    vel: jax.Array  # (2, 2)
    height: jax.Array  # (2,)
    heading: jax.Array  # (2,)


class _State(NamedTuple):
    """The game fields the spec v1 encoder reads."""

    bodies: _Bodies
    chaser: jax.Array  # ()
    step: jax.Array  # ()


def read_games(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """The games in one zip, skipping its header line."""
    with zipfile.ZipFile(path) as zf, zf.open(MEMBER) as member:
        header = json.loads(next(member))
        if list(header["columns"]) != list(COLUMNS):
            raise ValueError(f"{path} has columns {header['columns']}, expected {COLUMNS}")
        for i, line in enumerate(member):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)


def kept_seats(game: dict[str, Any]) -> list[int]:
    """The seats worth cloning: the winners of a game that did not end in a fall."""
    if game["ended_by"] == "fall":
        return []
    bots = (game["seat0_bot"], game["seat1_bot"])
    scores = game["scores"]
    return [
        seat for seat in (0, 1) if bots[seat] != "idle" and scores[seat] >= scores[1 - seat] - 1e-9
    ]


def game_arrays(game: dict[str, Any]) -> dict[str, np.ndarray] | None:
    """The states a game's decisions were made from, and the command chosen at each.

    Row k of the frames is the state after decision k, and holds decision k's command. So the
    state of row k - 1 pairs with the command in row k.
    """
    frames = np.asarray(game["frames"], dtype=np.float32)
    if len(frames) < 2:
        return None
    pos = frames[:, [_COL["x0"], _COL["y0"], _COL["x1"], _COL["y1"]]].reshape(-1, 2, 2)
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / DECISION_DT
    return {
        "pos": pos[:-1],
        "vel": vel[:-1],
        "height": frames[:-1, [_COL["height0"], _COL["height1"]]],
        "heading": frames[:-1, [_COL["heading0"], _COL["heading1"]]],
        "step": frames[:-1, _COL["step"]].astype(np.int32),
        "target": frames[1:, [_COL["raw0x"], _COL["raw0y"], _COL["raw1x"], _COL["raw1y"]]].reshape(
            -1, 2, 2
        ),
    }


def make_encoder(rules: Rules):
    """Spec v1 features of both seats for a batch of rebuilt states, (F, 2, 17)."""

    @jax.jit
    def encode(pos, vel, height, heading, chaser, step):
        def one(p, v, h, psi, c, s):
            state = _State(_Bodies(p, v, h, psi), c, s)
            return jnp.stack([SPEC_V1.encode(state, 0, rules), SPEC_V1.encode(state, 1, rules)])

        return jax.vmap(one)(pos, vel, height, heading, chaser, step)

    return encode


def load_pairs(
    data: Path, limits: tuple[int, ...], games_per_zip: int, log=print
) -> tuple[np.ndarray, np.ndarray]:
    """Features (K, 17) and target raw commands (K, 2) from every pairing at ``limits``."""
    feats, targets = [], []
    for limit in limits:
        encode = make_encoder(rules_for_limit(limit))
        for seat0, seat1 in pairings():
            path = data / zip_name(limit, seat0, seat1)
            if not path.exists():
                log(f"  missing {path.name}, skipped")
                continue
            parts: dict[str, list[np.ndarray]] = {
                k: []
                for k in ("pos", "vel", "height", "heading", "step", "target", "chaser", "keep")
            }
            for game in read_games(path, games_per_zip):
                seats = kept_seats(game)
                arrays = game_arrays(game) if seats else None
                if arrays is None:
                    continue
                n = len(arrays["step"])
                for k in ("pos", "vel", "height", "heading", "step", "target"):
                    parts[k].append(arrays[k])
                parts["chaser"].append(np.full(n, game["chaser"], dtype=np.int32))
                keep = np.zeros((n, 2), dtype=bool)
                keep[:, seats] = True
                parts["keep"].append(keep)
            if not parts["step"]:
                continue
            cat = {k: np.concatenate(v) for k, v in parts.items()}
            obs = np.asarray(
                encode(
                    cat["pos"],
                    cat["vel"],
                    cat["height"],
                    cat["heading"],
                    cat["chaser"],
                    cat["step"],
                )
            )
            feats.append(obs[cat["keep"]])
            targets.append(cat["target"][cat["keep"]])
        log(f"  {limit}s: {sum(len(f) for f in feats)} cloned decisions so far")
    return np.concatenate(feats).astype(np.float32), np.concatenate(targets).astype(np.float32)


def fit(
    obs: np.ndarray,
    target: np.ndarray,
    hidden: tuple[int, ...],
    epochs: int,
    batch: int,
    lr: float,
    seed: int,
    log=print,
) -> tuple[Agent, Normalizer, dict[str, float]]:
    """Fit an actor to the cloned commands. Returns it, its normaliser, and the fit's error."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(obs))
    n_val = max(len(obs) // 20, 1)
    val, tr = order[:n_val], order[n_val:]
    norm = Normalizer(
        mean=jnp.asarray(obs[tr].mean(axis=0)),
        var=jnp.asarray(obs[tr].var(axis=0)),
        count=jnp.float32(len(tr)),
    )
    layers = init_mlp(jax.random.PRNGKey(seed), (INPUT_DIM, *hidden, 2), out_gain=0.01)
    opt = Adam(
        m=jax.tree_util.tree_map(jnp.zeros_like, layers),
        v=jax.tree_util.tree_map(jnp.zeros_like, layers),
        t=jnp.zeros((), jnp.float32),
    )
    cfg = Config(hidden=hidden, max_grad_norm=1.0)

    def loss_fn(layers, x, y):
        # Compare commands after the tanh, the space the game applies.
        pred = jnp.tanh(mlp(layers, normalize(norm, x)))
        return jnp.mean(jnp.sum((pred - jnp.tanh(y)) ** 2, axis=-1))

    @jax.jit
    def step(layers, opt, x, y, lr):
        loss, grads = jax.value_and_grad(loss_fn)(layers, x, y)
        layers, opt = adam_step(layers, grads, opt, lr, cfg)
        return layers, opt, loss

    val_loss = jax.jit(loss_fn)
    x_val, y_val = jnp.asarray(obs[val]), jnp.asarray(target[val])
    steps_per_epoch = max(len(tr) // batch, 1)
    total = epochs * steps_per_epoch
    done = 0
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(tr)
        losses = []
        for i in range(steps_per_epoch):
            ids = perm[i * batch : (i + 1) * batch]
            rate = jnp.float32(lr * (1.0 - done / total))
            layers, opt, loss = step(
                layers, opt, jnp.asarray(obs[ids]), jnp.asarray(target[ids]), rate
            )
            losses.append(float(loss))
            done += 1
        v = float(val_loss(layers, x_val, y_val))
        log(f"  epoch {epoch:3d}: train {np.mean(losses):.4f}  validation {v:.4f}")
    # The squared error is in tanh units per component, at most 2 per command; 1.5 m/s per unit.
    rmse = float(np.sqrt(v / 2.0)) * 1.5
    agent = Agent(actor=layers, critic=[], log_std=jnp.zeros(2))
    return agent, norm, {"validation_loss": v, "command_rmse_mps": rmse}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True, help="folder of the dataset's zips")
    parser.add_argument("--out", type=Path, required=True, help="the planner file to write")
    parser.add_argument("--limits", default="20", help="round lengths to clone, e.g. 20,60,180")
    parser.add_argument("--games-per-zip", type=int, default=1000)
    parser.add_argument("--hidden", default="64,64")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=512)
    args = parser.parse_args(argv)
    limits = tuple(int(x) for x in args.limits.split(",") if x)
    hidden = tuple(int(x) for x in args.hidden.split(",") if x)
    log = lambda s: print(s, flush=True)  # noqa: E731
    start = time.monotonic()
    log(f"reading {args.data} for {limits}")
    obs, target = load_pairs(args.data, limits, args.games_per_zip, log)
    log(f"fitting {len(obs)} decisions")
    agent, norm, fit_stats = fit(
        obs, target, hidden, args.epochs, args.batch, args.lr, args.seed, log
    )
    cfg = Config(limits=limits, hidden=hidden, seed=args.seed, eval_games=args.eval_games)
    save_planner(args.out, agent, norm, args.out.stem, cfg, 0)
    log(f"saved {args.out}; command error {fit_stats['command_rmse_mps']:.3f} m/s")
    result = evaluate(make_evaluator(default_params()), agent, norm, cfg)
    for limit, row in result["matrix"].items():
        for label, r in row.items():
            log(
                f"  eval {limit:>3}s vs {label:<9} score {r['score']:.3f} "
                f"(chaser {r['score_chaser']:.3f}, runner {r['score_runner']:.3f})  "
                f"tag {r['tag']:.2f} fall {r['fall']:.2f} timeout {r['timeout']:.2f}"
            )
    log(f"  ranked score: {result['mean_score']:.4f}  ({time.monotonic() - start:.0f}s)")
    report = {"fit": fit_stats, "eval": result, "decisions": len(obs)}
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
