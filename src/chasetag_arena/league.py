"""A sports league for planners and bots: divisions by format, fixtures, and a standings table.

Each round length is a format with its own division, as cricket has T20, one-day and Test
matches: 20 s, 60 s and 180 s. A planner enters the divisions it was trained for (its file's
``round_limits_s``), and the scripted bots enter every division.

A fixture is two legs, like two innings: in one leg the home team chases, in the other it runs.
The home team plays from seat 0, the home end. Each leg is many games with the same roles, and a
team's result is its mean official score over both legs. The season is a double round-robin, so
every pair meets twice, once at each end.

A fixture is won by the higher total, unless the totals are within ``draw_margin``. A win is
worth 3 points and a draw 1. The table is ordered by points, then by score difference.

    uv run --no-sync python -m chasetag_arena.league --planner runs/k20/planner.npz --out table
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from chasetag_arena.agents.bots import BOTS
from chasetag_arena.arcade.game_loop import rules_for_limit
from chasetag_arena.core import game
from chasetag_arena.core.families.base import BodyParams
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.state import FALL, TAG, TIMEOUT, GameState
from chasetag_arena.planner import load_planner
from chasetag_arena.train import (
    BOT_NAMES,
    N_BOTS,
    League,
    Normalizer,
    batched_reset,
    batched_step,
    init_bots,
    mlp,
    normalize,
    seat_observations,
)

FORMATS: Final[dict[int, str]] = {20: "T20 (20 s)", 60: "one-day (60 s)", 180: "Test (180 s)"}
WIN_POINTS: Final[int] = 3
DRAW_POINTS: Final[int] = 1


@dataclass
class Team:
    """A bot, or a planner in one of the league's planner slots."""

    name: str
    code: int  # a bot's index in BOT_NAMES, or N_BOTS + its planner slot
    formats: tuple[int, ...]


@dataclass
class Standing:
    team: str
    played: int = 0
    won: int = 0
    drawn: int = 0
    lost: int = 0
    score_for: float = 0.0
    score_against: float = 0.0
    tags: int = 0
    falls: int = 0
    points: int = 0
    results: list[str] = field(default_factory=list)

    @property
    def difference(self) -> float:
        return self.score_for - self.score_against


def planner_league(paths: list[Path]) -> tuple[League, list[Team]]:
    """Load planner files into league slots. Every planner must share one network shape."""
    planners = [load_planner(p) for p in paths]
    if not planners:
        empty = League(
            actors=[(jnp.zeros((1, 2, 17)), jnp.zeros((1, 2)))],
            mean=jnp.zeros((1, 17)),
            var=jnp.ones((1, 17)),
            filled=jnp.zeros((1,), dtype=bool),
        )
        return empty, []
    shapes = {tuple(w.shape for w in p.weights) for p in planners}
    if len(shapes) != 1:
        raise ValueError("every planner in one league must have the same network shape")
    if any(p.activation != "tanh" for p in planners):
        raise ValueError("the league plays tanh planners only")
    actors = [
        (
            jnp.stack([jnp.asarray(p.weights[i], jnp.float32) for p in planners]),
            jnp.stack([jnp.asarray(p.biases[i], jnp.float32) for p in planners]),
        )
        for i in range(len(planners[0].weights))
    ]
    league = League(
        actors=actors,
        mean=jnp.stack([jnp.asarray(p.obs_mean, jnp.float32) for p in planners]),
        var=jnp.stack([jnp.asarray(p.obs_var, jnp.float32) for p in planners]),
        filled=jnp.ones((len(planners),), dtype=bool),
    )
    teams = []
    for slot, (path, planner) in enumerate(zip(paths, planners, strict=True)):
        teams.append(Team(planner.name, N_BOTS + slot, _planner_formats(path)))
    return league, teams


def _planner_formats(path: Path) -> tuple[int, ...]:
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(data["meta"].tobytes().decode("utf-8"))
    limits = meta.get("round_limits_s") or [20]
    return tuple(int(x) for x in limits)


def make_leg(params: BodyParams, limit_s: int):
    """Play ``n`` games between two teams with fixed seats and roles."""
    rules = rules_for_limit(limit_s)
    steps = rules.decisions_per_episode

    def seat_raw(code: Array, obs: Array, bots: Array, keys: Array, league: League):
        # Every bot and every planner acts, then each game keeps its team's command.
        n = obs.shape[0]
        bot_raw, new_bots = [], []
        for k, name in enumerate(BOT_NAMES):
            r, s = jax.vmap(BOTS[name].act)(obs, keys[:, k], bots[:, k])
            bot_raw.append(r)
            new_bots.append(s)
        bot_raw = jnp.stack(bot_raw, axis=1)[jnp.arange(n), jnp.clip(code, 0, N_BOTS - 1)]

        def one(layers, mean, var):
            return mlp(layers, normalize(Normalizer(mean, var, jnp.float32(1.0)), obs))

        planner_raw = jax.vmap(one)(league.actors, league.mean, league.var)
        slot = jnp.clip(code - N_BOTS, 0, league.filled.shape[0] - 1)
        planner_raw = planner_raw[slot, jnp.arange(n)]
        raw = jnp.where((code >= N_BOTS)[:, None], planner_raw, bot_raw)
        return raw, jnp.stack(new_bots, axis=1)

    @partial(jax.jit, static_argnames=("n",))
    def leg(codes: Array, chaser_seat: Array, league: League, key: Array, n: int):
        k_reset, k_bots, k_steps = jax.random.split(key, 3)
        states = batched_reset(jax.random.split(k_reset, n), params, rules)
        states = GameState(
            bodies=states.bodies,
            chaser=jnp.full((n,), chaser_seat, dtype=jnp.int32),
            step=states.step,
            ended_by=states.ended_by,
            key=states.key,
        )
        bots = jnp.stack([init_bots(jax.random.fold_in(k_bots, s), n) for s in (0, 1)], axis=1)

        def decide(carry, _):
            states, bots, key = carry
            key, k = jax.random.split(key)
            obs = seat_observations(states, rules)
            keys = jax.random.split(k, n * 2 * N_BOTS).reshape(n, 2, N_BOTS, 2)
            raws, new_bots = [], []
            for seat in (0, 1):
                code = jnp.full((n,), codes[seat])
                r, b = seat_raw(code, obs[:, seat], bots[:, seat], keys[:, seat], league)
                raws.append(r)
                new_bots.append(b)
            states, _ = batched_step(states, jnp.stack(raws, axis=1), params, rules)
            return (states, jnp.stack(new_bots, axis=1), key), None

        (states, _, _), _ = lax.scan(decide, (states, bots, k_steps), None, length=steps)
        scores = jax.vmap(lambda s: game.official_scores(s, rules))(states)
        fell = (states.ended_by == FALL)[:, None] & states.bodies.fallen
        return {
            "scores": jnp.mean(scores, axis=0),
            "tag": jnp.sum(states.ended_by == TAG),
            "fallen": jnp.sum(fell, axis=0),  # (2,) games in which each seat fell
            "timeout": jnp.sum(states.ended_by == TIMEOUT),
        }

    return leg


def play_season(
    teams: list[Team],
    league: League,
    limit_s: int,
    games: int,
    seed: int,
    draw_margin: float,
    log=print,
) -> dict[str, Any]:
    """A double round-robin in one format. Returns the fixtures and the ordered table."""
    entrants = [t for t in teams if limit_s in t.formats]
    leg = make_leg(default_params(), limit_s)
    table = {t.name: Standing(t.name) for t in entrants}
    fixtures = []
    for f, (home, away) in enumerate(itertools.permutations(entrants, 2)):
        codes = jnp.array([home.code, away.code], dtype=jnp.int32)
        key = jax.random.fold_in(jax.random.PRNGKey(seed), f)
        legs = []
        for chaser_seat in (0, 1):  # the home team chases, then runs
            out = leg(
                codes, jnp.int32(chaser_seat), league, jax.random.fold_in(key, chaser_seat), games
            )
            legs.append({k: np.asarray(v) for k, v in out.items()})
        home_total = float(sum(lg["scores"][0] for lg in legs))
        away_total = float(sum(lg["scores"][1] for lg in legs))
        # Leg 0 has the home team chasing, leg 1 the away team, so each leg's tags are its
        # chaser's. Falls belong to the seat whose body fell.
        tags = (int(legs[0]["tag"]), int(legs[1]["tag"]))
        falls = (int(sum(lg["fallen"][0] for lg in legs)), int(sum(lg["fallen"][1] for lg in legs)))
        if abs(home_total - away_total) <= draw_margin:
            outcome = "draw"
        else:
            outcome = "home" if home_total > away_total else "away"
        fixtures.append(
            {
                "home": home.name,
                "away": away.name,
                "home_score": round(home_total, 4),
                "away_score": round(away_total, 4),
                "result": outcome,
                "home_tags": tags[0],
                "away_tags": tags[1],
                "home_falls": falls[0],
                "away_falls": falls[1],
            }
        )
        for side, team, own, other, won in (
            (0, home, home_total, away_total, outcome == "home"),
            (1, away, away_total, home_total, outcome == "away"),
        ):
            s = table[team.name]
            s.played += 1
            s.score_for += own
            s.score_against += other
            s.tags += tags[side]
            s.falls += falls[side]
            if outcome == "draw":
                s.drawn += 1
                s.points += DRAW_POINTS
                s.results.append("D")
            elif won:
                s.won += 1
                s.points += WIN_POINTS
                s.results.append("W")
            else:
                s.lost += 1
                s.results.append("L")
        log(
            f"  {FORMATS.get(limit_s, limit_s)}: {home.name} {home_total:.3f} - "
            f"{away_total:.3f} {away.name}  ({outcome})"
        )
    ordered = sorted(table.values(), key=lambda s: (s.points, s.difference), reverse=True)
    return {
        "format": limit_s,
        "name": FORMATS.get(limit_s, f"{limit_s} s"),
        "games_per_leg": games,
        "fixtures": fixtures,
        "table": [
            {
                "position": i + 1,
                "team": s.team,
                "played": s.played,
                "won": s.won,
                "drawn": s.drawn,
                "lost": s.lost,
                "score_for": round(s.score_for, 3),
                "score_against": round(s.score_against, 3),
                "difference": round(s.difference, 3),
                "tags": s.tags,
                "falls": s.falls,
                "points": s.points,
            }
            for i, s in enumerate(ordered)
        ],
    }


def render_table(season: dict[str, Any]) -> str:
    """A season's table in Markdown."""
    lines = [
        f"### {season['name']}",
        "",
        "| # | Team | P | W | D | L | For | Against | Diff | Tags | Falls | Pts |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in season["table"]:
        lines.append(
            f"| {r['position']} | {r['team']} | {r['played']} | {r['won']} | {r['drawn']} | "
            f"{r['lost']} | {r['score_for']:.2f} | {r['score_against']:.2f} | "
            f"{r['difference']:+.2f} | {r['tags']} | {r['falls']} | {r['points']} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--planner", type=Path, action="append", default=[], help="a planner file; repeatable"
    )
    parser.add_argument("--formats", default="20,60,180", help="divisions, by round length")
    parser.add_argument("--games", type=int, default=256, help="games in each leg")
    parser.add_argument("--draw-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True, help="folder for the tables")
    args = parser.parse_args(argv)
    formats = tuple(int(x) for x in args.formats.split(",") if x)
    league, planners = planner_league(args.planner)
    teams = [Team(name, k, formats) for k, name in enumerate(BOT_NAMES)] + planners
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    seasons = []
    for limit_s in formats:
        log(f"season: {FORMATS.get(limit_s, limit_s)}")
        season = play_season(teams, league, limit_s, args.games, args.seed, args.draw_margin, log)
        seasons.append(season)
        log(render_table(season))
        (args.out / "standings.json").write_text(json.dumps(seasons, indent=2) + "\n")
        (args.out / "standings.md").write_text("\n\n".join(render_table(s) for s in seasons) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
