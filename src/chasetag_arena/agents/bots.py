"""Scripted opponents as pure JAX functions of the 17-dimensional strategy features.

Each bot is ``act(obs, key, state) -> (raw, next_state)``. ``obs`` is spec v1
(``SPEC_V1``), ``raw`` is the planner-style raw action that ``spec.commands.to_command``
maps to a command, and ``state`` is the bot's own carried state. Stateless bots pass it
through. The role comes from the observation, so one function serves both seats and both
roles, with ``jnp.where`` in place of Python branches. The bots therefore run under
``jit``, ``vmap`` and ``lax.scan``, and can be exported.

They mirror ``chasetag.agents.bots`` (greedy, intercept, wander, idle). Chasetag's bots
work in the world frame; these work in the heading frame, which the features already
use, and the geometry is the same. The runner's flight and the wall push match chasetag's.
Intercept's runner flees, as chasetag's does. The metrics note (section 3.5) describes a
perpendicular runner; that differs, and the code follows chasetag.

``fallback`` is the cautious planner that ships if learning fails (metrics section 3.5).
It caps its speed, which keeps falls rare, and as runner it heads for open space.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from chasetag_arena.core.spec.rules import OFFICIAL

# Index of each v1 feature. ``tests/test_bots.py`` checks these against the spec names.
ROLE = slice(0, 2)  # chaser bit first; [1, 0] chaser, [0, 1] runner
REL = slice(2, 4)  # opponent minus own, heading frame
REL_VEL = slice(4, 6)  # opponent velocity minus own, heading frame
DIST = 6  # 3D pelvis distance
OWN_NORM = slice(7, 9)  # own position divided by the arena half-width, world frame
OPP_NORM = slice(9, 11)
COS_SIN = slice(11, 13)  # cos and sin of the heading
OWN_VEL = slice(13, 15)  # own velocity, heading frame

_LIMIT = 1.0 - 1e-6  # tanh never reaches 1, so the inverse must clip short of it
_DT = OFFICIAL.decision_dt
# Chasetag's wall push for the runner: margin in metres, and the weight on the push.
_RUNNER_MARGIN = 2.0
_RUNNER_GAIN = 1.5


class View(NamedTuple):
    """The decoded features a bot reads. All fields are JAX arrays."""

    is_chaser: Array
    rel: Array
    rel_vel: Array
    dist: Array
    own_norm: Array
    cos: Array
    sin: Array
    own_vel: Array


def decode(obs: Array) -> View:
    return View(
        is_chaser=obs[0] > 0.5,
        rel=obs[REL],
        rel_vel=obs[REL_VEL],
        dist=obs[DIST],
        own_norm=obs[OWN_NORM],
        cos=obs[COS_SIN][0],
        sin=obs[COS_SIN][1],
        own_vel=obs[OWN_VEL],
    )


def _unit(vec: Array) -> Array:
    """Unit vector, or zero when the input is (numerically) zero."""
    norm = jnp.linalg.norm(vec)
    safe = jnp.where(norm > 1e-9, norm, 1.0)
    return jnp.where(norm > 1e-9, vec / safe, jnp.zeros_like(vec))


def _heading_frame(vec: Array, cos: Array, sin: Array) -> Array:
    """World to heading frame, as ``spec.frames.to_heading_frame`` does."""
    return jnp.stack([cos * vec[0] + sin * vec[1], -sin * vec[0] + cos * vec[1]])


def _world_wall_push(own_norm: Array, half: float, margin: float) -> Array:
    """Repulsion from the fence, growing linearly within ``margin`` metres of a wall.

    Chasetag's ``_wall_push``, taken from the normalised position. The push is in the
    world frame, and it adds up at a corner, so a runner there heads for the middle.
    """
    xy = own_norm * half
    gap_low = xy + half
    gap_high = half - xy
    push_low = jnp.maximum(margin - gap_low, 0.0) / margin
    push_high = jnp.maximum(margin - gap_high, 0.0) / margin
    return push_low - push_high


def raw_for_command(cmd_h: Array, max_speed: float = OFFICIAL.max_speed_mps) -> Array:
    """The raw action whose ``to_command`` is ``cmd_h`` (heading frame).

    ``to_command`` computes ``tanh(raw) * max_speed``, then rescales if the norm exceeds
    ``max_speed``. This inverts that. A command longer than ``max_speed`` is scaled down
    first. Components are clipped just inside ±1 before ``arctanh``, so a full-speed
    command gives a large finite raw action, not an infinite one.
    """
    norm = jnp.linalg.norm(cmd_h)
    scale = jnp.where(norm > max_speed, max_speed / jnp.where(norm > 0, norm, 1.0), 1.0)
    unit_scaled = jnp.clip(cmd_h * scale / max_speed, -_LIMIT, _LIMIT)
    return jnp.arctanh(unit_scaled)


def _flee_direction(v: View, half: float, margin: float, gain: float) -> Array:
    """Chasetag's ``_flee``, in the heading frame: away from the opponent, plus wall push."""
    away = _unit(-v.rel)
    push = _heading_frame(_world_wall_push(v.own_norm, half, margin), v.cos, v.sin)
    return _unit(away + gain * push)


def init_stateless(key: Array) -> Array:
    """Placeholder state for bots that carry none. Its shape matches the wander state."""
    del key
    return jnp.zeros(2)


# --- chasetag's four bots --------------------------------------------------------------


def greedy_act(
    obs: Array, key: Array, state: Array, *, max_speed: float = OFFICIAL.max_speed_mps
) -> tuple[Array, Array]:
    """Chaser: straight at the opponent. Runner: straight away, with wall push."""
    del key
    v = decode(obs)
    chase = _unit(v.rel) * max_speed
    flee = _flee_direction(v, OFFICIAL.arena_half_m, _RUNNER_MARGIN, _RUNNER_GAIN) * max_speed
    cmd = jnp.where(v.is_chaser, chase, flee)
    return raw_for_command(cmd, max_speed), state


def intercept_act(
    obs: Array, key: Array, state: Array, *, max_speed: float = OFFICIAL.max_speed_mps
) -> tuple[Array, Array]:
    """Chaser: aims where the opponent will be, at constant velocity. Runner: as greedy."""
    del key
    v = decode(obs)
    lead = jnp.minimum(v.dist / max(max_speed, 1e-3), 1.5)
    # The opponent's absolute velocity is its velocity relative to own, plus own velocity.
    target = v.rel + lead * (v.rel_vel + v.own_vel)
    chase = _unit(target) * max_speed
    flee = _flee_direction(v, OFFICIAL.arena_half_m, _RUNNER_MARGIN, _RUNNER_GAIN) * max_speed
    cmd = jnp.where(v.is_chaser, chase, flee)
    return raw_for_command(cmd, max_speed), state


def idle_act(obs: Array, key: Array, state: Array) -> tuple[Array, Array]:
    """Zero command, whatever the role."""
    del obs, key
    return jnp.zeros(2), state


class WanderConfig(NamedTuple):
    speed_frac: float = 0.6  # chasetag's wander runs at 60% of the maximum speed
    tau_s: float = 2.0  # correlation time of the heading drift
    sigma: float = 1.0  # drift noise scale
    dt: float = _DT
    half: float = OFFICIAL.arena_half_m
    margin: float = 2.0


WANDER = WanderConfig()


def wander_init(key: Array, cfg: WanderConfig = WANDER) -> Array:
    """Start the drift at a random direction, in the world frame."""
    del cfg
    return jax.random.normal(key, (2,))


def wander_act(
    obs: Array,
    key: Array,
    state: Array,
    *,
    cfg: WanderConfig = WANDER,
    max_speed: float = OFFICIAL.max_speed_mps,
) -> tuple[Array, Array]:
    """Smooth random walk: an Ornstein-Uhlenbeck drift in the world frame, kept inside."""
    v = decode(obs)
    eps = jax.random.normal(key, (2,))
    drift = state + (-state / cfg.tau_s) * cfg.dt + cfg.sigma * jnp.sqrt(cfg.dt) * eps
    push = _world_wall_push(v.own_norm, cfg.half, cfg.margin)
    direction_w = _unit(_unit(drift) + push)
    cmd_h = _heading_frame(direction_w, v.cos, v.sin) * (cfg.speed_frac * max_speed)
    return raw_for_command(cmd_h, max_speed), drift


# --- the cautious fallback planner -----------------------------------------------------


@dataclass(frozen=True)
class FallbackConfig:
    """Speed caps are below the 1.5 m/s maximum, which keeps the walker's falls rare."""

    chase_speed: float = 1.0  # m/s when chasing
    flee_speed: float = 1.0  # m/s when running
    half: float = OFFICIAL.arena_half_m
    wall_margin: float = 2.0  # m, chasetag's wall margin
    wall_gain: float = 2.0  # heavier than chasetag's 1.5, so the runner turns before the fence


FALLBACK = FallbackConfig()


def fallback_act(
    obs: Array, key: Array, state: Array, *, cfg: FallbackConfig = FALLBACK
) -> tuple[Array, Array]:
    """Chaser: greedy at a capped speed. Runner: flee to open space at a capped speed."""
    del key
    v = decode(obs)
    chase = _unit(v.rel) * cfg.chase_speed
    flee = _flee_direction(v, cfg.half, cfg.wall_margin, cfg.wall_gain) * cfg.flee_speed
    cmd = jnp.where(v.is_chaser, chase, flee)
    return raw_for_command(cmd, OFFICIAL.max_speed_mps), state


class Bot(NamedTuple):
    """A bot's initial state and its action function."""

    init_state: Callable[[Array], Array]
    act: Callable[[Array, Array, Array], tuple[Array, Array]]


BOTS: dict[str, Bot] = {
    "greedy": Bot(init_stateless, greedy_act),
    "intercept": Bot(init_stateless, intercept_act),
    "wander": Bot(wander_init, wander_act),
    "idle": Bot(init_stateless, idle_act),
    "fallback": Bot(init_stateless, fallback_act),
}


__all__ = [
    "BOTS",
    "FALLBACK",
    "WANDER",
    "Bot",
    "FallbackConfig",
    "View",
    "WanderConfig",
    "decode",
    "fallback_act",
    "greedy_act",
    "idle_act",
    "intercept_act",
    "raw_for_command",
    "wander_act",
    "wander_init",
]
