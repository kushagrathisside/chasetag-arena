"""The two-seat chase-tag game: reset, step, observe and the official score.

Everything here is a pure function of its arguments, so it works under ``jit`` and
``vmap``. Body dynamics come only from the ``BodyModel`` protocol, so a new family needs
no change here.

Official rules (design section 2.2): a tag ends the episode with the chaser scoring
1 - t/T and the runner t/T. A runner that survives T scores 1. A fall scores 0 for both
seats. Roles are drawn at random each episode.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from chasetag_arena.core.arena import start_layout
from chasetag_arena.core.dynamics import Noise, draw_noise, integrate_decision
from chasetag_arena.core.families.base import BodyParams
from chasetag_arena.core.state import FALL, RUNNING, TAG, TIMEOUT, GameState
from chasetag_arena.core.spec.commands import CommandSpec, to_command
from chasetag_arena.core.spec.features import FeatureSpec
from chasetag_arena.core.spec.rules import OFFICIAL, Rules


@partial(jax.jit, static_argnames=("rules",))
def reset(key: Array, params: BodyParams, rules: Rules = OFFICIAL) -> GameState:
    """A new episode: random chaser seat, official starts, bodies at rest."""
    k_role, k_rest = jax.random.split(key)
    chaser = jax.random.randint(k_role, (), 0, 2, dtype=jnp.int32)
    pos, heading = start_layout(rules)
    model, theta = params.model, params.theta
    bodies = jax.vmap(lambda p, h: model.init_state(theta, p, h, rules))(pos, heading)
    return GameState(
        bodies=bodies,
        chaser=chaser,
        step=jnp.int32(0),
        ended_by=jnp.int32(RUNNING),
        key=k_rest,
    )


def official_scores(state: GameState, rules: Rules = OFFICIAL) -> Array:
    """Official per-seat score, shape (2,). Zero for both seats while the episode runs."""
    dtype = state.bodies.pos.dtype
    frac = state.step.astype(dtype) * rules.control_dt / rules.episode_s
    tagged = state.ended_by == TAG
    timed_out = state.ended_by == TIMEOUT
    chaser_score = jnp.where(tagged, 1.0 - frac, 0.0)
    runner_score = jnp.where(tagged, frac, jnp.where(timed_out, 1.0, 0.0))
    is_chaser = jnp.arange(2) == state.chaser
    return jnp.where(is_chaser, chaser_score, runner_score).astype(dtype)


def advance(
    state: GameState,
    actions: Array,
    noise: Noise,
    params: BodyParams,
    rules: Rules = OFFICIAL,
) -> GameState:
    """One planner decision with explicit noise. ``state.key`` is left unchanged.

    ``actions`` holds raw planner outputs, shape (2, 2), one row per seat. Each finished
    episode is frozen, so calling this again changes nothing.
    """
    model, theta = params.model, params.theta
    dtype = state.bodies.pos.dtype
    cmd = to_command(actions.astype(dtype), state.bodies.heading, CommandSpec(rules.max_speed_mps))
    running = state.ended_by == RUNNING

    bodies, ended, end_step = integrate_decision(
        model, theta, state.bodies, cmd, noise.eps, rules, state.ended_by, state.step
    )

    # Stochastic fall, decided once per decision at its end. A tag earlier in the decision
    # has already ended the episode, so only still-running seats can fall here.
    still = ended == RUNNING
    elapsed = (state.step + rules.decision_substeps).astype(dtype) * rules.control_dt
    prev_cmd = state.bodies.last_cmd[:, 0]
    probs = jax.vmap(
        lambda bef, aft, pc, c: model.fall_probability(theta, bef, aft, pc, c, elapsed, rules)
    )(state.bodies, bodies, prev_cmd, cmd)
    fell = (noise.u < probs) & still
    bodies = replace(bodies, fallen=bodies.fallen | fell)
    ended = jnp.where(still & jnp.any(fell), FALL, ended)

    # A fallen pelvis drops below the official fall height.
    collapsed = 0.5 * rules.fall_height_m
    bodies = replace(
        bodies,
        height=jnp.where(bodies.fallen, jnp.minimum(bodies.height, collapsed), bodies.height),
    )

    # The end of the decision, or the time out.
    ended = jnp.where((ended == RUNNING) & (end_step >= rules.episode_steps), TIMEOUT, ended)

    last = jnp.stack([cmd, prev_cmd], axis=1)
    bodies = replace(bodies, last_cmd=jnp.where(running, last, state.bodies.last_cmd))
    return GameState(
        bodies=bodies,
        chaser=state.chaser,
        step=jnp.where(running, end_step, state.step),
        ended_by=ended,
        key=state.key,
    )


@partial(jax.jit, static_argnames=("rules",))
def step(
    state: GameState, actions: Array, params: BodyParams, rules: Rules = OFFICIAL
) -> tuple[GameState, Array]:
    """One planner decision with fresh randomness. Returns the state and the terminal scores.

    The second value is the official per-seat score on the step where the episode ends,
    and zeros on every other step.
    """
    key, sub = jax.random.split(state.key)
    nxt = advance(state, actions, draw_noise(sub, rules), params, rules)
    nxt = replace(nxt, key=key)
    ending = (state.ended_by == RUNNING) & (nxt.ended_by != RUNNING)
    scores = jnp.where(ending, official_scores(nxt, rules), 0.0)
    return nxt, scores


def observe(
    state: GameState, seat: Array | int, spec: FeatureSpec, rules: Rules = OFFICIAL
) -> Array:
    """The planner's input for ``seat``, encoded by ``spec``."""
    return spec.encode(state, seat, rules)


__all__ = [
    "advance",
    "observe",
    "official_scores",
    "reset",
    "step",
]
