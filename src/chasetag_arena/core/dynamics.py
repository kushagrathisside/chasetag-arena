"""One planner decision: a ``lax.scan`` over the control substeps of a decision.

Tag is checked after every substep, because the official score depends on the exact
tag time. A deterministic fall from a family's own rule is also checked per substep.
The stochastic fall of family K is decided once per decision in ``game.advance``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax

from chasetag_arena.core.state import FALL, RUNNING, TAG, BodyState
from chasetag_arena.core.spec.rules import Rules

# Standard normal draws per seat per substep: two for tracking, one for height.
NOISE_DIM = 3


class Noise(NamedTuple):
    """All randomness of one decision. Passing it explicitly makes the step reproducible."""

    eps: Array  # (decision_substeps, 2, NOISE_DIM), standard normal
    u: Array  # (2,), uniform on [0, 1); seat i falls if u[i] < its fall probability


def draw_noise(key: Array, rules: Rules) -> Noise:
    k_eps, k_u = jax.random.split(key)
    eps = jax.random.normal(k_eps, (rules.decision_substeps, 2, NOISE_DIM))
    u = jax.random.uniform(k_u, (2,))
    return Noise(eps=eps, u=u)


def pelvis_distance(bodies: BodyState) -> Array:
    """3D distance between the two pelvises, as the official tag test measures it."""
    dxy = bodies.pos[0] - bodies.pos[1]
    dz = bodies.height[0] - bodies.height[1]
    return jnp.sqrt(jnp.sum(dxy**2) + dz**2)


def integrate_decision(
    model: Any,
    theta: Any,
    bodies: BodyState,
    cmd: Array,
    eps: Array,
    rules: Rules,
    ended_by: Array,
    step0: Array,
) -> tuple[BodyState, Array, Array]:
    """Integrate the two bodies over one decision, checking tag and falls each substep.

    Returns the bodies, the updated ``ended_by`` and the control step at which the episode
    ended. If the episode did not end, that step is ``step0 + decision_substeps``.
    Bodies stop changing once the episode has ended, so a tag keeps the pose it happened in.
    """
    radius = model.tag_radius(theta, rules)
    substep_pair = jax.vmap(lambda b, c, e: model.substep(theta, b, c, e, rules))

    def one_substep(carry, eps_k):
        bodies, ended, end_step, k = carry
        running = ended == RUNNING
        new = substep_pair(bodies, cmd, eps_k)
        bodies = jax.tree_util.tree_map(lambda old, nxt: jnp.where(running, nxt, old), bodies, new)
        tagged = running & (pelvis_distance(bodies) <= radius)
        fell = running & ~tagged & jnp.any(bodies.fallen)
        ended = jnp.where(tagged, TAG, jnp.where(fell, FALL, ended))
        end_step = jnp.where(tagged | fell, step0 + k + 1, end_step)
        return (bodies, ended, end_step, k + 1), None

    init = (bodies, ended_by, step0 + rules.decision_substeps, jnp.int32(0))
    (bodies, ended, end_step, _), _ = lax.scan(one_substep, init, eps)
    return bodies, ended, end_step
