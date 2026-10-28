"""The fall hazard of family K (design section 2.3.4).

At each decision the fall probability is a logistic function of the decision's motion:

    lambda = sigmoid(b0 + bv·|v| + ba·|dv|/dt + bw·|omega| + bc·|dc| + bs·exp(-t/ts))

The exponential term covers the higher fall risk just after the start. The priors are
deliberately pessimistic until fitted to data.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from chasetag_arena.core.state import BodyState


def kinematic_fall_probability(
    theta: Any,
    before: BodyState,
    after: BodyState,
    prev_cmd: Array,
    cmd: Array,
    elapsed_s: Array,
    decision_dt: float,
) -> Array:
    """Fall probability for one decision. ``before`` and ``after`` bracket the decision."""
    speed = jnp.linalg.norm(after.vel)
    accel = jnp.linalg.norm(after.vel - before.vel) / decision_dt
    yaw = jnp.abs(after.yaw_rate)
    cmd_change = jnp.linalg.norm(cmd - prev_cmd)
    transient = jnp.exp(-elapsed_s / theta.t_s)
    eta = (
        theta.beta0
        + theta.beta_v * speed
        + theta.beta_a * accel
        + theta.beta_w * yaw
        + theta.beta_c * cmd_change
        + theta.beta_s * transient
    )
    return jax.nn.sigmoid(eta)
