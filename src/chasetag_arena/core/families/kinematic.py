"""Family K: the kinematic dot with limits, lag, delay, noise and a logistic fall hazard.

This is the baseline body (design section 2.3.3, body-models sections 5.1 and 5.2). Its
fitted parameters live in ``KParams``. The family's own settings, which are not fitted and
change the code path, live on ``KinematicFamily``.

Priors in ``default_params`` are placeholders for a pessimistic human-gait profile
(plan section 1.6, question 2). They are not fitted to any body. The hazard priors were
calibrated on 512 games of 20 s through ``game.step`` (rates in ``default_params``).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from chasetag_arena.core.arena import resolve_fence
from chasetag_arena.core.families.base import (
    BodyParams,
    clip_norm,
    heading_update,
    push_delay,
    rest_state,
)
from chasetag_arena.core.hazard import kinematic_fall_probability
from chasetag_arena.core.state import BodyState
from chasetag_arena.core.spec.frames import to_heading_frame, to_world_frame
from chasetag_arena.core.spec.rules import Rules


@dataclass(frozen=True)
class KParams:
    """Fitted parameters of family K. Speeds in m/s, accelerations in m/s², times in s."""

    v_fwd: Array  # forward speed limit, heading frame
    v_back: Array  # backward speed limit
    v_lat: Array  # lateral speed limit
    a_max: Array  # acceleration limit while speeding up
    a_dec: Array  # acceleration limit while slowing down
    tau: Array  # first-order tracking time constant
    delay: Array  # command delay, in control substeps (int)
    sigma_v: Array  # tracking noise, (m/s)/sqrt(s)
    k_psi: Array  # heading gain, 1/s
    omega_max: Array  # yaw rate limit, rad/s
    h0: Array  # standing pelvis height, m
    tag_margin: Array  # tag radius reduced by this (sway allowance), m
    beta0: Array  # hazard intercept
    beta_v: Array  # hazard weight on speed
    beta_a: Array  # hazard weight on acceleration
    beta_w: Array  # hazard weight on yaw rate
    beta_c: Array  # hazard weight on command change
    beta_s: Array  # hazard weight on the post-start transient
    t_s: Array  # transient time scale, s


jax.tree_util.register_dataclass(
    KParams, data_fields=[f.name for f in fields(KParams)], meta_fields=[]
)


@dataclass(frozen=True)
class KinematicFamily:
    """Family K's fixed settings. They are part of the model's identity, not fitted."""

    name: str = "K"
    aux_dim: int = 0
    buffer_len: int = 32  # substeps of command history; must exceed the largest delay
    v_min: float = 0.05  # m/s; below this the body does not turn
    sigma_h: float = 0.005  # m, standing pelvis height noise

    def init_state(self, theta: KParams, pos: Array, heading: Array, rules: Rules) -> BodyState:
        return rest_state(pos, heading, theta.h0, self.buffer_len, self.aux_dim)

    def substep(
        self, theta: KParams, body: BodyState, cmd_world: Array, eps: Array, rules: Rules
    ) -> BodyState:
        dt = rules.control_dt
        # 1. Limit the command in the heading frame, then rotate it back with the heading.
        c_h = to_heading_frame(cmd_world, body.heading)
        c_lim = jnp.stack(
            [
                jnp.clip(c_h[0], -theta.v_back, theta.v_fwd),
                jnp.clip(c_h[1], -theta.v_lat, theta.v_lat),
            ]
        )
        c_lim_w = to_world_frame(c_lim, body.heading)
        # 2. Delay: the body sees the command from `delay` substeps earlier.
        delay_buf, c_del = push_delay(body.delay_buf, c_lim_w, theta.delay)
        # 3. First-order tracking with separate acceleration and deceleration limits.
        delta = (c_del - body.vel) * (dt / theta.tau)
        accelerating = jnp.dot(delta, body.vel) >= 0
        a_lim = jnp.where(accelerating, theta.a_max, theta.a_dec)
        vel = body.vel + clip_norm(delta, a_lim * dt)
        # 4. Tracking noise.
        vel = vel + theta.sigma_v * jnp.sqrt(dt) * eps[:2]
        # 5. Heading follows the direction of motion.
        heading, yaw_rate = heading_update(
            body.heading, vel, dt, theta.k_psi, theta.omega_max, self.v_min
        )
        # 6. Position, then the fence.
        pos, vel = resolve_fence(body.pos + vel * dt, vel, rules.arena_half_m)
        # 7. Pelvis height: standing, with small noise.
        height = theta.h0 + self.sigma_h * eps[2]
        return BodyState(
            pos=pos,
            vel=vel,
            height=height,
            heading=heading,
            yaw_rate=yaw_rate,
            delay_buf=delay_buf,
            last_cmd=body.last_cmd,
            fallen=body.fallen,
            aux=body.aux,
        )

    def fall_probability(
        self,
        theta: KParams,
        before: BodyState,
        after: BodyState,
        prev_cmd: Array,
        cmd: Array,
        elapsed_s: Array,
        rules: Rules,
    ) -> Array:
        return kinematic_fall_probability(
            theta, before, after, prev_cmd, cmd, elapsed_s, rules.decision_dt
        )

    def tag_radius(self, theta: KParams, rules: Rules) -> Array:
        return rules.tag_radius_m - theta.tag_margin


KINEMATIC = KinematicFamily()


def default_params(dtype: DTypeLike | None = None) -> BodyParams:
    """Placeholder prior: pessimistic human-gait limits, a 100 ms delay and a fall hazard.

    The hazard was calibrated so that, over 20 s episodes of 512 games each, with the
    official rules: an idle body falls in about 1.0% of episodes (target below 2%), uniform
    random raw actions end about 16% of episodes in a fall (target below 20%), and bang-bang
    reversals end about 37% in a fall. The bang-bang rate is more than twice the random one.
    """

    def _f(x: float) -> Array:
        return jnp.asarray(x, dtype=dtype)

    theta = KParams(
        v_fwd=_f(1.5),
        v_back=_f(1.0),
        v_lat=_f(1.0),
        a_max=_f(2.0),
        a_dec=_f(3.0),
        tau=_f(0.3),
        delay=jnp.asarray(10, dtype=jnp.int32),
        sigma_v=_f(0.05),
        k_psi=_f(4.0),
        omega_max=_f(3.0),
        h0=_f(0.95),
        tag_margin=_f(0.0),
        beta0=_f(-9.5),
        beta_v=_f(0.1),
        beta_a=_f(0.2),
        beta_w=_f(0.3),
        beta_c=_f(0.5),
        beta_s=_f(0.5),
        t_s=_f(1.0),
    )
    return BodyParams(model=KINEMATIC, theta=theta)


__all__ = ["KINEMATIC", "KParams", "KinematicFamily", "default_params"]
