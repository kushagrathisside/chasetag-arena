"""Family LS: a linear inverted pendulum with stepping (body-models section 5.4).

The centre of mass ``x`` moves at constant height ``z0`` over a point foot ``p``, with
``x'' = omega² (x - p)`` and ``omega = sqrt(g / z0)``. Within a substep the motion is
propagated exactly, using the closed form, so there is no integration error.

Every ``T`` seconds the stance foot is replaced by the deadbeat capture-point rule:

    p_next = xi - b(v) - side * (w/2) * n,   b(v) = v T / (e^(omega T) - 1),

where ``n`` is the body's left-hand lateral axis and ``side`` alternates. The target is then
clipped onto the reach ellipse around the centre of mass. The capture point is
``xi = x + x'/omega``.

Falls come from loss of capturability. The capture offset ``xi - p`` must satisfy
``|xi - p| e^(-omega t) <= d_inf``, in reach-ellipse units, where
``d_inf = 1 / (e^(omega T) - 1)``. The factor ``e^(-omega t)`` scales the test back to the
step start, where the bound was derived. Once the test fails, the pelvis collapses to the
official fall height over ``collapse_s``, and the fall flag is set when it crosses.

Body state: ``pos`` and ``vel`` are the centre of mass. ``BodyState.aux`` holds
``[p_x, p_y, t_step, side, collapse_time]``, with ``collapse_time < 0`` while standing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from chasetag_arena.core.arena import resolve_fence
from chasetag_arena.core.families.base import BodyParams, push_delay, rest_state
from chasetag_arena.core.families.spring import HeadingParams, heading_update_2nd
from chasetag_arena.core.state import BodyState
from chasetag_arena.core.spec.frames import to_heading_frame, to_world_frame
from chasetag_arena.core.spec.rules import Rules


@dataclass(frozen=True)
class LSParams:
    """Fitted parameters of family LS. Lengths in m, times in s."""

    z0: Array  # standing pelvis (centre-of-mass) height, m
    T: Array  # step time, s
    w: Array  # step width, m
    r_f: Array  # forward reach, m
    r_l: Array  # lateral reach, m
    sigma_v: Array  # centre-of-mass velocity noise, (m/s)/sqrt(s)
    delay: Array  # command delay, in control substeps (int)
    tag_margin: Array  # tag radius reduced by this, m
    heading: HeadingParams


jax.tree_util.register_dataclass(
    LSParams, data_fields=[f.name for f in fields(LSParams)], meta_fields=[]
)


def _gauge(vec: Array, heading: Array, theta: LSParams) -> Array:
    """The reach-ellipse norm of a planar offset: 1 on the reach ellipse, scale-free otherwise."""
    fwd, lat = to_heading_frame(vec, heading)
    return jnp.sqrt((fwd / theta.r_f) ** 2 + (lat / theta.r_l) ** 2)


def _reach_clip(target: Array, com: Array, heading: Array, theta: LSParams) -> Array:
    """Pull a foot target onto the reach ellipse around the centre of mass, if it lies outside."""
    delta = target - com
    gauge = _gauge(delta, heading, theta)
    return com + jnp.where(gauge > 1.0, delta / jnp.where(gauge > 0, gauge, 1.0), delta)


@dataclass(frozen=True)
class SteppingFamily:
    """Family LS's fixed settings. They are part of the model's identity, not fitted."""

    name: str = "LS"
    aux_dim: int = 5  # [p_x, p_y, t_step, side, collapse_time]
    buffer_len: int = 32  # substeps of command history; must exceed the largest delay
    # Heading turns only above this speed. In-place stepping sways at about 0.5 m/s, and a
    # body released from a capture offset can briefly exceed 1 m/s. A lower gate lets the
    # heading turn during that sway and lock off-axis, which leaves a stopped body drifting.
    v_min: float = 0.8
    g: float = 9.81  # m/s²
    collapse_s: float = 0.2  # time for the pelvis to fall from standing height to the floor

    def init_state(self, theta: LSParams, pos: Array, heading: Array, rules: Rules) -> BodyState:
        state = rest_state(pos, heading, theta.z0, self.buffer_len, self.aux_dim)
        # The stance foot starts under the centre of mass, and the first step is to the left.
        aux = jnp.concatenate([pos, jnp.zeros(1), jnp.ones(1), -jnp.ones(1)]).astype(
            state.aux.dtype
        )
        return replace(state, aux=aux)

    def substep(
        self, theta: LSParams, body: BodyState, cmd_world: Array, eps: Array, rules: Rules
    ) -> BodyState:
        dt = rules.control_dt
        omega = jnp.sqrt(self.g / theta.z0)
        delay_buf, c_del = push_delay(body.delay_buf, cmd_world, theta.delay)

        foot = body.aux[0:2]
        t_step, side, collapse = body.aux[2], body.aux[3], body.aux[4]

        # 1. Exact propagation of the pendulum about the stance foot, plus velocity noise.
        rel = body.pos - foot
        ch, sh = jnp.cosh(omega * dt), jnp.sinh(omega * dt)
        pos = foot + rel * ch + (body.vel / omega) * sh
        vel = rel * (omega * sh) + body.vel * ch + theta.sigma_v * jnp.sqrt(dt) * eps[:2]

        # 2. At the end of a step, place the next foot on the deadbeat capture-point rule.
        t_next = t_step + dt
        boundary = t_next >= theta.T - 0.5 * dt
        capture = pos + vel / omega
        b_vec = c_del * (theta.T / jnp.expm1(omega * theta.T))
        left = to_world_frame(jnp.array([0.0, 1.0], dtype=pos.dtype), body.heading)
        target = capture - b_vec - side * (theta.w / 2.0) * left
        placed = _reach_clip(target, pos, body.heading, theta)
        foot = jnp.where(boundary, placed, foot)
        side = jnp.where(boundary, -side, side)
        t_next = jnp.where(boundary, 0.0, t_next)

        # 3. The arena fence holds the centre of mass, then the heading follows the motion.
        pos, vel = resolve_fence(pos, vel, rules.arena_half_m)
        heading, yaw = heading_update_2nd(
            body.heading, body.yaw_rate, vel, dt, theta.heading, self.v_min
        )

        # 4. Capturability. The offset is scaled back to the step start, where d_inf applies.
        capture = pos + vel / omega
        d_inf = 1.0 / jnp.expm1(omega * theta.T)
        exceeded = _gauge(capture - foot, heading, theta) * jnp.exp(-omega * t_next) > d_inf
        collapse = jnp.where(collapse >= 0, collapse + dt, jnp.where(exceeded, 0.0, -1.0))

        # 5. Pelvis height: standing, then collapsing to the floor after a loss of capture.
        floor = 0.5 * rules.fall_height_m
        drop = (theta.z0 - floor) * jnp.minimum(collapse / self.collapse_s, 1.0)
        height = jnp.where(collapse >= 0, theta.z0 - drop, theta.z0)
        fallen = body.fallen | (height < rules.fall_height_m)

        return BodyState(
            pos=pos,
            vel=vel,
            height=height,
            heading=heading,
            yaw_rate=yaw,
            delay_buf=delay_buf,
            last_cmd=body.last_cmd,
            fallen=fallen,
            aux=jnp.stack([foot[0], foot[1], t_next, side, collapse]),
        )

    def fall_probability(
        self,
        theta: LSParams,
        before: BodyState,
        after: BodyState,
        prev_cmd: Array,
        cmd: Array,
        elapsed_s: Array,
        rules: Rules,
    ) -> Array:
        # Capturability is deterministic and is checked per substep, so there is no hazard.
        return jnp.zeros((), dtype=before.pos.dtype)

    def tag_radius(self, theta: LSParams, rules: Rules) -> Array:
        return rules.tag_radius_m - theta.tag_margin


STEPPING = SteppingFamily()


def default_params(dtype: DTypeLike | None = None) -> BodyParams:
    """Human-scale priors. The physics tests check them.

    At z0 = 0.95 m and T = 0.5 s, omega = 3.21 1/s and d_inf = 0.25 in reach units. With a
    0.15 m step width and reach 0.9 m forward and 0.7 m lateral, the capture offset of a
    steady gait stays inside d_inf up to about 1.6 m/s forward and 0.8 m/s sideways.
    """

    def _f(x: float) -> Array:
        return jnp.asarray(x, dtype=dtype)

    theta = LSParams(
        z0=_f(0.95),
        T=_f(0.5),
        w=_f(0.15),
        r_f=_f(0.9),
        r_l=_f(0.7),
        sigma_v=_f(0.05),
        delay=jnp.asarray(10, dtype=jnp.int32),
        tag_margin=_f(0.0),
        heading=HeadingParams(k_p=_f(4.0), k_d=_f(4.0), alpha_max=_f(10.0), omega_max=_f(3.0)),
    )
    return BodyParams(model=STEPPING, theta=theta)


__all__ = [
    "STEPPING",
    "LSParams",
    "SteppingFamily",
    "default_params",
]
