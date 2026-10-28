"""Family SP: a spring pendulum on a wheeled base (body-models section 5.3).

A wheeled base at ``b`` follows the command through a velocity loop with an acceleration
limit. A point mass, the pelvis, hangs from the base on a massless rod of length ``l``, held
upright by a torsional spring and damper. The rod tilt is a vector ``theta`` with one
component per horizontal axis. The equations are non-linear, and their normalised parameters
are ``kappa = k/(m l^2)`` and ``gamma = c/(m l^2)``.

Pelvis state comes from the geometry, ``x = b + l sin|theta| theta_hat`` and
``h = l cos|theta|``. The fall rule is the tilt limit ``|theta| > arccos(h_fall / l)``.
The base carries the arena fence, and the pelvis follows it.

Body state: ``BodyState.aux`` holds ``[b (2), b_dot (2), theta (2), theta_dot (2)]``.
``BodyState.pos`` and ``vel`` are the pelvis, which is what the rules read.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from chasetag_arena.core.arena import resolve_fence
from chasetag_arena.core.families.base import (
    BodyParams,
    clip_norm,
    push_delay,
    rest_state,
    wrap_angle,
)
from chasetag_arena.core.state import BodyState
from chasetag_arena.core.spec.rules import Rules


@dataclass(frozen=True)
class HeadingParams:
    """The second-order heading law of body-models section 5.1, shared by SP and LS.

    The yaw acceleration is ``k_p * wrap(atan2(v) - psi) - k_d * yaw_rate``, clipped to
    ``±alpha_max``. The yaw rate is clipped to ``±omega_max``.
    """

    k_p: Array  # 1/s², steering stiffness
    k_d: Array  # 1/s, yaw damping
    alpha_max: Array  # rad/s², yaw acceleration limit
    omega_max: Array  # rad/s, yaw rate limit


jax.tree_util.register_dataclass(
    HeadingParams, data_fields=[f.name for f in fields(HeadingParams)], meta_fields=[]
)


@dataclass(frozen=True)
class SPParams:
    """Fitted parameters of family SP. Lengths in m, times in s."""

    length: Array  # rod length (pelvis height when upright), m
    kappa: Array  # k/(m l^2), 1/s²; the upright equilibrium is stable when kappa > g/l
    gamma: Array  # c/(m l^2), 1/s
    k_v: Array  # base velocity loop gain, 1/s
    a_max: Array  # base acceleration limit, m/s²
    sigma_v: Array  # base velocity noise, (m/s)/sqrt(s)
    delay: Array  # command delay, in control substeps (int)
    tag_margin: Array  # tag radius reduced by this, m
    heading: HeadingParams


jax.tree_util.register_dataclass(
    SPParams, data_fields=[f.name for f in fields(SPParams)], meta_fields=[]
)


def _norm(vec: Array) -> Array:
    """Euclidean norm with a zero gradient at the origin, not NaN."""
    sq = jnp.sum(vec * vec)
    pos = sq > 0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, sq, 1.0)), 0.0)


def pelvis_offset(tilt: Array, length: Array) -> Array:
    """The pelvis offset from the base, ``l sin|theta| theta_hat``.

    Written as ``l s(r) theta`` with ``s(r) = sin(r)/r``, which is 1 at the origin.
    """
    r = _norm(tilt)
    pos = r > 0
    safe = jnp.where(pos, r, 1.0)
    s = jnp.where(pos, jnp.sin(safe) / safe, 1.0)
    return length * s * tilt


def heading_update_2nd(
    heading: Array,
    yaw_rate: Array,
    vel: Array,
    dt: float,
    hp: HeadingParams,
    v_min: float,
) -> tuple[Array, Array]:
    """Second-order heading law. Returns (heading, yaw rate).

    The steering term acts only while the body moves faster than ``v_min``. Yaw damping
    always acts, so a body that stops keeps no yaw rate.
    """
    speed = jnp.linalg.norm(vel)
    err = wrap_angle(jnp.arctan2(vel[1], vel[0]) - heading)
    steer = jnp.where(speed > v_min, hp.k_p * err, 0.0)
    acc = jnp.clip(steer - hp.k_d * yaw_rate, -hp.alpha_max, hp.alpha_max)
    yaw = jnp.clip(yaw_rate + acc * dt, -hp.omega_max, hp.omega_max)
    return wrap_angle(heading + yaw * dt), yaw


@dataclass(frozen=True)
class SpringFamily:
    """Family SP's fixed settings. They are part of the model's identity, not fitted."""

    name: str = "SP"
    aux_dim: int = 8  # [b (2), b_dot (2), theta (2), theta_dot (2)]
    buffer_len: int = 32  # substeps of command history; must exceed the largest delay
    v_min: float = 0.05  # m/s; the steering law is inactive below this
    g: float = 9.81  # m/s²

    def init_state(self, theta: SPParams, pos: Array, heading: Array, rules: Rules) -> BodyState:
        state = rest_state(pos, heading, theta.length, self.buffer_len, self.aux_dim)
        return replace(state, aux=state.aux.at[0:2].set(state.pos))

    def substep(
        self, theta: SPParams, body: BodyState, cmd_world: Array, eps: Array, rules: Rules
    ) -> BodyState:
        dt = rules.control_dt
        base, base_vel = body.aux[0:2], body.aux[2:4]
        tilt, tilt_rate = body.aux[4:6], body.aux[6:8]

        # 1. Delay, then the base velocity loop with an acceleration limit on its norm.
        delay_buf, c_del = push_delay(body.delay_buf, cmd_world, theta.delay)
        acc = clip_norm(theta.k_v * (c_del - base_vel), theta.a_max)
        base_vel = base_vel + acc * dt + theta.sigma_v * jnp.sqrt(dt) * eps[:2]
        base = base + base_vel * dt
        base, base_vel = resolve_fence(base, base_vel, rules.arena_half_m)

        # 2. Tilt, one equation per axis. Semi-implicit Euler.
        tilt_acc = (
            (self.g / theta.length) * jnp.sin(tilt)
            - theta.kappa * tilt
            - theta.gamma * tilt_rate
            - (jnp.cos(tilt) / theta.length) * acc
        )
        tilt_rate = tilt_rate + tilt_acc * dt
        tilt = tilt + tilt_rate * dt

        # 3. Pelvis from the geometry. The rate is the exact derivative of the offset.
        offset, offset_rate = jax.jvp(
            lambda q: pelvis_offset(q, theta.length), (tilt,), (tilt_rate,)
        )
        pos = base + offset
        vel = base_vel + offset_rate
        r = _norm(tilt)
        height = theta.length * jnp.cos(r)

        # 4. Heading, then the tilt fall rule. A fall is sticky.
        heading, yaw = heading_update_2nd(
            body.heading, body.yaw_rate, vel, dt, theta.heading, self.v_min
        )
        theta_fall = jnp.arccos(jnp.minimum(rules.fall_height_m / theta.length, 1.0))
        fallen = body.fallen | (r > theta_fall)

        return BodyState(
            pos=pos,
            vel=vel,
            height=height,
            heading=heading,
            yaw_rate=yaw,
            delay_buf=delay_buf,
            last_cmd=body.last_cmd,
            fallen=fallen,
            aux=jnp.concatenate([base, base_vel, tilt, tilt_rate]),
        )

    def fall_probability(
        self,
        theta: SPParams,
        before: BodyState,
        after: BodyState,
        prev_cmd: Array,
        cmd: Array,
        elapsed_s: Array,
        rules: Rules,
    ) -> Array:
        # The tilt rule is deterministic and is checked per substep, so there is no hazard.
        return jnp.zeros((), dtype=before.pos.dtype)

    def tag_radius(self, theta: SPParams, rules: Rules) -> Array:
        return rules.tag_radius_m - theta.tag_margin


SPRING = SpringFamily()


def default_params(dtype: DTypeLike | None = None) -> BodyParams:
    """Human-scale priors: a 0.95 m rod, natural frequency 2.1 rad/s and damping ratio 0.4.

    With ``omega_n² = kappa - g/l`` and ``zeta = gamma / (2 omega_n)``, these give
    ``kappa = omega_n² + g/l ≈ 4.41 + g/l`` and ``gamma = 2 zeta omega_n = 1.68``.

    The natural frequency is set from the safe base acceleration. For a pelvis at the fall
    height, the tilt limit is ``theta_f = arccos(0.6 / 0.95) ≈ 0.887`` rad, and the
    linear lean that a constant base acceleration ``a`` produces at the damped peak gives
    ``a_safe ≈ l omega_n² theta_f / (1 + exp(-zeta pi / sqrt(1 - zeta²)))``. At
    ``omega_n = 2.1`` this is about 2.96 m/s², which matches ``a_max = 3``. At the earlier
    3 rad/s it was about 6.0 m/s², so a 3 m/s² base never tipped the rod.

    The base velocity loop has gain 4 1/s and an acceleration limit of 3 m/s². The steering
    uses ``k_p = 4``, ``k_d = 4``, ``alpha_max = 10`` and ``omega_max = 3``.
    """

    def _f(x: float) -> Array:
        return jnp.asarray(x, dtype=dtype)

    rod = 0.95
    omega_n, zeta = 2.1, 0.4
    g = SPRING.g
    theta = SPParams(
        length=_f(rod),
        kappa=_f(omega_n**2 + g / rod),
        gamma=_f(2.0 * zeta * omega_n),
        k_v=_f(4.0),
        a_max=_f(3.0),
        sigma_v=_f(0.05),
        delay=jnp.asarray(10, dtype=jnp.int32),
        tag_margin=_f(0.0),
        heading=HeadingParams(k_p=_f(4.0), k_d=_f(4.0), alpha_max=_f(10.0), omega_max=_f(3.0)),
    )
    return BodyParams(model=SPRING, theta=theta)


__all__ = [
    "SPRING",
    "HeadingParams",
    "SPParams",
    "SpringFamily",
    "default_params",
    "heading_update_2nd",
    "pelvis_offset",
]
