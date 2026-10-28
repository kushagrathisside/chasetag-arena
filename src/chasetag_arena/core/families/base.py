"""The body-model contract, and the pieces every family shares.

A family is a stateless object with pure methods. It takes a parameter pytree ``theta``
(the part a fitter differentiates through) and a ``BodyState``, and returns the next
``BodyState``. ``game.py`` depends only on this protocol, so a new family plugs in
without changes there.

Shared pieces (design section 2.3.3 and body-models section 5.1):

- ``push_delay``: the command delay line, common to every family;
- ``heading_update``: heading follows the direction of motion;
- ``clip_norm`` and ``wrap_angle``: small numerical helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from jax import Array

from chasetag_arena.core.state import BodyState
from chasetag_arena.core.spec.rules import Rules


class BodyModel(Protocol):
    """What a family provides. Methods are pure and must work under jit and vmap."""

    name: str
    aux_dim: int  # length of ``BodyState.aux``
    buffer_len: int  # length of ``BodyState.delay_buf``; must exceed the largest delay

    def init_state(self, theta: Any, pos: Array, heading: Array, rules: Rules) -> BodyState:
        """The body at rest at ``pos`` facing ``heading``."""
        ...

    def substep(
        self, theta: Any, body: BodyState, cmd_world: Array, eps: Array, rules: Rules
    ) -> BodyState:
        """Advance one control step. ``eps`` is (3,) standard normal; see ``dynamics.NOISE_DIM``."""
        ...

    def fall_probability(
        self,
        theta: Any,
        before: BodyState,
        after: BodyState,
        prev_cmd: Array,
        cmd: Array,
        elapsed_s: Array,
        rules: Rules,
    ) -> Array:
        """Per-decision fall probability. Families with deterministic falls return 0 here."""
        ...

    def tag_radius(self, theta: Any, rules: Rules) -> Array:
        """Pelvis distance at or below which a tag happens."""
        ...


@dataclass(frozen=True)
class BodyParams:
    """A family together with its parameters, passed as one argument to the game.

    ``model`` is static (it is part of the pytree structure, so it selects the code path
    under jit). ``theta`` is the dynamic parameter pytree.
    """

    model: BodyModel
    theta: Any


jax.tree_util.register_dataclass(BodyParams, data_fields=["theta"], meta_fields=["model"])


def rest_state(
    pos: Array, heading: Array, height: Array, buffer_len: int, aux_dim: int
) -> BodyState:
    """A body at rest: zero velocity and rate, empty delay line and command history."""
    dtype = jnp.result_type(pos, heading, height)
    return BodyState(
        pos=pos.astype(dtype),
        vel=jnp.zeros(2, dtype=dtype),
        height=jnp.asarray(height, dtype=dtype),
        heading=jnp.asarray(heading, dtype=dtype),
        yaw_rate=jnp.zeros((), dtype=dtype),
        delay_buf=jnp.zeros((buffer_len, 2), dtype=dtype),
        last_cmd=jnp.zeros((2, 2), dtype=dtype),
        fallen=jnp.zeros((), dtype=bool),
        aux=jnp.zeros((aux_dim,), dtype=dtype),
    )


def wrap_angle(angle: Array) -> Array:
    """Map an angle to [-pi, pi)."""
    return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi


def clip_norm(vec: Array, max_norm: Array) -> Array:
    """Scale ``vec`` down so that its norm is at most ``max_norm``."""
    norm = jnp.linalg.norm(vec)
    safe = jnp.where(norm > 0, norm, 1.0)
    return jnp.where(norm > max_norm, vec * (max_norm / safe), vec)


def push_delay(buf: Array, cmd: Array, delay: Array) -> tuple[Array, Array]:
    """Shift the command into the delay line and read the one from ``delay`` substeps ago.

    ``delay`` = 0 returns the current command. The read index is clipped to the buffer, so
    an oversized delay saturates instead of reading out of bounds.
    """
    new_buf = jnp.concatenate([cmd[None, :], buf[:-1]], axis=0)
    idx = jnp.clip(jnp.asarray(delay, dtype=jnp.int32), 0, buf.shape[0] - 1)
    return new_buf, new_buf[idx]


def heading_update(
    heading: Array,
    vel: Array,
    dt: float,
    k_psi: Array,
    omega_max: Array,
    v_min: float,
) -> tuple[Array, Array]:
    """Heading relaxes towards the direction of motion. Returns (heading, yaw rate).

    The yaw rate is ``k_psi * wrap(atan2(v) - psi)``, clipped to ``±omega_max``. It applies
    only while the body moves faster than ``v_min``. A body at rest stops turning.
    """
    speed = jnp.linalg.norm(vel)
    err = wrap_angle(jnp.arctan2(vel[1], vel[0]) - heading)
    omega = jnp.where(
        speed > v_min,
        jnp.clip(k_psi * err, -omega_max, omega_max),
        0.0,
    )
    return wrap_angle(heading + omega * dt), omega
