"""The fenced arena: an axis-aligned square, and the official start layout.

The fence is modelled as a hard wall at ``±arena_half_m``. A body that reaches it stops
there, and the outward velocity component is removed. Agent-to-agent collision is not
modelled (design section 2.2).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from chasetag_arena.core.spec.rules import Rules


def resolve_fence(pos: Array, vel: Array, half: float | Array) -> tuple[Array, Array]:
    """Clamp a planar position to the arena and zero any velocity pushing into a wall."""
    clamped = jnp.clip(pos, -half, half)
    hit = clamped != pos
    outward = hit & (vel * jnp.sign(clamped) > 0)
    return clamped, jnp.where(outward, 0.0, vel)


def start_layout(rules: Rules, dtype: DTypeLike | None = None) -> tuple[Array, Array]:
    """Official starts: seat 0 at (0, +s/2) facing -y, seat 1 at (0, -s/2) facing +y.

    This follows ``default_root_offsets`` in MyoSuite, where the two agents face each
    other along the Y axis. Returns positions (2, 2) and headings (2,).
    """
    half = rules.start_separation_m / 2.0
    pos = jnp.array([[0.0, half], [0.0, -half]], dtype=dtype)
    heading = jnp.array([-jnp.pi / 2, jnp.pi / 2], dtype=dtype)
    return pos, heading
