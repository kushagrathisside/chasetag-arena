"""Planar rotations between the world frame and a body's heading frame.

These match ``chasetag.env.features.to_heading_frame`` and ``to_world_frame`` exactly.
They accept any leading batch shape, and the heading may be a scalar or broadcastable.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array


def to_heading_frame(vec_xy: Array, heading: Array) -> Array:
    """Rotate world-frame planar vectors (..., 2) by -heading into the body frame."""
    c, s = jnp.cos(heading), jnp.sin(heading)
    x, y = vec_xy[..., 0], vec_xy[..., 1]
    return jnp.stack([c * x + s * y, -s * x + c * y], axis=-1)


def to_world_frame(vec_xy: Array, heading: Array) -> Array:
    """Rotate heading-frame planar vectors (..., 2) by +heading into the world frame."""
    return to_heading_frame(vec_xy, -heading)
