"""Planar velocity commands: the planner's raw output mapped to a world-frame command.

This is ``chasetag.env.commands.action_to_command`` in JAX. The norm rescale matters:
tanh bounds each component, so the norm can reach sqrt(2) times the maximum speed.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from chasetag_arena.core.spec.frames import to_world_frame


@dataclass(frozen=True)
class CommandSpec:
    max_speed: float = 1.5  # m/s, the largest planar command (rules.max_speed_mps)


def to_command(raw: Array, heading: Array, spec: CommandSpec) -> Array:
    """Map raw actions (..., 2) to world-frame commands (..., 2).

    ``heading`` has the batch shape of ``raw`` without its last axis.
    """
    cmd_h = jnp.tanh(raw) * spec.max_speed
    norm = jnp.linalg.norm(cmd_h, axis=-1, keepdims=True)
    safe = jnp.where(norm > 0, norm, 1.0)
    cmd_h = cmd_h * jnp.where(norm > spec.max_speed, spec.max_speed / safe, 1.0)
    return to_world_frame(cmd_h, heading)
