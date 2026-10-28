"""Pytree states for one body and for the two-seat game.

Both are registered with ``jax.tree_util.register_dataclass``, so they need no extra
dependency and pass through ``jit``, ``vmap`` and ``lax.scan`` unchanged. The per-seat
``BodyState`` leaves carry a leading seat axis of length 2 inside ``GameState``. A batch
of games adds one more leading axis from ``vmap``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array

# Values of ``GameState.ended_by``. An episode ends at most once.
RUNNING = 0
TAG = 1
FALL = 2
TIMEOUT = 3


@dataclass(frozen=True)
class BodyState:
    """One body. ``pos`` and ``height`` are the pelvis, which is what the rules read."""

    pos: Array  # (2,) pelvis planar position, m, world frame
    vel: Array  # (2,) pelvis planar velocity, m/s, world frame
    height: Array  # () pelvis height, m
    heading: Array  # () heading psi, rad, wrapped to (-pi, pi]
    yaw_rate: Array  # () heading rate omega, rad/s
    delay_buf: Array  # (L, 2) commands applied over the last L substeps, newest first
    last_cmd: Array  # (2, 2) last two decision commands, world frame, newest first
    fallen: Array  # () bool, set by a family's own fall rule or by the decision hazard
    aux: Array  # (A,) family-specific state; shape (0,) for family K


@dataclass(frozen=True)
class GameState:
    """The two-seat game. ``step`` is the control-step count at the episode's end, or so far."""

    bodies: BodyState  # leaves have shape (2, ...): seat 0 then seat 1
    chaser: Array  # () int32, seat index of the chaser this episode
    step: Array  # () int32, control steps elapsed
    ended_by: Array  # () int32, one of RUNNING, TAG, FALL, TIMEOUT
    key: Array  # (2,) uint32 PRNG key

    @property
    def done(self) -> Array:
        return self.ended_by != RUNNING


_BODY_FIELDS = (
    "pos",
    "vel",
    "height",
    "heading",
    "yaw_rate",
    "delay_buf",
    "last_cmd",
    "fallen",
    "aux",
)
_GAME_FIELDS = ("bodies", "chaser", "step", "ended_by", "key")

jax.tree_util.register_dataclass(BodyState, data_fields=list(_BODY_FIELDS), meta_fields=[])
jax.tree_util.register_dataclass(GameState, data_fields=list(_GAME_FIELDS), meta_fields=[])
