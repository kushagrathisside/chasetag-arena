"""Strategy features for the planner, in the same order as ``chasetag``'s ``Featurizer.strategy``.

Every quantity is expressed in the seat's heading frame, so a policy learned facing one
way transfers to every other. Spec v1 is 17-dimensional and must stay identical to
chasetag, because the exported planner reads chasetag's observation. Spec v2 appends the
last two decision commands (4 dimensions) to give a feed-forward planner the memory that
the body's command lag needs (design section 2.4).

The encoders are pure JAX functions of a game state and a seat index. They read only
``state.bodies`` and ``state.chaser``, so they work with any body family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from chasetag_arena.core.spec.frames import to_heading_frame
from chasetag_arena.core.spec.rules import OFFICIAL, Rules

if TYPE_CHECKING:
    from chasetag_arena.core.state import GameState

_V1_NAMES = (
    "role_chaser",
    "role_runner",
    "opp_rel_x_h",
    "opp_rel_y_h",
    "opp_rel_vx_h",
    "opp_rel_vy_h",
    "dist",
    "own_x_norm",
    "own_y_norm",
    "opp_x_norm",
    "opp_y_norm",
    "cos_heading",
    "sin_heading",
    "own_vx_h",
    "own_vy_h",
    "pelvis_height",
    "time_frac",
)
_V2_EXTRA_NAMES = (
    "last_cmd_1_x_h",
    "last_cmd_1_y_h",
    "last_cmd_2_x_h",
    "last_cmd_2_y_h",
)


def encode_v1(state: GameState, seat: Array | int, rules: Rules = OFFICIAL) -> Array:
    """The 17-dimensional chasetag strategy vector for ``seat``."""
    b = state.bodies
    opp = 1 - seat
    dtype = b.pos.dtype
    own_xy, opp_xy = b.pos[seat], b.pos[opp]
    own_vel, opp_vel = b.vel[seat], b.vel[opp]
    psi = b.heading[seat]

    rel_xy = opp_xy - own_xy
    dz = b.height[opp] - b.height[seat]
    dist = jnp.sqrt(jnp.sum(rel_xy**2) + dz**2)
    is_chaser = state.chaser == seat
    role = jnp.where(
        is_chaser,
        jnp.array([1.0, 0.0], dtype=dtype),
        jnp.array([0.0, 1.0], dtype=dtype),
    )
    time_frac = state.step.astype(dtype) * (rules.control_dt / rules.episode_s)
    half_arena = rules.arena_half_m
    return jnp.concatenate(
        [
            role,
            to_heading_frame(rel_xy, psi),
            to_heading_frame(opp_vel - own_vel, psi),
            dist[None],
            own_xy / half_arena,
            opp_xy / half_arena,
            jnp.stack([jnp.cos(psi), jnp.sin(psi)]),
            to_heading_frame(own_vel, psi),
            b.height[seat][None],
            time_frac[None],
        ]
    )


def encode_v2(state: GameState, seat: Array | int, rules: Rules = OFFICIAL) -> Array:
    """Spec v1 plus the last two decision commands, in the seat's current heading frame."""
    psi = state.bodies.heading[seat]
    last = to_heading_frame(state.bodies.last_cmd[seat], psi).reshape(-1)
    return jnp.concatenate([encode_v1(state, seat, rules), last])


@dataclass(frozen=True)
class FeatureSpec:
    """A named, versioned feature encoder. ``names`` and ``dim`` describe its output."""

    version: int
    names: tuple[str, ...]
    encode: Callable[..., Array]

    @property
    def dim(self) -> int:
        return len(self.names)


SPEC_V1 = FeatureSpec(version=1, names=_V1_NAMES, encode=encode_v1)
SPEC_V2 = FeatureSpec(version=2, names=_V1_NAMES + _V2_EXTRA_NAMES, encode=encode_v2)
