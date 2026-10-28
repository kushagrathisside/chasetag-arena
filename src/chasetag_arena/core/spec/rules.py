"""Official game rules, as read from MyoSuite 3.0.0 ``ChaseTagVsTaskConfig``.

The dataclass is frozen so it can be a static argument to ``jax.jit``. Each field keeps
the official value, so a variant (a 4 m start, a larger tag radius) is a new instance
and never an edit to ``OFFICIAL``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    """Episode, tag, fall, start and arena rules shared by every body family."""

    episode_s: float = 20.0
    control_dt: float = 0.01
    decision_substeps: int = 10  # control steps per planner decision (10 Hz)
    tag_radius_m: float = 0.5  # tag when the pelvis distance is at most this
    fall_height_m: float = 0.6  # pelvis below this is a fall
    start_separation_m: float = 2.0
    arena_half_m: float = 6.0  # fence centre lines span y in [-6, 6]
    max_speed_mps: float = 1.5  # largest planar velocity command

    @property
    def episode_steps(self) -> int:
        return round(self.episode_s / self.control_dt)

    @property
    def decision_dt(self) -> float:
        return self.decision_substeps * self.control_dt

    @property
    def decisions_per_episode(self) -> int:
        return self.episode_steps // self.decision_substeps


OFFICIAL = Rules()
