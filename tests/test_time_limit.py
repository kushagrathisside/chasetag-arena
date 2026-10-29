"""The round time limit: one minute by default, or three, five or ten minutes.

A round ends on a tag, on a fall, or when the limit passes. The limit is part of the rules, so a
round is scored and recorded under the rules of its own length. A change waits for the next round
and is refused in a logged session.
"""

from __future__ import annotations

import pytest

from chasetag_arena.arcade.controls import (
    DEFAULT_TIME_LIMIT_S,
    TIME_LIMITS_S,
    TimeLimitControl,
    parse_control,
)
from chasetag_arena.arcade.game_loop import GameLoop, rules_for_limit
from chasetag_arena.core.spec.rules import OFFICIAL
from chasetag_arena.core.state import TIMEOUT


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_default_limit_is_one_minute() -> None:
    assert DEFAULT_TIME_LIMIT_S == 60
    assert TIME_LIMITS_S == (60, 180, 300, 600)
    assert GameLoop(seed=1, selectors=("idle", "idle")).limit_s == 60


def test_rules_for_a_limit_change_only_the_round_length() -> None:
    rules = rules_for_limit(180)
    assert rules.episode_s == 180.0
    assert rules.episode_steps == 18000
    assert rules.tag_radius_m == OFFICIAL.tag_radius_m
    assert rules.fall_height_m == OFFICIAL.fall_height_m
    assert rules.start_separation_m == OFFICIAL.start_separation_m


def test_a_limit_outside_the_options_is_refused() -> None:
    with pytest.raises(ValueError, match="time limit must be one of"):
        rules_for_limit(45)


@pytest.mark.parametrize("value", [45, 120, True, "180", 60.5])
def test_the_control_accepts_only_the_listed_limits(value: object) -> None:
    with pytest.raises(ValueError, match="time limit must be one of"):
        parse_control({"kind": "limit", "value": value})


def test_the_control_parses_a_listed_limit() -> None:
    assert parse_control({"kind": "limit", "value": 300}) == TimeLimitControl(value=300)


def test_a_limit_change_waits_for_the_next_round() -> None:
    clock = _Clock()
    loop = GameLoop(seed=2, selectors=("idle", "idle"), clock=clock)
    loop.submit(TimeLimitControl(value=180))
    assert loop.limit_s == 60
    while loop.state.ended_by == 0:
        loop.decide()
        clock.now += OFFICIAL.decision_dt
    clock.now += 2.0  # past the intermission
    loop.decide()
    assert loop.limit_s == 180


def test_a_round_runs_to_its_limit_when_nobody_tags() -> None:
    clock = _Clock()
    loop = GameLoop(seed=3, selectors=("idle", "idle"), clock=clock)
    decisions = 0
    while int(loop.state.ended_by) == 0 and decisions < 1000:
        loop.decide()
        clock.now += OFFICIAL.decision_dt
        decisions += 1
    assert int(loop.state.ended_by) == TIMEOUT
    assert decisions == 600  # 60 s of game time at 0.1 s a decision


def test_the_serve_command_reads_the_time_limit() -> None:
    from chasetag_arena.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["serve"]).time_limit == 60
    assert parser.parse_args(["serve", "--time-limit", "600"]).time_limit == 600
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--time-limit", "45"])
