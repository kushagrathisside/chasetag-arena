"""Control parsing is strict, and the keyboard command expires."""

from __future__ import annotations

import math

import pytest

from chasetag_arena.arcade.controls import (
    KEYBOARD_STALE_S,
    KeyboardControl,
    KeyboardState,
    PauseControl,
    PlayControl,
    RestartControl,
    SelectorControl,
    SpeedControl,
    StepControl,
    parse_control,
)


@pytest.mark.parametrize(
    ("obj", "expected"),
    [
        ({"kind": "selector", "seat": 0, "value": "greedy"}, SelectorControl(0, "greedy")),
        ({"kind": "selector", "seat": 1, "value": "keyboard"}, SelectorControl(1, "keyboard")),
        ({"kind": "keyboard", "seat": 0, "cmd": [0.5, -1.0]}, KeyboardControl(0, (0.5, -1.0))),
        ({"kind": "pause"}, PauseControl()),
        ({"kind": "play"}, PlayControl()),
        ({"kind": "step"}, StepControl()),
        ({"kind": "restart"}, RestartControl()),
        ({"kind": "speed", "value": 2}, SpeedControl(2.0)),
    ],
)
def test_valid_controls_parse(obj: dict, expected: object) -> None:
    assert parse_control(obj) == expected


@pytest.mark.parametrize(
    "obj",
    [
        "pause",
        None,
        {},
        {"kind": "teleport"},
        {"kind": "pause", "seat": 0},  # unknown field
        {"kind": "selector", "seat": 2, "value": "greedy"},  # seat out of range
        {"kind": "selector", "seat": True, "value": "greedy"},  # bool is not a seat
        {"kind": "selector", "seat": 0, "value": "telepathy"},  # unknown selector
        {"kind": "keyboard", "seat": 0, "cmd": [1.0]},
        {"kind": "keyboard", "seat": 0, "cmd": [math.nan, 0.0]},
        {"kind": "keyboard", "seat": 0, "cmd": [9.0, 0.0]},
        {"kind": "speed", "value": 0},
        {"kind": "speed", "value": 100},
        {"kind": "speed", "value": math.inf},
    ],
)
def test_invalid_controls_raise(obj: object) -> None:
    with pytest.raises(ValueError):
        parse_control(obj)


def test_keyboard_command_holds_then_expires_after_a_quarter_second() -> None:
    assert KEYBOARD_STALE_S == 0.25
    keys = KeyboardState()
    keys.update(0, (1.0, 0.5), t_received=10.0)
    assert keys.current(0, now=10.2) == (1.0, 0.5)
    assert keys.current(0, now=10.26) == (0.0, 0.0)


def test_keyboard_update_replaces_only_its_seat() -> None:
    keys = KeyboardState()
    keys.update(0, (1.0, 0.0), t_received=0.0)
    keys.update(1, (0.0, 1.0), t_received=0.0)
    keys.update(0, (0.0, 0.0), t_received=0.1)
    assert keys.current(0, now=0.1) == (0.0, 0.0)
    assert keys.current(1, now=0.1) == (0.0, 1.0)


def test_unset_keyboard_seat_is_zero() -> None:
    assert KeyboardState().current(1, now=0.0) == (0.0, 0.0)
