"""The game thread: one two-seat game, decided in steps of game time, streamed to subscribers.

``decide`` is the whole of one decision: apply queued controls, start a round if the last one
has waited out its intermission, choose both seats' actions, step the core, and publish the
frame. ``run`` is the thread that calls it at the right wall-clock times. It parks while no
viewer is subscribed, so an unwatched game uses no CPU.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Final, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from chasetag_arena.agents.bots import BOTS, Bot, raw_for_command
from chasetag_arena.arcade.controls import (
    DEFAULT_TIME_LIMIT_S,
    SELECTORS,
    TIME_LIMITS_S,
    Control,
    KeyboardControl,
    KeyboardState,
    PauseControl,
    PlayControl,
    RestartControl,
    SelectorControl,
    SpeedControl,
    StepControl,
    TimeLimitControl,
)
from chasetag_arena.arcade.frames import Phase, state_to_frame
from chasetag_arena.core import game
from chasetag_arena.core.families.base import BodyParams
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.spec.features import encode_v1
from chasetag_arena.core.spec.frames import to_heading_frame
from chasetag_arena.core.spec.rules import OFFICIAL, Rules
from chasetag_arena.core.state import FALL, RUNNING, TAG, TIMEOUT, GameState
from chasetag_arena.planner import Planner

BODY_LABEL: Final[str] = "K/default"  # the default body; the limited preset is not vendored yet
INTERMISSION_S: Final[float] = 1.5
SUBSCRIBER_QUEUE_SIZE: Final[int] = 64
# With no planner file loaded, a planner seat plays the cautious fallback bot.
_PLANNER_STANDIN: Final[str] = "fallback"
_ZERO: Final[tuple[float, float]] = (0.0, 0.0)
logger = logging.getLogger(__name__)


def rules_for_limit(limit_s: int) -> Rules:
    """The official rules with a round limit of ``limit_s`` game seconds.

    Only the length of a round changes. Tags, falls, starts and the arena are the official
    values, and the score of a tag still splits by the fraction of the round that has passed.
    """
    if limit_s not in TIME_LIMITS_S:
        raise ValueError(f"time limit must be one of {TIME_LIMITS_S} seconds, got {limit_s}")
    return replace(OFFICIAL, episode_s=float(limit_s))


class RoundObserver(Protocol):
    """Told what happens in each round. Only a logged session attaches one."""

    def round_started(
        self, round_no: int, selectors: tuple[str, str], state: GameState
    ) -> None: ...

    def decision(self, round_no: int, state: GameState, actions: Array) -> None: ...

    def round_ended(
        self, round_no: int, state: GameState, scores: Array, ended_by: int
    ) -> None: ...


def _refuse_unlogged_seat(control: SelectorControl) -> None:
    """A logged session keeps the keyboard on seat 0, and seat 1 on a bot."""
    if control.seat == 0 and control.value != "keyboard":
        raise ValueError("in a logged session the keyboard stays on seat 0")
    if control.seat == 1 and control.value in ("keyboard", "planner"):
        raise ValueError("in a logged session seat 1 plays a bot")


@dataclass
class _Tally:
    """The match so far: score summed over rounds, and how the rounds ended."""

    seat0: float = 0.0
    seat1: float = 0.0
    tags: int = 0
    falls: int = 0
    timeouts: int = 0

    def add_round(self, scores: tuple[float, float], ended: int) -> None:
        self.seat0 += scores[0]
        self.seat1 += scores[1]
        if ended == TAG:
            self.tags += 1
        elif ended == FALL:
            self.falls += 1
        elif ended == TIMEOUT:
            self.timeouts += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "seat0": self.seat0,
            "seat1": self.seat1,
            "tags": self.tags,
            "falls": self.falls,
            "timeouts": self.timeouts,
        }


class GameLoop:
    """Owns one game and plays it on its own thread. Only the game thread touches game state.

    HTTP handlers call ``submit``, ``subscribe`` and ``unsubscribe``. They never read or write
    the state. Discrete controls go onto a queue that the game thread drains, and the keyboard
    and selector inputs have their own thread-safe holders.
    """

    def __init__(
        self,
        *,
        seed: int,
        selectors: tuple[str, str] = ("greedy", "greedy"),
        params: BodyParams | None = None,
        rules: Rules | None = None,
        clock: Callable[[], float] = time.monotonic,
        planner: Planner | None = None,
        recorder: RoundObserver | None = None,
    ) -> None:
        for name in selectors:
            if name not in SELECTORS:
                raise ValueError(f"unknown selector {name!r}")
        self._rules = rules_for_limit(DEFAULT_TIME_LIMIT_S) if rules is None else rules
        self._pending_limit: int | None = None
        self._params = default_params() if params is None else params
        self._clock = clock
        self._planner = planner
        self._planner_bot = None if planner is None else planner.as_bot()
        self._recorder = recorder
        self._recording_error: str | None = None

        self._lock = threading.Lock()  # guards _subscribers, _pending and _last_frame
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._pending: dict[int, str] = {}
        self._last_frame: dict[str, Any] | None = None
        self._controls: queue.Queue[Control] = queue.Queue()
        self._keyboard = KeyboardState()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._selectors = selectors
        self._speed = 1.0
        self._paused = False
        self._step_pending = 0
        self._decisions = 0
        self._seq = 0
        self._round_no = 0
        self._tally = _Tally()
        self._phase: Phase = "running"
        self._intermission_until = 0.0
        self._round_score: tuple[float, float] = _ZERO
        self._last_ended_by = RUNNING
        self._last_actions: Array | None = None

        # One key for the round resets, and one per seat for the bots. Each decision splits
        # the seat's key, so a bot's draws do not depend on the other seat.
        self._key, bot0, bot1 = jax.random.split(jax.random.PRNGKey(seed), 3)
        self._bot_keys: list[Array] = [bot0, bot1]
        self._bot_states: list[Array] = [jnp.zeros(2), jnp.zeros(2)]
        self._state: GameState
        self._start_round()

    # --- read-only views, for tests and the server ------------------------------------

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def selectors(self) -> tuple[str, str]:
        return self._selectors

    @property
    def planner(self) -> Planner | None:
        """The planner file that plays the planner seat, or None when none is loaded."""
        return self._planner

    @property
    def decisions(self) -> int:
        """Decisions made so far, across rounds. A parked loop does not increase it."""
        return self._decisions

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def last_actions(self) -> Array | None:
        """The (2, 2) raw actions the most recent decision used, or None before the first."""
        return self._last_actions

    @property
    def limit_s(self) -> int:
        """The longest a round may run now, in game seconds. A change waits for the next round."""
        return int(self._rules.episode_s)

    @property
    def recording_active(self) -> bool:
        """Whether rounds are being written to the store. False once a write has failed."""
        return self._recorder is not None

    @property
    def recording_error(self) -> str | None:
        """The failure that stopped the recording, or None while nothing has failed."""
        return self._recording_error

    # --- thread-safe entry points ------------------------------------------------------

    def submit(self, control: Control) -> None:
        """Accept a parsed control from any thread. Selectors and the time limit wait for the
        next round. A logged session refuses both changes that would alter its record."""
        if isinstance(control, SelectorControl):
            with self._lock:
                if self._recorder is not None:
                    _refuse_unlogged_seat(control)
                self._pending[control.seat] = control.value
        elif isinstance(control, TimeLimitControl):
            with self._lock:
                if self._recorder is not None:
                    raise ValueError("in a logged session the time limit is fixed")
                self._pending_limit = control.value
        elif isinstance(control, KeyboardControl):
            self._keyboard.update(control.seat, control.cmd, self._clock())
        else:
            self._controls.put(control)
            self._wake.set()

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        """A bounded queue of frames. The latest frame is queued at once, so a new viewer
        does not wait for the next decision."""
        frames: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            if self._last_frame is not None:
                frames.put_nowait(self._last_frame)
            self._subscribers.append(frames)
        self._wake.set()
        return frames

    def unsubscribe(self, frames: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if frames in self._subscribers:
                self._subscribers.remove(frames)

    def latest_frame(self) -> dict[str, Any] | None:
        """The last published frame, or None before the first. Does not subscribe, so it
        never wakes a parked loop."""
        with self._lock:
            return self._last_frame

    # --- the game thread ---------------------------------------------------------------

    def decide(self) -> bool:
        """Apply queued controls and make one decision if one is due. Returns whether it did.

        Only the game thread calls this, or a test that stands in for it.
        """
        self._drain_controls()
        if self._phase == "intermission":
            if self._clock() < self._intermission_until:
                return False
            self._start_round()
        if self._paused and self._step_pending == 0:
            return False
        if self._step_pending > 0:
            self._step_pending -= 1
        self._decide_once()
        return True

    def run(self) -> None:
        """The thread body. Decides every 0.1 s of game time, scaled by speed, while a viewer
        is connected or a step is pending. Otherwise it waits on the wake event."""
        due = self._clock()
        while not self._stop.is_set():
            self._wake.clear()
            self._drain_controls()
            with self._lock:
                viewed = bool(self._subscribers)
            if self._step_pending == 0 and (not viewed or self._paused):
                self._wake.wait()  # parked: a subscriber, a control or stop wakes it
                continue
            now = self._clock()
            if self._phase == "intermission" and now < self._intermission_until:
                self._wake.wait(self._intermission_until - now)
                continue
            if now < due:
                self._wake.wait(due - now)
                continue
            self.decide()
            period = self._rules.decision_dt / self._speed
            # After a stall, the missed deadlines are dropped rather than replayed in a burst.
            due = max(due + period, self._clock())

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="chasetag_arena-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # --- internals ---------------------------------------------------------------------

    def _bot_for(self, name: str) -> Bot:
        """The bot that plays a seat selected as ``name``. Not valid for ``keyboard``."""
        if name != "planner":
            return BOTS[name]
        if self._planner_bot is None:
            return BOTS[_PLANNER_STANDIN]
        return self._planner_bot

    def _drain_controls(self) -> None:
        """Apply queued controls in the order they arrived, then tell viewers of any change to
        the pause or the speed, since a paused game makes no decision that would carry it."""
        changed = False
        while True:
            try:
                control = self._controls.get_nowait()
            except queue.Empty:
                break
            if isinstance(control, PauseControl):
                self._paused = True
                changed = True
            elif isinstance(control, PlayControl):
                self._paused = False
                changed = True
            elif isinstance(control, StepControl):
                self._step_pending += 1
            elif isinstance(control, RestartControl):
                self._tally = _Tally()
                self._start_round()
            elif isinstance(control, SpeedControl):
                self._speed = control.value
                changed = True
        if changed:
            self._publish_status()

    def _publish_status(self) -> None:
        """Publish the current game again, with the pause and speed as they now are.

        The phase is the one the game is in now. An ended frame is never current here, because
        the intermission frame follows it in the same call, so this always repeats a running or
        an intermission frame.
        """
        if self._phase == "intermission":
            self._publish(
                self._frame(
                    self._state, _ZERO, "intermission", self._last_ended_by, self._round_score
                )
            )
        else:
            self._publish(self._frame(self._state, _ZERO, "running", RUNNING, _ZERO))

    def _start_round(self) -> None:
        """Apply pending selectors and time limit, reset the game, and publish its first frame."""
        with self._lock:
            if self._pending:
                selectors = list(self._selectors)
                for seat, name in self._pending.items():
                    selectors[seat] = name
                self._selectors = (selectors[0], selectors[1])
                self._pending = {}
            if self._pending_limit is not None:
                self._rules = rules_for_limit(self._pending_limit)
                self._pending_limit = None
        self._round_no += 1
        self._key, k_round = jax.random.split(self._key)
        self._state = game.reset(k_round, self._params, self._rules)
        for seat in (0, 1):
            self._bot_keys[seat], sub = jax.random.split(self._bot_keys[seat])
            name = self._selectors[seat]
            if name == "keyboard":
                self._bot_states[seat] = jnp.zeros(2)
            else:
                self._bot_states[seat] = self._bot_for(name).init_state(sub)
        self._record(lambda rec: rec.round_started(self._round_no, self._selectors, self._state))
        self._phase = "running"
        self._round_score = _ZERO
        self._last_ended_by = RUNNING
        self._publish(self._frame(self._state, _ZERO, "running", RUNNING, _ZERO))

    def _choose_actions(self, state: GameState, now: float) -> Array:
        """Raw (2, 2) actions for both seats, from each seat's selector."""
        rows: list[Array] = []
        for seat in (0, 1):
            name = self._selectors[seat]
            if name == "keyboard":
                # The held command is world frame; the core takes heading-frame commands.
                vx, vy = self._keyboard.current(seat, now)
                cmd_h = to_heading_frame(jnp.array([vx, vy]), state.bodies.heading[seat])
                rows.append(raw_for_command(cmd_h, OFFICIAL.max_speed_mps))
                continue
            obs = encode_v1(state, seat, self._rules)
            self._bot_keys[seat], sub = jax.random.split(self._bot_keys[seat])
            bot = self._bot_for(name)
            raw, self._bot_states[seat] = bot.act(obs, sub, self._bot_states[seat])
            rows.append(raw)
        return jnp.stack(rows)

    def _decide_once(self) -> None:
        actions = self._choose_actions(self._state, self._clock())
        self._last_actions = actions
        state, scores = game.step(self._state, actions, self._params, self._rules)
        self._state = state
        self._decisions += 1
        self._record(lambda rec: rec.decision(self._round_no, state, actions))
        ended = int(np.asarray(state.ended_by))
        if ended == RUNNING:
            self._publish(self._frame(state, _ZERO, "running", RUNNING, _ZERO))
        else:
            self._finish_round(state, scores, ended)

    def _finish_round(self, state: GameState, scores: Array, ended: int) -> None:
        """The terminal frame, then the intermission frame that holds its score."""
        self._record(lambda rec: rec.round_ended(self._round_no, state, scores, ended))
        final = (float(scores[0]), float(scores[1]))
        self._round_score = final
        self._last_ended_by = ended
        self._tally.add_round(final, ended)
        self._publish(self._frame(state, final, "ended", ended, final))
        self._phase = "intermission"
        self._intermission_until = self._clock() + INTERMISSION_S
        self._publish(self._frame(state, _ZERO, "intermission", ended, final))

    def _frame(
        self,
        state: GameState,
        scores: tuple[float, float],
        phase: Phase,
        ended_by: int,
        round_score: tuple[float, float],
    ) -> dict[str, Any]:
        self._seq += 1
        return state_to_frame(
            state,
            scores,
            seq=self._seq,
            t_server=self._clock(),
            round_no=self._round_no,
            phase=phase,
            ended_by=ended_by,
            selectors={"seat0": self._selectors[0], "seat1": self._selectors[1]},
            body_label=BODY_LABEL,
            match=self._tally.as_dict(),
            round_score=round_score,
            paused=self._paused,
            speed=self._speed,
            limit_s=self.limit_s,
        )

    def _record(self, write: Callable[[RoundObserver], None]) -> None:
        """Run one recorder hook. A failed write stops the recording, and the game plays on.

        The failure is logged and kept for ``recording_error``, so the arcade can say that the
        session is no longer being logged.
        """
        recorder = self._recorder
        if recorder is None:
            return
        try:
            write(recorder)
        except Exception as err:
            logger.exception("the session is no longer being logged; the game goes on")
            self._recorder = None
            self._recording_error = f"{type(err).__name__}: {err}"

    def _publish(self, frame: dict[str, Any]) -> None:
        """Fan a frame out to every subscriber. A full queue drops its oldest frame, which
        is safe because every frame is a full snapshot."""
        with self._lock:
            self._last_frame = frame
            subscribers = list(self._subscribers)
        for frames in subscribers:
            try:
                frames.put_nowait(frame)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    frames.get_nowait()
                frames.put_nowait(frame)
