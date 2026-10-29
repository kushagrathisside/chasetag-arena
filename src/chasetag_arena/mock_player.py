"""A mock player: one logged session, played headless, submitted, and checked in a store.

The participant is a stand-in, enrolled with a mock consent text, and it plays the keyboard seat.
A scripted command drives that seat: it runs at the opponent when it is the chaser, and away from
it when it is not. The other seat is the greedy bot. Each round is written to the store as the
game plays it, the session's play log is built and submitted, and then the store is read back and
checked. Nothing is written to the repository: the store is the only output.

Run it with a store folder and the migration file:

    python -m chasetag_arena.mock_player --migration ../groundtag/db/migrations/0001_ledger.sql
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from chasetag_arena import ledger_writer as lw
from chasetag_arena import recording
from chasetag_arena.arcade.controls import DEFAULT_TIME_LIMIT_S, TIME_LIMITS_S, KeyboardControl
from chasetag_arena.arcade.game_loop import INTERMISSION_S, GameLoop, rules_for_limit
from chasetag_arena.core.spec.rules import OFFICIAL
from chasetag_arena.core.state import GameState

MIGRATION_ENV: Final[str] = "CHASETAG_LEDGER_MIGRATION"
MOCK_LABEL: Final[str] = "mock-play-logs-v1"
MOCK_CONSENT: Final[bytes] = (
    b"Mock consent text. It stands in for the consent terms, so the store's consent rules can be "
    b"exercised. No person gave this consent.\n"
)
SELECTORS: Final[tuple[str, str]] = ("keyboard", "greedy")
PLAYER_SPEED: Final[float] = 1.2  # m/s, below the 1.5 m/s limit


class ManualClock:
    """A clock the caller moves on, so a headless game runs as fast as it can compute."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def scripted_command(state: GameState, speed: float = PLAYER_SPEED) -> tuple[float, float]:
    """The keyboard seat's command: at the opponent when chasing, away from it when running."""
    pos = np.asarray(state.bodies.pos, dtype=float)
    offset = pos[1] - pos[0]
    distance = float(np.linalg.norm(offset))
    if distance < 1e-9:
        return (0.0, 0.0)
    toward = offset / distance
    direction = toward if int(np.asarray(state.chaser)) == 0 else -toward
    return (float(direction[0] * speed), float(direction[1] * speed))


@dataclass(frozen=True)
class MockReport:
    store: Path
    session_id: int
    participant_id: int
    rounds: int
    seats: int
    tags: int
    falls: int
    timeouts: int
    chaser_seats: tuple[int, ...]
    submission: recording.Submission
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def render(self) -> str:
        chasers = " ".join(str(seat) for seat in self.chaser_seats)
        lines = [
            f"store: {self.store}",
            f"session {self.session_id}, participant {self.participant_id}",
            f"rounds: {self.rounds} (tags {self.tags}, falls {self.falls}, "
            f"timeouts {self.timeouts})",
            f"chaser seat by round: {chasers}",
            f"play log: submission {self.submission.id} {self.submission.status}",
        ]
        lines += [f"  {'PASS' if ok else 'FAIL'}  {name}" for name, ok in self.checks.items()]
        lines.append("result: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _checks(
    ledger: lw.Ledger,
    session_id: int,
    participant_id: int,
    rounds: int,
    submission: recording.Submission,
) -> dict[str, bool]:
    conn = ledger.conn
    session = conn.execute(
        "SELECT logged, participant_id FROM play_session WHERE id = ?", (session_id,)
    ).fetchone()
    rows = conn.execute(
        "SELECT r.id, r.ended_by, r.decisions, r.trajectory_artifact_id FROM round r "
        "WHERE r.play_session_id = ? ORDER BY r.round_index",
        (session_id,),
    ).fetchall()
    seat_rows = conn.execute(
        "SELECT rs.round_id, rs.score FROM round_seat rs JOIN round r ON r.id = rs.round_id "
        "WHERE r.play_session_id = ?",
        (session_id,),
    ).fetchall()
    scores: dict[int, float] = {}
    for seat in seat_rows:
        scores[int(seat["round_id"])] = scores.get(int(seat["round_id"]), 0.0) + float(
            seat["score"]
        )
    seats_per_round: dict[int, int] = {}
    for seat in seat_rows:
        key = int(seat["round_id"])
        seats_per_round[key] = seats_per_round.get(key, 0) + 1

    frames_match = True
    score_split = True
    for row in rows:
        frames = json_frames(ledger, int(row["trajectory_artifact_id"]))
        frames_match &= int(row["decisions"]) == len(frames) - 1
        expected = 0.0 if row["ended_by"] == "fall" else 1.0
        score_split &= abs(scores[int(row["id"])] - expected) < 1e-6

    trajectories = [int(row["trajectory_artifact_id"]) for row in rows]
    consent = conn.execute(
        "SELECT ce.event FROM consent_event ce JOIN consent_version cv "
        "ON cv.id = ce.consent_version_id WHERE ce.participant_id = ? AND cv.purpose = ? "
        "ORDER BY ce.id DESC LIMIT 1",
        (participant_id, recording.PURPOSE),
    ).fetchone()
    return {
        "session logged for the participant": (
            session is not None
            and int(session["logged"]) == 1
            and int(session["participant_id"]) == participant_id
        ),
        "every requested round is stored": len(rows) == rounds,
        "every round has two seats": all(seats_per_round.get(int(r["id"])) == 2 for r in rows),
        "scores split as the rules say": score_split,
        "decisions match each trajectory": frames_match,
        "every trajectory verifies": all(
            ledger.verify_artifact(artifact) == "verified" for artifact in trajectories
        ),
        "research consent is in force": consent is not None and consent["event"] == "granted",
        "the play log is accepted": submission.status == "accepted",
    }


def json_frames(ledger: lw.Ledger, artifact_id: int) -> list[dict[str, Any]]:
    """The frames of one round's trajectory file: the start, then one frame per decision."""
    return json.loads(ledger.read_object(artifact_id))["frames"]


def run_mock_play(
    store: Path,
    migration: Path,
    *,
    rounds: int = 3,
    seed: int = 11,
    limit_s: int = DEFAULT_TIME_LIMIT_S,
) -> MockReport:
    """Play ``rounds`` logged rounds, submit the play log, and check the store against it."""
    if rounds < 1:
        raise ValueError("a mock session needs at least one round")
    rules = rules_for_limit(limit_s)
    with recording.open_store(store, migration) as ledger:
        participant, _pseudonym, consent = recording.enrol(
            ledger, consent_label=MOCK_LABEL, consent_text=MOCK_CONSENT
        )
        session = recording.Recording(
            store,
            migration,
            participant_id=participant,
            seed=seed,
            selectors=SELECTORS,
            episode_s=limit_s,
            code_checkout=recording.REPO_ROOT,
        )
        clock = ManualClock()
        loop = GameLoop(seed=seed, selectors=SELECTORS, clock=clock, rules=rules, recorder=session)
        per_round = rules.decisions_per_episode + int(INTERMISSION_S / OFFICIAL.decision_dt) + 10
        for _ in range(rounds * per_round):
            if session.rounds_recorded >= rounds:
                break
            loop.submit(KeyboardControl(seat=0, cmd=scripted_command(loop.state)))
            loop.decide()
            clock.now += OFFICIAL.decision_dt
        if session.rounds_recorded < rounds:
            raise RuntimeError(f"only {session.rounds_recorded} of {rounds} rounds finished")
        session.finish()

        data = recording.build_manifest(ledger, session.session_id)
        submission = recording.submit_play_log(
            ledger, participant_id=participant, consent_version_id=consent, data=data
        )
        checks = _checks(ledger, session.session_id, participant, rounds, submission)
        outcomes = ledger.conn.execute(
            "SELECT ended_by, chaser_seat FROM round WHERE play_session_id = ? "
            "ORDER BY round_index",
            (session.session_id,),
        ).fetchall()
        ended = [str(row["ended_by"]) for row in outcomes]
        chasers = tuple(int(row["chaser_seat"]) for row in outcomes)
        seats = ledger.conn.execute(
            "SELECT COUNT(*) FROM round_seat rs JOIN round r ON r.id = rs.round_id "
            "WHERE r.play_session_id = ?",
            (session.session_id,),
        ).fetchone()[0]
        return MockReport(
            store=store,
            session_id=session.session_id,
            participant_id=participant,
            rounds=len(outcomes),
            seats=int(seats),
            tags=ended.count("tag"),
            falls=ended.count("fall"),
            timeouts=ended.count("timeout"),
            chaser_seats=chasers,
            submission=submission,
            checks=checks,
        )


def _env_migration() -> Path | None:
    value = os.environ.get(MIGRATION_ENV)
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Play one logged session headless, submit its play log, and check the store."
    )
    parser.add_argument("--store", type=Path, help="store folder; a new temporary one if omitted")
    parser.add_argument(
        "--migration",
        type=Path,
        default=_env_migration(),
        help=f"the ledger migration file (or set {MIGRATION_ENV})",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--limit",
        type=int,
        choices=TIME_LIMITS_S,
        default=DEFAULT_TIME_LIMIT_S,
        help="the longest a round may run, in game seconds (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.migration is None:
        parser.error(f"give --migration or set {MIGRATION_ENV}")
    store = args.store or Path(tempfile.mkdtemp(prefix="chasetag-mock-"))
    report = run_mock_play(
        store, args.migration, rounds=args.rounds, seed=args.seed, limit_s=args.limit
    )
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
