"""Recording a logged session in the ledger, and checking the play log a player submits.

These tests use a real store in a temporary folder. They need the ledger migration from groundtag:
set CHASETAG_LEDGER_MIGRATION to its path, or keep groundtag beside this repository. Without it,
the tests that touch the store are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import pytest

from chasetag_arena import ledger_writer as lw
from chasetag_arena import mock_player, recording
from chasetag_arena.core import game
from chasetag_arena.core.families.kinematic import default_params

SIBLING_MIGRATION = (
    Path(__file__).resolve().parents[2] / "groundtag" / "db" / "migrations" / "0001_ledger.sql"
)


@pytest.fixture
def migration() -> Path:
    value = os.environ.get("CHASETAG_LEDGER_MIGRATION")
    candidate = Path(value) if value else SIBLING_MIGRATION
    if not candidate.exists():
        pytest.skip("needs the groundtag migration: set CHASETAG_LEDGER_MIGRATION")
    return candidate


def _consent_version(ledger: lw.Ledger, participant: int) -> int:
    row = ledger.conn.execute(
        "SELECT consent_version_id FROM consent_event WHERE participant_id = ? ORDER BY id DESC",
        (participant,),
    ).fetchone()
    return int(row["consent_version_id"])


def test_a_mock_session_passes_every_check(tmp_path: Path, migration: Path) -> None:
    report = mock_player.run_mock_play(tmp_path / "store", migration, rounds=2)
    assert report.passed, report.render()
    assert report.rounds == 2
    assert report.seats == 4


def test_the_chaser_seat_is_chosen_at_random_each_round() -> None:
    """Both seats take the chaser role across rounds. The core draws it from each episode's key."""
    params = default_params()
    keys = jax.random.split(jax.random.PRNGKey(3), 64)
    chasers = {int(game.reset(key, params).chaser) for key in keys}
    assert chasers == {0, 1}


def test_an_unfinished_session_has_no_play_log(tmp_path: Path, migration: Path) -> None:
    with recording.open_store(tmp_path / "store", migration) as ledger:
        participant, _, _ = recording.enrol(
            ledger, consent_label=mock_player.MOCK_LABEL, consent_text=mock_player.MOCK_CONSENT
        )
        session = recording.Recording(
            tmp_path / "store",
            migration,
            participant_id=participant,
            seed=5,
            selectors=mock_player.SELECTORS,
            episode_s=60.0,
            code_checkout=recording.REPO_ROOT,
        )
        with pytest.raises(lw.LedgerError, match="has not finished"):
            recording.build_manifest(ledger, session.session_id)


def test_a_play_log_that_differs_from_the_store_is_rejected(
    tmp_path: Path, migration: Path
) -> None:
    report = mock_player.run_mock_play(tmp_path / "store", migration, rounds=1)
    with recording.open_store(tmp_path / "store", migration) as ledger:
        document = json.loads(recording.build_manifest(ledger, report.session_id))
        document["rounds"][0]["decisions"] += 1
        submission = recording.submit_play_log(
            ledger,
            participant_id=report.participant_id,
            consent_version_id=_consent_version(ledger, report.participant_id),
            data=json.dumps(document, sort_keys=True).encode(),
        )
    assert submission.status == "rejected"
    assert submission.rejection_reason == "the play log does not match the store"


def test_a_play_log_from_another_participant_is_rejected(tmp_path: Path, migration: Path) -> None:
    report = mock_player.run_mock_play(tmp_path / "store", migration, rounds=1)
    with recording.open_store(tmp_path / "store", migration) as ledger:
        other, _, consent = recording.enrol(
            ledger, consent_label=mock_player.MOCK_LABEL, consent_text=mock_player.MOCK_CONSENT
        )
        submission = recording.submit_play_log(
            ledger,
            participant_id=other,
            consent_version_id=consent,
            data=recording.build_manifest(ledger, report.session_id),
        )
    assert submission.status == "rejected"
    assert submission.rejection_reason == "the session belongs to another participant"


def test_a_session_needs_research_consent_in_force(tmp_path: Path, migration: Path) -> None:
    with recording.open_store(tmp_path / "store", migration) as ledger:
        participant, _, consent = recording.enrol(
            ledger, consent_label=mock_player.MOCK_LABEL, consent_text=mock_player.MOCK_CONSENT
        )
        ledger.record_consent(participant, consent, "withdrawn")
        with pytest.raises(lw.LedgerError, match="no research consent"):
            recording.Recording(
                tmp_path / "store",
                migration,
                participant_id=participant,
                seed=5,
                selectors=mock_player.SELECTORS,
                episode_s=60.0,
                code_checkout=recording.REPO_ROOT,
            )


def test_a_planner_seat_cannot_be_logged(tmp_path: Path, migration: Path) -> None:
    with recording.open_store(tmp_path / "store", migration) as ledger:
        participant, _, _ = recording.enrol(
            ledger, consent_label=mock_player.MOCK_LABEL, consent_text=mock_player.MOCK_CONSENT
        )
        with pytest.raises(lw.LedgerError, match="bot on seat 1"):
            recording.Recording(
                tmp_path / "store",
                migration,
                participant_id=participant,
                seed=5,
                selectors=("keyboard", "planner"),
                episode_s=60.0,
                code_checkout=recording.REPO_ROOT,
            )


def test_erasing_a_participant_removes_their_session_files(tmp_path: Path, migration: Path) -> None:
    report = mock_player.run_mock_play(tmp_path / "store", migration, rounds=1)
    with recording.open_store(tmp_path / "store", migration) as ledger:
        artifacts = [
            int(row["trajectory_artifact_id"])
            for row in ledger.conn.execute(
                "SELECT trajectory_artifact_id FROM round WHERE play_session_id = ?",
                (report.session_id,),
            ).fetchall()
        ]
        ledger.request_erasure(report.participant_id)
        erased = ledger.complete_erasure(report.participant_id)
        assert erased
        assert all(artifact in erased for artifact in artifacts)
        assert ledger.verify_artifact(artifacts[0]) == "erased"


def test_a_round_is_recorded_under_the_rules_of_its_length(tmp_path: Path, migration: Path) -> None:
    report = mock_player.run_mock_play(tmp_path / "store", migration, rounds=1, limit_s=60)
    with recording.open_store(tmp_path / "store", migration) as ledger:
        row = ledger.conn.execute(
            "SELECT v.name, v.episode_s FROM rules_version v "
            "JOIN play_session s ON s.rules_version_id = v.id WHERE s.id = ?",
            (report.session_id,),
        ).fetchone()
    assert (row["name"], row["episode_s"]) == ("arena-2d-60s", 60.0)
