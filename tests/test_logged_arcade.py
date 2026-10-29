"""The arcade logs a session when it is started for a participant, and refuses to break it.

The store tests need the ledger migration from groundtag: set CHASETAG_LEDGER_MIGRATION to its
path, or keep groundtag beside this repository. Without it they are skipped.
"""

from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from chasetag_arena import cli, recording
from chasetag_arena import ledger_writer as lw
from chasetag_arena.arcade.game_loop import GameLoop
from chasetag_arena.core.spec.rules import OFFICIAL
from chasetag_arena.mock_player import MOCK_CONSENT, MOCK_LABEL

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


def _enrol(store: Path, migration: Path) -> tuple[str, int]:
    with recording.open_store(store, migration) as ledger:
        participant, pseudonym, _ = recording.enrol(
            ledger, consent_label=MOCK_LABEL, consent_text=MOCK_CONSENT
        )
    return pseudonym, participant


# A local server is reached directly, whatever proxy the environment names for other hosts.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request(url: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET")
    if body is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with _LOCAL.open(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def _options(store: Path, migration: Path, pseudonym: str) -> cli.ServeOptions:
    return cli.ServeOptions(
        port=0,
        open_browser=False,
        store=store,
        migration=migration,
        participant=pseudonym,
    )


def test_a_logged_server_writes_its_session_and_ends_it_on_close(
    tmp_path: Path, migration: Path
) -> None:
    store = tmp_path / "store"
    pseudonym, _ = _enrol(store, migration)
    serving = cli.start_serving(_options(store, migration, pseudonym))
    try:
        status, info = _request(serving.server.url.rstrip("/") + "/ledger")
        assert status == 200
        assert info == {"recording": True, "error": None}
    finally:
        serving.close()
    conn = sqlite3.connect(store / "ledger.sqlite")
    try:
        ended = conn.execute("SELECT ended_at FROM play_session").fetchall()
    finally:
        conn.close()
    assert len(ended) == 1
    assert ended[0][0] is not None


def test_a_participant_without_consent_cannot_start_a_logged_server(
    tmp_path: Path, migration: Path
) -> None:
    store = tmp_path / "store"
    pseudonym, participant = _enrol(store, migration)
    with recording.open_store(store, migration) as ledger:
        consent = ledger.conn.execute("SELECT consent_version_id FROM consent_event").fetchone()[0]
        ledger.record_consent(participant, int(consent), "withdrawn")
    with pytest.raises(lw.LedgerError, match="no research consent"):
        cli.start_serving(_options(store, migration, pseudonym))


def test_the_arcade_refuses_selector_changes_that_break_a_logged_session(
    tmp_path: Path, migration: Path
) -> None:
    store = tmp_path / "store"
    pseudonym, _ = _enrol(store, migration)
    serving = cli.start_serving(_options(store, migration, pseudonym))
    control = serving.server.url.rstrip("/") + "/control"
    try:
        assert _request(control, {"kind": "selector", "seat": 0, "value": "greedy"})[0] == 400
        assert _request(control, {"kind": "selector", "seat": 1, "value": "keyboard"})[0] == 400
        assert _request(control, {"kind": "selector", "seat": 1, "value": "planner"})[0] == 400
        assert _request(control, {"kind": "selector", "seat": 1, "value": "idle"})[0] == 202
    finally:
        serving.close()


class _FailingRecorder:
    """Writes every hook except the end of a round, which fails as a full disk would."""

    def round_started(self, round_no: int, selectors: Any, state: Any) -> None:
        return None

    def decision(self, round_no: int, state: Any, actions: Any) -> None:
        return None

    def round_ended(self, round_no: int, state: Any, scores: Any, ended_by: int) -> None:
        raise OSError("disk full")


def test_a_failed_write_stops_the_recording_and_the_game_goes_on() -> None:
    clock_now = [0.0]
    loop = GameLoop(
        seed=1,
        selectors=("idle", "idle"),
        clock=lambda: clock_now[0],
        recorder=_FailingRecorder(),
    )
    for _ in range(4 * OFFICIAL.decisions_per_episode):
        if loop.recording_error is not None:
            break
        loop.decide()
        clock_now[0] += OFFICIAL.decision_dt
    assert loop.recording_error == "OSError: disk full"
    assert not loop.recording_active
    clock_now[0] += 2.0  # past the intermission, so the next round starts
    before = loop.decisions
    loop.decide()
    assert loop.decisions == before + 1


def test_enrol_prints_a_pseudonym_and_serve_needs_its_store(
    tmp_path: Path, migration: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    consent = tmp_path / "consent.txt"
    consent.write_bytes(MOCK_CONSENT)
    code = cli.main(
        [
            "enrol",
            "--store",
            str(tmp_path / "store"),
            "--migration",
            str(migration),
            "--consent-text",
            str(consent),
            "--label",
            MOCK_LABEL,
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.startswith("pseudonym: ")
    with pytest.raises(SystemExit):
        cli.main(["serve", "--no-browser", "--participant", "anything"])
    with pytest.raises(SystemExit):
        cli.main(["serve", "--no-browser", "--store", str(tmp_path / "store")])


def test_a_logged_session_refuses_a_new_time_limit(tmp_path: Path, migration: Path) -> None:
    store = tmp_path / "store"
    pseudonym, _ = _enrol(store, migration)
    serving = cli.start_serving(_options(store, migration, pseudonym))
    try:
        status, body = _request(
            serving.server.url.rstrip("/") + "/control", {"kind": "limit", "value": 180}
        )
        assert status == 400
        assert "time limit is fixed" in body["error"]
    finally:
        serving.close()
