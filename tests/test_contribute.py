"""Opt-in contribution: nothing is recorded without consent, and the player stays in control."""

from __future__ import annotations

import hashlib
import http.client
import json
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chasetag_arena import cli, contribute
from chasetag_arena.arcade.controls import SelectorControl
from chasetag_arena.arcade.game_loop import GameLoop, rules_for_limit
from chasetag_arena.arcade.server import ArcadeServer
from chasetag_arena.contribute import Contribution, ContributionError
from chasetag_arena.core.spec.rules import OFFICIAL


class Clock:
    """A clock the test moves on, so rounds run as fast as they compute."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def make(
    root: Path, selectors: tuple[str, str] = ("keyboard", "greedy")
) -> tuple[Contribution, GameLoop, Clock]:
    clock = Clock()
    contribution = Contribution(root / "contributions")
    loop = GameLoop(
        seed=1,
        selectors=selectors,
        clock=clock,
        rules=rules_for_limit(20),
        recorder=contribution,
    )
    contribution.bind(lambda: loop.limit_s)
    return contribution, loop, clock


def play(loop: GameLoop, clock: Clock, decisions: int) -> None:
    for _ in range(decisions):
        loop.decide()
        clock.now += OFFICIAL.decision_dt


def play_until_games(contribution: Contribution, loop: GameLoop, clock: Clock, games: int) -> None:
    for _ in range(2000):
        if contribution.status()["games"] >= games:
            return
        play(loop, clock, 1)
    raise AssertionError(f"fewer than {games} games were recorded")


def lines_of(contribution: Contribution) -> list[dict[str, Any]]:
    return [json.loads(line) for line in contribution.review_text().decode().splitlines()]


# --- consent -----------------------------------------------------------------------------


def test_nothing_is_recorded_before_consent(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    play(loop, clock, 400)  # more than one full round
    status = contribution.status()
    assert status["recording"] is False
    assert status["games"] == 0
    assert not (tmp_path / "contributions").exists()


def test_consent_needs_the_current_version(tmp_path: Path) -> None:
    contribution, _, _ = make(tmp_path)
    with pytest.raises(ContributionError, match="changed"):
        contribution.grant("not-the-version")
    assert contribution.status()["recording"] is False
    assert not (tmp_path / "contributions").exists()


def test_the_consent_text_is_a_marked_draft_with_a_stable_version() -> None:
    assert contribute.CONSENT_STATUS == "draft"
    digest = hashlib.sha256(contribute.CONSENT_TEXT.encode()).hexdigest()
    assert digest.startswith(contribute.CONSENT_VERSION)
    for must_say in ("18 or older", "CC BY-NC 4.0", "withdraw", "not recorded"):
        assert must_say in contribute.CONSENT_TEXT


# --- recording ---------------------------------------------------------------------------


def test_the_next_round_after_consent_is_recorded_in_the_dataset_layout(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    play(loop, clock, 5)  # a round is under way
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 1)
    first = lines_of(contribution)
    header, game = first[0], first[1]
    assert header["columns"] == list(contribute.COLUMNS)
    assert header["format"] == contribute.FORMAT
    assert len(first) == 2  # the round that was already running is not in it
    assert game["seat0_bot"] == "human"
    assert game["seat1_bot"] == "greedy"
    assert game["round_limit_s"] == 20
    assert game["ended_by"] in ("tag", "fall", "timeout")
    assert game["decisions"] == len(game["frames"]) - 1
    assert all(len(row) == len(contribute.COLUMNS) for row in game["frames"])
    assert game["frames"][0][9:] == [0.0, 0.0, 0.0, 0.0]  # the start has no command yet
    steps = [row[0] for row in game["frames"]]
    assert steps == sorted(steps)
    gaps = np.diff(steps)
    assert np.all(gaps[:-1] == 10)  # one decision is ten control steps
    assert 0 < gaps[-1] <= 10  # the last may stop short: a tag is timed to the step
    assert game["episode"] == 0
    # The last row is where the game stood when it ended.
    final = np.round(np.asarray(loop.state.bodies.pos, dtype=float), 4).reshape(-1)
    assert game["frames"][-1][1:5] == final.tolist()


def test_only_games_with_exactly_one_keyboard_seat_are_recorded(tmp_path: Path) -> None:
    for selectors in (("greedy", "wander"), ("keyboard", "keyboard")):
        contribution, loop, clock = make(tmp_path / "-".join(selectors), selectors)
        contribution.grant(contribute.CONSENT_VERSION)
        play(loop, clock, 450)
        assert contribution.status()["games"] == 0, selectors


def test_the_keyboard_can_be_on_either_seat_and_a_planner_opponent_is_named(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path, ("greedy", "keyboard"))
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 1)
    game = lines_of(contribution)[1]
    assert (game["seat0_bot"], game["seat1_bot"]) == ("greedy", "human")


def test_stopping_keeps_what_was_recorded_and_stops_new_games(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 1)
    play(loop, clock, 3)  # part of the next round
    after_stop = contribution.stop()
    assert after_stop["recording"] is False
    assert after_stop["games"] == 1
    play(loop, clock, 400)
    assert contribution.status()["games"] == 1
    assert len(lines_of(contribution)) == 2


def test_games_survive_a_restart_but_recording_stays_off_until_consent(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 1)
    ident = contribution.status()["contribution_id"]

    again = Contribution(tmp_path / "contributions")
    status = again.status()
    assert status["games"] == 1
    assert status["recording"] is False
    assert status["contribution_id"] == ident
    again.grant(contribute.CONSENT_VERSION)
    assert again.status()["contribution_id"] == ident  # the same person keeps the same ID


def test_a_write_failure_stops_recording_and_is_reported(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    contribution.grant(contribute.CONSENT_VERSION)
    games_file = tmp_path / "contributions" / contribute.GAMES_FILE
    games_file.unlink()
    games_file.mkdir()  # appending to a directory fails
    play(loop, clock, 260)  # the game goes on
    status = contribution.status()
    assert status["recording"] is False
    assert status["error"]


# --- withdrawal, review and export -------------------------------------------------------


def test_withdrawing_deletes_what_was_written_and_nothing_else(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 1)
    contribution.export()
    keep = tmp_path / "contributions" / "notes.txt"
    keep.write_text("not ours")

    status = contribution.withdraw()
    assert status["recording"] is False
    assert status["games"] == 0
    assert status["contribution_id"] is None
    remaining = sorted(p.name for p in (tmp_path / "contributions").rglob("*"))
    assert remaining == ["notes.txt"]
    with pytest.raises(ContributionError):
        contribution.review_text()


def test_withdrawing_removes_an_empty_folder(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 1)
    contribution.withdraw()
    assert not (tmp_path / "contributions").exists()


def test_an_export_holds_the_games_a_receipt_and_no_private_details(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    with pytest.raises(ContributionError, match="at least one game"):
        contribution.export()
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 2)
    info = contribution.export()
    assert info["games"] == 2
    assert info["consent_status"] == "draft"

    with zipfile.ZipFile(info["path"]) as archive:
        assert sorted(archive.namelist()) == ["README.txt", "episodes.jsonl", "receipt.json"]
        episodes = archive.read("episodes.jsonl")
        receipt = json.loads(archive.read("receipt.json"))
        readme = archive.read("README.txt").decode()
        everything = b"".join(archive.read(n) for n in archive.namelist())
    assert episodes == contribution.review_text()
    assert receipt["episodes_sha256"] == hashlib.sha256(episodes).hexdigest()
    assert receipt["games"] == 2
    assert receipt["consent_status"] == "draft"
    assert receipt["contribution_id"] == contribution.status()["contribution_id"]
    assert "not accepted" in readme
    # Nothing that names the person or the machine is in the package.
    for private in (str(Path.home()), str(tmp_path), Path.home().name):
        assert private.encode() not in everything
    assert info["file"].startswith("chasetag-contribution-")
    assert Path.home().name not in info["file"]


def test_the_latest_package_can_be_fetched_only_after_an_export(tmp_path: Path) -> None:
    contribution, loop, clock = make(tmp_path)
    contribution.grant(contribute.CONSENT_VERSION)
    play_until_games(contribution, loop, clock, 1)
    with pytest.raises(ContributionError, match="export a package first"):
        contribution.package_bytes()
    info = contribution.export()
    name, data = contribution.package_bytes()
    assert name == info["file"]
    assert data[:2] == b"PK"


# --- the game loop -----------------------------------------------------------------------


def test_the_player_stays_free_to_change_seats_and_round_length(tmp_path: Path) -> None:
    contribution, loop, _ = make(tmp_path)
    contribution.grant(contribute.CONSENT_VERSION)
    loop.submit(SelectorControl(seat=1, value="planner"))
    loop.submit(SelectorControl(seat=0, value="idle"))
    assert loop.recording_active is False  # the research ledger is not in use


def test_a_research_recorder_still_locks_the_seats() -> None:
    class Ledgerlike:
        def round_started(self, *args: Any) -> None: ...
        def decision(self, *args: Any) -> None: ...
        def round_ended(self, *args: Any) -> None: ...

    loop = GameLoop(seed=1, selectors=("keyboard", "greedy"), recorder=Ledgerlike())
    assert loop.recording_active is True
    with pytest.raises(ValueError, match="keyboard stays on seat 0"):
        loop.submit(SelectorControl(seat=0, value="idle"))


# --- the HTTP side -----------------------------------------------------------------------


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[ArcadeServer, Contribution, GameLoop, Clock]]:
    contribution, loop, clock = make(tmp_path)
    server = ArcadeServer(loop, "127.0.0.1", 0, None, contribution)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, contribution, loop, clock
    finally:
        server.shutdown()
        thread.join(5)


def call(
    server: ArcadeServer,
    method: str,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
    guarded: bool = True,
) -> tuple[int, dict[str, str], bytes]:
    sent = dict(headers or {})
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        sent["Content-Type"] = "application/json"
    if guarded and method == "POST":
        sent.setdefault("X-Requested-With", "chasetag-arena")
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request(method, path, body=payload, headers=sent)
        resp = conn.getresponse()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()
    finally:
        conn.close()


def test_status_tells_the_page_what_to_show(served: Any) -> None:
    server, *_ = served
    status, _, body = call(server, "GET", "/contribute")
    info = json.loads(body)
    assert status == 200
    assert info["enabled"] is True
    assert info["recording"] is False
    assert info["consent"]["status"] == "draft"
    assert info["consent"]["version"] == contribute.CONSENT_VERSION
    assert info["consent"]["text"] == contribute.CONSENT_TEXT


def test_the_whole_flow_over_http(served: Any) -> None:
    server, contribution, loop, clock = served
    consent = {"action": "consent", "version": contribute.CONSENT_VERSION}
    assert call(server, "POST", "/contribute", consent)[0] == 200
    play_until_games(contribution, loop, clock, 1)

    status, headers, body = call(server, "GET", "/contribute/review")
    assert status == 200
    assert json.loads(body.splitlines()[0])["format"] == contribute.FORMAT
    assert "inline" in headers["content-disposition"]

    assert call(server, "GET", "/contribute/package")[0] == 409  # nothing exported yet
    status, _, body = call(server, "POST", "/contribute", {"action": "export"})
    assert status == 200
    exported = json.loads(body)
    assert exported["games"] == 1

    status, headers, body = call(server, "GET", "/contribute/package")
    assert status == 200
    assert headers["content-type"] == "application/zip"
    assert exported["file"] in headers["content-disposition"]
    assert body[:2] == b"PK"

    assert (
        json.loads(call(server, "POST", "/contribute", {"action": "stop"})[2])["recording"] is False
    )
    status, _, body = call(server, "POST", "/contribute", {"action": "withdraw"})
    assert status == 200
    assert json.loads(body)["games"] == 0
    assert call(server, "GET", "/contribute/review")[0] == 409


def test_a_stale_consent_version_is_refused(served: Any) -> None:
    server, contribution, *_ = served
    status, _, body = call(server, "POST", "/contribute", {"action": "consent", "version": "old"})
    assert status == 409
    assert "changed" in json.loads(body)["error"]
    assert contribution.status()["recording"] is False


def test_other_websites_cannot_change_what_is_recorded(served: Any) -> None:
    server, contribution, *_ = served
    consent = {"action": "consent", "version": contribute.CONSENT_VERSION}
    # No marker header, as a plain cross-site form post would send.
    assert call(server, "POST", "/contribute", consent, guarded=False)[0] == 403
    # A page on another site, even if it could add the header.
    evil = {"Origin": "http://evil.example"}
    assert call(server, "POST", "/contribute", consent, headers=evil)[0] == 403
    # A name that points at this server through DNS rebinding.
    assert call(server, "POST", "/contribute", consent, headers={"Host": "evil.example"})[0] == 403
    assert call(server, "GET", "/contribute", headers={"Host": "evil.example"})[0] == 403
    assert call(server, "GET", "/contribute/review", headers={"Host": "evil.example"})[0] == 403
    assert contribution.status()["recording"] is False
    # The page itself is allowed.
    same = {"Origin": f"http://127.0.0.1:{server.port}"}
    assert call(server, "POST", "/contribute", consent, headers=same)[0] == 200


def test_bad_requests_are_refused_plainly(served: Any) -> None:
    server, *_ = served
    assert call(server, "POST", "/contribute", {"action": "delete-everything"})[0] == 400
    assert call(server, "POST", "/contribute", {"action": "export", "extra": 1})[0] == 400
    assert call(server, "POST", "/contribute", ["export"])[0] == 400
    assert call(server, "POST", "/contribute", {"action": "export"})[0] == 409  # no games yet
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request(
            "POST", "/contribute", body=b"{nope", headers={"X-Requested-With": "chasetag-arena"}
        )
        assert conn.getresponse().status == 400
    finally:
        conn.close()


def test_without_the_option_the_server_says_so_and_does_nothing(tmp_path: Path) -> None:
    loop = GameLoop(seed=1, clock=Clock())
    server = ArcadeServer(loop, "127.0.0.1", 0, None)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, _, body = call(server, "GET", "/contribute")
        assert (status, json.loads(body)) == (200, {"enabled": False})
        assert call(server, "GET", "/contribute/review")[0] == 404
        assert call(server, "POST", "/contribute", {"action": "export"})[0] == 404
    finally:
        server.shutdown()
        thread.join(5)


# --- the command line --------------------------------------------------------------------


def test_contribute_is_refused_off_this_computer_and_beside_the_ledger() -> None:
    for argv in (
        ["serve", "--contribute", "--host", "0.0.0.0"],
        ["serve", "--contribute", "--store", "s", "--participant", "p"],
        ["serve", "--contribute-dir", "somewhere"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(argv)
        assert exit_info.value.code == 2, argv


def test_serving_with_contribute_attaches_the_recorder_and_starts_with_the_keyboard(
    tmp_path: Path,
) -> None:
    options = cli.ServeOptions(
        port=0, open_browser=False, contribute=True, contribute_dir=tmp_path / "games"
    )
    serving = cli.start_serving(options)
    try:
        assert serving.contribution is not None
        assert serving.loop.selectors == ("keyboard", "greedy")
        status, _, body = call(serving.server, "GET", "/contribute")
        assert status == 200
        info = json.loads(body)
        assert info["recording"] is False
        assert info["folder"] == str(tmp_path / "games")
    finally:
        serving.close()
    assert not (tmp_path / "games").exists()  # nothing was written


def test_serving_without_contribute_has_no_recorder() -> None:
    serving = cli.start_serving(cli.ServeOptions(port=0, open_browser=False))
    try:
        assert serving.contribution is None
        assert json.loads(call(serving.server, "GET", "/contribute")[2]) == {"enabled": False}
    finally:
        serving.close()
