"""The HTTP server, tested over real sockets on port 0. Each test finishes in a few seconds."""

from __future__ import annotations

import http.client
import json
import math
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NoReturn

import pytest

from chasetag_arena.arcade import server as server_module
from chasetag_arena.arcade.controls import PauseControl
from chasetag_arena.arcade.game_loop import GameLoop
from chasetag_arena.arcade.server import MAX_BODY_BYTES, ArcadeServer
from chasetag_arena.planner import Planner, load_planner

FRAME_KEYS = {
    "seq",
    "t_server",
    "round",
    "phase",
    "t_event",
    "ended_by",
    "bodies",
    "chaser",
    "round_score",
    "match",
    "selectors",
    "paused",
    "speed",
}
PHASES = {"running", "ended", "intermission"}


def _reject_constant(name: str) -> NoReturn:
    raise AssertionError(f"non-finite JSON constant in an event: {name}")


def _get(server: ArcadeServer, path: str) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def _post(server: ArcadeServer, body: bytes) -> tuple[int, dict[str, str], Any]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("POST", "/control", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), json.loads(resp.read())
    finally:
        conn.close()


def _read_event(resp: http.client.HTTPResponse) -> tuple[str, Any]:
    """Read one server-sent event and check the framing: event line, one data line, blank."""
    name_line = resp.readline().decode("utf-8").rstrip("\r\n")
    assert name_line.startswith("event: "), name_line
    data_line = resp.readline().decode("utf-8").rstrip("\r\n")
    assert data_line.startswith("data: "), data_line
    end_line = resp.readline().decode("utf-8")
    assert end_line == "\n", (
        "a blank line must end the event, and only one data line may precede it"
    )
    data = json.loads(data_line[len("data: ") :], parse_constant=_reject_constant)
    return name_line[len("event: ") :], data


def _check_frame(frame: dict[str, Any]) -> None:
    assert set(frame) == FRAME_KEYS
    assert frame["phase"] in PHASES
    assert len(frame["bodies"]) == 2
    for body in frame["bodies"]:
        assert -math.pi < body["heading"] <= math.pi


@pytest.fixture
def start() -> Iterator[Any]:
    """Start a loop and a server on port 0. Teardown stops both, whatever the test did."""
    running: list[tuple[GameLoop, ArcadeServer, threading.Thread]] = []

    def _start(
        static_dir: Path | None = None, planner: Planner | None = None
    ) -> tuple[GameLoop, ArcadeServer]:
        loop = GameLoop(seed=11, selectors=("greedy", "greedy"), planner=planner)
        server = ArcadeServer(loop, "127.0.0.1", 0, static_dir)
        loop.start()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((loop, server, thread))
        return loop, server

    yield _start
    for loop, server, thread in running:
        server.shutdown()
        thread.join(5.0)
        loop.stop()


def test_healthz_returns_ok_and_version(start: Any) -> None:
    _, server = start()
    status, headers, body = _get(server, "/healthz")
    assert status == 200
    assert json.loads(body) == {"ok": True, "version": "0.1.0"}
    assert headers["Cache-Control"] == "no-store"


def test_planner_endpoint_reports_that_no_planner_is_loaded(start: Any) -> None:
    _, server = start()
    status, headers, body = _get(server, "/planner")
    assert status == 200
    assert json.loads(body) == {"loaded": False}
    assert headers["Cache-Control"] == "no-store"


def test_planner_endpoint_names_the_loaded_planner(start: Any, planner_file: Path) -> None:
    _, server = start(planner=load_planner(planner_file))
    status, _, body = _get(server, "/planner")
    assert status == 200
    assert json.loads(body) == {"loaded": True, "name": "test-planner", "file": "planner.npz"}


def test_events_send_a_snapshot_then_one_frame_per_decision(start: Any) -> None:
    _, server = start()
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("GET", "/events")
        resp = conn.getresponse()
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "text/event-stream"
        assert "no-cache" in resp.getheader("Cache-Control", "")

        name, snapshot = _read_event(resp)
        assert name == "snapshot"
        assert snapshot is not None, "the loop has published its first frame by construction"
        _check_frame(snapshot)

        name, frame = _read_event(resp)
        assert name == "frame"
        _check_frame(frame)
        assert frame["seq"] > snapshot["seq"], "the snapshot is not repeated as a frame"
    finally:
        conn.close()


def test_snapshot_is_null_before_any_frame_and_the_first_frame_follows(
    start: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop, server = start()
    monkeypatch.setattr(loop, "latest_frame", lambda: None)
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("GET", "/events")
        resp = conn.getresponse()
        name, snapshot = _read_event(resp)
        assert (name, snapshot) == ("snapshot", None)
        name, frame = _read_event(resp)
        assert name == "frame"
        _check_frame(frame)
    finally:
        conn.close()


def test_keepalive_comment_is_sent_when_the_stream_is_idle(
    start: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server_module, "KEEPALIVE_S", 0.2)
    loop, server = start()
    loop.submit(PauseControl())
    deadline = time.monotonic() + 5.0
    while not loop.paused and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)  # let any decision already in flight finish

    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("GET", "/events")
        resp = conn.getresponse()
        name, _ = _read_event(resp)
        assert name == "snapshot"
        assert resp.readline() == b": keepalive\n"
        assert resp.readline() == b"\n"
    finally:
        conn.close()


def test_control_is_accepted_and_reaches_the_loop(start: Any) -> None:
    loop, server = start()
    status, headers, body = _post(server, b'{"kind":"pause"}')
    assert status == 202
    assert body == {"accepted": True}
    assert headers["Cache-Control"] == "no-store"
    deadline = time.monotonic() + 5.0
    while not loop.paused and time.monotonic() < deadline:
        time.sleep(0.01)
    assert loop.paused


@pytest.mark.parametrize(
    "body",
    [
        b'{"kind":"selector","seat":1,"value":"idle"}',
        b'{"kind":"keyboard","seat":0,"cmd":[0.5,-0.5]}',
        b'{"kind":"speed","value":2}',
    ],
)
def test_valid_controls_return_202(start: Any, body: bytes) -> None:
    _, server = start()
    status, _, reply = _post(server, body)
    assert (status, reply) == (202, {"accepted": True})


@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b"\xff\xfe",
        b'{"kind":"keyboard","seat":0,"cmd":[NaN,0]}',
        b'{"kind":"teleport"}',
        b'{"kind":"selector","seat":0,"value":"telepathy"}',
        b'{"kind":"pause","extra":1}',
        b"[]",
    ],
)
def test_invalid_control_bodies_return_400_with_an_error(start: Any, body: bytes) -> None:
    _, server = start()
    status, _, reply = _post(server, body)
    assert status == 400
    assert isinstance(reply["error"], str) and reply["error"]


def test_body_at_the_limit_is_read_and_an_oversized_body_is_refused(start: Any) -> None:
    loop, server = start()
    # Whitespace at the limit is within size, so the failure is the JSON, not the length.
    status, _, _ = _post(server, b" " * MAX_BODY_BYTES)
    assert status == 400

    status, _, reply = _post(server, b" " * (MAX_BODY_BYTES + 1))
    assert status == 413
    assert "error" in reply
    assert not loop.paused


def test_post_without_content_length_is_refused(start: Any) -> None:
    _, server = start()
    with socket.create_connection(("127.0.0.1", server.port), timeout=10) as sock:
        sock.sendall(b"POST /control HTTP/1.0\r\nHost: x\r\n\r\n")
        status_line = sock.makefile("rb").readline().decode("ascii")
    assert status_line.split()[1] == "411"


def test_unknown_routes_and_wrong_methods(start: Any) -> None:
    _, server = start()
    assert _get(server, "/control")[0] == 405
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("POST", "/healthz", body=b"{}")
        assert conn.getresponse().status == 404
    finally:
        conn.close()


def test_without_a_static_directory_the_page_says_the_client_is_not_built(start: Any) -> None:
    _, server = start()
    for path in ("/", "/anything/else"):
        status, headers, body = _get(server, path)
        assert status == 200
        assert b"not built yet" in body
        assert headers["Content-Type"].startswith("text/plain")


def test_static_files_are_served_from_the_directory(start: Any, tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<p>arcade page</p>", encoding="utf-8")
    (site / "app.js").write_text("console.log(1);", encoding="utf-8")
    _, server = start(static_dir=site)

    status, headers, body = _get(server, "/")
    assert status == 200
    assert body == b"<p>arcade page</p>"
    assert headers["Content-Type"].startswith("text/html")
    assert "no-store" not in headers.get("Cache-Control", "")

    status, headers, _ = _get(server, "/app.js?v=3")
    assert status == 200
    assert "javascript" in headers["Content-Type"]

    assert _get(server, "/missing.css")[0] == 404


def test_static_paths_cannot_leave_the_directory(start: Any, tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<p>ok</p>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    (site / "escape.txt").symlink_to(tmp_path / "secret.txt")
    _, server = start(static_dir=site)

    for path in (
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/..%2fsecret.txt",
        "/sub/../../secret.txt",
        "/escape.txt",
        "/%00",
    ):
        status, _, body = _get(server, path)
        assert status == 404, path
        assert b"TOP SECRET" not in body, path


def test_shutdown_ends_an_open_stream_and_leaves_no_thread_running() -> None:
    before = set(threading.enumerate())
    loop = GameLoop(seed=12, selectors=("idle", "idle"))
    server = ArcadeServer(loop, "127.0.0.1", 0)
    loop.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    conn.request("GET", "/events")
    resp = conn.getresponse()
    assert _read_event(resp)[0] == "snapshot"

    started = time.monotonic()
    server.shutdown()
    thread.join(5.0)
    loop.stop()
    assert time.monotonic() - started < 3.0
    assert not thread.is_alive()
    conn.close()

    deadline = time.monotonic() + 2.0
    leftover = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    while leftover and time.monotonic() < deadline:
        time.sleep(0.02)
        leftover = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    assert leftover == []


def test_shutdown_before_serving_does_not_block() -> None:
    loop = GameLoop(seed=13, selectors=("idle", "idle"))
    server = ArcadeServer(loop, "127.0.0.1", 0)
    server.shutdown()
    server.serve_forever()  # returns at once: shutdown was requested first
    loop.stop()


def test_a_vanished_viewer_unsubscribes_and_the_loop_parks() -> None:
    loop = GameLoop(seed=14, selectors=("idle", "idle"))
    server = ArcadeServer(loop, "127.0.0.1", 0)
    loop.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
        conn.request("GET", "/events")
        resp = conn.getresponse()
        _read_event(resp)
        _read_event(resp)
        # http.client keeps the socket open through the response, so both must be closed.
        resp.close()
        conn.close()  # the client goes away without saying so

        time.sleep(1.0)  # the next writes fail, the stream unsubscribes, the loop parks
        settled = loop.decisions
        time.sleep(0.5)
        assert loop.decisions == settled
    finally:
        server.shutdown()
        thread.join(5.0)
        loop.stop()
