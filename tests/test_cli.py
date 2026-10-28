"""The command line: argument parsing, the serve lifecycle, and the browser hand-off."""

from __future__ import annotations

import http.client
import json
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

import pytest

from chasetag_arena import cli
from chasetag_arena.arcade.game_loop import GameLoop
from chasetag_arena.arcade.server import ArcadeServer
from chasetag_arena.cli import ServeOptions, build_parser, main, start_serving


def _settle(before: set[threading.Thread]) -> None:
    """Wait briefly for threads started by the test to finish, then assert none remain."""
    deadline = time.monotonic() + 2.0
    leftover = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    while leftover and time.monotonic() < deadline:
        time.sleep(0.02)
        leftover = [t for t in threading.enumerate() if t not in before and t.is_alive()]
    assert leftover == []


def _healthz(port: int) -> tuple[int, Any]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/healthz")
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_serve_defaults() -> None:
    args = build_parser().parse_args(["serve"])
    assert (args.host, args.port, args.no_browser, args.static_dir) == (
        "127.0.0.1",
        8765,
        False,
        None,
    )


def test_serve_options_are_read() -> None:
    args = build_parser().parse_args(
        ["serve", "--host", "0.0.0.0", "--port", "9000", "--no-browser", "--static-dir", "site"]
    )
    assert (args.host, args.port, args.no_browser, args.static_dir) == (
        "0.0.0.0",
        9000,
        True,
        Path("site"),
    )


@pytest.mark.parametrize("value", ["-1", "65536", "eight"])
def test_a_bad_port_is_a_usage_error(value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["serve", "--port", value])
    assert exc.value.code == 2


def test_help_prints_the_usage(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "serve" in capsys.readouterr().out


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_serve_answers_and_shuts_down_without_leftover_threads(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *_: opened.append(url) or True)
    before = set(threading.enumerate())

    serving = start_serving(ServeOptions(port=0, open_browser=False))
    try:
        status, body = _healthz(serving.server.port)
        assert status == 200
        assert body == {"ok": True, "version": "0.1.0"}
        assert f"serving at {serving.server.url}" in capsys.readouterr().out
    finally:
        serving.close()

    assert opened == [], "--no-browser must not open a browser"
    assert not serving.thread.is_alive()
    _settle(before)


def test_the_browser_opens_the_url_unless_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *_: opened.append(url) or True)
    serving = start_serving(ServeOptions(port=0, open_browser=True))
    try:
        assert opened == [serving.server.url]
    finally:
        serving.close()


def test_a_port_in_use_fails_before_the_game_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webbrowser, "open", lambda *_: True)
    before = set(threading.enumerate())
    holder = ArcadeServer(GameLoop(seed=1, selectors=("idle", "idle")), "127.0.0.1", 0)
    try:
        with pytest.raises(OSError):
            start_serving(ServeOptions(port=holder.port, open_browser=False))
    finally:
        holder.shutdown()
    _settle(before)


def test_main_passes_the_options_and_returns_zero_when_serving_ends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[ServeOptions] = []

    def finished_serving(options: ServeOptions) -> cli.Serving:
        seen.append(options)
        loop = GameLoop(seed=0, selectors=("idle", "idle"))
        server = ArcadeServer(loop, "127.0.0.1", 0)
        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        return cli.Serving(loop=loop, server=server, thread=thread)

    monkeypatch.setattr(cli, "start_serving", finished_serving)
    before = set(threading.enumerate())
    code = main(["serve", "--no-browser", "--port", "0", "--static-dir", "site"])
    assert code == 0
    assert seen == [ServeOptions(port=0, open_browser=False, static_dir=Path("site"))]
    _settle(before)


def test_close_returns_promptly_with_no_viewers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing must not wait for a request or a viewer that never arrives."""
    monkeypatch.setattr(webbrowser, "open", lambda *_: True)
    serving = start_serving(ServeOptions(port=0, open_browser=False))
    began = time.monotonic()
    serving.close()
    assert time.monotonic() - began < 5.0
