"""The browser client's files: they exist, they stay local, and the server serves them."""

from __future__ import annotations

import http.client
import re
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from chasetag_arena.arcade.game_loop import GameLoop
from chasetag_arena.arcade.server import ArcadeServer
from chasetag_arena.cli import PACKAGE_STATIC_DIR

CLIENT_FILES = ("index.html", "style.css", "logic.js", "render.js", "app.js")
URL_RE = re.compile(r"https?://|(?:src|href)\s*=\s*[\"']//", re.IGNORECASE)
# Names that must not reach the page or its comments.
_BANNED = ["cl" + "aude", "anth" + "ropic", "hai" + "ku", "son" + "net", "op" + "us", r"\bagent"]
NAME_RE = re.compile("|".join(_BANNED), re.IGNORECASE)


def _client_text(name: str) -> str:
    return (PACKAGE_STATIC_DIR / name).read_text(encoding="utf-8")


@pytest.fixture
def server() -> Iterator[ArcadeServer]:
    """The server as the CLI starts it, with the package's own client directory."""
    loop = GameLoop(seed=3)
    srv = ArcadeServer(loop, "127.0.0.1", 0, PACKAGE_STATIC_DIR)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    thread.join(5.0)
    loop.stop()


def _get(server: ArcadeServer, path: str) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        headers: dict[str, Any] = dict(resp.getheaders())
        return resp.status, headers, resp.read()
    finally:
        conn.close()


def test_the_client_files_exist() -> None:
    for name in CLIENT_FILES:
        assert (PACKAGE_STATIC_DIR / name).is_file(), name


def test_index_loads_the_stylesheet_and_the_scripts_with_defer() -> None:
    html = _client_text("index.html")
    assert 'href="style.css"' in html
    assert re.search(r'<script src="app\.js" defer></script>', html)
    for name in ("logic.js", "render.js"):
        assert re.search(rf'<script src="{re.escape(name)}" defer></script>', html), name


def test_no_static_file_refers_to_a_remote_url() -> None:
    for name in CLIENT_FILES:
        assert not URL_RE.search(_client_text(name)), f"{name} refers to a remote address"


def test_no_static_file_names_a_model_or_an_agent() -> None:
    for name in CLIENT_FILES:
        match = NAME_RE.search(_client_text(name))
        assert match is None, f"{name} contains {match.group(0) if match else ''!r}"


def test_the_root_route_serves_the_page(server: ArcadeServer) -> None:
    status, headers, body = _get(server, "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<title>chasetag-arena</title>" in body
    assert b"not built yet" not in body


def test_the_script_and_the_stylesheet_are_served(server: ArcadeServer) -> None:
    status, headers, _ = _get(server, "/app.js")
    assert status == 200
    assert "javascript" in headers["Content-Type"]

    status, headers, _ = _get(server, "/style.css")
    assert status == 200
    assert headers["Content-Type"].startswith("text/css")


def _page_body_label() -> tuple[str, str]:
    """The body the select shows, and the label in the status line under it."""
    html = _client_text("index.html")
    select = re.search(r'<select id="body-select"[^>]*>(.*?)</select>', html, re.DOTALL)
    assert select is not None, "the page has no body select"
    options = re.findall(
        r'<option value="([^"]*)"[^>]*\bselected\b[^>]*>([^<]*)</option>', select.group(1)
    )
    assert len(options) == 1, options
    value, text = options[0]
    assert value == text, "the option's value and its text differ"
    note = re.search(r'<span class="note" id="body-note">Running: ([^<]*)</span>', html)
    assert note is not None, "the page has no body status line"
    return text, note.group(1)


def test_the_body_label_in_the_frame_is_the_one_the_page_shows() -> None:
    loop = GameLoop(seed=3)
    try:
        assert loop.decide()
        frame = loop.latest_frame()
    finally:
        loop.stop()
    assert frame is not None
    label = frame["selectors"]["body"]
    shown_option, shown_note = _page_body_label()
    assert shown_option == label
    assert shown_note == label
