"""The HTTP side of the arcade: a frame stream, a control endpoint, and optional static files.

Only the standard library is used. Handlers never touch game state. They read frames from
their own subscription queue and hand controls to ``GameLoop.submit``, so a slow or stuck
browser cannot hold up the game thread.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, NoReturn, cast
from urllib.parse import unquote

from chasetag_arena import __version__
from chasetag_arena.arcade.controls import parse_control
from chasetag_arena.arcade.frames import encode_frame
from chasetag_arena.arcade.game_loop import GameLoop

MAX_BODY_BYTES: Final[int] = 4 * 1024
# An oversized body is read and discarded up to this many bytes, so the client receives the
# 413 instead of a reset connection. Anything beyond the cap is left unread.
_DRAIN_LIMIT_BYTES: Final[int] = 1024 * 1024
_READ_CHUNK_BYTES: Final[int] = 64 * 1024
KEEPALIVE_S: Final[float] = 15.0
# How often an event-stream handler checks for shutdown while it waits for a frame.
_POLL_S: Final[float] = 0.25
_SHUTDOWN_WAIT_S: Final[float] = 5.0
_SOCKET_TIMEOUT_S: Final[float] = 30.0
_SERVER_NAME: Final[str] = "chasetag-arena"
_NOT_BUILT: Final[bytes] = (
    b"The arcade client is not built yet. The game server is running; the page comes in a "
    b"later release.\n"
)
_TEXT: Final[str] = "text/plain; charset=utf-8"
_JSON: Final[str] = "application/json; charset=utf-8"


def _reject_constant(name: str) -> NoReturn:
    """Refuse NaN and Infinity in control bodies, which JSON does not define."""
    raise ValueError(f"{name} is not allowed in JSON")


def _route(raw_path: str) -> str:
    """The request path without its query string or fragment."""
    return raw_path.split("?", 1)[0].split("#", 1)[0]


def _static_file(root: Path, request_path: str) -> Path | None:
    """The file a request path names under ``root``, or None if it is missing or outside it.

    The check is made on the resolved path, so ``..`` segments, percent-encoded dots and
    symlinks that point outside the root are all refused.
    """
    rel = unquote(request_path).lstrip("/")
    if "\x00" in rel:
        return None
    if rel == "" or rel.endswith("/"):
        rel += "index.html"
    try:
        base = root.resolve()
        candidate = (base / rel).resolve()
        if not candidate.is_relative_to(base) or not candidate.is_file():
            return None
    except (OSError, ValueError):
        return None
    return candidate


def _content_type(path: Path) -> str:
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
        ctype += "; charset=utf-8"
    return ctype


class _Handler(BaseHTTPRequestHandler):
    """One request per connection. HTTP/1.0 keeps the framing simple: streams end on close."""

    server_version = _SERVER_NAME
    timeout = _SOCKET_TIMEOUT_S

    def version_string(self) -> str:
        return _SERVER_NAME

    def log_message(self, format: str, *args: Any) -> None:
        """Requests are not logged. The CLI prints the URL once and nothing per request."""

    def handle_one_request(self) -> None:
        # A client that disconnects mid-response is not an error worth a traceback.
        try:
            super().handle_one_request()
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    @property
    def _arcade(self) -> ArcadeServer:
        return cast(_Httpd, self.server).owner

    def do_GET(self) -> None:
        path = _route(self.path)
        if path == "/healthz":
            self._send_json(200, {"ok": True, "version": __version__})
        elif path == "/events":
            self._stream_events()
        elif path == "/planner":
            self._send_json(200, self._planner_info())
        elif path == "/ledger":
            self._send_json(200, self._ledger_info())
        elif path == "/control":
            self._send_json(405, {"error": "use POST"}, allow="POST")
        else:
            self._send_static(path)

    def _planner_info(self) -> dict[str, Any]:
        """Whether a planner file is loaded for the planner seat, and what the page shows for it."""
        planner = self._arcade.loop.planner
        if planner is None:
            return {"loaded": False}
        return {"loaded": True, "name": planner.name, "file": planner.file}

    def _ledger_info(self) -> dict[str, Any]:
        """Whether this session is being written to the research store, and why it may not be."""
        loop = self._arcade.loop
        return {"recording": loop.recording_active, "error": loop.recording_error}

    def do_POST(self) -> None:
        if _route(self.path) == "/control":
            self._post_control()
        else:
            self._discard_body()
            self._send_json(404, {"error": "not found"})

    # --- /control ----------------------------------------------------------------------

    def _post_control(self) -> None:
        length = self._declared_length()
        if length is None:
            self._send_json(411, {"error": "Content-Length is required"})
            return
        if length > MAX_BODY_BYTES:
            self._discard_body(length)
            self._send_json(413, {"error": f"body must be at most {MAX_BODY_BYTES} bytes"})
            return
        body = self.rfile.read(length)
        try:
            obj = json.loads(body, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            self._send_json(400, {"error": "body is not valid JSON"})
            return
        try:
            control = parse_control(obj)
            self._arcade.loop.submit(control)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(202, {"accepted": True})

    def _declared_length(self) -> int | None:
        """The Content-Length header as a non-negative int, or None when it is absent or bad."""
        raw = self.headers.get("Content-Length")
        if raw is None:
            return None
        try:
            length = int(raw)
        except ValueError:
            return None
        return length if length >= 0 else None

    def _discard_body(self, length: int | None = None) -> None:
        """Read and drop the request body, up to a cap, then close after the response."""
        if length is None:
            length = self._declared_length() or 0
        remaining = min(length, _DRAIN_LIMIT_BYTES)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                break
            remaining -= len(chunk)
        self.close_connection = True

    # --- /events -----------------------------------------------------------------------

    def _stream_events(self) -> None:
        """Send the latest frame as a snapshot, then every later frame as it is decided.

        The subscription queue already holds the newest frame at subscribe time, and the
        snapshot is read after subscribing, so frames at or below the snapshot's ``seq`` are
        duplicates and are skipped. Nothing is lost between the two reads.
        """
        arcade = self._arcade
        frames = arcade.loop.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.end_headers()

            snapshot = arcade.loop.latest_frame()
            self._write_event("snapshot", snapshot)
            last_seq = -1 if snapshot is None else int(snapshot["seq"])
            last_write = time.monotonic()
            while not arcade.stopping:
                try:
                    frame = frames.get(timeout=_POLL_S)
                except queue.Empty:
                    if time.monotonic() - last_write >= KEEPALIVE_S:
                        self._write(": keepalive\n\n")
                        last_write = time.monotonic()
                    continue
                if int(frame["seq"]) <= last_seq:
                    continue
                self._write_event("frame", frame)
                last_write = time.monotonic()
        finally:
            # Runs on a broken socket too, so a vanished viewer stops costing decisions.
            arcade.loop.unsubscribe(frames)

    def _write_event(self, name: str, frame: dict[str, Any] | None) -> None:
        data = "null" if frame is None else encode_frame(frame)
        self._write(f"event: {name}\ndata: {data}\n\n")

    def _write(self, text: str) -> None:
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    # --- static files and responses ----------------------------------------------------

    def _send_static(self, path: str) -> None:
        root = self._arcade.static_dir
        if root is None or not root.is_dir():
            self._send(200, _NOT_BUILT, _TEXT, no_store=False)
            return
        target = _static_file(root, path)
        if target is None:
            self._send(404, b"not found\n", _TEXT, no_store=False)
            return
        try:
            body = target.read_bytes()
        except OSError:
            self._send(404, b"not found\n", _TEXT, no_store=False)
            return
        self._send(200, body, _content_type(target), no_store=False)

    def _send_json(self, status: int, body: dict[str, Any], *, allow: str | None = None) -> None:
        payload = json.dumps(body, allow_nan=False).encode("utf-8")
        self._send(status, payload, _JSON, allow=allow)

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        no_store: bool = True,
        allow: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        if allow is not None:
            self.send_header("Allow", allow)
        self.end_headers()
        self.wfile.write(body)


class _Httpd(ThreadingHTTPServer):
    """A threading server whose request threads are daemonic and counted.

    Daemon threads cannot hold up interpreter exit. The count lets ``ArcadeServer.shutdown``
    wait for the handlers it has just ended, so no thread outlives the server.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], owner: ArcadeServer) -> None:
        self.owner = owner
        self._idle = threading.Condition()
        self._active = 0
        super().__init__(address, _Handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        with self._idle:
            self._active += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_done()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_done()

    def wait_idle(self, timeout: float) -> bool:
        """Wait until no request thread is running. False if the timeout passed first."""
        with self._idle:
            return self._idle.wait_for(lambda: self._active == 0, timeout)

    def _request_done(self) -> None:
        with self._idle:
            self._active -= 1
            self._idle.notify_all()


class ArcadeServer:
    """Serves the arcade over HTTP, using only the standard library.

    ``serve_forever`` blocks until ``shutdown`` is called from another thread. Port 0 binds a
    free port, which ``port`` reports.
    """

    def __init__(
        self,
        loop: GameLoop,
        host: str = "127.0.0.1",
        port: int = 8765,
        static_dir: Path | None = None,
    ) -> None:
        self.loop = loop
        self.static_dir = static_dir
        self._stopping = threading.Event()
        self._lock = threading.Lock()  # guards _serving against a concurrent shutdown
        self._serving = False
        self._httpd = _Httpd((host, port), self)

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}/"

    @property
    def stopping(self) -> bool:
        return self._stopping.is_set()

    def serve_forever(self) -> None:
        with self._lock:
            if self._stopping.is_set():
                return
            self._serving = True
        try:
            self._httpd.serve_forever(poll_interval=0.2)
        finally:
            with self._lock:
                self._serving = False

    def shutdown(self) -> None:
        """Stop serving and end every open stream. Call from a thread other than the one in
        ``serve_forever``. Safe to call more than once, and before ``serve_forever`` starts."""
        with self._lock:
            self._stopping.set()
            serving = self._serving
        if serving:
            self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd.wait_idle(_SHUTDOWN_WAIT_S)
