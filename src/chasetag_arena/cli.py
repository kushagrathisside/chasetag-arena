"""The command line. ``chasetag-arena serve`` runs the arcade on this machine.

``main`` only parses arguments and waits. ``start_serving`` does the work and returns a handle
that can be closed from code, which is how the tests stop a server they started.
"""

from __future__ import annotations

import argparse
import secrets
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chasetag_arena.arcade.game_loop import GameLoop
from chasetag_arena.arcade.server import ArcadeServer
from chasetag_arena.planner import PlannerError, load_planner

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8765
# The browser client that ships inside the package. ``--static-dir`` replaces it.
PACKAGE_STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "arcade" / "static"
_WAIT_POLL_S: Final[float] = 0.5  # short waits keep Ctrl+C responsive
_THREAD_JOIN_S: Final[float] = 5.0


@dataclass(frozen=True)
class ServeOptions:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    open_browser: bool = True
    static_dir: Path | None = None
    planner: Path | None = None


@dataclass
class Serving:
    """A running arcade: its game loop, its HTTP server, and the thread serving HTTP."""

    loop: GameLoop
    server: ArcadeServer
    thread: threading.Thread

    def close(self) -> None:
        """Stop HTTP first, so no viewer can subscribe, then stop the game thread."""
        self.server.shutdown()
        self.thread.join(_THREAD_JOIN_S)
        self.loop.stop()


def start_serving(options: ServeOptions) -> Serving:
    """Start the game and the HTTP server, announce the URL, and return without blocking.

    The planner file is read first and the server socket is bound before the game starts, so
    a planner that cannot run, or a port already in use, fails without leaving a thread behind.
    """
    planner = None if options.planner is None else load_planner(options.planner)
    loop = GameLoop(seed=secrets.randbits(31), planner=planner)
    static_dir = PACKAGE_STATIC_DIR if options.static_dir is None else options.static_dir
    server = ArcadeServer(loop, options.host, options.port, static_dir)
    loop.start()
    thread = threading.Thread(target=server.serve_forever, name="chasetag_arena-http", daemon=True)
    thread.start()
    print(f"chasetag-arena: serving at {server.url} (Ctrl+C to stop)", flush=True)
    if options.open_browser:
        webbrowser.open(server.url)
    return Serving(loop=loop, server=server, thread=thread)


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chasetag-arena", description="A grounded 2D chase-tag game, played in a browser."
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    serve = commands.add_parser("serve", help="run the arcade on this machine")
    serve.add_argument(
        "--host", default=DEFAULT_HOST, help="address to bind (default: %(default)s)"
    )
    serve.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help="port to bind; 0 picks a free one (default: %(default)s)",
    )
    serve.add_argument(
        "--no-browser", action="store_true", help="print the URL without opening a browser"
    )
    serve.add_argument(
        "--static-dir",
        type=Path,
        default=None,
        help="directory of arcade client files (default: the copy in the package)",
    )
    serve.add_argument(
        "--planner",
        type=Path,
        default=None,
        metavar="FILE",
        help="a planner file for the planner seat (default: none, the seat plays the fallback bot)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = ServeOptions(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        static_dir=args.static_dir,
        planner=args.planner,
    )
    try:
        serving = start_serving(options)
    except PlannerError as exc:
        print(f"chasetag-arena: {exc}", file=sys.stderr)
        return 2
    try:
        while serving.thread.is_alive():
            serving.thread.join(_WAIT_POLL_S)
    except KeyboardInterrupt:
        pass
    finally:
        serving.close()
    return 0
