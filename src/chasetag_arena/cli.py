"""The command line. ``chasetag-arena serve`` runs the arcade on this machine.

``chasetag-arena enrol`` adds a participant with research consent to a research store, and prints
the pseudonym that ``serve --participant`` needs to log a session.

``main`` only parses arguments and waits. ``start_serving`` does the work and returns a handle
that can be closed from code, which is how the tests stop a server they started.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sqlite3
import sys
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from chasetag_arena import ledger_writer as lw
from chasetag_arena import recording
from chasetag_arena.arcade.controls import DEFAULT_TIME_LIMIT_S, TIME_LIMITS_S
from chasetag_arena.arcade.game_loop import GameLoop, rules_for_limit
from chasetag_arena.arcade.server import ArcadeServer
from chasetag_arena.planner import PlannerError, load_planner

DEFAULT_HOST: Final[str] = "127.0.0.1"
DEFAULT_PORT: Final[int] = 8765
# The browser client that ships inside the package. ``--static-dir`` replaces it.
PACKAGE_STATIC_DIR: Final[Path] = Path(__file__).resolve().parent / "arcade" / "static"
_WAIT_POLL_S: Final[float] = 0.5  # short waits keep Ctrl+C responsive
_THREAD_JOIN_S: Final[float] = 5.0
MIGRATION_ENV: Final[str] = "CHASETAG_LEDGER_MIGRATION"
# A logged session has the participant at the keyboard, on seat 0, and a bot on seat 1.
LOGGED_SELECTORS: Final[tuple[str, str]] = ("keyboard", "greedy")
ENROL_LABEL: Final[str] = "play-logs-v1"


@dataclass(frozen=True)
class ServeOptions:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    open_browser: bool = True
    static_dir: Path | None = None
    planner: Path | None = None
    store: Path | None = None
    migration: Path | None = None
    participant: str | None = None
    time_limit_s: int = DEFAULT_TIME_LIMIT_S


@dataclass
class Serving:
    """A running arcade: its game loop, its HTTP server, the thread serving HTTP, and the
    logged session when there is one."""

    loop: GameLoop
    server: ArcadeServer
    thread: threading.Thread
    session: recording.Recording | None = None

    def close(self) -> None:
        """Stop HTTP first, so no viewer can subscribe, then stop the game thread.

        A logged session is marked ended only when its rounds were all written. A session whose
        recording failed is left open, since its record is incomplete.
        """
        self.server.shutdown()
        self.thread.join(_THREAD_JOIN_S)
        self.loop.stop()
        if self.session is not None and self.loop.recording_error is None:
            self.session.finish()


def _start_session(options: ServeOptions, seed: int) -> recording.Recording:
    """Check the participant and their consent, and open the session before the game starts."""
    if options.store is None or options.migration is None or options.participant is None:
        raise lw.LedgerError("a logged session needs --store, --migration and --participant")
    with recording.open_store(options.store, options.migration) as ledger:
        participant = recording.participant_for(ledger, options.participant)
    return recording.Recording(
        options.store,
        options.migration,
        participant_id=participant,
        seed=seed,
        selectors=LOGGED_SELECTORS,
        episode_s=options.time_limit_s,
        code_checkout=recording.REPO_ROOT,
    )


def start_serving(options: ServeOptions) -> Serving:
    """Start the game and the HTTP server, announce the URL, and return without blocking.

    The planner file is read first, and a logged session is opened, before the game and the
    server start. A planner that cannot run, a participant without consent, or a port already in
    use fails without leaving a thread behind.
    """
    planner = None if options.planner is None else load_planner(options.planner)
    seed = secrets.randbits(31)
    session = None if options.participant is None else _start_session(options, seed)
    rules = rules_for_limit(options.time_limit_s)
    if session is None:
        loop = GameLoop(seed=seed, planner=planner, rules=rules)
    else:
        loop = GameLoop(
            seed=seed,
            selectors=LOGGED_SELECTORS,
            planner=planner,
            rules=rules,
            recorder=session,
        )
    static_dir = PACKAGE_STATIC_DIR if options.static_dir is None else options.static_dir
    try:
        server = ArcadeServer(loop, options.host, options.port, static_dir)
    except OSError:
        if session is not None:
            session.finish()
        raise
    loop.start()
    thread = threading.Thread(target=server.serve_forever, name="chasetag_arena-http", daemon=True)
    thread.start()
    print(f"chasetag-arena: serving at {server.url} (Ctrl+C to stop)", flush=True)
    if session is not None:
        print("chasetag-arena: this session is being logged to the research store", flush=True)
    if options.open_browser:
        webbrowser.open(server.url)
    return Serving(loop=loop, server=server, thread=thread, session=session)


def enrol(store: Path, migration: Path, consent_text: Path, label: str) -> str:
    """Create a participant with research consent granted, and return the pseudonym."""
    text = consent_text.read_bytes()
    with recording.open_store(store, migration) as ledger:
        _participant, pseudonym, _consent = recording.enrol(
            ledger, consent_label=label, consent_text=text
        )
    return pseudonym


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return value


def _store_options(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        required=required,
        metavar="DIR",
        help="the research store folder",
    )
    parser.add_argument(
        "--migration",
        type=Path,
        default=None,
        metavar="FILE",
        help=f"the ledger migration file (default: the value of ${MIGRATION_ENV})",
    )


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
    serve.add_argument(
        "--time-limit",
        type=int,
        choices=TIME_LIMITS_S,
        default=DEFAULT_TIME_LIMIT_S,
        metavar="SECONDS",
        help=(
            "the longest a round may run: 20, 60, 180, 300 or 600 game seconds "
            "(default: %(default)s)"
        ),
    )
    _store_options(serve, required=False)
    serve.add_argument(
        "--participant",
        default=None,
        metavar="PSEUDONYM",
        help="log this session for the participant with this pseudonym (needs --store)",
    )
    enrol_cmd = commands.add_parser("enrol", help="add a participant with research consent")
    _store_options(enrol_cmd, required=True)
    enrol_cmd.add_argument(
        "--consent-text",
        type=Path,
        required=True,
        metavar="FILE",
        help="the consent text the participant accepted",
    )
    enrol_cmd.add_argument(
        "--label",
        default=ENROL_LABEL,
        help="the consent version's label (default: %(default)s)",
    )
    return parser


def _env_migration() -> Path | None:
    value = os.environ.get(MIGRATION_ENV)
    return Path(value) if value else None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "enrol":
        migration = args.migration or _env_migration()
        if migration is None:
            parser.error(f"give --migration or set {MIGRATION_ENV}")
        try:
            pseudonym = enrol(args.store, migration, args.consent_text, args.label)
        except (lw.LedgerError, sqlite3.Error) as exc:
            print(f"chasetag-arena: {exc}", file=sys.stderr)
            return 2
        print(f"pseudonym: {pseudonym}")
        print(
            "keep this pseudonym: pass it to serve --participant to log a session", file=sys.stderr
        )
        return 0
    if (args.participant is None) != (args.store is None):
        parser.error("--store and --participant are given together, to log a session")
    migration = args.migration
    if args.participant is not None:
        migration = args.migration or _env_migration()
        if migration is None:
            parser.error(f"give --migration or set {MIGRATION_ENV}")
    options = ServeOptions(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        static_dir=args.static_dir,
        planner=args.planner,
        store=args.store,
        migration=migration,
        participant=args.participant,
        time_limit_s=args.time_limit,
    )
    try:
        serving = start_serving(options)
    except (PlannerError, lw.LedgerError, sqlite3.Error) as exc:
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
