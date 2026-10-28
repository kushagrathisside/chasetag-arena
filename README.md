# chasetag-arena

chasetag-arena is a small, installable game of one-versus-one chase tag. Two bodies move on a
fenced 12 × 12 m arena, and roles swap each round. Each seat can be controlled by a scripted bot,
by your keyboard, or by a trained planner.

The game runs a local server and opens in your browser. It shows each round live, with the
official score, the tag and fall events, and the match tally.

Status: in development. The server, the browser client and the planner seat work. Saved rounds and
the logs submission are not built yet, and the interface will change.

The simulator core is vendored from the `groundtag` research repository. `vendor/SYNC.md` records
the commit it was copied from and the hash of every copied file.

## What works now

- Play rounds in the browser with bots (greedy, intercept, wander, idle, fallback), the keyboard, or
  a trained planner from a groundtag planner file: `uv run chasetag-arena serve --planner FILE`.
- Choose the player of each seat. A change applies at the next round.
- Pause, step, restart, and change the speed. The keys are listed on the playing page.
- The round and match scores, the tags, the falls and the time-outs.

## Quick start

Install the project once, in your checkout, then start the game:

```
uv sync
uv run chasetag-arena serve
```

The command prints the address, `http://127.0.0.1:8765/`, and opens it in your browser. Use
`--no-browser` to skip opening it. Press Ctrl+C to stop.

## Documentation

The documentation is in [`docs/`](docs/index.md). It covers the install, the controls, the
architecture, the HTTP API, the development workflow and the roadmap. It is built with MkDocs from
the `docs/` folder. See [`docs/development.md`](docs/development.md) for the build commands.

## Development

```
uv sync
uv run --no-sync python -m pytest
node --test tests/js/*.mjs
uv run --no-sync ruff check
uv run --no-sync ruff format --check
```

To refresh the vendored core from a `groundtag` checkout:

```
uv run --no-sync python scripts/sync_core.py /path/to/groundtag
```

## Licence

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may read, use, change and
share the game for any noncommercial purpose, including research, study and personal projects.
Commercial use needs a separate licence from the author. The simulator core is copied from
`groundtag`, a repository owned by the same author.
