# Development

## Package layout

```
src/chasetag_arena/
    __main__.py          python -m chasetag_arena
    cli.py               the serve command
    arcade/              the server and the browser client
        server.py        HTTP handlers, the event stream, static files
        game_loop.py     the game thread
        controls.py      control parsing and the keyboard state
        frames.py        frame building and JSON encoding
        static/          index.html, style.css, logic.js, render.js, app.js
    agents/bots.py       vendored: the scripted bots
    core/                vendored: the simulator core (spec/, families/, and the modules)
tests/                   Python tests, and tests/js for the Node tests of the client
scripts/                 sync_core.py, capture_screenshots.mjs, export_diagrams.sh
vendor/SYNC.md           the groundtag commit and the hash of each vendored file
docs/                    this site
```

Files under `core/` and `agents/bots.py` are vendored. Do not edit them by hand. See
[the vendored core](#the-vendored-core) below.

## Setup

In your own checkout, install the project once:

```
uv sync
```

The documentation tools are in their own group. Install them once, in your own checkout:

```
uv sync --group docs
```

Use `uv run --no-sync` for the other commands, so that nothing is reinstalled while you work.

## Tests

The Python tests use pytest:

```
uv run --no-sync python -m pytest
```

The client logic and the drawing functions are tested under Node 22:

```
node --test tests/js/*.mjs
```

## Lint and format

```
uv run --no-sync ruff check
uv run --no-sync ruff format --check
```

`uv run pre-commit install` runs the same checks before each commit. Run it once after cloning.

## The vendored core

The simulator core and the bots come from the groundtag research repository. `scripts/sync_core.py`
is the only way they change. Run it from this checkout, and give it the path to a groundtag checkout:

```
uv run --no-sync python scripts/sync_core.py /path/to/groundtag
```

The script copies the files, rewrites the imports, and writes `vendor/SYNC.md`. It stops if the
groundtag source tree has uncommitted changes. `tests/test_vendor.py` fails when a vendored file
differs from `vendor/SYNC.md`, and it fails when a hand edit is the cause. If that test fails, undo the
hand edit, or sync again.

## Screenshots

The screenshots in these pages come from a running server. Start one on a spare port, without a
browser window:

```
uv run --no-sync python -m chasetag_arena serve --no-browser --port 8799
```

In a second terminal, capture the pages:

```
node scripts/capture_screenshots.mjs --url http://127.0.0.1:8799/ --out docs/assets/screenshots
```

The script needs Google Chrome. Its options are `--chrome` (the browser path, default
`/usr/bin/google-chrome`), `--port` (the DevTools port, default 9333) and `--url`. It writes
`desktop-light.png`, `desktop-dark.png`, `phone-light.png` and `round-running.png`. The round capture
waits until a round has run for a few seconds, and it keeps only a shot taken while that round
was still running. Stop the server when you are done.

## Diagrams

The draw.io sources are in `docs/diagrams/*.drawio`. Edit them in the draw.io desktop app, or on
app.diagrams.net, then export the SVG files:

```
bash scripts/export_diagrams.sh
```

The script needs the draw.io desktop app on the path as `drawio`. Without a display, it runs under
`xvfb-run` when that is installed. It writes an SVG next to each source, and it removes the PNG copies
of the labels that draw.io adds, which the site does not need.

## Documentation site

Build the site with the same settings the site uses. A broken link or a missing page fails the build:

```
uv run --no-sync mkdocs build --strict
```

Preview the site while you write:

```
uv run --no-sync mkdocs serve
```
