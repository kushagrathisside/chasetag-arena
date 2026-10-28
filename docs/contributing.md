# Contributing

This page is for people who change the game, its documentation or its diagrams. To play, you do not
need any of it.

## Set up a checkout

Install the project and its test tools once:

```bash
uv sync
uv run --no-sync pre-commit install
```

Install the documentation tools when you work on the docs:

```bash
uv sync --group docs
```

Use `uv run --no-sync` for the other commands. It stops `uv` from changing the environment while you
work.

## Layout

```text
src/chasetag_arena/
    cli.py               the chasetag-arena command
    planner.py           loads and checks planner files
    arcade/              the server and the browser page
        server.py        HTTP handlers, the event stream, the static files
        game_loop.py     the game thread
        controls.py      parsing of controls and the held keyboard command
        frames.py        building frames and encoding them as JSON
        static/          index.html, style.css, logic.js, render.js, app.js
    agents/bots.py       the scripted bots (copied in, see below)
    core/                the simulator core (copied in, see below)
tests/                   Python tests; tests/js holds the Node tests of the page
scripts/                 screenshots, diagram export, and the core sync tool
docs/                    this site, its diagrams and screenshots
```

### The copied files

`src/chasetag_arena/core/` and `src/chasetag_arena/agents/bots.py` are copied from the author's
research repository. Do not edit them here. A test fails when a copied file changes, and the change
belongs upstream. The maintainer brings updates in with `scripts/sync_core.py`, which also records
the upstream commit in `vendor/SYNC.md`.

## Tests

The Python tests use pytest:

```bash
uv run --no-sync python -m pytest
```

The page's logic and drawing code have their own tests, which run under Node 20 or later:

```bash
node --test tests/js/*.mjs
```

## Lint and format

```bash
uv run --no-sync ruff check
uv run --no-sync ruff format --check
```

The `pre-commit` hooks run these checks, plus trailing-space and large-file checks, before each
commit.

## Continuous integration

Three workflows run on GitHub.

| Workflow | When it runs | What it does |
|---|---|---|
| **CI** | Every push to `main`, and every pull request | Runs ruff, the Python tests, the Node tests, and a strict docs build. |
| **Documentation** | Every push to `main` | Builds the site and publishes it to GitHub Pages. |
| **Release** | A tag named `vX.Y.Z` | Checks the tag matches the version in `pyproject.toml`, runs the tests, builds the wheel and sdist, and publishes them on a GitHub release. |

To publish a release, bump `version` in `pyproject.toml`, commit, and push a matching tag:

```bash
git tag v0.1.1
git push origin v0.1.1
```

## Documentation

Build the site the same way the Documentation workflow does. A broken link or a missing page fails
the build:

```bash
uv run --no-sync mkdocs build --strict
```

Preview it while you write:

```bash
uv run --no-sync mkdocs serve
```

### Diagrams

The diagrams are draw.io files in `docs/diagrams/`, one `.drawio` source for each `.svg`. Edit a
diagram in the draw.io desktop app, or on app.diagrams.net, then export every SVG:

```bash
bash scripts/export_diagrams.sh
```

The script needs the draw.io desktop app on your path as `drawio`. Without a display, it runs under
`xvfb-run` when that is installed. Commit the `.drawio` source together with its `.svg`.

### Screenshots

The screenshots come from a running server. Start one on a spare port, without a browser window:

```bash
uv run --no-sync chasetag-arena serve --no-browser --port 8799
```

In a second terminal, capture the pages:

```bash
node scripts/capture_screenshots.mjs --url http://127.0.0.1:8799/ --out docs/assets/screenshots
```

The script needs Google Chrome, and it takes `--chrome`, the path to the browser, and `--port`, the
DevTools port. Stop the server when you are done.

## Before you open a pull request

1. Run the tests, the lint and the strict docs build.
2. Describe what changed and why, and say how you checked it.
3. For a change to the page, attach a screenshot of the result.
