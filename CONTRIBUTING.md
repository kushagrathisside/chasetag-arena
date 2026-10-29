# Contributing to chasetag-arena

Bug reports, documentation fixes and tests are welcome. For a larger change, open an issue first, so
that the approach can be agreed before you write the code.

## Before your first contribution

Every contributor accepts the [Contributor Licence Agreement](CLA.md) before a contribution is merged.
The agreement lets the maintainer distribute the project, including under the commercial licence that the
README mentions. You keep the copyright in your work.

The sign-up method is [to be set up before outside contributions are merged].

## Files that are not edited here

The simulator core and the bots are copies from the groundtag research repository. They are listed, with
their hashes, in [vendor/SYNC.md](vendor/SYNC.md). `tests/test_vendor.py` fails when a copy changes. Do not
edit these files:

- `src/chasetag_arena/core/`
- `src/chasetag_arena/agents/bots.py`

If you find a fault in them, open an issue. The maintainer makes the change in the research repository.

## Setting up

```bash
uv sync
uv run pre-commit install
```

## Checks

Run these before you open a pull request:

```bash
uv run --no-sync python -m pytest
node --test tests/js/*.mjs
uv run --no-sync ruff check
uv run --no-sync ruff format --check
uv run --no-sync mkdocs build --strict
```

The pre-commit hooks run the Python checks before each commit. [docs/contributing.md](docs/contributing.md)
describes the rest of the workflow.

## Pull requests

- Keep one change in each pull request.
- Add or update a test for each change in behaviour.
- Change the documentation in the same pull request when behaviour changes.
- Describe what changed and why.

## Data

Match logs from the arcade can contain personal data. Do not paste them, or any pseudonym or consent
record, into an issue or a pull request.

## Licence

The project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). Contributions are
accepted under that licence and under the [CLA](CLA.md).
