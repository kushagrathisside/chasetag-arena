# chasetag-arena

A grounded 2D chase-tag game. Two bodies share a fenced arena: one chases, one runs, and a tag
or a fall ends the round. Each seat can be played by a scripted bot, by the keyboard, or by a
trained planner.

Status: in development. The package is not yet usable and its interface will change.

The simulator core is vendored from the `groundtag` research repository. `vendor/SYNC.md` records
the commit it was copied from and the hash of every copied file.

## Development

```
uv sync
uv run --no-sync pytest -q
uv run --no-sync ruff check
uv run --no-sync ruff format --check
```

To refresh the vendored core from a `groundtag` checkout:

```
uv run --no-sync python scripts/sync_core.py /path/to/groundtag
```
