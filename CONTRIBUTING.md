# Contributing to chasetag-arena

Bug reports, documentation fixes and tests are welcome. For a larger change, open an issue first, so
that the approach can be agreed before you write the code.

## Where to contribute

- **Code, tests and documentation** go here, as pull requests to this repository.
- **Data, trained planners and results** go to the ChaseTag Arena organization on Hugging Face,
  <https://huggingface.co/chasetag-arena>. Its contribution guide explains what each contribution
  needs and how it is reviewed.

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

The project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE), and contributions to
this repository are accepted under that licence. Data and models shared in the Hugging Face organization
are accepted under the licence of the repository they go to, which its card states.
