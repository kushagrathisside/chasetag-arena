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
uv run --no-sync python -m pytest -m "not slow"   # quick, while you work
uv run --no-sync python -m pytest                 # everything, before the pull request
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

The code is licensed under the [Apache License 2.0](LICENSE), an open-source licence.

By opening a pull request you agree that your contribution is licensed under the same licence, as
section 5 of the licence says. You keep the copyright in your work, and you confirm that you wrote it,
or that you have the right to share it under this licence. There is no separate agreement to sign. Do
not add code or files copied from elsewhere unless their licence allows it, and say where they came
from in the pull request.

Data and trained planners shared in the Hugging Face organization are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), unless the card of the repository they go
to says otherwise.
