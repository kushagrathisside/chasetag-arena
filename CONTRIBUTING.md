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

## Claiming an issue

Issues for newcomers carry the `help wanted` label, a difficulty label, an epic label and a domain
label (`domain: frontend`, `backend`, `simulation`, `ml`, `data`, `documentation` or `testing`). Filter
by domain to find the work that suits your skills.

- Take only an issue that is marked `status: ready`. An issue marked `status: blocked` is waiting for
  other issues, which it names under "Blocked by"; the [contributor dashboard](https://kushagrathisside.github.io/chasetag-arena/dashboard/)
  shows the state of every issue, live from GitHub.
- Comment on the issue to ask for it, and wait to be assigned. One person works on an issue at a
  time, and you can ask questions in the issue before you start.
- If you can no longer work on it, say so. An assigned issue with no activity for 14 days may be
  given to someone else, after a reminder in the issue. There is no deadline otherwise.
- An issue may need another issue of the same epic to be merged first. The dashboard and the "Blocked by" section say which.
- Some areas are kept by the maintainer (the simulator core, research logging, the game server,
  and the CI setup). A task never needs a change there; open an issue if you find a problem in one.

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
