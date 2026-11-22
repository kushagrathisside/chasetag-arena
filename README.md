# chasetag-arena

[![CI](https://github.com/kushagrathisside/chasetag-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/kushagrathisside/chasetag-arena/actions/workflows/ci.yml)
[![Licence: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/licence-PolyForm%20Noncommercial%201.0.0-blue.svg)](LICENSE)

chasetag-arena is a one-on-one chase-tag game that runs in a web browser on your own computer. Two
bodies share a fenced 12 m × 12 m arena. One chases and the other runs, and the roles change each
round. Each seat is played by a person at the keyboard, a scripted bot, or a trained planner.

![A round in progress in the arcade](docs/assets/screenshots/round-running.png)

**Documentation:** <https://kushagrathisside.github.io/chasetag-arena/>

## Contents

- [Quick start](#quick-start)
- [Playing](#playing)
- [Command-line reference](#command-line-reference)
- [Rules and scoring](#rules-and-scoring)
- [Planners](#planners)
- [Research logging](#research-logging)
- [Development](#development)
- [Contributing](#contributing)
- [Licence](#licence)

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). From a checkout of this repository:

```bash
uv sync
uv run chasetag-arena serve
```

The arcade opens at <http://127.0.0.1:8765/>. Press <kbd>Ctrl</kbd>+<kbd>C</kbd> in the terminal to
stop it. The [install guide](https://kushagrathisside.github.io/chasetag-arena/install/) explains each
step, and the [first round](https://kushagrathisside.github.io/chasetag-arena/getting-started/)
walks through a first game.

## Playing

The arcade has two seats, **A** and **B**. Choose who plays each seat from the menus above the arena.
A change applies at the start of the next round.

- Play with the arrow keys against bots, or watch two bots play.
- Choose the player of each seat. A change applies at the next round.
- Pause, step, restart, and change the speed.
- Round and match scores, with tags, falls and time-outs counted.
- Load a planner file with `--planner FILE` and watch it play a seat.
- Opt in, with `--contribute`, to record your own keyboard games on your computer and share them.
  Nothing is recorded until you accept a consent screen.

## Community and data

The data, trained planners and demos of chasetag-arena are shared by the **ChaseTag Arena**
organization on Hugging Face: <https://huggingface.co/chasetag-arena>.

- **Dataset:** [chasetag-arena-bot-episodes](https://huggingface.co/datasets/chasetag-arena/chasetag-arena-bot-episodes)
  holds 480,000 games between five scripted bots, in 24 seat pairings at 20 s and 60 s rounds.
- **Planners:** trained planners that you can load with `--planner FILE`, each with a model card.

Everyone who plays the game or builds on it is welcome to share data, models and results there. See
[Community and data](https://kushagrathisside.github.io/chasetag-arena/community/) in the
documentation.

## Development

```bash
uv sync
uv run --no-sync python -m pytest
node --test tests/js/*.mjs
uv run --no-sync ruff check
uv run --no-sync ruff format --check
uv run --no-sync mkdocs build --strict
```

The Python package is in `src/chasetag_arena/`:

| Path | Contents |
|---|---|
| `core/` | The two-seat simulator and the body models. |
| `agents/` | The scripted bots. |
| `arcade/` | The game loop, the HTTP server and the browser page. |
| `planner.py` | Loads and checks planner files. |
| `recording.py`, `ledger_writer.py`, `mock_player.py` | Research logging and its checks. |
| `cli.py` | The command-line interface. |

The [contributing guide](https://kushagrathisside.github.io/chasetag-arena/contributing/) covers the
layout in more detail, the diagrams, the screenshots and the release process.

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first. Contributions are accepted
under the project licence and the [Contributor Licence Agreement](CLA.md), which is a draft under
review.

## Licence

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, change and share the
game for any noncommercial purpose. Commercial use needs a separate licence from the author. The
[licence page](https://kushagrathisside.github.io/chasetag-arena/license/) explains the terms in plain
language.
