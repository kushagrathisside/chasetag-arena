# chasetag-arena

A one-on-one chase-tag game that runs in your browser, on your own computer. Two bodies share a
fenced 12 m × 12 m arena. One chases, the other runs, and the roles swap every round. Each seat is
played by you (keyboard), a scripted bot, or a trained planner.

**Documentation:** <https://kushagrathisside.github.io/chasetag-arena/>

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). From a checkout of this repository:

```bash
uv sync
uv run chasetag-arena serve
```

The game opens at `http://127.0.0.1:8765/`. Press Ctrl+C in the terminal to stop it.

The [install guide](https://kushagrathisside.github.io/chasetag-arena/install/) explains each step,
and the [first round](https://kushagrathisside.github.io/chasetag-arena/getting-started/) walks you
through your first game.

## Features

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
```

The [contributing guide](https://kushagrathisside.github.io/chasetag-arena/contributing/) covers the
layout, the diagrams, the screenshots, the docs build and the release process.

## Licence

Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE). You may use, change and share the
game for any noncommercial purpose. Commercial use needs a separate licence from the author. The
[license page](https://kushagrathisside.github.io/chasetag-arena/license/) explains the terms in
plain language.
