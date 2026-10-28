# chasetag-arena

chasetag-arena is a grounded 2D chase-tag game. Two bodies share a fenced arena of 12 m by 12 m.
One body chases and the other runs. A tag or a fall ends the round, and a round that lasts 20 s
of game time ends in a time-out.

Each seat is played by a scripted bot, by the keyboard, or by a trained planner file given with
`--planner`.

![The arcade page in light mode, with a round running](assets/screenshots/desktop-light.png)

## What works today

- The local server, started with `chasetag-arena serve`. It serves the page, a frame stream and
  a control endpoint.
- The game loop. It plays rounds one after another, with an intermission between them.
- The browser client: the arena, the controls, the keyboard seat, and the round and match scores.
- Seat selectors for the keyboard and the five bots: greedy, intercept, wander, idle and fallback.
- The planner seat, with a planner file given by `--planner FILE`.

## What does not work yet

- Planner releases. A planner is given by its file path, and downloads and checksums are not built.
- Saving rounds, the learning history, the compare panel, logs submission and a desktop
  installer.

See the [roadmap](roadmap.md) for the full list.

## Where to go next

- [Getting started](getting-started.md) installs the game and starts it.
- [Playing](playing.md) explains the controls and the screen.
- [Architecture](architecture.md) and the [API](api.md) describe how the parts fit together.
- [Development](development.md) covers the tests, the lint, the vendored core and the docs.
