# Roadmap

This page lists what the game does today and what is planned. A planned item is not in the game yet.

## Ready today

| Area | What works |
|---|---|
| **Playing** | Two seats, each a keyboard, a bot or a planner. Pause, Step, Restart and Speed. |
| **Rules** | Tags, falls, the round clock, and the official scores for each round. |
| **Scores** | Round scores and match totals, with tags, falls and time-outs counted. |
| **Bots** | Greedy, intercept, wander, idle and fallback. |
| **Planners** | A planner file, given with `--planner`, plays a seat, after the server checks it. |
| **Page** | The arena, the controls, the keyboard, the connection status, and a layout for phones. |
| **Server** | A local server with a live frame stream and a control endpoint. See the [API](api.md). |
| **Training** | Make a dataset of bot games, clone a starting planner, train it by reinforcement learning against every bot type, and rank planners in a league. See [Training planners](training.md). |
| **Community** | The ChaseTag Arena organization on Hugging Face shares the dataset and trained planners. See [Community and data](community.md). |
| **Recording your games** | Opt in with `--contribute`: a consent screen, local recording, review, export and withdrawal. The consent text is a draft, so packages are not accepted yet. |

## Planned

| Item | What it will do |
|---|---|
| **Planner downloads** | Download a published planner from inside the game, and check its checksum before use. Today, planners are published on Hugging Face and you download a file yourself. |
| **Learning history** | Show how each planner release scored over time. |
| **Saved rounds** | Keep a round so you can replay it later. |
| **Compare panel** | Watch two planners side by side, and compare how they scored. |
| **Preferences** | Pick the better of two rounds, to help judge which player is better. |
| **Sharing logs** | Send your recorded games to the project from inside the game, with a privacy notice first. Today, you export a package and contribute it by pull request, once the consent text is final. |
| **Desktop installer** | Install the game with one download, with no terminal needed. |
| **More bodies** | Bodies with more detail than the default one. |

Items may change or move as the project develops. Check the
[project's GitHub page](https://github.com/kushagrathisside/chasetag-arena) for the latest status.
