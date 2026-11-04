# Training planners

A [planner](planners.md) is made by training, outside the game. The repository includes the tools for
it. They run on the CPU with no browser. The steps are to make or download a dataset of bot games,
clone a starting planner from it, improve that planner by reinforcement learning against every bot
type and earlier copies of itself, and rank it in a league. The result is an ordinary planner file
that you load with `--planner`.

## The steps

### 1. Make a dataset of bot games

The cloning step reads recorded bot games, in one archive for each pairing of two bots at each round
length. You can use the [published dataset](https://huggingface.co/datasets/chasetag-arena/chasetag-arena-bot-episodes),
which covers 20 s and 60 s rounds, or make your own:

```bash
uv run --no-sync python -m chasetag_arena.dataset --out data --episodes 10000 --limits 20,60
```

It writes one zip for each of the 24 pairings, which are all ordered pairs of the five bots except
idle against idle, at each round length, with 10,000 games in each. It also writes a
`manifest.json` with a checksum and a seed for each file. It skips archives that are already
complete, so a stopped run can be started again.

### 2. Clone a starting planner

```bash
uv run --no-sync python -m chasetag_arena.bc --data data --limits 20 --out bc20.npz
```

This fits a network to the commands that the winning seat chose in games that did not end in a
fall. The idle bot is never cloned. The archives do not store velocities, so the planner's four
velocity features are estimated from the change in position over a decision. The command writes
`bc20.npz` and a report, `bc20.json`, with its scores against each bot.

### 3. Train

```bash
uv run --no-sync python -m chasetag_arena.train --run-dir runs/k20 --limits 20 --iters 400 \
  --init bc20.npz --critic-warmup 10 --init-log-std -1.0
```

Each training game draws an opponent: the planner itself, one of the five bots, or a frozen earlier
copy of the planner, drawn more often when the planner scores worse against it. The planner plays a
random seat. The reward is the official score, paid in small parts through the game so that long
rounds are learnable. Every 25 rounds the planner is scored against each bot, split by role, and the
best planner so far is saved.

The run folder holds `planner.npz` (the best planner), `checkpoint.pkl` (to resume), `metrics.jsonl`
(one line per round and per evaluation) and `config.json`. Useful options:

| Option | What it does |
|---|---|
| `--limits 20,60,180` | Train one planner for several round lengths. |
| `--iters N` | The number of rounds. Each round plays 512 games per round length. |
| `--init FILE` | Start from a planner file, such as one made by cloning. |
| `--resume` | Continue a stopped run from its checkpoint. |
| `--hidden 64,64` | The actor's layer sizes. |
| `--seed N` | The random seed. |

Starting from a cloned planner learns much faster than starting from nothing. In one test of 150
rounds at 20 s, the cloned start reached a score of 0.88 against the bots, and the run from scratch
reached 0.38.

### 4. Rank it in a league

```bash
uv run --no-sync python -m chasetag_arena.league --planner runs/k20/planner.npz \
  --formats 20 --out standings
```

Each round length is a division. Every pair of teams meets twice, and each meeting has two legs, so
each team chases once and runs once. A win is 3 points and a draw is 1. The standings are written
to `standings/standings.md` and `standings/standings.json`. Give `--planner` more than once to rank
several planners together.

### 5. Play it

```bash
uv run chasetag-arena serve --planner runs/k20/planner.npz --time-limit 20
```

## Train four planners in one go

`scripts/train_planners.sh DATA RUNS` trains a planner for 20 s, one for 60 s, one for 180 s and one
for all three, one after another. It needs 24 archives at each of 20 s, 60 s and 180 s in `DATA`. The
published dataset has no 180 s archives, so make those yourself with `--limits 20,60,180`. If the
run is stopped, start it again and it resumes.

To keep it running after you close the terminal:

```bash
setsid nohup scripts/train_planners.sh data runs > train.log 2>&1 < /dev/null &
```

Some systems end your processes when you log out. If yours does, stay logged in until it finishes.

## What to expect

These numbers are from one run of 400 rounds on one 20-core CPU, with 512 games per round length in
each round and the seed 0. The score is the mean official score against the greedy, intercept, wander
and fallback bots, on games that were not used to pick the saved planner. Your times and scores will
differ.

| Planner | Training time | Cloned start | After training |
|---|---|---|---|
| 20 s | about 31 minutes | 0.71 | 0.86 |
| 60 s | about 39 minutes | 0.53 | 0.86 |
| 180 s | about 40 minutes | 0.30 | 0.74 |
| All three lengths | about 83 minutes | 0.57 | 0.82 |

Making a 10,000-game archive took about half a minute at 20 s, and longer rounds take longer.

## Limits to keep in mind

- **The opponents are scripted bots.** A score against them says nothing about how the planner plays
  against a person.
- **One run is one seed.** Another seed gives a somewhat different planner.
- **A planner for all lengths trades strength for coverage.** In a league, the planner trained for 20 s
  beat the all-lengths planner at 20 s.
- **Only the actor is saved.** The trainer's value network, which sees more than the planner does, is
  used in training only and is not in the planner file.
