# Bodies

A body is the physics that carries a player around the arena. It decides how fast the body starts and
stops, how quickly it turns, how long it takes to follow a command, and whether it can fall. Every body
plays by the same rules and the same 1.5 m/s command limit. Only the physics differs.

Both seats play with the same body in a game, so the two sides always have equal physics. Choose the body
in the **Body** menu above the arena, or start the game with it:

```bash
uv run chasetag-arena serve --body dodger
```

A change in the menu applies at the start of the next round. A session that records games keeps the
`default` body, so the menu is locked there.

## The bodies

| Name | Family | What it is like |
|---|---|---|
| `default` | K | A person-like walker: 1.5 m/s, a tenth of a second of delay, and a small risk of falling. |
| `dot` | K | A point that follows commands at once and never falls. The easiest body to control. |
| `limited` | K | Like `default`, but with no command delay. |
| `sprinter` | K | Full speed forward, but slow to turn, to stop and to sidestep. |
| `dodger` | K | A bit slower, but quick to turn and sidestep. Sharp changes can make it fall. |
| `spring` | SP | A wheeled base carrying a pendulum body on a spring. A hard push tilts it, and it can tip over. |
| `stepping` | LS | An inverted pendulum that takes steps. Too much speed or a sharp turn loses balance. |

The label of a body is its family and its name, such as `K/dodger`. It appears in the page, in each frame
and in saved games.

## The three families

- **K, the kinematic body.** A point with speed limits, acceleration limits, a command delay, a tracking
  lag and noise. A fall is a chance that grows with hard turns, speed changes and sharp changes of
  command. The five K bodies are the same model with different numbers.
- **SP, the spring pendulum.** A wheeled base follows the command. A body hangs on it like an upright
  pendulum held by a spring. Accelerating tilts the body, and a tilt past a limit is a fall.
- **LS, the stepping pendulum.** The body balances on one foot and places the next foot every step. A fall
  happens when it can no longer catch its balance with the next step.

## Add a body

A new body is one function and one registration in `src/chasetag_arena/bodies.py`:

```python
def _heavy() -> BodyParams:
    """Slow to start and slow to stop."""
    return _k(a_max=1.0, a_dec=1.2, tau=0.6, delay=12)


register(BodySpec("heavy", "K", "Slow to start and to stop.", _heavy))
```

`_k` takes the default K body and changes the fields you name. The fields are in `KParams` in
`core/families/kinematic.py`. For a body with new physics, add a family in `core/families/` that follows
the `BodyModel` protocol in `base.py`, then register a body that uses it.

Then check it:

1. `tests/test_bodies.py` runs every registered body through a round, so a body that produces `nan` or
   leaves the arena fails there.
2. `tests/test_body_fairness.py` plays 20 s games with the scripted bots. A body must almost never fall at
   rest or with two greedy bots, a chaser must be able to catch a wandering runner, and neither side may
   win every game between equal bots.
3. Play it yourself with `--body heavy`.

Keep the summary to one sentence that ends in a full stop. The page shows it under the menu.
