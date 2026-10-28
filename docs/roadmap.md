# Roadmap

This page lists what is built and what is planned. A planned item is not part of the game today.

## Done

- **The server**: HTTP, the event stream, the control endpoint, and the static files of the page.
- **The frame contract**: one full frame per decision. See the [API](api.md).
- **The game loop**: rounds, the intermission between them, pause, step, restart and speed.
- **The browser client**: the arena, the controls, the status and the round and match scores.
- **The keyboard seat**: the arrow keys drive a body.
- **The selectors**: keyboard, greedy, intercept, wander, idle and fallback.

## Planned, not built

- **Planner releases**: downloading a planner release, checking its checksum, and the planner seat.
  Until then, the planner choice is disabled in the page. The server plays the fallback bot in its place.
- **Learning history panel**.
- **Saving rounds**.
- **Compare panel and preference rows**.
- **Logs submission and its validator**.
- **Desktop installer**.
- **Central training in the research repository**.
- **Other bodies**: only the default body, `K/default`, is in the code today.
