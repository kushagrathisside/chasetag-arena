# Research logging

A logged session writes the rounds it plays to a research store, on the computer that runs the
arcade. Nothing is logged unless the server is started for a participant, and a participant can
only be added with research consent.

!!! warning "Not for real participants yet"
    The consent terms are a draft that has not been reviewed, and the arcade page does not yet tell
    a player that a session is being logged. Use logging for test runs only, with a stand-in
    participant, until both are in place.

## What is written

For each round of a logged session:

- the controller in each seat (the participant's keyboard, or a bot);
- each seat's score and whether it fell;
- how the round ended (a tag, a fall or the time limit), the time it took, and the number of
  decisions;
- a trajectory file: each decision's pelvis positions and headings, and the actions that were taken.

For the session: the seed it started with, its start and end times, and the round limit. The
participant is identified only by a random pseudonym. The store holds no name, email address or
device identifier.

## Consent

`enrol` records a participant's research consent, and returns the pseudonym. A session is refused
unless the participant's latest research consent is a grant. Withdrawing consent stops any new
logging for that participant.

## Starting a logged session

Add a participant, then start the arcade for them:

```bash
chasetag-arena enrol --store ~/chasetag-data --migration FILE --consent-text FILE
chasetag-arena serve --store ~/chasetag-data --migration FILE --participant PSEUDONYM
```

`FILE` for `--migration` is the ledger migration file. `--consent-text` is the consent text that
the participant accepted. The `enrol` command prints the pseudonym once. Keep it: the `serve` command
needs it.

A logged session keeps the rules of its round limit. The keyboard stays on seat A and a bot plays
seat B, and the time limit cannot be changed, so each session is recorded under one rules row. The
server refuses a change that would break this, and the [API](api.md) describes the error.

Stopping the server ends the session. If a round fails to write, the game goes on, the session is
no longer logged, and [`GET /ledger`](api.md#get-ledger) says why. A session that failed is left
unfinished and cannot be submitted.

## Checking a store

The mock player plays a short logged session, submits its play log, and reads the store back to
check it. It needs a store folder of its own, or a new temporary one:

```bash
uv run --no-sync python -m chasetag_arena.mock_player --migration FILE --rounds 3 --limit 60
```

It exits with `0` when every check passes.

## Not built yet

- A consent screen in the browser. Consent is recorded with the `enrol` command only.
- A way for a participant to upload their play log. The store checks an uploaded play log against
  its own record, so the arcade must give the participant the log it produced.
- Logging for planner seats. A planner's policy is not recorded yet.
