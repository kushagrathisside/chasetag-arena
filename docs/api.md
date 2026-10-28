# API

The server speaks plain HTTP on the address set by `--host` and `--port`. The default is
`http://127.0.0.1:8765`. It uses JSON for the control requests and for the frames, and the Server-Sent
Events format for the frame stream. It has no authentication, so it should stay on `127.0.0.1`.

## `GET /`

Returns the arcade page, `index.html`. The page loads `style.css`, `logic.js`, `render.js` and
`app.js` from the same address. Any other path under the static folder is served too. A path that
does not exist returns `404` with the text `not found`.

## `GET /healthz`

```json
{"ok": true, "version": "0.1.0"}
```

## `GET /events`

A stream of frames, in the `text/event-stream` format. The server sends two kinds of event:

- `snapshot`: sent once, when the stream opens. Its data is the latest frame, or `null` before the
  first frame exists.
- `frame`: sent for each decision after that. Its data is one frame.

Every 15 s of silence, the server sends a comment line, `: keepalive`, so the connection stays open.

Each event's data is a single line of JSON. A frame looks like this:

```json
{
  "seq": 153,
  "t_server": 15470.148,
  "round": 4,
  "phase": "running",
  "t_event": 0.3,
  "ended_by": null,
  "bodies": [
    {"seat": 0, "x": -0.007, "y": 1.047, "heading": -2.1408, "height": 0.951, "fallen": false},
    {"seat": 1, "x": 0.005, "y": -0.957, "heading": 1.5036, "height": 0.945, "fallen": false}
  ],
  "chaser": 1,
  "round_score": [0.0, 0.0],
  "match": {"seat0": 1.238, "seat1": 1.763, "tags": 3, "falls": 0, "timeouts": 0},
  "selectors": {"seat0": "greedy", "seat1": "greedy", "body": "K/default"}
}
```

| Field | Type | Meaning |
|---|---|---|
| `seq` | integer | The frame's number. It rises by one for each frame, so a client can skip a duplicate. |
| `t_server` | number | The server's monotonic clock, in seconds, when the frame was made. |
| `round` | integer | The round number, from 1. |
| `phase` | string | `running`, `ended` or `intermission`. |
| `t_event` | number | The game time in this round when the frame was made, in seconds, to the millisecond. |
| `ended_by` | string or null | `tag`, `fall` or `timeout` in the `ended` and `intermission` phases. `null` while running. |
| `bodies` | array | Two objects, one for each seat, in seat order. |
| `bodies[].seat` | integer | `0` or `1`. |
| `bodies[].x`, `bodies[].y` | number | Position on the arena in metres, to the millimetre. The arena runs from -6 to 6 on both axes. |
| `bodies[].heading` | number | Heading in radians, in (-π, π], to four decimals. |
| `bodies[].height` | number | Pelvis height in metres, to the millimetre. |
| `bodies[].fallen` | boolean | `true` once the body has fallen. |
| `chaser` | integer | The seat that chases in this round. |
| `round_score` | array | The score of each seat. It is the round's score in the `ended` phase. In the other phases it is the score of the round that has just ended, or zero while a round runs. |
| `match` | object | `seat0` and `seat1` are the summed scores since the last restart. `tags`, `falls` and `timeouts` count the rounds. |
| `selectors` | object | `seat0` and `seat1` are the players now in each seat. `body` is `K/default`. |

Frames are full snapshots, so a viewer that misses one loses nothing once the next one arrives. A slow
viewer may miss frames, but it never blocks the game.

## `POST /control`

Sends one control. The body is a JSON object, and its `kind` field names the control:

| Kind | Body | Effect |
|---|---|---|
| `selector` | `{"kind": "selector", "seat": 0, "value": "greedy"}` | Sets who plays a seat. `value` is one of `keyboard`, `planner`, `greedy`, `intercept`, `wander`, `idle` or `fallback`. It applies at the next round start. `planner` is accepted, and it plays the `fallback` bot until planner releases exist. |
| `keyboard` | `{"kind": "keyboard", "seat": 0, "cmd": [0.0, 1.5]}` | Holds a velocity for a seat, in metres per second, in the world frame. Each component must be within ±1.5. The command expires 0.25 s after it arrives unless it is sent again. |
| `pause` | `{"kind": "pause"}` | Stops decisions. |
| `play` | `{"kind": "play"}` | Resumes decisions. |
| `step` | `{"kind": "step"}` | Makes one decision while paused. |
| `restart` | `{"kind": "restart"}` | Starts a new round and clears the match totals. |
| `speed` | `{"kind": "speed", "value": 2}` | Sets the speed multiplier. `value` must be above 0 and at most 8. |

`seat` is `0` or `1`. Numbers must be finite, and a boolean is not a number. A field that the kind does
not take is an error, so a typo does not pass silently.

Responses:

| Status | Body | When |
|---|---|---|
| `202` | `{"accepted": true}` | The control is queued. The game applies it on its next decision. |
| `400` | `{"error": "..."}` | The body is not valid JSON, or the control is not valid. The message says what is wrong. |
| `411` | `{"error": "Content-Length is required"}` | The request has no `Content-Length` header. |
| `413` | `{"error": "body must be at most 4096 bytes"}` | The body is larger than 4096 bytes. |

Other responses:

| Status | When |
|---|---|
| `405` | `GET /control`. The response has `Allow: POST`. |
| `404` | A `POST` to any other path. |

A `202` means the server took the control. It does not mean the game has applied it yet. The change
shows in the next frame.
