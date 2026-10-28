// Tests for the arcade client's pure functions. Run with: node --test tests/js/
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const logic = require(
  fileURLToPath(new URL("../../src/chasetag_arena/arcade/static/logic.js", import.meta.url)),
);

const PI = Math.PI;
const TAU = 2 * Math.PI;
const EPS = 1e-9;

function near(actual, expected, eps = EPS) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `expected ${actual} to be within ${eps} of ${expected}`,
  );
}

// The angular distance between two angles, on the circle.
function angularGap(a, b) {
  return Math.abs(logic.wrapAngle(a - b));
}

function frame(seq, round, bodies, extra = {}) {
  return {
    seq,
    t_server: seq * 0.1,
    round,
    phase: "running",
    t_event: seq * 0.1,
    ended_by: null,
    bodies,
    chaser: 0,
    round_score: [0, 0],
    match: { seat0: 0, seat1: 0, tags: 0, falls: 0, timeouts: 0 },
    selectors: { seat0: "keyboard", seat1: "greedy", body: "K/default" },
    ...extra,
  };
}

function body(seat, x, y, heading, fallen = false) {
  return { seat, x, y, heading, height: 0.95, fallen };
}

// --- wrapAngle ---------------------------------------------------------------------------

test("wrapAngle keeps angles inside (-pi, pi] and maps the seam to pi", () => {
  assert.equal(logic.wrapAngle(PI), PI);
  assert.equal(logic.wrapAngle(-PI), PI, "-pi is part of the interval as pi");
  assert.equal(logic.wrapAngle(PI + 0.1), -PI + 0.1);
  assert.equal(logic.wrapAngle(-PI - 0.1), PI - 0.1);
  near(logic.wrapAngle(3 * PI), PI);
  near(logic.wrapAngle(-3 * PI), PI);
  near(logic.wrapAngle(TAU), 0);
  near(logic.wrapAngle(7), 7 - TAU);
  assert.equal(logic.wrapAngle(0), 0);
  assert.equal(logic.wrapAngle(-PI / 2), -PI / 2);
});

test("wrapAngle output is always in (-pi, pi]", () => {
  for (let a = -20; a <= 20; a += 0.37) {
    const w = logic.wrapAngle(a);
    assert.ok(w > -PI && w <= PI, `wrapAngle(${a}) = ${w}`);
  }
});

// --- lerpAngle ---------------------------------------------------------------------------

test("lerpAngle goes the short way round the seam", () => {
  // From 3.0 to -3.0 the short arc crosses pi (about 0.28 rad), not the long way round.
  const mid = logic.lerpAngle(3.0, -3.0, 0.5);
  assert.ok(angularGap(mid, PI) < 1e-6, `midpoint ${mid} should be near pi`);
  near(logic.lerpAngle(3.0, -3.0, 0), 3.0);
  near(angularGap(logic.lerpAngle(3.0, -3.0, 1), -3.0), 0);
});

test("lerpAngle interpolates linearly when there is no seam", () => {
  near(logic.lerpAngle(0, 1, 0.5), 0.5);
  near(logic.lerpAngle(-0.5, 0.5, 0.5), 0);
  near(logic.lerpAngle(1, 1, 0.7), 1);
});

// --- interpolateFrames -------------------------------------------------------------------

test("interpolateFrames returns the older frame at 0, the midpoint at 0.5, the newer at 1", () => {
  const a = frame(10, 1, [body(0, 0, 0, 0), body(1, 2, 2, 0.2)]);
  const b = frame(11, 1, [body(0, 1, -1, 1.0), body(1, 4, 0, 0.6)]);

  const at0 = logic.interpolateFrames(a, b, 0);
  assert.deepEqual(at0.map((p) => [p.seat, p.x, p.y]), [
    [0, 0, 0],
    [1, 2, 2],
  ]);
  near(at0[0].heading, 0);
  near(at0[1].heading, 0.2);

  const half = logic.interpolateFrames(a, b, 0.5);
  near(half[0].x, 0.5);
  near(half[0].y, -0.5);
  near(half[0].heading, 0.5);
  near(half[1].x, 3);
  near(half[1].y, 1);
  near(half[1].heading, 0.4);

  const at1 = logic.interpolateFrames(a, b, 1);
  assert.deepEqual(at1.map((p) => [p.x, p.y]), [
    [1, -1],
    [4, 0],
  ]);
  near(at1[0].heading, 1.0);
});

test("interpolateFrames uses seq order, not argument order", () => {
  const older = frame(5, 1, [body(0, 0, 0, 0), body(1, 0, 0, 0)]);
  const newer = frame(6, 1, [body(0, 2, 0, 0), body(1, 0, 0, 0)]);
  const swapped = logic.interpolateFrames(newer, older, 0.25);
  near(swapped[0].x, 0.5, 1e-9);
});

test("interpolateFrames blends the heading along the short arc", () => {
  const a = frame(1, 1, [body(0, 0, 0, 3.0), body(1, 0, 0, 0)]);
  const b = frame(2, 1, [body(0, 0, 0, -3.0), body(1, 0, 0, 0)]);
  const mid = logic.interpolateFrames(a, b, 0.5)[0].heading;
  assert.ok(angularGap(mid, PI) < 1e-6, `heading midpoint ${mid} should be near pi`);
  assert.ok(mid > -PI && mid <= PI);
});

test("interpolateFrames holds the old positions across a round change, then snaps", () => {
  const old = frame(20, 3, [body(0, 1, 1, 0), body(1, -1, -1, 0)]);
  const fresh = frame(21, 4, [body(0, -5, 5, 0), body(1, 5, -5, 0)]);
  const early = logic.interpolateFrames(old, fresh, 0.5);
  assert.deepEqual(early.map((p) => [p.x, p.y]), [
    [1, 1],
    [-1, -1],
  ]);
  const late = logic.interpolateFrames(old, fresh, 1);
  assert.deepEqual(late.map((p) => [p.x, p.y]), [
    [-5, 5],
    [5, -5],
  ]);
});

test("interpolateFrames clamps t outside [0, 1]", () => {
  const a = frame(1, 1, [body(0, 0, 0, 0), body(1, 0, 0, 0)]);
  const b = frame(2, 1, [body(0, 2, 0, 0), body(1, 0, 0, 0)]);
  near(logic.interpolateFrames(a, b, -1)[0].x, 0);
  near(logic.interpolateFrames(a, b, 3)[0].x, 2);
});

// --- trails ------------------------------------------------------------------------------

test("pushTrail keeps only the newest max points and does not change its input", () => {
  let trail = [];
  for (let i = 0; i < 35; i += 1) {
    trail = logic.pushTrail(trail, { x: i, y: -i }, 30);
    assert.ok(trail.length <= 30);
  }
  assert.equal(trail.length, 30);
  assert.equal(trail[0].x, 5, "the oldest kept point is the 6th pushed");
  assert.equal(trail[29].x, 34);
  assert.equal(trail[29].y, -34);

  const base = [{ x: 1, y: 1 }];
  const grown = logic.pushTrail(base, { x: 2, y: 2 }, 30);
  assert.equal(base.length, 1, "the input array is not changed");
  assert.equal(grown.length, 2);
  assert.deepEqual(logic.pushTrail([], { x: 0, y: 0 }, 0), []);
});

test("advanceTrails resets both trails when the round changes", () => {
  let state = null;
  state = logic.advanceTrails(state, frame(1, 7, [body(0, 0, 0, 0), body(1, 1, 1, 0)]), 30);
  state = logic.advanceTrails(state, frame(2, 7, [body(0, 0.1, 0, 0), body(1, 1, 1, 0)]), 30);
  assert.equal(state.trails[0].length, 2);
  assert.equal(state.trails[1].length, 2);

  state = logic.advanceTrails(state, frame(3, 8, [body(0, -4, 4, 0), body(1, 4, -4, 0)]), 30);
  assert.equal(state.round, 8);
  assert.deepEqual(state.trails[0], [{ x: -4, y: 4 }]);
  assert.deepEqual(state.trails[1], [{ x: 4, y: -4 }]);
});

// --- render time and buffer ----------------------------------------------------------------

test("pickRenderTime is lagMs behind now", () => {
  assert.equal(logic.pickRenderTime(1000, 100), 900);
  assert.equal(logic.pickRenderTime(250.5, 100), 150.5);
});

test("bracketFrames finds the two frames around a time, and holds at both ends", () => {
  const buffer = [1, 2, 3].map((n) => ({ seq: n, t_server: n * 0.1 }));
  const mid = logic.bracketFrames(buffer, 250); // between t = 200 ms and 300 ms
  assert.equal(mid.prev.seq, 2);
  assert.equal(mid.next.seq, 3);
  near(mid.t, 0.5, 1e-9);

  const early = logic.bracketFrames(buffer, 0);
  assert.equal(early.prev, early.next);
  assert.equal(early.prev.seq, 1);
  assert.equal(early.t, 0);

  const late = logic.bracketFrames(buffer, 9999);
  assert.equal(late.prev, late.next);
  assert.equal(late.prev.seq, 3);

  assert.equal(logic.bracketFrames([], 0), null);
});

// --- ReconnectPolicy -----------------------------------------------------------------------

test("ReconnectPolicy backs off from 0.5 s to 8 s and resets", () => {
  const policy = new logic.ReconnectPolicy();
  const delays = [];
  for (let i = 0; i < 7; i += 1) delays.push(policy.next());
  assert.deepEqual(delays, [500, 1000, 2000, 4000, 8000, 8000, 8000]);
  policy.reset();
  assert.equal(policy.next(), 500);
  assert.equal(policy.next(), 1000);
});

// --- keyboard command ------------------------------------------------------------------------

test("keyboardCommandFromKeys maps each arrow key to a world-frame command", () => {
  assert.deepEqual(logic.keyboardCommandFromKeys(["ArrowUp"]), [0, 1.5]);
  assert.deepEqual(logic.keyboardCommandFromKeys(["ArrowDown"]), [0, -1.5]);
  assert.deepEqual(logic.keyboardCommandFromKeys(["ArrowLeft"]), [-1.5, 0]);
  assert.deepEqual(logic.keyboardCommandFromKeys(["ArrowRight"]), [1.5, 0]);
});

test("keyboardCommandFromKeys caps a diagonal at 1.5 m/s", () => {
  const [vx, vy] = logic.keyboardCommandFromKeys(new Set(["ArrowUp", "ArrowRight"]));
  near(Math.hypot(vx, vy), 1.5, 1e-9);
  near(vx, vy, 1e-12);
  assert.ok(vx > 0 && vy > 0);

  const [dx, dy] = logic.keyboardCommandFromKeys(new Set(["ArrowDown", "ArrowLeft"]));
  near(Math.hypot(dx, dy), 1.5, 1e-9);
  assert.ok(dx < 0 && dy < 0);
});

test("keyboardCommandFromKeys gives zero for no keys, blur, opposite keys and other keys", () => {
  assert.deepEqual(logic.keyboardCommandFromKeys(new Set()), [0, 0]);
  assert.deepEqual(logic.keyboardCommandFromKeys([]), [0, 0]);
  assert.deepEqual(logic.keyboardCommandFromKeys(new Set(["ArrowLeft", "ArrowRight"])), [0, 0]);
  assert.deepEqual(logic.keyboardCommandFromKeys(new Set(["a", "Shift"])), [0, 0]);
  assert.deepEqual(logic.keyboardCommandFromKeys(undefined), [0, 0]);
});

// --- frame validation and text -------------------------------------------------------------

test("validateFrame accepts a well-formed frame and rejects broken ones", () => {
  const good = frame(1, 1, [body(0, 0, 0, 0), body(1, 1, 1, 0)]);
  assert.equal(logic.validateFrame(good), true);
  assert.equal(logic.validateFrame(null), false);
  assert.equal(logic.validateFrame({ ...good, phase: "paused" }), false);
  assert.equal(logic.validateFrame({ ...good, bodies: [body(0, 0, 0, 0)] }), false);
  assert.equal(logic.validateFrame({ ...good, seq: "1" }), false);
  assert.equal(logic.validateFrame({ ...good, chaser: 2 }), false);
  assert.equal(logic.validateFrame({ ...good, bodies: [body(0, NaN, 0, 0), body(1, 0, 0, 0)] }), false);
  assert.equal(logic.validateFrame({ ...good, ended_by: "lost" }), false);
});

test("describePhase gives the status lines the page shows", () => {
  const running = frame(1, 1, [body(0, 0, 0, 0), body(1, 0, 0, 0)]);
  assert.deepEqual(logic.describePhase(running, false, 0), { headline: "Running", detail: "" });
  assert.deepEqual(logic.describePhase(running, true, 0), { headline: "Paused", detail: "" });

  const tagged = { ...running, phase: "intermission", ended_by: "tag" };
  assert.deepEqual(logic.describePhase(tagged, false, 0), {
    headline: "Round over: tagged",
    detail: "Next round in 1.5 s",
  });
  assert.equal(logic.describePhase({ ...tagged, ended_by: "fall" }, false, 0).headline, "Round over: fell");
  assert.equal(
    logic.describePhase({ ...tagged, ended_by: "timeout" }, false, 0).headline,
    "Round over: time out",
  );
  assert.equal(logic.describePhase(tagged, false, 0.5).detail, "Next round in 1.0 s");
  assert.equal(logic.describePhase(tagged, false, 9).detail, "Next round in 0.0 s");
  assert.equal(logic.describePhase({ ...tagged, phase: "ended" }, false, 0).detail, "");
});

test("a paused round over shows Paused, the result, and no countdown", () => {
  const tagged = frame(1, 1, [body(0, 0, 0, 0), body(1, 0, 0, 0)], {
    phase: "intermission",
    ended_by: "tag",
  });
  assert.deepEqual(logic.describePhase(tagged, true, 0), {
    headline: "Paused",
    detail: "Round over: tagged",
  });
  assert.deepEqual(logic.describePhase(tagged, true, 9), {
    headline: "Paused",
    detail: "Round over: tagged",
  });
  const ended = { ...tagged, phase: "ended" };
  assert.deepEqual(logic.describePhase(ended, true, 0), {
    headline: "Paused",
    detail: "Round over: tagged",
  });
});

test("keys 1 to 4 map to the speeds the help panel names", () => {
  assert.deepEqual(logic.SPEED_KEYS, { 1: 0.25, 2: 0.5, 3: 1, 4: 2 });
});

test("arenaSidePx is the column width when the room below the canvas is larger", () => {
  assert.equal(logic.arenaSidePx(640, 1000, 180), 640);
});

test("arenaSidePx is the room below the canvas top when the column is wider", () => {
  // A 1280 x 900 window with the canvas 183 px down: 717 px of room, not the 936 px column.
  assert.equal(logic.arenaSidePx(936, 900, 183), 717);
});

test("arenaSidePx never goes below 360 px, even with no room", () => {
  assert.equal(logic.arenaSidePx(936, 500, 300), 360);
  assert.equal(logic.ARENA_MIN_PX, 360);
});

test("arenaSidePx returns a whole number of pixels, rounded down", () => {
  assert.equal(logic.arenaSidePx(500.7, 900, 100), 500);
  assert.equal(logic.arenaSidePx(800, 900, 183.6), 716);
});

// --- planner option ----------------------------------------------------------------------

test("plannerOptionState enables the planner option only for a loaded, named planner", () => {
  assert.deepEqual(logic.plannerOptionState({ loaded: true, name: "walk-2d", file: "w.npz" }), {
    disabled: false,
    text: "planner (walk-2d)",
  });
  assert.deepEqual(logic.plannerOptionState({ loaded: false }), {
    disabled: true,
    text: "planner (not loaded)",
  });
  assert.deepEqual(logic.plannerOptionState({ loaded: true, name: "" }), {
    disabled: true,
    text: "planner (not loaded)",
  });
  assert.deepEqual(logic.plannerOptionState(null), { disabled: true, text: "planner (not loaded)" });
});
