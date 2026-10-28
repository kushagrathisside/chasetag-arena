// Tests for the arena's body size and role badge size. Run with: node --test tests/js/
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const render = require(
  fileURLToPath(new URL("../../src/chasetag_arena/arcade/static/render.js", import.meta.url)),
);

test("a body's radius is 3.5% of the arena width", () => {
  assert.ok(Math.abs(render.bodyRadiusPx(630) - 22.05) < 1e-9);
  assert.ok(Math.abs(render.bodyRadiusPx(358) / 358 - 0.035) < 1e-9);
});

// A stand-in for a 2D context's text measure: each character is `EM_PER_CHAR` ems wide, a
// generous figure for bold capitals such as CHASE and EVADE in the system UI font.
const EM_PER_CHAR = 0.72;
const stubContext = {
  font: "",
  measureText(text) {
    const px = Number(/(\d+)px/.exec(this.font)[1]);
    return { width: text.length * EM_PER_CHAR * px };
  },
};

test("a role badge is at most 1.6 body diameters wide at desktop arena sizes", () => {
  for (const sizePx of [480, 630, 760, 880, 1280]) {
    const r = render.bodyRadiusPx(sizePx);
    const metrics = render.badgeMetrics(r);
    for (const role of ["CHASE", "EVADE"]) {
      const { w } = render.badgeSize(stubContext, role, metrics);
      assert.ok(w <= 1.6 * 2 * r, `${role} is ${(w / (2 * r)).toFixed(2)} diameters at ${sizePx} px`);
    }
  }
});

test("the badge's font and padding scale with the body radius", () => {
  const small = render.badgeMetrics(render.bodyRadiusPx(630));
  const large = render.badgeMetrics(render.bodyRadiusPx(1260));
  assert.ok(Math.abs(large.fontPx / small.fontPx - 2) < 0.1, "the text size doubles with the arena");
  assert.ok(Math.abs(large.padPx / small.padPx - 2) < 1e-9, "the padding doubles with the arena");
});

test("the badge text never goes below 10 px", () => {
  assert.equal(render.badgeMetrics(render.bodyRadiusPx(200)).fontPx, 10);
});

const BOX = { x: 20, y: 20, w: 600, h: 600 };

test("a heading tick keeps its length when the arena has room for it", () => {
  const len = render.tickLengthPx({ px: 320, py: 320 }, 1, 0, 50, BOX);
  assert.equal(len, 50);
});

test("a heading tick stops at the fence it would cross", () => {
  // Centre 30 px from the right fence, aimed right: the tick stops at the fence.
  assert.equal(render.tickLengthPx({ px: 590, py: 320 }, 1, 0, 50, BOX), 30);
  // Diagonal toward the bottom-right corner: the nearer of the two fences limits it.
  const d = Math.SQRT1_2;
  const len = render.tickLengthPx({ px: 590, py: 600 }, d, d, 50, BOX);
  // The bottom fence is 20 px below the centre, so the diagonal reaches it at 20 * sqrt(2).
  assert.ok(Math.abs(len - 20 * Math.SQRT2) < 1e-9, `length ${len}`);
});

test("a heading tick at a wall stops with zero length when it points into the wall", () => {
  assert.equal(render.tickLengthPx({ px: 620, py: 320 }, 1, 0, 50, BOX), 0);
});

// A canvas context that records the path points, so the drawn tick can be read back.
function recordingContext(moves) {
  const ctx = {
    globalAlpha: 1,
    measureText: (text) => ({ width: text.length * 8 }),
  };
  return new Proxy(ctx, {
    get(target, prop) {
      if (prop in target) return target[prop];
      return (...args) => {
        if (prop === "moveTo") moves.push({ kind: "move", x: args[0], y: args[1] });
        if (prop === "lineTo") moves.push({ kind: "line", x: args[0], y: args[1] });
        return undefined;
      };
    },
    set(target, prop, value) {
      target[prop] = value;
      return true;
    },
  });
}

// The heading tick's tip: the lineTo that follows a moveTo at the body's centre.
function tickTip(moves, c) {
  const at = moves.findIndex(
    (m) => m.kind === "move" && Math.abs(m.x - c.px) < 1e-6 && Math.abs(m.y - c.py) < 1e-6,
  );
  assert.ok(at >= 0, "no tick found at the body centre");
  return moves[at + 1];
}

test("a body against the bottom wall draws its tick inside the fence", () => {
  const view = render.makeView(630, 6);
  const palette = render.readPalette(() => "");
  // A body at the bottom-right corner, heading down and to the right, toward the fence.
  const body = { seat: 0, x: 5.8, y: -5.8, heading: -Math.PI / 4, fallen: false };
  const c = view.toCanvas(body.x, body.y);
  const moves = [];
  const ctx = recordingContext(moves);
  render.drawArena(ctx, view, palette, {
    bodies: [body, { seat: 1, x: -3, y: 2, heading: 0, fallen: false }],
    chaser: 0,
    trails: [[], []],
    flash: null,
    clockS: 0,
  });
  const box = render.arenaBoxPx(view);
  const tip = tickTip(moves, c);
  assert.ok(tip.x <= box.x + box.w + 1e-6 && tip.y <= box.y + box.h + 1e-6, `tip ${tip.x}, ${tip.y} is past the fence`);
  assert.ok(tip.x >= box.x - 1e-6 && tip.y >= box.y - 1e-6);
});

test("a body in the middle draws its full heading tick", () => {
  const view = render.makeView(630, 6);
  const palette = render.readPalette(() => "");
  const body = { seat: 0, x: 0, y: 0, heading: 0, fallen: false };
  const c = view.toCanvas(0, 0);
  const moves = [];
  const ctx = recordingContext(moves);
  render.drawArena(ctx, view, palette, {
    bodies: [body, { seat: 1, x: -4, y: 4, heading: 0, fallen: false }],
    chaser: 0,
    trails: [[], []],
    flash: null,
    clockS: 0,
  });
  const tip = tickTip(moves, c);
  const r = render.bodyRadiusPx(630);
  assert.ok(Math.abs(tip.x - c.px - 2.4 * r) < 1e-6, `tick length ${tip.x - c.px}`);
});
