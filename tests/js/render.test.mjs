// Tests for the arena's body size and role badge placement. Run with: node --test tests/js/
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const render = require(
  fileURLToPath(new URL("../../src/chasetag_arena/arcade/static/render.js", import.meta.url)),
);

const SIZE_PX = 630;
const RADIUS_PX = render.bodyRadiusPx(SIZE_PX);
const SIZES = [
  { w: 60, h: 29 },
  { w: 62, h: 29 },
];

function boxOf(centre, size) {
  return { x: centre.cx - size.w / 2, y: centre.cy - size.h / 2, w: size.w, h: size.h };
}

function overlap(a, b) {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

test("a body's radius is 3.5% of the arena width", () => {
  assert.ok(Math.abs(render.bodyRadiusPx(630) - 22.05) < 1e-9);
  assert.ok(Math.abs(render.bodyRadiusPx(358) / 358 - 0.035) < 1e-9);
});

test("a lone badge goes above its body", () => {
  const anchors = [
    { px: 100, py: 200 },
    { px: 400, py: 500 },
  ];
  const [a] = render.placeBadges(anchors, RADIUS_PX, SIZES, [0, 1]);
  assert.ok(a.cy < 200, `badge centre ${a.cy} should be above the body at 200`);
  assert.equal(a.cx, 100);
});

test("the evader's badge moves below its body when the chaser's badge is above it", () => {
  // Two bodies stacked 12 px apart: the upper badge would cover the lower one, so the
  // chaser's badge (placed first) stays above and the evader's moves below.
  const anchors = [
    { px: 300, py: 300 },
    { px: 300, py: 312 },
  ];
  const [chaser, evader] = render.placeBadges(anchors, RADIUS_PX, SIZES, [0, 1]);
  assert.ok(chaser.cy < 300, "the chaser's badge is above its body");
  assert.ok(evader.cy > 312, "the evader's badge is below its body");
  assert.equal(overlap(boxOf(chaser, SIZES[0]), boxOf(evader, SIZES[1])), false);
});

test("the two badges never overlap, for any close pair of bodies and either chaser", () => {
  for (let dx = 0; dx <= 80; dx += 4) {
    for (let dy = -80; dy <= 80; dy += 2) {
      for (const order of [
        [0, 1],
        [1, 0],
      ]) {
        const anchors = [
          { px: 300, py: 300 },
          { px: 300 + dx, py: 300 + dy },
        ];
        const centres = render.placeBadges(anchors, RADIUS_PX, SIZES, order);
        const boxes = centres.map((c, i) => boxOf(c, SIZES[i]));
        assert.equal(
          overlap(boxes[0], boxes[1]),
          false,
          `badges overlap for dx ${dx}, dy ${dy}, order ${order}`,
        );
      }
    }
  }
});

test("no bodies gives no badges", () => {
  assert.deepEqual(render.placeBadges([], RADIUS_PX, [], []), []);
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
