// Tests for where the role badges go, including the canvas edges. Run with: node --test tests/js/
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const logic = require(
  fileURLToPath(new URL("../../src/chasetag_arena/arcade/static/logic.js", import.meta.url)),
);
const render = require(
  fileURLToPath(new URL("../../src/chasetag_arena/arcade/static/render.js", import.meta.url)),
);

const SIZE_PX = 630;
const CANVAS = { w: SIZE_PX, h: SIZE_PX };
const RADIUS_PX = render.bodyRadiusPx(SIZE_PX);
// A body's centre sits at the arena's edge, which is the view margin in from the canvas edge.
const EDGE_PX = Math.max(8, SIZE_PX * 0.03);
const MARGIN = logic.BADGE_EDGE_MARGIN_PX;
const SIZES = [
  { w: 60, h: 29 },
  { w: 62, h: 29 },
];

function boxOf(centre, size) {
  return logic.boxAt(centre, size);
}

function overlap(a, b) {
  return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}

function insideCanvas(box) {
  return (
    box.x >= MARGIN &&
    box.y >= MARGIN &&
    box.x + box.w <= CANVAS.w - MARGIN &&
    box.y + box.h <= CANVAS.h - MARGIN
  );
}

test("a lone badge goes above its body", () => {
  const anchors = [
    { px: 100, py: 200 },
    { px: 400, py: 500 },
  ];
  const [a] = logic.placeBadges(anchors, RADIUS_PX, SIZES, [0, 1], CANVAS);
  assert.ok(a.cy < 200, `badge centre ${a.cy} should be above the body at 200`);
  assert.equal(a.cx, 100);
});

test("a body at the bottom edge gets its badge above it", () => {
  const anchors = [{ px: SIZE_PX / 2, py: SIZE_PX - EDGE_PX }];
  const [a] = logic.placeBadges(anchors, RADIUS_PX, [SIZES[0]], [0], CANVAS);
  const box = boxOf(a, SIZES[0]);
  assert.ok(box.y + box.h <= anchors[0].py - RADIUS_PX, "the badge lies above the body's top");
  assert.ok(insideCanvas(box), "the badge is inside the canvas");
});

test("a body at the top edge gets its badge below it", () => {
  const anchors = [{ px: SIZE_PX / 2, py: EDGE_PX }];
  const [a] = logic.placeBadges(anchors, RADIUS_PX, [SIZES[0]], [0], CANVAS);
  const box = boxOf(a, SIZES[0]);
  assert.ok(box.y >= anchors[0].py + RADIUS_PX, "the badge lies below the body's bottom");
  assert.ok(insideCanvas(box), "the badge is inside the canvas");
});

test("the evader's badge moves below its body when the chaser's badge is above it", () => {
  // Two bodies stacked 12 px apart: the upper badge would cover the lower one, so the
  // chaser's badge (placed first) stays above and the evader's moves below.
  const anchors = [
    { px: 300, py: 300 },
    { px: 300, py: 312 },
  ];
  const [chaser, evader] = logic.placeBadges(anchors, RADIUS_PX, SIZES, [0, 1], CANVAS);
  assert.ok(chaser.cy < 300, "the chaser's badge is above its body");
  assert.ok(evader.cy > 312, "the evader's badge is below its body");
  assert.equal(overlap(boxOf(chaser, SIZES[0]), boxOf(evader, SIZES[1])), false);
});

test("a badge never leaves the canvas for bodies swept along each edge", () => {
  // Bodies sit within 30 px of each edge, and along the whole edge, including the corners.
  const bodies = [];
  for (let t = EDGE_PX; t <= SIZE_PX - EDGE_PX; t += 23) {
    for (let d = 0; d <= 30; d += 3) {
      bodies.push({ px: t, py: EDGE_PX + d });
      bodies.push({ px: t, py: SIZE_PX - EDGE_PX - d });
      bodies.push({ px: EDGE_PX + d, py: t });
      bodies.push({ px: SIZE_PX - EDGE_PX - d, py: t });
    }
  }
  for (const body of bodies) {
    const [a] = logic.placeBadges([body], RADIUS_PX, [SIZES[0]], [0], CANVAS);
    assert.ok(
      insideCanvas(boxOf(a, SIZES[0])),
      `the badge leaves the canvas for a body at (${body.px}, ${body.py})`,
    );
  }
});

test("badges stay inside the canvas and apart for pairs of bodies swept near each edge", () => {
  const offsets = [-40, -22, -12, 0, 12, 22, 40];
  for (let t = EDGE_PX; t <= SIZE_PX - EDGE_PX; t += 23) {
    for (let d = 0; d <= 30; d += 3) {
      for (const base of [
        { px: t, py: EDGE_PX + d },
        { px: t, py: SIZE_PX - EDGE_PX - d },
        { px: EDGE_PX + d, py: t },
        { px: SIZE_PX - EDGE_PX - d, py: t },
      ]) {
        for (const dx of offsets) {
          for (const dy of offsets) {
            const other = { px: base.px + dx, py: base.py + dy };
            // Both bodies stay inside the arena's fence, as they do in a round.
            const inFence = (p) =>
              p.px >= EDGE_PX && p.px <= SIZE_PX - EDGE_PX && p.py >= EDGE_PX && p.py <= SIZE_PX - EDGE_PX;
            if (!inFence(other)) continue;
            const anchors = [base, other];
            for (const order of [
              [0, 1],
              [1, 0],
            ]) {
              const centres = logic.placeBadges(anchors, RADIUS_PX, SIZES, order, CANVAS);
              const boxes = centres.map((c, i) => boxOf(c, SIZES[i]));
              for (const box of boxes) {
                assert.ok(insideCanvas(box), `badge leaves the canvas for ${JSON.stringify(anchors)}`);
              }
              assert.equal(
                overlap(boxes[0], boxes[1]),
                false,
                `badges overlap for ${JSON.stringify(anchors)}, order ${order}`,
              );
            }
          }
        }
      }
    }
  }
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
        const centres = logic.placeBadges(anchors, RADIUS_PX, SIZES, order, CANVAS);
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

// True when the badge's box touches the disc of radius RADIUS_PX centred on `p`.
function touchesBody(box, p) {
  const nx = Math.min(Math.max(p.px, box.x), box.x + box.w);
  const ny = Math.min(Math.max(p.py, box.y), box.y + box.h);
  return (nx - p.px) ** 2 + (ny - p.py) ** 2 < RADIUS_PX * RADIUS_PX;
}

test("a badge at the bottom wall beside another body never sits on a body", () => {
  // Two bodies side by side near the bottom wall. The lower badge would clamp up onto its own
  // body, so it must go to a clear spot instead, close to its body and off the other badge.
  const anchors = [
    { px: 40.9, py: 581.1 },
    { px: 64.9, py: 581.1 },
  ];
  for (const order of [
    [0, 1],
    [1, 0],
  ]) {
    const centres = logic.placeBadges(anchors, RADIUS_PX, SIZES, order, CANVAS);
    const boxes = centres.map((c, i) => boxOf(c, SIZES[i]));
    boxes.forEach((box, i) => {
      assert.ok(insideCanvas(box), `badge ${i} leaves the canvas for order ${order}`);
      for (const body of anchors) {
        assert.equal(touchesBody(box, body), false, `badge ${i} touches a body for order ${order}`);
      }
      const near = Math.hypot(centres[i].cx - anchors[i].px, centres[i].cy - anchors[i].py);
      assert.ok(near < 4 * RADIUS_PX, `badge ${i} is ${near.toFixed(1)} px from its body`);
    });
    assert.equal(overlap(boxes[0], boxes[1]), false, `badges overlap for order ${order}`);
  }
});

test("no bodies gives no badges", () => {
  assert.deepEqual(logic.placeBadges([], RADIUS_PX, [], [], CANVAS), []);
});
