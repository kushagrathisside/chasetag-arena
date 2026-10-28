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
