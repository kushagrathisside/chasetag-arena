/*
 * Drawing functions for the arena canvas. They take a 2D context and plain data, and they
 * hold no game logic. The arena is a square of side 2 * half metres, centred on the origin,
 * with +y up on the world side and down on the canvas.
 *
 * In the browser this file sets the global `ChasetagRender`. Under Node it exports the same
 * object.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module !== null && module.exports) {
    module.exports = api;
  } else {
    root.ChasetagRender = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const TAU = 2 * Math.PI;
  const BODY_RADIUS_FRAC = 0.035; // body radius as a share of the arena width
  const TICK_PER_RADIUS = 2.4; // heading tick length as a multiple of the body radius
  const CORNER_LENGTH_M = 0.7;
  const BADGE_GAP_FRAC = 0.25; // gap between a body and its badge, as a share of the radius
  const BADGE_MIN_GAP_PX = 3;
  const BADGE_FONT_FRAC = 0.7; // badge text size, as a share of the body radius
  const BADGE_PAD_FRAC = 0.3; // badge padding on each side of the text, as a share of the radius
  const BADGE_FONT_MIN_PX = 10; // the smallest text size the badge is drawn at
  const FLASH_MS = 600;
  const FLASH_MAX_RADIUS_M = 1.1;
  const GRID_STEP_M = 1;
  const CLOCK_INSET_PX = 14; // the clock's distance inside the fence lines

  // Names of the custom properties the drawing reads. Every colour comes from the page CSS.
  const PALETTE_NAMES = {
    arenaBg: "--arena-bg",
    grid: "--grid",
    fence: "--fence",
    seatA: "--seat-a",
    seatB: "--seat-b",
    seatText: "--seat-text",
    badgeBg: "--badge-bg",
    badgeText: "--badge-text",
    flash: "--flash",
    text: "--arena-text",
  };

  /**
   * A palette from a function that returns the value of a custom property by name. Missing
   * values fall back to a neutral grey, so the drawing still works without CSS.
   */
  function readPalette(getProperty) {
    const palette = {};
    for (const [key, name] of Object.entries(PALETTE_NAMES)) {
      const value = getProperty(name);
      palette[key] = value && value.trim() !== "" ? value.trim() : "#808080";
    }
    return palette;
  }

  /**
   * The mapping from world metres to canvas pixels for a square canvas of `sizePx` CSS
   * pixels. The fence is inset by a margin so its markers are not cut off.
   */
  function makeView(sizePx, half) {
    const margin = Math.max(8, sizePx * 0.03);
    const scale = (sizePx - 2 * margin) / (2 * half);
    return {
      sizePx,
      half,
      margin,
      scale,
      toCanvas(x, y) {
        return {
          px: margin + (x + half) * scale,
          py: margin + (half - y) * scale,
        };
      },
    };
  }

  function seatColour(palette, seat) {
    return seat === 0 ? palette.seatA : palette.seatB;
  }

  function seatLetter(seat) {
    return seat === 0 ? "A" : "B";
  }

  function drawBackground(ctx, view, palette) {
    ctx.fillStyle = palette.arenaBg;
    ctx.fillRect(0, 0, view.sizePx, view.sizePx);
  }

  function drawGrid(ctx, view, palette) {
    ctx.strokeStyle = palette.grid;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.55;
    ctx.beginPath();
    for (let m = -view.half + GRID_STEP_M; m < view.half; m += GRID_STEP_M) {
      const a = view.toCanvas(m, view.half);
      const b = view.toCanvas(m, -view.half);
      ctx.moveTo(a.px, a.py);
      ctx.lineTo(b.px, b.py);
      const c = view.toCanvas(-view.half, m);
      const d = view.toCanvas(view.half, m);
      ctx.moveTo(c.px, c.py);
      ctx.lineTo(d.px, d.py);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawFence(ctx, view, palette) {
    const h = view.half;
    const tl = view.toCanvas(-h, h);
    const br = view.toCanvas(h, -h);
    ctx.strokeStyle = palette.fence;
    ctx.lineWidth = 2;
    ctx.strokeRect(tl.px, tl.py, br.px - tl.px, br.py - tl.py);

    // Corner markers: a short L pointing into the arena at each corner.
    const k = CORNER_LENGTH_M;
    ctx.lineWidth = 3;
    const corners = [
      [-h, h, 1, -1],
      [h, h, -1, -1],
      [h, -h, -1, 1],
      [-h, -h, 1, 1],
    ];
    ctx.beginPath();
    for (const [cx, cy, dx, dy] of corners) {
      const p = view.toCanvas(cx, cy);
      const along = view.toCanvas(cx + dx * k, cy);
      const across = view.toCanvas(cx, cy + dy * k);
      ctx.moveTo(along.px, along.py);
      ctx.lineTo(p.px, p.py);
      ctx.lineTo(across.px, across.py);
    }
    ctx.stroke();
  }

  function drawTrails(ctx, view, trails, palette) {
    trails.forEach((trail, seat) => {
      if (trail.length < 2) return;
      ctx.strokeStyle = seatColour(palette, seat);
      ctx.lineWidth = 3;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      for (let i = 1; i < trail.length; i += 1) {
        // Older segments are fainter: the alpha rises from near 0 to 0.6 at the newest.
        ctx.globalAlpha = 0.6 * (i / (trail.length - 1));
        const a = view.toCanvas(trail[i - 1].x, trail[i - 1].y);
        const b = view.toCanvas(trail[i].x, trail[i].y);
        ctx.beginPath();
        ctx.moveTo(a.px, a.py);
        ctx.lineTo(b.px, b.py);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    });
  }

  function bodyRadiusPx(sizePx) {
    return BODY_RADIUS_FRAC * sizePx;
  }

  // The badge's text size and side padding grow with the body radius, so a badge keeps its
  // proportion to the body at any arena size. The text never goes below BADGE_FONT_MIN_PX.
  function badgeMetrics(radiusPx) {
    return {
      fontPx: Math.max(BADGE_FONT_MIN_PX, Math.round(radiusPx * BADGE_FONT_FRAC)),
      padPx: radiusPx * BADGE_PAD_FRAC,
    };
  }

  function drawBody(ctx, view, palette, body, c, r) {
    const colour = seatColour(palette, body.seat);
    ctx.save();
    ctx.globalAlpha = body.fallen ? 0.45 : 1;
    ctx.beginPath();
    ctx.arc(c.px, c.py, r, 0, TAU);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = palette.fence;
    if (body.fallen) {
      ctx.setLineDash([4, 3]);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Heading tick: world direction (cos h, sin h), drawn with the canvas y axis flipped.
    const tickPx = TICK_PER_RADIUS * r;
    const tipX = c.px + Math.cos(body.heading) * tickPx;
    const tipY = c.py - Math.sin(body.heading) * tickPx;
    ctx.beginPath();
    ctx.moveTo(c.px, c.py);
    ctx.lineTo(tipX, tipY);
    ctx.lineWidth = 3;
    ctx.strokeStyle = palette.fence;
    ctx.stroke();

    // Seat letter, so the body is identifiable without its colour.
    ctx.fillStyle = palette.seatText;
    ctx.font = `600 ${Math.round(r * 1.1)}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(seatLetter(body.seat), c.px, c.py + 1);
    ctx.restore();
  }

  // The size of a role badge: its text plus the metrics' padding on each side.
  function badgeSize(ctx, text, metrics) {
    ctx.font = `700 ${metrics.fontPx}px system-ui, sans-serif`;
    return {
      w: ctx.measureText(text).width + 2 * metrics.padPx,
      h: metrics.fontPx * 1.6,
    };
  }

  function boxAt(centre, size) {
    return { x: centre.cx - size.w / 2, y: centre.cy - size.h / 2, w: size.w, h: size.h };
  }

  function boxesOverlap(a, b) {
    return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  }

  // True when a disc of radius `r` centred at `p` touches the box.
  function boxTouchesDisc(box, p, r) {
    const nx = Math.min(Math.max(p.px, box.x), box.x + box.w);
    const ny = Math.min(Math.max(p.py, box.y), box.y + box.h);
    return (nx - p.px) ** 2 + (ny - p.py) ** 2 < r * r;
  }

  /**
   * Where each role badge goes, as centres in canvas pixels.
   *
   * `anchors` are the body centres, `sizes` the badge sizes, and `order` the indices in the
   * order they are placed, so the chaser's badge is placed first. A badge goes above its body
   * if it fits there, otherwise below, then left, then right. A candidate is clear when it
   * covers no badge already placed and no other body. If no candidate is clear, the first one
   * that covers no placed badge is used, so two badges never overlap while a placement exists.
   */
  function placeBadges(anchors, radiusPx, sizes, order) {
    const gap = Math.max(BADGE_MIN_GAP_PX, radiusPx * BADGE_GAP_FRAC);
    const placed = new Array(anchors.length);
    const boxes = [];
    for (const i of order) {
      const a = anchors[i];
      const s = sizes[i];
      const candidates = [
        { cx: a.px, cy: a.py - radiusPx - gap - s.h / 2 },
        { cx: a.px, cy: a.py + radiusPx + gap + s.h / 2 },
        { cx: a.px - radiusPx - gap - s.w / 2, cy: a.py },
        { cx: a.px + radiusPx + gap + s.w / 2, cy: a.py },
      ];
      const clearOfBadges = (c) => boxes.every((b) => !boxesOverlap(boxAt(c, s), b));
      const clear = (c) =>
        clearOfBadges(c) &&
        anchors.every((o, j) => j === i || !boxTouchesDisc(boxAt(c, s), o, radiusPx));
      const choice =
        candidates.find(clear) || candidates.find(clearOfBadges) || candidates[0];
      placed[i] = choice;
      boxes.push(boxAt(choice, s));
    }
    return placed;
  }

  function drawBadge(ctx, palette, centre, text, size, fontPx) {
    const box = boxAt(centre, size);
    const radius = box.h / 2;
    const { x, y, w: width, h: height } = box;
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
    ctx.fillStyle = palette.badgeBg;
    ctx.fill();
    ctx.font = `700 ${fontPx}px system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = palette.badgeText;
    ctx.fillText(text, centre.cx, centre.cy + 1);
  }

  /**
   * The flash on a tag or a fall. `flash.age` runs from 0 to 1 over FLASH_MS. With reduced
   * motion the ring is drawn at its final size and does not grow.
   */
  function drawFlash(ctx, view, palette, flash) {
    if (!flash) return;
    const c = view.toCanvas(flash.x, flash.y);
    const fade = 1 - flash.age;
    const radiusM = flash.animate
      ? 0.25 + (FLASH_MAX_RADIUS_M - 0.25) * flash.age
      : FLASH_MAX_RADIUS_M;
    ctx.save();
    ctx.globalAlpha = flash.animate ? fade : 0.9;
    ctx.beginPath();
    ctx.arc(c.px, c.py, radiusM * view.scale, 0, TAU);
    ctx.lineWidth = 4;
    ctx.strokeStyle = palette.flash;
    ctx.stroke();
    ctx.restore();
  }

  // The clock sits inside the fence, CLOCK_INSET_PX from both fence lines, so it never
  // touches the fence or its corner markers.
  function drawClock(ctx, view, palette, clockS) {
    ctx.fillStyle = palette.text;
    ctx.font = `600 ${Math.max(12, Math.round(view.sizePx * 0.035))}px system-ui, sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const inset = view.margin + CLOCK_INSET_PX;
    ctx.fillText(`${clockS.toFixed(1)} s`, inset, inset);
  }

  /**
   * Draw the whole arena.
   *
   * `scene` holds: `bodies` (seat, x, y, heading, fallen), `chaser` (0 or 1), `trails` (two
   * lists of {x, y}), `flash` (null, or {x, y, age, animate}), and `clockS`.
   * `ctx` must already be scaled so that one unit is one CSS pixel.
   */
  function drawArena(ctx, view, palette, scene) {
    drawBackground(ctx, view, palette);
    drawGrid(ctx, view, palette);
    drawFence(ctx, view, palette);
    drawTrails(ctx, view, scene.trails, palette);

    const r = bodyRadiusPx(view.sizePx);
    const anchors = scene.bodies.map((b) => view.toCanvas(b.x, b.y));
    const roles = scene.bodies.map((b) => (b.seat === scene.chaser ? "CHASE" : "EVADE"));
    const metrics = badgeMetrics(r);
    const sizes = roles.map((role) => badgeSize(ctx, role, metrics));
    // The chaser's badge is placed first, so when two badges clash the evader's moves.
    const order = scene.bodies.map((_, i) => i).sort((i, j) => (roles[j] === "CHASE") - (roles[i] === "CHASE"));
    const badges = placeBadges(anchors, r, sizes, order);

    scene.bodies.forEach((body, i) => drawBody(ctx, view, palette, body, anchors[i], r));
    scene.bodies.forEach((_, i) =>
      drawBadge(ctx, palette, badges[i], roles[i], sizes[i], metrics.fontPx),
    );
    drawFlash(ctx, view, palette, scene.flash);
    drawClock(ctx, view, palette, scene.clockS);
  }

  return {
    readPalette,
    makeView,
    drawArena,
    bodyRadiusPx,
    badgeMetrics,
    badgeSize,
    placeBadges,
    FLASH_MS,
    PALETTE_NAMES,
  };
});
