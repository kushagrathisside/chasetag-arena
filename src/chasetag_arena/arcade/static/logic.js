/*
 * Pure functions for the arcade client. No DOM access, so Node can test them.
 *
 * In the browser this file sets the global `ChasetagLogic`. Under Node, which provides
 * `module`, it exports the same object instead.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module !== null && module.exports) {
    module.exports = api;
  } else {
    root.ChasetagLogic = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const PI = Math.PI;
  const TAU = 2 * Math.PI;

  // Frames are stored as seq order. The newest frame is the one with the largest seq.
  const PHASES = ["running", "ended", "intermission"];
  const ENDED_BY = [null, "tag", "fall", "timeout"];
  const ARROWS = {
    ArrowLeft: "left",
    ArrowRight: "right",
    ArrowUp: "up",
    ArrowDown: "down",
  };
  const KEYBOARD_MAX_SPEED_MPS = 1.5;
  // The server's intermission between rounds, in wall-clock seconds.
  const INTERMISSION_S = 1.5;
  const BADGE_GAP_FRAC = 0.25; // gap between a body and its badge, as a share of the radius
  const BADGE_MIN_GAP_PX = 3;
  const BADGE_EDGE_MARGIN_PX = 4; // the least distance from a role badge to any canvas edge

  function clamp01(t) {
    if (!(t > 0)) return 0; // also catches NaN
    if (t > 1) return 1;
    return t;
  }

  /** Map an angle to (-pi, pi]. */
  function wrapAngle(a) {
    let x = a % TAU; // now in (-TAU, TAU)
    if (x > PI) {
      x -= TAU;
    } else if (x <= -PI) {
      x += TAU;
    }
    return x + 0; // turns -0 into 0
  }

  /** Interpolate from angle a to angle b by t, along the shorter arc. */
  function lerpAngle(a, b, t) {
    return wrapAngle(a + wrapAngle(b - a) * t);
  }

  /**
   * Body positions and headings between two frames, at t in [0, 1].
   *
   * t runs in seq order: 0 is the frame with the lower seq and 1 is the one with the higher
   * seq, whichever order the arguments come in. When the round changed between the two
   * frames there is nothing to blend, so the older positions are held until t reaches 1,
   * then the newer ones are shown.
   */
  function interpolateFrames(prev, next, t) {
    const u = clamp01(t);
    const [older, newer] = prev.seq <= next.seq ? [prev, next] : [next, prev];
    if (older.round !== newer.round) {
      const src = u < 1 ? older : newer;
      return src.bodies.map((b) => ({ seat: b.seat, x: b.x, y: b.y, heading: b.heading }));
    }
    return older.bodies.map((a, i) => {
      const b = newer.bodies[i];
      return {
        seat: a.seat,
        x: a.x + (b.x - a.x) * u,
        y: a.y + (b.y - a.y) * u,
        heading: lerpAngle(a.heading, b.heading, u),
      };
    });
  }

  /**
   * Append a point to a trail and keep only the newest `max` points.
   * Returns a new array; the input is not changed.
   */
  function pushTrail(trail, point, max) {
    if (!(max > 0)) return [];
    const grown = trail.concat([{ x: point.x, y: point.y }]);
    return grown.length > max ? grown.slice(grown.length - max) : grown;
  }

  /**
   * Trails for both bodies. A new round starts empty. `state` is `{round, trails}` or null.
   */
  function advanceTrails(state, frame, max) {
    const base =
      state !== null && state !== undefined && state.round === frame.round
        ? state
        : { round: frame.round, trails: [[], []] };
    const trails = base.trails.map((trail, i) => pushTrail(trail, frame.bodies[i], max));
    return { round: frame.round, trails };
  }

  /** The client-clock time to draw: `lagMs` behind `now`. Both are in milliseconds. */
  function pickRenderTime(now, lagMs) {
    return now - lagMs;
  }

  /**
   * The two buffered frames that bracket time `timeMs`, with the fraction `t` between them.
   *
   * `buffer` holds frames in seq order, and each frame's time is `t_server` in seconds. Before
   * the first frame or after the last one, both ends are the same frame and t is 0, so the
   * picture holds rather than extrapolating.
   */
  function bracketFrames(buffer, timeMs) {
    if (!buffer || buffer.length === 0) return null;
    const first = buffer[0];
    const last = buffer[buffer.length - 1];
    const at = (f) => f.t_server * 1000;
    if (buffer.length === 1 || timeMs <= at(first)) {
      return { prev: first, next: first, t: 0 };
    }
    if (timeMs >= at(last)) {
      return { prev: last, next: last, t: 0 };
    }
    for (let i = 0; i < buffer.length - 1; i += 1) {
      const a = buffer[i];
      const b = buffer[i + 1];
      if (timeMs >= at(a) && timeMs <= at(b)) {
        const span = at(b) - at(a);
        return { prev: a, next: b, t: span > 0 ? (timeMs - at(a)) / span : 1 };
      }
    }
    return { prev: last, next: last, t: 0 };
  }

  /** Exponential backoff from 0.5 s to 8 s. `reset` is called on each good message. */
  class ReconnectPolicy {
    constructor(minMs = 500, maxMs = 8000) {
      this.minMs = minMs;
      this.maxMs = maxMs;
      this.attempts = 0;
    }

    /** The wait before the next attempt, in ms. Each call doubles the wait up to the cap. */
    next() {
      const delay = Math.min(this.maxMs, this.minMs * 2 ** this.attempts);
      this.attempts += 1;
      return delay;
    }

    reset() {
      this.attempts = 0;
    }
  }

  /**
   * The keyboard command for the pressed arrow keys, as [vx, vy] in the world frame, m/s.
   * Up is +y and right is +x. Opposite keys cancel. A diagonal keeps the norm at 1.5 m/s.
   */
  function keyboardCommandFromKeys(pressed) {
    const set = pressed instanceof Set ? pressed : new Set(pressed || []);
    const held = new Set();
    for (const key of set) {
      if (Object.prototype.hasOwnProperty.call(ARROWS, key)) held.add(ARROWS[key]);
    }
    const x = (held.has("right") ? 1 : 0) - (held.has("left") ? 1 : 0);
    const y = (held.has("up") ? 1 : 0) - (held.has("down") ? 1 : 0);
    const scale = x !== 0 && y !== 0 ? KEYBOARD_MAX_SPEED_MPS / Math.SQRT2 : KEYBOARD_MAX_SPEED_MPS;
    return [x * scale + 0, y * scale + 0];
  }

  function isFiniteNumber(v) {
    return typeof v === "number" && Number.isFinite(v);
  }

  /** True when `obj` has the fields the drawing and the status text read. */
  function validateFrame(obj) {
    if (obj === null || typeof obj !== "object") return false;
    if (!isFiniteNumber(obj.seq) || !isFiniteNumber(obj.round)) return false;
    if (!isFiniteNumber(obj.t_server) || !isFiniteNumber(obj.t_event)) return false;
    if (!PHASES.includes(obj.phase)) return false;
    if (!ENDED_BY.includes(obj.ended_by)) return false;
    if (!Array.isArray(obj.bodies) || obj.bodies.length !== 2) return false;
    for (const b of obj.bodies) {
      if (!b || !isFiniteNumber(b.x) || !isFiniteNumber(b.y) || !isFiniteNumber(b.heading)) {
        return false;
      }
    }
    if (obj.chaser !== 0 && obj.chaser !== 1) return false;
    if (!Array.isArray(obj.round_score) || obj.round_score.length !== 2) return false;
    if (!obj.match || !obj.selectors) return false;
    return true;
  }

  function formatSeconds(t) {
    return `${t.toFixed(1)} s`;
  }

  /**
   * The planner option's label and whether it is enabled, from the server's `/planner` answer.
   * The option is enabled only when a planner is loaded and it has a name to show.
   */
  function plannerOptionState(info) {
    const named =
      info !== null &&
      typeof info === "object" &&
      info.loaded === true &&
      typeof info.name === "string" &&
      info.name !== "";
    return named
      ? { disabled: false, text: `planner (${info.name})` }
      : { disabled: true, text: "planner (not loaded)" };
  }

  /**
   * The status lines for a frame. `sinceIntermissionS` is the time since the intermission
   * frame arrived, and only matters in that phase.
   */
  function describePhase(frame, paused, sinceIntermissionS) {
    if (frame.phase === "running") {
      return { headline: paused ? "Paused" : "Running", detail: "" };
    }
    const reason = { tag: "tagged", fall: "fell", timeout: "time out" }[frame.ended_by];
    const headline = reason ? `Round over: ${reason}` : "Round over";
    if (frame.phase === "ended") {
      return { headline, detail: "" };
    }
    const left = Math.max(0, INTERMISSION_S - (sinceIntermissionS || 0));
    return { headline, detail: `Next round in ${left.toFixed(1)} s` };
  }

  const ARENA_MIN_PX = 360; // the arena's side is never fitted below this on a desktop

  /**
   * The arena's side in whole CSS pixels on a window wider than the phone layout. It is the
   * column's width, unless the room left below the canvas's top edge is smaller, in which case
   * it is that room. `topPx` is the canvas's distance from the top of the page.
   */
  function arenaSidePx(columnPx, viewportHeightPx, topPx) {
    const room = Math.max(ARENA_MIN_PX, viewportHeightPx - topPx);
    return Math.max(0, Math.floor(Math.min(columnPx, room)));
  }

  /** The key-to-speed map used by keys 1 to 4. */
  const SPEED_KEYS = { 1: 0.25, 2: 0.5, 3: 1, 4: 2 };

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
   * The badge centre moved, if needed, so the badge lies inside the canvas, which is `canvas.w`
   * by `canvas.h` pixels, with BADGE_EDGE_MARGIN_PX clear of every edge. A centre whose badge
   * is already inside comes back unchanged.
   */
  function clampBadgeCentre(centre, size, canvas) {
    const m = BADGE_EDGE_MARGIN_PX;
    const clampAxis = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
    return {
      cx: clampAxis(centre.cx, m + size.w / 2, canvas.w - m - size.w / 2),
      cy: clampAxis(centre.cy, m + size.h / 2, canvas.h - m - size.h / 2),
    };
  }

  // The fallback ring around a body has this many candidates, evenly spaced.
  const BADGE_RING_STEPS = 12;

  /**
   * Where each role badge goes, as centres in canvas pixels.
   *
   * `anchors` are the body centres, `sizes` the badge sizes, `order` the indices in the order
   * they are placed (so the chaser's badge is placed first), and `canvas` the `{w, h}` size of
   * the canvas. A badge goes above its body if it fits there, otherwise below, then left, then
   * right. Every candidate is clamped inside the canvas, so a badge that would leave one edge
   * slides in from it. A badge pushed onto a body, its own included, is not clear, so it moves
   * to the other side. A candidate is clear when it covers no badge already placed and touches
   * no body. If no side is clear, a ring of positions around the body is tried. If none of
   * those is clear, the first side that covers no placed badge is used. So two badges never
   * overlap while a placement exists.
   */
  function placeBadges(anchors, radiusPx, sizes, order, canvas) {
    const gap = Math.max(BADGE_MIN_GAP_PX, radiusPx * BADGE_GAP_FRAC);
    const placed = new Array(anchors.length);
    const boxes = [];
    for (const i of order) {
      const a = anchors[i];
      const s = sizes[i];
      const sides = [
        { cx: a.px, cy: a.py - radiusPx - gap - s.h / 2 },
        { cx: a.px, cy: a.py + radiusPx + gap + s.h / 2 },
        { cx: a.px - radiusPx - gap - s.w / 2, cy: a.py },
        { cx: a.px + radiusPx + gap + s.w / 2, cy: a.py },
      ].map((c) => clampBadgeCentre(c, s, canvas));
      const far = radiusPx + gap + Math.hypot(s.w, s.h) / 2;
      const ring = Array.from({ length: BADGE_RING_STEPS }, (_, k) => {
        const th = (k / BADGE_RING_STEPS) * TAU;
        const c = { cx: a.px + far * Math.cos(th), cy: a.py - far * Math.sin(th) };
        return clampBadgeCentre(c, s, canvas);
      });
      const clearOfBadges = (c) => boxes.every((b) => !boxesOverlap(boxAt(c, s), b));
      const clear = (c) =>
        clearOfBadges(c) && anchors.every((o) => !boxTouchesDisc(boxAt(c, s), o, radiusPx));
      const choice =
        sides.find(clear) ||
        ring.find(clear) ||
        sides.find(clearOfBadges) ||
        ring.find(clearOfBadges) ||
        sides[0];
      placed[i] = choice;
      boxes.push(boxAt(choice, s));
    }
    return placed;
  }

  return {
    boxAt,
    clampBadgeCentre,
    placeBadges,
    BADGE_EDGE_MARGIN_PX,
    wrapAngle,
    lerpAngle,
    interpolateFrames,
    pushTrail,
    advanceTrails,
    pickRenderTime,
    bracketFrames,
    ReconnectPolicy,
    keyboardCommandFromKeys,
    validateFrame,
    formatSeconds,
    describePhase,
    plannerOptionState,
    arenaSidePx,
    ARENA_MIN_PX,
    SPEED_KEYS,
    ARROWS,
  };
});
