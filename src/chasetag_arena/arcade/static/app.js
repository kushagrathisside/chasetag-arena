/*
 * The arcade page: the event stream, the controls, the keyboard, the playback clock and the
 * status text. Drawing is in render.js and the testable rules are in logic.js.
 */
(function () {
  "use strict";

  const L = globalThis.ChasetagLogic;
  const R = globalThis.ChasetagRender;

  const ARENA_HALF_M = 6; // the fence spans -6 to 6 m on both axes
  const BUFFER_SIZE = 5; // frames kept for interpolation
  const LAG_MS = 100; // draw this far behind the newest frame
  const TRAIL_LEN = 30; // decisions kept in each body's trail
  const KEY_SEND_MS = 100; // keyboard command rate while a key is held (about 10 Hz)
  const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

  const $ = (id) => document.getElementById(id);
  const els = {
    connection: $("connection"),
    connectionText: $("connection-text"),
    helpToggle: $("help-toggle"),
    help: $("help"),
    selects: [$("seat0-select"), $("seat1-select")],
    pending: [$("seat0-pending"), $("seat1-pending")],
    bodyNote: $("body-note"),
    playToggle: $("play-toggle"),
    step: $("step"),
    restart: $("restart"),
    speed: $("speed"),
    timeLimit: $("time-limit"),
    limitPending: $("limit-pending"),
    canvas: $("arena"),
    roundTitle: $("round-title"),
    phase: $("phase"),
    phaseDetail: $("phase-detail"),
    roundScoreA: $("round-score-a"),
    roundScoreB: $("round-score-b"),
    matchA: $("match-a"),
    matchB: $("match-b"),
    matchTags: $("match-tags"),
    matchFalls: $("match-falls"),
    matchTimeouts: $("match-timeouts"),
  };
  const ctx = els.canvas.getContext("2d");
  const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
  const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function freshSession() {
    // offset maps this page's clock to the server's clock, in ms. It is per connection,
    // because a restarted server counts its clock and its seq from zero again.
    return { offset: null, lastSeq: -1, buffer: [] };
  }

  const app = {
    es: null,
    retryTimer: null,
    policy: new L.ReconnectPolicy(500, 8000),
    connection: "reconnecting",
    session: freshSession(),
    latest: null,
    echo: null, // the selectors the server last reported, as [seat0, seat1]
    pending: [null, null], // selectors sent but not yet echoed
    pendingLimit: null, // a round limit sent but not yet started by a round
    paused: false, // the server's pause, from the newest frame; this page's Pause, Play or Step sets it first
    trails: null,
    flash: null,
    intermissionSince: null, // page time when the intermission frame arrived
    palette: null,
    pressed: new Set(),
    keyTimer: null,
    canvasCss: -1,
    canvasDpr: -1,
    lastPhaseKey: "",
    lastLabel: "",
  };

  // --- connection ----------------------------------------------------------------------

  function setConnection(state) {
    app.connection = state;
    els.connection.dataset.state = state;
    const text = { connected: "Connected", reconnecting: "Reconnecting", offline: "Offline" };
    els.connectionText.textContent = text[state];
  }

  // The planner option is enabled only while the server has a planner loaded. The answer is
  // read on each connection, so a server restarted with another planner shows at once.
  function applyPlannerInfo(info) {
    const state = L.plannerOptionState(info);
    for (const select of els.selects) {
      const option = select.querySelector('option[value="planner"]');
      option.disabled = state.disabled;
      option.textContent = state.text;
    }
    if (app.echo) reconcileSelectors(app.echo);
  }

  function loadPlannerInfo() {
    fetch("/planner", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : null))
      .then(applyPlannerInfo)
      .catch(() => applyPlannerInfo(null));
  }

  function connect() {
    app.retryTimer = null;
    app.session = freshSession();
    app.trails = null;
    app.flash = null;
    loadPlannerInfo();
    const es = new EventSource("/events");
    app.es = es;
    es.onopen = () => {
      if (app.es === es) setConnection("connected");
    };
    es.addEventListener("snapshot", (e) => onEvent(es, e, true));
    es.addEventListener("frame", (e) => onEvent(es, e, false));
    es.onerror = () => {
      if (app.es !== es) return;
      es.close();
      app.es = null;
      setConnection(navigator.onLine === false ? "offline" : "reconnecting");
      app.retryTimer = setTimeout(connect, app.policy.next());
    };
  }

  function onEvent(es, event, isSnapshot) {
    if (app.es !== es) return;
    let frame;
    try {
      frame = JSON.parse(event.data);
    } catch (err) {
      console.warn("chasetag-arena: ignored a malformed event", err);
      return;
    }
    app.policy.reset();
    setConnection("connected");
    if (frame === null) return; // the snapshot before the game's first frame
    if (!L.validateFrame(frame)) {
      console.warn("chasetag-arena: ignored a frame with an unexpected shape", frame);
      return;
    }
    if (frame.seq <= app.session.lastSeq) return; // a duplicate from the snapshot overlap
    app.session.lastSeq = frame.seq;
    ingest(frame, isSnapshot);
  }

  // --- frames --------------------------------------------------------------------------

  function ingest(frame, isSnapshot) {
    const now = performance.now();
    const session = app.session;
    const sample = now - frame.t_server * 1000;
    session.offset = session.offset === null ? sample : Math.min(session.offset, sample);
    session.buffer.push(frame);
    if (session.buffer.length > BUFFER_SIZE) session.buffer.shift();

    const prev = app.latest;
    app.latest = frame;
    app.paused = frame.paused;
    reconcileSpeed(frame.speed);
    reconcileLimit(frame.limit_s);

    if (frame.phase === "intermission") {
      const started = !prev || prev.round !== frame.round || prev.phase !== "intermission";
      if (started) app.intermissionSince = now;
    } else {
      app.intermissionSince = null;
    }

    // A tag or a fall flashes once, when its terminal frame arrives. A snapshot does not
    // replay an old flash.
    if (!isSnapshot && frame.phase === "ended" && (frame.ended_by === "tag" || frame.ended_by === "fall")) {
      app.flash = makeFlash(frame, now);
    }

    // The intermission frame repeats the terminal positions, so it adds no trail point. A
    // pause or speed change repeats the current frame, which adds none either.
    if (frame.phase !== "intermission" && !L.isStatusRepeat(prev, frame)) {
      app.trails = L.advanceTrails(app.trails, frame, TRAIL_LEN);
    }

    reconcileSelectors([frame.selectors.seat0, frame.selectors.seat1]);
    els.bodyNote.textContent = `Body: ${frame.selectors.body}`;
    updateHud(frame);
  }

  function makeFlash(frame, now) {
    let x;
    let y;
    const fallen = frame.bodies.find((b) => b.fallen);
    if (frame.ended_by === "fall" && fallen) {
      x = fallen.x;
      y = fallen.y;
    } else {
      x = (frame.bodies[0].x + frame.bodies[1].x) / 2;
      y = (frame.bodies[0].y + frame.bodies[1].y) / 2;
    }
    return { x, y, startedMs: now };
  }

  function reconcileSelectors(echo) {
    app.echo = echo;
    for (const seat of [0, 1]) {
      if (app.pending[seat] !== null && app.pending[seat] === echo[seat]) {
        app.pending[seat] = null;
      }
      const shown = app.pending[seat] !== null ? app.pending[seat] : echo[seat];
      if (els.selects[seat].value !== shown) els.selects[seat].value = shown;
      els.pending[seat].hidden = app.pending[seat] === null;
    }
  }

  // The select shows the server's speed. A speed with no option in the select is left as it is.
  function reconcileSpeed(speed) {
    const value = String(speed);
    const known = Array.from(els.speed.options).some((option) => option.value === value);
    if (known && els.speed.value !== value) els.speed.value = value;
  }

  // The round limit waits for the next round, like a selector. The select keeps the limit the
  // viewer chose until a round starts with it, or the server refuses it.
  function reconcileLimit(limit) {
    if (app.pendingLimit !== null) {
      if (app.pendingLimit !== limit) return;
      app.pendingLimit = null;
      els.limitPending.hidden = true;
    }
    const value = String(limit);
    const known = Array.from(els.timeLimit.options).some((option) => option.value === value);
    if (known && els.timeLimit.value !== value) els.timeLimit.value = value;
  }

  function onLimit(value) {
    const limit = Number(value);
    app.pendingLimit = limit;
    els.limitPending.hidden = false;
    send({ kind: "limit", value: limit }).then((ok) => {
      // A limit the server refused would stay pending for a round that never starts with it.
      if (ok || app.pendingLimit !== limit) return;
      app.pendingLimit = null;
      els.limitPending.hidden = true;
      if (app.latest) reconcileLimit(app.latest.limit_s);
    });
  }

  function formatScore(v) {
    return v.toFixed(3);
  }

  function updateHud(frame) {
    els.roundTitle.textContent = `Round ${frame.round}`;
    els.roundScoreA.textContent = formatScore(frame.round_score[0]);
    els.roundScoreB.textContent = formatScore(frame.round_score[1]);
    els.matchA.textContent = formatScore(frame.match.seat0);
    els.matchB.textContent = formatScore(frame.match.seat1);
    els.matchTags.textContent = String(frame.match.tags);
    els.matchFalls.textContent = String(frame.match.falls);
    els.matchTimeouts.textContent = String(frame.match.timeouts);
    const clock = L.formatSeconds(frame.t_event);
    app.lastLabel =
      `Round ${frame.round}, clock ${clock}, round score seat A ${formatScore(frame.round_score[0])}, ` +
      `seat B ${formatScore(frame.round_score[1])}. Match tags ${frame.match.tags}, ` +
      `falls ${frame.match.falls}, time-outs ${frame.match.timeouts}.`;
    els.canvas.setAttribute("aria-label", app.lastLabel);
    renderPhase();
  }

  // The button and the status text show the server's pause, so a pause sent from another page
  // shows here too, and a page that connects to a paused game shows it at once.
  function renderPhase() {
    const frame = app.latest;
    if (!frame) return;
    const now = performance.now();
    const since = app.intermissionSince === null ? 0 : (now - app.intermissionSince) / 1000;
    const label = frame.paused ? "Play" : "Pause";
    if (els.playToggle.textContent !== label) els.playToggle.textContent = label;
    const text = L.describePhase(frame, since);
    const key = `${text.headline}|${text.detail}`;
    if (key === app.lastPhaseKey) return;
    app.lastPhaseKey = key;
    els.phase.textContent = text.headline;
    els.phaseDetail.textContent = text.detail;
  }

  // --- controls ------------------------------------------------------------------------

  // Resolves to whether the server accepted the control.
  function send(body) {
    return fetch("/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      keepalive: true,
    })
      .then((res) => {
        if (!res.ok) console.warn("chasetag-arena: control refused", res.status, body);
        return res.ok;
      })
      .catch(() => {
        // The page is offline or the server is restarting. The next command retries.
        return false;
      });
  }

  function onSelect(seat, value) {
    const echoed = app.echo ? app.echo[seat] : null;
    app.pending[seat] = value === echoed ? null : value;
    els.pending[seat].hidden = app.pending[seat] === null;
    send({ kind: "selector", seat, value }).then((ok) => {
      // A change the server never took would stay pending for a round that never starts it.
      if (ok || app.pending[seat] !== value) return;
      app.pending[seat] = null;
      els.pending[seat].hidden = true;
      if (app.echo) reconcileSelectors(app.echo);
    });
  }

  // The page's target only: the display waits for the server's next frame to show the pause.
  // Two quick presses then toggle twice, even before the first frame has arrived.
  function setPaused(paused) {
    app.paused = paused;
    send({ kind: paused ? "pause" : "play" });
  }

  function togglePlay() {
    setPaused(!app.paused);
  }

  function stepOnce() {
    if (!app.paused) setPaused(true);
    send({ kind: "step" });
  }

  function restart() {
    send({ kind: "restart" });
  }

  function setSpeed(value) {
    els.speed.value = String(value);
    send({ kind: "speed", value: Number(value) });
  }

  // --- keyboard ------------------------------------------------------------------------

  // The keyboard seat is the one whose selector is keyboard, seat A first. Pending changes
  // count, because a command sent now is held until the round starts.
  function keyboardSeat() {
    for (const seat of [0, 1]) {
      const wanted = app.pending[seat] !== null ? app.pending[seat] : app.echo ? app.echo[seat] : null;
      if (wanted === "keyboard") return seat;
    }
    return null;
  }

  function sendKeyboard() {
    const seat = keyboardSeat();
    if (seat === null) return;
    send({ kind: "keyboard", seat, cmd: L.keyboardCommandFromKeys(app.pressed) });
  }

  function startKeyTimer() {
    if (app.keyTimer === null) {
      app.keyTimer = setInterval(sendKeyboard, KEY_SEND_MS);
    }
  }

  function stopKeyTimer() {
    if (app.keyTimer !== null) {
      clearInterval(app.keyTimer);
      app.keyTimer = null;
    }
  }

  function onKeyDown(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = e.target && e.target.tagName;
    if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
    if (hasOwn(L.ARROWS, e.key)) {
      e.preventDefault();
      app.pressed.add(e.key);
      startKeyTimer();
      sendKeyboard();
      return;
    }
    if (e.repeat) return;
    // Space and period belong to a focused button when one has focus, so the browser's
    // own click is the only action.
    if (tag === "BUTTON") return;
    if (e.key === " ") {
      e.preventDefault();
      togglePlay();
    } else if (e.key === ".") {
      e.preventDefault();
      stepOnce();
    } else if (hasOwn(L.SPEED_KEYS, e.key)) {
      setSpeed(L.SPEED_KEYS[e.key]);
    }
  }

  function onKeyUp(e) {
    if (!hasOwn(L.ARROWS, e.key) || !app.pressed.has(e.key)) return;
    app.pressed.delete(e.key);
    if (app.pressed.size === 0) stopKeyTimer();
    sendKeyboard();
  }

  function releaseKeys() {
    app.pressed.clear();
    stopKeyTimer();
    sendKeyboard(); // an empty command, so the body stops at once
  }

  // --- drawing -------------------------------------------------------------------------

  function refreshPalette() {
    const style = getComputedStyle(document.documentElement);
    app.palette = R.readPalette((name) => style.getPropertyValue(name));
  }

  // Below 600 px the phone layout gives the canvas its column's width. Wider, the canvas is
  // also held to the room left below its top edge, so the whole arena shows without scrolling.
  const phoneQuery = window.matchMedia("(max-width: 599px)");

  // Sets the canvas's CSS width. Its height follows from the square aspect ratio in style.css.
  function fitCanvas() {
    if (phoneQuery.matches) {
      els.canvas.style.width = "";
      return;
    }
    const column = els.canvas.parentElement.clientWidth;
    const top = els.canvas.getBoundingClientRect().top + window.scrollY;
    els.canvas.style.width = `${L.arenaSidePx(column, window.innerHeight, top)}px`;
  }

  function resizeCanvas() {
    fitCanvas();
    const css = Math.round(els.canvas.clientWidth);
    const dpr = window.devicePixelRatio || 1;
    if (css === app.canvasCss && dpr === app.canvasDpr) return;
    app.canvasCss = css;
    app.canvasDpr = dpr;
    const px = Math.max(1, Math.round(css * dpr));
    els.canvas.width = px;
    els.canvas.height = px;
  }

  // Positions at the render time: the last frame drawn is LAG_MS behind the server's now.
  function sceneAt(nowPerf) {
    const session = app.session;
    const trails = app.trails ? app.trails.trails : [[], []];
    if (session.buffer.length === 0 || session.offset === null) {
      return { bodies: [], chaser: 0, trails, flash: null, clockS: 0 };
    }
    const renderMs = L.pickRenderTime(nowPerf - session.offset, LAG_MS);
    const bracket = L.bracketFrames(session.buffer, renderMs);
    const bodies = L.interpolateFrames(bracket.prev, bracket.next, bracket.t);
    const discrete = bracket.t < 0.5 ? bracket.prev : bracket.next;
    const sameRound = bracket.prev.round === bracket.next.round;
    const clockS = sameRound
      ? bracket.prev.t_event + (bracket.next.t_event - bracket.prev.t_event) * bracket.t
      : discrete.t_event;
    return { bodies, chaser: discrete.chaser, trails, flash: flashAt(nowPerf), clockS };
  }

  function flashAt(nowPerf) {
    if (!app.flash) return null;
    const age = (nowPerf - app.flash.startedMs) / R.FLASH_MS;
    if (age >= 1) {
      app.flash = null;
      return null;
    }
    return { x: app.flash.x, y: app.flash.y, age, animate: !motionQuery.matches };
  }

  // Draws the arena at the canvas's current size. The frame loop and a resize both call it.
  function paint() {
    if (app.canvasCss <= 0) return;
    if (app.palette === null) refreshPalette();
    const nowPerf = performance.now();
    const view = R.makeView(app.canvasCss, ARENA_HALF_M);
    ctx.setTransform(app.canvasDpr, 0, 0, app.canvasDpr, 0, 0);
    R.drawArena(ctx, view, app.palette, sceneAt(nowPerf));
  }

  function frameLoop() {
    requestAnimationFrame(frameLoop);
    resizeCanvas();
    paint();
    renderPhase();
  }

  // A change of the column's width or of the window's height resizes the drawing and redraws
  // it at once, without waiting for the next animation frame.
  function watchSize() {
    const redraw = () => {
      resizeCanvas();
      paint();
    };
    window.addEventListener("resize", redraw);
    if (typeof ResizeObserver !== "function") return;
    new ResizeObserver(redraw).observe(els.canvas.parentElement);
  }

  // --- start ---------------------------------------------------------------------------

  function init() {
    setConnection("reconnecting");
    // After a choice, focus leaves the select, so the arrow keys drive the page again.
    els.selects[0].addEventListener("change", (e) => {
      onSelect(0, e.target.value);
      e.target.blur();
    });
    els.selects[1].addEventListener("change", (e) => {
      onSelect(1, e.target.value);
      e.target.blur();
    });
    els.playToggle.addEventListener("click", togglePlay);
    els.step.addEventListener("click", stepOnce);
    els.restart.addEventListener("click", restart);
    els.speed.addEventListener("change", (e) => {
      setSpeed(e.target.value);
      e.target.blur();
    });
    els.timeLimit.addEventListener("change", (e) => {
      onLimit(e.target.value);
      e.target.blur();
    });
    els.helpToggle.addEventListener("click", () => {
      const open = els.help.hidden;
      els.help.hidden = !open;
      els.helpToggle.setAttribute("aria-expanded", String(open));
    });

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", releaseKeys);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "hidden") releaseKeys();
    });
    window.addEventListener("offline", () => {
      if (app.connection !== "connected") setConnection("offline");
    });
    window.addEventListener("online", () => {
      if (app.connection === "offline") setConnection("reconnecting");
    });

    const onScheme = () => {
      refreshPalette();
    };
    if (darkQuery.addEventListener) darkQuery.addEventListener("change", onScheme);
    else darkQuery.addListener(onScheme);

    refreshPalette();
    watchSize();
    connect();
    requestAnimationFrame(frameLoop);
  }

  init();
})();
