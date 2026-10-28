// Capture the arcade page for the documentation. Node 22, no dependencies.
//
// Usage (with the server already running):
//   node scripts/capture_screenshots.mjs --url http://127.0.0.1:8765/ --out docs/assets/screenshots
//
// Options:
//   --url URL       page to capture (default http://127.0.0.1:8765/)
//   --out DIR       folder for the PNG files (default docs/assets/screenshots)
//   --chrome PATH   Chrome or Chromium executable (default /usr/bin/google-chrome)
//   --port N        DevTools port for the headless browser (default 9333)
//
// It writes desktop-light.png, desktop-dark.png, phone-light.png and round-running.png.
// The round-running capture waits until the page shows a running round, then waits a few
// more seconds, so the arena is mid-round in the picture.

import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";

function parseArgs(argv) {
  const opts = {
    url: "http://127.0.0.1:8765/",
    out: "docs/assets/screenshots",
    chrome: "/usr/bin/google-chrome",
    port: 9333,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    const value = () => {
      if (i + 1 >= argv.length) throw new Error(`${flag} needs a value`);
      i += 1;
      return argv[i];
    };
    if (flag === "--url") opts.url = value();
    else if (flag === "--out") opts.out = value();
    else if (flag === "--chrome") opts.chrome = value();
    else if (flag === "--port") opts.port = Number(value());
    else throw new Error(`unknown option: ${flag}`);
  }
  if (!Number.isInteger(opts.port) || opts.port <= 0) throw new Error("--port must be a positive integer");
  return opts;
}

const opts = parseArgs(process.argv.slice(2));
const OUT = resolve(opts.out);
const PROFILE = mkdtempSync(join(tmpdir(), "chasetag-shots-"));

// Each case sets the viewport, the colour scheme and the file name.
const CASES = [
  { name: "desktop-light", width: 1280, height: 900, mobile: false, scheme: "light" },
  { name: "desktop-dark", width: 1280, height: 900, mobile: false, scheme: "dark" },
  { name: "phone-light", width: 390, height: 844, mobile: true, scheme: "light" },
];
const RUNNING_CASE = { name: "round-running", width: 1280, height: 900, mobile: false, scheme: "light" };

mkdirSync(OUT, { recursive: true });
mkdirSync(PROFILE, { recursive: true });

const chrome = spawn(
  opts.chrome,
  [
    `--remote-debugging-port=${opts.port}`,
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--proxy-server=direct://",
    "--proxy-bypass-list=*",
    "--remote-allow-origins=*",
    `--user-data-dir=${PROFILE}`,
    "--no-first-run",
    "--disable-extensions",
    "about:blank",
  ],
  { stdio: "ignore" },
);
const exited = new Promise((resolveExit) => chrome.on("exit", resolveExit));

async function findPageTarget() {
  for (let i = 0; i < 150; i += 1) {
    try {
      const res = await fetch(`http://127.0.0.1:${opts.port}/json`);
      const targets = await res.json();
      const page = targets.find((t) => t.type === "page");
      if (page) return page;
    } catch {
      // Chrome is still starting. Retry.
    }
    await delay(100);
  }
  throw new Error("Chrome did not open its debugging port");
}

function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let nextId = 0;
  const pending = new Map();
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id !== undefined && pending.has(msg.id)) {
      const { resolveReply, rejectReply } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) rejectReply(new Error(msg.error.message));
      else resolveReply(msg.result);
    }
  };
  const opened = new Promise((resolveOpen, rejectOpen) => {
    ws.onopen = resolveOpen;
    ws.onerror = () => rejectOpen(new Error("could not open the DevTools WebSocket"));
  });
  return {
    opened,
    send(method, params = {}) {
      nextId += 1;
      const id = nextId;
      ws.send(JSON.stringify({ id, method, params }));
      return new Promise((resolveReply, rejectReply) => pending.set(id, { resolveReply, rejectReply }));
    },
    close() {
      ws.close();
    },
  };
}

async function evaluate(cdp, expression) {
  const res = await cdp.send("Runtime.evaluate", { expression, returnByValue: true });
  return res.result.value;
}

async function waitForLoad(cdp) {
  for (let i = 0; i < 100; i += 1) {
    if ((await evaluate(cdp, "document.readyState")) === "complete") return;
    await delay(100);
  }
  throw new Error("page did not finish loading");
}

// The page's status and the round clock, read from the DOM. The canvas label carries the clock.
async function readRound(cdp) {
  const raw = await evaluate(
    cdp,
    `JSON.stringify({
      phase: document.getElementById("phase")?.textContent ?? "",
      label: document.getElementById("arena")?.getAttribute("aria-label") ?? "",
    })`,
  );
  const { phase, label } = JSON.parse(raw);
  const clock = /clock ([0-9.]+) s/.exec(label);
  const round = /Round (\d+)/.exec(label);
  return {
    phase,
    round: round ? Number(round[1]) : -1,
    clockS: clock ? Number(clock[1]) : -1,
  };
}

// Wait until a round is running and its clock has reached minClockS. A round that ends
// during the wait is not used, so the picture never shows a tag or the intermission.
async function waitForRunning(cdp, minClockS) {
  for (let i = 0; i < 600; i += 1) {
    const r = await readRound(cdp);
    if (r.phase === "Running" && r.clockS >= minClockS) return r;
    await delay(100);
  }
  throw new Error(`the page never showed a running round past ${minClockS} s`);
}

// Capture a running round. The shot is kept only if the same round is still running when
// the capture ends, so a round that finished mid-capture is taken again.
async function captureRunning(cdp, file, minClockS) {
  for (let attempt = 0; attempt < 10; attempt += 1) {
    const before = await waitForRunning(cdp, minClockS);
    const data = await shoot(cdp, file);
    const after = await readRound(cdp);
    if (after.phase === "Running" && after.round === before.round) return data;
  }
  throw new Error(`could not capture a clean running round for ${file}`);
}

async function applyCase(cdp, c) {
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: c.width,
    height: c.height,
    deviceScaleFactor: 1,
    mobile: c.mobile,
  });
  await cdp.send("Emulation.setEmulatedMedia", {
    media: "screen",
    features: [{ name: "prefers-color-scheme", value: c.scheme }],
  });
}

async function shoot(cdp, file) {
  const shot = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
  writeFileSync(join(OUT, file), Buffer.from(shot.data, "base64"));
  return join(OUT, file);
}

const written = [];
let cdp = null;
try {
  const target = await findPageTarget();
  cdp = connect(target.webSocketDebuggerUrl);
  await cdp.opened;
  await cdp.send("Runtime.enable");

  for (const c of CASES) {
    await applyCase(cdp, c);
    await cdp.send("Page.navigate", { url: opts.url });
    await waitForLoad(cdp);
    await delay(1000); // let the stream arrive and the arena draw
    written.push(await captureRunning(cdp, `${c.name}.png`, 1.5));
  }

  // The round capture: the same desktop size, taken a few seconds into a running round.
  await applyCase(cdp, RUNNING_CASE);
  await cdp.send("Page.navigate", { url: opts.url });
  await waitForLoad(cdp);
  written.push(await captureRunning(cdp, `${RUNNING_CASE.name}.png`, 3.0));

  for (const file of written) console.log(file);
} finally {
  if (cdp) cdp.close();
  chrome.kill("SIGTERM");
  await Promise.race([exited, delay(5000)]);
  rmSync(PROFILE, { recursive: true, force: true });
}
