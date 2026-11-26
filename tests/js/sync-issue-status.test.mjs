import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import path from "node:path";

const run = promisify(execFile);
const script = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../scripts/sync_issue_status.mjs");

function fakeGitHub(issues) {
  const calls = [];
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => (body += chunk));
    req.on("end", () => {
      calls.push({ method: req.method, url: req.url, body: body ? JSON.parse(body) : null });
      res.setHeader("Content-Type", "application/json");
      if (req.method === "GET") res.end(JSON.stringify(req.url.includes("page=1") ? issues : []));
      else res.end("[]");
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve({ server, calls, url: `http://127.0.0.1:${server.address().port}` })));
}

const issue = (number, title, extra = {}) => ({ number, title, state: "open", labels: [], assignees: [], html_url: `https://github.com/o/r/issues/${number}`, ...extra });

test("the script relabels the issue that a closed issue unlocks, and nothing else", async () => {
  const fake = await fakeGitHub([
    issue(1, "[1.1] Allow the movement trails to be shown or hidden", { state: "closed", state_reason: "completed", labels: [{ name: "status: ready" }] }),
    issue(2, "[1.2] Allow the length of the movement trails to be chosen", { labels: [{ name: "status: blocked" }] }),
    issue(3, "[1.3] Provide a fullscreen view of the arena", { labels: [{ name: "status: ready" }] }),
  ]);
  try {
    const { stdout } = await run("node", [script], { env: { ...process.env, GITHUB_TOKEN: "t", GITHUB_REPOSITORY: "o/r", GITHUB_API_URL: fake.url } });
    assert.match(stdout, /#2 \(1\.2\)/);
    const writes = fake.calls.filter((c) => c.method !== "GET");
    assert.deepEqual(writes.map((c) => [c.method, c.url]), [
      ["POST", "/repos/o/r/issues/2/labels"],
      ["DELETE", "/repos/o/r/issues/2/labels/status%3A%20blocked"],
    ]);
    assert.deepEqual(writes[0].body, { labels: ["status: ready"] });
  } finally { fake.server.close(); }
});

test("a dry run changes nothing", async () => {
  const fake = await fakeGitHub([issue(2, "[5.2] Produce an inventory report of the dataset")]);
  try {
    const { stdout } = await run("node", [script, "--dry-run"], { env: { ...process.env, GITHUB_REPOSITORY: "o/r", GITHUB_API_URL: fake.url } });
    assert.match(stdout, /dry run/);
    assert.equal(fake.calls.filter((c) => c.method !== "GET").length, 0);
  } finally { fake.server.close(); }
});
