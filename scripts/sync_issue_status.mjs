#!/usr/bin/env node
/*
 * Keeps the `status: ready` and `status: blocked` labels of the contribution issues in step with the
 * plan in docs/assets/contribution-plan.json: an issue is blocked until the issues it needs are closed.
 *
 *   GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/name node scripts/sync_issue_status.mjs [--dry-run]
 *
 * With --dry-run it only prints what it would change. The token needs "issues: write".
 */
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PlanStatus = createRequire(import.meta.url)(path.join(root, "docs/assets/plan-status.js"));
const plan = JSON.parse(readFileSync(path.join(root, "docs/assets/contribution-plan.json"), "utf8"));
const dryRun = process.argv.includes("--dry-run");
const repo = process.env.GITHUB_REPOSITORY || plan.repo;
const token = process.env.GITHUB_TOKEN;
if (!dryRun && !token) { console.error("GITHUB_TOKEN is needed, unless --dry-run is given"); process.exit(2); }

const api = process.env.GITHUB_API_URL || "https://api.github.com";
const headers = { Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", ...(token ? { Authorization: `Bearer ${token}` } : {}) };
async function call(method, url, body) {
  const response = await fetch(api + url, { method, headers: { ...headers, ...(body ? { "Content-Type": "application/json" } : {}) }, body: body ? JSON.stringify(body) : undefined });
  if (!response.ok && !(method === "DELETE" && response.status === 404)) throw new Error(`${method} ${url}: ${response.status}`);
  return method === "DELETE" ? null : response.json();
}

const issues = [];
for (let page = 1; page <= 10; page += 1) {
  const batch = await call("GET", `/repos/${repo}/issues?labels=${encodeURIComponent(plan.label)}&state=all&per_page=100&page=${page}`);
  issues.push(...batch);
  if (batch.length < 100) break;
}
const changes = PlanStatus.labelChanges(plan, issues);
for (const c of changes) {
  console.log(`#${c.number} (${c.id}): add [${c.add.join(", ")}] remove [${c.remove.join(", ")}]`);
  if (dryRun) continue;
  if (c.add.length) await call("POST", `/repos/${repo}/issues/${c.number}/labels`, { labels: c.add });
  for (const name of c.remove) await call("DELETE", `/repos/${repo}/issues/${c.number}/labels/${encodeURIComponent(name)}`);
}
console.log(changes.length ? `${changes.length} issue(s) updated${dryRun ? " (dry run)" : ""}.` : "Every label is already correct.");
