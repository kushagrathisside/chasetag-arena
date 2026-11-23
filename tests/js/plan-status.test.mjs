import test from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const PlanStatus = require("../../docs/assets/plan-status.js");

const plan = {
  epics: [
    { id: "1", name: "One", issues: [
      { id: "1.1", title: "First", difficulty: "Beginner", needs: [] },
      { id: "1.2", title: "Second", difficulty: "Beginner", needs: ["1.1"] },
      { id: "1.3", title: "Third", difficulty: "Beginner", needs: [] },
    ] },
    { id: "2", name: "Two", issues: [
      { id: "2.1", title: "Reader", difficulty: "Intermediate", needs: [] },
      { id: "2.2", title: "Tool", difficulty: "Beginner", needs: ["2.1"] },
      { id: "2.3", title: "Other tool", difficulty: "Beginner", needs: ["2.1"] },
    ] },
  ],
  openEpic: { id: "7", name: "Open", purpose: "", modules: [{ title: "Core", summary: "" }, { title: "Server", summary: "" }] },
};

const issue = (title, extra = {}) => ({ number: 1, title, state: "open", html_url: "https://github.com/x/y/issues/1", assignees: [], ...extra });
const stateById = (result) => Object.fromEntries(result.epics.flatMap((e) => e.issues.map((i) => [i.id, i.state])));

test("an issue that has not been created is unpublished", () => {
  const result = PlanStatus.compute(plan, []);
  assert.equal(stateById(result)["1.1"], "unpublished");
  assert.equal(result.counts.unpublished, 6);
});

test("an open issue with nothing to wait for is ready, or taken when assigned", () => {
  const result = PlanStatus.compute(plan, [
    issue("[1.1] First"),
    issue("[1.3] Third", { assignees: [{ login: "ada" }], assignee: { login: "ada" } }),
  ]);
  const s = stateById(result);
  assert.equal(s["1.1"], "ready");
  assert.equal(s["1.3"], "taken");
});

test("an issue is blocked until what it needs is closed", () => {
  const open = [issue("[1.1] First"), issue("[1.2] Second")];
  assert.equal(stateById(PlanStatus.compute(plan, open))["1.2"], "blocked");
  const closed = [issue("[1.1] First", { state: "closed", state_reason: "completed" }), issue("[1.2] Second")];
  const s = stateById(PlanStatus.compute(plan, closed));
  assert.equal(s["1.1"], "done");
  assert.equal(s["1.2"], "ready");
});

test("an issue that was closed as not planned does not unlock others", () => {
  const issues = [issue("[1.1] First", { state: "closed", state_reason: "not_planned" }), issue("[1.2] Second")];
  const s = stateById(PlanStatus.compute(plan, issues));
  assert.equal(s["1.1"], "notplanned");
  assert.equal(s["1.2"], "blocked");
});

test("an issue needing a prerequisite that is not published is blocked", () => {
  const s = stateById(PlanStatus.compute(plan, [issue("[2.2] Tool")]));
  assert.equal(s["2.2"], "blocked");
});

test("an issue that unlocks others is marked as the place to start", () => {
  const result = PlanStatus.compute(plan, [issue("[2.1] Reader")]);
  const reader = result.epics[1].issues[0];
  assert.equal(reader.startHere, true);
  assert.deepEqual(reader.unlocks, ["2.2", "2.3"]);
  assert.equal(result.epics[0].issues[2].startHere, false);
});

test("pull requests are ignored, and counts add up", () => {
  const result = PlanStatus.compute(plan, [issue("[1.1] First", { pull_request: {} })]);
  assert.equal(stateById(result)["1.1"], "unpublished");
  const total = Object.values(result.counts).reduce((a, b) => a + b, 0);
  assert.equal(total, 6);
});

test("the open epic's module issues are matched by title", () => {
  const result = PlanStatus.compute(plan, [
    issue("[Testing] Core", { number: 9, html_url: "https://github.com/x/y/issues/9", sub_issues_summary: { total: 3 } }),
  ]);
  assert.equal(result.open.modules[0].number, 9);
  assert.equal(result.open.modules[0].subIssues, 3);
  assert.equal(result.open.modules[1].state, "unpublished");
});

test("the label to apply follows the state", () => {
  assert.equal(PlanStatus.desiredLabel("blocked"), "status: blocked");
  assert.equal(PlanStatus.desiredLabel("ready"), "status: ready");
  assert.equal(PlanStatus.desiredLabel("taken"), "status: ready");
  assert.equal(PlanStatus.desiredLabel("done"), null);
});

test("closing a prerequisite turns the label of the issue it unlocks from blocked to ready", () => {
  const issues = [
    issue("[1.1] First", { number: 11, state: "closed", state_reason: "completed", labels: [{ name: "status: ready" }] }),
    issue("[1.2] Second", { number: 12, labels: [{ name: "opencode" }, { name: "status: blocked" }] }),
    issue("[1.3] Third", { number: 13, labels: [{ name: "status: ready" }] }),
  ];
  const changes = PlanStatus.labelChanges(plan, issues);
  assert.deepEqual(changes, [{ number: 12, id: "1.2", add: ["status: ready"], remove: ["status: blocked"] }]);
});

test("a label is added when an issue has none, and closed issues are left alone", () => {
  const issues = [issue("[2.1] Reader", { number: 21 }), issue("[2.2] Tool", { number: 22 }), issue("[2.3] Other", { number: 23, state: "closed" })];
  const changes = PlanStatus.labelChanges(plan, issues);
  assert.deepEqual(changes.map((c) => [c.number, c.add]), [[21, ["status: ready"]], [22, ["status: blocked"]]]);
});
