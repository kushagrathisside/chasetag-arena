/*
 * The state of every planned issue, worked out from the plan and the GitHub issues.
 * It is used by the contributor dashboard (in the browser) and by the label-sync script (in Node).
 *
 * States:
 *   unpublished  the issue is in the plan but has not been created on GitHub yet
 *   done         the issue is closed
 *   notplanned   the issue was closed as "not planned"
 *   blocked      an issue that it needs has not been completed
 *   taken        nothing blocks it, and it is assigned to someone
 *   ready        nothing blocks it, and nobody has it
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PlanStatus = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const TASK_ID = /^\[(\d+\.\d+)\]/;
  const MODULE_TITLE = /^\[Testing\]\s*(.+)$/;

  /** Issues (not pull requests) by plan id, from a list of GitHub issues. */
  function indexIssues(issues) {
    const tasks = new Map();
    const modules = new Map();
    for (const issue of issues || []) {
      if (!issue || issue.pull_request) continue;
      const title = String(issue.title || "");
      const task = TASK_ID.exec(title);
      if (task) { tasks.set(task[1], issue); continue; }
      const mod = MODULE_TITLE.exec(title);
      if (mod) modules.set(mod[1].trim(), issue);
    }
    return { tasks, modules };
  }

  function isDone(issue) {
    return Boolean(issue) && issue.state === "closed" && issue.state_reason !== "not_planned";
  }

  /** The state of one issue of the plan. `needs` are plan ids. */
  function stateOf(id, needs, tasks) {
    const issue = tasks.get(id);
    if (!issue) return "unpublished";
    if (issue.state === "closed") return issue.state_reason === "not_planned" ? "notplanned" : "done";
    const waiting = needs.filter((n) => !isDone(tasks.get(n)));
    if (waiting.length > 0) return "blocked";
    return (issue.assignees && issue.assignees.length > 0) || issue.assignee ? "taken" : "ready";
  }

  /** The plan with a state on every issue. Epics 1 to 6 only: the open epic has modules instead. */
  function compute(plan, issues) {
    const { tasks, modules } = indexIssues(issues);
    const unlocks = new Map();
    for (const epic of plan.epics)
      for (const item of epic.issues)
        for (const n of item.needs) unlocks.set(n, (unlocks.get(n) || []).concat(item.id));
    const counts = { ready: 0, taken: 0, blocked: 0, done: 0, unpublished: 0, notplanned: 0 };
    const epics = plan.epics.map((epic) => {
      const items = epic.issues.map((item) => {
        const issue = tasks.get(item.id) || null;
        const state = stateOf(item.id, item.needs, tasks);
        counts[state] += 1;
        const waitingOn = item.needs.filter((n) => !isDone(tasks.get(n)));
        const startHere = (state === "ready" || state === "unpublished") && (unlocks.get(item.id) || []).length > 0;
        return {
          ...item, state, waitingOn, startHere,
          unlocks: unlocks.get(item.id) || [],
          number: issue ? issue.number : null,
          url: issue ? issue.html_url : null,
          assignee: issue && issue.assignee ? issue.assignee.login : null,
        };
      });
      const done = items.filter((i) => i.state === "done").length;
      return { ...epic, issues: items, done, total: items.length };
    });
    const open = plan.openEpic
      ? {
          ...plan.openEpic,
          modules: plan.openEpic.modules.map((m) => {
            const issue = modules.get(m.title) || null;
            const summary = issue && issue.sub_issues_summary ? issue.sub_issues_summary : null;
            return { ...m, number: issue ? issue.number : null, url: issue ? issue.html_url : null,
                     state: issue ? issue.state : "unpublished", subIssues: summary ? summary.total : null };
          }),
        }
      : null;
    return { epics, open, counts };
  }

  /** The label an open, published issue should carry: "status: blocked" or "status: ready". */
  function desiredLabel(state) {
    if (state === "blocked") return "status: blocked";
    if (state === "ready" || state === "taken") return "status: ready";
    return null;
  }

  const STATUS_LABELS = ["status: ready", "status: blocked"];

  /** The label changes that bring open, published issues in step with their state. */
  function labelChanges(plan, issues) {
    const { tasks } = indexIssues(issues);
    const changes = [];
    for (const epic of plan.epics) {
      for (const item of epic.issues) {
        const issue = tasks.get(item.id);
        if (!issue || issue.state !== "open") continue;
        const want = desiredLabel(stateOf(item.id, item.needs, tasks));
        const have = (issue.labels || []).map((l) => (typeof l === "string" ? l : l.name));
        const add = want && !have.includes(want) ? [want] : [];
        const remove = have.filter((l) => STATUS_LABELS.includes(l) && l !== want);
        if (add.length || remove.length) changes.push({ number: issue.number, id: item.id, add, remove });
      }
    }
    return changes;
  }

  return { TASK_ID, indexIssues, isDone, stateOf, compute, desiredLabel, labelChanges };
});
