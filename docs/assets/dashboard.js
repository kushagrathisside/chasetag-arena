/*
 * The contributor dashboard: the plan (docs/assets/contribution-plan.json) joined with the real
 * state of the issues on GitHub. Text from GitHub is only ever put on the page as text.
 */
(function () {
  "use strict";

  const CACHE_MS = 5 * 60 * 1000;
  const STATES = {
    ready: "Ready", taken: "Taken", blocked: "Blocked", done: "Done",
    unpublished: "Not published yet", notplanned: "Closed, not planned", unknown: "Not checked",
  };
  const FILTERS = [["all", "All"], ["ready", "Ready"], ["taken", "Taken"], ["blocked", "Blocked"], ["done", "Done"]];

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "text") node.textContent = v;
      else if (v !== false && v !== null && v !== undefined) node.setAttribute(k, v === true ? "" : v);
    }
    for (const child of children || []) node.append(child);
    return node;
  }

  function safeIssueUrl(url, repo) {
    const prefix = "https://github.com/" + repo + "/issues/";
    return typeof url === "string" && url.startsWith(prefix) ? url : null;
  }

  function readCache(key) {
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return null;
      const { at, issues } = JSON.parse(raw);
      return Date.now() - at < CACHE_MS && Array.isArray(issues) ? { at, issues } : null;
    } catch (_) { return null; }
  }
  function writeCache(key, issues) {
    try { sessionStorage.setItem(key, JSON.stringify({ at: Date.now(), issues })); } catch (_) { /* storage may be blocked */ }
  }

  async function fetchIssues(api, repo, label) {
    const key = "chasetag-dashboard:" + repo;
    const cached = readCache(key);
    if (cached) return { issues: cached.issues, at: cached.at };
    const all = [];
    for (let page = 1; page <= 5; page += 1) {
      const url = `${api}/repos/${repo}/issues?labels=${encodeURIComponent(label)}&state=all&per_page=100&page=${page}`;
      const response = await fetch(url, { headers: { Accept: "application/vnd.github+json" } });
      if (!response.ok) throw new Error("GitHub answered " + response.status);
      const batch = await response.json();
      all.push(...batch);
      if (batch.length < 100) break;
    }
    writeCache(key, all);
    return { issues: all, at: Date.now() };
  }

    /** One small chip for each domain of an issue (frontend, backend and so on). */
  function domainChips(domains) {
    return (domains || []).map((d) => el("span", { class: "dash-chip dash-domain", text: d }));
  }

function pill(state) {
    return el("span", { class: "dash-pill dash-" + state, text: STATES[state] });
  }

  function renderIssue(item, repo) {
    const url = safeIssueUrl(item.url, repo);
    const title = url ? el("a", { href: url, text: item.title }) : el("span", { text: item.title });
    const notes = [];
    if (item.state === "unknown") notes.push("The state could not be read from GitHub.");
    if (item.startHere) notes.push("Start here. It unlocks " + item.unlocks.join(", ") + ".");
    if (item.state === "blocked") notes.push("Waiting for " + item.waitingOn.join(", ") + ".");
    if (item.state === "taken" && item.assignee) notes.push("Taken by " + item.assignee + ".");
    const row = el("li", { class: "dash-issue", "data-state": item.state, "data-difficulty": item.difficulty.toLowerCase() }, [
      el("span", { class: "dash-id", text: item.id }),
      el("div", { class: "dash-main" }, [
        el("div", { class: "dash-title" }, [title]),
        el("div", { class: "dash-meta" }, [pill(item.state), el("span", { class: "dash-chip", text: item.difficulty })].concat(domainChips(item.domains))),
        notes.length ? el("div", { class: "dash-note", text: notes.join(" ") }) : null,
      ].filter(Boolean)),
    ]);
    if (item.startHere) row.classList.add("dash-start");
    return row;
  }

  function renderEpic(epic, repo) {
    const progress = el("progress", { max: String(epic.total), value: String(epic.done), "aria-label": `Epic ${epic.id}: ${epic.done} of ${epic.total} done` });
    const head = el("div", { class: "dash-epic-head" }, [
      el("h3", { text: `Epic ${epic.id} · ${epic.name}` }),
      el("div", { class: "dash-progress" }, [progress, el("span", { text: `${epic.done} of ${epic.total} done` })]),
    ]);
    const list = el("ul", { class: "dash-list" }, epic.issues.map((i) => renderIssue(i, repo)));
    const parts = [head, el("p", { class: "dash-purpose", text: epic.purpose }), list];
    if (epic.reserve) parts.push(el("p", { class: "dash-reserve", text: epic.reserve }));
    return el("section", { class: "dash-epic" }, parts);
  }

  function renderOpen(open, repo) {
    const items = open.modules.map((m) => {
      const url = safeIssueUrl(m.url, repo);
      const name = url ? el("a", { href: url, text: m.title }) : el("span", { text: m.title });
      const extra = m.state === "unpublished" ? "Not published yet" : (m.subIssues !== null ? `${m.subIssues} sub-issues` : "Open");
      return el("li", { class: "dash-module" }, [name, el("span", { class: "dash-chip", text: extra })].concat(domainChips(m.domains), [el("span", { class: "dash-note", text: m.summary })]));
    });
    return el("section", { class: "dash-epic dash-open" }, [
      el("div", { class: "dash-epic-head" }, [el("h3", { text: `Epic ${open.id} · ${open.name}` }), el("span", { class: "dash-chip", text: "Open epic" })]),
      el("p", { class: "dash-purpose", text: open.purpose }),
      el("ul", { class: "dash-modules" }, items),
    ]);
  }

  function renderSummary(counts, mode) {
    const order = [["ready", "Ready"], ["taken", "Taken"], ["blocked", "Blocked"], ["done", "Done"], ["unpublished", "Not published yet"]];
    return el("ul", { class: "dash-summary" }, order.map(([k, label]) => el("li", { class: "dash-card dash-" + k }, [
      el("strong", { text: mode === "unknown" ? "–" : String(counts[k]) }), el("span", { text: label }),
    ])));
  }

  function applyFilters(root) {
    const state = root.dataset.filter || "all";
    const beginner = root.dataset.beginner === "yes";
    for (const row of root.querySelectorAll(".dash-issue")) {
      const okState = state === "all" || row.dataset.state === state;
      const okLevel = !beginner || row.dataset.difficulty === "beginner";
      row.hidden = !(okState && okLevel);
    }
    for (const section of root.querySelectorAll(".dash-epic:not(.dash-open)")) {
      const any = Array.from(section.querySelectorAll(".dash-issue")).some((r) => !r.hidden);
      section.classList.toggle("dash-empty", !any);
    }
    for (const button of root.querySelectorAll(".dash-filter button")) {
      button.setAttribute("aria-pressed", String(button.dataset.value === state));
    }
  }

  function renderFilters(root) {
    const group = el("div", { class: "dash-filter", role: "group", "aria-label": "Show issues" },
      FILTERS.map(([value, label]) => el("button", { type: "button", "data-value": value, text: label, "aria-pressed": String(value === "all") })));
    group.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      root.dataset.filter = button.dataset.value;
      applyFilters(root);
    });
    const box = el("input", { type: "checkbox", id: "dash-beginner" });
    box.addEventListener("change", () => { root.dataset.beginner = box.checked ? "yes" : "no"; applyFilters(root); });
    return el("div", { class: "dash-controls" }, [group, el("label", { for: "dash-beginner" }, [box, " Beginner issues only"])]);
  }

  function withUnknown(model) {
    const epics = model.epics.map((e) => ({ ...e, issues: e.issues.map((i) => ({ ...i, state: "unknown", startHere: i.unlocks.length > 0 })) }));
    return { ...model, epics };
  }

  async function start(root) {
    const repo = root.dataset.repo;
    const api = root.dataset.api || "https://api.github.com";
    const status = el("p", { class: "dash-status", role: "status", "aria-live": "polite", text: "Reading the issues from GitHub…" });
    root.replaceChildren(status);
    let plan;
    try {
      plan = await (await fetch(new URL(root.dataset.plan, document.baseURI))).json();
    } catch (_) {
      status.textContent = "The plan could not be loaded. The issues are on GitHub: https://github.com/" + repo + "/issues";
      return;
    }
    let model, mode = "live", note;
    try {
      const { issues, at } = await fetchIssues(api, repo, plan.label);
      model = window.PlanStatus.compute(plan, issues);
      note = "Read from GitHub at " + new Date(at).toLocaleTimeString() + ". It is read again after five minutes.";
    } catch (error) {
      mode = "unknown";
      model = withUnknown(window.PlanStatus.compute(plan, []));
      note = "GitHub could not be reached (" + error.message + "), so the states cannot be shown. This is the plan only. Please try again later.";
    }
    status.textContent = note;
    const anyPublished = mode === "live" && model.counts.unpublished < Object.values(model.counts).reduce((a, b) => a + b, 0);
    const children = [status, renderSummary(model.counts, mode)];
    if (mode === "live" && !anyPublished) children.push(el("p", { class: "dash-banner", text: "The issues have not been published yet. This is the plan: it will fill in as the issues are created." }));
    children.push(renderFilters(root));
    for (const epic of model.epics) children.push(renderEpic(epic, repo));
    if (model.open) children.push(renderOpen(model.open, repo));
    root.replaceChildren(...children);
    root.dataset.filter = "all"; root.dataset.beginner = "no";
    applyFilters(root);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const root = document.getElementById("dashboard");
    if (root && window.PlanStatus) start(root);
  });
})();
