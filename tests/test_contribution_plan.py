"""The contribution plan that the dashboard and the label workflow read: it is well formed, and
every epic stands on its own."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLAN = json.loads((ROOT / "docs" / "assets" / "contribution-plan.json").read_text(encoding="utf-8"))
ISSUES = [issue for epic in PLAN["epics"] for issue in epic["issues"]]
EPIC_OF = {issue["id"]: epic["id"] for epic in PLAN["epics"] for issue in epic["issues"]}


def test_ids_are_unique_and_follow_their_epic() -> None:
    ids = [issue["id"] for issue in ISSUES]
    assert len(ids) == len(set(ids))
    for epic in PLAN["epics"]:
        for issue in epic["issues"]:
            assert re.fullmatch(rf"{epic['id']}\.\d+", issue["id"]), issue["id"]


@pytest.mark.parametrize("epic", PLAN["epics"], ids=lambda e: f"epic {e['id']}")
def test_an_epic_has_five_to_seven_issues(epic: dict) -> None:
    assert 5 <= len(epic["issues"]) <= 7


def test_every_issue_is_complete() -> None:
    for issue in ISSUES:
        assert issue["title"] and not issue["title"].endswith(".")
        assert issue["difficulty"] in {"Beginner", "Intermediate"}


def test_an_issue_needs_only_issues_of_its_own_epic() -> None:
    for issue in ISSUES:
        for other in issue["needs"]:
            assert other in EPIC_OF, f"{issue['id']} names an unknown issue {other}"
            assert EPIC_OF[other] == EPIC_OF[issue["id"]], f"{issue['id']} depends on another epic"


def test_a_prerequisite_is_numbered_before_the_issue_that_needs_it() -> None:
    def number(issue_id: str) -> tuple[int, ...]:
        return tuple(int(part) for part in issue_id.split("."))

    for issue in ISSUES:
        for other in issue["needs"]:
            assert number(other) < number(issue["id"]), f"{other} must come before {issue['id']}"


def test_the_issues_of_an_epic_are_numbered_without_gaps() -> None:
    for epic in PLAN["epics"]:
        numbers = [int(issue["id"].split(".")[1]) for issue in epic["issues"]]
        assert numbers == list(range(1, len(numbers) + 1))


def test_no_issue_has_a_helps_link() -> None:
    # Only a prerequisite links two issues, so that a student has one rule to learn.
    assert all("helps" not in issue for issue in ISSUES)


def test_the_order_of_issues_has_no_cycle() -> None:
    needs = {issue["id"]: issue["needs"] for issue in ISSUES}
    done: set[str] = set()
    pending = dict(needs)
    while pending:
        ready = [i for i, deps in pending.items() if all(d in done for d in deps)]
        assert ready, f"a cycle among {sorted(pending)}"
        for i in ready:
            done.add(i)
            del pending[i]


def test_the_open_epic_lists_its_modules() -> None:
    modules = PLAN["openEpic"]["modules"]
    assert len(modules) >= 10
    assert len({m["title"] for m in modules}) == len(modules)


def test_the_dashboard_files_the_site_needs_exist() -> None:
    for name in ("plan-status.js", "dashboard.js", "dashboard.css", "contribution-plan.json"):
        assert (ROOT / "docs" / "assets" / name).is_file(), name
    assert (ROOT / "docs" / "diagrams" / "contribution-map.svg").is_file()


DOMAINS = {"frontend", "backend", "simulation", "ml", "data", "documentation", "testing"}


def test_every_issue_and_module_has_a_known_domain() -> None:
    for issue in ISSUES:
        assert issue["domains"], f"{issue['id']} has no domain"
        assert set(issue["domains"]) <= DOMAINS, f"{issue['id']} has an unknown domain"
    for module in PLAN["openEpic"]["modules"]:
        assert module["domains"], f"{module['title']} has no domain"
        assert set(module["domains"]) <= DOMAINS, f"{module['title']} has an unknown domain"
