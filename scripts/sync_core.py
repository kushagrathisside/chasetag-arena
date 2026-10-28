"""Copy the simulator core from a groundtag checkout into this repository.

This script is the only way the vendored files change. It writes ``vendor/SYNC.md`` with the
groundtag commit and the SHA-256 of every copied file. ``tests/test_vendor.py`` fails when a
vendored file differs from that record, so a hand edit shows up as a failing test.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "chasetag_arena"
DEFAULT_GROUNDTAG = Path("/home/kushagra/groundtag")

# Imports are rewritten to the vendored location. Families stay in their own subpackage,
# so ``groundtag.core.families.base`` becomes ``chasetag_arena.core.families.base``.
_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgroundtag\.spec\b"), "chasetag_arena.core.spec"),
    (re.compile(r"\bgroundtag\.core\b"), "chasetag_arena.core"),
    (re.compile(r"\bgroundtag\.agents\b"), "chasetag_arena.agents"),
)
# A copied file must not import any package that is absent from this repository.
_FORBIDDEN = re.compile(r"^\s*(from|import)\s+(groundtag|chasetag|myosuite)\b", re.MULTILINE)


@dataclass(frozen=True)
class Copy:
    source: Path  # absolute path in the groundtag checkout
    dest: Path  # path relative to src/chasetag_arena


def plan(gt_src: Path) -> list[Copy]:
    """The files to copy, from groundtag's ``src/groundtag`` directory."""
    copies: list[Copy] = []
    for path in sorted((gt_src / "spec").glob("*.py")):
        copies.append(Copy(path, Path("core/spec") / path.name))
    for path in sorted((gt_src / "core").glob("*.py")):
        copies.append(Copy(path, Path("core") / path.name))
    for path in sorted((gt_src / "core" / "families").glob("*.py")):
        copies.append(Copy(path, Path("core/families") / path.name))
    copies.append(Copy(gt_src / "agents" / "bots.py", Path("agents/bots.py")))
    return copies


def rewrite(text: str) -> str:
    for pattern, replacement in _REWRITES:
        text = pattern.sub(replacement, text)
    return text


def git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync(groundtag: Path) -> int:
    gt_src = groundtag / "src" / "groundtag"
    if not gt_src.is_dir():
        sys.exit(f"not a groundtag checkout: {groundtag}")

    # A dirty source tree would make the recorded commit a lie about the copied content.
    dirty = subprocess.run(
        ["git", "-C", str(groundtag), "status", "--porcelain", "--", "src/groundtag"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        sys.exit(f"groundtag has uncommitted changes under src/groundtag:\n{dirty}")

    commit = git_head(groundtag)
    copies = plan(gt_src)

    # Clear the vendored set first so a file removed upstream does not linger here.
    for old in (PKG / "core").rglob("*.py"):
        old.unlink()
    (PKG / "agents" / "bots.py").unlink(missing_ok=True)

    records: list[tuple[str, str]] = []
    for copy in copies:
        text = copy.source.read_text(encoding="utf-8")
        text = rewrite(text)
        bad = _FORBIDDEN.search(text)
        if bad:
            sys.exit(f"{copy.source} still imports {bad.group(2)} after rewriting")
        data = text.encode("utf-8")
        target = PKG / copy.dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append((sha256(data), f"src/chasetag_arena/{copy.dest.as_posix()}"))

    write_sync_md(commit, records)
    print(f"copied {len(records)} files from groundtag {commit}")
    return 0


def write_sync_md(commit: str, records: list[tuple[str, str]]) -> None:
    lines = [
        "# Vendored core",
        "",
        "These files are copied from the groundtag research repository by",
        "`scripts/sync_core.py`. Do not edit them by hand: `tests/test_vendor.py` fails when a",
        "vendored file differs from this record.",
        "",
        f"- groundtag commit: {commit}",
        "",
        "| SHA-256 | File |",
        "|---|---|",
    ]
    lines.extend(f"| {digest} | {path} |" for digest, path in records)
    (REPO / "vendor" / "SYNC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    groundtag = Path(argv[1]) if len(argv) > 1 else DEFAULT_GROUNDTAG
    return sync(groundtag.resolve())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
