"""The vendored core matches the record in ``vendor/SYNC.md``.

A file edited by hand, a file added, a file removed, or a missing record all fail here. The
only way to change the vendored code is ``scripts/sync_core.py``, which rewrites the record too.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SYNC_MD = REPO / "vendor" / "SYNC.md"
COMMIT_RE = re.compile(r"^- groundtag commit: ([0-9a-f]{40})$", re.MULTILINE)
ROW_RE = re.compile(r"^\| ([0-9a-f]{64}) \| (src/chasetag_arena/\S+\.py) \|$", re.MULTILINE)
FORBIDDEN_IMPORT = re.compile(r"^\s*(from|import)\s+(groundtag|chasetag|myosuite)\b", re.MULTILINE)


def _vendored_on_disk() -> set[str]:
    """Every .py file under core/ and the bots module, as repository-relative paths."""
    pkg = REPO / "src" / "chasetag_arena"
    paths = {p.relative_to(REPO).as_posix() for p in (pkg / "core").rglob("*.py")}
    paths.add((pkg / "agents" / "bots.py").relative_to(REPO).as_posix())
    return paths


def _recorded() -> dict[str, str]:
    text = SYNC_MD.read_text(encoding="utf-8")
    return {path: digest for digest, path in ROW_RE.findall(text)}


def test_sync_records_a_groundtag_commit() -> None:
    assert COMMIT_RE.search(SYNC_MD.read_text(encoding="utf-8")), "no groundtag commit recorded"


def test_every_vendored_file_is_recorded_and_present() -> None:
    recorded = set(_recorded())
    on_disk = _vendored_on_disk()
    assert recorded, "vendor/SYNC.md lists no files"
    missing = sorted(recorded - on_disk)
    unlisted = sorted(on_disk - recorded)
    assert not missing, f"listed in SYNC.md but absent: {missing}"
    assert not unlisted, f"vendored but not listed in SYNC.md: {unlisted}"


def test_vendored_hashes_match_the_record() -> None:
    mismatched = []
    for path, digest in _recorded().items():
        actual = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        if actual != digest:
            mismatched.append(path)
    assert not mismatched, f"hash differs from SYNC.md (edit by hand?): {mismatched}"


def test_vendored_files_import_only_vendored_or_external_code() -> None:
    offenders = []
    for path in _vendored_on_disk():
        if FORBIDDEN_IMPORT.search((REPO / path).read_text(encoding="utf-8")):
            offenders.append(path)
    assert not offenders, f"imports a package this repository does not have: {offenders}"
