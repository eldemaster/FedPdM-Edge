#!/usr/bin/env python3
"""Fast engineering due-diligence checks for FedPdM-Edge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FILE_HASHES = {
    "experiments/reproduce.py": "0301de4cab66c12c5249ac6521adada3ae82b7241575061fe39867bd18a8a848",
    "results/reproducible/summary.csv": "67fc4b14fc1660365b19976e6c0ed6598a2f16e928a8892f18c3a03dcd032e86",
}
EXPECTED_PAPER_TREE_HASH = "f76d6bd93d7454a77fe9dc2c5237ec565c76321debf34b925f997371dd0a8d45"


class AuditError(ValueError):
    """Raised when an audit invariant is not satisfied."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def find_symlinks(root: Path) -> list[Path]:
    """Return every symlink below *root* without following symlinked directories."""
    root = Path(root)
    found: list[Path] = []
    if root.is_symlink():
        found.append(root)
        return found
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories + files:
            candidate = current_path / name
            if candidate.is_symlink():
                found.append(candidate)
    return sorted(found)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paper_files(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "paper",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    names = {name for name in result.stdout.decode().split("\0") if name}
    return [root / name for name in sorted(names)]


def paper_tree_sha256(root: Path) -> str:
    """Hash tracked and untracked, non-ignored paper sources and artifacts."""
    root = Path(root)
    paper = root / "paper"
    digest = hashlib.sha256()
    for path in _paper_files(root):
        if not path.is_file():
            continue
        digest.update(path.relative_to(paper).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def validate_protected_artifacts(root: Path) -> str:
    """Validate the canonical experiment, summary, and paper snapshots."""
    root = Path(root)
    for relative, expected in EXPECTED_FILE_HASHES.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise AuditError(f"{relative} must be a regular file")
        actual = _sha256(path)
        if actual != expected:
            raise AuditError(f"{relative} SHA-256 mismatch: {actual}")

    paper = root / "paper"
    if not paper.is_dir() or paper.is_symlink():
        raise AuditError("paper/ must be a regular directory")
    actual_paper_hash = paper_tree_sha256(root)
    if actual_paper_hash != EXPECTED_PAPER_TREE_HASH:
        raise AuditError(f"paper/ tree SHA-256 mismatch: {actual_paper_hash}")
    return "runner, summary, and paper tree match canonical SHA-256 values"


def _load_json(path: Path) -> Any:
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read valid JSON from {path}: {exc}") from exc


def _values_for_key(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                yield current_value
            yield from _values_for_key(current_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from _values_for_key(item, key)


def validate_dashboard_json(path: Path) -> str:
    """Validate dashboard attribution conservation and raw-data privacy fields."""
    data = _load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("engines"), list) or not data["engines"]:
        raise AuditError("dashboard data must contain a non-empty engines list")

    share_blocks = list(_values_for_key(data, "subsystem_share"))
    if not share_blocks:
        raise AuditError("dashboard data contains no subsystem_share series")
    rows_checked = 0
    for block_index, block in enumerate(share_blocks):
        if not isinstance(block, list) or not block:
            raise AuditError(f"subsystem_share block {block_index} must be a non-empty list")
        for row_index, row in enumerate(block):
            if not isinstance(row, list) or not row:
                raise AuditError(f"subsystem_share[{block_index}][{row_index}] must be a non-empty list")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in row):
                raise AuditError(f"subsystem_share[{block_index}][{row_index}] must be numeric")
            total = sum(row)
            if abs(total - 100.0) > 1e-6:
                raise AuditError(
                    f"subsystem_share[{block_index}][{row_index}] sums to {total}, expected 100.0"
                )
            rows_checked += 1

    privacy_values = list(_values_for_key(data, "raw_sensor_bytes_transmitted"))
    if not privacy_values:
        raise AuditError("dashboard data contains no raw_sensor_bytes_transmitted field")
    if any(value != 0 for value in privacy_values):
        raise AuditError(f"raw_sensor_bytes_transmitted must be 0 everywhere: {privacy_values}")
    return f"{rows_checked} attribution rows sum to 100.0; {len(privacy_values)} privacy fields are zero"


def validate_diagnostics_json(path: Path) -> str:
    """Validate the committed diagnostics network-isolation audit."""
    data = _load_json(path)
    audit = data.get("network_audit") if isinstance(data, dict) else None
    if not isinstance(audit, dict):
        raise AuditError("diagnostics data must contain network_audit")
    expected = {
        "total_socket_calls_during_diagnostics": 0,
        "raw_sensor_bytes_sent": 0,
    }
    for key, expected_value in expected.items():
        if key not in audit:
            raise AuditError(f"network_audit is missing {key}")
        if audit[key] != expected_value:
            raise AuditError(f"network_audit.{key} is {audit[key]!r}, expected {expected_value}")
    return "diagnostics recorded 0 socket calls and 0 raw sensor bytes sent"


def _run_check(name: str, function, *args) -> CheckResult:
    try:
        detail = function(*args)
        return CheckResult(name, True, str(detail))
    except Exception as exc:  # report every independent audit result in one run
        return CheckResult(name, False, str(exc))


def _check_symlinks(root: Path) -> str:
    links = find_symlinks(root)
    if links:
        relative = [str(path.relative_to(root)) for path in links]
        raise AuditError("found symlinks: " + ", ".join(relative))
    return "0 symlinks found"


def _check_cmapss_environment(required: bool) -> str:
    configured = os.environ.get("CMAPSS_ZIP")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise AuditError(f"CMAPSS_ZIP does not name a file: {path}")
        return f"enabled ({path})"
    if required:
        raise AuditError("--with-cmapss requires CMAPSS_ZIP to name an existing archive")
    return "not configured; dataset-gated tests may skip"


def _run_unittests(root: Path) -> CheckResult:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = subprocess.run(command, cwd=root, env=os.environ.copy())
    detail = "full unittest discovery"
    if os.environ.get("CMAPSS_ZIP"):
        detail += " with C-MAPSS enabled"
    return CheckResult("Unit tests", completed.returncode == 0, detail)


def _print_summary(results: list[CheckResult]) -> None:
    rows = [(result.name, "PASS" if result.passed else "FAIL", result.detail.replace("\n", " ")) for result in results]
    headers = ("CHECK", "STATUS", "DETAIL")
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(3)]
    rule = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print("\nPREFLIGHT AUDIT SUMMARY")
    print(rule)
    print("| " + " | ".join(headers[index].ljust(widths[index]) for index in range(3)) + " |")
    print(rule)
    for row in rows:
        print("| " + " | ".join(row[index].ljust(widths[index]) for index in range(3)) + " |")
    print(rule)
    print("OVERALL:", "PASS" if all(result.passed for result in results) else "FAIL")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-unittest",
        action="store_true",
        help="run artifact and schema checks only (used when unittest already ran)",
    )
    parser.add_argument(
        "--with-cmapss",
        action="store_true",
        help="require CMAPSS_ZIP; existing tests automatically consume the archive",
    )
    args = parser.parse_args(argv)

    results = [
        _run_check("Repository symlinks", _check_symlinks, ROOT),
        _run_check("Protected artifacts", validate_protected_artifacts, ROOT),
        _run_check("Dashboard schema", validate_dashboard_json, ROOT / "dashboard" / "demo_data.json"),
        _run_check(
            "Diagnostics privacy",
            validate_diagnostics_json,
            ROOT / "results" / "diagnostics_validation.json",
        ),
        _run_check("C-MAPSS input", _check_cmapss_environment, args.with_cmapss),
    ]
    if args.skip_unittest:
        results.append(CheckResult("Unit tests", True, "skipped by --skip-unittest"))
    else:
        results.append(_run_unittests(ROOT))

    _print_summary(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
