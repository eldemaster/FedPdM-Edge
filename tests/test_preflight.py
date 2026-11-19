"""Focused tests for the engineering preflight audit."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import preflight_check as preflight  # noqa: E402


class SymlinkDetectorTest(unittest.TestCase):
    def test_finds_file_and_directory_symlinks_without_following_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_file = root / "target.txt"
            target_file.write_text("target\n")
            target_dir = root / "target-dir"
            target_dir.mkdir()
            file_link = root / "file-link"
            directory_link = root / "directory-link"
            file_link.symlink_to(target_file)
            directory_link.symlink_to(target_dir, target_is_directory=True)

            self.assertEqual(
                preflight.find_symlinks(root),
                sorted([directory_link, file_link]),
            )


class JsonInvariantTest(unittest.TestCase):
    def _write_json(self, root: Path, name: str, value) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_dashboard_accepts_conserved_shares_and_zero_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_json(
                Path(temp_dir),
                "dashboard.json",
                {
                    "engines": [
                        {
                            "diagnostics": {
                                "subsystem_share": [[10.0, 20.0, 30.0, 40.0]],
                                "privacy": {"raw_sensor_bytes_transmitted": 0},
                            }
                        }
                    ]
                },
            )
            detail = preflight.validate_dashboard_json(path)
            self.assertIn("sum to 100.0", detail)

    def test_dashboard_rejects_bad_share_or_nonzero_raw_bytes(self):
        cases = (
            ([[10.0, 20.0, 30.0, 39.0]], 0, "sums to"),
            ([[10.0, 20.0, 30.0, 40.0]], 1, "must be 0"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (shares, raw_bytes, message) in enumerate(cases):
                with self.subTest(message=message):
                    path = self._write_json(
                        root,
                        f"dashboard-{index}.json",
                        {
                            "engines": [
                                {
                                    "diagnostics": {
                                        "subsystem_share": shares,
                                        "raw_sensor_bytes_transmitted": raw_bytes,
                                    }
                                }
                            ]
                        },
                    )
                    with self.assertRaisesRegex(preflight.AuditError, message):
                        preflight.validate_dashboard_json(path)

    def test_diagnostics_requires_zero_socket_calls_and_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid = self._write_json(
                root,
                "valid.json",
                {
                    "network_audit": {
                        "total_socket_calls_during_diagnostics": 0,
                        "raw_sensor_bytes_sent": 0,
                    }
                },
            )
            self.assertIn("0 socket calls", preflight.validate_diagnostics_json(valid))

            for key in ("total_socket_calls_during_diagnostics", "raw_sensor_bytes_sent"):
                with self.subTest(key=key):
                    report = {
                        "network_audit": {
                            "total_socket_calls_during_diagnostics": 0,
                            "raw_sensor_bytes_sent": 0,
                        }
                    }
                    report["network_audit"][key] = 1
                    invalid = self._write_json(root, f"invalid-{key}.json", report)
                    with self.assertRaisesRegex(preflight.AuditError, key):
                        preflight.validate_diagnostics_json(invalid)


class PreflightCommandTest(unittest.TestCase):
    def test_valid_repository_exits_cleanly_in_fast_mode(self):
        environment = os.environ.copy()
        environment.pop("CMAPSS_ZIP", None)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "scripts/preflight_check.py", "--skip-unittest"],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertIn("PREFLIGHT AUDIT SUMMARY", completed.stdout)
        self.assertIn("OVERALL: PASS", completed.stdout)


if __name__ == "__main__":
    unittest.main()
