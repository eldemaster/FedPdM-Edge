"""Tests and artifact-integrity checks for diagnostics validation.

The unit and artifact tests are self-contained.  The two end-to-end smoke
tests run only when CMAPSS_ZIP names an existing C-MAPSS archive.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import diagnostics as D  # noqa: E402
from common import MAX_RUL, SEQ_LEN, USEFUL_SENSORS, CmapssRegressor, set_seed  # noqa: E402
from experiments import validate_diagnostics as V  # noqa: E402


SEEDS = [7, 42, 100, 777, 2024]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
CMAPSS_ZIP = os.environ.get("CMAPSS_ZIP")
HAS_CMAPSS = bool(CMAPSS_ZIP and Path(CMAPSS_ZIP).is_file())


def synthetic_fleet(n_units=4, n_cycles=45, seed=0):
    """Small multi-unit raw fleet with a quiet start and an HPC EOL ramp."""
    rng = np.random.RandomState(seed)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, n_cycles + 1):
            ramp = max(0, cycle - 30) / 15.0
            values = {
                sensor: 100.0 + i * 5.0 + rng.normal(0, 0.2)
                for i, sensor in enumerate(D.FAULT_SENSORS)
            }
            values["s3"] += 4.0 * ramp
            values["s11"] += 3.0 * ramp
            rows.append({
                "unit": unit,
                "cycle": cycle,
                "setting1": rng.normal(0, 0.001),
                "setting2": rng.normal(0, 0.001),
                "setting3": 100.0 + rng.normal(0, 0.001),
                **values,
            })
    return pd.DataFrame(rows)


def scales_of(frame):
    return {
        "min": frame[USEFUL_SENSORS].min().to_dict(),
        "max": frame[USEFUL_SENSORS].max().to_dict(),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paper_tree_sha256():
    digest = hashlib.sha256()
    for path in sorted(p for p in (ROOT / "paper").rglob("*") if p.is_file()):
        digest.update(path.relative_to(ROOT / "paper").as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class SocketAuditTest(unittest.TestCase):
    def test_counts_local_create_connection_connect_and_send(self):
        # Use a real local socket pair for the send, plus no-op local connect
        # shims so this remains runnable in sandboxes that forbid TCP bind().
        left, right = socket.socketpair()

        def local_connect(sock, address):
            self.assertEqual(address, ("local-pair", 0))

        def local_create_connection(address, *args, **kwargs):
            left.connect(address)
            return left

        try:
            with mock.patch.object(socket.socket, "connect", local_connect), \
                    mock.patch.object(socket, "create_connection", local_create_connection):
                with V.SocketAudit() as audit:
                    client = socket.create_connection(("local-pair", 0))
                    # create_connection itself and its socket.connect call are both audited.
                    self.assertEqual(audit.calls, 2)
                    payload = b"local-pair-audit"
                    client.send(payload)
                    self.assertEqual(right.recv(len(payload)), payload)
                    self.assertEqual(audit.calls, 3)
                    self.assertEqual(audit.bytes, len(payload))
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair()
        try:
            with V.SocketAudit() as audit:
                left.send(b"pair")
                self.assertEqual(right.recv(4), b"pair")
            self.assertEqual(audit.calls, 1)
            self.assertEqual(audit.bytes, 4)
        finally:
            left.close()
            right.close()

    def test_baseline_and_diagnosis_have_no_socket_activity(self):
        fleet = synthetic_fleet()
        set_seed(3)
        with V.SocketAudit() as audit:
            baseline = D.fit_healthy_baseline(fleet, n_regimes=1)
            diagnostics = D.EdgeDiagnostics(CmapssRegressor(), scales_of(fleet), baseline)
            result = diagnostics.diagnose_engine(fleet[fleet.unit == 1], "unit-1")
        self.assertGreater(len(result["cycles"]), 0)
        self.assertEqual(audit.calls, 0)
        self.assertEqual(audit.bytes, 0)


class ValidationHelpersTest(unittest.TestCase):
    def test_engine_true_rul_caps_and_aligns_cycles(self):
        actual = V.engine_true_rul([1, 30, 50], last_cycle=50, final_rul=10)
        np.testing.assert_array_equal(actual, [59, 30, 10])
        capped = V.engine_true_rul([1, 2], last_cycle=300, final_rul=50)
        np.testing.assert_array_equal(capped, [MAX_RUL, MAX_RUL])

    @staticmethod
    def _single_fault_diags():
        hpc = np.zeros(14)
        fan = np.zeros(14)
        drift_hpc = np.zeros(14)
        drift_fan = np.zeros(14)
        for sensor, value in zip(("s3", "s7", "s9"), (50, 30, 20)):
            hpc[D.FAULT_SENSORS.index(sensor)] = value
            drift_hpc[D.FAULT_SENSORS.index(sensor)] = value / 10
        for sensor, value in zip(("s8", "s13", "s15"), (50, 30, 20)):
            fan[D.FAULT_SENSORS.index(sensor)] = value
            drift_fan[D.FAULT_SENSORS.index(sensor)] = value / 10

        def entry(sensor_row, drift_row, subsystem_row):
            return {
                "true_rul": np.array([80, 20, 10]),
                "sensor_share": np.vstack([sensor_row] * 3),
                "window_drift": np.vstack([drift_row] * 3),
                "subsystem_share": np.vstack([subsystem_row] * 3),
            }

        return {
            "hpc-unit": entry(hpc, drift_hpc, [0, 0, 100, 0]),
            "fan-unit": entry(fan, drift_fan, [100, 0, 0, 0]),
        }

    def test_single_fault_metrics_aggregate_and_filter_multiple_units(self):
        diags = self._single_fault_diags()
        all_units = V.single_fault_metrics(diags)
        self.assertEqual(all_units["n_windows_rul_lt_30"], 4)
        self.assertEqual(all_units["attr_primary_hpc_rate"], 0.5)
        self.assertEqual(all_units["attr_primary_core_rate"], 0.5)
        self.assertEqual(all_units["attr_top3_core_sensor_fraction"], 0.5)

        hpc_only = V.single_fault_metrics(diags, ["hpc-unit"])
        self.assertEqual(hpc_only["n_windows_rul_lt_30"], 2)
        self.assertEqual(hpc_only["attr_primary_hpc_rate"], 1.0)
        self.assertEqual(hpc_only["attr_top3_core_sensor_fraction"], 1.0)
        self.assertEqual(hpc_only["drift_top3_core_sensor_fraction"], 1.0)

    def test_eol_profiles_use_last_ten_windows_per_unit(self):
        diags = {}
        for unit, offset in (("u1", 0.0), ("u2", 100.0)):
            values = np.arange(12, dtype=float) + offset
            diags[unit] = {
                "window_drift": np.repeat(values[:, None], 14, axis=1),
                "sensor_share": np.repeat((values * 2)[:, None], 14, axis=1),
                "subsystem_share": np.repeat((values * 3)[:, None], 4, axis=1),
            }
        drift, attr, sub = V.eol_profiles(diags, ["u1", "u2"])
        np.testing.assert_allclose(drift[:, 0], [6.5, 106.5])
        np.testing.assert_allclose(attr[:, 0], [13.0, 213.0])
        np.testing.assert_allclose(sub[:, 0], [19.5, 319.5])

    @staticmethod
    def _two_mode_diags():
        ref = np.zeros(14)
        fan_profile = np.zeros(14)
        for sensor, value in (("s3", 3.0), ("s7", -4.0), ("s9", 5.0), ("s11", 4.0)):
            ref[D.FAULT_SENSORS.index(sensor)] = value
        for sensor, value in (("s8", 6.0), ("s13", 5.0), ("s15", -4.0)):
            fan_profile[D.FAULT_SENSORS.index(sensor)] = value

        diags = {}
        keys = []
        for index in range(8):
            is_hpc = index < 4
            base = ref if is_hpc else fan_profile
            drift = np.repeat(base[None, :], 12, axis=0)
            share = np.full(14, 0.1)
            dominant = ("s3", "s7", "s9") if is_hpc else ("s8", "s13", "s15")
            for sensor, value in zip(dominant, (50.0, 30.0, 18.9)):
                share[D.FAULT_SENSORS.index(sensor)] = value
            shares = np.repeat(share[None, :], 12, axis=0)
            key = f"unit-{index}"
            keys.append(key)
            diags[key] = {
                "window_drift": drift,
                "sensor_share": shares,
                "subsystem_share": D.subsystem_shares(shares),
            }
        return diags, keys, ref

    def test_orient_and_two_fault_metrics_separate_fault_modes(self):
        diags, keys, ref = self._two_mode_diags()
        drift, _, _ = V.eol_profiles(diags, keys)
        arbitrary_labels = np.array([1] * 4 + [0] * 4)
        np.testing.assert_array_equal(V.orient(arbitrary_labels, drift, ref), [0] * 4 + [1] * 4)

        metrics, labels = V.two_fault_metrics(diags, keys, seed=7, ref=ref)
        for view in ("drift", "attribution", "combined"):
            np.testing.assert_array_equal(labels[view], [0] * 4 + [1] * 4)
            self.assertGreater(metrics[view]["silhouette"], 0.95)
            self.assertEqual(metrics[view]["hpc_like"]["n_units"], 4)
            self.assertEqual(metrics[view]["fan_candidate"]["n_units"], 4)
        self.assertEqual(metrics["cross_view_ari_drift_vs_attribution"], 1.0)

    def test_saved_tensor_bytes_counts_autograd_storage(self):
        model = torch.nn.Linear(3, 1, bias=False)
        x = np.ones((2, 3), dtype=np.float32)
        measured = V.saved_tensor_bytes(model, x)
        self.assertGreaterEqual(measured, model.weight.numel() * model.weight.element_size())
        self.assertEqual(measured, V.saved_tensor_bytes(model, x))


class ArtifactIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / "results/diagnostics_validation.json").read_text())
        cls.docs = (ROOT / "docs/diagnostics.md").read_text()
        cls.experiments = (ROOT / "results/EXPERIMENTS.md").read_text()

    def test_network_seeds_and_latency_constraints(self):
        network = self.report["network_audit"]
        self.assertEqual(network["total_socket_calls_during_diagnostics"], 0)
        self.assertEqual(network["raw_sensor_bytes_sent"], 0)
        self.assertEqual(list(self.report["subsets"]), SUBSETS)
        for subset in SUBSETS:
            result = self.report["subsets"][subset]
            self.assertEqual([row["seed"] for row in result["per_seed"]], SEEDS)
            latency = result["latency"]
            self.assertEqual(latency["threads"], 1)
            self.assertEqual(latency["network_socket_calls"], 0)
            for name, value in latency.items():
                if name not in ("threads", "network_socket_calls"):
                    self.assertGreater(value, 0, f"{subset} {name}")

    def test_documented_rmse_and_single_fault_rates_match_json(self):
        subsets = self.report["subsets"]
        rmse = ", ".join(
            f"{name} {subsets[name]['test_rmse']['mean']:.2f} ± {subsets[name]['test_rmse']['std']:.2f}"
            for name in SUBSETS
        )
        self.assertIn(f"**Official test RMSE (mean ± sd over 5 seeds):** {rmse}.", self.docs)

        def pct(name, metric):
            value = subsets[name]["summary"]["baseline_x_gradient"][metric]
            return f"{100 * value['mean']:.1f} ± {100 * value['std']:.1f} %"

        self.assertIn(
            f"| Primary attribution subsystem = HPC | {pct('FD001', 'attr_primary_hpc_rate')} | "
            f"{pct('FD002', 'attr_primary_hpc_rate')} | 35.7 % |",
            self.docs,
        )
        self.assertIn(
            f"| Primary attribution subsystem ∈ core (HPC ∪ Turbine) | "
            f"{pct('FD001', 'attr_primary_core_rate')} | {pct('FD002', 'attr_primary_core_rate')} | 71.4 % |",
            self.docs,
        )

        def prose_pct(name, metric):
            value = subsets[name]["summary"]["baseline_x_gradient"][metric]
            if value["std"] == 0:
                return f"{100 * value['mean']:.1f}%"
            return f"{100 * value['mean']:.1f} ± {100 * value['std']:.1f}%"

        experiments_core = (
            f"Attribution puts the core subsystem first in "
            f"{prose_pct('FD001', 'attr_primary_core_rate')} / "
            f"{prose_pct('FD002', 'attr_primary_core_rate')} of windows."
        )
        experiments_hpc = (
            f"HPC is first in {prose_pct('FD001', 'attr_primary_hpc_rate')} / "
            f"{prose_pct('FD002', 'attr_primary_hpc_rate')} of windows"
        )
        self.assertIn(experiments_core, self.experiments)
        self.assertIn(experiments_hpc, self.experiments)

        drift_hpc = [
            100 * subsets[name]["summary"]["baseline_x_gradient"]["drift_primary_distribution"]["HPC"]
            for name in ("FD001", "FD002")
        ]
        self.assertIn(
            f"Model-free drift ranks HPC first in {drift_hpc[0]:.1f}% / {drift_hpc[1]:.1f}% of windows.",
            self.experiments,
        )

    def test_documented_clustering_metrics_match_json(self):
        f3 = self.report["subsets"]["FD003"]["summary"]
        f4 = self.report["subsets"]["FD004"]["summary"]
        self.assertIn(
            f"| drift (14 signed z) | silhouette | {f3['drift']['silhouette']['mean']:.2f} | "
            f"{f4['drift']['silhouette']['mean']:.2f} |",
            self.docs,
        )
        self.assertIn(
            f"| | cross-seed ARI (10 pairs) | {f3['drift']['cross_seed_ari']['mean']:.2f} | "
            f"{f4['drift']['cross_seed_ari']['mean']:.2f} ± {f4['drift']['cross_seed_ari']['std']:.2f} |",
            self.docs,
        )
        self.assertIn(
            f"| attribution (14 sensor shares) | silhouette | "
            f"{f3['attribution']['silhouette']['mean']:.2f} ± {f3['attribution']['silhouette']['std']:.2f} | "
            f"{f4['attribution']['silhouette']['mean']:.2f} ± {f4['attribution']['silhouette']['std']:.2f} |",
            self.docs,
        )
        self.assertIn(
            f"| | cross-seed ARI | {f3['attribution']['cross_seed_ari']['mean']:.2f} | "
            f"{f4['attribution']['cross_seed_ari']['mean']:.2f} ± "
            f"{f4['attribution']['cross_seed_ari']['std']:.2f} |",
            self.docs,
        )
        self.assertIn(
            f"| | KMeans vs GMM ARI | {f3['drift']['kmeans_vs_gmm_ari']['mean']:.2f} | "
            f"{f4['drift']['kmeans_vs_gmm_ari']['mean']:.2f} |",
            self.docs,
        )
        self.assertIn(
            f"| | KMeans vs GMM ARI | {f3['attribution']['kmeans_vs_gmm_ari']['mean']:.2f} | "
            f"{f4['attribution']['kmeans_vs_gmm_ari']['mean']:.2f} |",
            self.docs,
        )
        self.assertIn(
            f"| drift vs attribution clustering | cross-view ARI | "
            f"{f3['cross_view_ari_drift_vs_attribution']['mean']:.2f} | "
            f"{f4['cross_view_ari_drift_vs_attribution']['mean']:.2f} ± "
            f"{f4['cross_view_ari_drift_vs_attribution']['std']:.2f} |",
            self.docs,
        )
        self.assertIn(
            f"Silhouette {f3['drift']['silhouette']['mean']:.2f} / "
            f"{f4['drift']['silhouette']['mean']:.2f}. Cross-seed ARI "
            f"{f3['drift']['cross_seed_ari']['mean']:.2f} / {f4['drift']['cross_seed_ari']['mean']:.2f}.",
            self.experiments,
        )
        self.assertIn(
            f"Attribution-only clusters are stable on FD003 (ARI "
            f"{f3['attribution']['cross_seed_ari']['mean']:.2f}) but not on FD004 "
            f"({f4['attribution']['cross_seed_ari']['mean']:.2f} ± "
            f"{f4['attribution']['cross_seed_ari']['std']:.2f}).",
            self.experiments,
        )

    def test_documented_latency_and_autograd_bytes_match_json(self):
        latency = {name: self.report["subsets"][name]["latency"] for name in SUBSETS}
        rows = (
            ("inference only", "inference_only_ms_per_window"),
            ("+ attribution + summary", "inference_attribution_summary_ms_per_window"),
            ("+ drift recomputed from raw history + attribution + summary",
             "inference_attribution_drift_summary_ms_per_window"),
        )
        for label, key in rows:
            values = " | ".join(f"{latency[name][key]:.3f} ms" for name in SUBSETS)
            self.assertIn(f"| {label} | {values} |", self.docs)

        saved = {latency[name]["autograd_saved_tensor_bytes_per_window"] for name in SUBSETS}
        self.assertEqual(len(saved), 1)
        saved_bytes = saved.pop()
        self.assertIn(f"Autograd buffers add {saved_bytes:,} bytes per window", self.docs)
        inference = [latency[name]["inference_only_ms_per_window"] for name in SUBSETS]
        full = [latency[name]["inference_attribution_drift_summary_ms_per_window"] for name in SUBSETS]
        self.assertIn(
            f"Inference alone is {min(inference):.2f}–{max(inference):.2f} ms per window. "
            f"Inference + attribution + drift + summary is {min(full):.2f}–{max(full):.2f} ms per window. "
            f"Autograd buffers are {saved_bytes / 1000:.0f} KB per window.",
            self.experiments,
        )

    def test_protected_benchmark_artifacts_are_regular_and_unchanged(self):
        protected = [
            ROOT / "experiments/reproduce.py",
            ROOT / "results/reproducible/summary.csv",
            ROOT / "paper",
        ]
        for path in protected:
            self.assertTrue(path.exists(), path)
            self.assertFalse(path.is_symlink(), path)
            if path.is_dir():
                self.assertFalse(any(child.is_symlink() for child in path.rglob("*")), path)

        self.assertEqual(
            sha256(ROOT / "experiments/reproduce.py"),
            "0301de4cab66c12c5249ac6521adada3ae82b7241575061fe39867bd18a8a848",
        )
        self.assertEqual(
            sha256(ROOT / "results/reproducible/summary.csv"),
            "67fc4b14fc1660365b19976e6c0ed6598a2f16e928a8892f18c3a03dcd032e86",
        )
        self.assertEqual(
            paper_tree_sha256(),
            "b39b3f15a5b22d5c65a5edbb5975963efce3b10f30b21c24b68697b54e55b365",
        )


@unittest.skipUnless(HAS_CMAPSS, "set CMAPSS_ZIP to an existing CMAPSSData.zip for smoke tests")
class CmapssSmokeTest(unittest.TestCase):
    def run_command(self, command, timeout):
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return result.stdout

    def test_validate_diagnostics_smoke(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "diagnostics.json"
            self.run_command([
                sys.executable,
                "experiments/validate_diagnostics.py",
                "--subsets", "FD001", "FD003",
                "--seeds", "7",
                "--epochs", "1",
                "--threads", "1",
                "--out", str(output),
            ], timeout=1200)
            report = json.loads(output.read_text())
            self.assertEqual(report["config"]["subsets"], ["FD001", "FD003"])
            self.assertEqual(report["config"]["seeds"], [7])
            self.assertEqual(report["network_audit"]["total_socket_calls_during_diagnostics"], 0)
            self.assertEqual(report["network_audit"]["raw_sensor_bytes_sent"], 0)
            for subset in report["subsets"].values():
                self.assertEqual(subset["latency"]["threads"], 1)

    def test_reproducible_pipeline_quick_smoke_uses_temp_output(self):
        protected_before = {
            "runner": sha256(ROOT / "experiments/reproduce.py"),
            "summary": sha256(ROOT / "results/reproducible/summary.csv"),
            "paper": paper_tree_sha256(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "reproducible"
            self.run_command([
                sys.executable,
                "experiments/reproduce.py",
                "--data", str(CMAPSS_ZIP),
                "--out", str(output),
                "--quick",
                "--threads", "1",
            ], timeout=1800)
            raw = output / "fd001_iid_seed42.json"
            summary = output / "summary.csv"
            self.assertTrue(raw.is_file())
            self.assertTrue(summary.is_file())
            run = json.loads(raw.read_text())
            self.assertEqual(run["protocol"]["threads"], 1)
            self.assertEqual(run["dataset"], "FD001")
            self.assertEqual(run["seed"], 42)

        protected_after = {
            "runner": sha256(ROOT / "experiments/reproduce.py"),
            "summary": sha256(ROOT / "results/reproducible/summary.csv"),
            "paper": paper_tree_sha256(),
        }
        self.assertEqual(protected_after, protected_before)


if __name__ == "__main__":
    unittest.main()
