from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import CmapssRegressor, SEQ_LEN, USEFUL_SENSORS
from fed_norm import compute_local_stats
from service_bay_fl import (ServiceBayCoordinator, VehicleEdgeCheckpoint,
                            personalize_at_service_bay, staleness_weight)


class ServiceBayPrimitiveTests(unittest.TestCase):
    def test_checkpoint_round_trip_has_only_local_sufficient_stats(self):
        model = CmapssRegressor().state_dict()
        stats = compute_local_stats(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        checkpoint = VehicleEdgeCheckpoint("car-17", 5, model,
                                           {k: torch.zeros_like(v) for k, v in model.items()}, stats, 400)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vehicle.pt"
            checkpoint.save(path)
            loaded = VehicleEdgeCheckpoint.load(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(loaded.vehicle_id, "car-17")
        self.assertEqual(loaded.last_synced_round, 5)
        self.assertEqual(loaded.local_training_budget, 400)
        self.assertEqual(payload["raw_sensor_arrays"], 0)
        self.assertNotIn("sensor_rows", payload)
        self.assertTrue(torch.equal(loaded.model_weights["fc2.weight"], model["fc2.weight"]))
        np.testing.assert_array_equal(loaded.local_stats.mean, stats.mean)

    def test_staleness_weight_monotonicity(self):
        weights = [staleness_weight(100, tau) for tau in (0, 1, 3)]
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_fedavgm_and_scaffold_updates(self):
        start = {"w": torch.tensor([1.0])}
        coordinator = ServiceBayCoordinator(start, beta=0.9)
        cp = VehicleEdgeCheckpoint("v", 0, {"w": start["w"].clone()},
                                   {"w": torch.tensor([0.0])})
        coordinator.round = 1
        coordinator.aggregate([(cp, {"w": torch.tensor([3.0])}, 1)])
        self.assertAlmostEqual(coordinator.global_weights["w"].item(), 3.0)
        stale = VehicleEdgeCheckpoint("old", 0, {"w": torch.tensor([1.0])},
                                      {"w": torch.tensor([0.0])})
        coordinator.round = 3
        coordinator.global_weights["w"] = torch.tensor([5.0])
        coordinator.momentum["w"] = torch.tensor([0.0])
        coordinator.aggregate([(stale, {"w": torch.tensor([3.0])}, 1)])
        self.assertAlmostEqual(coordinator.global_weights["w"].item(), 7.0)
        cp2 = VehicleEdgeCheckpoint("v", 0, {"w": torch.tensor([1.0])},
                                    {"w": torch.tensor([0.0])})
        coord2 = ServiceBayCoordinator({"w": torch.tensor([1.0])}, scaffold=True)
        coord2.update_scaffold(cp2, {"w": torch.tensor([1.0])},
                              {"w": torch.tensor([0.7])}, local_steps=3, learning_rate=0.1)
        self.assertAlmostEqual(cp2.c_i["w"].item(), 1.0)
        untouched = VehicleEdgeCheckpoint("other", 0, {"w": torch.tensor([1.0])},
                                          {"w": torch.tensor([0.0])})
        coord2.refresh_global_control([cp2, untouched])
        self.assertAlmostEqual(coord2.global_c["w"].item(), 0.5)

    def test_personalization_changes_private_head_only(self):
        state = CmapssRegressor().state_dict()
        original = {k: v.clone() for k, v in state.items()}
        x = np.random.RandomState(1).normal(size=(6, SEQ_LEN, len(USEFUL_SENSORS))).astype(np.float32)
        y = np.linspace(10, 100, 6, dtype=np.float32)
        reference = CmapssRegressor(); reference.load_state_dict(state); reference.eval()
        with torch.no_grad():
            before_loss = torch.nn.functional.mse_loss(reference(torch.as_tensor(x)), torch.as_tensor(y)).item()
        personalized = personalize_at_service_bay(state, x, y, seed=2, epochs=2,
                                                  learning_rate=0.01, batch_size=6)
        reference.load_state_dict(personalized)
        with torch.no_grad():
            after_loss = torch.nn.functional.mse_loss(reference(torch.as_tensor(x)), torch.as_tensor(y)).item()
        self.assertFalse(torch.equal(personalized["fc2.weight"], original["fc2.weight"]))
        self.assertLess(after_loss, before_loss)
        for key in original:
            self.assertTrue(torch.equal(state[key], original[key]), key)
            if key.startswith("conv"):
                self.assertTrue(torch.equal(personalized[key], original[key]), key)

    def test_training_flattens_column_targets_to_match_model_output(self):
        from experiments.service_bay_benchmark import _train
        state = CmapssRegressor().state_dict()
        x = np.random.RandomState(3).normal(
            size=(4, SEQ_LEN, len(USEFUL_SENSORS))).astype(np.float32)
        y = np.asarray([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)
        updated = _train(state, x, y, seed=3, epochs=1, lr=1e-3, batch_size=4)
        self.assertTrue(all(torch.isfinite(value).all() for value in updated.values()))


class ServiceBayScheduleTests(unittest.TestCase):
    def test_vehicle_disjointness_and_partial_visits(self):
        from experiments.vehicle_swarm_benchmark import (VehicleProfile, assert_disjoint_partitions,
                                                         partition_vehicles)
        profiles = {i: VehicleProfile(100 + i, i % 3, i % 2) for i in range(1, 31)}
        partition = partition_vehicles(list(profiles), 5, "iid_swarm", profiles=profiles, seed=9)
        assert_disjoint_partitions(partition, profiles)
        from experiments.service_bay_benchmark import _schedule_staleness, _visit_schedule
        schedule = _visit_schedule(10, 0.3, 7, 20)
        self.assertTrue(all(len(visit) == 3 for visit in schedule))
        self.assertTrue(all(len(visit) < 10 for visit in schedule))
        trace = _schedule_staleness(schedule)
        self.assertTrue(all(row["tau"] == row["cycle"] - row["last_synced_round"] for row in trace))
        self.assertGreater(max(row["tau"] for row in trace), 1)

    @unittest.skipUnless(os.environ.get("CMAPSS_ZIP"), "CMAPSS_ZIP is not configured")
    def test_fd002_one_seed_smoke_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            command = [sys.executable, str(ROOT / "experiments/service_bay_benchmark.py"),
                       "--output", str(output), "--seeds", "7", "--cycles", "1",
                       "--local-epochs", "1", "--personalization-epochs", "1",
                       "--configs", "FD002-K10", "--schemes", "iid_swarm", "--batch-size", "256"]
            completed = subprocess.run(command, cwd=ROOT, env=os.environ.copy(),
                                       capture_output=True, text=True, timeout=900)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(output.is_file())
            import json
            result = json.loads(output.read_text())
            self.assertEqual(result["seeds"], [7])
            config = result["configurations"][0]
            self.assertEqual(config["per_seed"][0]["methods"]["fedavgm_staleness"]["visits"][0]["raw_sensor_rows_uploaded"], 0)


if __name__ == "__main__":
    unittest.main()
