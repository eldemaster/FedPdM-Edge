import os
import unittest
from unittest import mock

import numpy as np

from fed_norm import LocalStats
from experiments import vehicle_swarm_benchmark as swarm


class VehiclePartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.profiles = swarm.make_synthetic_swarm(120, seed=13)
        cls.units = list(cls.profiles)

    def test_every_partition_is_exhaustive_and_pairwise_disjoint(self):
        for k in (10, 20):
            for scheme in swarm.PARTITION_SCHEMES:
                with self.subTest(k=k, scheme=scheme):
                    partitions = swarm.partition_vehicles(
                        self.units, k, scheme, profiles=self.profiles, seed=7)
                    self.assertEqual(k, len(partitions))
                    self.assertGreaterEqual(min(map(len, partitions.values())), 2)
                    self.assertEqual(len(self.units), sum(map(len, partitions.values())))
                    for left in range(k):
                        for right in range(left + 1, k):
                            self.assertFalse(set(partitions[left]) & set(partitions[right]))
                    swarm.assert_disjoint_partitions(partitions, self.units)

    def test_non_iid_schemes_have_strictly_more_measured_skew_than_iid(self):
        metrics = {}
        for scheme in swarm.PARTITION_SCHEMES:
            partitions = swarm.partition_vehicles(
                self.units, 10, scheme, profiles=self.profiles, seed=7)
            metrics[scheme] = swarm.partition_heterogeneity(partitions, self.profiles)
        iid = metrics["iid_swarm"]["heterogeneity_score"]
        self.assertGreater(metrics["lifespan_quantity_skew"]["heterogeneity_score"], iid)
        self.assertGreater(metrics["regime_fault_skew"]["heterogeneity_score"], iid)
        self.assertGreater(
            metrics["lifespan_quantity_skew"]["lifespan_pairwise_js"],
            metrics["iid_swarm"]["lifespan_pairwise_js"],
        )
        self.assertGreater(
            metrics["regime_fault_skew"]["dominant_regime_pairwise_js"],
            metrics["iid_swarm"]["dominant_regime_pairwise_js"],
        )

    def test_partial_participation_is_exact_and_deterministic(self):
        for k, fraction, expected in ((10, 0.5, 5), (20, 0.3, 6), (3, 0.01, 1)):
            for round_index in range(1, 5):
                selected = swarm.select_round_clients(k, fraction, 42, round_index)
                self.assertEqual(expected, len(selected))
                self.assertEqual(selected, swarm.select_round_clients(k, fraction, 42, round_index))
                self.assertEqual(len(selected), len(set(selected)))
                self.assertTrue(all(0 <= client_id < k for client_id in selected))
        self.assertNotEqual(
            swarm.select_round_clients(10, 0.5, 42, 1),
            swarm.select_round_clients(10, 0.5, 42, 2),
        )


class DecentralizedNormalizationTest(unittest.TestCase):
    def test_server_aggregator_receives_only_local_stats(self):
        frame, profiles = swarm.make_synthetic_swarm(20, seed=5)
        partitions = swarm.partition_vehicles(
            list(profiles), 4, "iid_swarm", profiles=profiles, seed=5)
        frames = [frame[frame["unit"].isin(partitions[i])] for i in range(4)]
        observed = []
        original = swarm.aggregate_federated_stats

        def capture(stats, **kwargs):
            observed.extend(stats)
            return original(stats, **kwargs)

        with mock.patch.object(swarm, "aggregate_federated_stats", side_effect=capture):
            scaler, audit = swarm.build_decentralized_scaler(frames)
        self.assertEqual(4, len(observed))
        self.assertTrue(all(isinstance(value, LocalStats) for value in observed))
        self.assertEqual(0, audit["raw_sensor_rows_shared"])
        self.assertEqual("LocalStats", audit["aggregation_input_type"])
        transformed = scaler.transform(frames[0][swarm.USEFUL_SENSORS].to_numpy())
        self.assertTrue(np.isfinite(transformed).all())


class SwarmMethodSmokeTest(unittest.TestCase):
    def test_all_methods_run_and_return_finite_metrics(self):
        results = swarm.run_synthetic_smoke(seed=7)
        self.assertEqual(set(swarm.METHODS), set(results))
        required = {
            "rmse", "nasa_score", "late_prediction_rate",
            "client_worst_rmse", "client_p90_rmse",
        }
        for method, result in results.items():
            with self.subTest(method=method):
                self.assertTrue(required.issubset(result))
                self.assertTrue(all(np.isfinite(result[key]) for key in required))
        personalization = results["fedavg_asym_personalized"]["personalization"]
        self.assertIn("before", personalization)
        self.assertIn("after", personalization)
        self.assertTrue(np.isfinite(personalization["client_mean_rmse_delta"]))


@unittest.skipUnless(os.environ.get("CMAPSS_ZIP"), "CMAPSS_ZIP is not set")
class CmapssVehicleSwarmSmokeTest(unittest.TestCase):
    def test_fd002_real_data_partition_normalization_and_one_round(self):
        train, test, ruls, profiles = swarm.load_fleet_data("FD002", seed=3)
        self.assertEqual(260, len(profiles))
        partitions = swarm.partition_vehicles(
            list(profiles), 10, "regime_fault_skew", profiles=profiles, seed=3)
        swarm.assert_disjoint_partitions(partitions, profiles)
        frames = [train[train["unit"].isin(partitions[i])] for i in range(10)]
        scaler, audit = swarm.build_decentralized_scaler(frames)
        self.assertEqual(0, audit["raw_sensor_rows_shared"])
        cache = swarm._unit_sequences(swarm._scale_frame(train, scaler))
        clients = swarm.build_client_data(
            partitions, cache, seed=3, max_train_sequences=4, max_val_sequences=2)
        test_x, test_y = swarm._official_test_set(swarm._scale_frame(test, scaler), ruls)
        results = swarm.run_all_methods(
            clients, test_x[:8], test_y[:8], seed=3, fraction_fit=0.5,
            rounds=1, local_epochs=1, personalization_epochs=1, batch_size=4)
        self.assertTrue(all(swarm._finite_method_metrics(result) for result in results.values()))
        for method in ("fedavg_mse", "fedprox_mse", "fedavg_asym", "fedprox_asym"):
            self.assertEqual(5, results[method]["round_history"][0]["selected_count"])


if __name__ == "__main__":
    unittest.main()
