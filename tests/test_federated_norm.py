import os
import unittest

import numpy as np
import pandas as pd

from common import SEQ_LEN, USEFUL_SENSORS
from fed_norm import (aggregate_federated_stats, compare_federated_vs_centralized,
                      compute_local_stats, sanitize_telemetry_frame)


class FederatedStatsTests(unittest.TestCase):
    def _check_partition(self, sizes):
        rng = np.random.RandomState(731)
        clients = [rng.normal(loc=i * 2.5, scale=3.0, size=(n, 9, 5)) for i, n in enumerate(sizes)]
        raw = np.concatenate(clients, axis=0)
        stats = [compute_local_stats(part) for part in clients]
        scaler = aggregate_federated_stats(stats, sensors=[f"s{i}" for i in range(5)])
        # The aggregator accepts only LocalStats and does not retain client arrays.
        self.assertTrue(all(type(stat).__name__ == "LocalStats" for stat in stats))
        with self.assertRaises(TypeError):
            aggregate_federated_stats(clients)
        np.testing.assert_allclose(scaler.mean, np.mean(raw, axis=(0, 1)), atol=1e-6, rtol=0)
        np.testing.assert_allclose(scaler.std, np.std(raw, axis=(0, 1), ddof=0), atol=1e-6, rtol=0)
        np.testing.assert_allclose(scaler.min, np.min(raw, axis=(0, 1)), atol=1e-6, rtol=0)
        np.testing.assert_allclose(scaler.max, np.max(raw, axis=(0, 1)), atol=1e-6, rtol=0)
        check = compare_federated_vs_centralized(clients)
        self.assertTrue(check["equivalent"])

    def test_chan_merge_two_clients(self):
        self._check_partition([23, 41])

    def test_chan_merge_three_clients(self):
        self._check_partition([11, 37, 19])

    def test_chan_merge_uneven_clients(self):
        self._check_partition([3, 101, 17])


class TelemetrySanitizationTests(unittest.TestCase):
    def test_missing_infinite_leading_trailing_and_spikes(self):
        frame = pd.DataFrame({
            "unit": [1] * 5,
            "a": [np.nan, 2.0, np.inf, 1000.0, np.nan],
            "b": [-np.inf, 4.0, 5.0, 6.0, np.nan],
        })
        clean, report = sanitize_telemetry_frame(
            frame, sensors=["a", "b", "dropped"],
            healthy_means={"a": 3.0, "b": 5.0, "dropped": 7.0},
            clip_bounds={"a": (0.0, 10.0), "b": (0.0, 10.0), "dropped": (0.0, 10.0)},
        )
        self.assertEqual(clean.dtype, np.float32)
        self.assertEqual(clean.shape, (5, 3))
        self.assertTrue(np.isfinite(clean).all())
        np.testing.assert_allclose(clean[:, 0], [2.0, 2.0, 2.0, 10.0, 10.0])
        np.testing.assert_allclose(clean[:, 1], [4.0, 4.0, 5.0, 6.0, 6.0])
        np.testing.assert_allclose(clean[:, 2], 7.0)
        self.assertAlmostEqual(report["missing_fraction"], 10 / 15)
        self.assertAlmostEqual(report["clipped_fraction"], 2 / 15)
        self.assertEqual(report["imputed_channels"], ["a", "b", "dropped"])

    def test_frozen_channel_uses_healthy_fallback(self):
        frame = pd.DataFrame({"unit": [1, 1, 1, 2, 2, 2],
                              "sensor": [9.0, 9.0, 9.0, 1.0, 2.0, 3.0]})
        clean, report = sanitize_telemetry_frame(frame, sensors=["sensor"], healthy_means={"sensor": 2.5})
        np.testing.assert_array_equal(clean[:, 0], np.array([2.5, 2.5, 2.5, 1.0, 2.0, 3.0], dtype=np.float32))
        self.assertEqual(report["imputed_channels"], ["sensor"])


def _has_cmapss():
    path = os.environ.get("CMAPSS_ZIP")
    return bool(path and os.path.isfile(path))


@unittest.skipUnless(_has_cmapss(), "set CMAPSS_ZIP to an existing CMAPSSData.zip")
class CmapssParityTests(unittest.TestCase):
    def test_fd002_federated_minmax_and_two_epoch_model_parity(self):
        import torch

        import common
        from benchmark_asymmetric_loss import compute_scales_from_train
        from fed_norm import FederatedScaler

        train = common.load_cmapss_raw("train_FD002.txt")
        central = compute_scales_from_train(train)
        # Match the centralized benchmark's first-50-cycles-per-unit bounds.
        local_arrays = []
        for client_id in range(3):
            units = sorted(train.unit.unique())[client_id::3]
            local = train[train.unit.isin(units)].groupby("unit", sort=False).head(50)
            local_arrays.append(local[USEFUL_SENSORS].to_numpy(dtype=np.float64))
        scaler = aggregate_federated_stats(
            [compute_local_stats(part) for part in local_arrays], mode="minmax",
            clip_range=(0.0, 1.0), sensors=USEFUL_SENSORS,
        )
        np.testing.assert_allclose(scaler.min, [central["min"][s] for s in USEFUL_SENSORS], atol=1e-6, rtol=0)
        np.testing.assert_allclose(scaler.max, [central["max"][s] for s in USEFUL_SENSORS], atol=1e-6, rtol=0)

        # Evaluate both transforms with identical initialization and two epochs.
        test_raw = common.load_cmapss_raw("test_FD002.txt")
        rul = common.load_rul("RUL_FD002.txt")
        errors = {}
        for mode in ("zscore", "minmax"):
            model_scaler = FederatedScaler(scaler.mean, scaler.std, scaler.min, scaler.max,
                                           mode=mode, clip_range=(-5.0, 5.0) if mode == "zscore" else (0.0, 1.0))
            train_scaled = train.copy()
            train_scaled[USEFUL_SENSORS] = train_scaled[USEFUL_SENSORS].astype(np.float64)
            train_scaled[USEFUL_SENSORS] = model_scaler.transform(train[USEFUL_SENSORS].to_numpy(), clip=False)
            x_train, y_train = common.make_xy(train_scaled, sorted(train.unit.unique()))
            x_test, y_test = [], []
            for unit, group in test_raw.groupby("unit"):
                values = model_scaler.transform(group[USEFUL_SENSORS].to_numpy(), clip=False)
                if len(values) < SEQ_LEN:
                    values = np.concatenate([np.repeat(values[:1], SEQ_LEN - len(values), axis=0), values])
                x_test.append(values[-SEQ_LEN:])
                y_test.append(min(common.MAX_RUL, rul[int(unit) - 1]))
            common.set_seed(42)
            model = common.train_epochs(common.CmapssRegressor(),
                                        common.make_loader(x_train, y_train, 64, seed=42), epochs=2)
            with torch.no_grad():
                pred = model(torch.as_tensor(np.asarray(x_test, dtype=np.float32))).numpy()
            errors[mode] = common.rmse(y_test, pred)
        self.assertTrue(np.isfinite(list(errors.values())).all())
        self.assertLess(max(errors.values()), 2.5 * max(min(errors.values()), 1e-6))


if __name__ == "__main__":
    unittest.main()
