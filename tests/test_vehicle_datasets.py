"""Schema fixtures are synthetic; only CMAPSS_ZIP tests use a real archive."""

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
import torch

from common import AsymmetricRULLoss, CmapssRegressor, load_rul
from vehicle_datasets import (
    CmapssCombinedAdapter, NCMAPSSAdapter, SCANIA_FEATURE_COLS,
    ScaniaComponentXAdapter, VehicleFleetDataset, build_flexible_cnn,
    make_synthetic_scania_fleet,
)


def assert_disjoint(test, fleet, clients):
    ids = [v for vehicles in clients.values() for v in vehicles]
    test.assertEqual(len(ids), len(set(ids)))
    test.assertEqual(set(ids), set(fleet.vehicles))


class FlexibleCNNTests(unittest.TestCase):
    def test_baseline_parameter_and_forward_parity(self):
        baseline = CmapssRegressor().eval()
        model = build_flexible_cnn(17, 30).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 22_273)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 22_273)
        model.load_state_dict(baseline.state_dict(), strict=True)
        x = torch.randn(3, 30, 17)
        with torch.no_grad():
            self.assertEqual(model(x).shape, (3,))
            torch.testing.assert_close(model(x), baseline(x), rtol=0, atol=0)

    def test_arbitrary_channels_lengths_and_gradients(self):
        for channels, length in ((32, 30), (105, 20), (1, 4), (7, 31)):
            with self.subTest(channels=channels, length=length):
                model = build_flexible_cnn(channels, length)
                prediction = model(torch.randn(2, length, channels))
                self.assertEqual(prediction.shape, (2,))
                prediction.square().sum().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_invalid_dimensions(self):
        for channels, length in ((0, 30), (17, 3), (2, 0), (3.5, 30), (17, True)):
            with self.subTest(channels=channels, length=length), self.assertRaises(ValueError):
                build_flexible_cnn(channels, length)


class ScaniaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = make_synthetic_scania_fleet(self.temp.name)
        self.adapter = ScaniaComponentXAdapter(self.directory)

    def test_schema_censoring_windows_and_one_training_step(self):
        fleet = self.adapter.load()
        self.assertEqual(len(fleet.vehicles), 6)
        self.assertEqual(len(fleet.feature_cols), 105)
        self.assertEqual(fleet.setting_cols, [])
        self.assertIsNone(fleet.test_ruls)
        self.assertGreater(len({len(f) for f in fleet.vehicles.values()}), 1)
        self.assertTrue(any(f[fleet.feature_cols].isna().any().any() for f in fleet.vehicles.values()))
        self.assertFalse({"time_step", "tte_lower_bound", "rul", "in_study_repair", "Spec_0"} & set(fleet.feature_cols))
        clients = self.adapter.partition_by_specifications(fleet, ["Spec_0", "Spec_1"])
        self.assertEqual(len(clients), 2)
        assert_disjoint(self, fleet, clients)
        for ids in clients.values():
            self.assertEqual(len({fleet.metadata[v]["Spec_0"] for v in ids}), 1)
        expected_targets = []
        for vehicle, frame in fleet.vehicles.items():
            meta = fleet.metadata[vehicle]
            if meta["event_observed"]:
                np.testing.assert_allclose(frame.rul, meta["study_end"] - frame.time_step)
                expected_targets.extend(np.minimum(frame.rul.iloc[3:], 12.0))
            else:
                self.assertTrue(frame.rul.isna().all())
                self.assertTrue((frame.tte_lower_bound >= 0).all())
                x, y = fleet.windows([vehicle], seq_len=4)
                self.assertEqual(x.shape, (0, 4, 105))
                self.assertEqual(y.shape, (0,))
        with patch("vehicle_datasets.sanitize_telemetry_frame",
                   wraps=__import__("fed_norm").sanitize_telemetry_frame) as sanitizer:
            x, y = fleet.windows(seq_len=4, max_rul=12)
            self.assertEqual(sanitizer.call_count, 2 * len(y))
        self.assertEqual(x.shape, (len(expected_targets), 4, 105))
        np.testing.assert_allclose(y, expected_targets)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(y.dtype, np.float32)
        self.assertTrue(np.isfinite(x).all())
        # Disjoint clients' windows account for precisely the original population.
        self.assertEqual(sum(len(fleet.windows(ids, seq_len=4)[1]) for ids in clients.values()), len(y))
        model = build_flexible_cnn(105, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.conv1.weight.detach().clone()
        loss = AsymmetricRULLoss()(model(torch.from_numpy(x)), torch.from_numpy(y))
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.conv1.weight.detach()))

    def test_windows_are_causal_and_vehicle_local(self):
        fleet = self.adapter.load()
        first = "Scania-train-0"
        x, y = fleet.windows([first], seq_len=4)
        # All missing in an early window: neither future data nor other clients may fill it.
        fleet.vehicles[first].loc[:3, "167_0"] = np.nan
        early, _ = fleet.windows([first], seq_len=4)
        np.testing.assert_array_equal(early[0, :, 0], 0)
        fleet.vehicles[first].loc[4:, "167_0"] = 1e9
        fleet.vehicles["Scania-train-1"].loc[:, "167_0"] = -1e9
        after, _ = fleet.windows([first], seq_len=4)
        np.testing.assert_array_equal(early[0], after[0])
        self.assertEqual(x.shape[0], len(y))

    def test_endpoint_padding_stride_and_empty_output(self):
        fleet = self.adapter.load()
        x, y = fleet.windows(["Scania-train-0"], seq_len=20, last_only=True, max_rul=5)
        self.assertEqual(x.shape, (1, 20, 105))
        np.testing.assert_array_equal(x[0, 0], x[0, 15])
        np.testing.assert_array_equal(y, [5])
        all_x, all_y = fleet.windows(seq_len=4)
        subset_x, subset_y = fleet.windows(seq_len=4, stride=2)
        self.assertLess(len(subset_y), len(all_y))
        self.assertEqual(subset_x.shape[1:], all_x.shape[1:])
        x, y = fleet.windows([], seq_len=4)
        self.assertEqual(x.shape, (0, 4, 105))
        for kwargs in ({"seq_len": 0}, {"stride": 0}, {"max_rul": np.inf}, {"max_rul": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fleet.windows(**kwargs)
        with self.assertRaisesRegex(ValueError, "unique"):
            fleet.windows(["Scania-train-0"] * 2)

    def test_validation_classes_are_not_exact_rul(self):
        for suffix in ("operational_readouts", "specifications"):
            frame = pd.read_csv(self.directory / f"train_{suffix}.csv")
            frame.to_csv(self.directory / f"validation_{suffix}.csv", index=False)
        pd.DataFrame({"vehicle_id": range(6), "class_label": [0, 1, 2, 3, 4, 0]}).to_csv(
            self.directory / "validation_labels.csv", index=False)
        fleet = self.adapter.load(split="validation")
        self.assertEqual(fleet.metadata["Scania-validation-4"]["class_label"], 4)
        self.assertTrue(all(f.rul.isna().all() for f in fleet.vehicles.values()))
        self.assertEqual(fleet.windows(seq_len=4)[0].shape, (0, 4, 105))

    def test_rejects_missing_schema_bad_ids_and_impossible_outcomes(self):
        path = self.directory / "train_tte.csv"
        original = pd.read_csv(path)
        for bad in (original.drop(columns="in_study_repair"),
                    pd.concat([original, original.iloc[:1]]), original.iloc[1:],
                    original.assign(in_study_repair=2),
                    original.assign(length_of_study_time_step=0)):
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                self.adapter.load()
        original.to_csv(path, index=False)
        path = self.directory / "train_operational_readouts.csv"
        readouts = pd.read_csv(path)
        for bad in (readouts.assign(time_step=np.nan),
                    pd.concat([readouts, readouts.iloc[:1]]),
                    readouts.drop(columns=SCANIA_FEATURE_COLS[0])):
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                self.adapter.load()
        with self.assertRaisesRegex(FileNotFoundError, "researchdata.se"):
            ScaniaComponentXAdapter(self.directory / "absent").load()
        with self.assertRaises(ValueError):
            self.adapter.load(split="invalid")
        for cols in ([], ["Spec_9"], ["Spec_0", "Spec_0"]):
            with self.assertRaises(ValueError):
                self.adapter.partition_by_specifications(VehicleFleetDataset("empty", {}, [], [], {}), cols)

    def test_generator_is_deterministic(self):
        with TemporaryDirectory() as directory:
            make_synthetic_scania_fleet(directory)
            for file in self.directory.iterdir():
                self.assertEqual(file.read_bytes(), (Path(directory) / file.name).read_bytes())


class NCMAPSSAdapterTests(unittest.TestCase):
    def test_missing_file_and_optional_dependency_messages(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "N-CMAPSS_DS02.h5"
            with patch.dict("sys.modules", {"h5py": None}):
                with self.assertRaisesRegex(FileNotFoundError, "NASA repository item 17"):
                    NCMAPSSAdapter(path).load()
                path.touch()
                with self.assertRaisesRegex(ImportError, "pip install h5py"):
                    NCMAPSSAdapter(path).load()

    def test_flight_class_metadata_partition_without_h5py(self):
        fleet = VehicleFleetDataset("synthetic", {str(i): pd.DataFrame() for i in range(6)}, [], [],
                                    {str(i): {"Fc": i % 3 + 1} for i in range(6)})
        clients = NCMAPSSAdapter.partition_by_flight_class(fleet)
        self.assertEqual(clients, {"1": ["0", "3"], "2": ["1", "4"], "3": ["2", "5"]})
        assert_disjoint(self, fleet, clients)
        with self.assertRaises(KeyError):
            fleet.partition_by("nonexistent")

    @staticmethod
    def write_h5(path):
        """Tiny synthetic OFFICIAL-layout HDF5 fixture; no NASA measurements."""
        import h5py
        with h5py.File(path, "w") as handle:
            variable_names = {"W": ["alt", "Mach", "TRA", "T2"],
                              "X_s": ["Wf", "Nf"], "X_v": ["P30"],
                              "T": ["fan_eff_mod", "HPC_flow_mod"],
                              "A": ["unit", "cycle", "Fc", "hs"]}
            for group, names in variable_names.items():
                handle[f"{group}_var"] = np.array(names, dtype="S32")
            for split in ("dev", "test"):
                a = np.array([[u, 1 + row // 3, u, row < 3]
                              for u in (1, 2, 3) for row in range(6)], dtype=float)
                for group, names in variable_names.items():
                    handle[f"{group}_{split}"] = (a if group == "A" else
                        np.arange(18 * len(names), dtype=float).reshape(18, len(names)))
                handle[f"Y_{split}"] = np.tile([10, 10, 10, 9, 9, 9], 3).reshape(-1, 1)

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "optional h5py not installed")
    def test_hdf5_loading_health_truth_feature_isolation_and_chunking(self):
        import h5py
        with TemporaryDirectory() as directory:
            path = Path(directory) / "N-CMAPSS_synthetic.h5"
            self.write_h5(path)
            adapter = NCMAPSSAdapter(path)
            fleet = adapter.load(chunk_rows=4)
            self.assertEqual(len(fleet.vehicles), 3)
            self.assertEqual(len(fleet.feature_cols), 6)
            self.assertEqual(len(fleet.setting_cols), 4)
            assert_disjoint(self, fleet, adapter.partition_by_flight_class(fleet))
            self.assertEqual(set(fleet.partition_by("flight_class")), {"short", "medium", "long"})
            self.assertIsNone(fleet.test_ruls)
            for vehicle, frame in fleet.vehicles.items():
                self.assertEqual(len(frame), 6)
                self.assertIn("T:fan_eff_mod", frame)
                self.assertIn("X_v:P30", frame)
                self.assertEqual(fleet.metadata[vehicle]["health_parameter_cols"], ["T:fan_eff_mod", "T:HPC_flow_mod"])
                self.assertFalse(any(c.startswith(("T:", "X_v:")) or c in ("hs", "Fc", "rul", "cycle", "unit")
                                     for c in fleet.feature_cols))
            x, y = fleet.windows(seq_len=4, max_rul=8)
            self.assertEqual(x.shape, (9, 4, 6))
            np.testing.assert_array_equal(y, 8)
            virtual = adapter.load(split="test", include_virtual=True, row_stride=2, chunk_rows=2)
            self.assertEqual(len(virtual.feature_cols), 7)
            self.assertEqual(len(virtual.test_ruls), 3)
            self.assertTrue(all(len(f) == 3 for f in virtual.vehicles.values()))
            for vehicle, endpoint in virtual.test_ruls.items():
                self.assertEqual(endpoint, virtual.vehicles[vehicle].rul.iloc[-1])
            # Chunk boundaries must not alter retained samples.
            again = adapter.load(row_stride=2, chunk_rows=11)
            for vehicle in again.vehicles:
                pd.testing.assert_frame_equal(again.vehicles[vehicle], adapter.load(row_stride=2, chunk_rows=2).vehicles[vehicle])
            with h5py.File(path, "a") as handle:
                del handle["T_dev"]
            with self.assertRaisesRegex(ValueError, "T_dev"):
                adapter.load()

    @unittest.skipUnless(importlib.util.find_spec("h5py"), "optional h5py not installed")
    def test_rejects_misaligned_hdf5_and_changing_flight_class(self):
        import h5py
        with TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.h5"
            self.write_h5(path)
            with h5py.File(path, "a") as handle:
                del handle["Y_dev"]
                handle["Y_dev"] = np.zeros((2, 1))
            with self.assertRaisesRegex(ValueError, "aligned"):
                NCMAPSSAdapter(path).load()
            self.write_h5(path)
            with h5py.File(path, "a") as handle:
                handle["A_dev"][0, 2] = 2
            with self.assertRaisesRegex(ValueError, "flight class constant"):
                NCMAPSSAdapter(path).load()


class CmapssAdapterValidationTests(unittest.TestCase):
    def test_rejects_invalid_subset_or_split_before_reading(self):
        for subsets, split in (([], "train"), (["FD001"] * 2, "train"),
                               ("FD009", "train"), (None, "dev")):
            with self.subTest(subsets=subsets, split=split), self.assertRaises(ValueError):
                CmapssCombinedAdapter().load(subsets, split=split)


@unittest.skipUnless(os.environ.get("CMAPSS_ZIP") and Path(os.environ["CMAPSS_ZIP"]).is_file(),
                     "set CMAPSS_ZIP to an existing CMAPSSData.zip")
class CmapssArchiveIntegrationTests(unittest.TestCase):
    def test_archive_counts_collision_free_splits_and_non_iid(self):
        # Count raw archive IDs independently of pandas/common/the adapter.
        counts = {}
        with zipfile.ZipFile(os.environ["CMAPSS_ZIP"]) as archive:
            for subset in CmapssCombinedAdapter.SUBSETS:
                for split in ("train", "test"):
                    with archive.open(f"{split}_{subset}.txt") as handle:
                        counts[(subset, split)] = len({line.split()[0] for line in handle if line.strip()})
        self.assertEqual([counts[(s, "train")] for s in CmapssCombinedAdapter.SUBSETS], [100, 260, 100, 249])
        self.assertEqual([counts[(s, "test")] for s in CmapssCombinedAdapter.SUBSETS], [100, 259, 100, 248])
        adapter = CmapssCombinedAdapter()
        fleet = adapter.load(split="both")
        splits = fleet.partition_by("split")
        self.assertEqual(len(splits["train"]), 709)
        self.assertEqual(len(splits["test"]), 707)
        self.assertEqual(len(fleet.vehicles), 1416)
        assert_disjoint(self, fleet, splits)
        clients = fleet.partition_by("subset")
        assert_disjoint(self, fleet, clients)
        self.assertEqual(len(fleet.test_ruls), 707)
        for subset, ids in clients.items():
            self.assertEqual(len(ids), counts[(subset, "train")] + counts[(subset, "test")])
            for vehicle in ids:
                meta, frame = fleet.metadata[vehicle], fleet.vehicles[vehicle]
                self.assertTrue(vehicle.startswith(f"{subset}-{meta['split']}-U"))
                self.assertTrue(frame.cycle.is_monotonic_increasing)
                self.assertEqual(meta["operating_conditions"], 1 if subset in ("FD001", "FD003") else 6)
                self.assertEqual(meta["fault_mode_count"], 1 if subset in ("FD001", "FD002") else 2)
                expected_endpoint = 0 if meta["split"] == "train" else fleet.test_ruls[vehicle]
                self.assertEqual(frame.rul.iloc[-1], expected_endpoint)
        # All official labels survive mapping; no clipping until window creation.
        for subset in CmapssCombinedAdapter.SUBSETS:
            labels = load_rul(f"RUL_{subset}.txt")
            for unit, endpoint in enumerate(labels, 1):
                self.assertEqual(fleet.test_ruls[f"{subset}-test-U{unit:03d}"], endpoint)
        fd004 = adapter.load("FD004")
        self.assertEqual(len(fd004.vehicles), 249)
        self.assertIsNone(fd004.test_ruls)
        test = adapter.load("FD004", split="test")
        self.assertEqual(len(test.vehicles), 248)
        self.assertTrue(set(fd004.vehicles).isdisjoint(test.vehicles))
        x, y = test.windows(seq_len=30, last_only=True)
        self.assertEqual(x.shape, (248, 30, 17))
        self.assertTrue(np.isfinite(x).all())
        np.testing.assert_allclose(y, [min(v, 125) for v in test.test_ruls.values()])


if __name__ == "__main__":
    unittest.main()
