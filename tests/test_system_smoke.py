"""System smoke tests: training core, simulation, edge profiling, and paper tables.

Synthetic checks always run. Checks that need the NASA archive run only when
``CMAPSSData.zip`` is found ($CMAPSS_ZIP, ./, or ~/). Every check that writes
files does so in a temporary directory, so tracked files stay unchanged.
"""
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import common as C  # noqa: E402
import fed_sim  # noqa: E402
import sanity_check  # noqa: E402


def _cmapss_zip():
    try:
        return os.path.abspath(C.cmapss_zip_path())
    except FileNotFoundError:
        return None


CMAPSS_ZIP = _cmapss_zip()
NEEDS_DATA = "CMAPSSData.zip not found (set CMAPSS_ZIP=/path/to/CMAPSSData.zip)"


@contextlib.contextmanager
def _cwd(path):
    """Run code that loads repo files by relative path (global_scale.json, partition.json)."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _finite(value):
    return all(math.isfinite(float(v)) for v in np.ravel(value))


# --------------------------------------------------------------------------- common.py
class RegressorTest(unittest.TestCase):
    def test_forward_shape_and_finite(self):
        C.set_seed(0)
        model = C.CmapssRegressor()
        out = model(torch.randn(5, C.SEQ_LEN, len(C.USEFUL_SENSORS)))
        self.assertEqual(tuple(out.shape), (5,))
        self.assertTrue(torch.isfinite(out).all())

    def test_parameter_count(self):
        # conv1 17*32*3+32, conv2 32*64*3+64, fc1 (64*7)*32+32, fc2 32+1
        expected = 1664 + 6208 + 14368 + 33
        self.assertEqual(sum(p.numel() for p in C.CmapssRegressor().parameters()), expected)

    def test_seed_makes_init_deterministic(self):
        C.set_seed(7)
        a = C.CmapssRegressor().state_dict()
        C.set_seed(7)
        b = C.CmapssRegressor().state_dict()
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)


class LossTest(unittest.TestCase):
    def test_late_penalized_more_than_early(self):
        loss = C.AsymmetricRULLoss()
        y = torch.tensor([50.0, 80.0])
        late = loss(y + 10.0, y)
        early = loss(y - 10.0, y)
        self.assertGreater(late.item(), early.item())
        self.assertAlmostEqual(late.item(), 2.5 * 100.0, places=4)
        self.assertAlmostEqual(early.item(), 0.5 * 100.0, places=4)
        self.assertEqual(loss(y, y).item(), 0.0)

    def test_gradients_finite_on_large_errors(self):
        loss = C.AsymmetricRULLoss()
        pred = torch.tensor([1e3, -1e3, 0.0], requires_grad=True)
        loss(pred, torch.zeros(3)).backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        # late side has the larger gradient magnitude at equal |d|
        self.assertGreater(abs(pred.grad[0].item()), abs(pred.grad[1].item()))

    def test_invalid_alphas_rejected(self):
        for kwargs in ({"alpha_late": 0.5, "alpha_early": 2.5}, {"alpha_late": 1.0, "alpha_early": 1.0},
                       {"alpha_late": 2.0, "alpha_early": 0.0}, {"alpha_late": -1.0, "alpha_early": 0.5}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                C.AsymmetricRULLoss(**kwargs)

    def test_get_loss_fn(self):
        self.assertIsInstance(C.get_loss_fn(), torch.nn.MSELoss)
        self.assertIsInstance(C.get_loss_fn(None), torch.nn.MSELoss)
        self.assertIsInstance(C.get_loss_fn(" MSE "), torch.nn.MSELoss)
        for name in ("asymmetric", "ASYM", "asymmetric_rul"):
            self.assertIsInstance(C.get_loss_fn(name), C.AsymmetricRULLoss, name)
        custom = C.get_loss_fn("asymmetric", alpha_late=3.0, alpha_early=1.0)
        self.assertEqual((custom.alpha_late, custom.alpha_early), (3.0, 1.0))
        with self.assertRaisesRegex(ValueError, "Unknown loss function"):
            C.get_loss_fn("huber")


class MetricTest(unittest.TestCase):
    def test_nasa_score(self):
        self.assertEqual(C.nasa_score([10, 20], [10, 20]), 0.0)
        self.assertAlmostEqual(C.nasa_score([0], [10]), math.e - 1, places=12)        # late: exp(d/10)-1
        self.assertAlmostEqual(C.nasa_score([10], [0]), math.exp(10 / 13) - 1, places=12)  # early: exp(-d/13)-1
        self.assertGreater(C.nasa_score([0], [10]), C.nasa_score([10], [0]))
        self.assertIsInstance(C.nasa_score(np.array([1.0]), np.array([2.0])), float)

    def test_late_prediction_rate(self):
        self.assertEqual(C.late_prediction_rate([10, 10, 10, 10], [11, 10, 9, 12]), 0.5)
        self.assertEqual(C.late_prediction_rate([1, 2], [1, 2]), 0.0)
        self.assertEqual(C.late_prediction_rate([1, 2], [5, 6]), 1.0)


class SequenceTest(unittest.TestCase):
    @staticmethod
    def _engine(n_cycles):
        rows = {s: np.arange(n_cycles, dtype=float) + 100 * i for i, s in enumerate(C.USEFUL_SENSORS)}
        return pd.DataFrame({"unit": 1, "cycle": np.arange(1, n_cycles + 1), **rows})

    def test_short_sequence_left_padded(self):
        n = 5
        seq, rul = C.extract_last_sequence(self._engine(n), 300)
        self.assertEqual(seq.shape, (C.SEQ_LEN, len(C.USEFUL_SENSORS)))
        data = self._engine(n)[C.USEFUL_SENSORS].values
        np.testing.assert_array_equal(seq[:C.SEQ_LEN - n], np.repeat(data[:1], C.SEQ_LEN - n, axis=0))
        np.testing.assert_array_equal(seq[C.SEQ_LEN - n:], data)
        self.assertEqual(rul, C.MAX_RUL)

    def test_long_sequence_takes_last_window(self):
        df = self._engine(C.SEQ_LEN + 12)
        seq, rul = C.extract_last_sequence(df, 40)
        np.testing.assert_array_equal(seq, df[C.USEFUL_SENSORS].values[-C.SEQ_LEN:])
        self.assertEqual(rul, 40)

    def test_supervised_sequences_targets(self):
        X, y = C.extract_supervised_sequences(self._engine(C.SEQ_LEN + 2))
        self.assertEqual(len(X), 3)
        self.assertEqual(y, [2, 1, 0])


class SplitUnitsTest(unittest.TestCase):
    def test_disjoint_complete_and_deterministic(self):
        units = list(range(1, 51))
        tr, va = C.split_units(units, 0.1, seed=3)
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(sorted(tr + va), units)
        self.assertEqual(len(va), 5)
        self.assertEqual((tr, va), C.split_units(list(reversed(units)), 0.1, seed=3))
        self.assertNotEqual(va, C.split_units(units, 0.1, seed=4)[1])

    def test_at_least_one_validation_unit(self):
        tr, va = C.split_units([4, 9, 2], 0.1, seed=0)
        self.assertEqual(len(va), 1)
        self.assertEqual(sorted(tr + va), [2, 4, 9])


# --------------------------------------------------------------------------- sanity_check.py
class SanityCheckTest(unittest.TestCase):
    def test_loss_sanity_check(self):
        late, early = sanity_check.run_loss_sanity_check()
        self.assertGreater(late, early)

    @unittest.skipUnless(CMAPSS_ZIP, NEEDS_DATA)
    def test_quick_training_beats_constant_baseline(self):
        with _cwd(ROOT):
            n_test, baseline, untrained, trained = sanity_check.run_sanity_check(42)
        self.assertEqual(n_test, 259)
        self.assertGreater(untrained, baseline)
        self.assertLess(trained, baseline * 0.75)


# --------------------------------------------------------------------------- fed_sim.py
@unittest.skipUnless(CMAPSS_ZIP, NEEDS_DATA)
class FedSimSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scales = C.load_scales(str(ROOT / "global_scale.json"))
        cls.part = C.load_partition(str(ROOT / "partition.json"))
        cls.df = C.apply_scale(C.load_cmapss_raw("train_FD002.txt"), cls.scales)
        cls.test = C.build_test_set(cls.scales)

    def _run(self, mu, loss_name):
        hist = fed_sim.run(42, rounds=1, local_epochs=1, mu=mu, scales=self.scales, df=self.df,
                           test=self.test, part=self.part, loss_fn=C.get_loss_fn(loss_name))
        self.assertEqual(len(hist), 1)
        h = hist[0]
        self.assertEqual(h["round"], 1)
        for key in ("rmse", "nasa"):
            self.assertTrue(_finite(h[key]), f"{key}={h[key]}")
        for key in ("val_rmse", "train_time_s"):
            self.assertEqual(len(h[key]), len(self.part), key)
            self.assertTrue(_finite(h[key]), f"{key}={h[key]}")
        self.assertTrue(all(t > 0 for t in h["train_time_s"]))
        json.dumps(hist)  # the CLI writes this structure as JSON
        return h

    def test_fedavg_mse(self):
        self._run(0.0, "mse")

    def test_fedprox_asymmetric(self):
        self._run(0.01, "asymmetric")


# --------------------------------------------------------------------------- experiments/profile_edge.py
@unittest.skipUnless(CMAPSS_ZIP, NEEDS_DATA)
class ProfileEdgeSmokeTest(unittest.TestCase):
    def test_one_round_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "profile.json"
            proc = subprocess.run(
                [sys.executable, str(ROOT / "experiments" / "profile_edge.py"), "--data", CMAPSS_ZIP,
                 "--dataset", "FD002", "--partition", "regime", "--client", "0",
                 "--rounds", "1", "--epochs", "1", "--output", str(out)],
                cwd=tmp, capture_output=True, text=True, timeout=900)
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            record = json.loads(out.read_text())
        self.assertEqual(record["communication_tensor_payload_bytes_per_round"], 2 * record["model_tensor_bytes"])
        self.assertGreater(record["model_tensor_bytes"], 0)
        self.assertIs(record["mTLS"], False)
        self.assertEqual((record["dataset"], record["partition"], record["client"]), ("FD002", "regime", 0))
        self.assertEqual((record["rounds"], record["local_epochs"]), (1, 1))
        self.assertEqual(len(record["train_seconds_per_round"]), 1)
        self.assertEqual(record["train_seconds_std"], 0.0)
        self.assertGreater(record["training_windows"], 0)
        self.assertGreater(record["client_engines"], 0)
        self.assertGreater(record["peak_process_rss_bytes"], 0)


# --------------------------------------------------------------------------- paper/make_figures.py
class PaperFiguresTest(unittest.TestCase):
    """Run make_figures.py on a temp copy of paper/ + results/; tracked paper/ is never written."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        ignore = shutil.ignore_patterns("*.aux", "*.log", "*.out", "*.pdf", "*.bbl", "*.blg", "*.fls",
                                        "*.fdb_latexmk", "*.synctex.gz", "*.run.xml", "*.bcf", "__pycache__")
        shutil.copytree(ROOT / "paper", cls.root / "paper", ignore=ignore, symlinks=False)
        shutil.copytree(ROOT / "results", cls.root / "results", symlinks=False)
        shutil.rmtree(cls.root / "paper" / "generated")
        cls.proc = subprocess.run([sys.executable, "make_figures.py"], cwd=cls.root / "paper",
                                  capture_output=True, text=True, timeout=300)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_make_figures_regenerates_tracked_tables(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])
        generated = self.root / "paper" / "generated"
        tracked = sorted(p.name for p in (ROOT / "paper" / "generated").glob("*.tex"))
        self.assertTrue(tracked)
        self.assertEqual(sorted(p.name for p in generated.glob("*.tex")), tracked)
        for name in tracked:
            self.assertEqual((generated / name).read_text(encoding="utf-8"),
                             (ROOT / "paper" / "generated" / name).read_text(encoding="utf-8"), name)

    @unittest.skipUnless(shutil.which("latexmk") and shutil.which("pdflatex"), "latexmk/pdflatex not installed")
    def test_latexmk_builds_pdf(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-3000:])
        paper = self.root / "paper"
        proc = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                              cwd=paper, capture_output=True, text=True, timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:])
        self.assertGreater((paper / "main.pdf").stat().st_size, 0)
        log = (paper / "main.log").read_text(encoding="utf-8", errors="replace")
        self.assertNotIn("Undefined control sequence", log)
        self.assertNotRegex(log, r"Citation `[^']+' on page \d+ undefined")
        self.assertNotRegex(log, r"Reference `[^']+' on page \d+ undefined")


if __name__ == "__main__":
    unittest.main()
