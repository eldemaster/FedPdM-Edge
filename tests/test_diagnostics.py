"""Unit tests for diagnostics.py (synthetic data; no CMAPSSData.zip required)."""
import json
import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diagnostics as D  # noqa: E402
from common import SEQ_LEN, USEFUL_SENSORS, CmapssRegressor, cmapss_zip_path, load_cmapss_raw, set_seed  # noqa: E402


def _real_data_available():
    try:
        cmapss_zip_path()
        return True
    except FileNotFoundError:
        return False


def synthetic_fleet(n_units=6, n_cycles=80, n_regimes=1, seed=0, wild_settings=False):
    """Run-to-failure-like frame: flat healthy start then a ramp on HPC sensors s3/s11."""
    rng = np.random.RandomState(seed)
    regime_settings = np.array([[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100],
                                [25, 0.62, 60], [35, 0.84, 100], [42, 0.84, 100]], dtype=float)[:max(1, n_regimes)]
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            r = rng.randint(len(regime_settings))
            st = regime_settings[r] + rng.normal(0, 0.001, 3)
            if wild_settings:
                st = st + rng.normal(0, 50.0, 3)
            base = {s: 100.0 + 10 * i + 5.0 * r for i, s in enumerate(D.FAULT_SENSORS)}
            ramp = max(0, c - 40) / 40.0
            vals = {s: base[s] + rng.normal(0, 0.5) for s in D.FAULT_SENSORS}
            vals["s3"] += 8 * ramp
            vals["s11"] += 6 * ramp
            rows.append(dict(unit=u, cycle=c, setting1=st[0], setting2=st[1], setting3=st[2], **vals))
    return pd.DataFrame(rows)


def scales_of(df):
    return {"min": df[USEFUL_SENSORS].min().to_dict(), "max": df[USEFUL_SENSORS].max().to_dict()}


def make_model(seed=0):
    set_seed(seed)
    return CmapssRegressor()


class ChannelMapTest(unittest.TestCase):
    def test_fourteen_fault_sensors_no_settings(self):
        self.assertEqual(len(D.FAULT_SENSORS), 14)
        self.assertTrue(set(D.SETTINGS).isdisjoint(D.FAULT_SENSORS))
        self.assertEqual(set(D.SENSOR_INFO), set(D.FAULT_SENSORS))
        self.assertEqual(set(c for s in D.SUBSYSTEMS for c in D.SUBSYSTEM_SENSORS[s]), set(D.FAULT_SENSORS))
        self.assertEqual(D.SUBSYSTEMS, ("Fan", "LPC", "HPC", "Turbine"))
        # Table 2 spot checks
        self.assertEqual(D.SENSOR_INFO["s3"]["symbol"], "T30")
        self.assertEqual(D.SENSOR_INFO["s11"]["symbol"], "Ps30")
        self.assertEqual(D.SENSOR_INFO["s15"]["symbol"], "BPR")


class AttributionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.windows = rng.rand(7, SEQ_LEN, len(USEFUL_SENSORS)).astype(np.float32)
        self.model = make_model(3)

    def test_shapes_and_determinism(self):
        for method in D.ATTRIBUTION_METHODS:
            base = np.zeros_like(self.windows, dtype=np.float64) + 0.5
            p1, a1 = D.attribute_windows(self.model, self.windows, base, method=method, batch_size=3)
            p2, a2 = D.attribute_windows(self.model, self.windows, base, method=method)
            self.assertEqual(p1.shape, (7,))
            self.assertEqual(a1.shape, (7, len(USEFUL_SENSORS)))
            np.testing.assert_allclose(a1, a2, rtol=1e-6, atol=1e-10)
            np.testing.assert_allclose(p1, p2, rtol=1e-6)
            sh = D.sensor_shares(a1)
            self.assertEqual(sh.shape, (7, 14))
            self.assertEqual(D.subsystem_shares(sh).shape, (7, 4))

    def test_input_x_gradient_formula(self):
        x = torch.tensor(self.windows, requires_grad=True)
        (g,) = torch.autograd.grad(self.model(x).sum(), x)
        expected = np.abs(self.windows * g.numpy()).mean(1)
        expected[:, D.SETTING_IDX] = 0
        _, a = D.attribute_windows(self.model, self.windows, method="input_x_gradient")
        np.testing.assert_allclose(a, expected, rtol=1e-5, atol=1e-9)

    def test_baseline_x_gradient_formula_nontrivial_baseline(self):
        rng = np.random.RandomState(11)
        baseline = rng.rand(*self.windows.shape)
        self.assertFalse(np.allclose(baseline, self.windows))
        x = torch.tensor(self.windows, dtype=torch.float32, requires_grad=True)
        y = self.model(x)
        (g,) = torch.autograd.grad(y.sum(), x)
        expected = np.abs((x.detach().double().numpy() - baseline) * g.double().numpy()).mean(1)
        self.assertEqual(self.windows.shape[1], 30)
        expected[:, D.SETTING_IDX] = 0.0
        pred, a = D.attribute_windows(self.model, self.windows, baseline, method="baseline_x_gradient")
        np.testing.assert_allclose(a, expected, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(pred, y.detach().double().numpy(), rtol=1e-9)
        self.assertTrue(np.all(a[:, D.SETTING_IDX] == 0.0))
        self.assertTrue((a[:, D.FAULT_IDX] > 0).any())

    def test_empty_windows(self):
        empty = np.zeros((0, SEQ_LEN, len(USEFUL_SENSORS)), dtype=np.float32)
        for method in D.ATTRIBUTION_METHODS:
            pred, attr = D.attribute_windows(self.model, empty, empty.astype(np.float64), method=method)
            self.assertEqual(pred.shape, (0,))
            self.assertEqual(pred.dtype, np.float64)
            self.assertEqual(attr.shape, (0, len(USEFUL_SENSORS)))
            self.assertEqual(attr.dtype, np.float64)
        pred, attr = D.attribute_windows(self.model, empty)  # baseline=None fallback
        self.assertEqual((pred.shape, attr.shape), ((0,), (0, len(USEFUL_SENSORS))))
        with self.assertRaises(ValueError):
            D.attribute_windows(self.model, empty, method="bogus")

    def test_conservation_sum_100(self):
        _, a = D.attribute_windows(self.model, self.windows, method="gradient")
        sh = D.sensor_shares(a)
        np.testing.assert_allclose(sh.sum(1), 100.0, atol=1e-9)
        np.testing.assert_allclose(D.subsystem_shares(sh).sum(1), 100.0, atol=1e-9)
        self.assertTrue((sh >= 0).all())
        # degenerate all-zero attribution falls back to a uniform split, still 100
        z = D.sensor_shares(np.zeros((2, len(USEFUL_SENSORS))))
        np.testing.assert_allclose(z.sum(1), 100.0)
        # rounded shares used by the dashboard sum to exactly 100.0
        for row in D.subsystem_shares(sh):
            self.assertAlmostEqual(sum(D.round_to_total(row, 1)), 100.0, places=9)

    def test_settings_excluded_even_when_wild(self):
        w = self.windows.copy()
        w[:, :, D.SETTING_IDX] = np.random.RandomState(5).normal(0, 1e3, (7, SEQ_LEN, 3))
        for method in D.ATTRIBUTION_METHODS:
            _, a = D.attribute_windows(self.model, w, np.zeros_like(w, dtype=np.float64), method=method)
            self.assertTrue(np.all(a[:, D.SETTING_IDX] == 0.0))
            self.assertTrue(np.all(a[:, D.FAULT_IDX] >= 0.0))
            np.testing.assert_allclose(D.sensor_shares(a).sum(1), 100.0, atol=1e-9)

    def test_baseline_equal_input_gives_uniform(self):
        _, a = D.attribute_windows(self.model, self.windows, self.windows.astype(np.float64))
        self.assertTrue(np.allclose(a, 0.0))
        np.testing.assert_allclose(D.sensor_shares(a), 100.0 / 14)


class RoundToTotalTest(unittest.TestCase):
    def test_valid_shares_sum_exactly(self):
        out = D.round_to_total([33.333, 33.333, 33.334], 1)
        self.assertEqual(out, [33.3, 33.3, 33.4])
        self.assertEqual(round(sum(out), 9), 100.0)
        rng = np.random.RandomState(0)
        for _ in range(50):
            v = rng.dirichlet(np.ones(4)) * 100.0
            for dec in (0, 1, 2):
                self.assertAlmostEqual(sum(D.round_to_total(v, dec)), 100.0, places=9)

    def test_largest_remainder_choice(self):
        # floors 10,20,68 leave 2 units; largest remainders are .9 (idx0) and .6 (idx2)
        self.assertEqual(D.round_to_total([10.9, 20.5, 68.6], 0), [11.0, 20.0, 69.0])
        self.assertEqual(D.round_to_total([25.0, 25.0, 25.0, 25.0], 1), [25.0] * 4)
        self.assertEqual(D.round_to_total([5.0, 5.0], 1, total=10.0), [5.0, 5.0])

    def test_invalid_inputs_raise(self):
        for bad in ([-1.0, 101.0], [float("nan"), 50.0], [float("inf"), 1.0]):
            with self.assertRaises(ValueError):
                D.round_to_total(bad, 1)
        with self.assertRaises(ValueError):
            D.round_to_total([50.0, 50.0], 1, total=float("nan"))
        with self.assertRaises(ValueError):
            D.round_to_total([50.0, 50.0], 1, total=-100.0)
        # remainder above len(values): shares far below the total
        with self.assertRaises(ValueError):
            D.round_to_total([10.0, 10.0], 1, total=100.0)
        # remainder below zero: shares far above the total
        with self.assertRaises(ValueError):
            D.round_to_total([90.0, 90.0], 1, total=100.0)
        with self.assertRaises(ValueError):
            D.round_to_total([], 1)


class BaselineDriftTest(unittest.TestCase):
    def test_single_regime_drift(self):
        df = synthetic_fleet(n_regimes=1)
        bl = D.fit_healthy_baseline(df, n_regimes=1, healthy_cycles=30)
        self.assertEqual(bl.n_regimes, 1)
        z = bl.drift(df[df.unit == 1])
        self.assertEqual(z.shape, (80, 14))
        # healthy segment centered at 0, end-of-life HPC sensors drift up strongly
        np.testing.assert_allclose(z[:30].mean(0), 0.0, atol=1e-9)
        i3, i11, i8 = (D.FAULT_SENSORS.index(s) for s in ("s3", "s11", "s8"))
        self.assertGreater(z[-5:, i3].mean(), 5)
        self.assertGreater(z[-5:, i11].mean(), 5)
        self.assertLess(abs(z[-5:, i8].mean()), 2)

    def test_multi_regime_drift_removes_operating_condition(self):
        df = synthetic_fleet(n_regimes=6, n_units=8)
        bl = D.fit_healthy_baseline(df, n_regimes=6, healthy_cycles=30)
        self.assertEqual(bl.n_regimes, 6)
        g = df[df.unit == 2]
        z = bl.drift(g)
        i8 = D.FAULT_SENSORS.index("s8")
        # regime switching (+5 per regime) must not appear as drift on unaffected sensors
        self.assertLess(np.abs(z[:, i8]).max(), 5)
        self.assertGreater(z[-5:, D.FAULT_SENSORS.index("s3")].mean(), 5)
        # without regime normalization, regime switching swamps the healthy noise
        # estimate and the real fault drift on s3 becomes much less visible
        naive = D.fit_healthy_baseline(df, n_regimes=1).drift(g)
        i3 = D.FAULT_SENSORS.index("s3")
        self.assertGreater(z[-5:, i3].mean(), 3 * naive[-5:, i3].mean())

    def test_regimes_1d_and_empty_inputs(self):
        df = synthetic_fleet(n_regimes=6, n_units=8)
        bl = D.fit_healthy_baseline(df, n_regimes=6)
        two_d = bl.regimes(df[list(D.SETTINGS)].values[:10])
        self.assertEqual(two_d.shape, (10,))
        one_d = bl.regimes(df[list(D.SETTINGS)].values[0])
        self.assertEqual(one_d.shape, (1,))
        self.assertEqual(int(one_d[0]), int(two_d[0]))
        self.assertEqual(bl.regimes(np.zeros((0, 3))).shape, (0,))
        self.assertEqual(bl.regimes(np.zeros((0, 3))).dtype.kind, "i")
        self.assertEqual(bl.regimes(df[list(D.SETTINGS)].iloc[:0]).shape, (0,))
        with self.assertRaises(ValueError):
            bl.regimes(np.zeros((4, 2)))
        # empty frame flows through standardize
        u, reg = bl.standardize(df.iloc[:0])
        self.assertEqual((u.shape, reg.shape), ((0, 14), (0,)))

    def test_healthy_reference_settings_untouched(self):
        df = synthetic_fleet()
        sc = scales_of(df)
        bl = D.fit_healthy_baseline(df)
        g = df[df.unit == 1]
        ref = bl.healthy_reference(g, sc)
        x = D.scale_frame(g[USEFUL_SENSORS].values, sc)
        np.testing.assert_allclose(ref[:, D.SETTING_IDX], x[:, D.SETTING_IDX])
        self.assertEqual(ref.shape, x.shape)


class EngineAndSummaryTest(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_fleet(n_regimes=6, wild_settings=False)
        self.sc = scales_of(self.df)
        self.diag = D.EdgeDiagnostics(make_model(1), self.sc, D.fit_healthy_baseline(self.df, 6))

    def test_engine_outputs_aligned(self):
        out = self.diag.diagnose_engine(self.df[self.df.unit == 3], "SYN-U3")
        n = 80 - SEQ_LEN + 1
        self.assertEqual(len(out["cycles"]), n)
        self.assertEqual(out["cycles"][0], SEQ_LEN)
        for k, shape in [("pred_rul", (n,)), ("channel_attr", (n, 17)), ("sensor_share", (n, 14)),
                         ("subsystem_share", (n, 4)), ("window_drift", (n, 14)), ("cycle_drift", (80, 14))]:
            self.assertEqual(out[k].shape, shape, k)
        np.testing.assert_allclose(out["subsystem_share"].sum(1), 100.0, atol=1e-9)
        self.assertTrue(np.all(out["channel_attr"][:, D.SETTING_IDX] == 0))
        again = self.diag.diagnose_engine(self.df[self.df.unit == 3], "SYN-U3")
        np.testing.assert_allclose(out["sensor_share"], again["sensor_share"], rtol=1e-6)

    def test_short_engine_padded(self):
        out = self.diag.diagnose_engine(self.df[(self.df.unit == 2) & (self.df.cycle <= 12)])
        self.assertEqual(out["sensor_share"].shape, (1, 14))
        self.assertEqual(int(out["cycles"][0]), 12)

    def test_summary_structure(self):
        out = self.diag.diagnose_engine(self.df[self.df.unit == 3], "SYN-U3")
        s = self.diag.summarize(out)
        for key in ("engine_id", "cycle", "predicted_rul", "severity", "primary_subsystem",
                    "subsystem_shares_pct", "top_contributors", "top_drifting", "text", "privacy",
                    "evaluation_boundary", "drift_index", "primary_drift_subsystem"):
            self.assertIn(key, s)
        self.assertEqual(s["cycle"], 80)
        self.assertIn(s["severity"], ("critical", "warning", "healthy"))
        self.assertIn(s["primary_subsystem"], D.SUBSYSTEMS)
        self.assertAlmostEqual(sum(s["subsystem_shares_pct"].values()), 100.0, places=6)
        self.assertEqual(len(s["top_contributors"]), 3)
        self.assertEqual(len(s["top_drifting"]), 3)
        for e in s["top_contributors"] + s["top_drifting"]:
            self.assertIn(e["sensor"], D.FAULT_SENSORS)
            self.assertNotIn(e["sensor"], D.SETTINGS)
            self.assertEqual(e["subsystem"], D.SENSOR_INFO[e["sensor"]]["subsystem"])
        shares = [e["share_pct"] for e in s["top_contributors"]]
        self.assertEqual(shares, sorted(shares, reverse=True))
        # synthetic fault ramps s3/s11 -> they must be the top drifting sensors
        self.assertEqual({e["sensor"] for e in s["top_drifting"][:2]}, {"s3", "s11"})
        self.assertEqual(s["primary_drift_subsystem"], "HPC")
        self.assertEqual(s["privacy"]["raw_sensor_bytes_transmitted"], 0)
        self.assertFalse(s["privacy"]["contains_raw_sensor_values"])
        self.assertTrue(s["text"]["en"] and s["text"]["it"])
        json.dumps(s)  # serializable

    def test_summarize_non_final_index(self):
        out = self.diag.diagnose_engine(self.df[self.df.unit == 3], "SYN-U3")
        s0 = self.diag.summarize(out, index=0)
        self.assertEqual(s0["cycle"], SEQ_LEN)
        self.assertEqual(s0["engine_id"], "SYN-U3")
        self.assertEqual(s0["predicted_rul"], round(float(out["pred_rul"][0]), 2))
        self.assertAlmostEqual(sum(s0["subsystem_shares_pct"].values()), 100.0, places=6)
        self.assertEqual(len(s0["top_contributors"]), 3)
        s_last = self.diag.summarize(out)
        self.assertEqual(s_last["cycle"], 80)
        self.assertEqual(self.diag.summarize(out, index=-1)["cycle"], s_last["cycle"])
        self.assertEqual(self.diag.summarize(out, index=len(out["cycles"]) - 1), s_last)
        json.dumps(s0)

    def test_severity_thresholds(self):
        self.assertEqual(D.severity_for(10), "critical")
        self.assertEqual(D.severity_for(30), "warning")
        self.assertEqual(D.severity_for(90), "healthy")


@unittest.skipUnless(_real_data_available(), "CMAPSSData.zip not found (set CMAPSS_ZIP)")
class RealDataIntegrationTest(unittest.TestCase):
    """FD001 (single regime) and FD002 (multi-regime) engines through the full edge pipeline."""

    def _run(self, subset, n_regimes):
        df = load_cmapss_raw(f"train_{subset}.txt")
        baseline = D.fit_healthy_baseline(df, n_regimes=n_regimes)
        diag = D.EdgeDiagnostics(make_model(0), scales_of(df), baseline)
        unit = int(df["unit"].min())
        g = df[df["unit"] == unit]
        T = len(g)
        out = diag.diagnose_engine(g, f"{subset}-U{unit}")
        n = T - SEQ_LEN + 1
        for k, shape in [("pred_rul", (n,)), ("channel_attr", (n, 17)), ("sensor_share", (n, 14)),
                         ("subsystem_share", (n, 4)), ("window_drift", (n, 14)), ("cycle_drift", (T, 14))]:
            self.assertEqual(out[k].shape, shape, k)
            self.assertTrue(np.isfinite(out[k]).all(), k)
        self.assertEqual(len(out["cycles"]), n)
        np.testing.assert_allclose(out["sensor_share"].sum(1), 100.0, atol=1e-9)
        np.testing.assert_allclose(out["subsystem_share"].sum(1), 100.0, atol=1e-9)
        self.assertTrue(np.all(out["channel_attr"][:, D.SETTING_IDX] == 0.0))
        for idx in (0, -1):
            s = diag.summarize(out, index=idx)
            self.assertAlmostEqual(sum(s["subsystem_shares_pct"].values()), 100.0, places=6)
            self.assertEqual(s["engine_id"], f"{subset}-U{unit}")
            for e in s["top_contributors"] + s["top_drifting"]:
                self.assertNotIn(e["sensor"], D.SETTINGS)
            json.dumps(s)
        return baseline, g

    def test_fd001_engine(self):
        baseline, g = self._run("FD001", 1)
        self.assertEqual(baseline.n_regimes, 1)

    def test_fd002_multi_regime_engine(self):
        baseline, g = self._run("FD002", 6)
        self.assertEqual(baseline.n_regimes, 6)
        self.assertGreater(len(np.unique(baseline.regimes(g[list(D.SETTINGS)].values))), 1)


if __name__ == "__main__":
    unittest.main()
