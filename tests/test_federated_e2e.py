"""Federated runtime tests: client.SupervisedClient, server.weighted_avg, FL -> edge diagnostics.

Synthetic tests need neither CMAPSSData.zip nor a network. The Flower gRPC loopback test is
skipped unless CMAPSSData.zip and flwr are available (it spawns server.py + 2 client.py on
127.0.0.1 with an ephemeral port).
"""
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    import flwr  # noqa: F401
    HAVE_FLWR = True
except ImportError:  # pragma: no cover
    HAVE_FLWR = False

from common import (AsymmetricRULLoss, CmapssRegressor, SEQ_LEN, USEFUL_SENSORS, apply_scale,  # noqa: E402
                    cmapss_zip_path, get_loss_fn, make_loader, make_xy, set_seed)

if HAVE_FLWR:
    import client as client_mod  # noqa: E402
    import server as server_mod  # noqa: E402
import diagnostics as D  # noqa: E402


def _zip_path_or_none():
    try:
        return os.path.abspath(cmapss_zip_path())
    except FileNotFoundError:
        return None


ZIP = _zip_path_or_none()
FAULT = D.FAULT_SENSORS


def synthetic_plant(seed, n_units=5, n_cycles=60):
    """Raw run-to-failure-like frame (unit, cycle, settings, sensors) with a ramp on s3/s11."""
    rng = np.random.RandomState(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            ramp = max(0, c - 30) / 30.0
            vals = {s: 100.0 + 10 * i + rng.normal(0, 0.5) for i, s in enumerate(FAULT)}
            vals["s3"] += 8 * ramp
            vals["s11"] += 6 * ramp
            st = rng.normal(0, 0.001, 3) + np.array([0.0, 0.25, 100.0])
            rows.append(dict(unit=u, cycle=c, setting1=st[0], setting2=st[1], setting3=st[2], **vals))
    return pd.DataFrame(rows)


def scales_of(df):
    return {"min": df[USEFUL_SENSORS].min().to_dict(), "max": df[USEFUL_SENSORS].max().to_dict()}


def make_client(df, scales, seed, loss_fn=None, val_units=(5,)):
    """SupervisedClient on a synthetic plant: units in ``val_units`` held out for evaluation."""
    set_seed(seed)
    scaled = apply_scale(df, scales)
    tr = [u for u in sorted(df["unit"].unique()) if u not in val_units]
    X, y = make_xy(scaled, tr)
    Xv, yv = make_xy(scaled, list(val_units))
    loader = make_loader(X, y, 32, seed=seed)
    cl = client_mod.SupervisedClient(CmapssRegressor(), loader, torch.tensor(Xv), yv, loss_fn=loss_fn)
    return cl, len(X)


def snapshot(cl):
    """Copy of a client's parameters (get_parameters returns views of the live model tensors)."""
    return [p.copy() for p in cl.get_parameters({})]


def max_abs_diff(a, b):
    return max(float(np.abs(x - y).max()) for x, y in zip(a, b))


@unittest.skipUnless(HAVE_FLWR, "flwr not installed")
class SupervisedClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_plant(0)
        cls.scales = scales_of(cls.df)

    def test_parameters_roundtrip_strict(self):
        set_seed(1)
        src = CmapssRegressor()
        want = [v.numpy().copy() for v in src.state_dict().values()]
        cl, _ = make_client(self.df, self.scales, seed=2)
        self.assertGreater(max_abs_diff(cl.get_parameters({}), want), 0.0)  # different init
        cl.set_parameters(want)
        got = cl.get_parameters({})
        self.assertEqual(len(got), len(want))
        self.assertEqual(len(got), len(cl.model.state_dict()))
        for g, w in zip(got, want):
            self.assertEqual(g.shape, w.shape)
            np.testing.assert_array_equal(g, w)

    def test_set_parameters_is_strict(self):
        cl, _ = make_client(self.df, self.scales, seed=2)
        params = cl.get_parameters({})
        with self.assertRaises(RuntimeError):
            cl.set_parameters(params[:-1])  # missing tensor
        bad = list(params)
        bad[0] = np.zeros((1,) + bad[0].shape, dtype=np.float32)
        with self.assertRaises(RuntimeError):
            cl.set_parameters(bad)  # wrong shape

    def _check_fit(self, loss_fn, mu):
        cl, n_train = make_client(self.df, self.scales, seed=3, loss_fn=loss_fn)
        before = snapshot(cl)
        params, n, metrics = cl.fit(before, {"proximal_mu": mu, "local_epochs": 1})
        self.assertEqual(n, n_train)
        self.assertEqual(n, len(cl.train_loader.dataset))
        self.assertEqual(len(params), len(before))
        for k in ("train_time_s", "peak_rss_mb"):
            self.assertTrue(np.isfinite(metrics[k]), k)
            self.assertGreater(metrics[k], 0.0, k)
        for p in params:
            self.assertTrue(np.isfinite(p).all())
        self.assertGreater(max_abs_diff(params, before), 0.0, "weights did not update")
        return params, before

    def test_fit_mse_and_asymmetric_with_and_without_proximal(self):
        for name, loss_fn in (("mse", get_loss_fn("mse")), ("asymmetric", AsymmetricRULLoss())):
            for mu in (0.0, 0.1):
                with self.subTest(loss=name, mu=mu):
                    self._check_fit(loss_fn, mu)

    def test_client_default_loss_is_mse_and_asymmetric_is_used(self):
        cl, _ = make_client(self.df, self.scales, seed=3)
        self.assertIsInstance(cl.loss_fn, torch.nn.MSELoss)
        cl, _ = make_client(self.df, self.scales, seed=3, loss_fn=AsymmetricRULLoss())
        self.assertIsInstance(cl.loss_fn, AsymmetricRULLoss)

    def test_proximal_term_pulls_toward_global(self):
        """Large mu keeps the update closer to the global model than mu=0 (same seed/data)."""
        dist = {}
        for mu in (0.0, 5.0):
            cl, _ = make_client(self.df, self.scales, seed=4)
            before = snapshot(cl)
            after, _, _ = cl.fit(before, {"proximal_mu": mu, "local_epochs": 1})
            dist[mu] = sum(float(((a - b) ** 2).sum()) for a, b in zip(after, before))
        self.assertLess(dist[5.0], dist[0.0])

    def test_evaluate_returns_finite_val_rmse(self):
        cl, _ = make_client(self.df, self.scales, seed=5)
        loss, n, metrics = cl.evaluate(cl.get_parameters({}), {})
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(np.isfinite(metrics["val_rmse"]))
        self.assertEqual(loss, metrics["val_rmse"])
        self.assertEqual(n, len(cl.val_y))
        self.assertEqual(n, 60 - SEQ_LEN + 1)  # one held-out unit


@unittest.skipUnless(HAVE_FLWR, "flwr not installed")
class WeightedAvgTest(unittest.TestCase):
    def test_sample_weighted_exact(self):
        metrics = [(10, {"a": 1.0, "b": 10.0}), (30, {"a": 5.0, "b": 2.0})]
        out = server_mod.weighted_avg(metrics)
        self.assertEqual(set(out), {"a", "b"})
        self.assertAlmostEqual(out["a"], (10 * 1.0 + 30 * 5.0) / 40, places=12)
        self.assertAlmostEqual(out["b"], (10 * 10.0 + 30 * 2.0) / 40, places=12)

    def test_single_client_identity_and_equal_weights(self):
        self.assertEqual(server_mod.weighted_avg([(7, {"x": 3.5})]), {"x": 3.5})
        self.assertAlmostEqual(server_mod.weighted_avg([(4, {"x": 1.0}), (4, {"x": 3.0})])["x"], 2.0, places=12)

    def test_empty_list(self):
        self.assertEqual(server_mod.weighted_avg([]), {})


def fedavg(results):
    """Sample-weighted mean of [(params, n), ...] (what flwr's FedAvg does)."""
    tot = sum(n for _, n in results)
    return [sum(p[i].astype(np.float64) * n for p, n in results) / tot for i in range(len(results[0][0]))]


@unittest.skipUnless(HAVE_FLWR, "flwr not installed")
class FederatedToEdgeDiagnosticsTest(unittest.TestCase):
    def test_two_round_fedavg_then_local_diagnostics(self):
        plants = [synthetic_plant(10), synthetic_plant(11, n_units=6)]
        allv = pd.concat(plants)
        scales = scales_of(allv)  # shared global scaler, as global_scale.json
        clients, sizes = [], []
        for i, df in enumerate(plants):
            cl, n = make_client(df, scales, seed=20 + i, loss_fn=AsymmetricRULLoss(),
                                val_units=(int(df["unit"].max()),))
            clients.append(cl)
            sizes.append(n)
        self.assertNotEqual(sizes[0], sizes[1])

        set_seed(0)
        glob = [v.numpy() for v in CmapssRegressor().state_dict().values()]
        initial = [g.copy() for g in glob]
        for rnd in range(2):
            fits = [cl.fit(glob, {"proximal_mu": 0.0, "local_epochs": 1}) for cl in clients]
            for _, n, m in fits:
                self.assertTrue(np.isfinite(m["train_time_s"]))
            self.assertEqual([n for _, n, _ in fits], sizes)
            glob = [g.astype(np.float32) for g in fedavg([(p, n) for p, n, _ in fits])]
            evals = [cl.evaluate(glob, {}) for cl in clients]
            for loss, n, m in evals:
                self.assertTrue(np.isfinite(m["val_rmse"]), f"round {rnd}")
        self.assertGreater(max_abs_diff(glob, initial), 0.0)

        # Aggregated global model -> local edge diagnostics on each plant.
        model = CmapssRegressor()
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), glob)}, strict=True)
        for i, df in enumerate(plants):
            with self.subTest(plant=i):
                diag = D.EdgeDiagnostics(model, scales, D.fit_healthy_baseline(df, 1))
                for unit in sorted(df["unit"].unique())[:2]:
                    out = diag.diagnose_engine(df[df["unit"] == unit], f"P{i}-U{unit}")
                    n_win = 60 - SEQ_LEN + 1
                    self.assertEqual(out["pred_rul"].shape, (n_win,))
                    self.assertTrue(np.isfinite(out["pred_rul"]).all())
                    np.testing.assert_allclose(out["sensor_share"].sum(1), 100.0, atol=1e-9)
                    np.testing.assert_allclose(out["subsystem_share"].sum(1), 100.0, atol=1e-9)
                    self.assertTrue(np.all(out["channel_attr"][:, D.SETTING_IDX] == 0.0))
                    summ = diag.summarize(out)
                    self.assertAlmostEqual(sum(summ["subsystem_shares_pct"].values()), 100.0, places=6)
                    self.assertEqual(summ["privacy"]["raw_sensor_bytes_transmitted"], 0)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _kill(proc):
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=10)


@unittest.skipUnless(ZIP and HAVE_FLWR, "CMAPSSData.zip and flwr required for the gRPC loopback test")
class FlowerLoopbackTest(unittest.TestCase):
    TIMEOUT_S = 600

    def test_server_and_two_clients_one_round(self):
        port = _free_port()
        addr = f"127.0.0.1:{port}"
        base_env = dict(os.environ, SEED="42", CMAPSS_ZIP=ZIP, PYTHONUNBUFFERED="1")
        procs, logs = [], []
        with tempfile.TemporaryDirectory(prefix="fedpdm_e2e_") as tmp:
            def spawn(name, args, env):
                log = open(os.path.join(tmp, f"{name}.log"), "w+")
                logs.append(log)
                p = subprocess.Popen([sys.executable, "-u"] + args, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                procs.append((name, p))
                return p

            def tail(name):
                for log in logs:
                    if os.path.basename(log.name) == f"{name}.log":
                        log.flush()
                        log.seek(0)
                        return log.read()[-3000:]
                return ""

            try:
                srv = spawn("server", ["server.py"],
                            dict(base_env, ROUNDS="1", LOCAL_EPOCHS="1", MIN_CLIENTS="2",
                                 RESULTS_DIR=tmp, BIND=addr))
                deadline = time.time() + 120
                while not _port_open(port):
                    if srv.poll() is not None:
                        self.fail(f"server exited early ({srv.returncode}):\n{tail('server')}")
                    if time.time() > deadline:
                        self.fail(f"server never listened on {addr}:\n{tail('server')}")
                    time.sleep(0.5)
                # client.py reads its identity/endpoint from CLIENT_ID / FL_SERVER env vars.
                for cid in ("1", "2"):
                    spawn(f"client{cid}", ["client.py"], dict(base_env, CLIENT_ID=cid, FL_SERVER=addr))

                end = time.time() + self.TIMEOUT_S
                for name, p in procs:
                    try:
                        p.wait(timeout=max(1, end - time.time()))
                    except subprocess.TimeoutExpired:
                        self.fail(f"{name} timed out after {self.TIMEOUT_S}s:\n{tail(name)}")
                for name, p in procs:
                    self.assertEqual(p.returncode, 0, f"{name} exit {p.returncode}:\n{tail(name)}")
            finally:
                for _, p in procs:
                    _kill(p)
                for log in logs:
                    log.close()

            path = os.path.join(tmp, "results_physical_seed42.json")
            self.assertTrue(os.path.isfile(path), os.listdir(tmp))
            with open(path) as f:
                res = json.load(f)
            self.assertEqual(res["config"]["seed"], 42)
            self.assertEqual(res["config"]["rounds"], 1)
            mc = res["metrics_centralized"]
            self.assertIn("model_rmse", mc)
            self.assertEqual(len(mc["model_rmse"]), 2)  # round 0 (initial) + round 1
            for key in ("model_rmse", "baseline_rmse", "nasa_score"):
                for _, v in mc[key]:
                    self.assertTrue(np.isfinite(v), key)
            mf = res["metrics_distributed_fit"]
            for key in ("train_time_s", "peak_rss_mb"):
                self.assertIn(key, mf)
                self.assertEqual(len(mf[key]), 1)
                self.assertTrue(np.isfinite(mf[key][0][1]))


if __name__ == "__main__":
    unittest.main()
