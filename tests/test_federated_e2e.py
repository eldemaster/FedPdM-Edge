"""Federated runtime tests: client.SupervisedClient, server.weighted_avg, FL -> edge diagnostics.

Loopback tests spawn server.py and clients on an ephemeral localhost port. They use
CMAPSSData.zip when available, otherwise a tiny synthetic archive. All credentials,
datasets, results and logs are temporary and every subprocess is reaped on failure.
"""
import json
from datetime import datetime, timedelta, timezone
import ipaddress
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import zipfile

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

    def test_dp_clips_full_delta_and_reports_configuration(self):
        cl, _ = make_client(self.df, self.scales, seed=6)
        cl.dp_clip = 0.025
        before = snapshot(cl)
        after, _, metrics = cl.fit(before, {"local_epochs": 1})
        norm = np.sqrt(sum(np.square(a.astype(np.float64) - b).sum()
                           for a, b in zip(after, before)))
        self.assertGreater(norm, 0)
        self.assertLessEqual(norm, cl.dp_clip)
        self.assertGreater(norm, 0.99 * cl.dp_clip)  # clipping was exercised
        self.assertEqual(metrics["dp_clip"], cl.dp_clip)
        self.assertEqual(metrics["dp_noise_mult"], 0.0)
        self.assertIs(metrics["tls_enabled"], False)
        self.assertIs(metrics["mtls_enabled"], False)
        for actual, transmitted in zip(snapshot(cl), after):
            np.testing.assert_array_equal(actual, transmitted)
        # The input global arrays must remain unchanged (no noise or local aliasing).
        self.assertEqual(max_abs_diff(before, snapshot(make_client(self.df, self.scales, seed=6)[0])), 0)

    def test_dp_noise_is_calibrated_and_not_repeated_by_training_seed(self):
        cl, _ = make_client(self.df, self.scales, seed=7)
        cl.dp_clip, cl.dp_noise_mult = 0.05, 0.4
        reference = snapshot(cl)
        # Zero training epochs isolates the mechanism: each delta is pure Gaussian noise.
        first, _, metrics = cl.fit(reference, {"local_epochs": 0})
        set_seed(7)
        second, _, _ = cl.fit(reference, {"local_epochs": 0})
        noise = np.concatenate([(p.astype(np.float64) - g).ravel() for p, g in zip(first, reference)])
        expected_std = cl.dp_clip * cl.dp_noise_mult
        self.assertAlmostEqual(float(noise.std()), expected_std, delta=expected_std * 0.05)
        self.assertAlmostEqual(float(noise.mean()), 0.0, delta=expected_std * 0.05)
        self.assertGreater(max_abs_diff(first, second), 0)
        self.assertEqual(metrics["dp_noise_mult"], 0.4)

    def test_dp_disabled_preserves_update_and_small_bound_handles_rounding(self):
        cl, _ = make_client(self.df, self.scales, seed=7)
        reference = snapshot(cl)
        local = [p + 0.1 for p in reference]
        self.assertIs(cl.protect_update(local, reference), local)
        cl.dp_clip = 1e-12
        clipped = cl.protect_update(local, reference)
        norm = np.sqrt(sum(np.square(p.astype(np.float64) - g).sum()
                           for p, g in zip(clipped, reference)))
        self.assertLessEqual(norm, cl.dp_clip)

    def test_dp_validation(self):
        for clip, noise in ((0, 0), (-1, 0), (float("nan"), 0), (float("inf"), 0),
                            (1, -1), (1, float("nan")), (1, float("inf")),
                            (None, 1), (1e308, 1e308)):
            with self.subTest(clip=clip, noise=noise), self.assertRaises(ValueError):
                client_mod.validate_dp(clip, noise)

    def test_dp_nonfinite_update_is_not_transmitted(self):
        cl, _ = make_client(self.df, self.scales, seed=7)
        cl.dp_clip = 1
        reference = snapshot(cl)
        bad = snapshot(cl)
        bad[0].flat[0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            cl.protect_update(bad, reference)


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

    def test_security_configuration_is_not_averaged(self):
        a = {"train_time_s": 1.0, "tls_enabled": True, "mtls_enabled": True,
             "dp_clip": 0.5, "dp_noise_mult": 0.0}
        b = dict(a, dp_clip=2.0)
        metrics = server_mod.fit_metrics([(10, a), (30, b)])
        self.assertNotIn("dp_clip", metrics)
        self.assertIs(metrics["mtls_enabled"], True)
        strategy = server_mod.AuditedFedAvg()
        strategy.client_security = [a, b]
        self.assertIsNone(strategy.privacy_config()["dp_clip"])
        self.assertEqual(strategy.privacy_config()["dp_noise_mult"], 0.0)


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


def ephemeral_certificates(directory):
    """A short-lived CA and distinct server/edge identities; keys never leave tmp."""
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError:
        raise unittest.SkipTest("cryptography unavailable for ephemeral TLS certificates")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Ephemeral FedPdM Test CA")])

    def builder(subject, public_key):
        return (x509.CertificateBuilder().subject_name(subject).issuer_name(ca_name)
                .public_key(public_key).serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(hours=1)))

    ca = (builder(ca_name, ca_key.public_key())
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .sign(ca_key, hashes.SHA256()))
    ca_path = directory / "ca.pem"
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    paths = {"ca": str(ca_path)}
    for name in ("server", "client1", "client2"):
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        cert = (builder(subject, key.public_key())
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.ExtendedKeyUsage([
                    ExtendedKeyUsageOID.SERVER_AUTH if name == "server" else ExtendedKeyUsageOID.CLIENT_AUTH
                ]), critical=False))
        if name == "server":
            cert = cert.add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))
            ]), critical=False)
        cert_path, key_path = directory / f"{name}.pem", directory / f"{name}.key"
        cert_path.write_bytes(cert.sign(ca_key, hashes.SHA256()).public_bytes(serialization.Encoding.PEM))
        # Open with restrictive permissions from creation, not after writing the key.
        with os.fdopen(os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
        paths[name] = (str(cert_path), str(key_path))
    return paths


def synthetic_cmapss(directory):
    """Use the real entry points without adding a production synthetic-data mode."""
    directory = Path(directory)
    frame = synthetic_plant(45)
    columns = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    raw = frame.reindex(columns=columns, fill_value=0).to_csv(sep=" ", header=False, index=False)
    path = directory / "CMAPSSData.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("train_FD002.txt", raw)
        z.writestr("test_FD002.txt", raw)
        z.writestr("RUL_FD002.txt", "20\n" * 5)
    (directory / "global_scale.json").write_text(json.dumps({**scales_of(frame), "train_mean_rul": 30.0}))
    (directory / "partition.json").write_text(json.dumps({"1": [1, 2], "2": [3, 4]}))
    return str(path)


@unittest.skipUnless(HAVE_FLWR, "flwr required for transport configuration tests")
class TransportConfigurationTest(unittest.TestCase):
    def run_script(self, script, args, env=None):
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith(("FL_", "DP_")) and k != "CLIENT_ID"}
        return subprocess.run([sys.executable, os.path.join(ROOT, script), *args], env={**clean, **(env or {})},
                              capture_output=True, text=True, timeout=30)

    def test_plaintext_requires_opt_in_and_partial_credentials_fail(self):
        for script in ("server.py", "client.py"):
            for args in ([], ["--ca-cert", "absent.pem"], ["--cert", "absent.pem"],
                         ["--insecure", "--ca-cert", "absent.pem"], ["--unknown-tls-option"]):
                with self.subTest(script=script, args=args):
                    result = self.run_script(script, args)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("error:", result.stderr)

    def test_insecure_warning_and_env_conflict(self):
        result = self.run_script("client.py", ["--insecure"])
        self.assertNotEqual(result.returncode, 0)  # CLIENT_ID intentionally absent
        self.assertIn("WARNING: INSECURE plaintext", result.stderr)
        for script in ("server.py", "client.py"):
            result = self.run_script(script, ["--insecure"], {"FL_CA_CERT": "absent.pem"})
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("conflicts", result.stderr)

    def test_invalid_dp_cli_and_env_fail_before_loading_data(self):
        for args, env in ((["--dp-clip", "nan"], {}), (["--dp-clip", "0"], {}),
                          (["--dp-noise-mult", "1"], {}), ([], {"DP_NOISE_MULT": "-1"})):
            result = self.run_script("client.py", args, env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("dp_", result.stderr)

    def test_pem_validation_and_environment_paths(self):
        with tempfile.TemporaryDirectory(prefix="fedpdm_tls_config_") as tmp:
            certs = ephemeral_certificates(tmp)
            cert, key = certs["server"]
            with mock.patch.dict(os.environ, {"FL_CA_CERT": certs["ca"], "FL_SERVER_CERT": cert,
                                               "FL_SERVER_KEY": key}, clear=True):
                args, loaded = server_mod.transport_args([])
                self.assertTrue(args.require_client_cert)
                self.assertEqual(loaded, tuple(Path(p).read_bytes() for p in (certs["ca"], cert, key)))
                args, _ = server_mod.transport_args(["--no-require-client-cert"])
                self.assertFalse(args.require_client_cert)
            for script, role in (("server.py", "server"), ("client.py", "client1")):
                cert, _ = certs[role]
                result = self.run_script(script, ["--ca-cert", certs["ca"], "--cert", cert,
                                                  "--key", certs["client2"][1]])
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("Invalid TLS credentials", result.stderr)
            bad = Path(tmp) / "bad.pem"
            bad.write_text("not a certificate")
            result = self.run_script("server.py", ["--ca-cert", str(bad), "--cert", certs["server"][0],
                                                    "--key", certs["server"][1]])
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_scoped_grpc_adapters_restore_on_failure(self):
        import flwr.common.grpc as flower_grpc
        with tempfile.TemporaryDirectory(prefix="fedpdm_tls_adapter_") as tmp:
            certs = ephemeral_certificates(tmp)
            original = flower_grpc.grpc
            ca = Path(certs["ca"]).read_bytes()
            for role, adapter in (("server", server_mod.flower_server_credentials),
                                  ("client1", client_mod.flower_client_credentials)):
                cert, key = (Path(p).read_bytes() for p in certs[role])
                context = adapter((ca, cert, key), True) if role == "server" else adapter((ca, cert, key))
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    with context:
                        self.assertIsNot(flower_grpc.grpc, original)
                        if role == "server":
                            flower_grpc.grpc.ssl_server_credentials(((key, cert),), root_certificates=ca)
                        else:
                            flower_grpc.grpc.ssl_channel_credentials(ca)
                        raise RuntimeError("test failure")
                self.assertIs(flower_grpc.grpc, original)


@unittest.skipUnless(HAVE_FLWR, "flwr required for gRPC loopback tests")
class FlowerLoopbackTest(unittest.TestCase):
    TIMEOUT_S = 600

    def setUp(self):
        self.tmp_context = tempfile.TemporaryDirectory(prefix="fedpdm_e2e_")
        self.addCleanup(self.tmp_context.cleanup)
        self.tmp = Path(self.tmp_context.name)
        self.procs, self.logs = [], {}
        # This cleanup runs before TemporaryDirectory cleanup, even after assertions fail.
        self.addCleanup(self.stop_processes)
        self.certs = ephemeral_certificates(self.tmp / "certs")
        port = _free_port()
        self.port, self.addr = port, f"127.0.0.1:{port}"
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("FL_", "DP_"))}
        self.env.update(SEED="42", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if ZIP:
            self.env["CMAPSS_ZIP"] = ZIP
            for name in ("global_scale.json", "partition.json"):
                shutil.copyfile(os.path.join(ROOT, name), self.tmp / name)
        else:
            self.env["CMAPSS_ZIP"] = synthetic_cmapss(self.tmp)

    def stop_processes(self):
        for _, proc in self.procs:
            _kill(proc)
        for log in self.logs.values():
            log.close()

    def spawn(self, name, script, args=(), **env):
        log = open(self.tmp / f"{name}.log", "w+")
        self.logs[name] = log
        proc = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, script), *args],
                                cwd=self.tmp, env={**self.env, **env}, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append((name, proc))
        return proc

    def tail(self, name):
        log = self.logs[name]
        log.flush()
        log.seek(0)
        return log.read()[-5000:]

    def start_server(self, mode="mtls", env_paths=False):
        args, env = [], {}
        if mode == "insecure":
            args = ["--insecure"]
        else:
            cert, key = self.certs["server"]
            if env_paths:
                env = {"FL_CA_CERT": self.certs["ca"], "FL_SERVER_CERT": cert, "FL_SERVER_KEY": key}
            else:
                args = ["--ca-cert", self.certs["ca"], "--cert", cert, "--key", key]
            if mode == "tls":
                args.append("--no-require-client-cert")
        srv = self.spawn("server", "server.py", args, ROUNDS="1", LOCAL_EPOCHS="1", MIN_CLIENTS="2",
                         RESULTS_DIR=str(self.tmp), BIND=self.addr, **env)
        deadline = time.monotonic() + 120
        while not _port_open(self.port):
            if srv.poll() is not None:
                self.fail(f"server exited early ({srv.returncode}):\n{self.tail('server')}")
            if time.monotonic() > deadline:
                self.fail(f"server never listened on {self.addr}:\n{self.tail('server')}")
            time.sleep(0.1)
        return srv

    def start_clients(self, mode="mtls", env_paths=False):
        for cid in ("1", "2"):
            args, env = [], {}
            if mode == "insecure":
                args = ["--insecure"]
            elif mode == "tls":
                args = ["--ca-cert", self.certs["ca"]]
            else:
                cert, key = self.certs[f"client{cid}"]
                if env_paths:
                    env = {"FL_CA_CERT": self.certs["ca"], "FL_CLIENT_CERT": cert, "FL_CLIENT_KEY": key}
                else:
                    args = ["--ca-cert", self.certs["ca"], "--cert", cert, "--key", key]
            if env_paths:
                env.update(DP_CLIP="0.1", DP_NOISE_MULT="0.02")
            else:
                args.extend(["--dp-clip", "0.1", "--dp-noise-mult", "0"])
            self.spawn(f"client{cid}", "client.py", args, CLIENT_ID=cid, FL_SERVER=self.addr, **env)

    def assert_completed(self, tls_enabled, mtls_enabled, noise=0.0):
        end = time.monotonic() + self.TIMEOUT_S
        for name, proc in self.procs:
            try:
                proc.wait(timeout=max(1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                self.fail(f"{name} timed out after {self.TIMEOUT_S}s:\n{self.tail(name)}")
            self.assertEqual(proc.returncode, 0, f"{name} exit {proc.returncode}:\n{self.tail(name)}")
        path = self.tmp / "results_physical_seed42.json"
        self.assertTrue(path.is_file(), list(self.tmp.iterdir()))
        res = json.loads(path.read_text())
        self.assertEqual(res["config"]["seed"], 42)
        self.assertEqual(res["config"]["rounds"], 1)
        self.assertIs(res["config"]["tls_enabled"], tls_enabled)
        self.assertIs(res["config"]["mtls_enabled"], mtls_enabled)
        self.assertEqual(res["config"]["dp_clip"], 0.1)
        self.assertEqual(res["config"]["dp_noise_mult"], noise)
        mc = res["metrics_centralized"]
        self.assertEqual(len(mc["model_rmse"]), 2)
        for key in ("model_rmse", "baseline_rmse", "nasa_score"):
            for _, value in mc[key]:
                self.assertTrue(np.isfinite(value), key)
        mf = res["metrics_distributed_fit"]
        for key in ("train_time_s", "peak_rss_mb", *server_mod.SECURITY_KEYS):
            self.assertIn(key, mf)
            self.assertEqual(len(mf[key]), 1)
            self.assertTrue(np.isfinite(mf[key][0][1]))
        self.assertEqual(len(res["client_security"]), 2)
        self.assertEqual(len({r["client_id"] for r in res["client_security"]}), 2)
        for record in res["client_security"]:
            self.assertIs(record["tls_enabled"], tls_enabled)
            self.assertIs(record["mtls_enabled"], mtls_enabled)
            self.assertEqual(record["dp_clip"], 0.1)
            self.assertEqual(record["dp_noise_mult"], noise)

    def test_server_and_two_clients_one_round(self):
        self.start_server("insecure")
        self.start_clients("insecure")
        self.assert_completed(False, False)
        for name, _ in self.procs:
            self.assertIn("WARNING: INSECURE plaintext", self.tail(name))

    def test_mtls_server_and_two_clients_one_round(self):
        self.start_server()
        self.start_clients()
        self.assert_completed(True, True)

    def test_mtls_environment_paths_and_dp_noise(self):
        self.start_server(env_paths=True)
        self.start_clients(env_paths=True)
        self.assert_completed(True, True, noise=0.02)

    def test_explicit_server_only_tls(self):
        self.start_server("tls")
        self.start_clients("tls")
        self.assert_completed(True, False)

    def test_mtls_rejects_missing_untrusted_identity_and_untrusted_server(self):
        import grpc
        server = self.start_server()
        rogue = ephemeral_certificates(self.tmp / "rogue")
        ca = Path(self.certs["ca"]).read_bytes()
        good_cert, good_key = (Path(p).read_bytes() for p in self.certs["client1"])
        bad_cert, bad_key = (Path(p).read_bytes() for p in rogue["client1"])
        for name, credentials in (
            ("missing client certificate", grpc.ssl_channel_credentials(root_certificates=ca)),
            ("untrusted client certificate", grpc.ssl_channel_credentials(
                root_certificates=ca, private_key=bad_key, certificate_chain=bad_cert)),
            ("untrusted server", grpc.ssl_channel_credentials(
                root_certificates=Path(rogue["ca"]).read_bytes(), private_key=good_key,
                certificate_chain=good_cert)),
        ):
            with self.subTest(name=name), grpc.secure_channel(self.addr, credentials) as channel:
                # An actual RPC must fail, not merely an idle TCP readiness probe.
                from flwr.proto.transport_pb2_grpc import FlowerServiceStub
                from flwr.proto.transport_pb2 import ClientMessage
                with self.assertRaises(grpc.RpcError) as caught:
                    next(FlowerServiceStub(channel).Join(iter([ClientMessage()]), timeout=3))
                self.assertEqual(caught.exception.code(), grpc.StatusCode.UNAVAILABLE)
            self.assertIsNone(server.poll(), self.tail("server"))
            self.assertFalse((self.tmp / "results_physical_seed42.json").exists())
        # Recovery proves rejection did not break the listener or admit a rogue node.
        self.start_clients()
        self.assert_completed(True, True)


if __name__ == "__main__":
    unittest.main()
