"""Tests for the presentation dashboard (dashboard/): data invariants, generator, server, UI.

Run with the standard library runner:

    python3 -m unittest tests/test_dashboard.py -v
    CMAPSS_ZIP=/path/to/CMAPSSData.zip python3 -m unittest tests/test_dashboard.py -v

* Data / generator / server / ``node --check`` tests always run (``node`` must be on PATH).
* The bit-identity regeneration test runs only when ``CMAPSS_ZIP`` points at CMAPSSData.zip.
* The headless-browser test uses Selenium (Chrome, falling back to Safari) and is skipped
  only when no browser driver can be started.
"""
import functools
import http.server
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True  # keep dashboard/ free of __pycache__ artifacts

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"
DATA_PATH = DASH / "demo_data.json"
INDEX_PATH = DASH / "index.html"
SERVER_PATH = DASH / "server.py"
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import diagnostics as D  # noqa: E402

# load by path: the repo root also has a (Flower) server.py
G = _load("dashboard_generate_demo_data", DASH / "generate_demo_data.py")
dash_server = _load("dashboard_server", SERVER_PATH)
from common import SEQ_LEN, USEFUL_SENSORS, CmapssRegressor, set_seed  # noqa: E402

CMAPSS_ZIP = os.environ.get("CMAPSS_ZIP", "")
HAS_CMAPSS = bool(CMAPSS_ZIP) and os.path.isfile(CMAPSS_ZIP)

TAB_LABELS = {
    "it": {"tabAssetBtn": "1. RUL Asset Live (Quanto manca al guasto?)",
           "tabClusterBtn": "2. Cluster Federato (2x RPi4 + Hub)",
           "tabBusinessBtn": "3. Sicurezza & Impatto Aziendale",
           "diagTitle": "Diagnostica Edge: quale sottosistema guida il calo di RUL?"},
    "en": {"tabAssetBtn": "1. Live Asset RUL (Time to Failure)",
           "tabClusterBtn": "2. Federated Cluster (2x RPi4 + Hub)",
           "tabBusinessBtn": "3. Safety & Business Impact",
           "diagTitle": "Edge Diagnostics: which subsystem is driving the RUL drop?"},
}
SUBSYSTEM_IT = {"Fan": "Fan", "LPC": "LPC", "HPC": "HPC", "Turbine": "Turbina"}
SEVERITY_IT = {"crit": "FERMO PROGRAMMATO", "warn": "PRE-ALLARME", "ok": "OPERATIVO"}


def load_demo_data():
    with open(DATA_PATH) as f:
        return json.load(f)


# --------------------------------------------------------------------------- helpers


def synthetic_engine_frame(n_units=4, n_cycles=70, n_regimes=1, seed=0):
    """Raw C-MAPSS-like frame: flat healthy start, then an HPC (s3/s11) degradation ramp."""
    rng = np.random.RandomState(seed)
    regime_settings = np.array([[0, 0, 100], [10, 0.25, 100], [20, 0.7, 100]], dtype=float)[:n_regimes]
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            r = rng.randint(len(regime_settings))
            st = regime_settings[r] + rng.normal(0, 0.001, 3)
            vals = {s: 100.0 + 10 * i + 5.0 * r + rng.normal(0, 0.5) for i, s in enumerate(D.FAULT_SENSORS)}
            ramp = max(0, c - 35) / 35.0
            vals["s3"] += 8 * ramp
            vals["s11"] += 6 * ramp
            rows.append(dict(unit=u, cycle=c, setting1=st[0], setting2=st[1], setting3=st[2], **vals))
    return pd.DataFrame(rows)


def start_dashboard_server(timeout=20.0):
    """Launch dashboard/server.py on an ephemeral port; return (proc, base_url)."""
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_PATH), "--port", "0", "--no-browser"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    line_box = []
    reader = threading.Thread(target=lambda: line_box.append(proc.stdout.readline()), daemon=True)
    reader.start()
    reader.join(timeout)
    line = line_box[0] if line_box else ""
    m = re.search(r"http://127\.0\.0\.1:(\d+)/", line)
    if not m or m.group(1) == "0":
        stop_dashboard_server(proc)
        raise AssertionError(f"server did not report a bound port, first line: {line!r}")
    return proc, f"http://127.0.0.1:{m.group(1)}"


def stop_dashboard_server(proc, timeout=10.0):
    """Graceful Ctrl+C-style shutdown; return (returncode, remaining output)."""
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout)
    out = proc.stdout.read() if proc.stdout else ""
    if proc.stdout:
        proc.stdout.close()
    return proc.returncode, out


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def inline_scripts(html):
    return re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.S)


# --------------------------------------------------------------------------- 1. demo_data.json


class DemoDataInvariantsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_demo_data()

    def test_top_level_and_engines(self):
        for key in ("model_info", "engines", "diagnostics_meta", "cluster", "benchmarks"):
            self.assertIn(key, self.data)
        engines = self.data["engines"]
        self.assertEqual(len(engines), 8)
        self.assertEqual(sorted({e["subset"] for e in engines}), ["FD001", "FD002"])
        for subset in ("FD001", "FD002"):
            self.assertEqual(sum(e["subset"] == subset for e in engines), 4, subset)
        self.assertEqual(len({e["id"] for e in engines}), 8, "engine ids must be unique")
        for e in engines:
            self.assertTrue(e["id"].startswith(e["subset"] + "-U"), e["id"])
            self.assertIn(e["stage"], ("critical", "warning", "midlife", "healthy"))

    def test_diagnostics_meta(self):
        meta = self.data["diagnostics_meta"]
        self.assertEqual(meta["subsystems"], ["Fan", "LPC", "HPC", "Turbine"])
        self.assertEqual(meta["excluded_channels"], ["setting1", "setting2", "setting3"])
        sensors = [s for k in meta["subsystems"] for s in meta["subsystem_sensors"][k]]
        self.assertEqual(sorted(sensors), sorted(meta["sensor_info"]))
        self.assertEqual(meta["privacy"]["raw_sensor_bytes_transmitted"], 0)

    def test_per_engine_alignment_and_conservation(self):
        meta = self.data["diagnostics_meta"]
        known = set(meta["sensor_info"])
        for e in self.data["engines"]:
            with self.subTest(engine=e["id"]):
                d = e["diagnostics"]
                n = len(e["cycles"])
                self.assertGreater(n, 0)
                self.assertEqual(len(d["subsystem_share"]), n)
                self.assertEqual(len(d["top_contributors"]), n)
                self.assertEqual(len(d["top_drifting"]), n)
                self.assertEqual(len(d["drift_index"]), n)
                for key in ("true_rul", "pred_local_mse", "pred_fedavg_mse", "pred_fedavg_asym"):
                    self.assertEqual(len(e[key]), n, key)
                for s, arr in e["sensors"].items():
                    self.assertEqual(len(arr), n, s)
                self.assertEqual(e["cycles"], sorted(e["cycles"]))
                self.assertEqual(e["cycles"][-1], e["total_observed_cycles"])
                for i, row in enumerate(d["subsystem_share"]):
                    self.assertEqual(len(row), 4)
                    self.assertAlmostEqual(sum(row), 100.0, delta=1e-6, msg=f"cycle index {i}")
                    self.assertTrue(all(v >= 0 for v in row))
                for top, drift in zip(d["top_contributors"], d["top_drifting"]):
                    self.assertEqual(len(top), 3)
                    self.assertEqual(len(drift), 3)
                    # every code must resolve in sensorLabel() on the page
                    self.assertTrue({c for c, _ in top + drift} <= known)
                    pcts = [p for _, p in top]
                    self.assertEqual(pcts, sorted(pcts, reverse=True))
                    zs = [abs(z) for _, z in drift]
                    self.assertEqual(zs, sorted(zs, reverse=True))
                fs = d["final_summary"]
                self.assertEqual(fs["privacy"]["raw_sensor_bytes_transmitted"], 0)
                self.assertEqual(fs["engine_id"], e["id"])
                self.assertEqual(fs["cycle"], e["cycles"][-1])
                self.assertAlmostEqual(sum(fs["subsystem_shares_pct"].values()), 100.0, delta=1e-6)
                self.assertEqual(list(fs["subsystem_shares_pct"].values()), d["subsystem_share"][-1])


# --------------------------------------------------------------------------- 2. generate_demo_data.py


class GeneratorUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = synthetic_engine_frame()
        cls.df = df
        cls.scales = {"min": df[USEFUL_SENSORS].min().to_dict(), "max": df[USEFUL_SENSORS].max().to_dict()}
        set_seed(3)
        cls.model = CmapssRegressor()
        cls.baseline = D.fit_healthy_baseline(df, 1)
        cls.raw = df[df.unit == 2]
        cls.out = G.engine_diagnostics(cls.model, cls.scales, cls.baseline, cls.raw, "SYN-U2")

    def test_diagnostics_meta_matches_module_and_committed_json(self):
        meta = G.diagnostics_meta()
        self.assertEqual(meta["subsystems"], list(D.SUBSYSTEMS))
        self.assertEqual(meta["excluded_channels"], list(D.SETTINGS))
        self.assertEqual(set(meta["sensor_info"]), set(D.FAULT_SENSORS))
        self.assertEqual(meta["privacy"]["raw_sensor_bytes_transmitted"], 0)
        self.assertFalse(meta["privacy"]["contains_raw_sensor_values"])
        for key in ("model", "attribution_formula", "drift_formula", "evaluation_boundary", "subsystem_labels"):
            self.assertTrue(meta[key], key)
        # JSON round-trip equals what the dashboard ships with
        self.assertEqual(json.loads(json.dumps(meta)), load_demo_data()["diagnostics_meta"])

    def test_engine_diagnostics_shapes(self):
        out = self.out
        n = 70 - SEQ_LEN + 1
        self.assertEqual(out["method"], "baseline_x_gradient")
        for key in ("subsystem_share", "top_contributors", "top_drifting", "drift_index"):
            self.assertEqual(len(out[key]), n, key)
        for top, drift in zip(out["top_contributors"], out["top_drifting"]):
            self.assertEqual(len(top), 3)
            self.assertEqual(len(drift), 3)
            for code, _ in top + drift:
                self.assertIn(code, D.FAULT_SENSORS)
                self.assertNotIn(code, D.SETTINGS)
            self.assertEqual([p for _, p in top], sorted((p for _, p in top), reverse=True))
            self.assertEqual([abs(z) for _, z in drift], sorted((abs(z) for _, z in drift), reverse=True))
        fs = out["final_summary"]
        self.assertEqual(fs["engine_id"], "SYN-U2")
        self.assertEqual(fs["cycle"], 70)
        self.assertEqual(fs["privacy"]["raw_sensor_bytes_transmitted"], 0)
        self.assertEqual(list(fs["subsystem_shares_pct"].values()), out["subsystem_share"][-1])
        # injected HPC fault ramp must dominate the final drift ranking
        self.assertEqual({c for c, _ in out["top_drifting"][-1][:2]}, {"s3", "s11"})
        json.dumps(out)  # serializable, as written into demo_data.json

    def test_round_to_total_conservation(self):
        raw = D.EdgeDiagnostics(self.model, self.scales, self.baseline).diagnose_engine(self.raw, "SYN-U2")
        for i, (rounded, exact) in enumerate(zip(self.out["subsystem_share"], raw["subsystem_share"])):
            self.assertAlmostEqual(sum(rounded), 100.0, delta=1e-6, msg=f"index {i}")
            self.assertEqual(rounded, D.round_to_total(exact, 1))
            self.assertTrue(np.all(np.abs(np.asarray(rounded) - exact) < 0.1 + 1e-9))
            self.assertTrue(all(round(v, 1) == v for v in rounded))
        # adversarial: naive rounding of thirds loses 0.1, largest-remainder keeps 100.0
        self.assertAlmostEqual(sum(D.round_to_total([100 / 3] * 3, 1)), 100.0, delta=1e-9)
        self.assertAlmostEqual(sum(D.round_to_total([12.345, 12.345, 25.31, 50.0], 1)), 100.0, delta=1e-9)

    def test_engine_diagnostics_deterministic(self):
        again = G.engine_diagnostics(self.model, self.scales, self.baseline, self.raw, "SYN-U2")
        self.assertEqual(json.dumps(again, sort_keys=True), json.dumps(self.out, sort_keys=True))


@unittest.skipUnless(HAS_CMAPSS, "set CMAPSS_ZIP=/path/to/CMAPSSData.zip to regenerate demo_data.json")
class GeneratorRegenerationTest(unittest.TestCase):
    """Re-run generate_demo_data.main() into a temp dir; output must be bit-identical."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cwd = os.getcwd()
        try:
            # main() writes next to Path(__file__): redirect it to the temp dir
            with mock.patch.object(G, "__file__", os.path.join(cls.tmp.name, "generate_demo_data.py")):
                G.main()
        finally:
            os.chdir(cwd)
        cls.generated_path = Path(cls.tmp.name) / "demo_data.json"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_predictions_identical(self):
        with open(self.generated_path) as f:
            gen = json.load(f)
        ref = load_demo_data()
        self.assertEqual([e["id"] for e in gen["engines"]], [e["id"] for e in ref["engines"]])
        for g, r in zip(gen["engines"], ref["engines"]):
            for key in ("cycles", "true_rul", "pred_local_mse", "pred_fedavg_mse", "pred_fedavg_asym"):
                self.assertEqual(g[key], r[key], f"{g['id']} {key}")
            self.assertEqual(g["diagnostics"], r["diagnostics"], g["id"])

    def test_file_bit_identical(self):
        self.assertEqual(self.generated_path.read_bytes(), DATA_PATH.read_bytes())


# --------------------------------------------------------------------------- 3. server.py


class ServerSmokeTest(unittest.TestCase):
    def test_live_endpoints_and_clean_shutdown(self):
        proc, base = start_dashboard_server()
        try:
            status, headers, body = http_get(base + "/api/health")
            self.assertEqual(status, 200)
            self.assertIn("application/json", headers["Content-Type"])
            self.assertEqual(json.loads(body), {"status": "ok", "has_data": True})

            status, _, body = http_get(base + "/")
            self.assertEqual(status, 200)
            html = body.decode("utf-8")
            self.assertIn('id="diagCard"', html)
            self.assertEqual(html, INDEX_PATH.read_text(encoding="utf-8"))

            status, headers, body = http_get(base + "/demo_data.json")
            self.assertEqual(status, 200)
            self.assertEqual(len(json.loads(body)["engines"]), 8)

            status, _, _ = http_get(base + "/does-not-exist.json")
            self.assertEqual(status, 404)
        finally:
            code, out = stop_dashboard_server(proc)
        self.assertEqual(code, 0, out)
        self.assertIn("Shutting down", out)

    def test_no_directory_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "sub"))
            Path(tmp, "sub", "secret.txt").write_text("x")
            handler = functools.partial(dash_server.DemoHandler, directory=tmp)
            httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            th = threading.Thread(target=httpd.serve_forever, daemon=True)
            th.start()
            try:
                base = f"http://127.0.0.1:{httpd.server_address[1]}"
                status, _, body = http_get(base + "/sub/")
                self.assertEqual(status, 404)
                self.assertNotIn(b"secret.txt", body)
                self.assertEqual(http_get(base + "/sub/secret.txt")[0], 200)
            finally:
                httpd.shutdown()
                httpd.server_close()
                th.join(5)


# --------------------------------------------------------------------------- 4. index.html


class IndexStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX_PATH.read_text(encoding="utf-8")

    def test_inline_script_node_check(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "node must be on PATH for the mandatory `node --check`")
        scripts = inline_scripts(self.html)
        self.assertEqual(len(scripts), 1)
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "dashboard_inline.js")
            Path(good).write_text(scripts[0], encoding="utf-8")
            res = subprocess.run([node, "--check", good], capture_output=True, text=True, timeout=60)
            self.assertEqual(res.returncode, 0, res.stderr)
            # control: the check really rejects broken syntax
            bad = os.path.join(tmp, "broken.js")
            Path(bad).write_text(scripts[0] + "\nfunction (", encoding="utf-8")
            res = subprocess.run([node, "--check", bad], capture_output=True, text=True, timeout=60)
            self.assertNotEqual(res.returncode, 0)

    def test_script_dom_ids_exist(self):
        ids = set(re.findall(r'\bid="([^"]+)"', self.html))
        script = inline_scripts(self.html)[0]
        used = set(re.findall(r'getElementById\("([^"$]+)"\)', script))
        self.assertTrue(used, "no getElementById calls found")
        self.assertEqual(used - ids, set())
        for tab in ("tab-asset", "tab-cluster", "tab-business", "cycleSlider", "engineSelect", "langBtn",
                    "diagCard", "diagBar", "diagTop", "diagDrift", "diagTrend", "diagSummary"):
            self.assertIn(tab, ids)


ERROR_HOOK = """
window.__errs = [];
window.addEventListener('error', e => window.__errs.push('error: ' + e.message));
window.addEventListener('unhandledrejection', e => window.__errs.push('rejection: ' + (e.reason && e.reason.message || e.reason)));
"""

SCRUB_JS = """
const slider = document.getElementById('cycleSlider');
const max = parseInt(slider.max, 10);
const snaps = [];
for (let i = 0; i <= max; i++) {
  slider.value = String(i);
  slider.dispatchEvent(new Event('input', {bubbles: true}));
  const q = (id) => document.getElementById(id);
  snaps.push({
    value: parseInt(slider.value, 10),
    bar: [...q('diagBar').querySelectorAll('.diag-seg')].map(s => s.style.width),
    top: [...q('diagTop').querySelectorAll('strong')].map(s => s.textContent),
    drift: [...q('diagDrift').querySelectorAll('strong')].map(s => s.textContent),
    trendPaths: q('diagTrend').querySelectorAll('svg path').length,
    trend: q('diagTrend').innerHTML,
    summary: q('diagSummary').textContent,
    readout: q('cycleReadout').textContent,
    errs: window.__errs.length,
  });
}
return {max: max, snaps: snaps};
"""


def launch_driver():
    """Return (driver, kind) for a headless browser, or (None, reasons)."""
    reasons = []
    try:
        from selenium import webdriver
        from selenium.common.exceptions import WebDriverException
    except ImportError as exc:
        return None, [f"selenium not installed ({exc})"]

    def chrome_options():
        o = webdriver.ChromeOptions()
        for arg in ("--headless=new", "--no-sandbox", "--disable-gpu", "--window-size=1400,1000"):
            o.add_argument(arg)
        o.set_capability("goog:loggingPrefs", {"browser": "ALL"})
        return o

    try:
        return webdriver.Chrome(options=chrome_options()), "chrome"
    except WebDriverException as exc:
        reasons.append(f"chrome: {str(exc).splitlines()[0]}")
    # a stale chromedriver on PATH: let Selenium Manager resolve a matching one
    try:
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.selenium_manager import SeleniumManager
        paths = SeleniumManager().binary_paths(["--browser", "chrome", "--skip-driver-in-path"])
        return webdriver.Chrome(service=Service(paths["driver_path"]), options=chrome_options()), "chrome"
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"chrome via selenium-manager: {str(exc).splitlines()[0] if str(exc) else exc!r}")
    try:
        return webdriver.Safari(), "safari"
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"safari: {str(exc).splitlines()[0] if str(exc) else exc!r}")
    return None, reasons


class DashboardBrowserTest(unittest.TestCase):
    """Headless DOM interaction against a live dashboard/server.py."""

    @classmethod
    def setUpClass(cls):
        cls.data = load_demo_data()
        cls.driver, info = launch_driver()
        if cls.driver is None:
            raise unittest.SkipTest("no headless browser driver available: " + "; ".join(info))
        cls.kind = info
        cls.driver.set_script_timeout(120)
        if cls.kind == "chrome":
            cls.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": ERROR_HOOK})
        try:
            cls.proc, cls.base = start_dashboard_server()
        except Exception:
            cls.driver.quit()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()
        stop_dashboard_server(cls.proc)

    # -- helpers
    def load(self):
        from selenium.webdriver.support.ui import WebDriverWait
        self.driver.get(self.base + "/")
        if self.kind != "chrome":
            self.driver.execute_script(ERROR_HOOK)
        WebDriverWait(self.driver, 20).until(lambda d: d.execute_script(
            "return document.querySelectorAll('#engineSelect option').length === 8"
            " && document.querySelectorAll('#diagBar .diag-seg').length === 4"))

    def js_errors(self):
        errs = list(self.driver.execute_script("return window.__errs || []"))
        if self.kind == "chrome":
            errs += [m["message"] for m in self.driver.get_log("browser")
                     if m["level"] == "SEVERE" and "favicon.ico" not in m["message"]]
        return errs

    def text(self, el_id):
        return self.driver.execute_script(f"return document.getElementById('{el_id}').textContent")

    def setUp(self):
        self.load()

    def tearDown(self):
        self.assertEqual(self.js_errors(), [])

    # -- tests
    def test_tabs_switch(self):
        from selenium.webdriver.common.by import By
        d = self.driver
        for name in ("cluster", "business", "asset", "business", "cluster", "asset"):
            d.find_element(By.CSS_SELECTOR, f'[data-tab="{name}"]').click()
            for other in ("asset", "cluster", "business"):
                section = d.find_element(By.ID, f"tab-{other}")
                btn = d.find_element(By.CSS_SELECTOR, f'[data-tab="{other}"]')
                self.assertEqual(section.is_displayed(), other == name, f"{name}: tab-{other}")
                self.assertEqual("active" in btn.get_attribute("class").split(), other == name)
        d.find_element(By.CSS_SELECTOR, '[data-tab="cluster"]').click()
        self.assertTrue(d.find_elements(By.CSS_SELECTOR, "#roundRmseChart svg path"))
        d.find_element(By.CSS_SELECTOR, '[data-tab="business"]').click()
        self.assertGreater(len(d.find_elements(By.CSS_SELECTOR, "#asymTable tbody tr")), 0)
        self.assertGreater(len(d.find_elements(By.CSS_SELECTOR, "#imbalanceTable tbody tr")), 0)

    def test_diag_card_renders_on_tab1(self):
        from selenium.webdriver.common.by import By
        d = self.driver
        card = d.find_element(By.ID, "diagCard")
        self.assertTrue(card.is_displayed())
        self.assertNotIn("hidden", card.get_attribute("class").split())
        self.assertEqual(self.text("diagTitle"), TAB_LABELS["it"]["diagTitle"])
        self.assertEqual(len(d.find_elements(By.CSS_SELECTOR, "#diagTop .diag-row")), 3)
        self.assertEqual(len(d.find_elements(By.CSS_SELECTOR, "#diagDrift .diag-row")), 3)
        self.assertTrue(self.text("diagSummary"))

    def test_scrub_slider_all_engines(self):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import Select
        d = self.driver
        select = Select(d.find_element(By.ID, "engineSelect"))
        slider = d.find_element(By.ID, "cycleSlider")
        for idx, eng in enumerate(self.data["engines"]):
            with self.subTest(engine=eng["id"]):
                select.select_by_index(idx)  # real change event -> onEngineChange()
                n = len(eng["cycles"])
                self.assertEqual(int(slider.get_attribute("max")), n - 1)
                self.assertEqual(int(slider.get_attribute("value")), n - 1)
                self.assertIn(eng["id"], select.first_selected_option.text)

                # real keyboard input on the range control
                slider.send_keys(Keys.HOME)
                self.assertEqual(int(slider.get_attribute("value")), 0)
                self.assertIn(f"Ciclo {eng['cycles'][0]} /", self.text("cycleReadout"))
                slider.send_keys(Keys.END)
                self.assertIn(f"Ciclo {eng['cycles'][-1]} /", self.text("cycleReadout"))

                res = d.execute_script(SCRUB_JS)
                self.assertEqual(res["max"], n - 1)
                snaps = res["snaps"]
                self.assertEqual(len(snaps), n)
                diag = eng["diagnostics"]
                for i, s in enumerate(snaps):
                    shares = diag["subsystem_share"][i]
                    self.assertEqual(s["value"], i)
                    self.assertEqual(s["errs"], 0, f"JS error at index {i}")
                    self.assertEqual([float(w.rstrip("%")) for w in s["bar"]], shares, f"diagBar @ {i}")
                    self.assertEqual(s["top"], [c for c, _ in diag["top_contributors"][i]], f"diagTop @ {i}")
                    self.assertEqual(s["drift"], [c for c, _ in diag["top_drifting"][i]], f"diagDrift @ {i}")
                    self.assertEqual(s["trendPaths"], 4, f"diagTrend @ {i}")
                    prim = self.data["diagnostics_meta"]["subsystems"][shares.index(max(shares))]
                    p_safe = eng["pred_fedavg_asym"][i]
                    sev = SEVERITY_IT["crit" if p_safe <= 20 else "warn" if p_safe <= 45 else "ok"]
                    self.assertIn(f"Sottosistema principale {SUBSYSTEM_IT[prim]} ({max(shares):.1f}%",
                                  s["summary"], f"diagSummary @ {i}")
                    self.assertTrue(s["summary"].startswith(sev), f"diagSummary severity @ {i}")
                    self.assertIn(f"Ciclo {eng['cycles'][i]} / {eng['total_observed_cycles']}", s["readout"])
                if n > 1:
                    self.assertNotEqual(snaps[0]["trend"], snaps[-1]["trend"])

    def test_lang_toggle(self):
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import Select
        d = self.driver
        btn = d.find_element(By.ID, "langBtn")
        Select(d.find_element(By.ID, "engineSelect")).select_by_index(5)
        d.execute_script("const s=document.getElementById('cycleSlider'); s.value='10';"
                         "s.dispatchEvent(new Event('input'));")
        eng = self.data["engines"][5]

        def hours_labels():
            return d.execute_script("return [...document.querySelectorAll('#hoursSelect option')]"
                                    ".map(o => o.textContent)")

        self.assertEqual(btn.text, "EN")
        for lang, btn_text, cyc_word, summary_word in (("en", "IT", "Cycle", "Predicted RUL"),
                                                        ("it", "EN", "Ciclo", "RUL stimata"),
                                                        ("en", "IT", "Cycle", "Predicted RUL")):
            with self.subTest(lang=lang):
                btn.click()  # toggleLang()
                self.assertEqual(btn.text, btn_text)
                for el_id, label in TAB_LABELS[lang].items():
                    self.assertEqual(self.text(el_id), label)
                labels = hours_labels()
                if lang == "en":
                    self.assertEqual(labels, ["Cycles only", "1 cycle = 0.5 h", "1 cycle = 1.0 h",
                                              "1 cycle = 2.0 h", "1 cycle = 4.0 h"])
                else:
                    self.assertEqual(labels, ["Solo cicli", "1 ciclo = 0.5 h", "1 ciclo = 1.0 h",
                                              "1 ciclo = 2.0 h", "1 ciclo = 4.0 h"])
                self.assertIn(summary_word, self.text("diagSummary"))
                # engine + cycle position survive the relabel
                self.assertEqual(d.execute_script("return document.getElementById('engineSelect').selectedIndex"), 5)
                self.assertEqual(self.text("cycleReadout"),
                                 f"{cyc_word} {eng['cycles'][10]} / {eng['total_observed_cycles']}")
                self.assertEqual(len(d.find_elements(By.CSS_SELECTOR, "#engineSelect option")), 8)

    def test_lang_toggle_before_data_loaded(self):
        if self.kind != "chrome":
            self.skipTest("needs Chrome DevTools request blocking")
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        d = self.driver
        d.execute_cdp_cmd("Network.enable", {})
        d.execute_cdp_cmd("Network.setBlockedURLs", {"urls": ["*demo_data.json*"]})
        try:
            d.get(self.base + "/")
            WebDriverWait(d, 10).until(lambda drv: drv.execute_script("return window.__errs.length > 0"))
            d.execute_script("window.__errs = []")
            d.get_log("browser")  # drain the expected blocked-request / boot() failure
            d.find_element(By.ID, "langBtn").click()
            self.assertEqual(d.execute_script("return window.__errs"), [])
            self.assertEqual(self.text("tabAssetBtn"), TAB_LABELS["en"]["tabAssetBtn"])
            self.assertEqual(d.find_element(By.ID, "langBtn").text, "IT")
        finally:
            d.execute_cdp_cmd("Network.setBlockedURLs", {"urls": []})
            d.execute_cdp_cmd("Network.disable", {})
            self.load()


if __name__ == "__main__":
    unittest.main()
