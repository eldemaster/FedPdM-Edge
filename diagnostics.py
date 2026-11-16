"""Local edge diagnostics: sensor-level degradation attribution for FedPdM-Edge.

Answers "which physical module / sensor is driving the RUL drop?" for a single
engine, computed entirely on the edge node that owns the raw telemetry. Nothing
in this module opens a network connection; its outputs are aggregate shares,
standardized drift scores and a text summary, never raw sensor readings.

Pipeline per sliding window of ``SEQ_LEN`` (=30) cycles:

1. **Attribution** (model-based). For window ``x`` (W x C, min-max scaled, the
   exact input of ``CmapssRegressor``) and predicted RUL ``y = f(x)``::

       A_c = (1/W) * sum_t | (x[t,c] - b[t,c]) * dy/dx[t,c] |

   ``method='baseline_x_gradient'`` (default when a healthy baseline is known):
   ``b`` is the engine's healthy-baseline expectation for the same operating
   regime, so ``A_c`` is a first-order estimate of how much channel ``c``'s
   *deviation from healthy* moves the RUL estimate.
   ``method='input_x_gradient'``: ``b = 0`` (plain |x * grad|).
   ``method='gradient'``: ``A_c = (1/W) * sum_t |dy/dx[t,c]|`` (pure saliency).

   Operating settings ``setting1..3`` are model inputs but are *not* fault
   carriers: their attribution is forced to exactly 0 before normalization.
   Per-sensor shares are ``100 * A_c / sum_{c in fault sensors} A_c`` and
   subsystem shares are sums of their sensors' shares, so both sum to 100.

2. **Healthy-baseline drift** (model-free). Each raw cycle is first assigned
   an operating regime (nearest of ``k`` setting centroids; ``k=1`` for the
   single-condition FD001/FD003, ``k=6`` for FD002/FD004) and standardized
   with that regime's fleet mean/std. The engine's own first ``healthy_cycles``
   standardized cycles define its healthy offset; drift is::

       z[t,c] = (u[t,c] - mean_{t' < H} u[t',c]) / sigma_healthy_c

   where ``sigma_healthy_c`` is the pooled within-engine early-life noise of
   the local fleet. Window drift is the mean of ``z`` over the window.

3. **Maintenance summary**: top-3 contributing and top-3 drifting sensors with
   physical names, primary subsystem, severity from predicted RUL, and explicit
   privacy / evaluation-boundary metadata.

Channel definitions follow Table 2 of Saxena et al., "Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation" (PHM08), shipped in
``CMAPSSData.zip``; ``readme.txt`` defines columns 3-5 as operational settings
and columns 6-26 as sensor measurements 1..21 in that order.

Evaluation boundary: C-MAPSS gives only unit-level RUL. FD001/FD002 contain a
single fault mode (HPC degradation) and FD003/FD004 two (HPC, Fan), but no
per-unit fault-mode labels are published. Attribution is therefore validated
by physical consistency and unsupervised separability, not supervised
component-classification accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from common import SEQ_LEN, USEFUL_SENSORS, CmapssRegressor  # noqa: F401  (re-exported for callers)

# --------------------------------------------------------------------------- channel map

SETTINGS = ("setting1", "setting2", "setting3")
FAULT_SENSORS: List[str] = [c for c in USEFUL_SENSORS if c not in SETTINGS]
SUBSYSTEMS = ("Fan", "LPC", "HPC", "Turbine")

# Table 2 (Saxena et al. 2008), sensor index = position in the table.
SENSOR_INFO: Dict[str, Dict[str, str]] = {
    "s2": {"symbol": "T24", "name": "Total temperature at LPC outlet", "unit": "degR", "subsystem": "LPC"},
    "s3": {"symbol": "T30", "name": "Total temperature at HPC outlet", "unit": "degR", "subsystem": "HPC"},
    "s4": {"symbol": "T50", "name": "Total temperature at LPT outlet", "unit": "degR", "subsystem": "Turbine"},
    "s7": {"symbol": "P30", "name": "Total pressure at HPC outlet", "unit": "psia", "subsystem": "HPC"},
    "s8": {"symbol": "Nf", "name": "Physical fan speed", "unit": "rpm", "subsystem": "Fan"},
    "s9": {"symbol": "Nc", "name": "Physical core speed", "unit": "rpm", "subsystem": "HPC"},
    "s11": {"symbol": "Ps30", "name": "Static pressure at HPC outlet", "unit": "psia", "subsystem": "HPC"},
    "s12": {"symbol": "phi", "name": "Ratio of fuel flow to Ps30", "unit": "pps/psi", "subsystem": "Turbine"},
    "s13": {"symbol": "NRf", "name": "Corrected fan speed", "unit": "rpm", "subsystem": "Fan"},
    "s14": {"symbol": "NRc", "name": "Corrected core speed", "unit": "rpm", "subsystem": "HPC"},
    "s15": {"symbol": "BPR", "name": "Bypass ratio", "unit": "-", "subsystem": "Fan"},
    "s17": {"symbol": "htBleed", "name": "Bleed enthalpy", "unit": "-", "subsystem": "Turbine"},
    "s20": {"symbol": "W31", "name": "HPT coolant bleed", "unit": "lbm/s", "subsystem": "Turbine"},
    "s21": {"symbol": "W32", "name": "LPT coolant bleed", "unit": "lbm/s", "subsystem": "Turbine"},
}
SUBSYSTEM_LABELS = {
    "Fan": "Fan (Nf, NRf, BPR)",
    "LPC": "Low-pressure compressor (T24)",
    "HPC": "High-pressure compressor / core spool (T30, P30, Ps30, Nc, NRc)",
    "Turbine": "Combustor + HPT/LPT hot section (T50, phi, htBleed, W31, W32)",
}
SUBSYSTEM_SENSORS: Dict[str, List[str]] = {
    s: [c for c in FAULT_SENSORS if SENSOR_INFO[c]["subsystem"] == s] for s in SUBSYSTEMS
}

FAULT_IDX = np.array([USEFUL_SENSORS.index(c) for c in FAULT_SENSORS])
SETTING_IDX = np.array([USEFUL_SENSORS.index(c) for c in SETTINGS])
_SUB_MATRIX = np.array(
    [[1.0 if SENSOR_INFO[c]["subsystem"] == s else 0.0 for s in SUBSYSTEMS] for c in FAULT_SENSORS]
)  # (14, 4)

N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
SEVERITY_THRESHOLDS = {"critical": 20.0, "warning": 45.0}  # predicted-RUL cut-offs (cycles), as the dashboard
ATTRIBUTION_METHODS = ("baseline_x_gradient", "input_x_gradient", "gradient")
PRIVACY_METADATA = {
    "computed_on": "local_edge_node",
    "raw_telemetry_location": "local_edge_only",
    "raw_sensor_bytes_transmitted": 0,
    "network_calls": 0,
    "contains_raw_sensor_values": False,
    "shared_fields": "aggregate attribution shares (%), standardized drift scores (sigma), predicted RUL, text",
}
EVALUATION_BOUNDARY = (
    "C-MAPSS provides unit-level RUL only; no per-unit component fault labels exist. "
    "Subsystem attribution is a model/physics-based explanation, not a supervised component diagnosis."
)

# --------------------------------------------------------------------------- scaling helpers


def scale_frame(values: np.ndarray, scales: dict, columns: Sequence[str] = USEFUL_SENSORS) -> np.ndarray:
    mn = np.array([scales["min"][c] for c in columns], dtype=np.float64)
    mx = np.array([scales["max"][c] for c in columns], dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) - mn) / (mx - mn + 1e-7)


def sliding_windows(arr: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """(T, C) -> (T-seq_len+1, seq_len, C). Short series are left-padded with their first row."""
    arr = np.asarray(arr)
    if len(arr) < seq_len:
        arr = np.concatenate([np.repeat(arr[:1], seq_len - len(arr), axis=0), arr], axis=0)
    return np.lib.stride_tricks.sliding_window_view(arr, seq_len, axis=0).transpose(0, 2, 1).copy()


# --------------------------------------------------------------------------- healthy baseline


@dataclass
class HealthyBaseline:
    """Operating-regime normalizer + pooled early-life noise, fitted on the node's local fleet."""

    centroids: np.ndarray        # (k, 3) in standardized setting space
    set_mean: np.ndarray         # (3,)
    set_std: np.ndarray          # (3,)
    regime_mean: np.ndarray      # (k, 14) raw-unit fault-sensor mean per regime
    regime_std: np.ndarray       # (k, 14)
    healthy_sigma: np.ndarray    # (14,) pooled early-life noise in regime-standardized units
    healthy_cycles: int = 30

    @property
    def n_regimes(self) -> int:
        return len(self.centroids)

    def regimes(self, settings: np.ndarray) -> np.ndarray:
        s = np.asarray(settings, dtype=np.float64)
        if s.size == 0:
            return np.zeros(0, dtype=np.intp)
        if s.shape[-1:] != (3,) or s.ndim > 2:
            raise ValueError(f"settings must have shape (3,) or (T, 3), got {s.shape}")
        z = (np.atleast_2d(s) - self.set_mean) / self.set_std
        return ((z[:, None, :] - self.centroids[None]) ** 2).sum(-1).argmin(1)

    def standardize(self, frame: pd.DataFrame) -> tuple:
        """Return (u, regime) with u = (x - mu_regime) / sd_regime for the 14 fault sensors."""
        reg = self.regimes(frame[list(SETTINGS)].values)
        x = frame[FAULT_SENSORS].values.astype(np.float64)
        return (x - self.regime_mean[reg]) / self.regime_std[reg], reg

    def engine_offset(self, u: np.ndarray) -> np.ndarray:
        return u[: max(1, min(self.healthy_cycles, len(u)))].mean(0)

    def drift(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-cycle signed drift z-scores (T, 14) relative to this engine's own healthy start."""
        u, _ = self.standardize(frame)
        return (u - self.engine_offset(u)) / self.healthy_sigma

    def healthy_reference(self, frame: pd.DataFrame, scales: dict) -> np.ndarray:
        """Per-cycle healthy expectation in the model's scaled input space (T, 17).

        Fault sensors: regime mean + engine healthy offset; settings: the observed
        settings (zero deviation, so they can never carry baseline attribution).
        """
        u, reg = self.standardize(frame)
        healthy_raw = self.regime_mean[reg] + self.regime_std[reg] * self.engine_offset(u)
        ref = frame[USEFUL_SENSORS].values.astype(np.float64).copy()
        ref[:, FAULT_IDX] = healthy_raw
        return scale_frame(ref, scales)


def fit_healthy_baseline(df_raw: pd.DataFrame, n_regimes: int = 1, healthy_cycles: int = 30,
                         seed: int = 0) -> HealthyBaseline:
    """Fit on the local node's own run-to-failure history (raw, unscaled frame)."""
    df_raw = df_raw.sort_values(["unit", "cycle"])
    s = df_raw[list(SETTINGS)].values.astype(np.float64)
    set_mean, set_std = s.mean(0), s.std(0) + 1e-9
    zs = (s - set_mean) / set_std
    if n_regimes <= 1:
        centroids = np.zeros((1, 3))
    else:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=n_regimes, n_init=10, random_state=seed).fit(zs)
        order = np.lexsort(km.cluster_centers_.T[::-1])  # deterministic regime ids
        centroids = km.cluster_centers_[order]
    reg = ((zs[:, None, :] - centroids[None]) ** 2).sum(-1).argmin(1)
    x = df_raw[FAULT_SENSORS].values.astype(np.float64)
    k = len(centroids)
    regime_mean = np.stack([x[reg == r].mean(0) for r in range(k)])
    regime_std = np.stack([x[reg == r].std(0) for r in range(k)])
    regime_std = np.where(regime_std < 1e-6, 1.0, regime_std)

    bl = HealthyBaseline(centroids, set_mean, set_std, regime_mean, regime_std,
                         np.ones(len(FAULT_SENSORS)), healthy_cycles)
    resid = []
    for _, g in df_raw.groupby("unit"):
        u, _ = bl.standardize(g)
        head = u[:healthy_cycles]
        resid.append(head - head.mean(0))
    sigma = np.concatenate(resid).std(0)
    bl.healthy_sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return bl


# --------------------------------------------------------------------------- attribution


def attribute_windows(model: torch.nn.Module, windows: np.ndarray, baseline: Optional[np.ndarray] = None,
                      method: str = "baseline_x_gradient", batch_size: int = 512) -> tuple:
    """Return (pred_rul (N,), channel attribution (N, 17)) with settings forced to 0.

    ``windows`` and ``baseline`` share shape (N, W, 17) in model input space.
    """
    if method not in ATTRIBUTION_METHODS:
        raise ValueError(f"unknown attribution method {method!r}")
    if method == "baseline_x_gradient" and baseline is None:
        method = "input_x_gradient"
    if len(windows) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, len(USEFUL_SENSORS)), dtype=np.float64)
    model.eval()
    preds, attrs = [], []
    for i in range(0, len(windows), batch_size):
        x = torch.tensor(np.asarray(windows[i:i + batch_size]), dtype=torch.float32, requires_grad=True)
        y = model(x)
        (g,) = torch.autograd.grad(y.sum(), x)
        g = g.detach().double().numpy()
        if method == "gradient":
            a = np.abs(g)
        elif method == "input_x_gradient":
            a = np.abs(x.detach().double().numpy() * g)
        else:
            a = np.abs((x.detach().double().numpy() - baseline[i:i + batch_size]) * g)
        a = a.mean(1)
        a[:, SETTING_IDX] = 0.0
        preds.append(y.detach().double().numpy())
        attrs.append(a)
    return np.concatenate(preds), np.concatenate(attrs)


def sensor_shares(channel_attr: np.ndarray) -> np.ndarray:
    """(N, 17) channel attribution -> (N, 14) fault-sensor shares in percent, rows sum to 100."""
    a = np.atleast_2d(np.asarray(channel_attr, dtype=np.float64))[:, FAULT_IDX]
    a = np.clip(a, 0.0, None)
    tot = a.sum(1, keepdims=True)
    uniform = np.full_like(a, 1.0 / a.shape[1])
    return 100.0 * np.where(tot > 0, a / np.where(tot > 0, tot, 1.0), uniform)


def subsystem_shares(sensor_share: np.ndarray) -> np.ndarray:
    """(N, 14) sensor shares -> (N, 4) subsystem shares (order ``SUBSYSTEMS``)."""
    return np.atleast_2d(sensor_share) @ _SUB_MATRIX


def subsystem_drift(window_drift: np.ndarray) -> np.ndarray:
    """(N, 14) signed drift -> (N, 4) mean |z| per subsystem (sensor-count invariant)."""
    a = np.abs(np.atleast_2d(window_drift))
    return (a @ _SUB_MATRIX) / _SUB_MATRIX.sum(0)


def round_to_total(values: Sequence[float], decimals: int = 1, total: float = 100.0) -> List[float]:
    """Largest-remainder rounding so rounded shares still sum exactly to ``total``."""
    q = 10 ** decimals
    v = np.asarray(values, dtype=np.float64) * q
    if not (np.isfinite(v).all() and np.isfinite(total)):
        raise ValueError("round_to_total requires finite values and total")
    if (v < 0).any() or total < 0:
        raise ValueError("round_to_total requires non-negative values and total")
    fl = np.floor(v)
    rest = int(round(total * q - fl.sum()))
    if rest < 0 or rest > len(v):
        raise ValueError(f"values cannot be rounded to total {total}: remainder {rest} outside [0, {len(v)}]")
    order = np.argsort(-(v - fl))
    fl[order[:rest]] += 1
    return [round(float(x) / q, decimals) for x in fl]


# --------------------------------------------------------------------------- summary


def severity_for(pred_rul: float) -> str:
    if pred_rul <= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    if pred_rul <= SEVERITY_THRESHOLDS["warning"]:
        return "warning"
    return "healthy"


def _sensor_entry(c: str) -> dict:
    info = SENSOR_INFO[c]
    return {"sensor": c, "symbol": info["symbol"], "name": info["name"], "subsystem": info["subsystem"]}


def maintenance_summary(pred_rul: float, sensor_share: np.ndarray, window_drift: np.ndarray,
                        engine_id: Optional[str] = None, cycle: Optional[int] = None, top_k: int = 3) -> dict:
    """Structured, human-readable summary for one window (no raw sensor values inside)."""
    share = np.asarray(sensor_share, dtype=np.float64).ravel()
    drift = np.asarray(window_drift, dtype=np.float64).ravel()
    sub = subsystem_shares(share)[0]
    sub_r = round_to_total(sub, 1)
    primary = SUBSYSTEMS[int(np.argmax(sub))]
    dsub = subsystem_drift(drift)[0]
    top_c = [dict(_sensor_entry(FAULT_SENSORS[i]), share_pct=round(float(share[i]), 1))
             for i in np.argsort(-share, kind="stable")[:top_k]]
    top_d = [dict(_sensor_entry(FAULT_SENSORS[i]), drift_z=round(float(drift[i]), 2),
                  direction="up" if drift[i] >= 0 else "down")
             for i in np.argsort(-np.abs(drift), kind="stable")[:top_k]]
    sev = severity_for(float(pred_rul))
    drivers = ", ".join(f"{d['sensor']} {d['symbol']} ({d['share_pct']:.1f}%)" for d in top_c)
    drifts = ", ".join(f"{d['sensor']} {d['symbol']} ({d['drift_z']:+.1f} sigma)" for d in top_d)
    lead = {"critical": "Schedule maintenance now", "warning": "Plan inspection", "healthy": "No action"}[sev]
    lead_it = {"critical": "Pianificare subito la manutenzione", "warning": "Pianificare un'ispezione",
               "healthy": "Nessuna azione"}[sev]
    return {
        "engine_id": engine_id,
        "cycle": cycle,
        "predicted_rul": round(float(pred_rul), 2),
        "severity": sev,
        "primary_subsystem": primary,
        "primary_subsystem_label": SUBSYSTEM_LABELS[primary],
        "subsystem_shares_pct": dict(zip(SUBSYSTEMS, sub_r)),
        "primary_drift_subsystem": SUBSYSTEMS[int(np.argmax(dsub))],
        "subsystem_drift_mean_abs_z": {s: round(float(v), 2) for s, v in zip(SUBSYSTEMS, dsub)},
        "drift_index": round(float(np.abs(drift).mean()), 2),
        "top_contributors": top_c,
        "top_drifting": top_d,
        "text": {
            "en": (f"Predicted RUL {pred_rul:.1f} cycles ({sev.upper()}). {lead}. "
                   f"Primary contributor: {primary} ({sub_r[SUBSYSTEMS.index(primary)]:.1f}% of attribution). "
                   f"Top drivers: {drivers}. Largest drift vs healthy baseline: {drifts}."),
            "it": (f"RUL stimata {pred_rul:.1f} cicli ({sev.upper()}). {lead_it}. "
                   f"Sottosistema principale: {primary} ({sub_r[SUBSYSTEMS.index(primary)]:.1f}% dell'attribuzione). "
                   f"Sensori determinanti: {drivers}. Deriva maggiore vs baseline sana: {drifts}."),
        },
        "privacy": dict(PRIVACY_METADATA),
        "evaluation_boundary": EVALUATION_BOUNDARY,
    }


# --------------------------------------------------------------------------- engine-level driver


class EdgeDiagnostics:
    """Bundle model + scaler + healthy baseline for on-node per-window diagnosis."""

    def __init__(self, model: torch.nn.Module, scales: dict, baseline: HealthyBaseline,
                 method: str = "baseline_x_gradient"):
        self.model, self.scales, self.baseline, self.method = model, scales, baseline, method

    def diagnose_engine(self, frame: pd.DataFrame, engine_id: Optional[str] = None) -> dict:
        """Diagnose every W=30 window of one engine's raw (unscaled) history.

        Returns arrays aligned with window end cycles: ``pred_rul`` (N,),
        ``channel_attr`` (N,17), ``sensor_share`` (N,14), ``subsystem_share``
        (N,4), ``window_drift`` (N,14) and ``cycle_drift`` (T,14).
        """
        frame = frame.sort_values("cycle")
        x = scale_frame(frame[USEFUL_SENSORS].values, self.scales).astype(np.float32)
        ref = self.baseline.healthy_reference(frame, self.scales)
        drift = self.baseline.drift(frame)
        wx, wref, wd = sliding_windows(x), sliding_windows(ref), sliding_windows(drift)
        cycles = frame["cycle"].values.astype(int)
        end_cycles = cycles[SEQ_LEN - 1:] if len(cycles) >= SEQ_LEN else cycles[-1:]
        pred, attr = attribute_windows(self.model, wx, wref, self.method)
        sh = sensor_shares(attr)
        return {
            "engine_id": engine_id,
            "cycles": end_cycles,
            "pred_rul": pred,
            "channel_attr": attr,
            "sensor_share": sh,
            "subsystem_share": subsystem_shares(sh),
            "window_drift": wd.mean(1),
            "cycle_drift": drift,
        }

    def summarize(self, diag: dict, index: int = -1) -> dict:
        return maintenance_summary(diag["pred_rul"][index], diag["sensor_share"][index],
                                   diag["window_drift"][index], diag["engine_id"],
                                   int(diag["cycles"][index]))
