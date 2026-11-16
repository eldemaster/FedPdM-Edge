#!/usr/bin/env python3
"""5-seed validation of the local edge diagnostics module (diagnostics.py).

C-MAPSS has no per-unit component labels, so this script does NOT report a
supervised component accuracy. It reports:

a) FD001/FD002 (single fault mode, HPC degradation): how consistently the
   HPC / engine-core sensors dominate attribution and drift when true RUL < 30,
   against the sensor-count chance rate, plus the subsystem-share trend
   across RUL buckets.
b) FD003/FD004 (two fault modes, HPC + Fan, unlabeled): k=2 clustering of
   end-of-life drift and attribution profiles over train units and test units
   whose official RUL <= 30; silhouette, cross-seed ARI, cross-view ARI and
   the physical signature (Fan vs HPC drift/attribution) of each cluster.
c) Edge overhead: single-thread CPU latency per window for pure inference vs
   inference + attribution + drift + summary, autograd memory, and a socket
   audit proving the diagnostics path performs no network I/O.

Usage:
  CMAPSS_ZIP=/path/CMAPSSData.zip python experiments/validate_diagnostics.py \
      --out results/diagnostics_validation.json
"""
import argparse
import itertools
import json
import platform
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import diagnostics as D  # noqa: E402
from benchmark_asymmetric_loss import compute_scales_from_train  # noqa: E402
from common import (  # noqa: E402
    MAX_RUL, SEQ_LEN, USEFUL_SENSORS, CmapssRegressor, apply_scale, build_test_set, get_loss_fn,
    load_cmapss_raw, load_rul, make_loader, make_xy, predict, rmse, set_seed, split_units, train_epochs,
)

SEEDS = [7, 42, 100, 777, 2024]
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
CORE = ("HPC", "Turbine")
EOL_WINDOWS = 10
LOW_RUL = 30
REFERENCE = {"FD003": "FD001", "FD004": "FD002"}  # same operating conditions, HPC-only fault mode


# ----------------------------------------------------------------------------- network audit
class SocketAudit:
    """Count every outbound socket operation while active (connect/send*)."""

    NAMES = ("connect", "connect_ex", "send", "sendall", "sendto")

    def __init__(self):
        self.calls, self.bytes, self._orig = 0, 0, {}

    def __enter__(self):
        audit = self
        for name in self.NAMES:
            orig = getattr(socket.socket, name)
            self._orig[name] = orig

            def wrap(sock, *a, _orig=orig, **k):
                audit.calls += 1
                if a and isinstance(a[0], (bytes, bytearray, memoryview)):
                    audit.bytes += len(a[0])
                return _orig(sock, *a, **k)
            setattr(socket.socket, name, wrap)
        self._cc = socket.create_connection

        def cc(*a, **k):
            audit.calls += 1
            return self._cc(*a, **k)
        socket.create_connection = cc
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(socket.socket, name, orig)
        socket.create_connection = self._cc


# ----------------------------------------------------------------------------- helpers
def stats(vals):
    v = np.asarray(vals, dtype=np.float64)
    return {"mean": round(float(v.mean()), 4), "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
            "per_seed": [round(float(x), 4) for x in v]}


def engine_true_rul(cycles, last_cycle, final_rul):
    return np.minimum(MAX_RUL, final_rul + (last_cycle - np.asarray(cycles)))


def train_model(df_scaled, units, seed, epochs, loss):
    X, y = make_xy(df_scaled, units)
    set_seed(seed)
    return train_epochs(CmapssRegressor(), make_loader(X, y, 64, seed=seed), epochs, loss_fn=get_loss_fn(loss))


def diagnose_population(diag, frames):
    """frames: list of (key, frame, final_rul). Returns dict key -> diagnosis (+ true RUL per window)."""
    out = {}
    for key, frame, final_rul in frames:
        d = diag.diagnose_engine(frame, key)
        d["true_rul"] = engine_true_rul(d["cycles"], int(frame["cycle"].max()), final_rul)
        out[key] = d
    return out


# ----------------------------------------------------------------------------- (a) single fault
def single_fault_metrics(diags, hpc_only_units=None):
    keys = list(diags) if hpc_only_units is None else hpc_only_units
    sub_share, sens_share, drift, true = [], [], [], []
    for k in keys:
        d = diags[k]
        sub_share.append(d["subsystem_share"]); sens_share.append(d["sensor_share"])
        drift.append(d["window_drift"]); true.append(d["true_rul"])
    sub_share, sens_share = np.concatenate(sub_share), np.concatenate(sens_share)
    drift, true = np.concatenate(drift), np.concatenate(true)
    low = true < LOW_RUL
    prim = np.array(D.SUBSYSTEMS)[sub_share[low].argmax(1)]
    dsub = D.subsystem_drift(drift[low])
    dprim = np.array(D.SUBSYSTEMS)[dsub.argmax(1)]
    sens = np.array(D.FAULT_SENSORS)
    core_sensors = {c for s in CORE for c in D.SUBSYSTEM_SENSORS[s]}
    top3_attr = sens[np.argsort(-sens_share[low], 1, kind="stable")[:, :3]]
    top3_drift = sens[np.argsort(-np.abs(drift[low]), 1, kind="stable")[:, :3]]
    buckets = {"rul_ge_100": true >= 100, "rul_30_100": (true >= 30) & (true < 100), "rul_lt_30": low}
    return {
        "n_windows_rul_lt_30": int(low.sum()),
        "attr_primary_distribution": {s: float(np.mean(prim == s)) for s in D.SUBSYSTEMS},
        "attr_primary_hpc_rate": float(np.mean(prim == "HPC")),
        "attr_primary_core_rate": float(np.isin(prim, CORE).mean()),
        "attr_top3_core_sensor_fraction": float(np.isin(top3_attr, list(core_sensors)).mean()),
        "attr_mean_subsystem_share_by_rul": {
            b: {s: float(sub_share[m, i].mean()) for i, s in enumerate(D.SUBSYSTEMS)} for b, m in buckets.items() if m.any()},
        "drift_primary_distribution": {s: float(np.mean(dprim == s)) for s in D.SUBSYSTEMS},
        "drift_top3_core_sensor_fraction": float(np.isin(top3_drift, list(core_sensors)).mean()),
        "drift_mean_abs_z_by_subsystem_rul_lt_30": {s: float(dsub[:, i].mean()) for i, s in enumerate(D.SUBSYSTEMS)},
        "drift_top3_sensor_frequency": {c: float(np.mean((top3_drift == c).any(1))) for c in D.FAULT_SENSORS},
        "attr_top3_sensor_frequency": {c: float(np.mean((top3_attr == c).any(1))) for c in D.FAULT_SENSORS},
    }


# ----------------------------------------------------------------------------- (b) two fault modes
def eol_profiles(diags, keys):
    drift, attr, asub = [], [], []
    for k in keys:
        d = diags[k]
        drift.append(d["window_drift"][-EOL_WINDOWS:].mean(0))
        attr.append(d["sensor_share"][-EOL_WINDOWS:].mean(0))
        asub.append(d["subsystem_share"][-EOL_WINDOWS:].mean(0))
    return np.array(drift), np.array(attr), np.array(asub)


def cluster(features, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler

    z = StandardScaler().fit_transform(features)
    km = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(z)
    gmm = GaussianMixture(n_components=2, covariance_type="diag", random_state=seed, n_init=3).fit(z)
    from sklearn.metrics import adjusted_rand_score as ari
    return km.labels_, float(silhouette_score(z, km.labels_)), float(ari(km.labels_, gmm.predict(z)))


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def reference_signature(ref_subset):
    """Label-free HPC reference: mean end-of-life drift of a single-fault (HPC-only) subset."""
    df = load_cmapss_raw(f"train_{ref_subset}.txt")
    bl = D.fit_healthy_baseline(df, D.N_REGIMES[ref_subset])
    prof = []
    for _, g in df.groupby("unit"):
        z = D.sliding_windows(bl.drift(g.sort_values("cycle"))).mean(1)
        prof.append(z[-EOL_WINDOWS:].mean(0))
    return np.mean(prof, 0)


def orient(labels, drift, ref):
    """Name clusters physically: 'hpc_like' = mean signed drift closest (cosine) to the HPC reference."""
    sims = [cosine(drift[labels == c].mean(0), ref) if (labels == c).any() else -np.inf for c in (0, 1)]
    hpc = int(np.argmax(sims))
    return np.where(labels == hpc, 0, 1)  # 0 = hpc_like, 1 = fan_candidate


def signature(mask, drift, attr_sub, ref):
    dsub = D.subsystem_drift(drift[mask])
    mean_drift = drift[mask].mean(0)
    return {
        "n_units": int(mask.sum()),
        "cosine_to_hpc_reference": round(cosine(mean_drift, ref), 4),
        "drift_mean_abs_z": {s: round(float(dsub[:, i].mean()), 3) for i, s in enumerate(D.SUBSYSTEMS)},
        "drift_signed_z": {c: round(float(v), 3) for c, v in zip(D.FAULT_SENSORS, mean_drift)},
        "attr_subsystem_share_pct": {s: round(float(attr_sub[mask, i].mean()), 2) for i, s in enumerate(D.SUBSYSTEMS)},
    }


def two_fault_metrics(diags, keys, seed, ref):
    from sklearn.metrics import adjusted_rand_score as ari

    drift, attr, asub = eol_profiles(diags, keys)
    res, labels = {}, {}
    for name, feats in (("drift", drift), ("attribution", attr), ("combined", np.hstack([drift, attr]))):
        lab, sil, gmm_ari = cluster(feats, seed)
        lab = orient(lab, drift, ref)
        labels[name] = lab
        res[name] = {
            "silhouette": sil, "kmeans_vs_gmm_ari": gmm_ari,
            "hpc_like": signature(lab == 0, drift, asub, ref),
            "fan_candidate": signature(lab == 1, drift, asub, ref),
        }
    res["cross_view_ari_drift_vs_attribution"] = float(ari(labels["drift"], labels["attribution"]))
    return res, labels


# ----------------------------------------------------------------------------- (c) overhead
def saved_tensor_bytes(model, x):
    total = [0]

    def pack(t):
        total[0] += t.numel() * t.element_size()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        xr = torch.tensor(x, requires_grad=True)
        torch.autograd.grad(model(xr).sum(), xr)
    return total[0]


def latency(model, diag, frame, n=300, reps=3):
    torch.set_num_threads(1)
    diag.diagnose_engine(frame)  # warm-up
    x = D.sliding_windows(D.scale_frame(frame[USEFUL_SENSORS].values, diag.scales).astype(np.float32))
    ref = D.sliding_windows(diag.baseline.healthy_reference(frame, diag.scales))
    drift = D.sliding_windows(diag.baseline.drift(frame))
    idx = np.arange(min(n, len(x)))
    model.eval()

    def infer(i):
        with torch.no_grad():
            model(torch.from_numpy(x[i:i + 1]))

    def full(i):
        u = frame.iloc[: i + SEQ_LEN]  # drift recomputed from the raw history up to this cycle
        dr = diag.baseline.drift(u)[-SEQ_LEN:].mean(0)
        p, a = D.attribute_windows(model, x[i:i + 1], ref[i:i + 1])
        D.maintenance_summary(p[0], D.sensor_shares(a)[0], dr)

    def attr_only(i):
        p, a = D.attribute_windows(model, x[i:i + 1], ref[i:i + 1])
        D.maintenance_summary(p[0], D.sensor_shares(a)[0], drift[i].mean(0))

    out = {}
    for name, fn in (("inference_only", infer), ("inference_attribution_summary", attr_only),
                     ("inference_attribution_drift_summary", full)):
        for i in idx[:20]:
            fn(i)
        per = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for i in idx:
                fn(i)
            per.append((time.perf_counter() - t0) / len(idx) * 1e3)
        out[name + "_ms_per_window"] = round(float(np.median(per)), 4)
    out["overhead_ms_per_window"] = round(out["inference_attribution_drift_summary_ms_per_window"]
                                          - out["inference_only_ms_per_window"], 4)
    out["overhead_ratio"] = round(out["inference_attribution_drift_summary_ms_per_window"]
                                  / out["inference_only_ms_per_window"], 2)
    out["autograd_saved_tensor_bytes_per_window"] = saved_tensor_bytes(model, x[:1])
    out["input_window_bytes"] = int(x[:1].nbytes)
    out["windows_timed"] = int(len(idx))
    out["threads"] = torch.get_num_threads()
    return out


# ----------------------------------------------------------------------------- main
def run_subset(subset, seeds, epochs, loss, lat_seed):
    df_tr = load_cmapss_raw(f"train_{subset}.txt")
    df_te = load_cmapss_raw(f"test_{subset}.txt")
    ruls = load_rul(f"RUL_{subset}.txt")
    scales = compute_scales_from_train(df_tr)
    scaled = apply_scale(df_tr, scales)
    units = sorted(int(u) for u in df_tr.unit.unique())
    with SocketAudit() as audit:
        baseline = D.fit_healthy_baseline(df_tr, D.N_REGIMES[subset])
    test_frames = [(f"test-{int(u)}", g, int(ruls[int(u) - 1])) for u, g in df_te.groupby("unit")]
    train_frames = {u: (f"train-{u}", g, 0) for u, g in df_tr.groupby("unit")}

    per_seed, cluster_labels, lat = [], {}, None
    ref = None
    if subset in ("FD003", "FD004"):
        ref = reference_signature(REFERENCE[subset])
    for seed in seeds:
        t0 = time.time()
        tr_units, ho_units = split_units(units, 0.2, seed=seed)
        model = train_model(scaled, tr_units, seed, epochs, loss)
        Xt, yt = build_test_set(scales, subset)
        test_rmse = rmse(yt, np.clip(predict(model, Xt), 0, MAX_RUL))
        diag = D.EdgeDiagnostics(model, scales, baseline)
        rec = {"seed": seed, "test_rmse": test_rmse, "n_train_units": len(tr_units), "n_heldout_units": len(ho_units)}
        with SocketAudit() as audit_seed:
            if subset in ("FD001", "FD002"):
                frames = [train_frames[u] for u in ho_units] + test_frames
                diags = diagnose_population(diag, frames)
                rec["baseline_x_gradient"] = single_fault_metrics(diags)
                grad_diag = D.EdgeDiagnostics(model, scales, baseline, method="gradient")
                g = single_fault_metrics(diagnose_population(grad_diag, frames))
                rec["gradient"] = {k: g[k] for k in ("attr_primary_distribution", "attr_primary_hpc_rate",
                                                      "attr_primary_core_rate", "attr_top3_core_sensor_fraction")}
            else:
                frames = list(train_frames.values()) + [f for f in test_frames if f[2] <= LOW_RUL]
                diags = diagnose_population(diag, frames)
                keys = [f[0] for f in frames]
                rec["clustering"], labels = two_fault_metrics(diags, keys, seed, ref)
                rec["n_units_clustered"] = {"train": len(train_frames), "test_rul_le_30": len(keys) - len(train_frames)}
                cluster_labels[seed] = (keys, labels)
        rec["network_socket_calls"] = audit_seed.calls
        rec["network_bytes_sent"] = audit_seed.bytes
        if seed == lat_seed:
            u = max(ho_units, key=lambda k: len(train_frames[k][1]))
            threads = torch.get_num_threads()
            with SocketAudit() as audit_lat:
                lat = latency(model, diag, train_frames[u][1].sort_values("cycle"))
            torch.set_num_threads(threads)
            lat["network_socket_calls"] = audit_lat.calls
        rec["elapsed_s"] = round(time.time() - t0, 1)
        per_seed.append(rec)
        print(f"  {subset} seed {seed}: test RMSE {test_rmse:.2f} ({rec['elapsed_s']} s)", flush=True)

    out = {"n_regimes": baseline.n_regimes, "baseline_fit_network_socket_calls": audit.calls,
           "per_seed": per_seed, "latency": lat}
    if subset in ("FD001", "FD002"):
        agg = {}
        for meth in ("baseline_x_gradient", "gradient"):
            agg[meth] = {k: stats([r[meth][k] for r in per_seed])
                         for k in ("attr_primary_hpc_rate", "attr_primary_core_rate", "attr_top3_core_sensor_fraction")}
        agg["baseline_x_gradient"]["drift_top3_core_sensor_fraction"] = stats(
            [r["baseline_x_gradient"]["drift_top3_core_sensor_fraction"] for r in per_seed])
        agg["baseline_x_gradient"]["attr_primary_distribution"] = {
            s: stats([r["baseline_x_gradient"]["attr_primary_distribution"][s] for r in per_seed]) for s in D.SUBSYSTEMS}
        agg["baseline_x_gradient"]["drift_primary_distribution"] = per_seed[0]["baseline_x_gradient"]["drift_primary_distribution"]
        n_core = sum(len(D.SUBSYSTEM_SENSORS[s]) for s in CORE)
        agg["chance_rates_by_sensor_count"] = {
            "primary_hpc": len(D.SUBSYSTEM_SENSORS["HPC"]) / len(D.FAULT_SENSORS),
            "primary_core": n_core / len(D.FAULT_SENSORS),
            "top3_core_sensor_fraction": n_core / len(D.FAULT_SENSORS)}
        agg["note"] = ("drift metrics are model-independent (identical across seeds); chance rates assume "
                       "attribution mass proportional to sensor count per subsystem")
        out["summary"] = agg
    else:
        from sklearn.metrics import adjusted_rand_score as ari
        agg = {}
        for name in ("drift", "attribution", "combined"):
            pair = [ari(cluster_labels[a][1][name], cluster_labels[b][1][name])
                    for a, b in itertools.combinations(seeds, 2)]
            n_tot = [sum(r["n_units_clustered"].values()) for r in per_seed]
            entry = {
                "silhouette": stats([r["clustering"][name]["silhouette"] for r in per_seed]),
                "kmeans_vs_gmm_ari": stats([r["clustering"][name]["kmeans_vs_gmm_ari"] for r in per_seed]),
                "cross_seed_ari": stats(pair),
                "fan_candidate_fraction": stats([r["clustering"][name]["fan_candidate"]["n_units"] / n
                                                 for r, n in zip(per_seed, n_tot)]),
            }
            for cl in ("hpc_like", "fan_candidate"):
                rows = [r["clustering"][name][cl] for r in per_seed]
                entry[cl] = {
                    "cosine_to_hpc_reference": stats([x["cosine_to_hpc_reference"] for x in rows]),
                    "drift_mean_abs_z": {s: stats([x["drift_mean_abs_z"][s] for x in rows]) for s in D.SUBSYSTEMS},
                    "drift_signed_z": {c: stats([x["drift_signed_z"][c] for x in rows]) for c in D.FAULT_SENSORS},
                    "attr_subsystem_share_pct": {s: stats([x["attr_subsystem_share_pct"][s] for x in rows])
                                                 for s in D.SUBSYSTEMS},
                }
            agg[name] = entry
        agg["cross_view_ari_drift_vs_attribution"] = stats(
            [r["clustering"]["cross_view_ari_drift_vs_attribution"] for r in per_seed])
        agg["hpc_reference"] = {"subset": REFERENCE[subset],
                                "eol_signed_drift_z": {c: round(float(v), 3) for c, v in zip(D.FAULT_SENSORS, ref)}}
        out["summary"] = agg
    out["test_rmse"] = stats([r["test_rmse"] for r in per_seed])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--loss", default="asymmetric", choices=["mse", "asymmetric"])
    ap.add_argument("--threads", type=int, default=4, help="torch threads for training (latency is always 1)")
    ap.add_argument("--out", default=str(ROOT / "results" / "diagnostics_validation.json"))
    args = ap.parse_args()

    t0 = time.time()
    result = {
        "description": "Edge diagnostics validation (attribution + healthy-baseline drift); no supervised component labels exist in C-MAPSS",
        "evaluation_boundary": D.EVALUATION_BOUNDARY,
        "config": {
            "seeds": args.seeds, "subsets": args.subsets, "epochs": args.epochs, "loss": args.loss,
            "model": "CmapssRegressor (common.py)", "window": SEQ_LEN, "rul_cap": MAX_RUL,
            "attribution_formula": "A_c = mean_t |(x[t,c] - b[t,c]) * dRUL/dx[t,c]|, b = regime-aware healthy baseline; settings forced to 0; shares = 100*A_c/sum_c A_c",
            "drift_formula": "z[t,c] = (u[t,c] - mean_{t<30} u[t,c]) / sigma_healthy_c, u = per-regime standardized sensor",
            "train_units": "80% of training units (split_units, val_frac=0.2, per seed)",
            "single_fault_population": "held-out training units (run to failure) + all official test units; windows with true RUL < 30",
            "two_fault_population": f"all training units + test units with official RUL <= {LOW_RUL}; profile = mean of last {EOL_WINDOWS} windows",
            "clustering": "StandardScaler -> KMeans(k=2, n_init=10, random_state=seed); GMM(diag) agreement; clusters oriented by cosine similarity of mean signed EOL drift to the label-free HPC-only reference (FD001 for FD003, FD002 for FD004); the other cluster is the fan-mode candidate",
            "subsystem_map": {s: D.SUBSYSTEM_SENSORS[s] for s in D.SUBSYSTEMS},
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "machine": platform.machine(), "system": platform.system(),
                        "processor": platform.processor() or platform.machine()},
        "subsets": {},
    }
    torch.set_num_threads(args.threads)
    for subset in args.subsets:
        print(f"[{subset}]", flush=True)
        torch.set_num_threads(args.threads)
        result["subsets"][subset] = run_subset(subset, args.seeds, args.epochs, args.loss, lat_seed=args.seeds[0])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["process_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    result["network_audit"] = {
        "total_socket_calls_during_diagnostics": sum(
            r["network_socket_calls"] for s in result["subsets"].values() for r in s["per_seed"])
        + sum(s["baseline_fit_network_socket_calls"] for s in result["subsets"].values())
        + sum((s["latency"] or {}).get("network_socket_calls", 0) for s in result["subsets"].values()),
        "raw_sensor_bytes_sent": sum(r["network_bytes_sent"] for s in result["subsets"].values() for r in s["per_seed"]),
        "method": "socket.socket.connect/connect_ex/send/sendall/sendto and socket.create_connection instrumented while fitting baselines, diagnosing every engine and timing latency",
    }
    result["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out} in {result['elapsed_s']} s")


if __name__ == "__main__":
    main()
