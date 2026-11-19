"""Generate demo_data.json for the FedPdM-Edge presentation dashboard.

Trains lightweight reference models on FD001 and FD002 (seed 42) reusing common.py
helpers, computes cycle-by-cycle RUL trajectories on representative held-out test
engines, and packages measured physical cluster & reproducible benchmark metrics.

Each traced engine also carries local edge diagnostics (``diagnostics.py``):
per-cycle subsystem responsibility shares (Fan / LPC / HPC / Turbine, summing to
100%), top-3 contributing and top-3 drifting sensors, and a maintenance summary,
computed from the FedAvg (Asymmetric) model on the engine's own telemetry.
"""
import base64
import csv
import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import diagnostics as diag_mod  # noqa: E402
from benchmark_asymmetric_loss import compute_scales_from_train  # noqa: E402
from common import (  # noqa: E402
    MAX_RUL,
    SEQ_LEN,
    USEFUL_SENSORS,
    CmapssRegressor,
    apply_scale,
    get_loss_fn,
    load_cmapss_raw,
    load_partition,
    load_rul,
    load_scales,
    make_loader,
    make_xy,
    predict,
    set_seed,
    split_units,
    train_epochs,
)

DEMO_SENSORS = ["s2", "s3", "s4", "s11"]
FALLBACK_TAG = '<script id="demoDataGzip" type="application/octet-stream">'


def update_file_replay_fallback(data_path: Path) -> None:
    """Embed a deterministic compressed copy for browsers that block file:// fetches."""
    index_path = data_path.parent / "index.html"
    if not index_path.is_file():
        return
    html = index_path.read_text(encoding="utf-8")
    start = html.index(FALLBACK_TAG)
    content_start = start + len(FALLBACK_TAG)
    end = html.index("</script>", content_start)
    encoded = base64.b64encode(gzip.compress(data_path.read_bytes(), compresslevel=9, mtime=0)).decode("ascii")
    wrapped = "\n" + "\n".join(encoded[i:i + 120] for i in range(0, len(encoded), 120)) + "\n"
    index_path.write_text(html[:content_start] + wrapped + html[end:], encoding="utf-8")


def train_subset_models(subset: str, seed: int = 42, rounds: int = 5, local_epochs: int = 2):
    """Train Isolated Local (MSE), FedAvg (MSE), and FedAvg (Asymmetric) on a subset."""
    if subset == "FD002":
        scales = load_scales(str(ROOT / "global_scale.json"))
        part = load_partition(str(ROOT / "partition.json"))
        df_raw = load_cmapss_raw("train_FD002.txt")
    else:
        df_raw = load_cmapss_raw(f"train_{subset}.txt")
        scales = compute_scales_from_train(df_raw)
        units = sorted(int(u) for u in df_raw["unit"].unique())
        part = {"1": [u for u in units if u % 2 == 1], "2": [u for u in units if u % 2 == 0]}

    df = apply_scale(df_raw, scales)
    client_data = {}
    for cid, units in sorted(part.items()):
        tr, _ = split_units(units, 0.1, seed=seed)
        client_data[cid] = make_xy(df, tr)

    # Isolated local model trained only on Plant 1 data
    cid0 = sorted(client_data)[0]
    set_seed(seed + int(cid0))
    local_model = train_epochs(
        CmapssRegressor(),
        make_loader(*client_data[cid0], 64, seed=seed + int(cid0)),
        rounds * local_epochs,
        loss_fn=get_loss_fn("mse"),
    )

    fed_models = {}
    for loss_name in ("mse", "asymmetric"):
        loss_fn = get_loss_fn(loss_name)
        set_seed(seed)
        gmodel = CmapssRegressor()
        clients = []
        for cid, (X, y) in sorted(client_data.items()):
            set_seed(seed + int(cid))
            m = CmapssRegressor()
            clients.append(
                dict(
                    model=m,
                    loader=make_loader(X, y, 64, seed=seed + int(cid)),
                    n=len(X),
                    opt=torch.optim.Adam(m.parameters(), lr=1e-3),
                )
            )
        for _ in range(rounds):
            gstate = {k: v.clone() for k, v in gmodel.state_dict().items()}
            states, ns = [], []
            for c in clients:
                c["model"].load_state_dict(gstate)
                train_epochs(c["model"], c["loader"], local_epochs, optimizer=c["opt"], loss_fn=loss_fn)
                states.append({k: v.clone() for k, v in c["model"].state_dict().items()})
                ns.append(c["n"])
            tot = sum(ns)
            gmodel.load_state_dict({k: sum(s[k].float() * n for s, n in zip(states, ns)) / tot for k in gstate})
        fed_models[loss_name] = gmodel

    baseline = diag_mod.fit_healthy_baseline(df_raw, diag_mod.N_REGIMES[subset])
    return scales, local_model, fed_models["mse"], fed_models["asymmetric"], baseline


def engine_diagnostics(model, scales: dict, baseline, raw_group, engine_id: str) -> dict:
    """Per-cycle subsystem shares, top-3 contributing / drifting sensors, final summary."""
    edge = diag_mod.EdgeDiagnostics(model, scales, baseline)
    d = edge.diagnose_engine(raw_group, engine_id)
    sens = diag_mod.FAULT_SENSORS
    top_c, top_d, shares = [], [], []
    for sh, sub, dr in zip(d["sensor_share"], d["subsystem_share"], d["window_drift"]):
        shares.append(diag_mod.round_to_total(sub, 1))
        top_c.append([[sens[i], round(float(sh[i]), 1)] for i in np.argsort(-sh, kind="stable")[:3]])
        top_d.append([[sens[i], round(float(dr[i]), 2)] for i in np.argsort(-np.abs(dr), kind="stable")[:3]])
    return {
        "method": edge.method,
        "subsystem_share": shares,
        "top_contributors": top_c,
        "top_drifting": top_d,
        "drift_index": [round(float(np.abs(dr).mean()), 2) for dr in d["window_drift"]],
        "final_summary": edge.summarize(d),
    }


def diagnostics_meta() -> dict:
    return {
        "subsystems": list(diag_mod.SUBSYSTEMS),
        "subsystem_labels": diag_mod.SUBSYSTEM_LABELS,
        "subsystem_sensors": diag_mod.SUBSYSTEM_SENSORS,
        "sensor_info": diag_mod.SENSOR_INFO,
        "excluded_channels": list(diag_mod.SETTINGS),
        "model": "FedAvg (Asymmetric) global model, evaluated locally on the edge node",
        "attribution_formula": "A_c = mean_t |(x[t,c] - b[t,c]) * dRUL/dx[t,c]|; b = regime-aware healthy baseline; settings forced to 0; share_c = 100*A_c/sum A",
        "drift_formula": "z = (regime-standardized sensor - engine healthy mean over first 30 cycles) / pooled healthy noise; window mean",
        "privacy": diag_mod.PRIVACY_METADATA,
        "evaluation_boundary": diag_mod.EVALUATION_BOUNDARY,
    }


def pick_representative_engines(subset: str, scales: dict, local_model, fed_mse, fed_asym, baseline=None):
    """Select 4 diverse held-out test engines and trace cycle-by-cycle RUL predictions."""
    df_test_raw = load_cmapss_raw(f"test_{subset}.txt")
    df_test = apply_scale(df_test_raw, scales)
    ruls = load_rul(f"RUL_{subset}.txt")

    candidates = []
    for u, grp in df_test.groupby("unit"):
        u = int(u)
        if len(grp) < 85:
            continue
        final_rul = int(min(MAX_RUL, ruls[u - 1]))
        candidates.append((u, final_rul, grp))

    buckets = [
        ("critical", lambda r: r <= 20),
        ("warning", lambda r: 22 <= r <= 48),
        ("midlife", lambda r: 52 <= r <= 85),
        ("healthy", lambda r: r >= 95),
    ]
    selected = []
    used = set()
    for label, pred_fn in buckets:
        matches = [c for c in candidates if pred_fn(c[1]) and c[0] not in used]
        if matches:
            matches.sort(key=lambda c: len(c[2]), reverse=True)
            selected.append((label, *matches[0]))
            used.add(matches[0][0])

    traces = []
    for label, unit_id, final_rul, grp in selected:
        grp = grp.sort_values("cycle")
        vals = grp[USEFUL_SENSORS].values.astype(np.float32)
        cycles = grp["cycle"].values.astype(int)
        t_end = int(cycles[-1])

        windows = np.array([vals[i : i + SEQ_LEN] for i in range(len(vals) - SEQ_LEN + 1)], dtype=np.float32)
        eval_cycles = [int(cycles[i + SEQ_LEN - 1]) for i in range(len(vals) - SEQ_LEN + 1)]
        true_rul = [float(min(MAX_RUL, final_rul + (t_end - c))) for c in eval_cycles]

        p_local = [float(np.clip(v, 0.0, MAX_RUL)) for v in predict(local_model, windows)]
        p_fed_mse = [float(np.clip(v, 0.0, MAX_RUL)) for v in predict(fed_mse, windows)]
        p_fed_asym = [float(np.clip(v, 0.0, MAX_RUL)) for v in predict(fed_asym, windows)]

        sensor_series = {
            s: [round(float(v), 4) for v in grp[s].values[SEQ_LEN - 1 :]] for s in DEMO_SENSORS
        }
        traces.append(
            {
                "id": f"{subset}-U{unit_id}",
                "subset": subset,
                "unit": unit_id,
                "stage": label,
                "final_true_rul": final_rul,
                "total_observed_cycles": t_end,
                "cycles": eval_cycles,
                "true_rul": [round(v, 2) for v in true_rul],
                "pred_local_mse": [round(v, 2) for v in p_local],
                "pred_fedavg_mse": [round(v, 2) for v in p_fed_mse],
                "pred_fedavg_asym": [round(v, 2) for v in p_fed_asym],
                "sensors": sensor_series,
            }
        )
        if baseline is not None:
            raw_grp = df_test_raw[df_test_raw["unit"] == unit_id]
            diag = engine_diagnostics(fed_asym, scales, baseline, raw_grp, f"{subset}-U{unit_id}")
            assert len(diag["subsystem_share"]) == len(eval_cycles)
            traces[-1]["diagnostics"] = diag
    return traces


def load_cluster_rounds():
    """Load real physical 5-round traces and 15-round simulation trace."""
    phys_dir = ROOT / "results" / "fd002"
    seed_files = sorted(phys_dir.glob("results_physical_seed*.json"))
    physical_runs = []
    for path in seed_files:
        with open(path) as f:
            d = json.load(f)
        seed = d.get("config", {}).get("seed", path.stem)
        rmse_pts = d["metrics_centralized"]["model_rmse"]
        nasa_pts = d["metrics_centralized"]["nasa_score"]
        fit_t = dict(d.get("metrics_distributed_fit", {}).get("train_time_s", []))
        rss = dict(d.get("metrics_distributed_fit", {}).get("peak_rss_mb", []))
        physical_runs.append(
            {
                "seed": seed,
                "rounds": [
                    {
                        "round": int(r),
                        "rmse": round(float(val), 2),
                        "nasa": round(float(nasa_pts[idx][1]), 1),
                        "avg_train_s": round(float(fit_t.get(r, 0.0)), 2),
                        "avg_rss_mb": round(float(rss.get(r, 0.0)), 1),
                    }
                    for idx, (r, val) in enumerate(rmse_pts)
                ],
            }
        )

    sim15_path = phys_dir / "sim_fedavg_r15_e2.json"
    sim15 = []
    if sim15_path.exists():
        with open(sim15_path) as f:
            d15 = json.load(f)
        run42 = d15.get("runs", {}).get("42", [])
        sim15 = [
            {
                "round": int(item["round"]),
                "rmse": round(float(item["rmse"]), 2),
                "nasa": round(float(item["nasa"]), 1),
            }
            for item in run42
        ]
    return {"physical_5r": physical_runs, "sim_15r_seed42": sim15}


def load_summary_highlights():
    """Read key benchmark aggregates from summary.csv and asymmetric_loss_comparison.json."""
    asym_path = ROOT / "results" / "asymmetric_loss_comparison.json"
    with open(asym_path) as f:
        asym_data = json.load(f)

    repro_rows = []
    csv_path = ROOT / "results" / "reproducible" / "summary.csv"
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["scope"] in ("global", "client") and row["mean"]:
                repro_rows.append(
                    {
                        "dataset": row["dataset"],
                        "partition": row["partition"],
                        "method": row["method"],
                        "scope": row["scope"],
                        "client_id": row["client_id"],
                        "metric": row["metric"],
                        "mean": round(float(row["mean"]), 2),
                        "std": round(float(row["std"]), 2),
                    }
                )
    subsets = asym_data["subsets"]
    late_rates = {
        loss: [subsets[dataset]["losses"][loss]["summary"]["fedavg"]["late_rate_pct_mean"]
               for dataset in sorted(subsets)]
        for loss in ("mse", "asymmetric")
    }
    imbalance_rmse = {
        row["method"]: row["mean"]
        for row in repro_rows
        if row["dataset"] == "FD002"
        and row["partition"] == "engine_imbalance"
        and row["scope"] == "client"
        and row["client_id"] == "0"
        and row["metric"] == "rmse"
        and row["method"] in ("local_only", "fedavg")
    }
    return {
        "asymmetric": subsets,
        "reproducible": repro_rows,
        "highlights": {
            "fedavg_late_rate_pct": {
                loss: {"min": round(min(values), 1), "max": round(max(values), 1)}
                for loss, values in late_rates.items()
            },
            "fd002_imbalance_client_0_rmse": {
                "isolated_local": imbalance_rmse["local_only"],
                "federated": imbalance_rmse["fedavg"],
            },
        },
    }


def main():
    os.chdir(ROOT)
    model_tmp = CmapssRegressor()
    param_count = sum(p.numel() for p in model_tmp.parameters())
    tensor_bytes = sum(p.numel() * p.element_size() for p in model_tmp.parameters())

    engines = []
    for subset in ("FD001", "FD002"):
        scales, m_loc, m_fed_mse, m_fed_asym, baseline = train_subset_models(subset, seed=42)
        engines.extend(pick_representative_engines(subset, scales, m_loc, m_fed_mse, m_fed_asym, baseline))

    model_info = {
        "architecture": "1D-CNN (2 Conv layers 32/64 ch + MLP)",
        "window_cycles": SEQ_LEN,
        "rul_cap_cycles": MAX_RUL,
        "sensor_channels": len(USEFUL_SENSORS),
        "param_count": param_count,
        "tensor_payload_bytes": tensor_bytes,
        "round_exchange_bytes": tensor_bytes * 2,
    }
    cluster = load_cluster_rounds()
    cluster["highlights"] = {
        "tensor_payload_kb": round(tensor_bytes / 1000, 1),
        "round_trip_tensor_kb": round(tensor_bytes * 2 / 1000, 1),
    }
    payload = {
        "model_info": model_info,
        "engines": engines,
        "diagnostics_meta": diagnostics_meta(),
        "cluster": cluster,
        "benchmarks": load_summary_highlights(),
    }

    out_path = Path(__file__).resolve().parent / "demo_data.json"
    text = json.dumps(payload, indent=2)
    # keep the file diff-friendly but compact: inline innermost scalar arrays
    text = re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", lambda m: "[" + re.sub(r"\s*\n\s*", " ", m.group(1)) + "]", text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    update_file_replay_fallback(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KiB, {len(engines)} engines)")


if __name__ == "__main__":
    main()
