"""Constant / Local-only / Centralized baselines on official test_FD002 (all 259 engines), plus the
federated rows read from results/. Writes results/fd002/benchmark_e<epochs>.json + .csv.
All three trained models use the same 260-unit partition and the same epoch budget."""
import argparse, csv, glob, json, os
import numpy as np
from common import (CmapssRegressor, set_seed, load_cmapss_raw, load_scales, load_partition, apply_scale,
                    build_test_set, make_xy, make_loader, split_units, train_epochs, predict, rmse, nasa_score,
                    get_loss_fn)


def run_benchmark(seed, epochs, df, scales, part, X_te, y_te, loss_fn=None):
    set_seed(seed)
    base = np.full_like(y_te, scales['train_mean_rul'])
    res = {"baseline": (rmse(y_te, base), nasa_score(y_te, base))}
    data = {}
    for cid, units in sorted(part.items()):
        tr, _ = split_units(units, 0.1, seed=seed)       # same train units as federated clients
        data[cid] = make_xy(df, tr)
    allX = np.concatenate([d[0] for d in data.values()]); ally = np.concatenate([d[1] for d in data.values()])
    local = []
    for cid, (X, y) in data.items():
        set_seed(seed + int(cid))
        m = train_epochs(CmapssRegressor(), make_loader(X, y, 64, seed + int(cid)), epochs, loss_fn=loss_fn)
        p = predict(m, X_te); local.append((rmse(y_te, p), nasa_score(y_te, p)))
    res["local"] = tuple(np.mean(local, axis=0))
    set_seed(seed)
    m = train_epochs(CmapssRegressor(), make_loader(allX, ally, 64, seed), epochs, loss_fn=loss_fn)
    p = predict(m, X_te); res["central"] = (rmse(y_te, p), nasa_score(y_te, p))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024, 7, 777])
    ap.add_argument("--loss-fn", default=os.environ.get("LOSS_FN", "mse"), choices=["mse", "asymmetric"])
    ap.add_argument("--alpha-late", type=float, default=None)
    ap.add_argument("--alpha-early", type=float, default=None)
    a = ap.parse_args()
    loss_kwargs = {}
    if a.loss_fn == "asymmetric":
        if a.alpha_late is not None:
            loss_kwargs["alpha_late"] = a.alpha_late
        elif "ALPHA_LATE" in os.environ:
            loss_kwargs["alpha_late"] = float(os.environ["ALPHA_LATE"])
        if a.alpha_early is not None:
            loss_kwargs["alpha_early"] = a.alpha_early
        elif "ALPHA_EARLY" in os.environ:
            loss_kwargs["alpha_early"] = float(os.environ["ALPHA_EARLY"])
    loss_fn = get_loss_fn(a.loss_fn, **loss_kwargs)

    scales, part = load_scales(), load_partition()
    df = apply_scale(load_cmapss_raw('train_FD002.txt'), scales)
    X_te, y_te = build_test_set(scales)
    per_seed = {s: run_benchmark(s, a.epochs, df, scales, part, X_te, y_te, loss_fn=loss_fn) for s in a.seeds}
    for s, r in per_seed.items(): print(s, {k: (round(v[0], 2), round(v[1])) for k, v in r.items()}, flush=True)
    rows = []
    for k in ["baseline", "local", "central"]:
        arr = np.array([per_seed[s][k] for s in a.seeds])
        rows.append(dict(method=k, epochs=a.epochs, n=len(a.seeds), rmse_mean=arr[:, 0].mean(), rmse_std=arr[:, 0].std(),
                         nasa_mean=arr[:, 1].mean(), nasa_std=arr[:, 1].std()))
    # federated rows: physical-node runs found on disk (missing seeds are reported, not hidden)
    for path_glob, name in [("results/fd002/results_physical_seed*.json", "federated_physical_fedavg")]:
        r, n, found = [], [], []
        for f in sorted(glob.glob(path_glob)):
            d = json.load(open(f))['metrics_centralized']
            r.append(d['model_rmse'][-1][1]); n.append(d['nasa_score'][-1][1]); found.append(f)
        if r:
            rows.append(dict(method=name, epochs=None, n=len(r), rmse_mean=np.mean(r), rmse_std=np.std(r),
                             nasa_mean=np.mean(n), nasa_std=np.std(n)))
        else:
            print(f"WARNING: no federated result files match {path_glob}")
    os.makedirs("results/fd002", exist_ok=True)
    suffix = "" if a.loss_fn == "mse" else f"_{a.loss_fn}"
    json.dump({"per_seed": {str(s): v for s, v in per_seed.items()}, "summary": rows, "loss_fn": a.loss_fn},
              open(f"results/fd002/benchmark_e{a.epochs}{suffix}.json", "w"), indent=1, default=float)
    with open(f"results/fd002/benchmark_e{a.epochs}{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    for r_ in rows:
        print(f"{r_['method']:28s} RMSE {r_['rmse_mean']:.2f} ± {r_['rmse_std']:.2f} | NASA {r_['nasa_mean']:.0f} ± {r_['nasa_std']:.0f} (n={r_['n']})")
    print("Note: True RUL capped at 125 cycles; test = last window of all 259 FD002 engines.")
