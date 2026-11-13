"""Multi-seed comparison of symmetric MSE vs. AsymmetricRULLoss on FD001 and FD002.

Evaluates Isolated Local, FedAvg (5 rounds x 2 local epochs), and Centralized
(10 epochs) across 5 seeds (42, 123, 456, 789, 1024), recording Test RMSE,
NASA Asymmetric Score, and late-prediction rate (% of test units with y_pred > y_true).

Saves structured results to results/asymmetric_loss_comparison.json.
"""
import argparse
import json
import os
import time
import numpy as np
import torch

from common import (
    CmapssRegressor,
    USEFUL_SENSORS,
    apply_scale,
    build_test_set,
    extract_supervised_sequences,
    get_loss_fn,
    late_prediction_rate,
    load_cmapss_raw,
    load_partition,
    load_scales,
    make_loader,
    make_xy,
    nasa_score,
    predict,
    rmse,
    set_seed,
    split_units,
    train_epochs,
)


def compute_scales_from_train(df_raw):
    """Compute healthy-regime min/max scales (first 50 cycles per unit) and mean RUL."""
    head = df_raw.groupby("unit").head(50)[USEFUL_SENSORS]
    all_y = []
    for _, group in df_raw.groupby("unit"):
        _, y = extract_supervised_sequences(group)
        all_y.extend(y)
    return {
        "min": head.min().to_dict(),
        "max": head.max().to_dict(),
        "train_mean_rul": float(np.mean(all_y)),
    }


def eval_metrics(y_true, y_pred):
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "late_rate_pct": late_prediction_rate(y_true, y_pred) * 100.0,
    }


def run_fedavg(seed, df, part, X_te, y_te, loss_fn, rounds=5, local_epochs=2, lr=1e-3):
    set_seed(seed)
    gmodel = CmapssRegressor()
    clients = []
    for cid, units in sorted(part.items()):
        tr, _ = split_units(units, 0.1, seed=seed)
        X, y = make_xy(df, tr)
        set_seed(seed + int(cid))
        m = CmapssRegressor()
        clients.append(
            dict(
                cid=cid,
                model=m,
                loader=make_loader(X, y, 64, seed=seed + int(cid)),
                n=len(X),
                opt=torch.optim.Adam(m.parameters(), lr=lr),
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
    p = predict(gmodel, X_te)
    return eval_metrics(y_te, p)


def run_subset(subset, seeds, epochs=10, rounds=5, local_epochs=2, alpha_late=2.5, alpha_early=0.5):
    if subset == "FD002":
        scales = load_scales("global_scale.json")
        part = load_partition("partition.json")
        df_raw = load_cmapss_raw("train_FD002.txt")
    else:
        df_raw = load_cmapss_raw(f"train_{subset}.txt")
        scales = compute_scales_from_train(df_raw)
        units = sorted(int(u) for u in df_raw["unit"].unique())
        part = {
            "1": [u for u in units if u % 2 == 1],
            "2": [u for u in units if u % 2 == 0],
        }

    df = apply_scale(df_raw, scales)
    X_te, y_te = build_test_set(scales, subset=subset)
    base_pred = np.full_like(y_te, scales["train_mean_rul"])
    baseline_metrics = eval_metrics(y_te, base_pred)

    subset_res = {
        "n_test_engines": int(len(y_te)),
        "baseline": baseline_metrics,
        "losses": {},
    }

    for loss_name in ("mse", "asymmetric"):
        loss_kwargs = {"alpha_late": alpha_late, "alpha_early": alpha_early} if loss_name == "asymmetric" else {}
        loss_fn = get_loss_fn(loss_name, **loss_kwargs)
        per_seed = {}
        for seed in seeds:
            set_seed(seed)
            data = {}
            for cid, units in sorted(part.items()):
                tr, _ = split_units(units, 0.1, seed=seed)
                data[cid] = make_xy(df, tr)
            allX = np.concatenate([d[0] for d in data.values()])
            ally = np.concatenate([d[1] for d in data.values()])

            # 1. Isolated Local
            local_runs = []
            for cid, (X, y) in data.items():
                set_seed(seed + int(cid))
                m = train_epochs(
                    CmapssRegressor(),
                    make_loader(X, y, 64, seed=seed + int(cid)),
                    epochs,
                    loss_fn=loss_fn,
                )
                local_runs.append(eval_metrics(y_te, predict(m, X_te)))
            local_avg = {
                k: float(np.mean([r[k] for r in local_runs]))
                for k in ("rmse", "nasa_score", "late_rate_pct")
            }

            # 2. FedAvg
            fed_metrics = run_fedavg(
                seed, df, part, X_te, y_te, loss_fn=loss_fn, rounds=rounds, local_epochs=local_epochs
            )

            # 3. Centralized
            set_seed(seed)
            m_cent = train_epochs(
                CmapssRegressor(),
                make_loader(allX, ally, 64, seed=seed),
                epochs,
                loss_fn=loss_fn,
            )
            cent_metrics = eval_metrics(y_te, predict(m_cent, X_te))

            per_seed[str(seed)] = {
                "local": local_avg,
                "fedavg": fed_metrics,
                "central": cent_metrics,
            }
            print(
                f"[{subset} | {loss_name:10s} | seed={seed:4d}] "
                f"Local: RMSE={local_avg['rmse']:.2f} NASA={local_avg['nasa_score']:.0f} Late={local_avg['late_rate_pct']:.1f}% | "
                f"FedAvg: RMSE={fed_metrics['rmse']:.2f} NASA={fed_metrics['nasa_score']:.0f} Late={fed_metrics['late_rate_pct']:.1f}% | "
                f"Central: RMSE={cent_metrics['rmse']:.2f} NASA={cent_metrics['nasa_score']:.0f} Late={cent_metrics['late_rate_pct']:.1f}%",
                flush=True,
            )

        summary = {}
        for setup in ("local", "fedavg", "central"):
            summary[setup] = {}
            for metric in ("rmse", "nasa_score", "late_rate_pct"):
                vals = [per_seed[str(s)][setup][metric] for s in seeds]
                summary[setup][f"{metric}_mean"] = float(np.mean(vals))
                summary[setup][f"{metric}_std"] = float(np.std(vals))

        subset_res["losses"][loss_name] = {
            "per_seed": per_seed,
            "summary": summary,
        }

    return subset_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024])
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--alpha-late", type=float, default=2.5)
    ap.add_argument("--alpha-early", type=float, default=0.5)
    ap.add_argument("--out", default="results/asymmetric_loss_comparison.json")
    args = ap.parse_args()

    t0 = time.perf_counter()
    results = {
        "config": {
            "seeds": args.seeds,
            "subsets": args.subsets,
            "epochs": args.epochs,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "alpha_late": args.alpha_late,
            "alpha_early": args.alpha_early,
        },
        "subsets": {},
    }

    for subset in args.subsets:
        results["subsets"][subset] = run_subset(
            subset,
            args.seeds,
            epochs=args.epochs,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            alpha_late=args.alpha_late,
            alpha_early=args.alpha_early,
        )

    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {args.out} in {results['elapsed_s']}s")


if __name__ == "__main__":
    main()
