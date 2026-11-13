"""In-process federated simulation (FedAvg / FedProx) using the exact same local-training code
as client.py. No network: used for seed sweeps / ablations. Real multi-node runs use server.py+client.py."""
import json, os, sys, time, argparse
import numpy as np
import torch
from common import (CmapssRegressor, set_seed, load_cmapss_raw, load_scales, load_partition, apply_scale,
                    build_test_set, make_xy, make_loader, split_units, train_epochs, predict, rmse, nasa_score,
                    get_loss_fn)


def run(seed, rounds=5, local_epochs=2, mu=0.0, lr=1e-3, scales=None, df=None, test=None, part=None, loss_fn=None):
    scales = scales or load_scales(); part = part or load_partition()
    df = df if df is not None else apply_scale(load_cmapss_raw('train_FD002.txt'), scales)
    X_te, y_te = test or build_test_set(scales)
    set_seed(seed)
    gmodel = CmapssRegressor()
    clients = []
    for cid, units in sorted(part.items()):
        tr, va = split_units(units, 0.1, seed=seed)
        X, y = make_xy(df, tr); Xv, yv = make_xy(df, va)
        set_seed(seed + int(cid))
        m = CmapssRegressor()
        clients.append(dict(cid=cid, model=m, loader=make_loader(X, y, 64, seed=seed + int(cid)), n=len(X),
                            opt=torch.optim.Adam(m.parameters(), lr=lr), Xv=Xv, yv=yv))
    hist = []
    for r in range(1, rounds + 1):
        gstate = {k: v.clone() for k, v in gmodel.state_dict().items()}
        states, ns, times, vals = [], [], [], []
        for c in clients:
            c['model'].load_state_dict(gstate)
            glob = [p.detach().clone() for p in c['model'].parameters()]
            t0 = time.perf_counter()
            train_epochs(c['model'], c['loader'], local_epochs, mu=mu, global_params=glob, optimizer=c['opt'],
                         loss_fn=loss_fn)
            times.append(time.perf_counter() - t0)
            states.append({k: v.clone() for k, v in c['model'].state_dict().items()}); ns.append(c['n'])
        tot = sum(ns)
        gmodel.load_state_dict({k: sum(s[k].float() * n for s, n in zip(states, ns)) / tot for k in gstate})
        for c in clients:
            c['model'].load_state_dict(gmodel.state_dict())
            vals.append(rmse(c['yv'], predict(c['model'], c['Xv'])))
        p = predict(gmodel, X_te)
        hist.append(dict(round=r, rmse=rmse(y_te, p), nasa=nasa_score(y_te, p), val_rmse=vals, train_time_s=times))
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2024, 7, 777])
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mu", type=float, default=0.0); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-fn", default=os.environ.get("LOSS_FN", "mse"), choices=["mse", "asymmetric"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    loss_fn = get_loss_fn(a.loss_fn)
    scales = load_scales(); part = load_partition()
    df = apply_scale(load_cmapss_raw('train_FD002.txt'), scales); test = build_test_set(scales)
    out = {"config": vars(a), "runs": {}}
    for s in a.seeds:
        h = run(s, a.rounds, a.epochs, a.mu, a.lr, scales, df, test, part, loss_fn=loss_fn)
        out["runs"][str(s)] = h
        print(f"seed {s}: RMSE {h[-1]['rmse']:.2f} NASA {h[-1]['nasa']:.0f}", flush=True)
    r = [v[-1]['rmse'] for v in out["runs"].values()]; n = [v[-1]['nasa'] for v in out["runs"].values()]
    out["summary"] = dict(rmse_mean=float(np.mean(r)), rmse_std=float(np.std(r)), nasa_mean=float(np.mean(n)), nasa_std=float(np.std(n)))
    print(out["summary"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
