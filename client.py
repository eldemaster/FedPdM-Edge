import argparse
import os
import sys
import time
import resource
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch

from common import (CmapssRegressor, USEFUL_SENSORS, set_seed, load_cmapss_raw, load_scales,
                    load_partition, apply_scale, make_xy, make_loader, split_units,
                    train_epochs, predict, rmse, get_loss_fn)


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # bytes on macOS, KiB on Linux


class SupervisedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_x, val_y, lr=1e-3, loss_fn=None):
        self.model = model
        self.train_loader = train_loader
        self.val_x, self.val_y = val_x, val_y
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn if loss_fn is not None else get_loss_fn("mse")

    def get_parameters(self, config):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        sd = OrderedDict({k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        mu = float(config.get("proximal_mu", 0.0))
        epochs = int(config.get("local_epochs", 2))
        glob = [p.detach().clone() for p in self.model.parameters()]
        t0 = time.perf_counter()
        train_epochs(self.model, self.train_loader, epochs, mu=mu, global_params=glob,
                     optimizer=self.optimizer, loss_fn=self.loss_fn)
        dt = time.perf_counter() - t0
        print(f"FIT mu={mu} epochs={epochs} train_time_s={dt:.2f} peak_rss_mb={peak_rss_mb():.1f}", flush=True)
        return self.get_parameters({}), len(self.train_loader.dataset), \
            {"train_time_s": dt, "peak_rss_mb": peak_rss_mb()}

    def evaluate(self, parameters, config):
        """RMSE on held-out *units* of this client (never trained on)."""
        self.set_parameters(parameters)
        r = rmse(self.val_y, predict(self.model, self.val_x))
        return r, len(self.val_y), {"val_rmse": r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss-fn", default=os.environ.get("LOSS_FN", "mse"), choices=["mse", "asymmetric"])
    ap.add_argument("--alpha-late", type=float, default=None)
    ap.add_argument("--alpha-early", type=float, default=None)
    args, _ = ap.parse_known_args()

    loss_kwargs = {}
    if args.loss_fn == "asymmetric":
        if args.alpha_late is not None:
            loss_kwargs["alpha_late"] = args.alpha_late
        elif "ALPHA_LATE" in os.environ:
            loss_kwargs["alpha_late"] = float(os.environ["ALPHA_LATE"])
        if args.alpha_early is not None:
            loss_kwargs["alpha_early"] = args.alpha_early
        elif "ALPHA_EARLY" in os.environ:
            loss_kwargs["alpha_early"] = float(os.environ["ALPHA_EARLY"])
    loss_fn = get_loss_fn(args.loss_fn, **loss_kwargs)

    seed = int(os.environ.get("SEED", 42))
    cid = os.environ.get("CLIENT_ID")
    if cid is None:
        sys.exit("CLIENT_ID must be set (matches a key in partition.json)")
    fl_server = os.environ.get("FL_SERVER")
    if not fl_server:
        sys.exit("FL_SERVER must be set, e.g. FL_SERVER=<server-ip>:8081")
    part = load_partition()
    if cid not in part:
        sys.exit(f"CLIENT_ID={cid!r} not in partition.json keys {list(part)}")
    set_seed(seed + int(cid))

    df = apply_scale(load_cmapss_raw('train_FD002.txt'), load_scales())
    tr_units, va_units = split_units(part[cid], 0.1, seed=seed)
    X, y = make_xy(df, tr_units)
    Xv, yv = make_xy(df, va_units)
    loader = make_loader(X, y, 64, seed=seed + int(cid))
    print(f"client {cid}: {len(tr_units)} train units ({len(X)} windows), {len(va_units)} val units, loss={args.loss_fn}", flush=True)
    client = SupervisedClient(CmapssRegressor(), loader, torch.tensor(Xv), yv, loss_fn=loss_fn)
    fl.client.start_client(server_address=fl_server, client=client.to_client())
