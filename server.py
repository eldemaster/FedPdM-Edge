import json
import os
import sys

import flwr as fl
import numpy as np
import torch
from collections import OrderedDict

from common import (CmapssRegressor, set_seed, load_scales, build_test_set, nasa_score, rmse, predict)


def get_evaluate_fn():
    scales = load_scales()
    X_te, y_te = build_test_set(scales)
    X_te_t = torch.tensor(X_te)
    base = np.full_like(y_te, scales['train_mean_rul'])
    b_rmse, b_nasa = rmse(y_te, base), nasa_score(y_te, base)

    def evaluate(server_round, parameters, config):
        model = CmapssRegressor()
        sd = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(sd, strict=True)
        p = predict(model, X_te_t)
        r, n = rmse(y_te, p), nasa_score(y_te, p)
        print(f"Round {server_round} - Model RMSE: {r:.2f}, NASA Score: {n:.2f}", flush=True)
        return r, {"model_rmse": r, "baseline_rmse": float(b_rmse), "nasa_score": float(n)}
    return evaluate


def weighted_avg(metrics):
    tot = sum(n for n, _ in metrics)
    keys = metrics[0][1].keys() if metrics else []
    return {k: sum(n * m[k] for n, m in metrics) / tot for k in keys}


if __name__ == "__main__":
    seed = int(os.environ.get("SEED", 777))
    rounds = int(os.environ.get("ROUNDS", 5))
    n_clients = int(os.environ.get("MIN_CLIENTS", 2))
    mu = float(os.environ.get("PROXIMAL_MU", 0.0))          # >0 => FedProx local objective
    local_epochs = int(os.environ.get("LOCAL_EPOCHS", 2))
    out_dir = os.environ.get("RESULTS_DIR", "results/fd002")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    # Seeded initial weights from the server: otherwise Flower asks a random client,
    # making round 0 depend on which client connects first.
    init = fl.common.ndarrays_to_parameters([v.numpy() for v in CmapssRegressor().state_dict().values()])
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=n_clients, min_evaluate_clients=n_clients, min_available_clients=n_clients,
        initial_parameters=init,
        evaluate_fn=get_evaluate_fn(),
        on_fit_config_fn=lambda r: {"proximal_mu": mu, "local_epochs": local_epochs},
        fit_metrics_aggregation_fn=weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )
    history = fl.server.start_server(server_address=os.environ.get("BIND", "0.0.0.0:8081"),
                                     config=fl.server.ServerConfig(num_rounds=rounds), strategy=strategy)
    if not history.metrics_centralized:
        sys.exit("No centralized metrics recorded - clients never completed a round")
    tag = "fedprox" if mu > 0 else "fedavg"
    name = f"results_physical_seed{seed}.json" if mu == 0 else f"results_physical_{tag}_mu{mu}_seed{seed}.json"
    with open(os.path.join(out_dir, name), "w") as f:
        json.dump({"metrics_centralized": history.metrics_centralized,
                   "metrics_distributed_fit": history.metrics_distributed_fit,
                   "metrics_distributed": history.metrics_distributed,
                   "config": {"seed": seed, "rounds": rounds, "mu": mu, "local_epochs": local_epochs}}, f)
