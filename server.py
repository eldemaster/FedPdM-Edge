import os
import flwr as fl
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
from common import load_cmapss_raw, CmapssRegressor, USEFUL_SENSORS, set_seed, extract_last_sequence, nasa_score

def get_evaluate_fn():
    with open('global_scale.json', 'r') as f:
        scales = json.load(f)
    min_vals = pd.Series(scales['min'])
    max_vals = pd.Series(scales['max'])
    train_mean_rul = scales['train_mean_rul']
    
    # Load official test set and RULs
    df_test = load_cmapss_raw('test_FD002.txt')
    df_test[USEFUL_SENSORS] = (df_test[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
    import zipfile
    with zipfile.ZipFile('/home/eldemaster/CMAPSSData.zip', 'r') as z:
        with z.open('RUL_FD002.txt') as f:
            true_ruls = pd.read_csv(f, header=None).values.flatten()
            
    X_te, y_te = [], []
    for u, group in df_test.groupby('unit'):
        idx = int(u) - 1
        x, y = extract_last_sequence(group, true_ruls[idx])
        if x is not None:
            X_te.append(x)
            y_te.append(y)
            
    X_te_tensor = torch.tensor(np.array(X_te, dtype=np.float32))
    y_te = np.array(y_te)
    
    baseline_preds = np.full_like(y_te, train_mean_rul)
    baseline_rmse = np.sqrt(mean_squared_error(y_te, baseline_preds))
    baseline_nasa = nasa_score(y_te, baseline_preds)
    
    def evaluate(server_round, parameters, config):
        model = CmapssRegressor()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        with torch.no_grad():
            preds = model(X_te_tensor).numpy()
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            n_score = nasa_score(y_te, preds)
            
        print(f"Round {server_round} - Model RMSE: {rmse:.2f}, NASA Score: {n_score:.2f}")
        return float(rmse), {"model_rmse": float(rmse), "baseline_rmse": float(baseline_rmse), "nasa_score": float(n_score)}
    
    return evaluate

if __name__ == "__main__":
    seed = int(os.environ.get("SEED", 777))
    set_seed(seed)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn()
    )
    history = fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )
    with open(f"results/fd002/results_physical_seed{seed}.json", "w") as f:
        json.dump({"metrics_centralized": history.metrics_centralized}, f)
