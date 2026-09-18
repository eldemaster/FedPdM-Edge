import flwr as fl
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
from common import load_cmapss_raw, CmapssRegressor, USEFUL_SENSORS, set_seed, extract_supervised_sequences

def get_evaluate_fn():
    df_raw = load_cmapss_raw()
    unique_units = df_raw['unit'].unique()
    np.random.shuffle(unique_units)
    global_test_units = unique_units[-10:]
    train_units = unique_units[:-10]
    
    with open('global_scale.json', 'r') as f:
        scales = json.load(f)
    min_vals = pd.Series(scales['min'])
    max_vals = pd.Series(scales['max'])
    
    df_test = df_raw[df_raw['unit'].isin(global_test_units)].copy()
    df_test[USEFUL_SENSORS] = (df_test[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
    X_te, y_te = [], []
    for u, group in df_test.groupby('unit'):
        x, y = extract_supervised_sequences(group)
        X_te.extend(x)
        y_te.extend(y)
    
    X_te_tensor = torch.tensor(np.array(X_te, dtype=np.float32))
    y_te = np.array(y_te)
    
    df_train = df_raw[df_raw['unit'].isin(train_units)].copy()
    df_train_rul = []
    for u, group in df_train.groupby('unit'):
        _, y = extract_supervised_sequences(group)
        df_train_rul.extend(y)
    train_mean_rul = np.mean(df_train_rul)
    baseline_preds = np.full_like(y_te, train_mean_rul)
    baseline_rmse = np.sqrt(mean_squared_error(y_te, baseline_preds))
    
    def evaluate(server_round, parameters, config):
        model = CmapssRegressor()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        with torch.no_grad():
            preds = model(X_te_tensor).numpy()
            rmse = np.sqrt(mean_squared_error(y_te, preds))
            
        print(f"Round {server_round} - Model RMSE: {rmse:.2f}, Baseline RMSE: {baseline_rmse:.2f}")
        return float(rmse), {"model_rmse": float(rmse), "baseline_rmse": float(baseline_rmse)}
    
    return evaluate

if __name__ == "__main__":
    set_seed(42)
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
    
    res = {
        "losses_distributed": history.losses_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_centralized": history.metrics_centralized,
    }
    with open("results_physical.json", "w") as f:
        json.dump(res, f)
