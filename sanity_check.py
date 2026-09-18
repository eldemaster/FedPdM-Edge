import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from common import load_cmapss_raw, CmapssRegressor, USEFUL_SENSORS, set_seed, extract_supervised_sequences

def run_sanity_check(seed):
    set_seed(seed)
    df_raw = load_cmapss_raw()
    unique_units = df_raw['unit'].unique()
    np.random.shuffle(unique_units)
    
    global_test_units = unique_units[-10:]
    train_units = unique_units[:-10]
    
    with open('global_scale.json', 'r') as f:
        scales = json.load(f)
    min_vals = pd.Series(scales['min'])
    max_vals = pd.Series(scales['max'])
    
    df_train = df_raw[df_raw['unit'].isin(train_units)].copy()
    df_train[USEFUL_SENSORS] = (df_train[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    X_tr, y_tr = [], []
    for u, group in df_train.groupby('unit'):
        x, y = extract_supervised_sequences(group)
        X_tr.extend(x)
        y_tr.extend(y)
    
    df_test = df_raw[df_raw['unit'].isin(global_test_units)].copy()
    df_test[USEFUL_SENSORS] = (df_test[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    X_te, y_te = [], []
    for u, group in df_test.groupby('unit'):
        x, y = extract_supervised_sequences(group)
        X_te.extend(x)
        y_te.extend(y)
    
    y_te = np.array(y_te)
    train_mean_rul = np.mean(y_tr)
    baseline_preds = np.full_like(y_te, train_mean_rul)
    baseline_rmse = np.sqrt(mean_squared_error(y_te, baseline_preds))
    
    model = CmapssRegressor()
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(np.array(X_te, dtype=np.float32))).numpy()
    untrained_rmse = np.sqrt(mean_squared_error(y_te, preds))
    
    return baseline_rmse, untrained_rmse

if __name__ == "__main__":
    baseline, untrained = run_sanity_check(42)
    print(f"Sanity Check -> Baseline RMSE: {baseline:.2f}, Untrained NN RMSE: {untrained:.2f}")
    assert untrained > baseline, "untrained net should be worse than a constant"
    print("Sanity check passed with strict assertions!")
