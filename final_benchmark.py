import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json
import os
from common import load_cmapss_raw, CmapssRegressor, USEFUL_SENSORS, set_seed, extract_supervised_sequences, extract_last_sequence, nasa_score
from sklearn.metrics import mean_squared_error

def train_model(train_loader, epochs=10):
    model = CmapssRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        preds = model(X_te).numpy()
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    score = nasa_score(y_te, preds)
    return rmse, score

def run_benchmark(seed):
    set_seed(seed)
    df_raw = load_cmapss_raw('train_FD002.txt')
    
    with open('global_scale.json', 'r') as f:
        scales = json.load(f)
    min_vals = pd.Series(scales['min'])
    max_vals = pd.Series(scales['max'])
    train_mean_rul = scales['train_mean_rul']
    
    with open('partition.json', 'r') as f:
        part = json.load(f)
        units_c1 = part["1"]
        units_c2 = part["2"]
        # FIX P23: Only use partition units for all!
        valid_units = set(units_c1 + units_c2)
        
    df_train = df_raw.copy()
    df_train[USEFUL_SENSORS] = (df_train[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
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
    
    b_preds = np.full_like(y_te, train_mean_rul)
    b_rmse = np.sqrt(mean_squared_error(y_te, b_preds))
    b_score = nasa_score(y_te, b_preds)
    
    X_c1, y_c1, X_c2, y_c2, X_all, y_all = [], [], [], [], [], []
    for u, group in df_train.groupby('unit'):
        u_int = int(u)
        if u_int not in valid_units: continue
        x, y = extract_supervised_sequences(group)
        if u_int in units_c1:
            X_c1.extend(x); y_c1.extend(y)
        elif u_int in units_c2:
            X_c2.extend(x); y_c2.extend(y)
        X_all.extend(x); y_all.extend(y)
        
    loader_c1 = DataLoader(TensorDataset(torch.tensor(np.array(X_c1, dtype=np.float32)), torch.tensor(np.array(y_c1, dtype=np.float32))), batch_size=64, shuffle=True)
    loader_c2 = DataLoader(TensorDataset(torch.tensor(np.array(X_c2, dtype=np.float32)), torch.tensor(np.array(y_c2, dtype=np.float32))), batch_size=64, shuffle=True)
    loader_all = DataLoader(TensorDataset(torch.tensor(np.array(X_all, dtype=np.float32)), torch.tensor(np.array(y_all, dtype=np.float32))), batch_size=64, shuffle=True)
    
    m_c1 = train_model(loader_c1, epochs=10)
    m_c2 = train_model(loader_c2, epochs=10)
    m_all = train_model(loader_all, epochs=10)
    
    rmse_c1, score_c1 = evaluate_model(m_c1, X_te_tensor, y_te)
    rmse_c2, score_c2 = evaluate_model(m_c2, X_te_tensor, y_te)
    rmse_all, score_all = evaluate_model(m_all, X_te_tensor, y_te)
    
    local_rmse = (rmse_c1 + rmse_c2) / 2
    local_score = (score_c1 + score_c2) / 2
    
    return b_rmse, b_score, local_rmse, local_score, rmse_all, score_all

seeds = [42, 100, 2024, 7, 777]
results = []
for s in seeds:
    results.append(run_benchmark(s))

results = np.array(results)
means = np.mean(results, axis=0)
stds = np.std(results, axis=0)

# Process 5 federated seeds
fed_rmses = []
fed_scores = []
for s in seeds:
    try:
        with open(f'results/fd002/results_physical_seed{s}.json', 'r') as f:
            fed_data = json.load(f)['metrics_centralized']
            fed_rmses.append(fed_data['model_rmse'][-1][1])
            fed_scores.append(fed_data['nasa_score'][-1][1])
    except Exception as e:
        pass

if len(fed_rmses) > 0:
    fed_rmse_mean, fed_rmse_std = np.mean(fed_rmses), np.std(fed_rmses)
    fed_score_mean, fed_score_std = np.mean(fed_scores), np.std(fed_scores)
else:
    fed_rmse_mean, fed_rmse_std, fed_score_mean, fed_score_std = 0,0,0,0

print("\n--- FINAL BENCHMARK TABLE (Official test_FD002.txt) ---")
print(f"Constant Baseline : RMSE = {means[0]:.2f} ± {stds[0]:.2f} | NASA Score = {means[1]:.2f} ± {stds[1]:.2f}")
print(f"Local-Only (avg)  : RMSE = {means[2]:.2f} ± {stds[2]:.2f} | NASA Score = {means[3]:.2f} ± {stds[3]:.2f}")
print(f"Federated (FedAvg): RMSE = {fed_rmse_mean:.2f} ± {fed_rmse_std:.2f} | NASA Score = {fed_score_mean:.2f} ± {fed_score_std:.2f}")
print(f"Centralized (10 e): RMSE = {means[4]:.2f} ± {stds[4]:.2f} | NASA Score = {means[5]:.2f} ± {stds[5]:.2f}")
print("Note: True RUL is capped at 125 cycles during evaluation as per standard CMAPSS practice.")
