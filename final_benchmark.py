import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json
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
    df_raw = load_cmapss_raw('train_FD001.txt')
    
    with open('global_scale.json', 'r') as f:
        scales = json.load(f)
    min_vals = pd.Series(scales['min'])
    max_vals = pd.Series(scales['max'])
    train_mean_rul = scales['train_mean_rul']
    
    with open('partition.json', 'r') as f:
        part = json.load(f)
        units_c1 = part["1"]
        units_c2 = part["2"]
        
    df_train = df_raw.copy()
    df_train[USEFUL_SENSORS] = (df_train[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
    df_test = load_cmapss_raw('test_FD001.txt')
    df_test[USEFUL_SENSORS] = (df_test[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
    import zipfile
    with zipfile.ZipFile('/home/eldemaster/CMAPSSData.zip', 'r') as z:
        with z.open('RUL_FD001.txt') as f:
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
    
    # Baseline
    b_preds = np.full_like(y_te, train_mean_rul)
    b_rmse = np.sqrt(mean_squared_error(y_te, b_preds))
    b_score = nasa_score(y_te, b_preds)
    
    # Datasets
    X_c1, y_c1, X_c2, y_c2, X_all, y_all = [], [], [], [], [], []
    for u, group in df_train.groupby('unit'):
        x, y = extract_supervised_sequences(group)
        if int(u) in units_c1:
            X_c1.extend(x); y_c1.extend(y)
        elif int(u) in units_c2:
            X_c2.extend(x); y_c2.extend(y)
        X_all.extend(x); y_all.extend(y)
        
    loader_c1 = DataLoader(TensorDataset(torch.tensor(np.array(X_c1, dtype=np.float32)), torch.tensor(np.array(y_c1, dtype=np.float32))), batch_size=64, shuffle=True)
    loader_c2 = DataLoader(TensorDataset(torch.tensor(np.array(X_c2, dtype=np.float32)), torch.tensor(np.array(y_c2, dtype=np.float32))), batch_size=64, shuffle=True)
    loader_all = DataLoader(TensorDataset(torch.tensor(np.array(X_all, dtype=np.float32)), torch.tensor(np.array(y_all, dtype=np.float32))), batch_size=64, shuffle=True)
    
    # Train
    m_c1 = train_model(loader_c1, epochs=10) # 10 epochs equals 5 rounds * 2 local epochs
    m_c2 = train_model(loader_c2, epochs=10)
    m_all = train_model(loader_all, epochs=10)
    
    rmse_c1, score_c1 = evaluate_model(m_c1, X_te_tensor, y_te)
    rmse_c2, score_c2 = evaluate_model(m_c2, X_te_tensor, y_te)
    rmse_all, score_all = evaluate_model(m_all, X_te_tensor, y_te)
    
    local_rmse = (rmse_c1 + rmse_c2) / 2
    local_score = (score_c1 + score_c2) / 2
    
    return b_rmse, b_score, local_rmse, local_score, rmse_all, score_all

seeds = [42, 100, 2024]
results = []
for s in seeds:
    results.append(run_benchmark(s))

results = np.array(results)
means = np.mean(results, axis=0)
stds = np.std(results, axis=0)

# Read federated RMSE from last run
try:
    with open('results/results_physical.json', 'r') as f:
        fed_data = json.load(f)['metrics_centralized']
        fed_rmse = fed_data['model_rmse'][-1][1]
        fed_score = fed_data['nasa_score'][-1][1]
except Exception as e:
    fed_rmse, fed_score = 0.0, 0.0

print("\n--- FINAL BENCHMARK TABLE (Official test_FD001.txt) ---")
print(f"Constant Baseline : RMSE = {means[0]:.2f} ± {stds[0]:.2f} | NASA Score = {means[1]:.2f} ± {stds[1]:.2f}")
print(f"Local-Only (avg)  : RMSE = {means[2]:.2f} ± {stds[2]:.2f} | NASA Score = {means[3]:.2f} ± {stds[3]:.2f}")
print(f"Federated (FedAvg): RMSE = {fed_rmse:.2f}         | NASA Score = {fed_score:.2f}")
print(f"Centralized (10 e): RMSE = {means[4]:.2f} ± {stds[4]:.2f} | NASA Score = {means[5]:.2f} ± {stds[5]:.2f}")
