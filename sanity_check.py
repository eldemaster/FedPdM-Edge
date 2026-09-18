import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from common import load_cmapss_raw, CmapssRegressor, USEFUL_SENSORS, set_seed, extract_supervised_sequences, extract_last_sequence

def run_sanity_check(seed=42):
    set_seed(seed)
    df_raw = load_cmapss_raw('train_FD001.txt')
    
    with open('global_scale.json', 'r') as f:
        scales = json.load(f)
    min_vals = pd.Series(scales['min'])
    max_vals = pd.Series(scales['max'])
    train_mean_rul = scales['train_mean_rul']
    
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
    
    b_preds = np.full_like(y_te, train_mean_rul)
    baseline_rmse = np.sqrt(mean_squared_error(y_te, b_preds))
    
    # Untrained
    model = CmapssRegressor()
    model.eval()
    with torch.no_grad():
        preds = model(X_te_tensor).numpy()
    untrained_rmse = np.sqrt(mean_squared_error(y_te, preds))
    
    # Quick Train (2 epochs to beat 75% baseline)
    X_tr, y_tr = [], []
    for u, group in df_train.groupby('unit'):
        x, y = extract_supervised_sequences(group)
        X_tr.extend(x)
        y_tr.extend(y)
    
    loader = DataLoader(TensorDataset(torch.tensor(np.array(X_tr, dtype=np.float32)), torch.tensor(np.array(y_tr, dtype=np.float32))), batch_size=64, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for _ in range(2):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds_t = model(X_te_tensor).numpy()
    trained_rmse = np.sqrt(mean_squared_error(y_te, preds_t))
    
    return baseline_rmse, untrained_rmse, trained_rmse

if __name__ == "__main__":
    baseline, untrained, trained = run_sanity_check(42)
    print(f"Sanity Check -> Baseline RMSE: {baseline:.2f}, Untrained RMSE: {untrained:.2f}, Quick-Trained RMSE: {trained:.2f}")
    assert untrained > baseline, "untrained net should be worse than a constant"
    assert trained < baseline * 0.75, "model must beat the constant baseline by 25%"
    print("Sanity check passed with strict physical assertions!")
