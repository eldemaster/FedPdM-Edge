import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from common import load_cmapss_raw, CmapssRegressor, USEFUL_SENSORS, set_seed, extract_supervised_sequences

class SupervisedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(2):
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(self.train_loader.dataset), {}

if __name__ == "__main__":
    set_seed(42)
    df_raw = load_cmapss_raw()
    unique_units = df_raw['unit'].unique()
    np.random.shuffle(unique_units)
    
    if "CLIENT_ID" not in os.environ:
        raise ValueError("CLIENT_ID must be set!")
        
    partition = int(os.environ["CLIENT_ID"])
    if partition == 1:
        units = unique_units[:45]
    else:
        units = unique_units[45:90]
        
    with open('global_scale.json', 'r') as f:
        scales = json.load(f)
    min_vals = pd.Series(scales['min'])
    max_vals = pd.Series(scales['max'])
        
    df_sub = df_raw[df_raw['unit'].isin(units)].copy()
    # USE GLOBAL SCALES INSTEAD OF LOCAL!
    df_sub[USEFUL_SENSORS] = (df_sub[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
    X, y = [], []
    for u, group in df_sub.groupby('unit'):
        x_seq, y_rul = extract_supervised_sequences(group)
        X.extend(x_seq)
        y.extend(y_rul)
        
    X_tensor = torch.tensor(np.array(X, dtype=np.float32))
    y_tensor = torch.tensor(np.array(y, dtype=np.float32))
    
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)
    
    fl.client.start_numpy_client(
        server_address="192.168.1.144:8081",
        client=SupervisedClient(CmapssRegressor(), train_loader)
    )
