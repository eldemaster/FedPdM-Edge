import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from collections import OrderedDict

DATA_FILE = "/home/eldemaster/CMAPSSData/train_FD001.txt"
SEQ_LEN = 30
USEFUL_SENSORS = [
    's2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21'
]
INPUT_DIM = len(USEFUL_SENSORS)

def load_cmapss_raw():
    cols = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(DATA_FILE, sep=r'\s+', header=None, names=cols)
    return df

class CmapssAutoencoder(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, seq_len=SEQ_LEN):
        super(CmapssAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        flat_dim = input_dim * seq_len
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, flat_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        encoded = self.encoder(x_flat)
        decoded = self.decoder(encoded)
        return decoded.view(batch_size, self.seq_len, self.input_dim)

class CmapssClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, mu):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.mu = mu
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
        
        # Save global weights for FedProx
        global_weights = [param.clone().detach() for param in self.model.parameters()]
        
        self.model.train()
        for epoch in range(2):
            for batch in self.train_loader:
                x = batch[0]
                self.optimizer.zero_grad()
                out = self.model(x)
                
                loss = self.criterion(out, x)
                
                # Add Proximal Term for FedProx
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), global_weights):
                    proximal_term += ((local_param - global_param) ** 2).sum()
                loss += (self.mu / 2) * proximal_term
                
                loss.backward()
                self.optimizer.step()
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch[0]
                out = self.model(x)
                total_loss += self.criterion(out, x).item() * x.size(0)
        avg_loss = total_loss / len(self.val_loader.dataset)
        return float(avg_loss), len(self.val_loader.dataset), {"mse": float(avg_loss)}

df_raw = load_cmapss_raw()
unique_units = df_raw['unit'].unique()
np.random.shuffle(unique_units)

# Holdout for Global Evaluation
global_test_units = unique_units[-10:]
client_units = np.array_split(unique_units[:-10], 5)

def extract_sequences(df_group, is_healthy):
    data = df_group[USEFUL_SENSORS].values
    seqs = []
    max_cycle = len(data)
    for i in range(max_cycle - SEQ_LEN):
        seq = data[i:i+SEQ_LEN]
        if is_healthy and i < 50:
            seqs.append(seq)
        elif not is_healthy and i > max_cycle - 50:
            seqs.append(seq)
    return seqs

def build_client_data(units):
    df_sub = df_raw[df_raw['unit'].isin(units)].copy()
    # PER-CLIENT NORMALIZATION
    min_vals = df_sub[USEFUL_SENSORS].min()
    max_vals = df_sub[USEFUL_SENSORS].max()
    df_sub[USEFUL_SENSORS] = (df_sub[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
    healthy = []
    for u, group in df_sub.groupby('unit'):
        healthy.extend(extract_sequences(group, is_healthy=True))
    
    return np.array(healthy, dtype=np.float32)

def client_fn(context: fl.common.Context) -> fl.client.Client:
    cid = int(context.node_config["partition-id"])
    units = client_units[cid]
    
    X_healthy = build_client_data(units)
    
    train_size = int(len(X_healthy) * 0.8)
    train_data = X_healthy[:train_size]
    val_data = X_healthy[train_size:]
    
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(val_data)), batch_size=64)
    
    return CmapssClient(CmapssAutoencoder(), train_loader, val_loader, mu=0.1).to_client()

def get_evaluate_fn():
    df_sub = df_raw[df_raw['unit'].isin(global_test_units)].copy()
    min_vals = df_sub[USEFUL_SENSORS].min()
    max_vals = df_sub[USEFUL_SENSORS].max()
    df_sub[USEFUL_SENSORS] = (df_sub[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)
    
    healthy = []
    anomalous = []
    for u, group in df_sub.groupby('unit'):
        healthy.extend(extract_sequences(group, is_healthy=True))
        anomalous.extend(extract_sequences(group, is_healthy=False))
        
    X_test = np.array(healthy + anomalous, dtype=np.float32)
    y_test = np.array([0]*len(healthy) + [1]*len(anomalous))
    
    def evaluate(server_round, parameters, config):
        model = CmapssAutoencoder()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        with torch.no_grad():
            x_tensor = torch.tensor(X_test)
            out = model(x_tensor)
            mse_per_seq = torch.mean((out - x_tensor)**2, dim=(1,2)).numpy()
            
            # ROC AUC computation
            auc = roc_auc_score(y_test, mse_per_seq)
            
        return float(np.mean(mse_per_seq)), {"roc_auc": float(auc)}
    
    return evaluate

if __name__ == "__main__":
    strategy = fl.server.strategy.FedProx(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        proximal_mu=0.1,
        evaluate_fn=get_evaluate_fn()
    )
    
    print("--- FL PdM CMAPSS (FIXED) ---")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
