import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

DATA_FILE = "/home/eldemaster/CMAPSSData/train_FD001.txt"
SEQ_LEN = 30
USEFUL_SENSORS = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
INPUT_DIM = len(USEFUL_SENSORS)

cols = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
df = pd.read_csv(DATA_FILE, sep=r'\s+', header=None, names=cols)

# Normalize
min_vals = df[USEFUL_SENSORS].min()
max_vals = df[USEFUL_SENSORS].max()
df[USEFUL_SENSORS] = (df[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)

# Train a quick Autoencoder on Unit 2-100 (healthy)
class CmapssAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        flat_dim = INPUT_DIM * SEQ_LEN
        self.encoder = nn.Sequential(nn.Linear(flat_dim, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, flat_dim))
    def forward(self, x):
        b = x.size(0)
        return self.decoder(self.encoder(x.reshape(b, -1))).reshape(b, SEQ_LEN, INPUT_DIM)

healthy_seqs = []
for u, group in df[df['unit'] != 1].groupby('unit'):
    data = group[USEFUL_SENSORS].values
    for i in range(min(50, len(data) - SEQ_LEN)): # only healthy
        healthy_seqs.append(data[i:i+SEQ_LEN])

X_train = torch.tensor(np.array(healthy_seqs, dtype=np.float32))
loader = DataLoader(TensorDataset(X_train), batch_size=64, shuffle=True)

model = CmapssAutoencoder()
opt = optim.Adam(model.parameters(), lr=0.001)
crit = nn.MSELoss()

model.train()
for epoch in range(5):
    for batch in loader:
        opt.zero_grad()
        loss = crit(model(batch[0]), batch[0])
        loss.backward()
        opt.step()

# Evaluate on Unit 1 (Entire Lifecycle)
unit1_data = df[df['unit'] == 1][USEFUL_SENSORS].values
mse_scores = []
cycles = []

model.eval()
with torch.no_grad():
    for i in range(len(unit1_data) - SEQ_LEN):
        seq = torch.tensor(unit1_data[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0)
        out = model(seq)
        mse = torch.mean((out - seq)**2).item()
        mse_scores.append(mse)
        cycles.append(i + SEQ_LEN)

plt.figure(figsize=(10, 5))
plt.plot(cycles, mse_scores, color='red', linewidth=2, label='Reconstruction Error (MSE)')
plt.axvline(x=len(unit1_data)-50, color='black', linestyle='--', label='Actual Failure Window (-50 cycles)')
plt.title('Autoencoder Anomaly Detection: Engine Unit 1 Degradation')
plt.xlabel('Engine Cycle')
plt.ylabel('Reconstruction MSE')
plt.grid(True, alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('/home/eldemaster/engine_degradation.png', dpi=300)
print("Saved engine_degradation.png")
