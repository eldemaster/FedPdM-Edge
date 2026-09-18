import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DATA_FILE = "/home/eldemaster/CMAPSSData/train_FD001.txt"
SEQ_LEN = 30
USEFUL_SENSORS = ['s2', 's3', 's4', 's6', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
INPUT_DIM = len(USEFUL_SENSORS)

# Let's say s14 is a critical sensor to visualize (index 10 in USEFUL_SENSORS)
SENSOR_IDX = 10 

cols = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
df = pd.read_csv(DATA_FILE, sep=r'\s+', header=None, names=cols)

min_vals = df[USEFUL_SENSORS].min()
max_vals = df[USEFUL_SENSORS].max()
df[USEFUL_SENSORS] = (df[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)

# We need the trained model. Let's just do a fast 3 epoch train again so it runs standalone
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
for u, group in df[df['unit'].isin([2,3,4,5,6])].groupby('unit'):
    data = group[USEFUL_SENSORS].values
    for i in range(min(50, len(data) - SEQ_LEN)): 
        healthy_seqs.append(data[i:i+SEQ_LEN])
X_train = torch.tensor(np.array(healthy_seqs, dtype=np.float32))

model = CmapssAutoencoder()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
model.train()
for epoch in range(25): # short fast training
    opt.zero_grad()
    loss = nn.MSELoss()(model(X_train), X_train)
    loss.backward()
    opt.step()

model.eval()

# Analyze Unit 1
unit1_data = df[df['unit'] == 1][USEFUL_SENSORS].values
raw_s14 = unit1_data[:, SENSOR_IDX]

mse_scores = []
reconstructions = []
cycles = []

with torch.no_grad():
    for i in range(len(unit1_data) - SEQ_LEN):
        seq = torch.tensor(unit1_data[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0)
        out = model(seq)
        mse = torch.mean((out - seq)**2).item()
        mse_scores.append(mse)
        reconstructions.append(out.squeeze().numpy())
        cycles.append(i + SEQ_LEN)

fig = plt.figure(figsize=(15, 12))

# Subplot 1: Raw Sensor Data Drift
plt.subplot(3, 1, 1)
plt.plot(range(len(raw_s14)), raw_s14, color='royalblue', alpha=0.7, label='Normalized Sensor 14 (Core Speed/Vibration)')
plt.axvline(x=len(raw_s14)-50, color='red', linestyle='--', label='Failure Window Starts')
plt.title('1. Raw Telemetry: Sensor Drift over Engine Lifecycle')
plt.ylabel('Sensor Amplitude')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 2: Autoencoder MSE
plt.subplot(3, 1, 2)
plt.plot(cycles, mse_scores, color='darkred', linewidth=2, label='Autoencoder Reconstruction Error (MSE)')
plt.axhline(y=0.015, color='orange', linestyle='-', linewidth=2, label='Anomaly Alert Threshold')
plt.axvline(x=len(raw_s14)-50, color='red', linestyle='--')
plt.title('2. Anomaly Score: How the AI flags the impending failure')
plt.ylabel('MSE Error')
plt.grid(True, alpha=0.3)
plt.legend()

# Subplot 3 & 4: Why MSE goes up (Reconstruction vs Original)
# Healthy Window (cycle 20 to 50)
healthy_orig = unit1_data[20:20+SEQ_LEN, SENSOR_IDX]
healthy_recon = reconstructions[20][:, SENSOR_IDX]

plt.subplot(3, 2, 5)
plt.plot(healthy_orig, color='blue', label='Actual Sensor')
plt.plot(healthy_recon, color='green', linestyle='--', label='AI Reconstruction')
plt.title('3. Healthy State (Cycle 20-50)')
plt.xlabel('Time Step (in window)')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.legend()

# Faulty Window (last 30 cycles before failure)
faulty_idx = len(unit1_data) - SEQ_LEN - 1
faulty_orig = unit1_data[faulty_idx:faulty_idx+SEQ_LEN, SENSOR_IDX]
faulty_recon = reconstructions[faulty_idx][:, SENSOR_IDX]

plt.subplot(3, 2, 6)
plt.plot(faulty_orig, color='blue', label='Actual Sensor')
plt.plot(faulty_recon, color='red', linestyle='--', label='AI Reconstruction')
plt.title(f'4. Failing State (Cycle {faulty_idx}-{faulty_idx+30})')
plt.xlabel('Time Step (in window)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('/home/eldemaster/explanatory_pdm.png', dpi=300)
print("Saved explanatory plot.")
