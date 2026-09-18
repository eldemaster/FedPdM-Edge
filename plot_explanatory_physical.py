import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from common import load_cmapss_raw, CmapssAutoencoder, USEFUL_SENSORS, SEQ_LEN

# Load data and global test holdout (last 10 units for seed 42)
df_raw = load_cmapss_raw()
unique_units = df_raw['unit'].unique()
np.random.seed(42)
np.random.shuffle(unique_units)
global_test_units = unique_units[-10:]

# Pick the very last unit for plotting
test_unit = global_test_units[-1]
df_sub = df_raw[df_raw['unit'].isin(global_test_units)].copy()

healthy_stats = df_sub.groupby('unit').head(50)
min_vals = healthy_stats[USEFUL_SENSORS].min()
max_vals = healthy_stats[USEFUL_SENSORS].max()
df_sub[USEFUL_SENSORS] = (df_sub[USEFUL_SENSORS] - min_vals) / (max_vals - min_vals + 1e-7)

unit_data = df_sub[df_sub['unit'] == test_unit][USEFUL_SENSORS].values
cycles = df_sub[df_sub['unit'] == test_unit]['cycle'].values
max_cycle = len(unit_data)

# Load Federated Model
model = CmapssAutoencoder()
model.load_state_dict(torch.load("global_model.pt", weights_only=True))
model.eval()

mse_scores = []
reconstructions = []
for i in range(max_cycle - SEQ_LEN):
    seq = torch.tensor(unit_data[i:i+SEQ_LEN], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(seq)
    mse = torch.mean((out - seq)**2).item()
    mse_scores.append(mse)
    reconstructions.append(out.squeeze().numpy())

fig, axs = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 2, 1.5, 1.5]})
sensor_idx = 0  # Plot the first useful sensor (s2)

# Panel 1: Raw Sensor
axs[0].plot(cycles, unit_data[:, sensor_idx], color='gray', label='Sensor s2 (Normalized)')
axs[0].set_title(f'Engine {test_unit}: Raw Sensor Drift over Lifetime')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Panel 2: MSE Anomaly Score
axs[1].plot(cycles[SEQ_LEN:], mse_scores, color='red', linewidth=2, label='Autoencoder Reconstruction Error')
axs[1].axhline(y=np.percentile(mse_scores[:50], 95) * 2, color='black', linestyle='--', label='Dynamic Alert Threshold')
axs[1].set_title('Global Federated Model: Anomaly Score')
axs[1].legend()
axs[1].grid(True, alpha=0.3)

# Panel 3: Healthy zoom
h_idx = 10
axs[2].plot(range(SEQ_LEN), unit_data[h_idx:h_idx+SEQ_LEN, sensor_idx], 'b-', label='Actual Signal (Healthy)')
axs[2].plot(range(SEQ_LEN), reconstructions[h_idx][:, sensor_idx], 'r--', label='Model Reconstruction')
axs[2].set_title(f'Healthy State Zoom (Cycle {h_idx} to {h_idx+SEQ_LEN})')
axs[2].legend()
axs[2].grid(True, alpha=0.3)

# Panel 4: Faulty zoom
f_idx = max_cycle - SEQ_LEN - 10
axs[3].plot(range(SEQ_LEN), unit_data[f_idx:f_idx+SEQ_LEN, sensor_idx], 'b-', label='Actual Signal (Failing)')
axs[3].plot(range(SEQ_LEN), reconstructions[f_idx][:, sensor_idx], 'r--', label='Model Reconstruction')
axs[3].set_title(f'Failing State Zoom (Cycle {f_idx} to {f_idx+SEQ_LEN})')
axs[3].legend()
axs[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('explanatory_physical.png', dpi=300)
