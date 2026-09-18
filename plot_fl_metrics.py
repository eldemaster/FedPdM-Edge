import matplotlib.pyplot as plt

# Hardcoding the extracted metrics from the log for quick plotting, 
# since the history object is lost from the previous memory space.

rounds = [1, 2, 3, 4, 5]

# From History (loss, distributed)
loss_distributed = [0.045165, 0.014258, 0.013311, 0.013015, 0.013181]

# From History (loss, centralized)
loss_centralized = [0.073340, 0.040173, 0.039694, 0.039368, 0.037924]

# From History (metrics, centralized) - roc_auc starts at round 0 but we plot 1-5
roc_auc = [1.0, 1.0, 1.0, 1.0, 1.0]

plt.figure(figsize=(12, 5))

# Plot 1: Losses
plt.subplot(1, 2, 1)
plt.plot(rounds, loss_distributed, marker='o', label='Distributed Loss (Client Avg)')
plt.plot(rounds, loss_centralized, marker='s', label='Centralized Loss (Global Holdout)')
plt.title('Federated Autoencoder Loss over Rounds')
plt.xlabel('FL Round')
plt.ylabel('MSE Reconstruction Loss')
plt.xticks(rounds)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plot 2: ROC-AUC
plt.subplot(1, 2, 2)
plt.plot(rounds, roc_auc, marker='^', color='green', label='ROC-AUC (Anomaly Detection)')
plt.title('Global Anomaly Detection Performance')
plt.xlabel('FL Round')
plt.ylabel('Area Under Curve (AUC)')
plt.xticks(rounds)
plt.ylim(0.8, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('/home/eldemaster/pdm_metrics.png', dpi=300)
print("Plot saved to pdm_metrics.png")
