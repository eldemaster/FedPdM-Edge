import json
import matplotlib.pyplot as plt

with open("results_physical.json", "r") as f:
    data = json.load(f)

rounds = list(range(len(data['losses_centralized'])))
model_rmse = [x[1] for x in data['metrics_centralized']['model_rmse']]
baseline_rmse = [x[1] for x in data['metrics_centralized']['baseline_rmse']]

plt.figure(figsize=(10, 6))
plt.plot(rounds, model_rmse, 's-', color='blue', label='Federated RUL Model (RMSE)', linewidth=2, markersize=8)
plt.plot(rounds, baseline_rmse, '--', color='red', label='Baseline (Mean RUL)', linewidth=2)

plt.title('Federated Learning (Supervised) - RUL Prediction Error over Rounds')
plt.xlabel('Communication Round')
plt.ylabel('RMSE (Cycles)')
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('rul_metrics.png', dpi=300)
