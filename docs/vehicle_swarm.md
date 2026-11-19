# Non-IID vehicle-swarm federated learning benchmark

## Scenario and privacy boundary

This benchmark models a manufacturer improving one remaining-useful-life (RUL) model across customer fleets whose vehicles never move between customers. A client is a customer fleet, not a telemetry shard: every C-MAPSS unit ID is assigned to exactly one of `K=10` or `K=20` clients, train/validation splitting is then performed by whole vehicle, and an assertion rejects any duplicate or missing unit.

The manufacturer does not receive raw sensor rows for normalization. Each customer calls `compute_local_stats` on its local `USEFUL_SENSORS` rows and sends only a `LocalStats` object; `aggregate_federated_stats` constructs a `FederatedScaler` from those sufficient statistics. The audit field `raw_sensor_rows_shared` is consequently zero in every seed run. Model updates are sample-count-weighted. In each round a seeded sampler selects exactly `round(K * fraction_fit)` clients: 5 of 10 for `fraction_fit=0.5`, and 6 of 20 for `fraction_fit=0.3`.

The implementation also supports the combined 709-unit FD001--FD004 training fleet. It prefixes IDs as `FD00x:<unit>` before partitioning so unit numbers reused by NASA cannot collide.

## Partitions and heterogeneity

- `iid_swarm` randomly shuffles whole vehicles and splits them uniformly.
- `lifespan_quantity_skew` draws client fleet sizes from Dirichlet `alpha=0.5`, orders vehicles by observed training lifespan, and assigns contiguous lifespan bands. This represents customers with systematically short- or long-lived duty profiles.
- `regime_fault_skew` fits six operating-setting clusters from `setting1..3`. FD004 additionally fits a label-free, two-cluster K-Means model to end-of-life standardized drift using the local diagnostics baseline. Each `(dominant regime, drift cluster)` group draws a client allocation from Dirichlet `alpha=0.3`. The drift cluster is deliberately not called a component fault label.

The table reports the five-seed mean. `count CV/Gini` uses supervised-window counts, `life JS` is mean pairwise Jensen-Shannon divergence between client lifespan histograms, `life span` is the range of client mean lifespans in cycles, and `regime JS` / `drift JS` compare categorical profiles. Larger values mean more inter-client skew.

| Dataset | K | Partition | Count CV | Gini | Life JS | Life span | Regime JS | Drift JS |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| FD002 | 10 | IID | 0.055 | 0.030 | 0.080 | 31.3 | 0.011 | 0.000 |
| FD002 | 10 | Lifespan/quantity | 1.147 | 0.556 | 0.770 | 170.4 | 0.024 | 0.000 |
| FD002 | 10 | Regime/drift | 1.141 | 0.561 | 0.327 | 55.5 | 0.056 | 0.000 |
| FD004 | 10 | IID | 0.064 | 0.035 | 0.089 | 45.5 | 0.004 | 0.014 |
| FD004 | 10 | Lifespan/quantity | 1.168 | 0.567 | 0.776 | 256.9 | 0.002 | 0.200 |
| FD004 | 10 | Regime/drift | 0.980 | 0.488 | 0.280 | 120.4 | 0.007 | 0.388 |
| FD004 | 20 | IID | 0.099 | 0.056 | 0.175 | 72.6 | 0.004 | 0.041 |
| FD004 | 20 | Lifespan/quantity | 1.193 | 0.537 | 0.786 | 299.6 | 0.001 | 0.215 |
| FD004 | 20 | Regime/drift | 1.058 | 0.521 | 0.395 | 172.1 | 0.006 | 0.372 |

Both intended Non-IID partitions have a strictly larger combined heterogeneity score than IID in every configuration. FD002 has only one published dataset-level fault mode, so no two-cluster drift feature is used there. On FD004 almost every vehicle has the same most-frequent operating regime (248 of 249 select the same cluster); therefore its `regime_fault_skew` signal comes primarily from the unsupervised drift cluster. The benchmark reports this limitation rather than manufacturing per-vehicle component labels.

## Five-seed benchmark

All entries are mean ± population standard deviation over seeds `7, 42, 100, 777, 2024`. Each cell is `global-test RMSE / NASA score per test vehicle / late-prediction rate`. The JSON also retains total NASA score, client-matched metrics, worst-client and 90th-percentile RMSE, every round trajectory, exact round participants, and every vehicle partition.

The bounded protocol uses five rounds, one local epoch per selected client, at most 512 deterministically sampled training windows and 128 validation windows per client, Adam at `1e-3`, FedProx `mu=0.05`, and two personalization epochs at one-quarter learning rate. Local-only models receive five epochs. A local-only global result is a training-sample-weighted ensemble of the customer models; its client metrics use each customer's own held-out vehicles.

### FD002, K=10, 50% participation

| Partition | Method | RMSE | NASA / vehicle | Late rate |
|---|---|---:|---:|---:|
| IID | Local only | 47.30 ± 2.54 | 1076.8 ± 656.2 | 0.59 ± 0.03 |
| IID | FedAvg MSE | 43.73 ± 1.00 | 291.1 ± 169.2 | 0.50 ± 0.05 |
| IID | FedProx MSE | 43.73 ± 1.00 | 291.1 ± 169.2 | 0.50 ± 0.05 |
| IID | FedAvg asymmetric | 50.09 ± 1.46 | 129.9 ± 22.6 | 0.33 ± 0.02 |
| IID | FedProx asymmetric | 50.09 ± 1.45 | 129.9 ± 22.5 | 0.33 ± 0.02 |
| IID | Personalized asymmetric | 48.53 ± 0.19 | 105.7 ± 2.0 | 0.34 ± 0.00 |
| Lifespan/quantity | Local only | 44.47 ± 1.36 | 404.3 ± 190.3 | 0.53 ± 0.05 |
| Lifespan/quantity | FedAvg MSE | 43.92 ± 1.38 | 307.3 ± 238.8 | 0.49 ± 0.06 |
| Lifespan/quantity | FedProx MSE | 43.92 ± 1.38 | 307.0 ± 238.2 | 0.49 ± 0.06 |
| Lifespan/quantity | FedAvg asymmetric | 49.63 ± 2.01 | 125.3 ± 26.9 | 0.34 ± 0.03 |
| Lifespan/quantity | FedProx asymmetric | 49.63 ± 2.01 | 125.3 ± 26.9 | 0.34 ± 0.03 |
| Lifespan/quantity | Personalized asymmetric | 49.64 ± 1.34 | 122.6 ± 20.4 | 0.33 ± 0.01 |
| Regime/drift | Local only | 44.11 ± 1.28 | 343.9 ± 222.9 | 0.51 ± 0.04 |
| Regime/drift | FedAvg MSE | 44.06 ± 0.66 | 282.5 ± 183.2 | 0.49 ± 0.06 |
| Regime/drift | FedProx MSE | 44.06 ± 0.66 | 282.1 ± 183.2 | 0.49 ± 0.06 |
| Regime/drift | FedAvg asymmetric | 49.91 ± 2.06 | 129.4 ± 31.3 | 0.33 ± 0.03 |
| Regime/drift | FedProx asymmetric | 49.91 ± 2.06 | 129.3 ± 31.3 | 0.33 ± 0.03 |
| Regime/drift | Personalized asymmetric | 48.63 ± 0.55 | 107.1 ± 7.3 | 0.34 ± 0.01 |

### FD004, K=10, 50% participation

| Partition | Method | RMSE | NASA / vehicle | Late rate |
|---|---|---:|---:|---:|
| IID | Local only | 51.81 ± 2.39 | 3279.6 ± 1548.7 | 0.60 ± 0.02 |
| IID | FedAvg MSE | 46.09 ± 1.58 | 714.9 ± 499.0 | 0.51 ± 0.03 |
| IID | FedProx MSE | 46.09 ± 1.57 | 714.9 ± 497.7 | 0.51 ± 0.03 |
| IID | FedAvg asymmetric | 49.50 ± 2.21 | 112.9 ± 22.1 | 0.34 ± 0.01 |
| IID | FedProx asymmetric | 49.50 ± 2.21 | 113.0 ± 22.1 | 0.34 ± 0.01 |
| IID | Personalized asymmetric | 48.41 ± 0.59 | 96.9 ± 5.0 | 0.35 ± 0.00 |
| Lifespan/quantity | Local only | 46.61 ± 1.78 | 841.0 ± 458.7 | 0.53 ± 0.04 |
| Lifespan/quantity | FedAvg MSE | 46.49 ± 2.34 | 833.7 ± 824.7 | 0.50 ± 0.06 |
| Lifespan/quantity | FedProx MSE | 46.49 ± 2.35 | 834.1 ± 825.1 | 0.50 ± 0.06 |
| Lifespan/quantity | FedAvg asymmetric | 50.44 ± 2.69 | 127.4 ± 27.2 | 0.34 ± 0.02 |
| Lifespan/quantity | FedProx asymmetric | 50.44 ± 2.69 | 127.4 ± 27.2 | 0.34 ± 0.02 |
| Lifespan/quantity | Personalized asymmetric | 49.97 ± 2.05 | 116.5 ± 24.2 | 0.34 ± 0.01 |
| Regime/drift | Local only | 48.93 ± 2.66 | 1769.3 ± 1112.9 | 0.57 ± 0.03 |
| Regime/drift | FedAvg MSE | 46.55 ± 1.23 | 798.7 ± 360.0 | 0.52 ± 0.03 |
| Regime/drift | FedProx MSE | 46.55 ± 1.23 | 798.1 ± 359.5 | 0.52 ± 0.03 |
| Regime/drift | FedAvg asymmetric | 48.81 ± 1.15 | 102.4 ± 10.5 | 0.35 ± 0.01 |
| Regime/drift | FedProx asymmetric | 48.81 ± 1.15 | 102.4 ± 10.5 | 0.35 ± 0.01 |
| Regime/drift | Personalized asymmetric | 48.79 ± 0.59 | 100.3 ± 7.0 | 0.35 ± 0.00 |

### FD004, K=20, 30% participation

| Partition | Method | RMSE | NASA / vehicle | Late rate |
|---|---|---:|---:|---:|
| IID | Local only | 51.47 ± 2.22 | 3025.5 ± 1289.9 | 0.59 ± 0.02 |
| IID | FedAvg MSE | 45.37 ± 1.05 | 497.0 ± 274.5 | 0.50 ± 0.02 |
| IID | FedProx MSE | 45.37 ± 1.05 | 497.1 ± 274.6 | 0.50 ± 0.02 |
| IID | FedAvg asymmetric | 48.86 ± 1.83 | 108.6 ± 7.5 | 0.35 ± 0.02 |
| IID | FedProx asymmetric | 48.86 ± 1.83 | 108.6 ± 7.5 | 0.35 ± 0.02 |
| IID | Personalized asymmetric | 48.58 ± 0.28 | 98.5 ± 3.7 | 0.35 ± 0.00 |
| Lifespan/quantity | Local only | 46.03 ± 1.44 | 674.1 ± 341.6 | 0.52 ± 0.03 |
| Lifespan/quantity | FedAvg MSE | 45.66 ± 1.15 | 428.2 ± 330.5 | 0.47 ± 0.06 |
| Lifespan/quantity | FedProx MSE | 45.66 ± 1.15 | 428.3 ± 330.6 | 0.47 ± 0.06 |
| Lifespan/quantity | FedAvg asymmetric | 51.30 ± 3.59 | 143.7 ± 58.3 | 0.33 ± 0.02 |
| Lifespan/quantity | FedProx asymmetric | 51.30 ± 3.59 | 143.7 ± 58.3 | 0.33 ± 0.02 |
| Lifespan/quantity | Personalized asymmetric | 49.61 ± 0.66 | 108.5 ± 9.9 | 0.35 ± 0.00 |
| Regime/drift | Local only | 46.80 ± 1.73 | 924.4 ± 537.6 | 0.53 ± 0.03 |
| Regime/drift | FedAvg MSE | 45.75 ± 1.32 | 362.8 ± 251.5 | 0.46 ± 0.06 |
| Regime/drift | FedProx MSE | 45.75 ± 1.32 | 362.6 ± 251.3 | 0.46 ± 0.06 |
| Regime/drift | FedAvg asymmetric | 49.68 ± 1.12 | 110.3 ± 10.9 | 0.34 ± 0.01 |
| Regime/drift | FedProx asymmetric | 49.68 ± 1.12 | 110.3 ± 10.9 | 0.34 ± 0.01 |
| Regime/drift | Personalized asymmetric | 48.47 ± 0.49 | 97.3 ± 5.0 | 0.35 ± 0.00 |

## What helped, and what did not

`FedAvg MSE` is the better choice when symmetric RMSE is the primary objective. Against the asymmetric variants it is generally 2.3--6.4 RMSE cycles better, and on IID data it also clearly improves over isolated local training. Under the skewed partitions, local-only global ensembles sometimes come within 0.1--0.6 RMSE cycles of FedAvg, but their worst-client validation errors remain much larger (roughly 69--96 cycles).

`AsymmetricRULLoss(alpha_late=2.5, alpha_early=0.5)` helps when late maintenance is the costly failure mode. Across the nine configurations, FedAvg asymmetric reduces late-prediction rate from the MSE model's 0.46--0.52 range to 0.33--0.35 and reduces NASA score per vehicle from 282--834 to 102--144. That safety improvement comes with higher symmetric RMSE and sometimes higher worst-client RMSE.

`FedProx(mu=0.05)` does not materially help in this run. FedProx and FedAvg differ by less than displayed RMSE precision in every configuration, and NASA changes are negligible. Five rounds and one local epoch do not create enough local drift for this proximal strength to matter; a longer local-training sweep would be needed before concluding that FedProx is ineffective generally.

Customer-side personalization improves matched-client mean validation RMSE in eight of nine configurations: the mean change ranges from -2.20 cycles to a neutral +0.002 cycles. It also generally improves global-ensemble NASA risk over unpersonalized asymmetric FedAvg. It is not uniformly fair: worst-client RMSE worsens in a few settings, so deployment should gate personalization per customer rather than assume every fleet benefits.

Round histories show most best global checkpoints at rounds 4--5. Mean first-to-final RMSE improvement is about 38.8--41.6 cycles for MSE and 33.2--38.3 for asymmetric optimization. This indicates that the five-round horizon is useful but not a convergence proof.

## Caveats and reproduction

- C-MAPSS units are simulated turbofan engines, used here as stand-ins for vehicles. Results do not establish road-vehicle validity, fleet connectivity behavior, or production maintenance economics.
- FD004 documents two dataset-level degradation modes but publishes no per-unit component label. The two end-of-life groups in this benchmark are unsupervised drift clusters and must not be interpreted as ground-truth fan/HPC diagnoses.
- The 512-window cap equalizes compute and keeps the 45-case experiment reproducible; it is not a full-data accuracy claim. Sample selection is seeded and the complete partitions and round selections are retained in the JSON.
- Only training-fleet raw rows contribute to normalization. Official C-MAPSS test vehicles are transformed after federated statistics have been aggregated.
- NASA score grows exponentially and is reported both as the canonical total and per test vehicle. The tables use the latter so FD002 and FD004 are easier to compare.

Run the full experiment with:

```bash
CMAPSS_ZIP=/path/to/CMAPSSData.zip python3 experiments/vehicle_swarm_benchmark.py
```

The machine-readable output is `results/vehicle_swarm_benchmark.json`.
