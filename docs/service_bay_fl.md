# Service-Bay Intermittent-Connectivity Federated Learning

## Architecture

Vehicles retain their telemetry, complete accumulated local training history, model checkpoint, and SCAFFOLD control variate while offline. A service visit is the only synchronization event. On check-in, the edge node reports its last synchronized cycle and trains from that saved model; the coordinator computes `tau = current_cycle - last_synced_round`, applies the submitted model delta, and returns the latest global model. A vehicle that did not visit does not participate in that cycle. Local training uses the full vehicle-disjoint fleet history; no 512-window cap is applied.

Normalization follows the repository's decentralized path: each customer computes `fed_norm.LocalStats` locally and only those sufficient statistics reach `aggregate_federated_stats`. Checkpoint files contain model tensors, the vehicle's control variate, the last sync cycle, and optional LocalStats. They do not contain sensor rows. Benchmark records assert `raw_sensor_rows_uploaded: 0`.

## Why these algorithms fit service visits

- **FedAvgM with staleness weighting:** a vehicle may return with a several-cycle-old model after multiple local epochs. Server momentum (`beta=0.9`) smooths irregular updates; `n_i * (1 + tau_i)^(-0.5)` reduces the influence of old updates without requiring an always-on client connection.
- **SCAFFOLD:** persistent client control variates compensate for local gradient drift. This is useful when a service visit can afford three or more epochs over customer-specific operating histories.
- **Service-bay personalization:** after the global model is returned, each visiting customer fine-tunes a private fully connected head on its own accumulated history. The shared convolutional backbone and coordinator weights remain unchanged by that private step.
- **Local cosine schedule and budget:** the local learning rate decays across each multi-epoch visit. The checkpoint records consumed training-window budget, while training continues to use the entire retained history.

This protocol models sparse offline-first vehicles rather than synchronous FL, where clients are assumed to be reachable each round. Here, participation is 30% per cycle for K=10 and 20% for FD004 K=20, over 20 cycles. The runner reports the actual visit intervals and staleness for each update. A stale vehicle starts from its most recently synchronized model, submits its trained state, and receives the current global state after aggregation.

## Benchmark protocol and results

The benchmark evaluates seeds 7, 42, 100, 777, and 2024 on FD002 K=10, FD004 K=10, and FD004 K=20. Each configuration is repeated with `iid_swarm`, `lifespan_quantity_skew`, and `regime_fault_skew` vehicle-disjoint partitions. Every customer reserves vehicles for local validation; these held-out vehicles never supply training windows. The official engine test set provides the global evaluation.

Methods are `centralized_oracle`, `local_only`, `fedavg_intermittent`, `fedavgm_staleness`, `scaffold_service_bay`, `fedavgm_personalized`, and `fedavgm_asym_personalized`. Metrics and per-seed round curves are recorded in [`results/service_bay_benchmark.json`](../results/service_bay_benchmark.json). The personalized methods include both official-test evaluation and held-out per-customer validation. JSON aggregates use population mean and standard deviation across seeds.

### Aggregate table

This table is generated from the JSON artifact so the reported values match it exactly. Each row is mean ± standard deviation across the five seeds, grouped by dataset, K, and partition scheme.

| Dataset | K | Partition Scheme | Method | Official-test RMSE | NASA Score / Sample | Late Prediction Rate | Average Staleness $\tau$ |
|---|---:|---|---|---:|---:|---:|---:|
| FD002 | 10 | iid_swarm | centralized_oracle | 27.09 ± 2.92 | 40.35 ± 17.21 | 0.51 ± 0.02 | 0.00 ± 0.00 |
| FD002 | 10 | iid_swarm | local_only | 44.13 ± 0.46 | 322.19 ± 123.88 | 0.52 ± 0.05 | 0.00 ± 0.00 |
| FD002 | 10 | iid_swarm | fedavg_intermittent | 45.83 ± 2.07 | 620.21 ± 491.97 | 0.53 ± 0.09 | 2.89 ± 0.12 |
| FD002 | 10 | iid_swarm | fedavgm_staleness | 56.72 ± 9.83 | 7341.48 ± 8803.59 | 0.52 ± 0.22 | 2.89 ± 0.12 |
| FD002 | 10 | iid_swarm | scaffold_service_bay | 85.01 ± 0.07 | 4288.36 ± 26.85 | 0.00 ± 0.00 | 2.89 ± 0.12 |
| FD002 | 10 | iid_swarm | fedavgm_personalized (ensemble)* | 62.09 ± 9.62 | 15960.99 ± 7931.71 | 0.71 ± 0.14 | 2.89 ± 0.12 |
| FD002 | 10 | iid_swarm | fedavgm_asym_personalized (ensemble)* | 54.60 ± 10.00 | 8163.22 ± 9301.03 | 0.62 ± 0.15 | 2.89 ± 0.12 |
| FD002 | 10 | lifespan_quantity_skew | centralized_oracle | 29.93 ± 2.75 | 54.91 ± 19.24 | 0.51 ± 0.02 | 0.00 ± 0.00 |
| FD002 | 10 | lifespan_quantity_skew | local_only | 43.26 ± 0.56 | 180.55 ± 57.32 | 0.47 ± 0.03 | 0.00 ± 0.00 |
| FD002 | 10 | lifespan_quantity_skew | fedavg_intermittent | 44.60 ± 1.09 | 228.84 ± 205.84 | 0.46 ± 0.07 | 2.89 ± 0.12 |
| FD002 | 10 | lifespan_quantity_skew | fedavgm_staleness | 84.80 ± 0.59 | 4218.45 ± 216.35 | 0.00 ± 0.00 | 2.89 ± 0.12 |
| FD002 | 10 | lifespan_quantity_skew | scaffold_service_bay | 84.72 ± 0.09 | 4181.21 ± 34.19 | 0.00 ± 0.00 | 2.89 ± 0.12 |
| FD002 | 10 | lifespan_quantity_skew | fedavgm_personalized (ensemble)* | 67.13 ± 6.86 | 9373.82 ± 8665.80 | 0.51 ± 0.30 | 2.89 ± 0.12 |
| FD002 | 10 | lifespan_quantity_skew | fedavgm_asym_personalized (ensemble)* | 66.60 ± 12.19 | 8700.99 ± 8853.84 | 0.41 ± 0.32 | 2.89 ± 0.12 |
| FD002 | 10 | regime_fault_skew | centralized_oracle | 30.27 ± 4.21 | 60.65 ± 31.71 | 0.52 ± 0.02 | 0.00 ± 0.00 |
| FD002 | 10 | regime_fault_skew | local_only | 43.47 ± 0.25 | 222.50 ± 80.70 | 0.49 ± 0.04 | 0.00 ± 0.00 |
| FD002 | 10 | regime_fault_skew | fedavg_intermittent | 43.78 ± 0.70 | 204.14 ± 140.85 | 0.47 ± 0.06 | 2.89 ± 0.12 |
| FD002 | 10 | regime_fault_skew | fedavgm_staleness | 75.41 ± 16.46 | 3117.36 ± 1658.81 | 0.09 ± 0.18 | 2.89 ± 0.12 |
| FD002 | 10 | regime_fault_skew | scaffold_service_bay | 84.68 ± 0.17 | 4165.95 ± 60.86 | 0.00 ± 0.00 | 2.89 ± 0.12 |
| FD002 | 10 | regime_fault_skew | fedavgm_personalized (ensemble)* | 57.71 ± 11.18 | 8202.02 ± 9576.04 | 0.54 ± 0.22 | 2.89 ± 0.12 |
| FD002 | 10 | regime_fault_skew | fedavgm_asym_personalized (ensemble)* | 61.77 ± 13.20 | 5262.27 ± 7454.37 | 0.40 ± 0.28 | 2.89 ± 0.12 |
| FD004 | 10 | iid_swarm | centralized_oracle | 30.71 ± 1.03 | 80.46 ± 24.36 | 0.51 ± 0.01 | 0.00 ± 0.00 |
| FD004 | 10 | iid_swarm | local_only | 45.13 ± 0.49 | 474.83 ± 135.08 | 0.51 ± 0.02 | 0.00 ± 0.00 |
| FD004 | 10 | iid_swarm | fedavg_intermittent | 44.87 ± 0.92 | 206.83 ± 119.50 | 0.44 ± 0.05 | 2.89 ± 0.12 |
| FD004 | 10 | iid_swarm | fedavgm_staleness | 79.62 ± 13.40 | 3306.13 ± 1656.93 | 0.07 ± 0.13 | 2.89 ± 0.12 |
| FD004 | 10 | iid_swarm | scaffold_service_bay | 88.47 ± 0.09 | 4916.05 ± 39.37 | 0.00 ± 0.00 | 2.89 ± 0.12 |
| FD004 | 10 | iid_swarm | fedavgm_personalized (ensemble)* | 59.50 ± 10.64 | 4015.47 ± 6121.32 | 0.43 ± 0.22 | 2.89 ± 0.12 |
| FD004 | 10 | iid_swarm | fedavgm_asym_personalized (ensemble)* | 56.53 ± 15.05 | 680.27 ± 755.73 | 0.29 ± 0.14 | 2.89 ± 0.12 |
| FD004 | 10 | lifespan_quantity_skew | centralized_oracle | 30.88 ± 1.47 | 88.05 ± 40.53 | 0.50 ± 0.01 | 0.00 ± 0.00 |
| FD004 | 10 | lifespan_quantity_skew | local_only | 44.51 ± 0.26 | 292.18 ± 109.92 | 0.47 ± 0.05 | 0.00 ± 0.00 |
| FD004 | 10 | lifespan_quantity_skew | fedavg_intermittent | 45.01 ± 0.94 | 293.88 ± 244.28 | 0.44 ± 0.07 | 2.89 ± 0.12 |
| FD004 | 10 | lifespan_quantity_skew | fedavgm_staleness | 83.88 ± 10.04 | 7333.91 ± 4454.78 | 0.15 ± 0.29 | 2.89 ± 0.12 |
| FD004 | 10 | lifespan_quantity_skew | scaffold_service_bay | 88.29 ± 0.06 | 4840.15 ± 25.86 | 0.00 ± 0.00 | 2.89 ± 0.12 |
| FD004 | 10 | lifespan_quantity_skew | fedavgm_personalized (ensemble)* | 63.48 ± 12.63 | 4097.08 ± 6111.89 | 0.35 ± 0.24 | 2.89 ± 0.12 |
| FD004 | 10 | lifespan_quantity_skew | fedavgm_asym_personalized (ensemble)* | 69.49 ± 15.62 | 4988.71 ± 5835.92 | 0.27 ± 0.28 | 2.89 ± 0.12 |
| FD004 | 10 | regime_fault_skew | centralized_oracle | 30.69 ± 1.08 | 78.99 ± 24.79 | 0.50 ± 0.01 | 0.00 ± 0.00 |
| FD004 | 10 | regime_fault_skew | local_only | 45.05 ± 0.81 | 454.38 ± 195.32 | 0.50 ± 0.03 | 0.00 ± 0.00 |
| FD004 | 10 | regime_fault_skew | fedavg_intermittent | 49.75 ± 5.66 | 286.61 ± 219.48 | 0.39 ± 0.09 | 2.89 ± 0.12 |
| FD004 | 10 | regime_fault_skew | fedavgm_staleness | 85.88 ± 5.71 | 4304.34 ± 1459.06 | 0.03 ± 0.06 | 2.89 ± 0.12 |
| FD004 | 10 | regime_fault_skew | scaffold_service_bay | 88.40 ± 0.13 | 4885.77 ± 54.17 | 0.00 ± 0.00 | 2.89 ± 0.12 |
| FD004 | 10 | regime_fault_skew | fedavgm_personalized (ensemble)* | 60.84 ± 6.01 | 7512.36 ± 6612.94 | 0.52 ± 0.23 | 2.89 ± 0.12 |
| FD004 | 10 | regime_fault_skew | fedavgm_asym_personalized (ensemble)* | 56.93 ± 11.11 | 607.56 ± 435.30 | 0.32 ± 0.13 | 2.89 ± 0.12 |
| FD004 | 20 | iid_swarm | centralized_oracle | 30.89 ± 1.13 | 82.88 ± 32.31 | 0.51 ± 0.01 | 0.00 ± 0.00 |
| FD004 | 20 | iid_swarm | local_only | 46.68 ± 0.78 | 682.59 ± 361.03 | 0.50 ± 0.07 | 0.00 ± 0.00 |
| FD004 | 20 | iid_swarm | fedavg_intermittent | 49.18 ± 5.72 | 508.03 ± 468.15 | 0.43 ± 0.11 | 3.94 ± 0.09 |
| FD004 | 20 | iid_swarm | fedavgm_staleness | 63.47 ± 0.65 | 15640.17 ± 1206.44 | 0.73 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | iid_swarm | scaffold_service_bay | 88.66 ± 0.07 | 5000.01 ± 30.79 | 0.00 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | iid_swarm | fedavgm_personalized (ensemble)* | 63.79 ± 0.00 | 16243.40 ± 0.00 | 0.73 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | iid_swarm | fedavgm_asym_personalized (ensemble)* | 63.79 ± 0.00 | 16243.40 ± 0.00 | 0.73 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | lifespan_quantity_skew | centralized_oracle | 30.92 ± 1.28 | 87.33 ± 37.03 | 0.51 ± 0.01 | 0.00 ± 0.00 |
| FD004 | 20 | lifespan_quantity_skew | local_only | 45.10 ± 0.53 | 326.69 ± 129.03 | 0.47 ± 0.05 | 0.00 ± 0.00 |
| FD004 | 20 | lifespan_quantity_skew | fedavg_intermittent | 49.70 ± 6.95 | 3658.50 ± 5360.12 | 0.53 ± 0.11 | 3.94 ± 0.09 |
| FD004 | 20 | lifespan_quantity_skew | fedavgm_staleness | 68.82 ± 10.06 | 14018.97 ± 4448.84 | 0.58 ± 0.29 | 3.94 ± 0.09 |
| FD004 | 20 | lifespan_quantity_skew | scaffold_service_bay | 88.54 ± 0.14 | 4946.34 ± 60.57 | 0.00 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | lifespan_quantity_skew | fedavgm_personalized (ensemble)* | 63.79 ± 0.00 | 16243.40 ± 0.00 | 0.73 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | lifespan_quantity_skew | fedavgm_asym_personalized (ensemble)* | 63.80 ± 0.01 | 16243.40 ± 0.01 | 0.73 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | regime_fault_skew | centralized_oracle | 30.80 ± 1.03 | 87.02 ± 35.07 | 0.51 ± 0.00 | 0.00 ± 0.00 |
| FD004 | 20 | regime_fault_skew | local_only | 45.50 ± 0.50 | 447.36 ± 197.82 | 0.49 ± 0.05 | 0.00 ± 0.00 |
| FD004 | 20 | regime_fault_skew | fedavg_intermittent | 48.03 ± 2.47 | 1363.68 ± 1045.08 | 0.52 ± 0.08 | 3.94 ± 0.09 |
| FD004 | 20 | regime_fault_skew | fedavgm_staleness | 63.96 ± 7.71 | 10103.93 ± 7534.66 | 0.53 ± 0.26 | 3.94 ± 0.09 |
| FD004 | 20 | regime_fault_skew | scaffold_service_bay | 88.51 ± 0.14 | 4935.86 ± 61.66 | 0.00 ± 0.00 | 3.94 ± 0.09 |
| FD004 | 20 | regime_fault_skew | fedavgm_personalized (ensemble)* | 62.64 ± 2.30 | 14700.08 ± 3086.63 | 0.72 ± 0.02 | 3.94 ± 0.09 |
| FD004 | 20 | regime_fault_skew | fedavgm_asym_personalized (ensemble)* | 64.50 ± 1.40 | 13135.97 ± 6214.86 | 0.62 ± 0.21 | 3.94 ± 0.09 |

*\*Note: For personalized methods (`fedavgm_personalized`, `fedavgm_asym_personalized`), official-test metrics report the `personalized_global_test` ensemble evaluation over customer-specific models.*

### Centralized gap closure

For each configuration and method, gap closure is computed as `(local_only RMSE - federated RMSE) / (local_only RMSE - centralized_oracle RMSE)`. Values above 1 indicate the federated result beats the centralized reference on that evaluation; negative values indicate a regression against local-only. The JSON contains the seed-level inputs to this comparison. The values below are filled from the benchmark artifact at generation time.

| Dataset | K | Partition Scheme | Plain FedAvg | FedAvgM staleness | SCAFFOLD | FedAvg personalized | Asymmetric personalized |
|---|---:|---|---:|---:|---:|---:|---:|
| FD002 | 10 | iid_swarm | -9.95% ± 10.01% | -84.48% ± 77.65% | -250.44% ± 60.57% | -114.46% ± 68.29% | -72.90% ± 78.29% |
| FD002 | 10 | lifespan_quantity_skew | -9.88% ± 9.06% | -320.66% ± 54.45% | -319.45% ± 51.56% | -181.78% ± 49.18% | -178.90% ± 89.84% |
| FD002 | 10 | regime_fault_skew | -2.31% ± 4.49% | -286.83% ± 175.99% | -354.57% ± 142.97% | -134.54% ± 128.39% | -134.79% ± 69.26% |
| FD004 | 10 | iid_swarm | 1.16% ± 8.82% | -231.81% ± 79.12% | -304.71% ± 39.94% | -102.51% ± 75.66% | -75.08% ± 97.58% |
| FD004 | 10 | lifespan_quantity_skew | -3.13% ± 6.96% | -283.91% ± 57.56% | -324.40% ± 34.09% | -133.73% ± 90.15% | -175.50% ± 112.92% |
| FD004 | 10 | regime_fault_skew | -32.54% ± 38.76% | -291.39% ± 67.92% | -306.77% ± 44.01% | -109.69% ± 42.14% | -89.99% ± 89.51% |
| FD004 | 20 | iid_swarm | -19.10% ± 41.41% | -107.57% ± 14.05% | -268.49% ± 27.68% | -109.63% ± 13.53% | -109.63% ± 13.53% |
| FD004 | 20 | lifespan_quantity_skew | -31.50% ± 48.11% | -169.11% ± 78.17% | -307.84% ± 21.32% | -132.46% ± 9.21% | -132.47% ± 9.23% |
| FD004 | 20 | regime_fault_skew | -17.28% ± 19.75% | -128.24% ± 59.97% | -294.69% ± 25.67% | -117.23% ± 16.38% | -130.40% ± 18.16% |

### Customer-held-out validation

Per-customer held-out RMSE evaluated across the five per-seed runs (using `global.customer_validation.mean_rmse` for non-personalized methods, and `personalized_customer_validation.mean_rmse` for personalized methods):

| Dataset | K | Partition Scheme | Centralized Oracle | Local-only | FedAvg | FedAvgM staleness | SCAFFOLD | FedAvg personalized | Asymmetric personalized |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| FD002 | 10 | iid_swarm | 26.94 ± 2.39 | 43.37 ± 1.32 | 44.52 ± 1.27 | 56.00 ± 9.18 | 89.80 ± 1.58 | 57.25 ± 3.97 | 57.28 ± 2.75 |
| FD002 | 10 | lifespan_quantity_skew | 31.11 ± 2.70 | 40.18 ± 0.96 | 44.07 ± 0.87 | 84.34 ± 3.08 | 84.25 ± 2.74 | 76.34 ± 4.33 | 77.15 ± 3.39 |
| FD002 | 10 | regime_fault_skew | 29.17 ± 4.24 | 43.69 ± 0.91 | 43.65 ± 1.49 | 79.90 ± 18.90 | 90.36 ± 0.65 | 75.54 ± 9.94 | 79.08 ± 9.11 |
| FD004 | 10 | iid_swarm | 27.14 ± 0.54 | 42.59 ± 1.49 | 43.95 ± 1.89 | 86.62 ± 13.88 | 96.37 ± 1.91 | 76.66 ± 11.54 | 82.40 ± 6.30 |
| FD004 | 10 | lifespan_quantity_skew | 29.15 ± 2.40 | 39.43 ± 0.62 | 43.26 ± 0.57 | 86.41 ± 11.36 | 89.89 ± 3.41 | 79.01 ± 8.51 | 81.46 ± 8.66 |
| FD004 | 10 | regime_fault_skew | 27.77 ± 0.91 | 42.92 ± 1.63 | 50.27 ± 7.57 | 92.43 ± 5.89 | 95.09 ± 2.72 | 78.02 ± 6.52 | 82.87 ± 3.26 |
| FD004 | 20 | iid_swarm | 26.71 ± 1.32 | 43.95 ± 2.39 | 48.68 ± 8.54 | 55.45 ± 1.32 | 96.47 ± 1.28 | 57.98 ± 2.19 | 56.85 ± 1.71 |
| FD004 | 20 | lifespan_quantity_skew | 28.65 ± 1.68 | 41.48 ± 1.68 | 46.99 ± 5.60 | 65.32 ± 12.77 | 91.01 ± 0.46 | 65.14 ± 7.67 | 64.63 ± 8.32 |
| FD004 | 20 | regime_fault_skew | 26.05 ± 1.38 | 43.75 ± 2.03 | 44.46 ± 1.64 | 60.82 ± 10.15 | 95.15 ± 1.22 | 61.09 ± 5.29 | 63.92 ± 10.55 |

### Benchmark interpretation

- **Overall gap closure:** The overall centralized gap was not closed under intermittent service-bay federated learning. Across nearly all configurations, federated model synchronization regressed performance relative to the isolated local-only training baseline.
- **Range of gap_closed_fraction:** For the non-baseline FL methods (`fedavg_intermittent`, `fedavgm_staleness`, `scaffold_service_bay`, `fedavgm_personalized`, `fedavgm_asym_personalized`), the configuration-mean `gap_closed_fraction` ranged from **-354.57% ± 142.97%** (SCAFFOLD on FD002 K=10 regime_fault_skew) to **+1.16% ± 8.82%** (plain FedAvg on FD004 K=10 iid_swarm), with individual per-seed runs spanning between **-626.07%** and **+16.11%**.
- **Comparison against Centralized Oracle:** No method beats the centralized oracle on per-customer held-out RMSE in any configuration or seed run (0 out of 45 runs). Centralized oracle achieves 26.05 ± 1.38 to 31.11 ± 2.70 cycles, whereas local-only achieves 39.43 ± 0.62 to 43.95 ± 2.39 cycles, and federated methods range between 43.26 ± 0.57 and 96.47 ± 1.28 cycles.
- **Compute device:** The GPU XPU benchmark run executed on device `Intel(R) Graphics` using PyTorch XPU acceleration (`device: "xpu"`).

## Caveats

The C-MAPSS benchmarks are a controlled proxy for real service schedules. They do not capture geographic service-bay arrival processes, vehicle-specific hardware limits, communication failures during a visit, or manufacturer deployment constraints. The assumed update transport is model tensors and compact sufficient statistics; no secure aggregation or differential privacy claim is made. Centralized training is an upper-bound reference because it pools training windows that remain distributed in the federated methods. Personalized global-test scores use an ensemble of customer-specific models and are a diagnostic, while per-customer held-out vehicle scores are the primary personalization measure. Benchmark values should be interpreted with both RMSE and the asymmetric NASA/late-prediction metrics, especially for safety-related comparisons.
