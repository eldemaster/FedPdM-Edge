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

<!-- SERVICE_BAY_RESULTS_TABLE -->

### Centralized gap closure

For each configuration and method, gap closure is computed as `(local_only RMSE - federated RMSE) / (local_only RMSE - centralized_oracle RMSE)`. Values above 1 indicate the federated result beats the centralized reference on that evaluation; negative values indicate a regression against local-only. The JSON contains the seed-level inputs to this comparison. The values below are filled from the benchmark artifact at generation time.

<!-- SERVICE_BAY_GAP_TABLE -->

## Caveats

The C-MAPSS benchmarks are a controlled proxy for real service schedules. They do not capture geographic service-bay arrival processes, vehicle-specific hardware limits, communication failures during a visit, or manufacturer deployment constraints. The assumed update transport is model tensors and compact sufficient statistics; no secure aggregation or differential privacy claim is made. Centralized training is an upper-bound reference because it pools training windows that remain distributed in the federated methods. Personalized global-test scores use an ensemble of customer-specific models and are a diagnostic, while per-customer held-out vehicle scores are the primary personalization measure. Benchmark values should be interpreted with both RMSE and the asymmetric NASA/late-prediction metrics, especially for safety-related comparisons.
