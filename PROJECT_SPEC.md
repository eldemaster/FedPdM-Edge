# FedPdM-Edge: Project Specification and Author's Guide

This document explains the checked-in project as it exists on `master`, including its research benchmark, Flower training demo, streaming edge runtime, privacy controls, dashboard, paper pipeline, and verification gates. It distinguishes implemented code from components that are present but not yet wired together.

**Source snapshot:** `master` at `bb27348` (merge of PR #26, including PR #25). Source paths and API names below refer to that tree. Test counts are exact counts of `test_*` methods and test classes in all ten checked-in `tests/test_*.py` files. The checked-in `docs/PILOT_RUNBOOK.md` and parts of the manuscript still contain pre-PR descriptions; their drift is called out below rather than repeated as current implementation.

## 1. Start here: what this project does

FedPdM-Edge is a research proof of concept for estimating the remaining useful life (RUL) of an engine from recent sensor readings. The model is a compact one-dimensional convolutional neural network. In the federated training demo, each client trains on its own C-MAPSS windows and sends model parameters to a coordinator. In edge inference, a rolling buffer turns incoming readings into fixed-size windows, a local model estimates RUL, and local diagnostics explain which sensors and subsystems most influenced the estimate. Only a small, allow-listed diagnostic summary is sent to the dashboard; raw sensor readings and model windows are rejected by that alert path.

The codebase has several related but separate paths:

| Path | What it is for | Current implementation boundary |
|---|---|---|
| `server.py` + `client.py` | Flower/gRPC federated training against FD002 | TLS/mTLS and optional client-update clipping/noise are implemented. The server strategy is FedAvg; clients can add a FedProx local penalty. |
| `fed_sim.py`, `experiments/reproduce.py` | In-process experiments and benchmark comparisons | These are simulations; they do not exercise the Flower network transport. |
| `experiments/vehicle_swarm_benchmark.py` | Strictly vehicle-disjoint, partially participating customer-fleet simulation | Runs `K=10` and `K=20` swarms with IID, lifespan/quantity-skewed, and regime/drift-skewed partitions; compares local-only, FedAvg, FedProx, asymmetric-loss, and personalized variants. It does not exercise Flower transport. |
| `fed_norm.py` | Federated aggregation of channel statistics and telemetry sanitation | Implemented as a reusable module with tests. It is not currently called by `client.py` or `edge_runtime.py`. |
| `vehicle_datasets.py` | Opt-in vehicle-fleet dataset contract, adapters, and flexible CNN | Implements combined C-MAPSS, Scania Component X, and N-CMAPSS adapters without downloading data or changing the baseline pipeline. Only C-MAPSS was locally available for real-data verification; external-adapter tests use synthetic schema fixtures. |
| `edge_runtime.py` + `diagnostics.py` | Streaming/replayed local inference, diagnostics, and debounced alerts | Implemented. The runner uses the existing `common.load_scales()` interface; it does not yet consume a `FederatedScaler` directly. |
| `dashboard/` | Offline PoC replay and a live summary API | Implemented with a standard-library HTTP server. Live updates are in-memory summaries, not a durable monitoring service. |

### 1.1 How Everything Fits Together

#### One operational edge cycle

The following is the intended end-to-end operational story, with the current integration boundary made explicit:

1. **A sensor snapshot arrives.** A file replay, MQTT broker, or JSON Lines input provides an asset ID, cycle/timestamp, and some or all of the 17 channels in `common.USEFUL_SENSORS`. `TelemetrySample.from_mapping()` in `edge_runtime.py` validates the asset and cycle and converts usable readings to numbers. Unknown channels and missing/unparseable values are ignored. No PLC/OPC-UA adapter is included.
2. **The stream becomes a stable window.** `RollingWindowBuffer` keeps a bounded per-asset history. Missing or non-finite channels use last-observation-carried-forward (LOCF), with the configured initial fill value before any value has been seen. A cold-start history is left-padded by repeating its first reading until it reaches `SEQ_LEN` (30). Cycles and timestamps must increase for each asset.
3. **Signal hygiene and scaling are selected.** `fed_norm.sanitize_telemetry_frame()` can normalize missing, infinite, dropped, and optionally stuck channels; `compute_local_stats()` and `aggregate_federated_stats()` can produce a shared `FederatedScaler` without passing raw arrays to the aggregator. That scaler supports `zscore` or `minmax` and defaults to clipping scaled values to [-5, 5]. These are available building blocks, not an active `EdgeInferenceRunner` stage today. The runner currently passes the rolling raw-unit frame to `diagnostics.EdgeDiagnostics`, which uses the legacy scale dictionary loaded through `common.load_scales()`.
4. **The model estimates RUL locally.** `EdgeDiagnostics` prepares scaled 30-by-17 windows and runs `common.CmapssRegressor`. It bounds the operational final prediction to [0, 125] cycles. The prediction and all sensor readings stay in the edge process.
5. **Diagnostics explain and contextualize the prediction.** `diagnostics.attribute_windows()` uses `baseline_x_gradient` by default when a healthy reference exists. The healthy baseline is operating-regime-aware. The module also computes signed sensor drift against the engine's early-life offset and produces subsystem shares, top contributors, top drifting sensors, and a maintenance summary. Operating settings are excluded from fault responsibility.
6. **The alert transition is debounced.** `AlertManager(confirm_cycles=3)` requires three consecutive observations of a changed severity before changing its accepted state. A new candidate severity restarts the count; observing the current severity clears a pending transition. Both escalation and recovery are debounced.
7. **Only an aggregate summary is attempted across the edge boundary.** `WebhookDashboardSink` sends the validated alert payload to `dashboard/server.py` at `POST /api/edge_update`, with no raw values. The runner payload requires `primary_subsystem`, `summary_text`, and `privacy`; the dashboard accepts `summary` instead and does not allow the other two fields. As checked in, the endpoint therefore responds HTTP 400 to the unmodified `EdgeInferenceRunner` webhook payload. An adapter/schema reconciliation is needed before this edge-to-dashboard route works end to end. `JsonLinesAlertSink` remains a working local output path.
8. **The browser can read accepted summaries.** The dashboard accepts supported summary JSON bodies up to 64 KiB and stores the most recent accepted summary per `asset_id` in process memory. `GET /api/live` returns `{"mode":"live","assets":...}` when an update has arrived, or `{"mode":"poc","assets":{}}` otherwise. The UI can switch between **PoC Replay** and **Live Edge Stream**; replay uses committed `dashboard/demo_data.json`, while live mode polls the API for aggregate summaries. It currently cannot display alerts from the runner webhook until the payload mismatch above is reconciled.

The terms “one operational cycle” and “one model prediction” refer to one incoming sample and the resulting current rolling window. Alert delivery occurs only when the debounced severity state changes, not for every sample.

#### Offline federated training cycle

1. `server.py` loads its configured transport credentials, initializes the 22,273-parameter `CmapssRegressor`, and starts Flower's gRPC server. TLS is required unless `--insecure` is explicitly selected. Client certificates are required by default when TLS is active.
2. Each `client.py` process reads its `CLIENT_ID`, `FL_SERVER`, local partition and FD002 training data. It receives the current global weights and trains locally with Adam. The loss defaults to MSE; `--loss-fn asymmetric` selects `AsymmetricRULLoss`. A positive proximal coefficient in the server's `PROXIMAL_MU` adds the FedProx penalty.
3. Each client returns its updated model parameters, its number of training windows, and timing/resource/security metadata. Raw training windows are not part of this return payload.
4. If `--dp-clip C` is configured, the client clips the concatenated full model delta to L2 norm at most `C`. If `--dp-noise-mult σ` is also positive, it adds independent Gaussian noise with per-coordinate standard deviation `C × σ` to that client update before returning it.
5. Flower's FedAvg strategy aggregates model parameters weighted by the client example counts and evaluates the global model on the official test set each round. Transport encryption/authentication protects the link; clipping/noise modifies transmitted updates. These controls do not provide secure aggregation or an accounted formal differential-privacy guarantee.

## 2. Architecture, transport security, and privacy

### 2.1 Code-level data flow

```text
                          FEDERATED TRAINING (separate offline path)
  FD002 files -> client.py / local training -> model deltas + counts -> server.py / Flower FedAvg
                    TLS/mTLS; optional whole-delta clip + Gaussian noise

                          EDGE INFERENCE (local runtime)
  JSONL / CSV replay / MQTT -> TelemetrySample -> RollingWindowBuffer (30 x 17)
     -> EdgeDiagnostics -> CmapssRegressor + baseline_x_gradient + HealthyBaseline drift
     -> AlertManager (3 confirmations) -> allow-listed summary only
     -> WebhookDashboardSink -> POST /api/edge_update -> GET /api/live -> browser

  fed_norm.py provides separately tested sanitation and federated-statistics tools;
  it is not currently connected to the two paths above.
```

### 2.2 Flower TLS and mTLS controls

`server.py` and `client.py` both reject missing or partial TLS configuration unless plaintext was explicitly requested. There is no silent plaintext fallback.

| Process | CLI options | Environment defaults | Meaning |
|---|---|---|---|
| `server.py` | `--ca-cert`, `--cert`, `--key`, `--require-client-cert` / `--no-require-client-cert`, `--insecure` | `FL_CA_CERT`, `FL_SERVER_CERT`, `FL_SERVER_KEY` | Server certificate/key and trusted client CA. Client certificate verification is on by default with TLS. `--no-require-client-cert` permits server-only TLS explicitly. |
| `client.py` | `--ca-cert`, `--cert`, `--key`, `--insecure` | `FL_CA_CERT`, `FL_CLIENT_CERT`, `FL_CLIENT_KEY` | The CA is required for TLS. Client certificate and key must be supplied together; including them enables client-authenticated mTLS. CA-only client TLS is permitted when the server was explicitly configured without client authentication. |

`--insecure` is local-development-only and conflicts with any certificate option or corresponding environment variable. The process prints an explicit plaintext warning. The code validates PEMs before starting and uses scoped adapters for Flower's gRPC credential factory; it refuses unsupported Flower TLS implementations instead of silently weakening verification.

`deploy.sh` requires `--ca-cert`, `--cert`, and `--key` unless `--insecure` is explicit. It accepts `FL_CA_CERT`, `FL_CLIENT_CERT`, and `FL_CLIENT_KEY` as defaults, excludes conventional key/certificate files from rsync, and prints the corresponding client launch options. Certificates must be provisioned separately on each edge node. `DP_CLIP` and `DP_NOISE_MULT` can set the client update controls described below.

### 2.3 Optional client-update clipping and noise

`client.py` accepts `--dp-clip C` (environment `DP_CLIP`) and `--dp-noise-mult σ` (environment `DP_NOISE_MULT`, default `0`). `C` must be finite and positive. `σ` must be finite and non-negative; positive noise requires a clip bound. With both enabled, the standard deviation is `Cσ` for each coordinate of the full parameter delta.

This is a client-update clipping and Gaussian-noise mechanism intended as a DP component. It can bound the norm of each client's transmitted delta and perturb that delta. **The repository has no formal ε/δ privacy accountant or privacy budget calibration**, so this implementation does not support a formal end-to-end DP guarantee. It also has no secure aggregation: the coordinator receives individual client updates. mTLS protects transport and authenticates participating endpoints but does not hide updates from the coordinator.

## 3. Data, model, and preprocessing

### 3.1 C-MAPSS inputs and targets

`common.py` reads 26 whitespace-separated columns: `unit`, `cycle`, `setting1..3`, and `s1..s21`. `USEFUL_SENSORS` selects 17 model channels in this order: `setting1`, `setting2`, `setting3`, `s2`, `s3`, `s4`, `s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20`, `s21`. The settings describe operating conditions; the other 14 channels are physical sensors. The input sequence length is `SEQ_LEN = 30`; the target cap is `MAX_RUL = 125`.

For a training window ending at cycle `t` on an engine with final cycle `T`, the target is `min(125, T - t)`. Training windows use stride one. The official test set uses one terminal 30-cycle window per engine; shorter histories are left-padded by repeating their first observed row.

The dataset archive path is resolved in `common.cmapss_zip_path()` from `CMAPSS_ZIP`, then `./CMAPSSData.zip`, then `~/CMAPSSData.zip`. `experiments/reproduce.py` also has its own `--data` interface.

### 3.2 Model and local training

`common.CmapssRegressor` takes `(batch, 30, 17)`, transposes for `Conv1d`, and applies:

| Layer | Definition | Output shape after layer (excluding batch) |
|---|---|---|
| `conv1` + ReLU | `Conv1d(17, 32, kernel_size=3, padding=1)` | `(32, 30)` |
| `pool1` | `MaxPool1d(2)` | `(32, 15)` |
| `conv2` + ReLU | `Conv1d(32, 64, kernel_size=3, padding=1)` | `(64, 15)` |
| `pool2` | `MaxPool1d(2)` | `(64, 7)` |
| flatten | — | `448` |
| `fc1` + ReLU | `Linear(448, 32)` | `32` |
| `fc2` | `Linear(32, 1)` and squeeze | scalar per example |

The model has 22,273 trainable parameters. `common.train_epochs()` uses Adam with default learning rate `1e-3`; loader batch size varies by path (64 in the Flower client and legacy demo; 128 in `experiments/reproduce.py`). `client.py` defaults to two local epochs per round.

### 3.3 Loss and reported metrics

For residual `d = y_pred - y_true`, `common.AsymmetricRULLoss` computes the mean of:

```text
alpha_late * d²    when d >= 0   (late: RUL is overestimated)
alpha_early * d²   when d < 0    (early: RUL is underestimated)
```

The defaults are `alpha_late=2.5` and `alpha_early=0.5`, both positive. The loss is quadratic on both sides and continuous with continuous first derivative at zero. It is selectable in the Flower client; MSE remains the default.

`common.nasa_score()` reports the benchmark's asymmetric exponential sum. With `d = y_pred - y_true`, each sample contributes `exp(-d/13)-1` for `d < 0`, or `exp(d/10)-1` for `d >= 0`. The smaller divisor makes late errors grow faster. RMSE is `sqrt(mean((y_pred-y_true)^2))`; `common.late_prediction_rate()` is the fraction with `y_pred > y_true`. The NASA score is a sum, so its absolute value depends on test-set size.

The experiment suite also contains a capped-exponent score-aware training ablation. Do not confuse that objective with the quadratic `AsymmetricRULLoss`, or either training loss with the uncapped NASA evaluation metric.

### 3.4 Vehicle-fleet dataset contract and adapters (`vehicle_datasets.py`)

`VehicleFleetDataset` keeps one time-ordered frame and metadata record per anonymous vehicle. It exposes `feature_cols`, `setting_cols`, optional official endpoint `test_ruls`, whole-vehicle `partition_by()`, and `windows()`. Window construction is vehicle-local and causal: it sanitizes only the observed portion of each window, omits unknown targets, left-pads short `last_only=True` histories, and returns `float32` arrays shaped `(N, W, C)` and `(N,)`. Vehicles must be partitioned before windows or training scalers are constructed.

| Adapter / symbol | Implemented contract | Evidence boundary |
|---|---|---|
| `CmapssCombinedAdapter` | Loads FD001–FD004 with collision-free IDs such as `FD002-train-U001`; supports train, test, or both splits and subset grouping. The local archive contains **709 training engines and 707 official test engines** across the four subsets. | Combining subsets increases the controlled simulated fleet and mixes experimental populations. It does not create 709 real vehicles or real customer ownership. |
| `ScaniaComponentXAdapter` | Reads the published commercial heavy-truck CSV schema: 105 operational features, eight categorical specifications, irregular `time_step`, repair events, censoring, and validation/test interval classes. `partition_by_specifications()` forms whole-truck cohorts. | Scania data was not downloaded or benchmarked here. Censored trucks retain lower bounds and do not produce invented point-RUL targets. Specifications are cohort proxies, not customer IDs or proof of one chassis model. |
| `NCMAPSSAdapter` | Reads official `W`, `X_s`, optional `X_v`, `T`, `Y`, and `A` HDF5 arrays; partitions whole engines by flight class. Physical health parameters and health-state metadata remain outside model features. | N-CMAPSS data was not downloaded or benchmarked here. Its component-health truth comes from simulated degradation under recorded flight envelopes, not observed vehicle failures. The eager DataFrame adapter is not out-of-core training. |
| `build_flexible_cnn(in_channels, seq_len)` | Reuses the baseline Conv32/Conv64/FC32 forward path with configurable channel count and sequence length. At `(17, 30)` it has state-dict and output parity with `CmapssRegressor` and **22,273 parameters**. | All clients in one federated run still need identical feature order, channel count, and sequence length. Arbitrary-channel support does not make different dataset schemas directly aggregatable. |

`docs/DATASET_EVALUATION.md` records provenance, schemas, licenses, download locations, target semantics, memory limits, and the adapter examples. The external Scania and N-CMAPSS fixtures in `tests/test_vehicle_datasets.py` are explicitly synthetic schema tests; a successful optimizer step proves interface compatibility, not predictive validity.

## 4. Algorithms and the reason for the current choices

### 4.1 Design Rationale Cheat-Sheet

| Decision | Chosen method | Alternative considered or tested | Why this choice fits the project |
|---|---|---|---|
| Temporal model | 1D-CNN (`CmapssRegressor`) | Compact CNN-LSTM in `experiments/reproduce.py` | The convolutional model is compact, simple to deploy, and is the repository's main model. The CNN-LSTM is retained as an ablation; the manuscript reports it performed worse under the tested training budget. No Pi efficiency claim follows without Pi measurements. |
| Risk-aware local objective | `AsymmetricRULLoss` | MSE; direct raw exponential NASA score as a training objective | MSE treats early and late residuals symmetrically. The quadratic asymmetric loss increases the cost of overestimating RUL while keeping a smooth, linearly growing gradient. The NASA exponential is useful as an evaluation metric but can produce extreme gradients during optimization; the separate score-aware ablation caps its exponent. |
| Federated optimizer | FedAvg baseline; FedProx penalty option | FedAdam | FedAvg is the transparent sample-count-weighted reference. FedProx adds a local penalty against client drift without changing the aggregation rule. FedAdam is implemented in the paired simulation suite, but its fixed server learning-rate configuration produced much higher RMSE (78.23 on FD002 and 56.62 on FD004); this is a negative result for that configuration, not all FedAdam settings. |
| Attribution | `baseline_x_gradient` | Plain `gradient` (and `input_x_gradient`) | Multiplying the gradient by distance from a healthy, regime-aware baseline asks which sensor deviations from expected healthy behavior affect the estimate. Plain gradient measures local sensitivity around the observed input without that healthy reference. All methods exclude the three operating settings from fault responsibility. |
| Shared normalization | Chan/Welford federated sufficient statistics in `fed_norm.py` | Centrally computed `global_scale.json` | Clients can aggregate counts, means, second-moment accumulators, minima and maxima without sending raw samples to the aggregation function. This supports a distributed fitting path and reduces direct central collection. The current legacy Flower path still uses `global_scale.json`; the federated scaler is not yet wired into it, and aggregate statistics are not protected by secure aggregation. |
| Alert stability | `AlertManager(confirm_cycles=3)` | Immediate raw threshold transition | Three consecutive samples with the candidate severity suppress a one-cycle fluctuation from changing the accepted alert state. The same rule applies to recovery. Three is the implemented default, not a site-calibrated maintenance SLA. |

### 4.2 Federated averaging and FedProx

In the Flower demo, `server.py` configures `flwr.server.strategy.FedAvg` with all configured clients participating in fit and evaluation. Client updates are weighted by the number of local training examples. A positive `PROXIMAL_MU` value configures `common.train_epochs()` to add `mu/2 * ||w - w_global||²` to each local batch objective. A zero value disables the proximal term. The `fed_sim.py` and `experiments/reproduce.py` paths test additional methods, including FedAdam, but those are local experiments rather than the Flower transport path.

### 4.3 Vehicle-swarm benchmark (`experiments/vehicle_swarm_benchmark.py`)

The vehicle-swarm benchmark models one manufacturer coordinating customer fleets while preserving the vehicle boundary. Every C-MAPSS unit belongs to exactly one client; train/validation splitting then uses whole vehicles. The committed configurations are FD002 with `K=10`, FD004 with `K=10`, and FD004 with `K=20`. Per round, deterministic partial participation selects exactly `round(K * fraction_fit)` clients: 5 of 10 at `fraction_fit=0.5`, or 6 of 20 at `fraction_fit=0.3`.

The three partition schemes are `iid_swarm`, `lifespan_quantity_skew`, and `regime_fault_skew`. Lifespan/quantity skew draws client fleet sizes from a Dirichlet distribution with `alpha=0.5` and assigns ordered lifespan bands. Regime/drift skew fits six operating-setting clusters and draws each profile's allocation with Dirichlet `alpha=0.3`; FD004 adds a label-free two-cluster end-of-life drift grouping. That grouping is not a component-fault label. The JSON reports vehicle counts, supervised-window count CV and Gini, pairwise lifespan/regime/drift Jensen-Shannon divergence, allocation entropy, and a combined heterogeneity score. Both intended Non-IID schemes have a strictly higher combined score than IID in every committed configuration.

The compared methods are `local_only`, `fedavg_mse`, `fedprox_mse`, `fedavg_asym`, `fedprox_asym`, and `fedavg_asym_personalized`. The bounded protocol uses five seeds (`7`, `42`, `100`, `777`, `2024`), five rounds, one local epoch, at most 512 training windows and 128 validation windows per client, `mu=0.05`, and two local personalization epochs at one-quarter of the training learning rate. Federated normalization calls `compute_local_stats()` inside each fleet and gives the server only `LocalStats` objects; every recorded run reports `raw_sensor_rows_shared: 0`. This is a code-path audit, not secure aggregation or a formal privacy proof.

Across the nine dataset/client-count/partition configurations, asymmetric FedAvg lowers mean late-prediction rate from the MSE model's 0.46–0.52 range to 0.33–0.35 and lowers NASA score per test vehicle from 282–834 to 102–144, while increasing symmetric RMSE by 2.3–6.4 cycles. `FedProx(mu=0.05)` is indistinguishable at the displayed precision in this short protocol. Customer fine-tuning improves matched-client mean validation RMSE in eight of nine configurations, but worsens some worst-client results; personalization must therefore be gated rather than assumed beneficial. Exact per-seed partitions, participants, trajectories, metrics, and aggregates are in `results/vehicle_swarm_benchmark.json`; the rounded interpretation is in `docs/vehicle_swarm.md`.

## 5. Federated normalization and signal hygiene (`fed_norm.py`)

### 5.1 Local sufficient statistics and merge

`LocalStats` stores per-channel `count`, `mean`, `M2` (sum of squared deviations from the mean), `min`, and `max`. `compute_local_stats(training_array)` treats the final axis as channels, flattens leading sample dimensions, and excludes non-finite values per channel.

`aggregate_federated_stats(client_stats, ...)` accepts only `LocalStats`, not raw sample arrays. For each new client's count `n_b`, mean `m_b`, and accumulator `M2_b`, it combines the existing aggregate `(n, m, M2)` using Chan's parallel-variance merge:

```text
delta = m_b - m
n_total = n + n_b
m_total = m + delta * n_b / n_total
M2_total = M2 + M2_b + delta² * n * n_b / n_total
```

The implementation applies these elementwise per channel and merges minima/maxima independently. The resulting variance is population variance `M2/count`; `std = sqrt(max(variance, 0))`. Every channel must have at least one finite observation overall.

### 5.2 Transform and sanitation

`FederatedScaler` supports `mode="zscore"` (`(x - mean) / std`) and `mode="minmax"` (`(x - min) / (max - min)`), checks the channel count, and returns `float32`. Constant standard deviations or min/max spans use a denominator of 1 to avoid division by zero. The default `clip_range` is `(-5.0, 5.0)` and can be disabled with `None` or bypassed per call with `clip=False`. `as_dict()` serializes the mode, clipping range, and per-sensor statistics.

`sanitize_telemetry_frame(frame, sensors=..., healthy_means=..., clip_bounds=..., stuck_tolerance=...)` coerces sensor columns to numeric, treats infinities as missing, creates missing channels, and can flag near-constant per-unit signals as stuck. It forward-fills then back-fills within each `unit` (or across the frame if no unit column exists). Remaining gaps fall back to supplied healthy means, otherwise available finite means, otherwise zero. Optional raw bounds can clip readings. It returns a float32 array and a report with imputation/missing/clipped information.

### 5.3 Why the legacy path remains

`experiments/reproduce.py` intentionally retains its established centralized scaling path for benchmark continuity: changing preprocessing would change the experiment protocol and invalidate direct comparison with its tracked results. The Flower demo also continues to load the committed `global_scale.json`. The separate vehicle-swarm benchmark does use `compute_local_stats()`, `aggregate_federated_stats()`, and `FederatedScaler` to demonstrate decentralized normalization across simulated fleets. `fed_norm.py` includes a parity probe, but neither the scaler nor `sanitize_telemetry_frame()` is automatically inserted into `reproduce.py`, `client.py`, or `EdgeInferenceRunner`. Connecting it to those operational paths requires an explicit deployment/experiment integration decision and corresponding validation.

## 6. Streaming edge runtime and local diagnostics

### 6.1 Sources, windows, and inference

`edge_runtime.py` defines:

| Symbol | Role |
|---|---|
| `TelemetrySample` | Normalized asset ID, non-negative integer cycle, normalized timestamp, and recognized sensor values. Timestamps may be numeric or ISO-8601. |
| `JsonLinesStreamSource` | Parses one telemetry JSON object per non-empty line. |
| `CsvReplaySource` | Replays rows with `asset_id` (or `unit`), `cycle`, and sensor columns. |
| `MqttSource` | Optional MQTT adapter. `paho-mqtt` is imported only when needed; a client can be injected for testing. |
| `RollingWindowBuffer` | Thread-safe, bounded per-asset rolling window; default size is `SEQ_LEN`; fills absent/non-finite values with per-channel LOCF and cold-start pads by repeating the first reading. |
| `EdgeInferenceRunner` | Runs diagnostics for each incoming sample, clips final RUL to `[0, MAX_RUL]`, updates the alert state, and returns an in-process result. |
| `JsonLinesAlertSink` | Writes validated allow-listed alert JSON to stdout or a file. |
| `WebhookDashboardSink` | POSTs the same validated summary payload to a dashboard endpoint. |

The CLI in `edge_runtime.py` requires exactly one of `--jsonl PATH` or `--replay-cmapss {FD001,FD002,FD003,FD004}`. Optional arguments are `--model`, `--scales` (default `global_scale.json`), `--alerts` (default `-`, stdout), `--confirm-cycles` (default `3`), `--max-samples`, and `--interval-ms` (default `0.0`). Without `--model`, it warns that it is using an untrained demonstration model. JSONL CLI mode uses a scale-derived neutral baseline and warns that production drift scoring should fit `HealthyBaseline` on local healthy history.

### 6.2 Attribution and healthy-baseline drift (`diagnostics.py`)

`diagnostics.attribute_windows()` provides `baseline_x_gradient`, `input_x_gradient`, and `gradient`. The default baseline method falls back to input-times-gradient if no baseline is supplied. For each sensor channel, the baseline attribution averages `abs((x - baseline) * d(predicted_rul)/dx)` over window positions. The three operating settings are explicitly set to zero. `HealthyBaseline` assigns cycles to operating regimes, normalizes fault sensors within each regime, estimates an engine's early-life offset, and computes signed drift relative to pooled healthy noise.

The four subsystem groups are Fan, LPC, HPC, and Turbine. `EdgeDiagnostics` exposes prediction, per-sensor contribution, subsystem shares, drifting sensors, and bilingual maintenance text. These are explanations and drift indicators, not supervised component-fault diagnoses: C-MAPSS has no per-engine component labels.

### 6.3 Alert state and privacy boundary

`AlertManager` has an independent current/pending state for each asset. `diagnostics.severity_for()` labels predictions at or below 20 cycles `critical`, at or below 45 cycles `warning`, and above 45 `healthy`; candidate transitions are accepted after `confirm_cycles` consecutive updates. `EdgeInferenceRunner` emits to its configured sink only when an accepted state change occurs.

The payload fields are `asset_id`, `cycle`, `predicted_rul`, `severity`, `primary_subsystem`, `subsystem_shares_pct`, `top_contributors`, `top_drifting`, `summary_text`, and `privacy`. `_validate_alert_payload()` requires exactly this set, checks the privacy metadata, recursively rejects raw sensor channel names and raw-value/window fields, then JSON-round-trips the detached object. This prevents accidental raw values in this specific alert path; it is not a claim that every possible process or transport in the project is private.

## 7. Dashboard, paper, tests, and CI

### 7.1 Dashboard (`dashboard/`)

`dashboard/server.py` is a Python standard-library threaded HTTP server. It serves `dashboard/index.html` and committed `dashboard/demo_data.json`, blocks directory listings, and provides:

| Route | Behavior |
|---|---|
| `GET /api/health` | Returns `{"status":"ok","has_data":...}`. |
| `POST /api/edge_update` | Accepts JSON summaries with allow-listed fields; body must be 1–65,536 bytes. Larger bodies return HTTP 413. Invalid JSON, unsupported fields, malformed asset IDs, and nested raw telemetry keys (`raw_sensors`, `raw_window`, `windows`, `telemetry_raw`) return HTTP 400. |
| `GET /api/live` | Returns the in-memory latest summary per asset and mode `live` if nonempty, otherwise `poc`. |

The POST API accepts only the field allow-list in `dashboard/server.py` and does not store a history. The edge webhook and dashboard API currently have incompatible schemas: `EdgeInferenceRunner` emits `primary_subsystem`, `summary_text`, and `privacy`, while the API allows `summary` and rejects unsupported keys. No adapter is present, so direct webhook posts receive HTTP 400. The server validates nested raw telemetry key names (`raw_sensors`, `raw_window`, `windows`, `telemetry_raw`) and rejects unsupported top-level fields. It does not authenticate callers; bind to an appropriate trusted interface/network for a pilot.

The browser UI provides a **PoC Replay / Live Edge Stream** toggle. Replay uses the committed demo snapshot. Live mode polls `GET /api/live` and renders the latest summaries. The normal asset replay charts still use the fixed demo trajectories and sensors. Cluster and business-tab headline metrics are data-driven from `DATA.cluster` and `DATA.benchmarks`; some labels and explanatory copy remain static. Cadence conversion is user-configurable and visibly labeled as a configurable conversion from cycles to a pilot cadence; C-MAPSS itself defines cycles, not elapsed hours.

`dashboard/generate_demo_data.py` regenerates reference demo data and requires the C-MAPSS archive. It is not required to serve the committed dashboard. Commands:

```bash
python3 dashboard/server.py
python3 dashboard/server.py --port 0 --no-browser
CMAPSS_ZIP=/path/to/CMAPSSData.zip python3 dashboard/generate_demo_data.py
```

### 7.2 Paper pipeline (`paper/`)

`paper/main.tex` is an IEEEtran manuscript whose sole listed author is Alessandro De Martini, with a Mid Sweden University affiliation and the ResilientEdge acknowledgement. The paper covers the paired RUL benchmark, asymmetric-risk loss, local attribution/drift method, its evaluation boundary, and generated local-diagnostics results/tables. It uses `paper/make_figures.py` to regenerate scalar macros and tables under `paper/generated/` from committed result artifacts. `paper/Makefile` runs the generator and then `latexmk -pdf main.tex`; generated files should not be edited manually.

There is exactly one open `\todo{...}` marker in the current `paper/*.tex` source: `\todo{AI-use disclosure pending}` in `paper/main.tex`. No co-author or affiliation TODO remains. The manuscript's prose has not been fully reconciled with PRs #17–#26: in particular, `paper/sections/architecture.tex`, `paper/sections/related_work.tex`, and `paper/sections/setup.tex` still say TLS/mTLS, DP, and federated normalization are absent, and the paper does not cover the new vehicle-swarm or vehicle-dataset work. Treat source code as authoritative for implementation status; those manuscript statements are known documentation drift. This spec update does not alter `paper/`.

### 7.3 Test suites: exact current inventory

All ten files use `unittest`. The table reports **test classes / methods whose names start with `test_`**, counted directly from the checked-in files.

| File | Classes / methods | Main coverage |
|---|---:|---|
| `tests/test_diagnostics.py` | **6 / 22** | Sensor map, attribution methods and invariants, baseline drift, summaries, and real-data integration. |
| `tests/test_dashboard.py` | **6 / 26** | Demo-data invariants, generator, regeneration, HTTP server, static UI checks, and browser behavior. |
| `tests/test_diagnostics_validation.py` | **5 / 16** | Socket audit, validation helpers, artifact integrity, paper digest, and C-MAPSS smoke checks. |
| `tests/test_system_smoke.py` | **9 / 21** | Model/loss/metrics, sequence and split behavior, simulation, edge profiler, and paper build. |
| `tests/test_federated_e2e.py` | **5 / 26** | Client updates, weighted averaging, transport and privacy configuration, synthetic edge path, and Flower loopback including mTLS cases. |
| `tests/test_federated_norm.py` | **3 / 6** | Federated stats/parity and telemetry sanitation. |
| `tests/test_edge_runtime.py` | **5 / 9** | Rolling-window buffering, alert hysteresis, end-to-end safe JSONL alert, and injected MQTT behavior. |
| `tests/test_preflight.py` | **3 / 5** | Symlink detection, dashboard/diagnostics JSON invariants, and fast preflight command. |
| `tests/test_vehicle_swarm.py` | **4 / 6** | Exhaustive/disjoint vehicle partitions, measured Non-IID skew, deterministic partial participation, `LocalStats`-only normalization, all-method synthetic smoke, and a C-MAPSS-gated FD002 round. |
| `tests/test_vehicle_datasets.py` | **5 / 15** | Flexible-CNN parity/dimensions, Scania schema/censoring/causal windows, N-CMAPSS HDF5/feature isolation, and combined C-MAPSS counts/split isolation. |
| **Total** | **51 / 152** | Exact static inventory; runtime skips depend on available data and optional tools. |

Run the suite with `python3 -m unittest discover -s tests -v`. C-MAPSS-gated tests require a valid `CMAPSS_ZIP`; browser checks can depend on Selenium/driver availability. Test count is a source inventory, not a claim that every gated integration test ran in a given environment.

### 7.4 Preflight and continuous integration

`scripts/preflight_check.py` can run its full unittest discovery or a fast artifact-only pass. It checks repository symlinks, SHA-256 pins for the canonical experiment/summary/paper tree, dashboard attribution conservation and zero-raw-byte metadata, the diagnostics socket/privacy audit, C-MAPSS environment configuration, and optionally the tests. Flags are `--skip-unittest` and `--with-cmapss`. `--with-cmapss` requires `CMAPSS_ZIP` to name an existing archive; without it, data-gated tests may skip.

`.github/workflows/ci.yml` runs on pushes to `master` and pull requests using Ubuntu, Python 3.11, and Node.js 20. It installs `requirements.txt`, runs `python3 -m unittest discover -s tests -v`, then runs `python3 scripts/preflight_check.py --skip-unittest`. Thus CI includes the fast preflight checks as a gate after unit tests.

### 7.5 Useful commands and environment variables

| Entry point | Verified invocation/configuration |
|---|---|
| Flower coordinator | `python3 server.py [--ca-cert PATH --cert PATH --key PATH]`; configure `SEED`, `ROUNDS`, `MIN_CLIENTS`, `PROXIMAL_MU`, `LOCAL_EPOCHS`, `RESULTS_DIR`, and `BIND`; TLS paths can come from `FL_CA_CERT`, `FL_SERVER_CERT`, `FL_SERVER_KEY`; `--insecure` is explicit local-only plaintext. |
| Flower client | `CLIENT_ID=1 FL_SERVER=host:8081 python3 client.py [--loss-fn mse\|asymmetric]`; optional `--alpha-late`, `--alpha-early`, TLS flags, and DP flags. Environment: `SEED`, `LOSS_FN`, `ALPHA_LATE`, `ALPHA_EARLY`, `FL_CA_CERT`, `FL_CLIENT_CERT`, `FL_CLIENT_KEY`, `DP_CLIP`, `DP_NOISE_MULT`. |
| Deployment helper | `./deploy.sh --ca-cert PATH --cert PATH --key PATH` or explicit `./deploy.sh --insecure` for local development. |
| Edge runtime | `python3 edge_runtime.py --jsonl PATH [--model PATH] [--scales PATH] [--alerts PATH] [--confirm-cycles N] [--max-samples N] [--interval-ms MS]`, or use `--replay-cmapss FD001` (FD001–FD004 accepted). |
| Dashboard | `python3 dashboard/server.py [--host HOST] [--port PORT] [--no-browser]`. |
| Artifact preflight | `python3 scripts/preflight_check.py [--skip-unittest] [--with-cmapss]`. |
| Paired research suite | `python3 experiments/reproduce.py --data PATH --out DIR [--threads N]`; `--summarize-only` and `--quick` are also defined by that script. |
| Vehicle-swarm benchmark | `CMAPSS_ZIP=/path/to/CMAPSSData.zip python3 experiments/vehicle_swarm_benchmark.py`; optional flags are `--output PATH`, `--rounds N`, `--max-train-sequences N`, `--max-val-sequences N`, `--seeds ...`, and `--synthetic-smoke`. |
| Vehicle dataset tests | `python3 -m unittest tests/test_vehicle_datasets.py -v`; set `CMAPSS_ZIP` to enable the archive integration case, and install optional `h5py` to enable HDF5 fixture cases. |

## 8. Repository map and current status

### 8.1 File map

| Path | Responsibility / important symbols |
|---|---|
| `common.py` | Shared C-MAPSS loading/scaling, `USEFUL_SENSORS`, `SEQ_LEN`, `MAX_RUL`, `CmapssRegressor`, `AsymmetricRULLoss`, metrics, windowing, unit splitting, training. |
| `server.py` | Flower coordinator, `AuditedFedAvg`, certificate validation, TLS/mTLS setup, `--insecure` handling, result output. |
| `client.py` | Flower `SupervisedClient`, local fit/evaluation, transport credentials, whole-delta clipping/noise, client CLI. |
| `deploy.sh` | Client rsync deployment and explicit transport/certificate launch options. |
| `fed_norm.py` | `LocalStats`, `compute_local_stats`, `aggregate_federated_stats`, `FederatedScaler`, `sanitize_telemetry_frame`, parity probe. |
| `edge_runtime.py` | `TelemetrySample`, JSONL/CSV/MQTT sources, `RollingWindowBuffer`, `EdgeInferenceRunner`, `AlertManager`, JSONL and webhook sinks. |
| `diagnostics.py` | `HealthyBaseline`, `fit_healthy_baseline`, attribution methods, `EdgeDiagnostics`, maintenance summaries and evaluation boundary. |
| `fed_sim.py` | Local FedAvg/FedProx simulation, separate from networked Flower transport. |
| `experiments/reproduce.py` | Paired FD001–FD004 experiments, centralized preprocessing, partitions, CNN/CNN-LSTM and aggregation/loss comparisons. |
| `experiments/vehicle_swarm_benchmark.py` | Strictly vehicle-disjoint `K=10`/`K=20` partial-participation benchmark, Non-IID partitions, decentralized normalization, six method variants, personalization, and heterogeneity metrics. |
| `results/vehicle_swarm_benchmark.json` | Complete five-seed vehicle-swarm artifact: exact partitions, selected clients, round histories, global/client metrics, personalization, and heterogeneity aggregates. |
| `docs/vehicle_swarm.md` | Protocol, rounded benchmark tables, interpretation, caveats, and reproduction command for the vehicle swarm. |
| `vehicle_datasets.py` | `VehicleFleetDataset`, `build_flexible_cnn`, `CmapssCombinedAdapter`, `ScaniaComponentXAdapter`, `NCMAPSSAdapter`, and synthetic Scania schema-fixture generator. |
| `docs/DATASET_EVALUATION.md` | Dataset decision matrix, provenance, exact schemas, licensing/download notes, adapter contract, and evidence boundaries for C-MAPSS, Scania Component X, N-CMAPSS, and rejected/next-step alternatives. |
| `dashboard/server.py` | Standard-library static/API server: `/api/health`, `/api/edge_update`, `/api/live`. |
| `dashboard/generate_demo_data.py` | Offline demo snapshot generator; requires C-MAPSS data to regenerate. |
| `dashboard/index.html` | PoC Replay / Live Edge Stream interface, three tabs, English/Italian copy. |
| `dashboard/demo_data.json` | Committed snapshot used for offline PoC replay. |
| `scripts/preflight_check.py` | Artifact, symlink, JSON, privacy and optional test preflight. |
| `.github/workflows/ci.yml` | Python/Node CI test and preflight gate. |
| `docs/PILOT_RUNBOOK.md` | Pilot planning guide; currently stale about TLS/mTLS and federated normalization and should be reconciled in a separate docs change. |
| `tests/test_diagnostics.py` | Diagnostics unit and data-gated integration tests. |
| `tests/test_dashboard.py` | Dashboard generator, server, UI and browser tests. |
| `tests/test_diagnostics_validation.py` | Diagnostics validation, artifact and paper-tree checks. |
| `tests/test_system_smoke.py` | Model, loss, metrics, simulation, profiler and paper smoke tests. |
| `tests/test_federated_e2e.py` | Flower client, transport, DP-control and loopback tests. |
| `tests/test_federated_norm.py` | Federated statistics and telemetry sanitation tests. |
| `tests/test_edge_runtime.py` | Streaming buffer, alerting and source-to-sink tests. |
| `tests/test_preflight.py` | Preflight checks. |
| `tests/test_vehicle_swarm.py` | Vehicle partition, heterogeneity, participation, decentralized normalization, method, and real-data smoke tests. |
| `tests/test_vehicle_datasets.py` | Flexible-CNN and combined C-MAPSS/Scania/N-CMAPSS adapter tests. |
| `paper/main.tex`, `paper/sections/*.tex` | IEEE manuscript and its method/results. Some security/preprocessing descriptions have not caught up with code. |
| `paper/make_figures.py`, `paper/generated/` | Reproducible macros/tables generated from committed artifacts. |
| `docs/` | Project and pilot documentation; consult §8.2 for known drift. |
| `results/`, `experiments/` | Research inputs and outputs; unchanged by this specification. |

### 8.2 Implemented capabilities and remaining limits

| Area | Current status |
|---|---|
| **mTLS / transport** | Implemented in the Flower path. Server TLS and client-authentication are required by default when not using explicit `--insecure`; server-only TLS is an explicit opt-out. No silent plaintext fallback. |
| **Update privacy** | Optional full-client-delta L2 clipping and Gaussian noise are implemented. There is no formal epsilon/delta accountant, no configured privacy budget, and no secure aggregation. The coordinator can see individual client updates. |
| **Federated normalization** | `fed_norm.py` implements tested Chan/Welford sufficient-statistics aggregation, transforms and sanitation, and the vehicle-swarm benchmark uses it for decentralized fleet normalization. The production-like Flower client, paired experiment suite, and current edge runner still use legacy centralized scales; operational integration remains open. Statistics themselves can reveal information and are not protected by secure aggregation. |
| **Streaming inference** | JSONL, CSV replay and optional MQTT source adapters, bounded rolling windows, local inference, diagnostics and severity hysteresis are implemented. Plant-specific PLC/OPC-UA acquisition, calibrated cadence and deployment supervision are not included. |
| **Live dashboard API** | Summary POST and live GET endpoints are implemented. The API is in-memory, has no authentication or persistence, and should be deployed only on a suitably restricted interface/network. Raw telemetry field names are rejected. |
| **CI / preflight** | `.github/workflows/ci.yml` runs the unit suite and preflight on master pushes and pull requests. `scripts/preflight_check.py` checks symlinks and pinned artifact/schema invariants. |
| **C-MAPSS vs plant data** | C-MAPSS is a public turbofan benchmark, not a partner plant dataset. Asset schema mapping, time/cadence calibration, data quality policy and validation on real plant failures remain necessary. No automatic shutdown/control action is implemented or appropriate. |
| **Hardware evidence** | Physical Flower runs on Raspberry Pi clients exist in committed research artifacts, but `profile_edge.py` and `diagnostics.py` do not have a measured Pi 4 profile. Runtime latency, energy, thermals, and transport overhead on deployment hardware remain unmeasured. |
| **Flower API lifecycle** | The code still uses Flower's `fl.server.start_server` and `fl.client.start_client` entry points. Migration from the deprecated server/client API remains open. |
| **Scale of federation** | Physical evidence remains two Raspberry Pi 4 clients for five rounds. The in-process vehicle-swarm benchmark adds `K=10` with 50% participation and `K=20` with 30% participation, five rounds and five seeds, with whole-vehicle Non-IID clients and exact dropout-by-design sampling. It does not exercise network transport, availability failures, contention, or secure aggregation; it is not evidence of 10 or 20 physical customer sites. SCAFFOLD, clustered FL, longer-round sweeps, and personalization fairness controls remain open. |
| **Dataset generality** | The measured results remain C-MAPSS simulations. `CmapssCombinedAdapter` supports the 709-engine combined training fleet and 707-engine official test fleet; Scania Component X and N-CMAPSS adapters plus `build_flexible_cnn` are implemented and schema-tested. The external datasets were not downloaded or benchmarked, so no real-truck or N-CMAPSS accuracy/generalization claim is supported. Cross-schema federation is not implemented. |
| **Research scope** | Five-round experiments and five-seed aggregates do not establish broad industrial generalization or state-of-the-art performance. `FedAdam` tuning, formal privacy accounting, secure aggregation, networked larger-federation trials, longer horizons, and labeled component-attribution evaluation remain research gaps. |
| **Documentation drift** | `docs/PILOT_RUNBOOK.md` still says TLS/mTLS and federated normalization must be added. Several manuscript paragraphs also say TLS, DP, and federated normalization are absent. The code and this spec describe current implementation; those other files need their own review/update. |

## 9. Verification record for this edit

This specification and `docs/PRESENTATION_ROADMAP.md` were checked against the repository source snapshot named at the top. Static source inventory contains **10 test files, 51 test classes, and 152 `test_*` methods**. `python3 scripts/preflight_check.py --skip-unittest` passed: **0 symlinks**, protected artifacts match their canonical SHA-256 values, 1,798 dashboard attribution rows sum to 100.0, 9 privacy fields are zero, and diagnostics report 0 socket calls and 0 raw sensor bytes sent. C-MAPSS was not configured, and unit tests were intentionally skipped by the requested preflight flag. The working diff contains only `PROJECT_SPEC.md` and `docs/PRESENTATION_ROADMAP.md`.
