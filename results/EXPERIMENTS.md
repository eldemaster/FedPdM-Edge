# Experiment record

## Reproducible suite

The current runner is `experiments/reproduce.py`. It uses the official C-MAPSS files and holds a fixed preprocessing path (training-set global min/max, 30-cycle windows, left-padding short test sequences with their first row, 125-cycle target cap) across centralized, local-only and federated methods. FedAvg, FedProx, FedAdam, asymmetric loss, and a compact CNN-LSTM are evaluated on the same seed-specific initialization and client partition. Raw runs include global and per-client RMSE and NASA score. The summary reports mean, sample standard deviation, and paired exact sign-flip p-values.

Paired tests compare seed-matched means under sign-exchangeability of the differences under the null. With five seeds, the minimum two-sided exact p-value is 0.0625. Tests are exploratory; no multiple-comparison adjustment is applied.

Coverage is deliberate: the three required baselines run on IID splits of all four datasets; FedProx and FedAdam run on FD002/FD004 regime and engine-count-imbalanced splits; score-aware loss and CNN-LSTM are evaluated on FD002/FD004 regime splits. Missing method/partition combinations are not imputed.

Run command:

```bash
python experiments/reproduce.py --data /path/to/CMAPSSData.zip \
  --out results/reproducible --threads 1
```

The default matrix is FD001–FD004, seeds `42 100 2024 7 777`, and partitions IID, regime, and engine-count imbalance. Regime partitions apply only to FD002 and FD004. They assign each window to a client using the endpoint operating condition, derived from six k-means clusters over cycle-level settings. An engine can therefore contribute windows to both clients; this partition tests operating-regime shift and does not model disjoint plant fleets. FD003 and FD004 are run over their complete datasets, which contain both fault modes.

To recompute `summary.csv` from all completed top-level raw JSON files (for example, after parallel dataset runs), use `python experiments/reproduce.py --summarize-only`.

## Runs and status

- The first one-seed smoke output used a 32/48-channel CNN and was superseded when the runner was aligned with the repository's 32/64-channel model. It is quarantined under `results/reproducible/interrupted/pre_32-64-cnn-alignment/` and excluded from summaries. `--quick` now writes into a separate `smoke/` subdirectory so it cannot overwrite the full result matrix.
- An initial partial run exposed a false regime partition for single-regime FD001; inspecting FD002 showed that per-engine medians would also yield an extreme 17/243 engine split. Those first outputs are quarantined under `results/reproducible/interrupted/seed42_pre_regime-fix/`. A second partial run was stopped when the final method-by-partition coverage was narrowed; its outputs are under `results/reproducible/interrupted/pre_variant-filtering/`. Both directories are excluded from summaries.
- A later audit found the loader omitted short official test sequences. All pre-padding runs and their summary are quarantined under `results/reproducible/interrupted/pre_short-test-padding/` and excluded from study tables. The current run left-pads those test sequences so every official test engine is scored.
- The corrected matrix completed: 40 raw JSON files cover five seeds for every planned dataset/partition condition. The top-level summary excludes all quarantined and smoke outputs, and every run scores the complete official test population (FD001 100, FD002 259, FD003 100, FD004 248 engines).
- IID centralized / local-only / FedAvg global RMSE means are respectively 14.85 / 19.80 / 19.27 on FD001, 22.12 / 23.35 / 23.33 on FD002, 18.00 / 21.98 / 21.60 on FD003, and 23.91 / 28.43 / 27.07 on FD004. Full mean ± sample standard deviation values are in `summary.csv` and README.
- On regime partitions, FedProx was close to FedAvg. Tested FedAdam had much larger error (RMSE 78.23 on FD002, 56.62 on FD004) under the fixed server learning rate. The asymmetric objective reduced mean NASA score relative to FedAvg on both datasets while increasing RMSE. The five-seed exact paired tests had minimum attainable two-sided p=0.0625 and no comparison was below 0.05.
- The edge profiler passed a one-round, one-epoch smoke run on an Apple Silicon Mac (FD002 regime, client 0): 3.646 s training time, 426,196,992-byte peak process RSS, and 178,184 model-tensor payload bytes for downlink plus uplink. Its raw output is `results/edge/local-smoke-macos.json`; it is not a Raspberry Pi measurement and is excluded from edge claims.

## Edge measurement

`experiments/profile_edge.py` records one-client training time per round, peak process RSS, and the tensor payload for one model download plus one update upload. Payload excludes gRPC framing and is not a wire capture. This code path does not enable TLS.

The remote helpers were attempted. The PC helper returned a Tailscale authentication check and did not run a workload. Both Pi helpers remained unreachable and exited with SSH status 255. No Raspberry Pi timing or memory measurement is claimed until JSON files from those devices exist under `results/edge/`.

The algorithm comparison uses a deterministic local PyTorch simulation of client updates; it does not launch a Flower server or measure actual network traffic. The tensor payload is calculated from the model parameter shapes and dtypes.

## Environment

The local run used Python 3.13.2, PyTorch 2.15.0, pandas 3.1.1, and NumPy 2.5.4 in a temporary environment outside the repository. The dataset archive was read from a local path and is not copied into this repository.

## Privacy

The Flower demo does not configure TLS/mTLS. The experiment runner simulates client updates locally; it does not implement encrypted transport, secure aggregation, or differential privacy. Only the data-flow statement that raw training windows are not serialized as client updates is supported.
