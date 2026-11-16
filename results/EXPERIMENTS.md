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

## Edge diagnostics validation (sensor/subsystem attribution)

`experiments/validate_diagnostics.py` validates `diagnostics.py`. It is separate from the reproducible suite and does not change `reproduce.py` or `summary.csv`.

Setup:
- Seeds 7, 42, 100, 777, 2024, on FD001–FD004.
- One isolated `CmapssRegressor` per seed, trained with `AsymmetricRULLoss` for 8 epochs on 80% of training units.
- Raw output: `results/diagnostics_validation.json`. Method and full tables: `docs/diagnostics.md`.

C-MAPSS has no per-unit component labels, so no supervised component accuracy is reported.

- **FD001/FD002, windows with true RUL < 30:**
  - Attribution puts the core subsystem first in 99.4 ± 0.8% / 100.0% of windows. Core = HPC ∪ combustor/turbine; chance by sensor count is 71.4%.
  - HPC is first in 26.6 ± 5.5% / 59.6 ± 5.0% of windows, against 35.7% chance.
  - Model-free drift ranks HPC first in 51.4% / 76.2% of windows.
- **FD003/FD004, k=2 clustering of end-of-life drift profiles:**
  - Silhouette 0.51 / 0.49. Cross-seed ARI 1.00 / 0.99.
  - The HPC-like cluster matches the FD001/FD002 HPC drift signature (cosine 0.98 / 1.00).
  - The other cluster reverses the P30/phi/bleed drift direction and lowers BPR (fan-mode candidate, not ground truth).
  - Attribution-only clusters are stable on FD003 (ARI 1.00) but not on FD004 (0.70 ± 0.31).
- **Edge overhead:**
  - Measured on the Apple Silicon host, single thread, not on a Raspberry Pi.
  - Inference alone is 0.05–0.07 ms per window. Inference + attribution + drift + summary is 0.49–0.93 ms per window. Autograd buffers are 117 KB per window.
  - Socket instrumentation recorded 0 calls and 0 bytes during the diagnostics path.

## Edge measurement

`experiments/profile_edge.py` records one-client training time per round, peak process RSS, and the tensor payload for one model download plus one update upload. Payload excludes gRPC framing and is not a wire capture. This code path does not enable TLS.

The remote helpers were attempted. The PC helper returned a Tailscale authentication check and did not run a workload. Both Pi helpers remained unreachable and exited with SSH status 255. No Raspberry Pi timing or memory measurement is claimed until JSON files from those devices exist under `results/edge/`.

The algorithm comparison uses a deterministic local PyTorch simulation of client updates; it does not launch a Flower server or measure actual network traffic. The tensor payload is calculated from the model parameter shapes and dtypes.

## Environment

The local run used Python 3.13.2, PyTorch 2.15.0, pandas 3.1.1, and NumPy 2.5.4 in a temporary environment outside the repository. The dataset archive was read from a local path and is not copied into this repository.

## Privacy

The Flower demo does not configure TLS/mTLS. The experiment runner simulates client updates locally; it does not implement encrypted transport, secure aggregation, or differential privacy. Only the data-flow statement that raw training windows are not serialized as client updates is supported.

---

## Earlier FD002 physical and local runs (2026-11-12, official `test_FD002`, all 259 engines, RUL cap 125)

Seeds: 42, 100, 2024, 7, 777. Data: CMAPSSData.zip (not committed).

## Real, on physical nodes (2026-11-12)
Server: Ubuntu PC (`eldemaster-ubuntu`, flwr 1.37). Clients: RPi `edgedataspace` (client 1), `edgedataspace2` (client 2), Cortex-A72, 3.7 GiB, torch 2.14 cpu.
`PROXIMAL_MU=0 ROUNDS=5 ./run_fd002_federated.sh` -> `results/fd002/results_physical_seed*.json`, logs `server_mu0_*.log`.
Result FedAvg: RMSE 22.27 ± 0.44, NASA 6037 ± 1030. Train time/round: RPi1 12.8 s, RPi2 17.3 s (seed 42 client logs); peak RSS 570-620 MB.

## Local (Mac, in-process, `fed_sim.py` / `final_benchmark.py --epochs 10`)
- `results/fd002/benchmark_e10.{json,csv}`: baseline 43.51; local 25.01 ± 0.54; centralized 20.39 ± 1.11.
- `sim_fedavg_r5_e2.json` 22.20 ± 0.42 (matches physical run); `sim_fedprox0.01_r5_e2.json` 22.20 ± 0.58; `sim_fedprox0.1_r5_e2.json` 22.86 ± 0.93; `sim_fedavg_r15_e2.json` 19.27 ± 1.30 NASA 2742.
- Commands: `python fed_sim.py --mu M --rounds R --out ...`, `python final_benchmark.py --epochs 10`.

## Started / NOT finished (no numbers claimed)
- Physical 15-round run (`ROUNDS=15 RESULTS_DIR=results/fd002_r15`) was launched on the PC; results not collected.
- `final_benchmark.py --epochs 30` and `fed_sim.py --rounds 30` were running locally; not collected.
- FD001 not re-run. FedProx not run on physical nodes. Early stopping by client val RMSE: no gain in sim (19.05 vs 19.27, from r15 history).
- Old pre-fix results kept in `results/fd002/legacy_pre_fix/` (evaluated 253 of 259 engines).
