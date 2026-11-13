# FedPdM-Edge

Research code for federated Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan data. The repository includes a Flower deployment demo and a paired PyTorch experiment suite.

## Scope and privacy

In the Flower demo, clients train on local windows and send model parameters to the coordinator. The paired benchmark simulates client updates in one local process and does not measure a Flower network session. Keeping raw windows out of the parameter exchange does not prevent leakage from model updates. The checked-in implementation does **not** enable TLS/mTLS, encrypted weights, differential privacy, or secure aggregation. It makes no formal privacy guarantee. The scaler in the research runner is fitted on all training engines, so the preprocessing step is centralized.

The evaluated datasets are FD001–FD004. Targets use the conventional 125-cycle cap and the official held-out test files. The NASA asymmetric score is reported alongside RMSE. See [results/EXPERIMENTS.md](results/EXPERIMENTS.md) for the exact runs, raw outputs, and unavailable hardware. Legacy FD001/FD002 files remain under `results/fd001/` and `results/fd002/`; they are not combined with the paired experiment suite.

## Experiments

`experiments/reproduce.py` contains the shared preprocessing and training pipeline for:

- Centralized, per-client local-only, and FedAvg baselines.
- FedProx and FedAdam server optimization.
- MSE versus an asymmetric NASA-score-aware loss, plus a small CNN-LSTM comparison.
- Five paired seeds and IID, regime-grouped window (FD002/FD004), and engine-count-imbalanced partitions.
- Global and client-shard RMSE/NASA score, and exact paired sign-flip tests in the summary CSV.

For FD002/FD004, the regime split groups windows by the nearest of six operating-condition clusters at each window endpoint. A single engine can contribute windows to both clients, so this split tests regime shift rather than disjoint plant fleets.

The centralized/local/FedAvg baselines run on IID partitions for all four datasets. FedProx and FedAdam are added on the FD002/FD004 regime and engine-count-imbalanced partitions; the score-aware loss and CNN-LSTM are also compared on those regime partitions. This coverage is reflected in the summary rather than imputed across missing method/partition combinations.

## Measured results

All entries below are mean ± sample standard deviation over five paired seeds. Lower is better for both RMSE and the NASA score. The NASA score is summed over each dataset's official test engines.

| Dataset | Method | RMSE | NASA score |
|---|---|---:|---:|
| FD001 | Centralized | 14.85 ± 0.97 | 439.85 ± 139.24 |
| FD001 | Local-only | 19.80 ± 0.83 | 1,334.44 ± 328.37 |
| FD001 | FedAvg | 19.27 ± 1.14 | 1,063.17 ± 268.40 |
| FD002 | Centralized | 22.12 ± 1.78 | 4,180.30 ± 2,858.02 |
| FD002 | Local-only | 23.35 ± 0.41 | 5,283.19 ± 443.40 |
| FD002 | FedAvg | 23.33 ± 0.24 | 4,179.66 ± 680.06 |
| FD003 | Centralized | 18.00 ± 0.85 | 1,093.74 ± 311.96 |
| FD003 | Local-only | 21.98 ± 0.30 | 3,368.98 ± 477.38 |
| FD003 | FedAvg | 21.60 ± 0.55 | 2,959.02 ± 559.00 |
| FD004 | Centralized | 23.91 ± 0.81 | 4,758.71 ± 1,169.51 |
| FD004 | Local-only | 28.43 ± 1.49 | 15,523.43 ± 8,611.70 |
| FD004 | FedAvg | 27.07 ± 0.56 | 8,059.20 ± 2,868.94 |

On the regime-grouped partitions, FD002 FedAvg reached 23.50 ± 0.63 RMSE / 4,425.37 ± 1,367.59 NASA score; FedProx reached 23.30 ± 0.48 / 4,581.90 ± 1,331.95. FD004 FedAvg reached 26.78 ± 0.20 / 7,654.82 ± 693.72; FedProx reached 26.88 ± 0.81 / 7,638.47 ± 1,490.37. FedAdam with the tested server learning rate degraded sharply (RMSE 78.23 on FD002 and 56.62 on FD004). The asymmetric objective reduced the mean NASA score relative to FedAvg in both datasets while increasing RMSE. With five seeds, the exact two-sided paired sign-flip test's minimum p-value is 0.0625; no tested comparison reached p < 0.05. Full partition, per-client, and algorithm tables are in [results/reproducible/summary.csv](results/reproducible/summary.csv).

Install the versions used by the local run, then execute the full matrix:

```bash
python -m pip install -r experiments/requirements.txt
python experiments/reproduce.py --data /path/to/CMAPSSData.zip
```

The runner writes raw JSON per dataset/partition/seed and `summary.csv` under `results/reproducible/`. A smoke run is available with `--quick`; it is for pipeline validation and must not be presented as a five-seed result.

After running disjoint dataset groups in parallel or resuming a partial run, rebuild the aggregate summary with:

```bash
python experiments/reproduce.py --summarize-only
```

For Raspberry Pi profiling, run the following on each device with the same dataset and Python environment:

```bash
python experiments/profile_edge.py \
  --data /path/to/CMAPSSData.zip --dataset FD002 \
  --partition regime --client 0 --output results/edge/rpi_fd002.json
```

The profiler records per-round CPU training time, peak process RSS, and the exact model-tensor payload for one downlink and one uplink. The byte count excludes gRPC framing and transport overhead. It does not imply that transport encryption is enabled.

## Existing Flower demo

The root-level `server.py`, `client.py`, and `run_fd002_federated.sh` are the earlier Flower demo. Their configuration and archived results predate the paired runner. They are useful for deployment experiments but should not be used as the sole basis for the tables in a new study.

### NASA C-MAPSS FD002 (official test set, all 259 engines)
*5 seeds, mean ± std, 10 local epochs / 5 federated rounds x 2 epochs; measured, see `results/EXPERIMENTS.md`*:

The underlying FD002 dataset contains six operating conditions, but the archived client partition provenance remains unresolved; see `docs/literature_review.md`. Do not interpret these results as an operating-condition partition until the assignments are recovered and recorded.

| Architecture | Test RMSE | NASA Score |
| :--- | :---: | :---: |
| Constant Baseline | 43.51 | 63,730 |
| Isolated Local Edge | 25.01 ± 0.54 | 11,937 ± 2,207 |
| FedAvg, physical 2xRPi4 + Ubuntu server | 22.27 ± 0.44 | 6,037 ± 1,030 |
| Centralized (10 epochs) | 20.39 ± 1.11 | 3,395 ± 972 |

In-process simulation (same training code), FedAvg 5 rounds 22.20 ± 0.42; FedProx mu=0.01 22.20 ± 0.58, mu=0.1 22.86 ± 0.93 (no gain); FedAvg 15 rounds 19.27 ± 1.30 (not yet compared with an equal-epoch centralized run).
Per-node cost (physical, 2 local epochs/round): RPi1 ~12.8 s (16.7k windows), RPi2 ~17.3 s (25.0k windows), peak RSS ~570-620 MB.

> The FD001 table above is from the pre-audit pipeline and was **not re-run** here.

### Asymmetric Risk-Aware Training Loss (`mse` vs. `asymmetric`)
To align local and federated optimization with the safety-critical NASA Asymmetric Scoring Function, `common.py` provides `AsymmetricRULLoss(alpha_late=2.5, alpha_early=0.5)` (selectable via `LOSS_FN=asymmetric` or `--loss-fn asymmetric` in `client.py`, `final_benchmark.py`, and `fed_sim.py`; default remains `mse`). It applies a $C^1$-smooth asymmetric quadratic penalty ($\alpha_{\text{late}} d^2$ for late predictions $d = \hat{y} - y \ge 0$ vs. $\alpha_{\text{early}} d^2$ for early predictions $d < 0$), avoiding the gradient explosion of raw exponential objectives.

*Controlled 5-seed comparison (`42, 123, 456, 789, 1024`), 10 local epochs (FedAvg: 5 rounds × 2 epochs), recorded in `results/asymmetric_loss_comparison.json` (`benchmark_asymmetric_loss.py`):*

| Dataset | Setup | Loss | Test RMSE (Cycles) | NASA Score | Late-Prediction Rate ($\hat{y} > y$) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **FD001** (100 engines) | Isolated Local | `mse` | 17.58 ± 1.01 | 722.9 ± 120.5 | 56.2% ± 6.6% |
| **FD001** (100 engines) | Isolated Local | `asymmetric` | **17.15 ± 1.50** | **435.0 ± 85.6** (-39.8%) | **33.8% ± 4.5%** |
| **FD001** (100 engines) | FedAvg (5r × 2e) | `mse` | 16.98 ± 1.31 | 690.8 ± 173.4 | 63.8% ± 7.1% |
| **FD001** (100 engines) | FedAvg (5r × 2e) | `asymmetric` | **16.25 ± 1.23** | **369.1 ± 43.5** (-46.6%) | **34.2% ± 8.0%** |
| **FD001** (100 engines) | Centralized (10e) | `mse` | **15.16 ± 0.86** | 450.4 ± 91.8 | 56.6% ± 3.3% |
| **FD001** (100 engines) | Centralized (10e) | `asymmetric` | 15.80 ± 0.97 | **338.7 ± 52.0** (-24.8%) | **30.8% ± 5.2%** |
| **FD002** (259 engines) | Isolated Local | `mse` | **23.73 ± 1.22** | 7,610.8 ± 3,611.4 | 50.3% ± 7.8% |
| **FD002** (259 engines) | Isolated Local | `asymmetric` | 26.30 ± 2.08 | **3,932.2 ± 829.7** (-48.3%) | **29.5% ± 6.0%** |
| **FD002** (259 engines) | FedAvg (5r × 2e) | `mse` | **22.33 ± 1.46** | 5,499.1 ± 2,962.8 | 48.9% ± 8.7% |
| **FD002** (259 engines) | FedAvg (5r × 2e) | `asymmetric` | 26.07 ± 1.69 | **3,412.6 ± 429.8** (-38.0%) | **25.3% ± 6.3%** |
| **FD002** (259 engines) | Centralized (10e) | `mse` | **20.17 ± 0.31** | **2,769.5 ± 494.6** | 40.8% ± 5.0% |
| **FD002** (259 engines) | Centralized (10e) | `asymmetric` | 25.79 ± 2.74 | 3,630.2 ± 1,185.2 (+31.1%) | **20.5% ± 6.1%** |

> **Findings:** `AsymmetricRULLoss` consistently cuts the late-prediction rate roughly in half across all setups and datasets (from 40.8%–63.8% down to 20.5%–34.2%) and sharply reduces NASA Score mean and variance in decentralized settings (FD001 FedAvg: -46.6%, FD002 FedAvg: -38.0% with std dropping from 2,962.8 to 429.8). On FD002, this safety margin trades off higher symmetric RMSE (+10.8% to +27.9%) due to conservative early-prediction bias.

## Repository layout

```text
common.py                            Existing model, AsymmetricRULLoss, and C-MAPSS helpers
client.py / server.py                Earlier Flower client/server demo
fed_sim.py / final_benchmark.py      In-process Flower-parity simulation and 4-way benchmark
benchmark_asymmetric_loss.py         Controlled 5-seed MSE vs. AsymmetricRULLoss benchmark
compute_global_scale.py              Global min-max scaling across operating conditions
run_fd002_federated.sh               Multi-client local/remote test runner
sanity_check.py                      Verification gate for model convergence and data pipelines
experiments/reproduce.py             Paired FD001–FD004 experiments
experiments/profile_edge.py          Edge training/RSS/payload profiler
docs/literature_review.md            Literature review and FD002 partition audit notes
results/                             Raw results, asymmetric loss comparison, and experiment log
paper/                               Manuscript source and generated tables
```

## Paper

Build from `paper/` with `make`. `python paper/make_figures.py` generates manuscript tables from `results/reproducible/summary.csv`; only completed raw runs should appear in that file.

## Project

Developed within the EU Interreg Aurora project ResilientEdge (grant 20373282), with Mid Sweden University and the University of Oulu.
