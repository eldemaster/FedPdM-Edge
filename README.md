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

## Repository layout

```text
common.py                    Existing model and C-MAPSS helpers
client.py / server.py        Earlier Flower client/server demo
experiments/reproduce.py     Paired FD001–FD004 experiments
experiments/profile_edge.py  Edge training/RSS/payload profiler
results/                     Raw results and experiment log
paper/                       Manuscript source and generated tables
```

## Paper

Build from `paper/` with `make`. `python paper/make_figures.py` generates manuscript tables from `results/reproducible/summary.csv`; only completed raw runs should appear in that file.

## Project

Developed within the EU Interreg Aurora project ResilientEdge (grant 20373282), with Mid Sweden University and the University of Oulu.
