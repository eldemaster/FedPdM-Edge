# FedPdM-Edge — Technical Specification

One-file, end-to-end specification of the FedPdM-Edge repository: federated Remaining-Useful-Life (RUL) prediction on NASA C-MAPSS, a safety-oriented asymmetric loss, local edge diagnostics, an offline dashboard, the paper pipeline and the test suite.

**How to read the evidence labels**

| Label | Meaning |
|---|---|
| **MEASURED** | A number copied from a committed artifact (`results/**`, `dashboard/demo_data.json`) and cross-checked against that file. |
| **ANALYSIS** | Reasoning from the code or the maths, not a measurement. |
| **UNMEASURED / PoC** | Not measured, not implemented, or only a proof of concept. No claim is made. |

**Provenance.** Every path, symbol, dimension, formula, hyperparameter and number below was checked against the source on `master` (`6bf5e5b`): `common.py`, `client.py`, `server.py`, `fed_sim.py`, `experiments/*.py`, `diagnostics.py`, `dashboard/*`, `tests/*`, `paper/make_figures.py`, `results/*.json|csv`, `global_scale.json`, `partition.json`, and the prose in `README.md`, `docs/*.md`, `PDM_FIXES.md`, `results/EXPERIMENTS.md`. Parameter counts were recomputed by arithmetic (§4.1). Summary tables were compared with `results/reproducible/summary.csv`, `results/asymmetric_loss_comparison.json`, `results/diagnostics_validation.json`, `results/edge/local-smoke-macos.json` and `dashboard/demo_data.json`. No experiment or test was re-run to write this file. The C-MAPSS archive (`CMAPSSData.zip`) is not in the repository.

**Contents**

1. [Executive summary and why FedPdM-Edge exists](#1-executive-summary-and-why-fedpdm-edge-exists)
2. [System architecture and end-to-end data flow](#2-system-architecture-and-end-to-end-data-flow)
3. [Dataset and preprocessing specifications](#3-dataset-and-preprocessing-specifications-nasa-c-mapss)
4. [ML algorithms, model architectures and training objectives](#4-ml-algorithms-model-architectures-and-training-objectives)
5. [Local edge degradation attribution and drift (`diagnostics.py`)](#5-local-edge-degradation-attribution-and-drift-diagnosticspy)
6. [Dashboard, paper pipeline and test suite](#6-dashboard-paper-pipeline-and-test-suite)
7. [Repository file map, bug-fix evolution and known limitations](#7-repository-file-map-bug-fix-evolution-and-known-limitations)

---

## 1. Executive summary and why FedPdM-Edge exists

### 1.1 The problem

Industrial predictive maintenance (PdM) asks, per machine, "how many operating cycles remain before failure?" (RUL). Three constraints shape this project (`paper/sections/introduction.tex`, `README.md`):

1. **Telemetry is sensitive and failure data is scarce per site.** Each plant sees few run-to-failure trajectories. Pooling them would help accuracy, but raw per-cycle sensor streams are operationally sensitive. Federated learning (FL) keeps raw telemetry on the plant edge node. Only model parameters travel to a coordinator.
2. **Data locality is not privacy.** Model updates can still leak information. The repository implements **no** TLS/mTLS, differential privacy (DP) or secure aggregation. It claims only that raw training windows are not serialized in the parameter exchange (`README.md`, "Scope and privacy").
3. **Maintenance errors are asymmetric.** A *late* prediction (`ŷ > y`, RUL overestimated) can defer service past the real failure. An *early* prediction (`ŷ < y`) wastes useful life. RMSE alone hides this difference.

### 1.2 Why NASA C-MAPSS `FD001`–`FD004`

C-MAPSS is the standard public run-to-failure benchmark for turbofan RUL. The four subsets span the difficulty axes the project needs (conditions and fault modes as stated in the archive `readme.txt`, quoted in `docs/project_guide_part2.md` §1.3):

| Subset | Operating conditions | Fault modes | Why it is used here |
|---|---|---|---|
| FD001 | 1 | 1 (HPC degradation) | Easy, near-IID baseline. |
| FD002 | 6 | 1 (HPC) | Heterogeneous operating regimes, the natural non-IID case. |
| FD003 | 1 | 2 (HPC, Fan) | Two fault modes, used for unsupervised mode separation. |
| FD004 | 6 | 2 (HPC, Fan) | Hardest: both heterogeneity sources. |

It also defines the asymmetric NASA score (Saxena et al. 2008) that the project reports next to RMSE. **Boundary:** C-MAPSS publishes only a unit-level RUL target. There are no per-unit component or fault-mode labels, so no supervised component-diagnosis accuracy exists anywhere in this repository (§5.8).

### 1.3 Core capabilities

| Capability | Where | Status |
|---|---|---|
| Federated RUL prediction with a 22,273-parameter 1D-CNN (FedAvg, FedProx, FedAdam) | `common.py`, `client.py`, `server.py`, `fed_sim.py`, `experiments/reproduce.py` | Implemented. Real gRPC run on 2× Raspberry Pi 4 + Ubuntu server (FD002, 5 seeds); paired in-process suite on FD001–FD004. |
| Safety-critical asymmetric loss | `common.AsymmetricRULLoss` (`alpha_late=2.5`, `alpha_early=0.5`) | Implemented, default off (`mse` is the default). |
| Local subsystem/sensor diagnostics (attribution + healthy-baseline drift) | `diagnostics.py`, validated by `experiments/validate_diagnostics.py` | Implemented, validated on a development host. |
| Offline presentation dashboard | `dashboard/` | Implemented, standard-library server, no network dependency at run time. |

**Positioning.** The repository's own literature review states the contribution is an end-to-end physical testbed plus reproducibility, not algorithmic novelty or state-of-the-art accuracy (`docs/literature_review.md`, `paper/sections/related_work.tex`). Federated C-MAPSS, FedProx on RUL and personalized FL for RUL all have prior work.

---

## 2. System architecture and end-to-end data flow

### 2.1 Diagram

```
 PLANT / EDGE NODE (raw telemetry never leaves)                    COORDINATOR
 ┌────────────────────────────────────────────────────────────┐    ┌──────────────────────────────┐
 │ raw per-cycle rows: unit, cycle, setting1-3, s1..s21       │    │ server.py (Flower, gRPC)     │
 │        │ common.load_cmapss_raw (26 cols)                  │    │  strategy: FedAvg            │
 │        ▼                                                   │    │  seeded init weights         │
 │ 17 channels (USEFUL_SENSORS) → min-max scale               │    │  evaluate_fn: official test  │
 │        ▼                                                   │    │  set every round             │
 │ windows (30 × 17), target min(125, RUL)  [train]           │    └───────────▲──────────┬───────┘
 │ last window, left-padded if short        [test]            │                │ uplink   │ downlink
 │        ▼                                                   │  parameters    │ 89,092 B │ 89,092 B
 │ client.py SupervisedClient.fit  ──────────────────────────────────────────►│          │
 │   local epochs (Adam 1e-3, batch 64, optional FedProx)     │◄───────────────────────────┘
 │   loss: MSE (default) or AsymmetricRULLoss                 │
 │        ▼ global model weights received each round          │
 │ diagnostics.py  (local, no sockets)                        │
 │   attribution + drift → shares %, z-scores, EN/IT text     │
 │        ▼  aggregate JSON only                              │
 └────────────┬───────────────────────────────────────────────┘
              ▼  (offline, precomputed)
   dashboard/generate_demo_data.py → dashboard/demo_data.json
   dashboard/server.py (stdlib HTTP, 127.0.0.1:8765) → dashboard/index.html (3 tabs, IT/EN)
```

### 2.2 Walkthrough

1. **Raw telemetry.** `common.load_cmapss_raw` reads the first 26 whitespace-separated columns of a C-MAPSS file: `unit`, `cycle`, `setting1..3`, `s1..s21` (§3.1).
2. **Local windowing.** On each client, 17 selected channels are min-max scaled and cut into 30-cycle windows. Training targets are the RUL at the window's last cycle, capped at 125 (§3.2–3.4).
3. **Federated training.** `server.py` starts Flower with `FedAvg`. Each `client.py` loads the global weights, trains `local_epochs` (default 2) on its own windows, and returns weights plus its window count. The server averages them weighted by window count and scores the global model on the official test set every round.
4. **Local diagnostics.** With the global model and its own raw history, the edge node runs `diagnostics.py`: it computes per-sensor attribution shares, healthy-baseline drift and a bilingual maintenance summary. This code opens no sockets (§5.8).
5. **Presentation.** `dashboard/generate_demo_data.py` trains reference models offline, runs the diagnostics on 8 test engines and writes `dashboard/demo_data.json`. `dashboard/server.py` serves that file with `dashboard/index.html`.

**Two code paths, one model.** The *physical* path is `server.py` + `client.py` over gRPC. The *simulated* path is `fed_sim.py` (same local-training code, no network) and `experiments/reproduce.py` (a separate paired suite, no network). The README warns that the older Flower files predate the paired runner and should not be the sole basis for tables in a new study.

### 2.3 Network payload specification

| Quantity | Value | Source |
|---|---|---|
| Model parameters | 22,273 float32 | recomputed in §4.1 |
| Model tensor payload | **89,092 bytes** (22,273 × 4) | `results/edge/local-smoke-macos.json` `model_tensor_bytes` |
| Per round, downlink + uplink per client | **178,184 bytes** (2 × 89,092) | same file, `communication_tensor_payload_bytes_per_round` |
| Raw telemetry bytes shared | **0** — only parameters, a sample count and two timing/RSS scalars cross the link | `client.py::fit` return value |
| Fit metadata per round | `train_time_s`, `peak_rss_mb` | `client.py::fit` |
| TLS / mTLS | **Off** (`"mTLS": false`) | `profile_edge.py`, `reproduce.py` protocol, server log "SSL is disabled" |

The payload is computed from tensor shapes and dtypes (`2 * sum(p.numel() * p.element_size())` in `experiments/profile_edge.py`). It **excludes** gRPC framing and is not a wire capture. No wire-level traffic measurement exists (UNMEASURED).

---

## 3. Dataset and preprocessing specifications (NASA C-MAPSS)

### 3.1 Columns → 17 `USEFUL_SENSORS`

`common.load_cmapss_raw` names 26 columns: `['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]`.

```python
# common.py:9  (identical list: experiments/reproduce.py::SENSORS)
USEFUL_SENSORS = ['setting1','setting2','setting3','s2','s3','s4','s7','s8','s9',
                  's11','s12','s13','s14','s15','s17','s20','s21']        # 17 channels
SEQ_LEN = 30
MAX_RUL = 125
```

- **3 operating settings + 14 sensors = 17 model input channels.** `unit` and `cycle` are identifiers.
- Sensors **not** used: `s1, s5, s6, s10, s16, s18, s19`. The repository records no variance table or written rationale for dropping them. The usual reason is that these sensors are (near-)constant in C-MAPSS, but no file here computes that, so treat the list as a fixed design input.
- The settings are included "for heterogeneous operating conditions" (comment in `common.py`). They are model inputs but never fault carriers in diagnostics (§5.3).
- The dataset archive is located by `common.cmapss_zip_path()`: `$CMAPSS_ZIP`, then `./CMAPSSData.zip`, then `~/CMAPSSData.zip`; otherwise `FileNotFoundError`. `experiments/reproduce.py` requires `--data`.

### 3.2 Windowing and the RUL target

- **Training windows** (`common.extract_supervised_sequences`): stride-1 windows of `SEQ_LEN = 30` cycles. Target = `min(MAX_RUL, max_cycle − cycle_at_window_end)`. `experiments/reproduce.read_data` does the same with `sliding_window_view` and `np.minimum(CAP, max_cycle − ends)`.
- **Piecewise-linear target.** RUL is capped at `MAX_RUL = 125`: `y_t = min(125, T − t)`. The cap also applies to **test** labels (`extract_last_sequence` returns `min(MAX_RUL, true_rul)`).
- **Engines shorter than 30 cycles** give no training windows.
- Why 30 and 125 are standard protocol values; no ablation of either is in the repository.

### 3.3 Test set and short-engine left-padding

The official test evaluation scores **one terminal window per engine** (`common.build_test_set`). `extract_last_sequence` handles short engines:

```python
if len(data) < SEQ_LEN:
    pad = np.repeat(data[:1], SEQ_LEN - len(data), axis=0)   # repeat first reading
    data = np.concatenate([pad, data], axis=0)
seq = data[-SEQ_LEN:]
```

Short test engines were once silently dropped, biasing the metric to longer-lived engines. The audit fix keeps every engine (`results/EXPERIMENTS.md`). Engines now scored: **FD001 100, FD002 259, FD003 100, FD004 248**. `sanity_check.py` asserts `n == 259` for FD002. The pre-fix FD002 Flower results (253 of 259 engines) are kept under `results/fd002/legacy_pre_fix/`.

### 3.4 Unit-level train/validation split (`split_units`)

```python
def split_units(units, val_frac=0.1, seed=0):   # common.py
    # sorts + shuffles unit IDs with RandomState(seed); holds out max(1, round(val_frac*n)) units
```

The split is by **engine**, not by window: overlapping 30-cycle windows from one engine are highly correlated, so a window-level split would leak near-duplicates into validation. Used by `client.py`, `fed_sim.py`, `final_benchmark.py`, `benchmark_asymmetric_loss.py` and `dashboard/generate_demo_data.py` (`val_frac=0.1`). `experiments/validate_diagnostics.py` uses `val_frac=0.2`. The paired suite (`reproduce.py`) has **no validation split** and no validation-based hyperparameter selection.

### 3.5 Min-max scaling: two procedures

| | Flower demo, controlled loss study, dashboard FD002 | Paired suite (`reproduce.py`, `profile_edge.py`) |
|---|---|---|
| Producer | `compute_global_scale.py` → `global_scale.json` (FD002). `benchmark_asymmetric_loss.compute_scales_from_train` (FD001, validation, dashboard FD001). | Inline in `read_data` / `profile_edge.load_client` |
| Rows | **First 50 cycles of every training engine** (`groupby('unit').head(50)`); test data never used | **All rows of the training split** (`tr[SENSORS].min()/max()`) |
| Formula | `(x − min) / (max − min + 1e-7)` (`common.apply_scale`, no clipping) | same, then `.clip(-5, 5)` |
| Extra | `train_mean_rul` (mean capped target over all training windows), used for the constant baseline. In `global_scale.json`: `80.74004197407993` | — |

`global_scale.json` holds min/max for all 17 channels (e.g. `setting3` min 60.0, `s9` max 9081.2). Because the first-50-cycle scaler sees only healthy readings, degraded readings can exceed [0, 1].

**PoC caveat.** Both procedures compute statistics over **all** training engines in one place, and `deploy.sh` rsyncs `global_scale.json` to both Pis. Preprocessing is centralized. Federated or per-client normalization is not implemented.

### 3.6 Partition schemes

**Paired suite** (`reproduce.partitions(train, seed)`; always called with `seed=0`, so the split is the same for all five seeds — seeds change only initialization and data order):

| Scheme | Construction | Datasets |
|---|---|---|
| `iid` | Shuffle unit IDs with `default_rng(0)`, alternate them (`[::2]`, `[1::2]`). Test engines alternate by sorted ID. | FD001–FD004 |
| `engine_imbalance` | First `len//4` shuffled engines → client 0, the rest → client 1 (1:3 engine ratio, engines kept whole). Test IDs use a separately shuffled 1:3 cut. | FD002, FD004 only |
| `regime` | See below. | FD002, FD004 only |

**`regime` (`k = 6`).** `operating_clusters(frame, k=6)` z-scores the cycle-level `setting1..3` and fits a deterministic k-means: the first center is the row with minimum z-scored `setting1`, each next center is the row farthest from the existing ones (farthest-point init), then up to 50 Lloyd iterations. `window_regime_data` labels each *window* by the nearest center at its **endpoint** (`x[:, -1, :3]`) and groups windows into pseudo-units with key `-(unit*6 + regime + 1)` (`decode_regime_key` inverts it). Centers are sorted lexicographically: the first 3 regimes go to client 0, the last 3 to client 1.

**MEASURED consequence (seed 42 raw JSON):** every training engine contributes windows to both clients. FD002 `client_engine_overlap` = 260, `client_training_windows` = [20840, 25379]; FD004 overlap = 249, windows = [24250, 29778]. The split therefore tests regime shift, **not** disjoint plant fleets (`README.md`). It is undefined for FD001/FD003 (skipped).

**Flower demo** (`partition.json`): keys `"1"` and `"2"`, 130 FD002 unit IDs each. Its provenance is unresolved: `setup_fd002.py` would write a lifespan-sorted split, while logs describe an operating-condition split (`docs/literature_review.md`, `README.md`). Do not read the physical results as an operating-condition partition. `benchmark_asymmetric_loss.py` and the dashboard use `partition.json` for FD002 and an **odd/even unit-ID split** for FD001.

---

## 4. ML algorithms, model architectures and training objectives

### 4.1 Primary model: `CmapssRegressor` (`common.py`) ≡ `CNN` (`experiments/reproduce.py`)

Input `(B, 30, 17)`; `forward` transposes to `(B, 17, 30)` for `Conv1d`.

| Layer | Definition | Output | Parameters |
|---|---|---|---:|
| `conv1` + ReLU | `Conv1d(17, 32, kernel_size=3, padding=1)` | (B, 32, 30) | 1,664 |
| `pool1` | `MaxPool1d(2)` | (B, 32, 15) | 0 |
| `conv2` + ReLU | `Conv1d(32, 64, kernel_size=3, padding=1)` | (B, 64, 15) | 6,208 |
| `pool2` | `MaxPool1d(2)` (floor) | (B, 64, 7) | 0 |
| `flatten` | | (B, **448**) | 0 |
| `fc1` + ReLU | `Linear(448, 32)` | (B, 32) | 14,368 |
| `fc2` | `Linear(32, 1)`, then `.squeeze(-1)` | (B,) | 33 |
| **Total** | | | **22,273** |

No BatchNorm or dropout, so `state_dict()` (what Flower sends) and `parameters()` hold the same tensors. 22,273 × 4 B = 89,092 B.

**Training defaults.** Adam `lr=1e-3`. Batch size **64** in `common.make_loader` (Flower demo, `fed_sim.py`, loss study, dashboard) and **128** in `reproduce.py`. `client.py` uses 2 local epochs per round by default (server config `LOCAL_EPOCHS`).

**Edge-suitability is a design choice, not a measurement.** No file compares 32/64 channels against Pi latency or energy. `docs/literature_review.md`: claim "physically deployed on Raspberry Pi clients", not "resource-efficient", until measured.

### 4.2 Architecture ablation: `CNNLSTM` (`experiments/reproduce.py`)

```python
self.conv = nn.Sequential(nn.Conv1d(17, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))   # (B,16,15)
self.lstm = nn.LSTM(16, 24, batch_first=True)
self.head = nn.Linear(24, 1)                                                              # on z[:, -1]
```

Parameters: conv 832 + LSTM 4·24·(16+24) + 2·4·24 = 4,032 + head 25 = **4,889**. It runs only as a *centralized* ablation on the FD002/FD004 regime partitions (10 epochs, MSE).

**MEASURED (`summary.csv`, n=5):** FD002 RMSE **43.14 ± 0.02**, NASA 25,313.76 ± 294.55. FD004 RMSE **43.00 ± 0.01**, NASA 43,970.10 ± 652.10. Versus the 1D-CNN centralized runs (21.46 and 23.41 RMSE) it is far worse. The near-zero cross-seed std at RMSE ≈ 43 suggests a near-constant predictor under this 10-epoch budget (ANALYSIS; the repository does not diagnose it further). The paper reports only that it "performs worse than the 1D-CNN under this training budget".

### 4.3 Loss functions

Let `d = ŷ − y`. Late: `d ≥ 0`. Early: `d < 0`.

#### 4.3.1 `MSELoss`

`nn.MSELoss` — the default everywhere (`get_loss_fn("mse")`). Symmetric: it weights `+d` and `−d` equally, so it cannot prefer conservative errors.

#### 4.3.2 NASA score (evaluation metric)

```
S = Σ_i  ( exp(−d_i/13) − 1   if d_i < 0
         ( exp( d_i/10) − 1   if d_i ≥ 0
```

(`common.nasa_score`; `reproduce.nasa` uses `expm1`, equivalent.) Late errors use the smaller divisor (10), so they grow faster. `S` is a sum, so it is not comparable across test sets of different size. `common.late_prediction_rate` = mean of `ŷ > y`.

**Why late predictions are dangerous.** `ŷ > y` means the model claims more remaining life than exists, so maintenance may be scheduled after the failure. The metric makes the same judgement: at `|d| = 20`, a late error costs `e² − 1 ≈ 6.4` and an early one `e^(20/13) − 1 ≈ 3.7`.

#### 4.3.3 Capped exponential NASA-aligned loss (paired suite only)

In `reproduce.train(..., loss_kind='asymmetric')`:

```python
loss = where(d < 0, exp(clamp(-d/13, max=20)) - 1,
                    exp(clamp( d/10, max=20)) - 1).mean()
```

It trains directly on the NASA score; the `clamp(max=20)` caps the exponent. Used by the variants `asymmetric` (centralized) and `federated_asymmetric` (FedAvg), both only on FD002/FD004 regime partitions. The paper labels them "Central + exp." and "FedAvg + exp." (`paper/make_figures.METHOD_LABELS`).

#### 4.3.4 `AsymmetricRULLoss` (`common.py`)

```python
d = y_pred - y_true
weights = torch.where(d >= 0, self.alpha_late, self.alpha_early)   # 2.5 / 0.5
return torch.mean(weights * (d ** 2))
```

```
L = (1/B) Σ_i w_i · d_i² ,    w_i = 2.5 (alpha_late) if d_i ≥ 0 ;  0.5 (alpha_early) if d_i < 0
```

- Constructor raises `ValueError` if either alpha is ≤ 0 or if `alpha_late <= alpha_early`.
- Late:early penalty ratio = 2.5 / 0.5 = **5×**.
- Gradient `2·w_i·d_i/B` is linear in `|d|` and 0 at `d = 0` from both sides, so the loss is continuously differentiable (`C¹`); only its second derivative jumps.
- Selected by `get_loss_fn("asymmetric" | "asym" | "asymmetric_rul")`, or `client.py --loss-fn asymmetric` / `LOSS_FN`, with `--alpha-late/--alpha-early` or `ALPHA_LATE/ALPHA_EARLY`. `fed_sim.py` accepts `--loss-fn` but always uses the default alphas.

**Why not train on the raw NASA score.** Its gradient grows as `exp(|d|/10)` and explodes on the large errors of early training (`common.py` docstring). In `results/fd002/server_mu0_42.log` the untrained model (round 0, RMSE 85.29) has NASA score 1,139,425.52. The quadratic surrogate keeps gradients linear in `|d|`.

**Two different "asymmetric" objectives — do not confuse them.** `AsymmetricRULLoss` is a weighted quadratic used by the Flower/loss-study/dashboard code. The capped exponential (§4.3.3) is used only by the paired suite. The paper states they are not shown to be interchangeable.

#### 4.3.5 Empirical trade-off (MEASURED)

`results/asymmetric_loss_comparison.json` (`benchmark_asymmetric_loss.py`): seeds 42, 123, 456, 789, 1024; local 10 epochs (average of 2 clients), FedAvg 5 rounds × 2 epochs, centralized 10 epochs; Adam 1e-3, batch 64, 90/10 unit split; **population std (ddof = 0)**. Constant baseline (train-mean RUL): FD001 RMSE 40.55 / NASA 18,365.49 / late 45.0 %; FD002 RMSE 43.51 / NASA 63,730.17 / late 50.19 %.

| Dataset | Setup | Loss | RMSE | NASA | Late rate % |
|---|---|---|---:|---:|---:|
| FD001 | Local | mse | 17.58 ± 1.01 | 722.87 ± 120.55 | 56.2 ± 6.59 |
| FD001 | Local | asym | 17.15 ± 1.50 | 434.98 ± 85.58 | 33.8 ± 4.52 |
| FD001 | FedAvg | mse | 16.98 ± 1.31 | 690.81 ± 173.43 | 63.8 ± 7.11 |
| FD001 | FedAvg | asym | 16.25 ± 1.23 | 369.13 ± 43.50 | 34.2 ± 7.98 |
| FD001 | Central | mse | 15.16 ± 0.86 | 450.37 ± 91.81 | 56.6 ± 3.32 |
| FD001 | Central | asym | 15.80 ± 0.97 | 338.73 ± 51.98 | 30.8 ± 5.15 |
| FD002 | Local | mse | 23.73 ± 1.22 | 7,610.84 ± 3,611.39 | 50.27 ± 7.75 |
| FD002 | Local | asym | 26.30 ± 2.08 | 3,932.18 ± 829.67 | 29.5 ± 6.02 |
| FD002 | FedAvg | mse | 22.33 ± 1.46 | 5,499.06 ± 2,962.80 | 48.88 ± 8.73 |
| FD002 | FedAvg | asym | 26.07 ± 1.69 | 3,412.62 ± 429.82 | 25.25 ± 6.33 |
| FD002 | Central | mse | 20.17 ± 0.31 | 2,769.50 ± 494.61 | 40.85 ± 4.97 |
| FD002 | Central | asym | 25.79 ± 2.74 | 3,630.22 ± 1,185.19 | 20.54 ± 6.15 |

Reading it:

- **Late rate falls in all six setups**, from 40.8–63.8 % to 20.5–34.2 %.
- **NASA score** drops for 5 of 6 setups (FD001: −39.8 % / −46.6 % / −24.8 % for Local / FedAvg / Central; FD002: −48.3 % / −38.0 % for Local / FedAvg) and **rises +31.1 %** for FD002 Central (the README attributes it to conservative early-prediction outliers such as seed 1024). FD002 FedAvg NASA std falls from 2,962.8 to 429.8.
- **RMSE** improves for FD001 Local and FedAvg, rises slightly for FD001 Central (15.16 → 15.80), and **rises in every FD002 setup** (+10.8 % to +27.9 %). The loss trades symmetric accuracy for safety margin.
- No significance test is run for this study.

Capped-exponential objective (`summary.csv`, regime partitions, n=5, sample std):

| Dataset | Method | RMSE | NASA |
|---|---|---:|---:|
| FD002 | fedavg (MSE) | 23.50 ± 0.63 | 4,425.37 ± 1,367.59 |
| FD002 | federated_asymmetric | 27.44 ± 0.69 | 4,104.37 ± 534.95 |
| FD002 | centralized (MSE) | 21.46 ± 1.18 | 5,762.79 ± 2,135.75 |
| FD002 | asymmetric (central) | 25.62 ± 1.11 | 4,009.00 ± 422.46 |
| FD004 | fedavg (MSE) | 26.78 ± 0.20 | 7,654.82 ± 693.72 |
| FD004 | federated_asymmetric | 29.75 ± 0.36 | 6,513.13 ± 332.91 |
| FD004 | centralized (MSE) | 23.41 ± 0.49 | 4,329.01 ± 1,650.39 |
| FD004 | asymmetric (central) | 29.95 ± 0.82 | 7,243.98 ± 3,590.53 |

It lowers mean NASA versus FedAvg on both datasets and raises RMSE; for centralized FD004 the mean NASA **increases** (4,329 → 7,244). The paired suite records no late-prediction rate.

### 4.4 Federated learning algorithms

#### 4.4.1 FedAvg — sample-weighted parameter averaging

Per round: each client `k` loads the global weights `w`, trains locally, and returns `w_k` and `n_k` (its training-window count). The server computes

```
w ← Σ_k (n_k / Σ_j n_j) · w_k
```

- Live: Flower `fl.server.strategy.FedAvg` (`fraction_fit = fraction_evaluate = 1.0`, all `min_*_clients = MIN_CLIENTS`, default 2).
- `fed_sim.py` does the same average by hand: `sum(s[k].float() * n) / tot`.
- `reproduce.run_variant` weights by `len(tensors(train_data, units))`.

#### 4.4.2 FedProx — proximal term (`PROXIMAL_MU`)

The server strategy stays FedAvg. FedProx is purely the client-side objective:

```
L_k = crit(f(X), y) + (μ/2) · Σ_j ‖ w_j − w_j^(t) ‖²
```

where `w^(t)` is the global weights received this round. In `common.train_epochs`: `loss + 0.5 * mu * sum(((p - r) ** 2).sum() ...)` with `r` detached clones of the global parameters, active when `mu > 0`. The server passes `proximal_mu` through `on_fit_config_fn`; setting `PROXIMAL_MU > 0` also renames the output to `results_physical_fedprox_mu{mu}_seed{seed}.json`. The paired suite uses `μ = 0.01` (`train_prox`, MSE data term).

**MEASURED:** FedProx is close to FedAvg (e.g. FD002 regime 23.30 vs 23.50 RMSE; FD004 26.88 vs 26.78). In the in-process FD002 simulation, μ = 0.01 gives 22.20 ± 0.58 against FedAvg 22.20 ± 0.42, and μ = 0.1 gives 22.86 ± 0.93: no gain. FedProx was **not** run on the physical nodes.

#### 4.4.3 FedAdam — server-side Adam (`reproduce.run_variant`, `name == 'fedadam'`)

The client update is FedAvg; the server treats the average as a pseudo-gradient:

```
g  = w_global − avg(w_k)
m1 = 0.9·m1 + 0.1·g          m2 = 0.99·m2 + 0.01·g²
w_new = w_global − 0.01 · (m1/(1−0.9^t)) / ( sqrt(m2/(1−0.99^t)) + 1e-8 )
```

β1 = 0.9, β2 = 0.99, server lr **0.01**, eps 1e-8, bias-corrected, moments persist across rounds.

**Why it degraded (MEASURED outcome, ANALYSIS for the cause).** With the fixed server lr, FedAdam diverged:

| Partition | FedAdam RMSE | FedAdam NASA |
|---|---:|---:|
| FD002 regime | 78.23 ± 23.17 | 3.29 × 10⁸ ± 6.99 × 10⁸ |
| FD004 regime | 56.62 ± 16.01 | 9.00 × 10⁶ ± 1.84 × 10⁷ |
| FD002 engine_imbalance | 59.37 ± 16.21 | 1.05 × 10⁷ ± 2.08 × 10⁷ |
| FD004 engine_imbalance | 49.66 ± 12.60 | 2.14 × 10⁶ ± 4.49 × 10⁶ |

Adam normalizes by `sqrt(m2)`, so each step has magnitude about `lr` per parameter regardless of how small the averaged client change is. With 5 rounds and no lr tuning, a fixed 0.01 step on every weight can overshoot (ANALYSIS). The repository has **no server-lr sweep**, so this is a negative result for this configuration only, not evidence that all FedAdam settings fail (`paper/sections/results.tex`).

#### 4.4.4 Server-seeded initialization

`server.py` calls `set_seed(seed)` and builds `initial_parameters` from a fresh `CmapssRegressor()`. Otherwise Flower asks a random client for the initial weights, so round 0 would depend on which client connects first. Round-0 RMSE across the physical seeds is 85.18–85.39. In `reproduce.py` one seeded `init` state is shared by every variant (`initial = clone(init)`).

#### 4.4.5 `fed_sim.py` vs live Flower `server.py` / `client.py`

| | `server.py` + `client.py` (live) | `fed_sim.py` | `experiments/reproduce.py` |
|---|---|---|---|
| Transport | Flower gRPC, no TLS | in-process, no network | in-process, no network |
| Local training | `common.train_epochs`, **persistent** Adam per client | same code, persistent Adam | own `train`, **new Adam each round**, batch 128 |
| Aggregation | `FedAvg` (sample-weighted) | manual weighted average | manual (FedAvg/FedProx/FedAdam) |
| Data / partition | FD002, `partition.json` | FD002, `partition.json` | FD001–FD004, `iid` / `regime` / `engine_imbalance` |
| Scaler | `global_scale.json` | `global_scale.json` | full-train min/max, clipped to [-5, 5] |
| Output | `results_physical_seed{seed}.json` | `--out` JSON with `runs` and `summary` (`np.std`, ddof = 0) | `results/reproducible/*.json` + `summary.csv` (sample std, ddof = 1) |

**MEASURED parity.** `sim_fedavg_r5_e2.json` gives 22.20 ± 0.42 RMSE against the physical 22.27 ± 0.44 (5 rounds × 2 epochs, 5 seeds, `np.std`).

**MEASURED FD002 results (legacy Flower pipeline, n=5):**

| Setup | RMSE | NASA |
|---|---:|---:|
| Constant baseline | 43.51 | 63,730 |
| Isolated local (`final_benchmark.py`, 10 epochs) | 25.01 ± 0.54 | 11,937 ± 2,207 |
| FedAvg, physical 2× RPi 4 + Ubuntu server | 22.27 ± 0.44 | 6,037 ± 1,030 |
| FedAvg in-process, 5 rounds | 22.20 ± 0.42 | 5,990.93 ± 704.43 |
| FedAvg in-process, 15 rounds | 19.27 ± 1.30 | 2,741.56 ± 905.88 |
| Centralized (10 epochs) | 20.39 ± 1.11 | 3,395 ± 972 |

The 15-round simulation has not been compared with an equal-epoch centralized run. A 15-round physical run was launched but its results were not collected.

**MEASURED paired suite (`summary.csv`, global, mean ± sample std, n=5), IID partitions:**

| Dataset | Centralized | Local-only | FedAvg |
|---|---|---|---|
| FD001 | 14.85 ± 0.97 / 439.85 ± 139.24 | 19.80 ± 0.83 / 1,334.44 ± 328.37 | 19.27 ± 1.14 / 1,063.17 ± 268.40 |
| FD002 | 22.12 ± 1.78 / 4,180.30 ± 2,858.02 | 23.35 ± 0.41 / 5,283.19 ± 443.40 | 23.33 ± 0.24 / 4,179.66 ± 680.06 |
| FD003 | 18.00 ± 0.85 / 1,093.74 ± 311.96 | 21.98 ± 0.30 / 3,368.98 ± 477.38 | 21.60 ± 0.55 / 2,959.02 ± 559.00 |
| FD004 | 23.91 ± 0.81 / 4,758.71 ± 1,169.51 | 28.43 ± 1.49 / 15,523.43 ± 8,611.70 | 27.07 ± 0.56 / 8,059.20 ± 2,868.94 |

(cell = RMSE / NASA score). Regime and engine-imbalance FedAvg/FedProx rows are in `results/reproducible/summary.csv`; the regime figures are quoted above.

**Paired-test floor.** `reproduce.paired_p` is an exact two-sided sign-flip test. With n = 5 there are 2⁵ = 32 sign vectors and at least 2 (the observed pattern and its mirror) always count as extreme, so the minimum p-value is 2/32 = **0.0625**. No comparison can reach p < 0.05; the tests are exploratory with no multiple-comparison correction.

**Paired-suite protocol.** Datasets FD001–FD004; seeds 42, 100, 2024, 7, 777; centralized/local 10 epochs; federated 5 rounds × 2 local epochs; `--quick` runs FD001, seed 42, `iid` into `results/reproducible/smoke/`. Variants: `centralized`, `local_only`, `fedavg` everywhere; `fedprox`, `fedadam` on FD002/FD004 `regime` and `engine_imbalance`; `federated_asymmetric`, `asymmetric`, `cnn_lstm` on FD002/FD004 `regime`.

### 4.5 Unsupervised ML in diagnostics

| Use | Algorithm | Where | Details |
|---|---|---|---|
| Operating-regime discovery for drift | `sklearn.cluster.KMeans(n_clusters=6, n_init=10, random_state=seed)` on standardized `setting1..3`; centroids re-ordered with `np.lexsort`. `k = 1` for FD001/FD003. | `diagnostics.fit_healthy_baseline` | §5.5 |
| Regime-based client partition | Deterministic farthest-point k-means, `k = 6` (own code, not sklearn) | `reproduce.operating_clusters` | §3.6 |
| Unlabeled fault-mode separation on FD003/FD004 | `StandardScaler` → `KMeans(n_clusters=2, n_init=10, random_state=seed)`; agreement check with `GaussianMixture(n_components=2, covariance_type="diag", n_init=3, random_state=seed)` | `experiments/validate_diagnostics.cluster` | §5.7 |

---

## 5. Local edge degradation attribution and drift (`diagnostics.py`)

### 5.1 Purpose

A scalar RUL says *when*, not *which module*. `diagnostics.py` answers "which physical module or sensor is driving the RUL drop?" for one engine, entirely on the edge node that holds the raw telemetry. Its outputs are aggregate shares (%), standardized drift scores (σ), the predicted RUL and text. `PRIVACY_METADATA` records `network_calls: 0`, `raw_sensor_bytes_transmitted: 0`, `contains_raw_sensor_values: False`.

### 5.2 Physical channel map (Saxena et al. 2008, Table 2)

`FAULT_SENSORS` = the 14 non-setting channels; `SUBSYSTEMS = ("Fan", "LPC", "HPC", "Turbine")`.

| Subsystem | Sensors (symbol) | Count |
|---|---|---:|
| Fan | s8 Nf, s13 NRf, s15 BPR | 3 |
| LPC | s2 T24 | 1 |
| HPC | s3 T30, s7 P30, s9 Nc, s11 Ps30, s14 NRc | 5 |
| Turbine | s4 T50, s12 phi, s17 htBleed, s20 W31, s21 W32 | 5 |

Design notes:

- "Turbine" merges the combustor with the HPT/LPT hot section (`SUBSYSTEM_LABELS`).
- Sensors are grouped by **where they are measured**, not by which module is faulty: in the gas path a fault in one module shifts sensors in others. A subsystem share says which instrumented group dominates the model's evidence, not which component is faulty.
- `s12 phi` (fuel flow / Ps30) is assigned to Turbine (combustor). This placement decides the HPC-versus-Turbine split on FD001 (§5.7).
- Group sizes differ (3/1/5/5), so every validation rate is shown next to a sensor-count chance rate: HPC 5/14 = 35.7 %, core (HPC ∪ Turbine) 10/14 = 71.4 %.

### 5.3 Why `setting1..3` are model inputs but forced to 0.0 in attribution

The settings tell the model the operating context (FD002/FD004 cycle through six conditions), so they stay among the 17 inputs. But an operating setting is a flight condition chosen by the operator, not a degradation symptom. Two mechanisms keep it out of fault responsibility:

1. **Hard zero.** `attribute_windows` sets `a[:, SETTING_IDX] = 0.0` after computing attribution and before normalization, for all three methods. `sensor_shares` uses only `FAULT_IDX`.
2. **Zero baseline deviation.** `HealthyBaseline.healthy_reference` copies the *observed* settings into the reference, so `x − b = 0` on settings channels under `baseline_x_gradient`.

### 5.4 Attribution equations

Notation: `x ∈ ℝ^(W×17)` scaled window (W = 30), `ŷ = f(x)`, `g = ∂ŷ/∂x` from `torch.autograd.grad(y.sum(), x)`, `b` baseline window.

| `method` | Per-channel attribution `A_c` |
|---|---|
| `baseline_x_gradient` (default) | `A_c = (1/W) Σ_t | (x[t,c] − b[t,c]) · g[t,c] |` |
| `input_x_gradient` | `A_c = (1/W) Σ_t | x[t,c] · g[t,c] |` (b = 0) |
| `gradient` | `A_c = (1/W) Σ_t | g[t,c] |` |

then `A_c := 0` for `c ∈ SETTINGS`.

- `baseline_x_gradient` is a first-order estimate of how much a channel's *deviation from the engine's healthy expectation* moves the RUL estimate (Taylor: `f(x) − f(b) ≈ Σ (x−b)·g`). Because the absolute value is taken per element, the shares rank magnitudes; they are not a signed decomposition.
- An unknown method raises `ValueError`. If `method == "baseline_x_gradient"` and `baseline is None`, it **silently falls back** to `input_x_gradient`. An empty window array returns shapes `(0,)` and `(0, 17)`.
- One backward pass of `y.sum()` yields per-sample gradients because the model has no cross-sample layers and runs in `eval()` (ANALYSIS; the test `test_shapes_and_determinism` confirms `batch_size=3` equals `batch_size=512`).

**Shares and conservation.**

```python
a = clip(A[:, FAULT_IDX], 0, None);  share = 100 * a / a.sum()      # (N,14); uniform 100/14 if a.sum()==0
subsystem_share = sensor_share @ _SUB_MATRIX                          # (N,4); sums of the sensors' shares
subsystem_drift = (|z| @ _SUB_MATRIX) / _SUB_MATRIX.sum(0)            # mean |z| per subsystem
```

Sensor shares and subsystem shares each sum to 100.

**`round_to_total(values, decimals=1, total=100.0)` — largest-remainder rounding.** Naive rounding of `[100/3]*3` gives 99.9. The function floors `v·10^d`, computes `rest = round(total·10^d − Σ floor)`, requires `0 ≤ rest ≤ len(v)` (else `ValueError`; also rejects negative, NaN, inf and the empty list), adds one unit to the `rest` entries with the largest fractional remainders, and divides back. Example (`test_largest_remainder_choice`): `[10.9, 20.5, 68.6]` at 0 decimals → `[11, 20, 69]`. Used by `maintenance_summary` and by every per-cycle `subsystem_share` row in `demo_data.json`, so displayed shares sum to exactly **100.0 %**.

### 5.5 Regime-aware healthy-baseline drift

`fit_healthy_baseline(df_raw, n_regimes=1, healthy_cycles=30, seed=0)` is fitted on the node's own raw run-to-failure history:

1. Standardize `setting1..3` (`set_mean`, `set_std + 1e-9`).
2. Regime centroids: one at the origin if `k ≤ 1`, else `KMeans(k, n_init=10, random_state=seed)`, ordered by `lexsort` (deterministic IDs). `N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}`.
3. Per-regime `regime_mean`, `regime_std` of the 14 fault sensors (std < 1e-6 → 1.0).
4. Pooled early-life noise `healthy_sigma`: per unit, regime-standardize, take the first `healthy_cycles` rows, subtract their mean; the std of all residuals (std < 1e-6 → 1.0).

For an engine:

```
u[t,c]  = (x[t,c] − μ_{r(t),c}) / σ_{r(t),c}                       regime-standardized (r(t)= nearest centroid)
off_c   = mean_{t < H} u[t,c]          H = max(1, min(healthy_cycles, T))
z[t,c]  = (u[t,c] − off_c) / healthy_sigma_c                        drift, in units of healthy noise
window drift = mean of z over the window
```

The attribution baseline `b[t]` is the same engine's healthy expectation in model space: fault sensors `μ_{r(t)} + σ_{r(t)}·off` mapped through `scale_frame`, settings = observed values.

Why per-regime standardization: on FD002/FD004 a raw sensor jumps whenever the operating regime switches. A single fleet-wide mean/std (`k = 1`) would inflate the standard deviations and hide the slow fault signal. A synthetic unit test (`test_multi_regime_drift_removes_operating_condition`) shows the regime-aware s3 end-of-life drift exceeds 3× the `k = 1` value. **Boundary:** this comparison is synthetic only; the validation JSON has no naive-baseline ablation on real data.

### 5.6 Bilingual maintenance summary

`maintenance_summary(pred_rul, sensor_share, window_drift, engine_id, cycle, top_k=3)` returns a JSON-serializable dict with no raw sensor values: `predicted_rul`, `severity`, `primary_subsystem`, `subsystem_shares_pct` (via `round_to_total`), `primary_drift_subsystem`, `subsystem_drift_mean_abs_z`, `drift_index`, `top_contributors` (top 3 by share), `top_drifting` (top 3 by |z|, with `direction` up/down), `text.en`, `text.it`, `privacy`, `evaluation_boundary`.

`SEVERITY_THRESHOLDS = {"critical": 20.0, "warning": 45.0}`:

```
pred_rul <= 20 → "critical"   EN "Schedule maintenance now"   IT "Pianificare subito la manutenzione"
pred_rul <= 45 → "warning"    EN "Plan inspection"            IT "Pianificare un'ispezione"
otherwise      → "healthy"    EN "No action"                  IT "Nessuna azione"
```

`EdgeDiagnostics(model, scales, baseline, method)` is the driver: `diagnose_engine(raw_frame, id)` returns aligned arrays (`cycles`, `pred_rul`, `channel_attr` (N,17), `sensor_share` (N,14), `subsystem_share` (N,4), `window_drift` (N,14), `cycle_drift` (T,14)); `summarize(diag, index=-1)` renders one window.

### 5.7 Validation results (MEASURED, `results/diagnostics_validation.json`)

Protocol (`experiments/validate_diagnostics.py`): seeds `[7, 42, 100, 777, 2024]`; FD001–FD004; one isolated `CmapssRegressor` per seed and subset, trained with `AsymmetricRULLoss`, 8 epochs, batch 64, on 80 % of training units (`split_units(units, 0.2, seed)`); `compute_scales_from_train`; `baseline_x_gradient` (plus `gradient` for comparison on FD001/FD002). Environment: Python 3.11.4, torch 2.15.0, `machine: arm64`, `system: Darwin`; `elapsed_s` 675.2. Note: this model is **not** the paper's federated model.

**Official test RMSE** (last window per test engine, mean ± sd over 5 seeds): FD001 16.80 ± 1.42, FD002 23.74 ± 3.08, FD003 17.46 ± 2.68, FD004 26.44 ± 4.79.

**FD001 / FD002 single fault mode, windows with true RUL < 30** (population: held-out training units + all official test units):

| Rate (mean ± sd, 5 seeds) | `baseline_x_gradient` FD001 | FD002 | `gradient` FD001 | FD002 | Chance |
|---|---:|---:|---:|---:|---:|
| Primary subsystem = HPC | 26.6 ± 5.5 % | 59.6 ± 5.0 % | 0.8 ± 1.5 % | 1.4 ± 3.0 % | 35.7 % |
| Primary subsystem ∈ core (HPC ∪ Turbine) | 99.4 ± 0.8 % | 100.0 ± 0.0 % | 98.7 ± 2.1 % | 100.0 ± 0.0 % | 71.4 % |
| Top-3 attributed sensors in core | 94.9 ± 5.1 % | 73.3 ± 1.7 % | 83.6 ± 12.3 % | 66.7 ± 0.0 % | 71.4 % |
| Top-3 drifting sensors in core (model-free) | 82.0 ± 1.7 % | 76.7 ± 0.6 % | — | — | 71.4 % |

Model-free drift ranks HPC first in 51.4 % (FD001) and 76.2 % (FD002) of windows. Plain gradient saliency almost never ranks HPC first, i.e. it describes model sensitivity rather than engine state. HPC-only attribution on FD001 is *below* chance because phi (s12, mapped to Turbine) moves together with P30/Ps30 in the HPC fault signature (§5.2). `input_x_gradient` is unit-tested but **not** in the validation JSON.

**FD003 / FD004 two fault modes (unlabeled), k = 2 clustering of end-of-life profiles** (profile = mean of the last `EOL_WINDOWS = 10` windows; FD003 120 profiles, FD004 302). Clusters are named without labels: the cluster whose mean signed drift has the higher cosine to the HPC-only sibling reference (`REFERENCE = {"FD003": "FD001", "FD004": "FD002"}`) is `hpc_like`, the other is the `fan_candidate`.

| View | Metric | FD003 | FD004 |
|---|---|---:|---:|
| drift | silhouette | 0.51 | 0.49 |
| drift | cross-seed ARI (10 pairs) | 1.00 | 0.99 ± 0.01 |
| drift | KMeans vs GMM ARI | 0.93 | 0.93 |
| drift | cosine to HPC ref (`hpc_like`) | 0.98 | 1.00 |
| attribution | cross-seed ARI | 1.00 | 0.70 ± 0.31 |
| drift vs attribution | cross-view ARI | 0.93 | 0.37 ± 0.11 |

In the fan candidate, P30, phi and coolant bleeds W31/W32 reverse sign relative to the HPC-only signature and BPR falls (FD003: s7 +21.89 vs −3.21; s12 +26.86 vs −3.45; s15 −5.39 vs +3.15). This is physically consistent with a second, fan-side fault mode, but **no ground truth exists** to confirm the cluster-to-mode mapping. Subsystem share bars alone do not separate the modes (they sum absolute attributions and lose sign); the dashboard bar is not a per-window mode label. On FD004, attribution-only clustering is not reliably seed-stable.

### 5.8 Empirical boundary, privacy audit and edge cost

**Strict boundary (`diagnostics.EVALUATION_BOUNDARY`):** "C-MAPSS provides unit-level RUL only; no per-unit component fault labels exist. Subsystem attribution is a model/physics-based explanation, not a supervised component diagnosis." **No supervised component accuracy is claimed anywhere.** Validation is by physical consistency (FD001/FD002) and unsupervised separability (FD003/FD004).

**`SocketAudit`** (`experiments/validate_diagnostics.py`) monkey-patches `socket.socket.connect`, `connect_ex`, `send`, `sendall`, `sendto` and `socket.create_connection`, counting calls and bytes. It wraps the baseline fit, the diagnosis of every engine, and the latency timing. **MEASURED:** `network_audit = {total_socket_calls_during_diagnostics: 0, raw_sensor_bytes_sent: 0}`. Scope: only those instrumented entry points in one process; other I/O paths (e.g. `sendmsg`, subprocesses) are not instrumented. It does not change the repository's privacy scope (no TLS/DP/secure aggregation).

**Latency (MEASURED on an Apple Silicon `arm64` Darwin host; single thread; median over 300 windows; the FD001 timing may overlap a concurrent job):**

| ms per window | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| inference only | 0.0513 | 0.0664 | 0.0602 | 0.0565 |
| + attribution + summary | 0.2193 | 0.3284 | 0.2743 | 0.2446 |
| + drift recomputed from raw history + attribution + summary | **0.4911** | **0.9319** | **0.6473** | **0.5895** |

Autograd saved-tensor memory: 117,368 bytes per window; an input window is 2,040 bytes; model weights 89,092 bytes. Whole-process peak RSS (703,873,024 bytes) includes training and is not the diagnostics footprint.

**UNMEASURED:** no Raspberry Pi 4 measurement exists. The Pi helpers were unreachable (SSH status 255, `results/EXPERIMENTS.md`). That sub-millisecond host cost is acceptable on Pi-class hardware, given one window per engine cycle, is an inference, not a Pi result. Energy, temperature/throttling, wire bytes and TLS overhead are also unmeasured.

**Also noted:** in validation the healthy baseline is fitted once per subset on *all* training units (including units held out from model training); a deployment would fit it on the node's own history.

---

## 6. Dashboard, paper pipeline and test suite

### 6.1 Dashboard (`dashboard/`)

```
dashboard/generate_demo_data.py → dashboard/demo_data.json → dashboard/server.py → dashboard/index.html
   (offline, needs CMAPSSData.zip)     (committed, 674,295 B)     (stdlib HTTP)        (vanilla HTML/SVG/JS)
```

**`generate_demo_data.py`.** `train_subset_models(subset, seed=42, rounds=5, local_epochs=2)` trains three models: `Local MSE` (client "1" only, `rounds × local_epochs = 10` epochs), `FedAvg MSE` and `FedAvg Asym` (in-process, persistent Adam, window-count-weighted average). FD001 uses `compute_scales_from_train` and an odd/even split; FD002 uses `global_scale.json` and `partition.json`. It fits a healthy baseline per subset and runs `diagnostics` with the FedAvg-asymmetric model on **8 test engines** (4 per subset, longest unused engine with ≥ 85 observed cycles per bucket by true final RUL: `critical` ≤ 20, `warning` 22–48, `midlife` 52–85, `healthy` ≥ 95). Prediction series are clipped to [0, 125]; `final_summary.predicted_rul` is the raw model output.

**`demo_data.json`** top-level keys: `model_info` (22,273 params, 89,092 and 178,184 bytes), `engines` (8, each with `cycles`, `true_rul`, `pred_local_mse`, `pred_fedavg_mse`, `pred_fedavg_asym`, four scaled `sensors` s2/s3/s4/s11, and a `diagnostics` sub-tree of shares, z-scores and `final_summary`), `diagnostics_meta`, `cluster` (`physical_5r`: 5 runs from `results/fd002/results_physical_seed*.json`; `sim_15r_seed42`: 15 rounds), `benchmarks` (`asymmetric` from `asymmetric_loss_comparison.json`; `reproducible`: 228 rows of `summary.csv`). The `diagnostics` sub-tree has no sensor values, but the `sensors` series do contain min-max-scaled traces of four sensors. Committed engines (final true RUL / severity from predicted RUL): FD001-U81 (8, critical), U91 (38, warning), U93 (85, warning), U12 (124, healthy); FD002-U65 (11, critical), U58 (26, critical), U140 (55, warning), U119 (106, healthy). `stage` (true RUL bucket) and `severity` (predicted RUL) can differ.

**`server.py`.** Standard library only. `DemoHandler(SimpleHTTPRequestHandler)` serves `dashboard/`; `GET /api/health` → `{"status": "ok", "has_data": <demo_data.json exists>}`; `/` → `/index.html`; `list_directory()` returns 404, so directory listings are blocked (individual files are still served by name). CLI: `--host` (default `127.0.0.1`), `--port` (default `8765`; `--port 0` prints the actual bound port), `--no-browser`; Ctrl+C prints "Shutting down dashboard.".

**`index.html`.** One inline script, no external scripts/styles/fonts; `boot()` fetches `/demo_data.json`, so it must be served by `server.py`. Language starts as Italian (`<html lang="it">`, `LANG = "it"`) with an EN/IT toggle. Three tabs:

1. **Live Asset RUL** — engine select (8), cycle slider, play/pause (90 ms/step), KPI cards, RUL chart (true vs three predictions, critical band < 20), sensor chart, and the diagnostics card (stacked subsystem bar, top-3 contributors, top-3 drifting with ▲/▼ and σ, share trend, bilingual summary, evaluation-boundary text). Status uses the same ≤ 20 / ≤ 45 thresholds as `diagnostics.py`. An hours selector (cycles only, 0.5, **1 default**, 2, 4 h/cycle) is a presentation scale — C-MAPSS defines cycles, not hours.
2. **Federated Cluster** — five physical 5-round FD002 runs plus the 15-round simulation, round slider (650 ms/step), RMSE/NASA per round.
3. **Safety & Business Impact** — MSE vs asymmetric table, FD002 engine-imbalance table, engineering note.

Boundary: several tab 2/3 strings are literals in `index.html` (e.g. "(Raspberry Pi 4)" node titles, "~16.7k windows", "12.8s/round", "Baseline 43.51"), not read from `demo_data.json`.

**Commands**

```bash
CMAPSS_ZIP=/path/to/CMAPSSData.zip python dashboard/generate_demo_data.py   # regenerate (optional)
python dashboard/server.py                                                  # http://127.0.0.1:8765/
python dashboard/server.py --port 0 --no-browser
```

### 6.2 Paper pipeline (`paper/`)

- `paper/main.tex` (IEEEtran) includes `paper/sections/*.tex` and `paper/references.bib`.
- `paper/make_figures.py` (standard library only) reads `results/reproducible/summary.csv`, `results/asymmetric_loss_comparison.json`, `results/edge/local-smoke-macos.json`. It skips rows with `n < 5` and fails the build on incomplete or unexpected five-seed inputs. It writes `paper/generated/macros.tex` (scalar macros such as `\EdgeSmokePeakRssMiB`) and `table_main.tex`, `table_algorithms.tex`, `table_partitions.tex`, `table_clients.tex`, `table_significance.tex`, `table_asymmetric.tex`. Do not edit `generated/` by hand.
- `paper/Makefile`: `make` runs `python3 make_figures.py` then `latexmk -pdf main.tex`.
- Open `\todo` markers (all in `main.tex`): University of Oulu co-authors, Oulu affiliation(s), AI-use disclosure. None in `paper/sections/`.
- The paper avoids claims of SOTA accuracy, Raspberry Pi efficiency, encrypted transport and formal privacy.

### 6.3 Test suites (`tests/`)

All are `unittest`-based and also run under pytest. Counts are `test_*` methods in each file.

| File | Classes / tests | What it covers | Data gating |
|---|---|---|---|
| `tests/test_diagnostics.py` | 6 / 22 | Channel map vs Table 2, three attribution methods (formulas, shapes, determinism, empty input), 100 % conservation, settings forced to 0, `round_to_total`, baseline drift (single and 6-regime synthetic), engine/summary structure, severity thresholds | `RealDataIntegrationTest` needs the C-MAPSS zip |
| `tests/test_dashboard.py` | 6 / 18 | `demo_data.json` invariants, generator unit tests, byte-identical regeneration, server smoke (`/api/health`, no directory listing, clean SIGINT), `index.html` `node --check` and DOM ids, headless-browser scrubbing/language tests | `node` required; regeneration needs `CMAPSS_ZIP`; browser tests skip if no Selenium driver |
| `tests/test_diagnostics_validation.py` | 5 / 16 | `SocketAudit`, validation helpers, JSON-vs-docs number integrity (`docs/diagnostics.md`, `results/EXPERIMENTS.md`), SHA-256 pins of `experiments/reproduce.py`, `results/reproducible/summary.csv` and the `paper/` tree digest, no symlinks | `CmapssSmokeTest` needs the zip |
| `tests/test_system_smoke.py` | 9 / 21 | Model shape and 22,273-parameter count, seeded determinism, loss asymmetry and gradients, `nasa_score`, `late_prediction_rate`, short-sequence padding, `split_units`, `sanity_check`, `fed_sim` (FedAvg and FedProx+asymmetric), `profile_edge.py` one-round run, `make_figures.py` reproduces tracked tables, `latexmk` PDF build | Most need the zip; `latexmk` test needs `latexmk` and `pdflatex` |
| `tests/test_federated_e2e.py` | 4 / 11 | `SupervisedClient` parameter round-trip and strictness, fit with MSE/asymmetric ± proximal, proximal pull, `server.weighted_avg`, a 2-round FedAvg then local diagnostics on synthetic plants, and a Flower gRPC loopback (server + 2 clients on 127.0.0.1) | Loopback needs the zip and `flwr`; the rest is synthetic |

Run (from the repository root):

```bash
python -m pip install -r requirements.txt          # flwr, torch, numpy, pandas, scikit-learn
export CMAPSS_ZIP=/path/to/CMAPSSData.zip          # optional; enables data-gated tests
python -m unittest discover -s tests -v            # or: python -m pytest tests
python -m unittest tests.test_diagnostics          # a single module
python sanity_check.py                             # FD002 gate: n == 259, trained RMSE < 0.75 × baseline, loss checks
```

Tests that write files use temporary directories. The SHA-256 pins in `test_diagnostics_validation.py` mean intentional edits to `experiments/reproduce.py`, `results/reproducible/summary.csv` or `paper/` must update the pins deliberately.

### 6.4 Experiment commands

```bash
python experiments/reproduce.py --data /path/to/CMAPSSData.zip --out results/reproducible --threads 1
python experiments/reproduce.py --summarize-only
python experiments/reproduce.py --data /path/to/CMAPSSData.zip --quick          # smoke only, not a result
python experiments/validate_diagnostics.py --out results/diagnostics_validation.json   # needs CMAPSS_ZIP
python experiments/profile_edge.py --data /path/to/CMAPSSData.zip --dataset FD002 --partition regime --client 0 --output results/edge/rpi_fd002.json
python benchmark_asymmetric_loss.py                  # controlled MSE vs AsymmetricRULLoss study
python fed_sim.py --rounds 5 --epochs 2 --mu 0 --out results/fd002/sim_fedavg_r5_e2.json
python final_benchmark.py --epochs 10
SEED=42 ROUNDS=5 python server.py                    # coordinator
CLIENT_ID=1 FL_SERVER=<ip>:8081 python client.py     # each client (CLIENT_ID in partition.json)
```

Pinned versions for the paired run: `experiments/requirements.txt` (`numpy==2.5.4`, `pandas==3.1.1`, `torch==2.15.0`). The root `requirements.txt` is unpinned (`flwr>=1.7`, `torch>=2.0`, `numpy`, `pandas`, `scikit-learn`).

---

## 7. Repository file map, bug-fix evolution and known limitations

### 7.1 File map

| Path | Purpose | Key symbols |
|---|---|---|
| `common.py` | Model, data I/O, loss, metrics, shared training | `USEFUL_SENSORS`, `SEQ_LEN`, `MAX_RUL`, `CmapssRegressor`, `extract_supervised_sequences`, `extract_last_sequence`, `nasa_score`, `late_prediction_rate`, `AsymmetricRULLoss`, `get_loss_fn`, `apply_scale`, `build_test_set`, `split_units`, `train_epochs`, `make_loader`, `cmapss_zip_path` |
| `server.py` | Flower coordinator (FedAvg, seeded init, per-round test eval) | `get_evaluate_fn`, `weighted_avg`; env `SEED ROUNDS MIN_CLIENTS PROXIMAL_MU LOCAL_EPOCHS RESULTS_DIR BIND` |
| `client.py` | Flower edge client | `SupervisedClient`, `peak_rss_mb`; env `CLIENT_ID FL_SERVER SEED LOSS_FN ALPHA_LATE ALPHA_EARLY` |
| `fed_sim.py` | In-process FedAvg/FedProx, same local code as `client.py` | `run`; CLI `--seeds --rounds --epochs --mu --lr --loss-fn --out` |
| `final_benchmark.py` | Constant / local / centralized baselines vs physical FedAvg JSONs | `run_benchmark` |
| `benchmark_asymmetric_loss.py` | Controlled 5-seed MSE vs `AsymmetricRULLoss` study | `compute_scales_from_train`, `eval_metrics`, `run_fedavg`, `run_subset` |
| `compute_global_scale.py` | Writes `global_scale.json` (FD002, first 50 cycles, `train_mean_rul`) | script |
| `setup_fd002.py` | Lifespan-sorted 2-way split → `partition.json` (hard-coded zip path) | script |
| `sanity_check.py` | Verification gate | `run_sanity_check`, `run_loss_sanity_check` |
| `inspect_fd002.py`, `plot_rul.py` | Small helpers; `plot_rul.py` expects a legacy `results_physical.json` schema | scripts |
| `deploy.sh`, `run_fd002_federated.sh` | rsync to two Pis; multi-seed physical run (server + `ssh -f -n` clients, 1200 s timeout per seed) | shell |
| `partition.json`, `global_scale.json` | FD002 client split (130 + 130 units); healthy-regime min/max + `train_mean_rul` | data |
| `diagnostics.py` | Local attribution, drift, summaries | `SUBSYSTEMS`, `SENSOR_INFO`, `SUBSYSTEM_SENSORS`, `N_REGIMES`, `SEVERITY_THRESHOLDS`, `PRIVACY_METADATA`, `EVALUATION_BOUNDARY`, `HealthyBaseline`, `fit_healthy_baseline`, `attribute_windows`, `sensor_shares`, `subsystem_shares`, `subsystem_drift`, `round_to_total`, `severity_for`, `maintenance_summary`, `EdgeDiagnostics` |
| `experiments/reproduce.py` | Paired FD001–FD004 suite | `CNN`, `CNNLSTM`, `read_data`, `operating_clusters`, `window_regime_data`, `partitions`, `train`, `train_prox`, `run_variant`, `paired_p`, `summarize` |
| `experiments/profile_edge.py` | One-client training time / RSS / payload profiler | `load_client`, `peak_rss_bytes` |
| `experiments/validate_diagnostics.py` | 5-seed diagnostics validation | `SocketAudit`, `single_fault_metrics`, `eol_profiles`, `cluster`, `reference_signature`, `orient`, `two_fault_metrics`, `saved_tensor_bytes`, `latency`, `run_subset` |
| `dashboard/` | Offline presentation dashboard | `generate_demo_data.py` (`train_subset_models`, `pick_representative_engines`, `engine_diagnostics`), `demo_data.json`, `server.py` (`DemoHandler`), `index.html` |
| `tests/` | Five test modules (§6.3) | — |
| `paper/` | IEEEtran manuscript; `make_figures.py`; `generated/` | — |
| `results/` | `reproducible/` (40 raw JSON + `summary.csv`, quarantined `interrupted/`), `fd001/`, `fd002/` (+`legacy_pre_fix/`), `edge/local-smoke-macos.json`, `asymmetric_loss_comparison.json`, `diagnostics_validation.json`, `EXPERIMENTS.md` | — |
| `docs/` | `diagnostics.md`, `literature_review.md` (partly stale), `project_guide_part1.md`, `project_guide_part2.md` | — |
| `PDM_FIXES.md` | Audit/fix log of the legacy Flower pipeline | — |

### 7.2 Key historical fixes

**Legacy Flower pipeline (`PDM_FIXES.md`, rounds 8 and 9, 2026-09-18 and 2026-10-23).**

| ID | Problem | Resolution recorded |
|---|---|---|
| P23 | Baselines trained on different units than federated | `final_benchmark.py` filters to the partition units (same train units as the federated clients). |
| P24, P11 | Federated row not comparable; seed count | Five physical seeds aggregated for every row. |
| P26 | RUL cap undocumented | Benchmark prints "True RUL is capped at 125 cycles during evaluation as per standard CMAPSS practice." |
| P21 | Sanity gate crashed, had a duplicated assert and an inert `... if "trained_rmse" in locals() else True`, and tested the wrong holdout | Rewritten against the official FD002 test set (`n == 259`, `untrained > baseline`, `trained < 0.75·baseline`); `PDM_FIXES.md` round 9 marks it closed. |
| P27 | FD001 is IID, nothing for FL to correct | Pipeline moved to FD002; results split into `results/fd001/` and `results/fd002/`. |
| P20 | Client `evaluate()` returned 0.0; deprecated Flower API | `evaluate` now returns validation RMSE on held-out units. API deprecation **still open**: `server.py` still calls `fl.server.start_server`. |
| Seeds | Client seed was fixed at 42 | Client seed is `SEED + CLIENT_ID`. |

**Paired suite (`results/EXPERIMENTS.md`, quarantined runs under `results/reproducible/interrupted/`).** All are excluded from `summary.csv`, which globs only top-level `fd*_seed*.json`:

- `pre_32-64-cnn-alignment/` — first smoke used a 32/48-channel CNN; superseded by the 32/64 `CmapssRegressor`. `--quick` now writes to `smoke/`.
- `seed42_pre_regime-fix/` — false regime partition for single-regime FD001; per-engine medians would give an extreme 17/243 FD002 split, which led to the window-level regime split.
- `pre_variant-filtering/` — partial run stopped when method-by-partition coverage was narrowed.
- `pre_short-test-padding/` — loader omitted short official test engines; fixed by left-padding (§3.3).

Superseded numbers: the round-9 FD002 FedAvg figure in `PDM_FIXES.md` (21.98 ± 0.99) is replaced by the post-padding physical rerun (22.27 ± 0.44). The README FD001 legacy table is from the pre-audit pipeline and was not re-run.

### 7.3 Known limitations and production/PoC gaps

| Area | Status |
|---|---|
| **Transport security** | `mTLS = False` everywhere. `start_server`/`start_client` get no certificates; the server log says "SSL is disabled". |
| **Privacy** | No differential privacy, no secure aggregation, no privacy accountant. Model updates can leak information; only "raw windows are not serialized" is supported. |
| **Preprocessing** | Centralized min-max scaling (`global_scale.json`, or full-train min/max in `reproduce.py`), shipped to clients. No federated/per-client normalization. |
| **Raspberry Pi 4 hardware** | Physical FedAvg ran on 2 Pis (round-5 `train_time_s` 15.32–16.46 s, `peak_rss_mb` 598.1–600.7 across seeds), but `profile_edge.py` and the diagnostics latency have **no Pi measurements**. Energy, thermals, wire bytes and TLS overhead are unmeasured. The Mac smoke profile (3.646 s, 426,196,992 B RSS) is not Pi evidence. |
| **Scale of federation** | Two clients, five rounds. No SCAFFOLD, personalization or clustered FL. No round-count sweep; a 15-round physical run was not collected. |
| **Statistics** | 5 seeds; the exact paired test cannot go below p = 0.0625; no multiple-comparison correction. |
| **FedAdam** | Fixed server lr 0.01 diverged; no lr sweep. |
| **Partition provenance** | `partition.json` origin unresolved; regime split shares engines across clients (overlap 260 / 249). |
| **Diagnostics** | No component labels, so no supervised accuracy. Subsystem shares reflect sensor placement, not fault location. FD003/FD004 fan-mode naming is a hypothesis. Baseline fit in validation includes held-out units. Naive-drift failure shown on synthetic data only. |
| **Dashboard** | Static snapshot with 8 engines; hours are a presentation scale; some tab 2/3 figures are hard-coded; directory listings are blocked but files are served by name. |
| **Inconsistencies to know** | Two scalers; two "asymmetric" losses; two std conventions (sample, ddof = 1, in `summary.csv`; population, ddof = 0, in `fed_sim.py`, `final_benchmark.py`, `benchmark_asymmetric_loss.py`); two seed sets (`42 100 2024 7 777` vs `42 123 456 789 1024` in the loss study); batch 64 vs 128; persistent vs reset Adam; `docs/literature_review.md` predates `AsymmetricRULLoss`, FedProx/FedAdam and the paired tests. |
| **Flower API** | Uses deprecated `fl.server.start_server`; migration to `ServerApp`/`ClientApp` is open. |
| **Models** | Compared only the 1D-CNN and a compact CNN-LSTM. Published centralized FD002 models report lower error under different protocols; this repository does not claim SOTA accuracy. |
