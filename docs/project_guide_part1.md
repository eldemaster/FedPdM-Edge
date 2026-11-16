# FedPdM-Edge Project Guide — Part 1: Core Pipeline, Federation, Objectives and Evidence

> **Scope of this part.** Part 1 covers the research motivation, the C-MAPSS data pipeline, the model, the safety-oriented loss, the Flower deployment and in-process simulation, the reproducible benchmark suite, the bug-fix history and the manuscript status. **Part 2** (a separate document) covers `diagnostics.py`, `dashboard/` and `tests/`; those files are referenced here only when another source file mentions them.
>
> **Evidence rule.** Every path, symbol, hyperparameter and number below comes from a file in this repository. Each claim names its source. Where the repository does not support a claim, the guide says so. Items that were **not measured** (Raspberry Pi profiler output, energy, wire traffic, TLS overhead) are marked **UNMEASURED**.
>
> **Sources read for this part:** `README.md`, `PDM_FIXES.md`, `docs/literature_review.md`, `requirements.txt`, `experiments/requirements.txt`, `common.py`, `client.py`, `server.py`, `fed_sim.py`, `compute_global_scale.py`, `setup_fd002.py`, `sanity_check.py`, `benchmark_asymmetric_loss.py`, `final_benchmark.py`, `inspect_fd002.py`, `plot_rul.py`, `deploy.sh`, `run_fd002_federated.sh`, `partition.json`, `global_scale.json`, `experiments/reproduce.py`, `experiments/profile_edge.py`, `results/EXPERIMENTS.md`, `results/reproducible/summary.csv` (plus the `*_seed42.json` raw files for partition metadata), `results/asymmetric_loss_comparison.json`, `results/edge/local-smoke-macos.json`, `results/fd002/results_physical_seed*.json`, `results/fd002/benchmark_e10.csv`, `results/fd002/sim_*.json`, `results/fd002/server_mu0_42.log`, `results/fd001/benchmark_table.txt`, `paper/main.tex`, `paper/sections/*.tex`, `paper/make_figures.py`, `paper/generated/macros.tex`, `paper/README.md`, `paper/Makefile`.

---

## Table of contents

1. [Project purpose, context and research motivation](#1-project-purpose-context-and-research-motivation)
2. [Dataset and preprocessing decisions (C-MAPSS FD001–FD004)](#2-dataset-and-preprocessing-decisions-nasa-c-mapss-fd001fd004)
3. [Model architecture and edge resource decisions](#3-model-architecture-and-edge-resource-decisions)
4. [Safety-critical objective: `AsymmetricRULLoss`](#4-safety-critical-objective-asymmetricrulloss)
5. [Federated learning design and deployment](#5-federated-learning-design-and-deployment)
6. [Reproducible benchmark suite, bug-fix history and paper status](#6-reproducible-benchmark-suite-bug-fix-evolution-and-paper-status)
7. [Appendix A — File-by-file reference](#appendix-a--file-by-file-reference)
8. [Appendix B — Known inconsistencies and caveats found while writing this guide](#appendix-b--known-inconsistencies-and-caveats-found-while-writing-this-guide)
9. [Appendix C — Verification ledger](#appendix-c--verification-ledger)

---

## 1. Project purpose, context and research motivation

### 1.1 What the repository is

`README.md` describes the project as "Research code for federated Remaining Useful Life (RUL) prediction on NASA C-MAPSS turbofan data". It contains two separate codebases:

| Codebase | Files | Role | Network? |
|---|---|---|---|
| **Flower deployment demo** (older) | `server.py`, `client.py`, `common.py`, `fed_sim.py`, `final_benchmark.py`, `benchmark_asymmetric_loss.py`, `run_fd002_federated.sh`, `deploy.sh` | Physical federation over gRPC: two Raspberry Pi clients and an Ubuntu coordinator. Also an in-process simulation with the same code. | Yes (gRPC, no TLS) |
| **Paired experiment suite** (newer) | `experiments/reproduce.py`, `experiments/profile_edge.py` | Five paired seeds across FD001–FD004. Covers centralized, local-only, FedAvg, FedProx, FedAdam, a capped-exponential loss and a CNN-LSTM. | No: client updates are simulated in one process |

The README warns that the root-level Flower files "predate the paired runner" and "should not be used as the sole basis for the tables in a new study" (`README.md`, "Existing Flower demo").

### 1.2 Funding and institutional context

- `paper/main.tex` acknowledgment and `\thanks`: "This work was supported by the Interreg Aurora project ResilientEdge (grant 20373282)."
- `paper/main.tex` authors: Alessandro De Martini and Stefan Forsström, Mid Sweden University, Sundsvall, Sweden. The University of Oulu co-authors and affiliations are still `\todo{University of Oulu co-authors}` and `\todo{Oulu affiliation(s)}`.
- `README.md`, "Project": "Developed within the EU Interreg Aurora project ResilientEdge (grant 20373282), with Mid Sweden University and the University of Oulu."

### 1.3 Motivation as stated in the repository

The paper's introduction (`paper/sections/introduction.tex`) gives the motivation:

1. **Sensitive telemetry and scarce failure data per site.** "Telemetry can be operationally sensitive, while individual sites may have limited examples of degradation and failure. Federated learning (FL) lets clients train on local data and share model updates with a coordinator."
2. **Data locality is not privacy.** "Exchanging model updates is not itself a formal privacy guarantee". RUL work "must therefore distinguish data locality from protections such as differential privacy and secure aggregation."
3. **Asymmetric maintenance cost.** "A late prediction ($\hat{y} > y$) may defer service beyond the actual failure point, while an early prediction ($\hat{y} < y$) may consume useful life." Reporting RMSE alone "can therefore hide relevant differences."

The research questions (`introduction.tex`) are:
1. How do centralized, local-only and FedAvg models compare under matched preprocessing and seeds?
2. How do client partition and aggregation method affect global and per-client RUL accuracy?
3. How does asymmetric-risk optimization change symmetric accuracy and late-prediction risk?

> **Not found in the repository.** The phrases "industrial SMEs" and "multi-plant" do not appear in any file read for this guide. A case-insensitive search for `SME|small and medium|multi-plant|plant` found only the README/EXPERIMENTS.md sentence saying the regime split "does not model disjoint plant fleets". The SME/multi-plant framing in the task brief is therefore **not a documented claim of this repository**. This guide uses only the "sites" and "telemetry" wording above.

### 1.4 Gap analysis and positioning (from `docs/literature_review.md`)

The literature review was checked "against publisher pages, PHM Society proceedings, PMLR, ACM, and arXiv records on 12 November 2026". Prior work it cites:

| Area | Prior work (as cited) | Implication for FedPdM-Edge (per the review) |
|---|---|---|
| Benchmark and metric | Saxena et al. 2008 (two PHM papers) | Origin of C-MAPSS and the asymmetric NASA score. |
| Centralized deep RUL | Li, Ding & Sun 2018 CNN (RMSE 18.45 / 30.29 / 19.82 / 29.16); Wang et al. 2024 SA-TCN (11.50 / 16.45 / 11.62 / 15.47); Liu, Xu & Geng 2025 MoEFormer (15.13 / 13.71 / 13.88 / 14.47) | Centralized FD002 models in published work report lower error than the repository's FedAvg. The protocols differ, so the values are context, not a leaderboard. |
| Federated RUL | Llasag Rosero et al. 2020 (FedAvg/FedProx, μ≈0.01); Söderkvist Vermelin et al. 2024 (C-MAPSS, personalization layers); Chen et al. 2023 (pruning + FedAvg); Landau et al. 2025 (N-CMAPSS, six airlines); Sorrenti et al. 2026 FedCMAPSS preprint (five standardized federated tasks) | "The current paper cannot claim first C-MAPSS FL or first personalized RUL FL." FedCMAPSS has "the most direct overlap with the algorithmic/heterogeneity contribution." |
| Privacy | Nikulins et al. 2026 (DP-FL for RUL); Abadi et al. 2016 (DP-SGD); Bonawitz et al. 2017 (secure aggregation) | "The current repository implements neither DP nor an accountant". TLS and secure aggregation are separate guarantees. |
| Edge systems | Gao et al. 2020 (FL vs SplitNN on Raspberry Pi); Sebbio et al. 2023 (RPi 4 power with INA219) | Physical overhead must be measured before claiming "resource-efficient". |
| Optimizers | Karimireddy et al. 2020 (SCAFFOLD); Reddi et al. 2021 (FedAdam/FedYogi); Sattler et al. 2021 (clustered FL) | Candidate baselines. Clustered FL "is not scientifically informative with only two clients". |

**Stated positioning** (`literature_review.md`, "Gap analysis"):

> "The study's defensible angle is an end-to-end physical testbed with two Raspberry Pi clients and an Ubuntu coordinator. The model itself is a small conventional 1D-CNN, and FedAvg is a standard method. […] the study's value depends on stronger reproducibility and system-level evidence rather than claiming algorithmic novelty or SOTA prediction quality."

The paper agrees (`related_work.tex`, "Positioning"): "The present contribution is an engineering evaluation rather than a new optimizer."

**Gaps listed by the review, and their status in the current code.** The status column comes from reading the code. It is not a claim the review itself makes.

| Gap in `literature_review.md` | Status in current code (verified by reading) |
|---|---|
| Unverified FD002 split provenance | **Still open.** See §2.5. |
| Only two subsets, two clients | Paired suite now covers FD001–FD004. Still **two clients** everywhere. |
| Five rounds only | Still 5 rounds in `reproduce.py` and in the physical runs. A 15-round simulation exists (`results/fd002/sim_fedavg_r15_e2.json`). The 15-round physical run was "launched … results not collected" (`EXPERIMENTS.md`). |
| No FedProx/SCAFFOLD/FedAdam/personalized baseline | FedProx and FedAdam are now in `reproduce.py`. SCAFFOLD and personalization are **not implemented**. |
| No paired statistical analysis | `reproduce.py::paired_p` adds an exact sign-flip test (see §6.3). |
| No observed edge-resource metrics | Physical client logs give per-round train time and RSS (§3.4). The `profile_edge.py` Pi output is **UNMEASURED**. |
| NASA score only reported, never trained | `AsymmetricRULLoss` was added (§4). The review's line "training currently minimizes symmetric MSE" is now **stale**: MSE is still the default, but the loss can be switched. |
| No formal privacy control | **Still open.** No TLS, DP or secure aggregation (§5.6). |

---

## 2. Dataset and preprocessing decisions (NASA C-MAPSS `FD001`–`FD004`)

### 2.1 Raw data loading

- The dataset is **not stored in the repository**. `common.cmapss_zip_path()` looks for `CMAPSSData.zip` in `$CMAPSS_ZIP`, then `./CMAPSSData.zip`, then `~/CMAPSSData.zip`, and raises `FileNotFoundError` if none exists. `experiments/reproduce.py` requires `--data` ("dataset archives are not stored in this repository").
- `common.load_cmapss_raw(filename)` reads the first 26 whitespace-separated columns as `['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + ['s1'..'s21']`. `reproduce.py::COLS` is identical.
- `common.load_rul(filename)` reads `RUL_FDxxx.txt` as a flat vector indexed by `unit - 1`.

### 2.2 Why 17 input channels (`USEFUL_SENSORS`)

```python
# common.py:9 (identical list: experiments/reproduce.py::SENSORS)
USEFUL_SENSORS = ['setting1', 'setting2', 'setting3', 's2', 's3', 's4', 's7', 's8',
                  's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
```

- **3 operating settings + 14 sensors = 17 channels**, out of 26 parsed columns. `unit` and `cycle` are identifiers, so 24 candidate signal columns exist. The 7 sensors not used are `s1, s5, s6, s10, s16, s18, s19`.
- **Why the settings are included.** The comment at `common.py:8` reads: "Added setting1, setting2, setting3 for heterogeneous operating conditions". The paper's method section describes the input as "three operating settings and 14 sensors".
- **Why those 7 sensors are excluded.** The repository gives no written rationale or computed variance table for this choice. The brief's phrase "14 non-constant sensors" matches the usual C-MAPSS practice, but **no file here computes or states sensor variance**, so this guide treats the exclusion as a fixed design input and not a verified property. (The README says `diagnostics.py` attributes predictions to "the 14 fault-bearing sensors"; see Part 2.)

### 2.3 Window length `SEQ_LEN = 30` and RUL cap `MAX_RUL = 125`

```python
SEQ_LEN = 30     # common.py:10   (reproduce.py: WINDOW = 30)
MAX_RUL = 125    # common.py:11   (reproduce.py: CAP = 125)
```

- **Training targets** (`common.extract_supervised_sequences`): sliding windows of 30 cycles with stride 1. The target is `min(MAX_RUL, max_cycle - cycles[i + SEQ_LEN - 1])`, so RUL is measured at the window's **last** cycle. `reproduce.read_data` does the same with `np.lib.stride_tricks.sliding_window_view`, `ends = g.cycle[WINDOW-1:]` and `np.minimum(CAP, max - ends)`.
- **Piecewise-linear target.** `paper/sections/method.tex` defines $y_t=\min(R_{\max}, T-t)$ with $R_{\max}=125$ as a "piece-wise linear target". The README calls it "the conventional 125-cycle cap". `PDM_FIXES.md` P26 required the benchmark to print "True RUL is capped at 125 cycles during evaluation as per standard CMAPSS practice."
- **The cap applies to test labels too.** `extract_last_sequence` returns `min(MAX_RUL, true_rul)`, and `reproduce.py` uses `min(CAP, rul[unit-1])`.
- **Why 30.** The repository gives no ablation or written rationale for 30 beyond calling it the fixed protocol. `related_work.tex` says only: "Our paired suite uses a 30-cycle input, a 125-cycle cap, and one terminal window per official test engine". A 30-cycle window also sets the CNN's flatten size (30 → 15 → 7 after two `MaxPool1d(2)`; see §3.1).
- **Training engines shorter than 30 cycles** produce no windows: `range(len(data) - SEQ_LEN + 1)` is empty, and `reproduce.py` checks `if len(v) >= WINDOW`.

### 2.4 Short test engines: left-padding instead of dropping (audit fix)

The current code in `common.extract_last_sequence`:

```python
if len(data) < SEQ_LEN:
    # Short test engines used to be silently dropped (biasing the metric to
    # longer-lived engines). Left-pad by repeating the first reading instead.
    pad = np.repeat(data[:1], SEQ_LEN - len(data), axis=0)
    data = np.concatenate([pad, data], axis=0)
seq = data[-SEQ_LEN:]
```

`reproduce.read_data` has the same branch: "Keep every official test engine in evaluation. Repeat its first observed row as deterministic left padding."

**Why.** Dropping short test engines removes the engines observed for only a few cycles. The code comment says this biases the metric "to longer-lived engines".

**Audit trail.**
- `results/EXPERIMENTS.md`: "A later audit found the loader omitted short official test sequences. All pre-padding runs and their summary are quarantined under `results/reproducible/interrupted/pre_short-test-padding/` and excluded from study tables. The current run left-pads those test sequences so every official test engine is scored."
- Legacy Flower side: "Old pre-fix results kept in `results/fd002/legacy_pre_fix/` (evaluated 253 of 259 engines)." (`EXPERIMENTS.md`). Six FD002 engines were missing before the fix.
- A regression gate exists: `sanity_check.py` asserts `n == 259` ("FD002 test set must contain all 259 engines").
- Engines now scored (`EXPERIMENTS.md`, and `test_engines` in the raw JSON): **FD001 100, FD002 259, FD003 100, FD004 248**.

> `PDM_FIXES.md` does not describe the padding fix itself. Its rounds 8–9 cover P20–P28. The fix is documented in `results/EXPERIMENTS.md` and in the code comments.

### 2.5 Scaling: `global_scale.json`, and why centralized scaling is a known PoC caveat

There are **two different scaling procedures**, one per codebase.

| | Flower demo / controlled loss study | Paired suite (`reproduce.py`, `profile_edge.py`) |
|---|---|---|
| Producer | `compute_global_scale.py` → `global_scale.json` (FD002). `benchmark_asymmetric_loss.compute_scales_from_train` for FD001. | Computed inline in `read_data` / `load_client` |
| Rows used | **First 50 cycles of every training engine** (`groupby('unit').head(50)`), commented "healthy regime. Test data is never used." | **All rows of the training split** (`tr[SENSORS].min()/max()`) |
| Formula | `(x - min) / (max - min + 1e-7)` (`common.apply_scale`) | Same, then `.clip(-5, 5)` |
| Extra field | `train_mean_rul` = mean capped target over all training windows (used for the constant baseline) | — |

Contents of `global_scale.json`: FD002 healthy-regime min/max for all 17 channels (for example, `setting3` min 60.0 / max 100.0, `s9` min 7990.36 / max 9081.2) and `"train_mean_rul": 80.74004197407993`.

**Why it is a caveat.** Both procedures compute statistics over **all clients' training engines in one place**:
- `README.md`: "The scaler in the research runner is fitted on all training engines, so the preprocessing step is centralized."
- `literature_review.md`: "The coordinator also receives global scaling values calculated centrally by `compute_global_scale.py`."
- `discussion.tex`: "The global min–max scaler uses all training engines and therefore represents a centralized preprocessing step."
- `architecture.tex`: "centrally computed scaling statistics are shared with clients."

Practical consequence: `deploy.sh` rsyncs the whole working tree, including `global_scale.json` and `partition.json`, to both Pis. Clients do not compute their own scaling. A deployment that is federated end to end would need federated or per-client normalization. **The repository does not implement this.**

> Using the first 50 cycles means later (degraded) readings can fall outside [0, 1]. The paired suite explicitly clips to [-5, 5]. `common.apply_scale` does **not** clip.

### 2.6 Operating-regime clustering (`k = 6`, FD002/FD004)

`reproduce.operating_clusters(frame, k=6)`:
1. Takes cycle-level `setting1..3` from the **raw** (pre-split, unscaled) training frame and z-scores them (any std of 0 is replaced by 1).
2. **Deterministic farthest-point initialization.** The first center is the row with minimum z-scored `setting1`. Each next center is the row farthest from the existing centers.
3. Up to 50 Lloyd iterations, stopping on `np.allclose`.
4. Returns `centers, mean, scale`.

`window_regime_data` labels each window by the nearest center at its **endpoint** (`x[:, -1, :3]`). It groups windows into pseudo-units with key `-(unit*6 + regime + 1)`. `decode_regime_key` inverts the key.

Client assignment: centers are sorted lexicographically (`sorted(range(len(centers)), key=lambda i: tuple(centers[i]))`). The first 3 regimes go to client 0 and the last 3 to client 1.

**Why window-level and not engine-level.** `EXPERIMENTS.md`: "An initial partial run exposed a false regime partition for single-regime FD001; inspecting FD002 showed that per-engine medians would also yield an extreme 17/243 engine split." Those outputs are in `interrupted/seed42_pre_regime-fix/`.

**Consequence: the split is not a set of disjoint sites.** From `results/reproducible/fd002_regime_seed42.json`:

| Dataset | `client_engines` | `client_engine_overlap` | `client_train_groups` | `client_training_windows` | `client_test_engines` |
|---|---|---|---|---|---|
| FD002 regime | [260, 260] | **260** | [780, 780] | [20840, 25379] | [112, 147] |
| FD004 regime | [249, 249] | **249** | [747, 747] | [24250, 29778] | [102, 146] |

Every training engine contributes windows to both clients. `README.md`: "this split tests regime shift rather than disjoint plant fleets." The regime split is only defined for FD002/FD004. FD001/FD003 are single-condition, and `reproduce.py` prints "has no valid regime split; skipped".

### 2.7 Non-IID partition designs

**Paired suite** (`reproduce.partitions(train, seed)`, always called with `seed=0`, so **the split is the same for all five seeds**; seeds change only initialization and data order):

| Name | Construction | Datasets run | Seed-42 metadata |
|---|---|---|---|
| `iid` | Shuffle unit IDs with `default_rng(0)` and alternate them (`[::2]`, `[1::2]`). Test split: sorted test IDs alternated. | FD001–FD004 | FD001 [50, 50] engines / [8976, 8755] windows. FD002 [130, 130] / [22365, 23854]. FD003 [50, 50] / [9917, 11903]. FD004 [125, 124] / [27369, 26659]. |
| `engine_imbalance` | First `len//4` shuffled engines → client 0, the rest → client 1 ("1:3 engine ratio", whole engines kept intact). Test split: separately shuffled test IDs with the same 1:3 cut. | FD002, FD004 only (others skipped) | FD002 [65, 195] / [12008, 34211]. FD004 [62, 187] / [14222, 39806]. |
| `regime` | §2.6 | FD002, FD004 only | §2.6 |

**Flower demo** (`partition.json`): keys `"1"` and `"2"`, 130 FD002 unit IDs each (260 total, IDs 1–260). `common.load_partition` casts the IDs to `int`.

**Provenance warning for `partition.json`.** `setup_fd002.py` would write a **lifespan-sorted** split (shortest half → client 1, longest half → client 2; it hard-codes `/home/eldemaster/CMAPSSData.zip`). `literature_review.md` states: "This is lifespan/label skew, not an operating-condition partition. The result provenance remains unresolved because logs describe an operating-condition split." The README repeats: "the archived client partition provenance remains unresolved … Do not interpret these results as an operating-condition partition until the assignments are recovered and recorded." The unit IDs in the committed `partition.json` are listed in non-monotonic order. Without the dataset in the repository, this guide **cannot verify** whether the file matches `setup_fd002.py`'s output. `inspect_fd002.py` only prints `describe()` of the FD002 per-unit lifespans.

**Controlled loss study** (`benchmark_asymmetric_loss.run_subset`): FD002 uses `partition.json`. FD001 uses an **odd/even unit-ID split** (`u % 2 == 1` → client 1, even → client 2).

### 2.8 Train/validation split by engine (`split_units`)

```python
def split_units(units, val_frac=0.1, seed=0):
    """Unit-level (not window-level) split so val windows never share an engine with train."""
```

`split_units` sorts and shuffles unit IDs with `RandomState(seed)` and holds out `max(1, round(0.1·n))` units. Overlapping 30-cycle windows from one engine are highly correlated. A window-level split would leak near-duplicate windows into validation. A unit-level split prevents this. `client.py`, `fed_sim.py`, `final_benchmark.py` and `benchmark_asymmetric_loss.py` all use it. The paired suite (`reproduce.py`) has **no validation split**: `discussion.tex` lists "no validation-based hyperparameter selection".

---

## 3. Model architecture and edge resource decisions

### 3.1 `CmapssRegressor` (`common.py`) ≡ `CNN` (`experiments/reproduce.py`)

Input tensor: `(batch, 30, 17)`. `forward` transposes it to `(batch, 17, 30)` for `Conv1d`.

| Layer | Definition | Output shape | Parameters |
|---|---|---|---|
| `conv1` | `Conv1d(17, 32, kernel_size=3, padding=1)` + ReLU | (B, 32, 30) | 17·32·3 + 32 = **1,664** |
| `pool1` | `MaxPool1d(2)` | (B, 32, 15) | 0 |
| `conv2` | `Conv1d(32, 64, kernel_size=3, padding=1)` + ReLU | (B, 64, 15) | 32·64·3 + 64 = **6,208** |
| `pool2` | `MaxPool1d(2)` (floor) | (B, 64, 7) | 0 |
| `flatten` | | (B, 448) | 0 |
| `fc1` | `Linear(64*7=448, 32)` + ReLU | (B, 32) | 448·32 + 32 = **14,368** |
| `fc2` | `Linear(32, 1)`, then `.squeeze(-1)` | (B,) | 32 + 1 = **33** |
| **Total** | | | **22,273** |

- Sizes: 22,273 × 4 bytes (float32) = **89,092 bytes** (`model_tensor_bytes` in `results/edge/local-smoke-macos.json`). One downlink plus one uplink = **178,184 bytes** per round (`communication_tensor_payload_bytes_per_round`). That is ≈178.2 kB in decimal units, or ≈174.0 KiB.
- `paper/sections/method.tex` states the same: "(22 273 parameters) … The float32 model is 89 092 bytes; one downlink plus one uplink is 178 184 tensor bytes per round, excluding protocol overhead."
- **Payload definition** (`profile_edge.py`): `payload = 2 * sum(p.numel() * p.element_size())`. It is computed from tensor shapes, "excludes gRPC/TLS framing", and is "not a wire capture" (`EXPERIMENTS.md`).
- The 1D-CNN has no BatchNorm or other buffers, so `state_dict()` (what Flower sends) and `parameters()` (what the payload counts) contain the same tensors.

### 3.2 `CNNLSTM` baseline (`experiments/reproduce.py`)

```python
self.conv = nn.Sequential(nn.Conv1d(17, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
self.lstm = nn.LSTM(16, 24, batch_first=True)
self.head = nn.Linear(24, 1)          # applied to z[:, -1]
```

Parameters: conv 832 + LSTM 4·24·(16+24) + 2·4·24 = 4,032 + head 25 = **4,889**. This matches the docstring "~4.9k params" and `method.tex` "4 889 parameters". It runs only as a centralized ablation on the FD002/FD004 regime partitions, compared with `centralized`.

**Result** (`summary.csv`, global, n=5): FD002 RMSE **43.14 ± 0.018**, NASA 25,313.76 ± 294.55. FD004 RMSE **43.00 ± 0.005**, NASA 43,970.10 ± 652.10. The cross-seed std is close to zero and RMSE is close to 43, which suggests a near-constant predictor under this 10-epoch budget. The paper says: "The compact CNN-LSTM performs worse than the 1D-CNN under this training budget." The repository does not diagnose this further.

### 3.3 Why a compact 1D-CNN for Raspberry Pi-class nodes

What the repository states:
- `literature_review.md`: "The model itself is a small conventional 1D-CNN". Its value is the end-to-end physical testbed, not the architecture.
- The physical clients are recorded as "RPi `edgedataspace` (client 1), `edgedataspace2` (client 2), Cortex-A72, 3.7 GiB, torch 2.14 cpu" (`EXPERIMENTS.md`).
- The parameter count is 22k, and the per-round tensor payload is 178,184 bytes.

What the repository **does not** contain: a comparative study of model size versus Pi latency, energy or accuracy that justifies 32/64 channels. The design choice is documented, but the edge-optimality claim is **not measured**. `literature_review.md`: "The scientific claim should therefore be 'physically deployed on Raspberry Pi clients,' not 'resource-efficient' until measured."

### 3.4 Three levels of edge evidence

| Evidence | Source | Values | Status |
|---|---|---|---|
| **(a) Mac smoke profile** | `results/edge/local-smoke-macos.json` from `profile_edge.py` | macOS-27.0.1-arm64, Python 3.13.2, PyTorch 2.15.0. FD002 regime, client 0, 260 engines, 20,840 windows, 1 round × 1 epoch. **3.646 s** training. Peak RSS **426,196,992 B** (= 406.5 MiB, `\EdgeSmokePeakRssMiB`). Payload 178,184 B. `"mTLS": false`. | Measured, but **not Pi evidence**: "it is not a Raspberry Pi measurement and is excluded from edge claims" (`EXPERIMENTS.md`) |
| **(b) Physical Flower runs** | `results/fd002/results_physical_seed{42,100,2024,7,777}.json`, client logs as quoted in `EXPERIMENTS.md`/README | Train time per round: RPi1 ~12.8 s (16.7k windows), RPi2 ~17.3 s (25.0k windows) (seed-42 client logs). Peak RSS ~570–620 MB. The round-5 weighted-average fit metrics in the JSON are `train_time_s` 15.32–16.46 s and `peak_rss_mb` 598.1–600.7 across seeds. | **Measured on real Pis**, but through `client.py` instrumentation (`time.perf_counter`, `ru_maxrss`), not through `profile_edge.py` |
| **(c) `profile_edge.py` on Pi 4** | Would be `results/edge/rpi_fd002.json` | — | **UNMEASURED.** "Both Pi helpers remained unreachable and exited with SSH status 255. No Raspberry Pi timing or memory measurement is claimed until JSON files from those devices exist under `results/edge/`." |

Still **UNMEASURED** at every level: energy/power, temperature and throttling, real wire bytes (gRPC framing, retransmissions), TLS overhead, coordinator cost and inference latency on a Pi. (`EXPERIMENTS.md` reports Mac-only diagnostics inference timing; see Part 2.)

Implementation note: `client.peak_rss_mb()` divides `ru_maxrss` by 1e6 on macOS (bytes) and by 1e3 on Linux (KiB). On Linux the "MB" figure is therefore KiB/1000. `profile_edge.peak_rss_bytes()` multiplies by 1024 on non-Darwin systems.

---

## 4. Safety-critical objective (`AsymmetricRULLoss`)

### 4.1 Why symmetric MSE is not enough

Let $d=\hat{y}-y$. **Late** predictions ($d>0$) overestimate remaining life. `common.late_prediction_rate` documents this as "late failure prediction: y_pred > y_true", and the `AsymmetricRULLoss` docstring calls it "overestimating RUL and risking unpredicted failure". MSE weights $\pm d$ equally, so it cannot prefer conservative errors.

The evaluation metric (`common.nasa_score`; `reproduce.nasa` uses `expm1`, which is equivalent):

$$
S=\sum_i \begin{cases} e^{-d_i/13}-1, & d_i<0 \\ e^{d_i/10}-1, & d_i\ge 0\end{cases}
$$

Late errors use the smaller divisor (10), so they grow faster. `method.tex` notes that $S$ is a sum "and thus is not comparable across test sets of different size". Under MSE the measured late rates are **40.8%–63.8%** (§4.4).

### 4.2 Why not train on the raw NASA score

`common.py` docstring: "Unlike the raw exponential NASA score—whose gradients grow as `exp(|d|/10)` and explode on large early-training errors—this loss uses a smooth, differentiable asymmetric squared-error surrogate". `related_work.tex`: "Direct exponential optimization can be numerically unstable because its gradients grow exponentially with error magnitude."

Scale of the problem in a recorded run: in `results/fd002/server_mu0_42.log` the untrained model (round 0, RMSE 85.29) has NASA score **1,139,425.52**.

### 4.3 Exact formulation (`common.AsymmetricRULLoss`)

$$
\mathcal{L}(\hat{\mathbf y},\mathbf y)=\frac{1}{B}\sum_{i=1}^{B} w_i\,d_i^2,\qquad
w_i=\begin{cases}\alpha_{\text{late}} = 2.5, & d_i\ge 0\\ \alpha_{\text{early}} = 0.5, & d_i<0\end{cases}
$$

```python
d = y_pred - y_true
weights = torch.where(d >= 0, self.alpha_late, self.alpha_early)
return torch.mean(weights * (d ** 2))
```

- **Validation:** `ValueError` if either alpha is ≤ 0 or if `alpha_late <= alpha_early`.
- **Ratio:** 2.5 / 0.5 = 5× penalty on late errors (`method.tex`: "a $5\times$ penalty ratio").
- **Gradient:** $2w_i d_i / B$. It is linear in $|d|$ and equals 0 on both sides of $d=0$, so it is continuous there. The README/paper call this "$C^1$-smooth": the loss is continuously differentiable, while its second derivative jumps at 0.
- **How to select it:** `common.get_loss_fn(name)` accepts `"mse"`, or `"asymmetric"` / `"asym"` / `"asymmetric_rul"`. Clients: `client.py --loss-fn asymmetric` or `LOSS_FN=asymmetric`, with `--alpha-late/--alpha-early` or `ALPHA_LATE/ALPHA_EARLY` (`final_benchmark.py` works the same way). `fed_sim.py` accepts `--loss-fn` but **not** custom alphas: it calls `get_loss_fn(a.loss_fn)` with the defaults. **The default everywhere is `mse`.**
- **Sanity gate:** `sanity_check.run_loss_sanity_check()` asserts loss(+10) > loss(−10), finite gradients, and gradients on every `CmapssRegressor` parameter.

### 4.4 Controlled 5-seed results (`results/asymmetric_loss_comparison.json`)

Protocol (`benchmark_asymmetric_loss.py`): seeds **42, 123, 456, 789, 1024**. Local-only 10 epochs, averaged over 2 clients. FedAvg 5 rounds × 2 local epochs. Centralized 10 epochs. Batch 64, Adam lr 1e-3, 90/10 unit split. Total `elapsed_s` 1867.36. **Std is population std (`np.std`, ddof=0)**, unlike the paired suite.

| Dataset | Setup | Loss | RMSE | NASA | Late rate % |
|---|---|---|---:|---:|---:|
| FD001 (100) | Local | mse | 17.58 ± 1.01 | 722.87 ± 120.55 | 56.2 ± 6.59 |
| FD001 | Local | asym | 17.15 ± 1.50 | 434.98 ± 85.58 | 33.8 ± 4.52 |
| FD001 | FedAvg | mse | 16.98 ± 1.31 | 690.81 ± 173.43 | 63.8 ± 7.11 |
| FD001 | FedAvg | asym | 16.25 ± 1.23 | 369.13 ± 43.50 | 34.2 ± 7.98 |
| FD001 | Central | mse | 15.16 ± 0.86 | 450.37 ± 91.81 | 56.6 ± 3.32 |
| FD001 | Central | asym | 15.80 ± 0.97 | 338.73 ± 51.98 | 30.8 ± 5.15 |
| FD002 (259) | Local | mse | 23.73 ± 1.22 | 7,610.84 ± 3,611.39 | 50.27 ± 7.75 |
| FD002 | Local | asym | 26.30 ± 2.08 | 3,932.18 ± 829.67 | 29.5 ± 6.02 |
| FD002 | FedAvg | mse | 22.33 ± 1.46 | 5,499.06 ± 2,962.80 | 48.88 ± 8.73 |
| FD002 | FedAvg | asym | 26.07 ± 1.69 | 3,412.62 ± 429.82 | 25.25 ± 6.33 |
| FD002 | Central | mse | 20.17 ± 0.31 | 2,769.50 ± 494.61 | 40.85 ± 4.97 |
| FD002 | Central | asym | 25.79 ± 2.74 | 3,630.22 ± 1,185.19 | 20.54 ± 6.15 |

Constant baselines (train-mean RUL): FD001 RMSE 40.55, NASA 18,365.49, late 45.0%. FD002 RMSE 43.51, NASA 63,730.17, late 50.19%.

**Trade-offs** (paper macros in `paper/generated/macros.tex`, README "Findings"):
- The late rate falls in all six setups: from 40.8–63.8% to 20.5–34.2%.
- NASA score changes vs MSE: FD001 Local −39.8%, FedAvg −46.6%, Central −24.8%. FD002 Local −48.3%, FedAvg −38.0%, Central **+31.1%**. The README attributes the Central increase to "conservative early-prediction outliers such as seed 1024".
- FD002 FedAvg NASA std falls from 2,962.8 to 429.8.
- RMSE: on FD001 it improves for Local and FedAvg and rises slightly for Central (15.16 → 15.80). On FD002 it **rises in every setup**, by +10.8% to +27.9% (README).
- The repository runs no significance test for this study.

### 4.5 Two different "asymmetric" objectives — do not confuse them

| | `AsymmetricRULLoss` (`common.py`) | `'asymmetric'` in `reproduce.train` |
|---|---|---|
| Form | Weighted quadratic, α=2.5/0.5 | **Capped exponential NASA score**: `exp(clamp(-d/13, max=20)) - 1` for d<0, `exp(clamp(d/10, max=20)) - 1` for d≥0, averaged |
| Used by | `client.py`, `fed_sim.py`, `final_benchmark.py`, `benchmark_asymmetric_loss.py` | Paired suite variants `asymmetric` (central) and `federated_asymmetric` |
| Paper label | `AsymmetricRULLoss` | "Central + exp." / "FedAvg + exp." (`make_figures.METHOD_LABELS`) |

The paper keeps them separate on purpose: "does not establish that the two asymmetric objectives are interchangeable" (`discussion.tex`). The `tab:algorithms` caption says "'exp.' denotes the paired suite's capped exponential NASA-score training objective, not `AsymmetricRULLoss`."

**Results for the capped-exponential objective** (`summary.csv`, regime partitions, n=5, sample std):

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

The capped-exponential objective lowers the mean NASA score vs FedAvg on both datasets and raises RMSE. For **centralized FD004** the mean NASA score **increases** (4,329 → 7,244). The paired suite does not record late-prediction rate.

---

## 5. Federated learning design and deployment

### 5.1 Topology

```
             Ubuntu coordinator (server.py, Flower gRPC, BIND default 0.0.0.0:8081)
              │  evaluate_fn: official test_FD002 (259 engines) every round
              │  FedAvg (fraction_fit=1.0, min clients = MIN_CLIENTS=2)
     ┌────────┴─────────┐
 RPi client 1        RPi client 2         (client.py, CLIENT_ID=1|2, FL_SERVER=<ip>:8081)
 partition.json["1"] partition.json["2"]  model params only; gRPC, "SSL is disabled"
```

Hosts in the scripts: the server checks `TARGET_IP` (default `192.168.1.144`). Pis are `RPI1_HOST` default `eldemaster@192.168.1.147` and `RPI2_HOST` default `eldemaster@192.168.1.115` (`run_fd002_federated.sh`, `deploy.sh`). Server software: "`eldemaster-ubuntu`, flwr 1.37" (`EXPERIMENTS.md`).

### 5.2 Server (`server.py`)

- Environment variables: `SEED` (default **777**), `ROUNDS` (5), `MIN_CLIENTS` (2), `PROXIMAL_MU` (0.0), `LOCAL_EPOCHS` (2), `RESULTS_DIR` (`results/fd002`), `BIND` (`0.0.0.0:8081`).
- **Server-seeded initial weights.** After `set_seed(seed)` the server builds `initial_parameters` from a fresh `CmapssRegressor()`. The code comment explains why: "otherwise Flower asks a random client, making round 0 depend on which client connects first." The round-0 RMSE in the physical runs is consistent across seeds (85.18–85.39).
- **Strategy:** `fl.server.strategy.FedAvg` with `fraction_fit=fraction_evaluate=1.0` and all `min_*_clients = n_clients`. `on_fit_config_fn` returns `{"proximal_mu": mu, "local_epochs": local_epochs}`. Fit and evaluate metrics are aggregated with `weighted_avg` (sample-weighted mean).
- **FedProx via `PROXIMAL_MU`.** The server strategy stays FedAvg. FedProx exists only as the client-side proximal term (§5.3) that `proximal_mu > 0` turns on. The output name changes to `results_physical_fedprox_mu{mu}_seed{seed}.json`.
- **Central evaluation** (`get_evaluate_fn`): the global model is scored on `build_test_set(scales)` (FD002 default) each round. It returns `model_rmse`, `baseline_rmse` (constant `train_mean_rul`) and `nasa_score`.
- **Output:** JSON with `metrics_centralized`, `metrics_distributed_fit`, `metrics_distributed` and `config`. The server exits with an error if no centralized metrics were recorded.
- **Caveat from the review:** "`server.py` evaluates the official test set every round. The test metrics must not be used for tuning or model selection".

### 5.3 Client (`client.py::SupervisedClient`)

- Required environment variables: `CLIENT_ID` (must be a key in `partition.json`) and `FL_SERVER`. Optional: `SEED` (default 42). The client seed is `set_seed(seed + int(cid))`. `PDM_FIXES.md` round 9: "the client seed used to be fixed at 42, so the old FD001 std only reflects the server seed."
- Data: `train_FD002.txt` scaled with `global_scale.json`, then `split_units(part[cid], 0.1, seed=seed)` and `make_loader(X, y, 64, seed=seed+cid)`.
- `fit`: loads the global parameters and **snapshots them as `glob` for the proximal reference**. It trains `local_epochs` with `train_epochs(..., mu=mu, global_params=glob, optimizer=self.optimizer, loss_fn=self.loss_fn)` and returns parameters, `len(dataset)` and `{"train_time_s", "peak_rss_mb"}`.
- **The Adam optimizer persists across rounds** (`self.optimizer` is created once in `__init__`). The paired suite resets Adam each round (§6.1).
- `evaluate`: RMSE on **held-out validation units** of that client. This closes `PDM_FIXES.md` P20 ("Client `evaluate()` returns a hardcoded `0.0`").
- Launched with `fl.client.start_client(server_address=fl_server, client=client.to_client())`. No certificates are passed.

**FedProx term** (`common.train_epochs`):

$$\mathcal{L}_k=\text{crit}(f(\mathbf X),\mathbf y)+\tfrac{\mu}{2}\sum_j\lVert \mathbf w_j-\mathbf w^{(t)}_j\rVert_2^2$$

`loss + 0.5 * mu * sum(((p - r) ** 2).sum() ...)`, with `ref` = detached clones of `global_params`.

### 5.4 In-process parity (`fed_sim.py`)

Docstring: "In-process federated simulation (FedAvg / FedProx) using the exact same local-training code as client.py. No network". Per round, each client loads the global state, trains with `train_epochs` (persistent per-client Adam, same as `client.py`), and the server does a sample-weighted average `sum(s[k]*n)/tot`. Each round also logs global test RMSE/NASA, per-client val RMSE and train time. **Summary std is `np.std` (ddof=0).**

Parity check (`EXPERIMENTS.md`): `sim_fedavg_r5_e2.json` gives RMSE **22.20 ± 0.42** "(matches physical run)" vs physical **22.27 ± 0.44**.

| File | Config | RMSE | NASA |
|---|---|---:|---:|
| `sim_fedavg_r5_e2.json` | FedAvg, 5 r × 2 e | 22.20 ± 0.42 | 5,990.93 ± 704.43 |
| `sim_fedprox0.01_r5_e2.json` | μ = 0.01 | 22.20 ± 0.58 | 6,051.78 ± 806.98 |
| `sim_fedprox0.1_r5_e2.json` | μ = 0.1 | 22.86 ± 0.93 | 6,977.31 ± 1,660.30 |
| `sim_fedavg_r15_e2.json` | FedAvg, 15 r × 2 e | 19.27 ± 1.30 | 2,741.56 ± 905.88 |

The README notes that FedProx gave "no gain" and that the 15-round result is "not yet compared with an equal-epoch centralized run". `EXPERIMENTS.md`: "Early stopping by client val RMSE: no gain in sim (19.05 vs 19.27, from r15 history)."

### 5.5 Deployment scripts

- **`deploy.sh`**: `rsync -avz` of the working tree to `~/fedpdm_client/` on both Pis, excluding `.git`, `results`, `venv` and `__pycache__`. This ships `partition.json` and `global_scale.json`.
- **`run_fd002_federated.sh`** (`set -euo pipefail`):
  1. Aborts unless `ip -4 addr show` contains `TARGET_IP`.
  2. For each seed in `SEEDS` (default `42 100 2024 7 777`): `pkill` any old `client.py` on both Pis, then start `server.py` in the background with `SEED/ROUNDS/PROXIMAL_MU`, logging to `server_mu${PROXIMAL_MU}_${seed}.log`.
  3. Wait 5 s, then launch the clients with **`ssh -f -n`** and `nohup`. `PDM_FIXES.md` round 9 lists this as required: "or the first ssh blocks".
  4. Poll the server PID with a **1200 s** timeout per seed. After the run, kill the leftover clients.
  - The script has one `ln -sf` call, which writes `server_seed.log` as a symlink to the latest log **on the server host at runtime**. The committed `results/fd002/server_seed.log` is a regular file (git mode `100644`), not a symlink.
- Physical results (`EXPERIMENTS.md`, recomputed from the five JSONs with `np.std`): **RMSE 22.27 ± 0.44, NASA 6,037 ± 1,030**. Per-seed final RMSE: 42 → 21.62, 100 → 22.91, 2024 → 21.99, 7 → 22.38, 777 → 22.47. Seed 7 has a non-monotone round-2 RMSE of 47.79.
- Legacy local baselines (`results/fd002/benchmark_e10.csv`, `final_benchmark.py --epochs 10`): constant 43.51 / 63,730. Local 25.01 ± 0.54 / 11,937 ± 2,207. Central 20.39 ± 1.11 / 3,395 ± 972.

### 5.6 Security and privacy boundaries

| Control | Status | Evidence |
|---|---|---|
| TLS / mTLS | **Not configured** | `start_client`/`start_server` are called without certificates. The server log says "gRPC server running (5 rounds), SSL is disabled" (`server_mu0_42.log`). `profile_edge.py` and `reproduce.py` record `"mTLS": False`. |
| Secure aggregation | **Not implemented** | `literature_review.md`, `architecture.tex` |
| Differential privacy | **Not implemented** | Same |
| Raw data leaves client | Not in parameter exchange | `README.md`: keeping raw windows out "does not prevent leakage from model updates" |
| Preprocessing privacy | **Centralized** | §2.5 |

`architecture.tex`: "Earlier README claims of mTLS and encrypted weights were removed because the checked-in code does not implement them." `literature_review.md` ranks TLS as "low–medium" cost and DP/secure aggregation as "medium–high".

Flower API status: the server log shows the warning "DEPRECATED FEATURE: flwr.server.start_server() is deprecated." `PDM_FIXES.md` P20 notes migration to `ServerApp`/`ClientApp` (flwr 1.37). **It is still open:** the code still calls `start_server`, and `client.py` calls `start_client` (no longer `start_numpy_client`).

---

## 6. Reproducible benchmark suite, bug-fix evolution and paper status

### 6.1 `experiments/reproduce.py` protocol

From the code and the `protocol` field written to every raw JSON:

| Item | Value |
|---|---|
| Datasets | FD001, FD002, FD003, FD004 (FD003/FD004 in full, both fault modes) |
| Seeds | 42, 100, 2024, 7, 777 |
| Partitions | `iid` (all); `regime`, `engine_imbalance` (FD002/FD004 only) |
| Variants | All runs: `centralized`, `local_only`, `fedavg`. Plus `fedprox`, `fedadam` on FD002/FD004 regime and engine_imbalance. Plus `federated_asymmetric`, `asymmetric`, `cnn_lstm` on FD002/FD004 regime. |
| Budgets | Centralized/local: 10 epochs. Federated: 5 rounds × 2 local epochs. |
| Optimizer | Adam lr 1e-3, batch **128**, **new client Adam each round** |
| FedProx | μ = 0.01 (`train_prox`) |
| FedAdam | Pseudo-gradient `g = w_global − avg`, β1 = 0.9, β2 = 0.99, server lr **0.01**, eps 1e-8, with bias correction. Moments persist across rounds. |
| Aggregation weights | Number of training windows per client |
| Initialization | One seeded `init` state shared by every variant (`initial = clone(init)`) |
| Threads | `--threads` (default 1) |
| Smoke | `--quick` → FD001, seed 42, iid, written to `results/reproducible/smoke/` |
| Metrics | Global and per-client RMSE + NASA. `local_only` global = pooled predictions of each client model on its own test split (`pooled_metrics`). |

Commands (README/EXPERIMENTS.md):

```bash
python -m pip install -r experiments/requirements.txt   # numpy==2.5.4 pandas==3.1.1 torch==2.15.0
python experiments/reproduce.py --data /path/to/CMAPSSData.zip --out results/reproducible --threads 1
python experiments/reproduce.py --summarize-only
```

Recorded environment (`EXPERIMENTS.md`): Python 3.13.2, PyTorch 2.15.0, pandas 3.1.1, NumPy 2.5.4. The root `requirements.txt` is unpinned (`flwr>=1.7`, `torch>=2.0`, `numpy`, `pandas`, `scikit-learn`). Physical Pis ran torch 2.14 (cpu).

### 6.2 5-seed summary (`results/reproducible/summary.csv`, global scope, mean ± sample std, n = 5)

The file has 288 data rows plus a header. It contains global rows, per-client rows and paired-test rows. `centralized` trains on all engines, so its value is the same across partitions of a dataset.

**IID baselines (README / paper `tab:main`)**

| Dataset | Method | RMSE | NASA |
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

**Regime partitions (FD002/FD004)**

| Dataset | Method | RMSE | NASA |
|---|---|---:|---:|
| FD002 | Centralized | 21.46 ± 1.18 | 5,762.79 ± 2,135.75 |
| FD002 | Local-only | 23.04 ± 0.43 | 5,033.02 ± 818.81 |
| FD002 | FedAvg | 23.50 ± 0.63 | 4,425.37 ± 1,367.59 |
| FD002 | FedProx | 23.30 ± 0.48 | 4,581.90 ± 1,331.95 |
| FD002 | FedAdam | **78.23 ± 23.17** | 3.29 × 10⁸ ± 6.99 × 10⁸ |
| FD004 | Centralized | 23.41 ± 0.49 | 4,329.01 ± 1,650.39 |
| FD004 | Local-only | 26.85 ± 0.80 | 8,006.51 ± 2,168.07 |
| FD004 | FedAvg | 26.78 ± 0.20 | 7,654.82 ± 693.72 |
| FD004 | FedProx | 26.88 ± 0.81 | 7,638.47 ± 1,490.37 |
| FD004 | FedAdam | **56.62 ± 16.01** | 9.00 × 10⁶ ± 1.84 × 10⁷ |

(The exp-loss and CNN-LSTM rows are in §4.5 and §3.2.)

> **Centralized differs between iid and regime.** The centralized RMSE is the same for FD002 iid and engine_imbalance (22.12) but different for regime (21.46). The regime run trains on regime-grouped pseudo-units (`tr_run`), which changes how windows are grouped and so the loader order. It is also scored on the regime-keyed test set. The training windows themselves are the same.

**Engine-count imbalance (FD002/FD004)**

| Dataset | Method | RMSE | NASA |
|---|---|---:|---:|
| FD002 | Local-only | 26.90 ± 1.83 | 13,062.97 ± 11,870.18 |
| FD002 | FedAvg | 22.36 ± 0.39 | 4,073.82 ± 970.12 |
| FD002 | FedProx | 22.30 ± 0.82 | 4,170.31 ± 1,185.30 |
| FD002 | FedAdam | 59.37 ± 16.21 | 1.05 × 10⁷ ± 2.08 × 10⁷ |
| FD004 | Local-only | 27.82 ± 1.29 | 13,933.31 ± 5,764.98 |
| FD004 | FedAvg | 26.55 ± 0.73 | 8,205.18 ± 4,119.58 |
| FD004 | FedProx | 26.33 ± 0.77 | 7,748.58 ± 3,981.96 |
| FD004 | FedAdam | 49.66 ± 12.60 | 2.14 × 10⁶ ± 4.49 × 10⁶ |

### 6.3 Paired exact sign-flip test and its limitation

```python
def paired_p(a, b):
    d = a - b                       # seed-matched differences
    observed = |mean(d)|
    # enumerate all 2^n sign vectors s in {-1, +1}^n
    p = #{ |mean(d*s)| >= observed - 1e-12 } / 2^n
```

The test raises an error for n > 20. With **n = 5** there are 2⁵ = 32 sign patterns. The observed pattern and its full sign flip always give the same |mean| (|mean(−d)| = |mean(d)|), so at least 2 of 32 patterns count as extreme. The minimum p is therefore **2/32 = 0.0625**. That is above 0.05, so **no comparison can reach p < 0.05** regardless of effect size. `EXPERIMENTS.md`: "Tests are exploratory; no multiple-comparison adjustment is applied." The test assumes sign-exchangeability of paired differences under the null.

Comparisons computed: `fedavg_vs_{centralized, local_only, fedprox, fedadam, federated_asymmetric}`, `centralized_vs_{asymmetric, cnn_lstm}`. Many reach the floor of 0.0625. Examples: every FedAvg-vs-FedAdam comparison, and FD004 iid FedAvg vs both baselines on both metrics. Others are well above the floor, for example FD002 iid `fedavg_vs_local_only` RMSE p = 0.8125 and FD004 regime `fedavg_vs_fedprox` p = 1.0.

`literature_review.md` recommends "at least 10–20 paired seeds". The current suite has 5.

### 6.4 FedAdam divergence

Under the fixed server lr = 0.01, FedAdam reaches RMSE 78.23 (FD002 regime) and 56.62 (FD004 regime), and NASA scores in the millions to hundreds of millions. The paper (`results.tex`) frames this narrowly: "this is a negative result for this configuration, not evidence that all FedAdam settings fail." The repository has **no server-lr sweep**.

### 6.5 Quarantined runs (`results/reproducible/interrupted/`)

All four are excluded from `summary.csv` because `summarize()` globs only top-level `fd*_seed*.json`.

| Directory | Defect / reason (`EXPERIMENTS.md`) |
|---|---|
| `pre_32-64-cnn-alignment/` | First one-seed smoke used a **32/48-channel** CNN. Superseded when the runner was aligned with the 32/64 `CmapssRegressor`. `--quick` now writes to `smoke/` so it cannot overwrite the full matrix. |
| `seed42_pre_regime-fix/` | "False regime partition for single-regime FD001". Per-engine medians would give an extreme 17/243 FD002 split, which led to the window-level regime split. |
| `pre_variant-filtering/` | Partial run stopped "when the final method-by-partition coverage was narrowed". |
| `pre_short-test-padding/` | Loader omitted short official test sequences (§2.4). |

### 6.6 Bug-fix history from `PDM_FIXES.md` (legacy Flower pipeline)

Audit round 8 (2026-09-18) and round 9 (2026-10-23):

| ID | Problem | Resolution recorded |
|---|---|---|
| P23 | Data parity: baselines trained on different units than federated | `final_benchmark.py` filters to the same partition units. Current code uses `split_units(units, 0.1, seed)`, "same train units as federated clients". |
| P24 | Federated row not comparable | Five physical seeds aggregated (FD001: 14.92 ± 0.46 RMSE, NASA 394.8 ± 51.1). |
| P11 | Seed count | 5 seeds for all rows |
| P26 | Capping undocumented | Note printed with the table |
| P25 | Repo hygiene | `__pycache__/` gitignored. `plot_explanatory_physical.py` deleted. |
| **P21** | Sanity gate crashed (`load_cmapss_raw()` missing `filename`), had a duplicated assert and an inert `assert ... if "trained_rmse" in locals() else True`, and tested the internal holdout instead of the official test set | **Closed** in round 9 ("15ad636"). The current `sanity_check.py` uses the official FD002 test set, asserts `n == 259`, `untrained > baseline` and `trained < 0.75·baseline`, plus the loss checks. |
| P28 | Work left uncommitted | Closed in round 9 |
| P27 | FD001 is IID, so nothing for FL to correct | Moved to FD002. Results split into `results/fd001/` and `results/fd002/`. |
| P20 | `evaluate()` returned 0.0; deprecated Flower API | `evaluate` fixed (§5.3). API deprecation **still open**. |

Round-9 legacy FD002 numbers in `PDM_FIXES.md` (FedAvg 21.98 ± 0.99) are **superseded** by the post-padding physical rerun (22.27 ± 0.44, `EXPERIMENTS.md`). The FD001 table "is from the pre-audit pipeline and was **not re-run**" (README).

### 6.7 Paper status (`paper/`)

- **Build:** `make` runs `python3 make_figures.py` and then `latexmk -pdf`. `make_figures.py` uses only the standard library. It reads `summary.csv`, `asymmetric_loss_comparison.json` and `local-smoke-macos.json`, skips rows with `n < 5`, and "incomplete or unexpected five-seed inputs fail the build" (`paper/README.md`). It writes `paper/generated/{macros,table_main,table_algorithms,table_partitions,table_clients,table_significance,table_asymmetric}.tex`.
- **Remaining `\todo` markers** (`grep '\\todo' paper/`):
  1. `main.tex:20` — `\todo{University of Oulu co-authors}`
  2. `main.tex:21` — `\todo{Oulu affiliation(s)}`
  3. `main.tex:39` — `\todo{AI-use disclosure pending}`

  No `\todo` remains in `paper/sections/*.tex`.
- **Claims the paper explicitly avoids:** SOTA accuracy, Raspberry Pi efficiency, encrypted transport and formal privacy (abstract, introduction, discussion).

---

## Appendix A — File-by-file reference

| File | Key symbols / behavior |
|---|---|
| `common.py` | `USEFUL_SENSORS`, `SEQ_LEN`, `MAX_RUL`, `set_seed`, `cmapss_zip_path`, `load_cmapss_raw`, `CmapssRegressor`, `extract_supervised_sequences`, `extract_last_sequence`, `nasa_score`, `late_prediction_rate`, `AsymmetricRULLoss`, `get_loss_fn`, `load_rul`, `apply_scale`, `build_test_set(scales, subset='FD002')`, `load_scales`, `load_partition`, `make_xy`, `make_loader(batch_size=64)`, `split_units`, `train_epochs(mu, global_params, optimizer, loss_fn)`, `predict`, `rmse` |
| `client.py` | `SupervisedClient(NumPyClient)`, `peak_rss_mb`. Env: `CLIENT_ID`, `FL_SERVER`, `SEED`, `LOSS_FN`, `ALPHA_LATE`, `ALPHA_EARLY` |
| `server.py` | `get_evaluate_fn`, `weighted_avg`. Env: `SEED`, `ROUNDS`, `MIN_CLIENTS`, `PROXIMAL_MU`, `LOCAL_EPOCHS`, `RESULTS_DIR`, `BIND` |
| `fed_sim.py` | `run(seed, rounds, local_epochs, mu, lr, …, loss_fn)`. CLI `--seeds --rounds --epochs --mu --lr --loss-fn --out` |
| `final_benchmark.py` | `run_benchmark`. Writes `results/fd002/benchmark_e{epochs}[_{loss}].{json,csv}`. Reads physical FedAvg JSONs. |
| `benchmark_asymmetric_loss.py` | `compute_scales_from_train`, `eval_metrics`, `run_fedavg`, `run_subset`, `main` |
| `compute_global_scale.py` | FD002 first-50-cycle min/max + `train_mean_rul` → `global_scale.json` |
| `setup_fd002.py` | Lifespan-sorted 2-way split → `partition.json` (hard-coded zip path) |
| `sanity_check.py` | `run_loss_sanity_check`, `run_sanity_check` + asserts |
| `inspect_fd002.py` | Prints FD002 lifespan `describe()` |
| `plot_rul.py` | Reads `results_physical.json` (keys `losses_centralized`, `metrics_centralized`), writes `rul_metrics.png`. **Legacy**: that filename/schema is not what the current `server.py` writes (no `losses_centralized` key). |
| `deploy.sh`, `run_fd002_federated.sh` | §5.5 |
| `experiments/reproduce.py` | `CNN`, `CNNLSTM`, `read_data`, `nasa`, `metrics`, `pooled_metrics`, `operating_clusters`, `window_regime_data`, `decode_regime_key`, `partitions`, `tensors`, `train`, `clone`, `run_variant`, `train_prox`, `paired_p`, `main`, `summarize` |
| `experiments/profile_edge.py` | `load_client`, `peak_rss_bytes`, `main`. Defaults: FD002, regime, client 0, 5 rounds, 2 epochs, 1 thread. Each round trains a **fresh** `CNN` (cost profiling, not convergence). |

## Appendix B — Known inconsistencies and caveats found while writing this guide

1. **Two "asymmetric" losses** (§4.5): `AsymmetricRULLoss` (quadratic) vs `reproduce.py` `'asymmetric'` (capped exponential).
2. **Two scalers** (§2.5): first-50-cycle FD002 statistics (`global_scale.json`) vs full-train statistics with clipping (`reproduce.py`).
3. **Two std conventions:** `summary.csv` uses sample std (ddof=1). `fed_sim.py`, `final_benchmark.py`, `benchmark_asymmetric_loss.py` and the README's legacy FD002 table use `np.std` (ddof=0).
4. **Two seed sets:** paired suite and physical runs use `42 100 2024 7 777`. The controlled loss study uses `42 123 456 789 1024`.
5. **Batch size 64** (Flower/legacy/loss study) vs **128** (paired suite).
6. **Adam state:** persistent across rounds in `client.py`/`fed_sim.py`/`benchmark_asymmetric_loss.run_fedavg`. Reset every round in `reproduce.py`.
7. **`partition.json` provenance** is unresolved (§2.7).
8. **`docs/literature_review.md` is partly stale.** It predates `AsymmetricRULLoss`, FedProx/FedAdam in the suite and the paired tests (§1.4).
9. **`plot_rul.py`** expects a legacy result schema.
10. **`paired_p` floor** = 0.0625 at n = 5 (§6.3).
11. **Alpha overrides are not uniform.** `client.py` and `final_benchmark.py` accept `--alpha-late/--alpha-early` (or `ALPHA_LATE/ALPHA_EARLY`). `fed_sim.py` accepts only `--loss-fn`, so it always uses α = 2.5/0.5. The README lists all three scripts as able to select the loss, but it does not say which ones accept alpha overrides.

## Appendix C — Verification ledger

Each cross-check below was done by reading the cited file during authoring (and, where noted, by recomputing).

| Claim | Check |
|---|---|
| 22,273 params / 89,092 B / 178,184 B | Recomputed layer by layer (§3.1). Matches `local-smoke-macos.json` and `method.tex`. |
| CNN-LSTM 4,889 params | Recomputed (§3.2). Matches `method.tex`. |
| Physical FD002 22.27 ± 0.44 / 6,037 ± 1,030 | Recomputed from 5 JSONs with `np.std` → 22.273 ± 0.438, 6,037.1 ± 1,029.9 |
| Peak RSS 406.5 MiB | 426,196,992 / 1024² = 406.45. Matches `\EdgeSmokePeakRssMiB`. |
| Summary tables | Copied from `summary.csv` global rows (rounded to 2 dp) |
| Asymmetric table | Copied from `asymmetric_loss_comparison.json` `summary` |
| Partition metadata | Read from `results/reproducible/*_seed42.json` |
| `\todo` list | `grep -rn '\\todo' paper/` |
| "SSL is disabled" | `results/fd002/server_mu0_42.log` |
| SME / multi-plant | Searched; not present (§1.3) |
