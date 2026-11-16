# Local edge diagnostics: which subsystem is driving the RUL drop?

`diagnostics.py` adds per-engine degradation attribution to FedPdM-Edge. It runs on the edge node that owns the telemetry, reuses `common.py` (`CmapssRegressor`, `USEFUL_SENSORS`, `SEQ_LEN`), and outputs aggregate shares, standardized drift scores and a text summary. It does not output raw sensor readings and does not open a network connection.

## Evaluation boundary

C-MAPSS provides a unit-level RUL target only. `readme.txt` in `CMAPSSData.zip` says FD001/FD002 have one fault mode (HPC degradation) and FD003/FD004 have two (HPC degradation, fan degradation), but no per-unit fault-mode or component label is published. **No supervised component accuracy is claimed.** The module is validated in three ways instead:

- physical consistency on the single-fault datasets
- unsupervised fault-mode separability on the two-fault datasets
- edge cost

The attribution explains the model's prediction. It is not a component diagnosis.

## Channel verification and subsystem map

`readme.txt` defines the 26 columns as unit, cycle, operational settings 1–3, then sensor measurements 1–21. The 21 sensors appear in the same order as Table 2 of Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation* (PHM08), which is also in the archive. The sensors are therefore read as s1 = T2 … s21 = W32, the usual reading of that table.

`common.USEFUL_SENSORS` has 17 model inputs. `setting1..3` are operating conditions. The model receives them as inputs, but they are **never** assigned fault responsibility. The 14 sensors that can carry fault responsibility are mapped to four subsystems. Python (`diagnostics.SUBSYSTEMS`) and the dashboard use the same names.

| Sensor | Symbol | Description (Table 2) | Subsystem |
|---|---|---|---|
| s2 | T24 | Total temperature at LPC outlet | LPC |
| s3 | T30 | Total temperature at HPC outlet | HPC |
| s7 | P30 | Total pressure at HPC outlet | HPC |
| s9 | Nc | Physical core speed | HPC |
| s11 | Ps30 | Static pressure at HPC outlet | HPC |
| s14 | NRc | Corrected core speed | HPC |
| s8 | Nf | Physical fan speed | Fan |
| s13 | NRf | Corrected fan speed | Fan |
| s15 | BPR | Bypass ratio | Fan |
| s4 | T50 | Total temperature at LPT outlet | Turbine |
| s12 | phi | Ratio of fuel flow to Ps30 | Turbine (combustor) |
| s17 | htBleed | Bleed enthalpy | Turbine |
| s20 | W31 | HPT coolant bleed | Turbine |
| s21 | W32 | LPT coolant bleed | Turbine |

"Turbine" here means the combustor plus the HPT/LPT hot section. Sensors are assigned by **where they are measured**. In the gas path, a fault in one module also shifts sensors located in other modules. For example, HPC degradation moves fan speed and the coolant bleeds. A subsystem share therefore says which instrumented module dominates the model's evidence. It does not say which module is faulty. Subsystems also have different sensor counts (Fan 3, LPC 1, HPC 5, Turbine 5), so the validation reports the sensor-count chance rate next to every rate.

## Method

For each 30-cycle window (`SEQ_LEN`):

**a) Attribution.** `x` is the min-max-scaled window that `CmapssRegressor` sees, `ŷ = f(x)` is the predicted RUL, and `b` is the engine's healthy-baseline expectation in the same scaled space (see b). Then

```
A_c     = (1/W) Σ_t | (x[t,c] − b[t,c]) · ∂ŷ/∂x[t,c] |        (method="baseline_x_gradient", default)
A_c     = 0  for c ∈ {setting1, setting2, setting3}
share_c = 100 · A_c / Σ_{14 sensors} A_c
share_s = Σ_{c ∈ s} share_c       (Fan, LPC, HPC, Turbine; sums to 100)
```

Two other methods are available. `method="input_x_gradient"` sets `b = 0`, giving `|x·∂ŷ/∂x|`. `method="gradient"` uses the plain saliency `|∂ŷ/∂x|`. If all attributions are zero, the shares fall back to a uniform split. `round_to_total` uses largest-remainder rounding so that the dashboard's one-decimal shares still add up to exactly 100.0.

**b) Healthy-baseline drift.** This step does not use the model.

1. Each raw cycle is assigned to the nearest operating regime. Regimes are k-means centroids over standardized settings: k = 1 for FD001/FD003 and k = 6 for FD002/FD004.
2. Each cycle is standardized with that regime's fleet mean and standard deviation, giving `u`.
3. The engine's mean `u` over its first 30 cycles is its healthy offset.
4. Drift is `z = (u − offset) / σ_healthy`. Here `σ_healthy` is the pooled within-engine noise over the first 30 cycles across the local fleet.
5. Window drift is the mean of `z` over the window.

The same healthy offset gives the attribution baseline: `b = scale(μ_regime + σ_regime · offset)`. For the settings channels, `b` is set equal to `x`. Because of the regime step, a regime switch in FD002/FD004 does not show up as drift (see `tests/test_diagnostics.py`).

**c) Maintenance summary** (`maintenance_summary`, `EdgeDiagnostics.summarize`). The summary contains:

- predicted RUL and severity: critical ≤ 20 cycles, warning ≤ 45, otherwise healthy (the dashboard's thresholds)
- primary subsystem and subsystem shares
- top-3 contributing sensors and top-3 drifting sensors, with Table 2 names and signed σ
- primary drift subsystem and drift index
- EN/IT text
- `privacy` metadata (`raw_sensor_bytes_transmitted: 0`, `contains_raw_sensor_values: false`)
- the evaluation-boundary string

Usage:

```python
import diagnostics as D
bl = D.fit_healthy_baseline(df_train_raw, D.N_REGIMES["FD002"])
edge = D.EdgeDiagnostics(model, scales, bl)
diag = edge.diagnose_engine(df_engine_raw, "FD002-U65")
summary = edge.summarize(diag)          # latest window
```

## Validation (`experiments/validate_diagnostics.py`)

```bash
CMAPSS_ZIP=/path/CMAPSSData.zip python experiments/validate_diagnostics.py --out results/diagnostics_validation.json
```

The validation setup:

- **Seeds:** 7, 42, 100, 777, 2024.
- **Model:** one `CmapssRegressor` per seed, trained with `AsymmetricRULLoss` for 8 epochs on 80% of training units. This is an isolated edge-node model, not the paper's federated runs.
- **Scaler:** `compute_scales_from_train`.
- **Official test RMSE (mean ± sd over 5 seeds):** FD001 16.80 ± 1.42, FD002 23.74 ± 3.08, FD003 17.46 ± 2.68, FD004 26.44 ± 4.79.

Raw per-seed outputs are in `results/diagnostics_validation.json`. `experiments/reproduce.py`, `results/reproducible/` and `paper/` were not touched.

### a) Single fault mode (FD001, FD002): consistency as true RUL falls below 30

The population is windows with true RUL < 30 from held-out training units (run to failure) and official test units: 907 windows for FD001 and 2,586 for FD002 (seed 7 split).

| Metric (mean ± sd, 5 seeds) | FD001 | FD002 | Chance (sensor count) |
|---|---:|---:|---:|
| Primary attribution subsystem = HPC | 26.6 ± 5.5 % | 59.6 ± 5.0 % | 35.7 % |
| Primary attribution subsystem ∈ core (HPC ∪ Turbine) | 99.4 ± 0.8 % | 100.0 ± 0.0 % | 71.4 % |
| Top-3 attributed sensors in core | 94.9 ± 5.1 % | 73.3 ± 1.7 % | 71.4 % |
| Top-3 drifting sensors in core (model-free) | 82.0 % | 76.7 % | 71.4 % |
| Primary *drift* subsystem = HPC (model-free) | 51.4 % | 76.2 % | 35.7 % |

The table supports four findings:

- **The core dominates.** Attribution near end of life is almost entirely in the core (HPC + combustor/turbine) on both datasets, against a 71% chance rate. Fan and LPC are almost never primary.
- **HPC versus Turbine.** On FD002, HPC is the primary subsystem in about 60% of low-RUL windows, against 36% chance. On FD001, attribution mass sits mostly on the hot-section side: phi (s12), and P30 / Ps30 (s7/s11). Ps30 (s11) is in the top-3 in 79% of windows (seed 7), and P30 / phi move together in the HPC fault signature. HPC alone is primary in only 27% of windows, which is below chance.
  - This depends on assigning phi to the combustor. Read the HPC-only rate with the mapping caveat above in mind.
- **Drift agrees with physics.** The model-free drift ranks HPC first (FD001 mean |z| 4.4 for HPC vs 2.9 for Fan; FD002 3.7 vs 2.5). Ps30 (s11) is the most frequent top-3 drifting sensor on both datasets (85% / 78%).
- **The baseline matters.** Plain gradient saliency (`|∂ŷ/∂x|`) almost never ranks HPC first (0.8% / 1.4%) but still ranks the core first. It describes model sensitivity rather than the engine's state. The baseline-relative form `(x − b)·∂ŷ/∂x` is therefore the default.

Subsystem shares by true-RUL bucket (seed 7, ≥100 → 30–100 → <30):

- FD001: HPC 30.3 → 32.2 → 33.4%, Fan 16.9 → 21.7%, Turbine 47.6 → 37.7%.
- FD002: HPC 37.2 → 40.2 → 41.8%.

HPC share rises as RUL falls, but the trend is small. The large signal is the core-versus-Fan/LPC split.

### b) Two fault modes (FD003, FD004): unsupervised separation

**Population.** Each clustered unit's profile is the mean of its last 10 windows. FD003 uses 100 training units plus 20 test units with official RUL ≤ 30. FD004 uses 249 training units plus 53 test units.

**Clustering.** Standardized features → KMeans(k=2, n_init=10, `random_state=seed`). A diagonal GMM is fitted as an agreement check.

**Cluster naming.** No labels are used. The cluster whose mean signed drift is closest (cosine) to a **label-free HPC reference** is called *HPC-like*. The reference is the mean end-of-life drift of the single-fault sibling dataset, which has the same operating conditions: FD001 for FD003, FD002 for FD004. The other cluster is the *fan-mode candidate*.

| Features | Metric | FD003 | FD004 |
|---|---|---:|---:|
| drift (14 signed z) | silhouette | 0.51 | 0.49 |
| | cross-seed ARI (10 pairs) | 1.00 | 0.99 ± 0.01 |
| | KMeans vs GMM ARI | 0.93 | 0.93 |
| | fan-mode candidate fraction | 45.8 % | 39.0 % |
| | cosine to HPC ref (HPC-like / candidate) | 0.98 / 0.02 | 1.00 / 0.43 |
| attribution (14 sensor shares) | silhouette | 0.55 ± 0.08 | 0.39 ± 0.09 |
| | cross-seed ARI | 1.00 | 0.70 ± 0.31 |
| | KMeans vs GMM ARI | 1.00 | 0.91 |
| drift vs attribution clustering | cross-view ARI | 0.93 | 0.37 ± 0.11 |

The drift-profile clusters are seed-stable. The drift features do not depend on the model, so the cross-seed ARI only measures k-means initialisation. The two clusters have physically opposite signatures. Mean signed EOL z by cluster on FD003 (FD004 in parentheses):

| Sensor | HPC-like | Fan-mode candidate |
|---|---:|---:|
| P30 (s7) | −3.2 (−1.3) | +21.9 (+9.8) |
| phi (s12) | −3.5 (−1.4) | +26.9 (+11.9) |
| BPR (s15) | +3.2 (+3.3) | −5.4 (−6.5) |
| W31/W32 (s20/s21) | ≈ −3.1 (−1.4) | ≈ +4.2 (+2.1) |
| Nf / NRf (s8/s13) | +1.5 (+1.9) | +5.3 (+5.5) |
| Nc / NRc (s9/s14) | +4.7 (+4.0) | +12.1 (+11.6) |

In the HPC-like cluster, the coolant bleeds W31/W32 (s20/s21) are ≈ −3.0 / −3.1 on FD003 and ≈ −1.4 on FD004.

The HPC-like cluster has the same signature as FD001/FD002 (cosine 0.98–1.00). The candidate cluster shows these changes:

- P30, phi and the coolant bleeds reverse direction.
- Bypass ratio falls.
- Fan speed drifts 2.5–3.5× more than in the HPC-like cluster.
- Core speed rises strongly.

This is consistent with a second, fan-side fault mode, but no ground truth exists to confirm the cluster-to-mode mapping.

**Limitation.** In mean |z| terms, the fan-mode candidate has larger drift in *every* subsystem, including HPC (from Nc/NRc and P30). Its attribution subsystem shares differ only slightly from the HPC-like cluster (FD003 Fan 11.4 vs 10.2%, HPC 43.1 vs 38.2%). The subsystem *share* bar therefore does **not** by itself tell the two modes apart. The separation comes from the sensor-level drift *direction* signature. On FD004, attribution-only clustering is not reliably seed-stable (ARI 0.70 ± 0.31) and agrees only weakly with the drift clustering (0.37). Use drift profiles, or drift and attribution combined, to separate modes on multi-regime data.

### c) Edge overhead and data flow

Measurements were single-thread CPU (`torch.set_num_threads(1)`) on the development host: Apple Silicon arm64, Python 3.11.4, torch 2.15.0. Each figure is the median over 3 repetitions × 300 windows. The FD001 timing may overlap a concurrent dashboard-data generation run.

| Per-window latency | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| inference only | 0.051 ms | 0.066 ms | 0.060 ms | 0.057 ms |
| + attribution + summary | 0.219 ms | 0.328 ms | 0.274 ms | 0.245 ms |
| + drift recomputed from raw history + attribution + summary | 0.491 ms | 0.932 ms | 0.647 ms | 0.590 ms |

The relative overhead is about 10–14× pure inference, because attribution needs a backward pass and drift is recomputed in pandas. The absolute cost is under 1 ms per window, and a C-MAPSS window advances once per engine cycle (one flight). Autograd buffers add 117,368 bytes per window, against a 2,040-byte input window and 89,092 bytes of model weights. **No Raspberry Pi 4 measurement was made** (see `results/EXPERIMENTS.md` → Edge measurement). Saying the overhead is small enough for Pi-class hardware is an inference from these sub-millisecond host figures plus the per-cycle cadence. It is not a measured Pi result.

During baseline fitting, diagnosis of every engine and latency timing, `socket.socket.connect/connect_ex/send/sendall/sendto` and `socket.create_connection` were instrumented. The audit recorded **0 socket calls and 0 bytes sent**. The diagnostics code path does no network I/O. This audit covers only that code path. It does not change the repository's privacy scope (no TLS/DP/secure aggregation; see README).

## Dashboard

`dashboard/generate_demo_data.py` runs `EdgeDiagnostics` with the FedAvg (Asymmetric) global model on each of the 8 traced test engines. For each engine it writes into `demo_data.json`:

- per-cycle `subsystem_share` (Fan/LPC/HPC/Turbine, rounded to sum to 100.0)
- `top_contributors`, `top_drifting` and `drift_index`
- a `final_summary`

It also writes a top-level `diagnostics_meta` block with the sensor map, formulas, privacy and boundary.

The healthy baseline and regime statistics are fitted on the subset's training engines. This matches the repository's centralized-scaler caveat. In a deployment, each node fits them on its own history.

Tab 1 ("Live Asset RUL") shows the stacked subsystem bar, the top-3 contributing and drifting sensors, a stacked responsibility-over-time chart and the maintenance line. All of these follow `cycleSlider` and Play. IT/EN switching goes through the existing `I18N`/`applyLabels`/`renderAsset` path. The existing RUL predictions in `demo_data.json` regenerate bit-identically.

## Tests

```bash
python -m pytest tests/test_diagnostics.py
```

The tests use synthetic data and do not need the archive. They cover:

- deterministic output shapes
- the exact input×gradient formula
- shares summing to 100% (raw, degenerate and rounded)
- `setting1..3` getting zero attribution even when settings vary wildly
- single- and multi-regime drift
- short-engine padding
- the maintenance-summary schema
