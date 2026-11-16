# FedPdM-Edge Technical & Architectural Handbook — Part 2

## Local edge diagnostics, validation, presentation dashboard and test architecture

> **Scope of Part 2.** Part 1 of this handbook (written in parallel) covers the project motivation, the federated-learning architecture, `AsymmetricRULLoss`, the reproducible benchmarks and `paper/`. This part covers the layer built on top of the RUL model:
>
> - `diagnostics.py`: local degradation attribution and healthy-baseline drift.
> - `experiments/validate_diagnostics.py` and `results/diagnostics_validation.json`: 5-seed validation.
> - `dashboard/`: offline presentation dashboard.
> - `tests/`: the three test modules on `master`.
>
> **Provenance rule.** Every path, symbol, formula and number below comes from a file read directly in the repository at commit `9182089`, or from `readme.txt` / `Damage Propagation Modeling.pdf` inside `CMAPSSData.zip`. Numbers are copied from the committed `results/diagnostics_validation.json` and `dashboard/demo_data.json`. **No experiment was re-run to write this guide.** Analytical remarks that are not measurements are labelled *(analysis)*. Empirical and hardware limits are marked **⚠ Boundary**.

---

## Table of contents

1. [Why local edge degradation attribution exists, and its empirical boundary](#1-why-local-edge-degradation-attribution-exists-and-its-empirical-boundary)
2. [Physical channel mapping and design decisions (`diagnostics.py`)](#2-physical-channel-mapping-and-design-decisions-diagnosticspy)
3. [Model-free healthy-baseline drift and multi-regime normalization](#3-model-free-healthy-baseline-drift-and-multi-regime-normalization)
4. [5-seed quantitative validation, clustering and edge overhead](#4-5-seed-quantitative-validation-clustering-and-edge-overhead)
5. [Offline presentation dashboard architecture (`dashboard/`)](#5-offline-presentation-dashboard-architecture-dashboard)
6. [Testing and verification architecture (`tests/`)](#6-testing-and-verification-architecture-tests)
7. [Appendix A: symbol and file index](#appendix-a-symbol-and-file-index)
8. [Appendix B: consolidated list of boundaries and caveats](#appendix-b-consolidated-list-of-boundaries-and-caveats)

---

## 1. Why local edge degradation attribution exists, and its empirical boundary

### 1.1 The operational gap: *when* versus *which*

The model in `common.py` is `CmapssRegressor`. It is a 1D-CNN:

- `Conv1d(17→32, k=3)` → ReLU → `MaxPool1d(2)`
- `Conv1d(32→64, k=3)` → ReLU → `MaxPool1d(2)`
- `Linear(64·7→32)` → ReLU → `Linear(32→1)`

`dashboard/demo_data.json` → `model_info.param_count` gives 22,273 parameters. The model maps one window of `SEQ_LEN = 30` cycles × 17 channels (`USEFUL_SENSORS`) to **one scalar**: the Remaining Useful Life (RUL), capped at `MAX_RUL = 125`.

A scalar RUL tells a maintenance planner *when* an asset is expected to fail. It does not say *which physical module or sensor* is driving the estimate down. A planner who reads "RUL 12 cycles" still has to decide what to inspect.

The module docstring of `diagnostics.py` states the question it answers:

> "which physical module / sensor is driving the RUL drop?" for a single engine, computed entirely on the edge node that owns the raw telemetry.

### 1.2 Why the computation stays on the edge node

The privacy model of FedPdM-Edge is that raw telemetry never leaves the plant; only model weights travel (Part 1). Diagnostics follows the same rule:

- **Attribution needs raw inputs.** Attribution needs the exact scaled input window `x`, the per-engine healthy baseline `b`, and a backward pass through the model. The model can be the federated global model, which the node already holds. The raw sensor history is only on the node.
- **Drift needs each engine's own history.** Drift is computed against each engine's own first `healthy_cycles` cycles and the local fleet's regime statistics (§3). Both require that local history.
- **Outputs are aggregates.** `diagnostics.py` emits only aggregate shares (%), standardized drift scores (σ), the predicted RUL and text. This is declared in `PRIVACY_METADATA`:

```python
PRIVACY_METADATA = {
    "computed_on": "local_edge_node",
    "raw_telemetry_location": "local_edge_only",
    "raw_sensor_bytes_transmitted": 0,
    "network_calls": 0,
    "contains_raw_sensor_values": False,
    "shared_fields": "aggregate attribution shares (%), standardized drift scores (sigma), predicted RUL, text",
}
```

- **No network code.** The module imports only `dataclasses`, `typing`, `numpy`, `pandas`, `torch`, `common`, and lazily `sklearn.cluster.KMeans`. It opens no socket. §4.4 shows the socket audit that checks this empirically.

> **⚠ Boundary (privacy scope).** `docs/diagnostics.md` says the socket audit covers only the diagnostics code path. It "does not change the repository's privacy scope (no TLS/DP/secure aggregation; see README)".

### 1.3 The strict C-MAPSS empirical boundary

`readme.txt` in `CMAPSSData.zip` was read verbatim for this guide. It defines the four subsets:

| Subset | Conditions (readme) | Fault modes (readme) | Train / test trajectories (readme) | Unit IDs actually in files |
|---|---|---|---|---|
| FD001 | ONE (Sea Level) | ONE (HPC Degradation) | 100 / 100 | — |
| FD002 | SIX | ONE (HPC Degradation) | 260 / 259 | train 260, test 259 |
| FD003 | ONE (Sea Level) | TWO (HPC Degradation, Fan Degradation) | 100 / 100 | — |
| FD004 | SIX | TWO (HPC Degradation, Fan Degradation) | 248 / 249 | **train 249, test 248** |

The last column was counted from the distinct unit IDs in `train_FD002/FD004.txt` and `test_FD002/FD004.txt` inside the archive. For FD004, `readme.txt` swaps the train and test counts. The validation JSON agrees with the files: FD004 `n_units_clustered.train = 249`.

C-MAPSS publishes **only a unit-level RUL target** (`RUL_FD00x.txt` for test units; run-to-failure for training units). No file gives a per-unit fault-mode or per-component label. As a result:

- **FD001 / FD002.** Every unit has the same single fault mode (HPC degradation). Validation can only check *physical consistency*: does attribution concentrate on the engine core as RUL falls? There is no per-unit label to score.
- **FD003 / FD004.** Two fault modes exist, but no file says which unit has which. Validation can only check *unsupervised separability* (k = 2 clustering) and name clusters by physical signature.

`diagnostics.py` encodes this as a constant that travels in every summary and every validation report:

```python
EVALUATION_BOUNDARY = (
    "C-MAPSS provides unit-level RUL only; no per-unit component fault labels exist. "
    "Subsystem attribution is a model/physics-based explanation, not a supervised component diagnosis."
)
```

> **⚠ Boundary.** **No supervised component-classification accuracy is claimed anywhere in the repository.** `docs/diagnostics.md`: "The attribution explains the model's prediction. It is not a component diagnosis." `experiments/validate_diagnostics.py` docstring: "this script does NOT report a supervised component accuracy."

---

## 2. Physical channel mapping and design decisions (`diagnostics.py`)

### 2.1 Column layout: verification against `readme.txt`

`readme.txt` says each row has 26 space-separated columns:

1. unit number
2. time, in cycles
3. to 5. operational settings 1–3
6. onward: sensor measurements

The readme's enumeration ends with the typo "26) sensor measurement 26". Arithmetic gives 26 − 5 = 21 sensors. `common.load_cmapss_raw` names the columns `['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]` and reads `usecols = range(26)`.

### 2.2 Verification against Saxena et al. (PHM08), Table 2

The archive ships `Damage Propagation Modeling.pdf` (Saxena, Goebel, Simon, Eklund, PHM08). The text of Table 2 ("C-MAPSS outputs to measure system response") was taken from the PDF's compressed content streams. It lists the 21 participant-visible outputs in this order:

| s# | Symbol | Table 2 description | In `USEFUL_SENSORS`? | `diagnostics.SENSOR_INFO` subsystem |
|---|---|---|---|---|
| s1 | T2 | Total temperature at fan inlet | no | — |
| s2 | T24 | Total temperature at LPC outlet | yes | LPC |
| s3 | T30 | Total temperature at HPC outlet | yes | HPC |
| s4 | T50 | Total temperature at LPT outlet | yes | Turbine |
| s5 | P2 | Pressure at fan inlet | no | — |
| s6 | P15 | Total pressure in bypass-duct | no | — |
| s7 | P30 | Total pressure at HPC outlet | yes | HPC |
| s8 | Nf | Physical fan speed | yes | Fan |
| s9 | Nc | Physical core speed | yes | HPC |
| s10 | epr | Engine pressure ratio (P50/P2) | no | — |
| s11 | Ps30 | Static pressure at HPC outlet | yes | HPC |
| s12 | phi | Ratio of fuel flow to Ps30 | yes | Turbine |
| s13 | NRf | Corrected fan speed | yes | Fan |
| s14 | NRc | Corrected core speed | yes | HPC |
| s15 | BPR | Bypass Ratio | yes | Fan |
| s16 | farB | Burner fuel-air ratio | no | — |
| s17 | htBleed | Bleed Enthalpy | yes | Turbine |
| s18 | Nf_dmd | Demanded fan speed | no | — |
| s19 | PCNfR_dmd | Demanded corrected fan speed | no | — |
| s20 | W31 | HPT coolant bleed | yes | Turbine |
| s21 | W32 | LPT coolant bleed | yes | Turbine |

The symbol and description of every one of the 14 sensors in `SENSOR_INFO` match the table row at the same index. `tests/test_diagnostics.py::ChannelMapTest` spot-checks `s3→T30`, `s11→Ps30` and `s15→BPR`.

The rationale for dropping s1, s5, s6, s10, s16, s18 and s19 from `USEFUL_SENSORS` is not documented in the files covered by this part. Only the resulting list is reproduced here.

Table 1 of the same PDF says C-MAPSS can inject faults into "five rotating components (Fan, LPC, HPC, HPT, and LPT)". Its caption says HPC degradation is simulated with the HPC flow and efficiency modifiers.

### 2.3 The four-subsystem map

```python
SETTINGS      = ("setting1", "setting2", "setting3")
FAULT_SENSORS = [c for c in USEFUL_SENSORS if c not in SETTINGS]   # 14 sensors
SUBSYSTEMS    = ("Fan", "LPC", "HPC", "Turbine")
```

| Subsystem | Sensors (`SUBSYSTEM_SENSORS`) | Count | `SUBSYSTEM_LABELS` |
|---|---|---:|---|
| Fan | s8, s13, s15 | 3 | Fan (Nf, NRf, BPR) |
| LPC | s2 | 1 | Low-pressure compressor (T24) |
| HPC | s3, s7, s9, s11, s14 | 5 | High-pressure compressor / core spool (T30, P30, Ps30, Nc, NRc) |
| Turbine | s4, s12, s17, s20, s21 | 5 | Combustor + HPT/LPT hot section (T50, phi, htBleed, W31, W32) |

Design decisions recorded in `docs/diagnostics.md`:

- **"Turbine" is a merged group.** It merges the combustor with the HPT/LPT hot section. Table 1's five rotating components are reduced to four instrumented groups.
- **Sensors are grouped by where they are measured, not by which module is faulty.** "In the gas path, a fault in one module also shifts sensors located in other modules… HPC degradation moves fan speed and the coolant bleeds." A subsystem share therefore says which instrumented module dominates the model's evidence. It does not say which module is faulty.
- **phi (s12) goes to Turbine (combustor).** phi is the ratio of fuel flow to Ps30. Its placement drives the HPC-versus-Turbine split on FD001 (§4.2.3).
- **Group sizes differ (3/1/5/5).** Every validation rate is therefore reported next to a sensor-count chance rate: HPC 5/14 = 35.7 %, core (HPC ∪ Turbine) 10/14 = 71.4 %.

Index helpers, precomputed at import time:

- `FAULT_IDX`: positions of the 14 sensors inside the 17-channel input.
- `SETTING_IDX`: positions 0, 1, 2.
- `_SUB_MATRIX`: a 14 × 4 one-hot sensor-to-subsystem matrix.

### 2.4 Settings are model inputs but never fault carriers

**Why the settings are inputs.** `common.py` adds `setting1..3` to `USEFUL_SENSORS` with the comment "Added setting1, setting2, setting3 for heterogeneous operating conditions". `readme.txt` says FD002/FD004 run under six conditions and "three operational settings that have a substantial effect on engine performance". The RUL model needs these channels to interpret a sensor value in its operating context.

**Why they are excluded from attribution.** An operating setting is a flight condition chosen by the operator, not a degradation symptom. Two mechanisms keep it out:

1. **Hard zeroing.** `attribute_windows` sets `a[:, SETTING_IDX] = 0.0` after computing the attribution and before any normalization. This holds for all three methods. `sensor_shares` then uses only `FAULT_IDX`, so the settings cannot enter the denominator either.
2. **Zero baseline deviation.** `HealthyBaseline.healthy_reference` copies the *observed* settings into the reference (`ref = frame[USEFUL_SENSORS]…copy()`, then only `ref[:, FAULT_IDX]` is overwritten). So `x − b = 0` on settings channels under `baseline_x_gradient`, even before the hard zero.

Tests:

- `test_settings_excluded_even_when_wild` injects settings with σ = 1000 and checks exact zeros for every method.
- `test_healthy_reference_settings_untouched` checks that the reference equals the scaled observed settings.

### 2.5 Scaling and windowing helpers

- `scale_frame(values, scales, columns=USEFUL_SENSORS)`: min-max scaling `(x − min) / (max − min + 1e-7)`, the same transform as `common.apply_scale`. In the validation and the FD001 dashboard path, the scales come from `benchmark_asymmetric_loss.compute_scales_from_train`, whose docstring reads "healthy-regime min/max scales (first 50 cycles per unit)". The FD002 dashboard model uses `global_scale.json` (§5.1).
- `sliding_windows(arr, seq_len=SEQ_LEN)`: `(T, C) → (T − 30 + 1, 30, C)` via `np.lib.stride_tricks.sliding_window_view`. Series shorter than 30 are **left-padded by repeating the first row**, matching `common.extract_last_sequence`. A 12-cycle engine therefore yields exactly one window, labelled with cycle 12 (`test_short_engine_padded`).

### 2.6 Attribution: the three methods compared

`attribute_windows(model, windows, baseline=None, method="baseline_x_gradient", batch_size=512)` returns `(pred_rul (N,), channel_attr (N, 17))`.

Notation:

- `x ∈ ℝ^{W×17}`: the scaled window, W = 30.
- `ŷ = f(x)`: the predicted RUL.
- `g = ∂ŷ/∂x`: the input gradient, from `torch.autograd.grad(y.sum(), x)`.

| `method` | Per-channel attribution `A_c` | Meaning |
|---|---|---|
| `"baseline_x_gradient"` (default) | `A_c = (1/W) Σ_t |(x[t,c] − b[t,c]) · g[t,c]|` | first-order effect of the channel's **deviation from the engine's healthy expectation** |
| `"input_x_gradient"` | `A_c = (1/W) Σ_t |x[t,c] · g[t,c]|` (b = 0) | deviation from the scaled-space origin (the training min) |
| `"gradient"` | `A_c = (1/W) Σ_t |g[t,c]|` | pure saliency: local sensitivity of the model |

Then `A_c := 0` for `c ∈ SETTINGS`.

Implementation details from the code:

- An unknown `method` raises `ValueError`.
- **Silent fallback.** If `method == "baseline_x_gradient"` and `baseline is None`, the method falls back to `"input_x_gradient"`.
- An empty `windows` array returns float64 arrays of shape `(0,)` and `(0, 17)` (`test_empty_windows`).
- Inputs are cast to `float32` with `requires_grad=True`. Attributions are computed in `float64`.
- **Why one backward pass of `y.sum()` gives per-sample gradients** *(analysis)*. `CmapssRegressor` has no batch-norm or dropout (only Conv1d, ReLU, MaxPool1d, Linear), and the module is put in `eval()`. Each output therefore depends only on its own window, so the gradient of the sum equals the stack of per-sample gradients. `test_shapes_and_determinism` confirms that `batch_size=3` and `batch_size=512` give identical attributions.

**Why the baseline form is the default.** The module docstring says it is "a first-order estimate of how much channel c's deviation from healthy moves the RUL estimate". *(analysis)* This follows from the first-order Taylor expansion `f(x) − f(b) ≈ Σ_{t,c} (x[t,c] − b[t,c]) · g[t,c]`. Because the absolute value is taken per element before summing, `W · Σ_c A_c ≥ |first-order term|`. The shares therefore rank the magnitude of each channel's contribution. They are not a signed decomposition that sums to `f(x) − f(b)`. A channel sitting exactly at its healthy expectation gets zero attribution, however sensitive the model is to it. When the whole window equals the baseline, every attribution is 0 and the uniform fallback applies (`test_baseline_equal_input_gives_uniform`).

**Why plain gradient saliency fails to separate engine state from model sensitivity.** `|∂ŷ/∂x|` describes the model's local Jacobian. It does not include how far the engine has moved from healthy. The measured result on FD001/FD002 (`results/diagnostics_validation.json`, low-RUL windows, mean ± sd over 5 seeds):

| Rate | `gradient` FD001 | `gradient` FD002 | `baseline_x_gradient` FD001 | `baseline_x_gradient` FD002 | Chance |
|---|---:|---:|---:|---:|---:|
| Primary subsystem = HPC | 0.8 ± 1.5 % | 1.4 ± 3.0 % | 26.6 ± 5.5 % | 59.6 ± 5.0 % | 35.7 % |
| Primary subsystem ∈ core | 98.7 ± 2.1 % | 100.0 ± 0.0 % | 99.4 ± 0.8 % | 100.0 ± 0.0 % | 71.4 % |
| Top-3 attributed sensors in core | 83.6 ± 12.3 % | 66.7 ± 0.0 % | 94.9 ± 5.1 % | 73.3 ± 1.7 % | 71.4 % |

Plain saliency still points at the core, but it almost never ranks HPC first. `docs/diagnostics.md` concludes that it "describes model sensitivity rather than the engine's state." On FD002 its top-3 core fraction (66.7 %) is below chance.

> **⚠ Boundary.** `input_x_gradient` is implemented and unit-tested (`test_input_x_gradient_formula`) but **not** evaluated in the validation JSON. Only `baseline_x_gradient` and `gradient` are compared there.

### 2.7 From channel attribution to shares

```python
def sensor_shares(channel_attr):           # (N,17) -> (N,14), rows sum to 100
    a = clip(channel_attr[:, FAULT_IDX], 0, None)
    tot = a.sum(1)
    return 100 * (a / tot  if tot > 0  else  1/14)      # uniform fallback

def subsystem_shares(sensor_share):        # (N,14) -> (N,4)
    return sensor_share @ _SUB_MATRIX
```

This is a simplified rendering of `diagnostics.py:260–271`.

- **Conservation.** Sensor shares sum to 100, and so do subsystem shares, because each sensor belongs to exactly one subsystem. Tests: `test_conservation_sum_100` and `test_engine_outputs_aligned`, tolerance 1e-9.
- **Uniform zero-attribution fallback.** When all 14 attributions are 0 (for example `x == b` everywhere), each sensor gets 100/14 ≈ 7.14 %. This avoids a 0/0 division. Test: `test_baseline_equal_input_gives_uniform`.
- **Drift aggregation by subsystem.** `subsystem_drift(window_drift)` is `(|z| @ _SUB_MATRIX) / _SUB_MATRIX.sum(0)`: the *mean* |z| per subsystem. Its docstring calls this "sensor-count invariant", so a 5-sensor group is not favoured over the 1-sensor LPC. Attribution shares, by contrast, are *sums*, which is why the chance rates in §2.3 are needed.

### 2.8 `round_to_total`: largest-remainder rounding

The dashboard shows subsystem shares to one decimal. Naive rounding can produce totals of 99.9 or 100.1. For example, `[100/3]*3` rounds to 33.3 × 3 = 99.9. `round_to_total(values, decimals=1, total=100.0)` avoids this:

1. Scale: `v = values · 10^decimals`.
2. Validate: every value and `total` must be finite and non-negative, otherwise `ValueError`.
3. Floor: `fl = floor(v)`. Compute `rest = round(total · 10^decimals − Σ fl)`.
4. Require `0 ≤ rest ≤ len(v)`, otherwise `ValueError`. This rejects inputs that cannot reach the total by rounding, such as `[10, 10] → 100` or `[90, 90] → 100`. It also rejects the empty list, where `rest = 1000 > 0`.
5. Add 1 unit to the `rest` entries with the largest fractional remainders `v − fl`.
6. Return `[round(x / 10^decimals, decimals) …]`.

Worked example from `test_largest_remainder_choice`: `[10.9, 20.5, 68.6]` at 0 decimals has floors 10/20/68 and `rest = 2`. The two largest remainders are 0.9 and 0.6, so the result is `[11, 20, 69]`.

Users:

- `maintenance_summary` (`subsystem_shares_pct`).
- `dashboard/generate_demo_data.py` (every per-cycle `subsystem_share` row).
- `test_round_to_total_conservation` checks every rounded value is within 0.1 of the exact value and is a one-decimal number.

### 2.9 End-to-end driver: `EdgeDiagnostics`

```python
edge = D.EdgeDiagnostics(model, scales, baseline, method="baseline_x_gradient")
diag = edge.diagnose_engine(df_engine_raw, "FD002-U65")   # raw, unscaled frame for one engine
summary = edge.summarize(diag)                             # latest window (index=-1)
```

`diagnose_engine` performs these steps:

1. Sort by `cycle`.
2. Compute `x = scale_frame(...)` (float32), `ref = baseline.healthy_reference(...)` and `drift = baseline.drift(...)`.
3. Window all three.
4. Call `attribute_windows`.
5. Return arrays aligned on window end cycles:

| Key | Shape | Content |
|---|---|---|
| `cycles` | (N,) | window end cycles (`cycles[29:]`, or the last cycle if T < 30) |
| `pred_rul` | (N,) | raw model output (not clipped to [0, 125]) |
| `channel_attr` | (N, 17) | `A_c`, settings = 0 |
| `sensor_share` | (N, 14) | % |
| `subsystem_share` | (N, 4) | % in `SUBSYSTEMS` order |
| `window_drift` | (N, 14) | mean signed z over each window |
| `cycle_drift` | (T, 14) | per-cycle signed z |

`summarize(diag, index=-1)` calls `maintenance_summary` for any window index. `test_summarize_non_final_index` checks index 0, −1 and N − 1.

---

## 3. Model-free healthy-baseline drift and multi-regime normalization

Attribution explains the *model*. Drift is a second, model-independent view of the *engine*. It is also what `baseline_x_gradient` uses as its baseline `b`.

### 3.1 Why naive single-regime drift fails on FD002/FD004

FD002 and FD004 cycle through six operating conditions (`readme.txt`), and `N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}`. The raw value of a sensor jumps whenever the engine switches regime.

If drift were measured against one fleet-wide mean and standard deviation (`k = 1`), these jumps would:

- inflate the per-sensor σ used for standardization, and
- inflate the early-life noise estimate `σ_healthy`.

The slow fault signal would then be hidden under operating-condition variance.

`tests/test_diagnostics.py::test_multi_regime_drift_removes_operating_condition` reproduces this on a synthetic 6-regime fleet where every sensor shifts by +5 per regime and an HPC ramp is injected on s3/s11. It checks two things:

- With `n_regimes=6`, the unaffected sensor s8 stays within |z| < 5 and the s3 end-of-life drift exceeds 5σ.
- With `n_regimes=1`, the s3 end-of-life drift is **less than one third** of the regime-aware value.

> **⚠ Boundary.** The naive-versus-regime comparison is demonstrated on **synthetic data only** (the unit test). The validation JSON reports only regime-aware drift on real FD002/FD004. It has no naive-baseline ablation.

### 3.2 `fit_healthy_baseline`: the fitting procedure

`fit_healthy_baseline(df_raw, n_regimes=1, healthy_cycles=30, seed=0)` is fitted on the node's own raw run-to-failure history:

1. **Standardize settings.** Compute `set_mean` and `set_std` (+1e-9) over all rows, then `zs = (s − set_mean) / set_std`.
2. **Find regime centroids.**
   - `k ≤ 1`: one centroid at the origin.
   - Otherwise: `KMeans(n_clusters=k, n_init=10, random_state=seed)` on `zs`.
   - Centroids are re-ordered with `np.lexsort(km.cluster_centers_.T[::-1])`, which sorts by setting1, then setting2, then setting3. Regime IDs are therefore deterministic.
3. **Assign regimes.** Each row goes to its nearest centroid (squared Euclidean distance in standardized setting space).
4. **Per-regime statistics.** For each regime, compute `regime_mean` and `regime_std` of the 14 fault sensors over **all** rows of that regime (healthy and degraded cycles alike). A std below 1e-6 is replaced by 1.0, which handles constant sensors.
5. **Pooled healthy noise.** For each unit:
   - standardize its rows: `u = (x − μ_regime) / σ_regime`;
   - take the first `healthy_cycles` rows;
   - subtract their mean.

   Concatenate these residuals over all units. Their std is `healthy_sigma` (14,), the pooled within-engine early-life noise. A value below 1e-6 becomes 1.0.

The fitted `HealthyBaseline` dataclass holds:

- `centroids` (k, 3)
- `set_mean`, `set_std` (3,)
- `regime_mean`, `regime_std` (k, 14)
- `healthy_sigma` (14,)
- `healthy_cycles` (default 30)

### 3.3 Applying the baseline to one engine

| Method | Formula |
|---|---|
| `regimes(settings)` | nearest-centroid ID. Accepts shape `(3,)` or `(T, 3)`. Returns an empty int array for empty input. Raises `ValueError` for other shapes (`test_regimes_1d_and_empty_inputs`). |
| `standardize(frame)` | `u[t,c] = (x[t,c] − μ_{r(t),c}) / σ_{r(t),c}`, also returns `r(t)` |
| `engine_offset(u)` | `mean_{t < H} u[t,:]` with `H = max(1, min(healthy_cycles, T))` |
| `drift(frame)` | `z[t,c] = (u[t,c] − offset_c) / healthy_sigma_c` |
| `healthy_reference(frame, scales)` | fault sensors: `μ_{r(t)} + σ_{r(t)} · offset`, mapped back to raw units. Settings: observed values. Result passed through `scale_frame` → (T, 17). |

Key properties:

- **Each engine is its own reference.** Subtracting the engine's early-life offset removes unit-to-unit manufacturing and initial-wear variation. `readme.txt` says this variation "is considered normal, i.e., it is not considered a fault condition". Test `test_single_regime_drift` checks that `z[:30]` has mean ≈ 0 (atol 1e-9).
- **Drift is in units of healthy noise.** z = 3 means the window has moved three times the typical early-life cycle-to-cycle noise of the local fleet.
- **A regime switch is not drift.** Standardization is per regime, so a switch between conditions does not show up as drift. `docs/diagnostics.md` refers to `tests/test_diagnostics.py` for this.
- **The attribution baseline is time-varying and regime-aware.** `b[t]` follows the engine's regime at cycle `t`. A regime change in the input is matched by the same change in `b`, so it does not create attribution mass.

> **⚠ Boundary (fit population).** In `experiments/validate_diagnostics.py::run_subset`, `fit_healthy_baseline(df_tr, …)` is fitted **once per subset on all training units**, including the units held out from model training in each seed. The fit uses no labels, but held-out units do contribute to the regime and noise statistics. In `dashboard/generate_demo_data.py` the baseline is fitted on the subset's training engines and applied to test engines. `docs/diagnostics.md` notes that a deployment would fit it "on its own history" per node.

### 3.4 Structured bilingual `maintenance_summary`

`maintenance_summary(pred_rul, sensor_share, window_drift, engine_id=None, cycle=None, top_k=3)` returns a JSON-serializable dict. It contains no raw sensor values:

| Field | Source |
|---|---|
| `engine_id`, `cycle` | arguments |
| `predicted_rul` | `round(pred_rul, 2)` |
| `severity` | `severity_for(pred_rul)` |
| `primary_subsystem`, `primary_subsystem_label` | `argmax` of the unrounded subsystem shares, plus `SUBSYSTEM_LABELS` |
| `subsystem_shares_pct` | `round_to_total(subsystem_shares, 1)` as `{Fan, LPC, HPC, Turbine}` |
| `primary_drift_subsystem` | `argmax(subsystem_drift)` |
| `subsystem_drift_mean_abs_z` | mean |z| per subsystem, 2 dp |
| `drift_index` | mean |z| over the 14 sensors, 2 dp |
| `top_contributors` | top-3 by share (stable sort): `sensor`, `symbol`, `name`, `subsystem`, `share_pct` |
| `top_drifting` | top-3 by |z| (stable sort): the same fields plus `drift_z` and `direction` (`up`/`down`) |
| `text.en` / `text.it` | rendered sentences (below) |
| `privacy` | a copy of `PRIVACY_METADATA` |
| `evaluation_boundary` | `EVALUATION_BOUNDARY` |

**Severity cut-offs.** `SEVERITY_THRESHOLDS = {"critical": 20.0, "warning": 45.0}` (cycles of predicted RUL):

```
pred_rul <= 20  -> "critical"  ("Schedule maintenance now"    / "Pianificare subito la manutenzione")
pred_rul <= 45  -> "warning"   ("Plan inspection"             / "Pianificare un'ispezione")
otherwise       -> "healthy"   ("No action"                   / "Nessuna azione")
```

The source comment says these thresholds match the dashboard. `index.html::renderAsset` uses `pSafe <= 20` / `pSafe <= 45` for the status KPI.

**Text template (EN).** `"Predicted RUL {r:.1f} cycles ({SEV}). {lead}. Primary contributor: {primary} ({pct:.1f}% of attribution). Top drivers: {s# SYM (x.x%), …}. Largest drift vs healthy baseline: {s# SYM (+z.z sigma), …}."`

The IT template has the same structure: "RUL stimata … cicli", "Sottosistema principale", "Sensori determinanti", "Deriva maggiore vs baseline sana".

---

## 4. 5-seed quantitative validation, clustering and edge overhead

Source: `experiments/validate_diagnostics.py` → `results/diagnostics_validation.json`.

Command, from the script docstring and `docs/diagnostics.md`:

```bash
CMAPSS_ZIP=/path/CMAPSSData.zip python experiments/validate_diagnostics.py --out results/diagnostics_validation.json
```

### 4.1 Protocol (from `config` and the code)

| Item | Value |
|---|---|
| Seeds | `[7, 42, 100, 777, 2024]` |
| Subsets | FD001, FD002, FD003, FD004 |
| Model | `CmapssRegressor` (common.py), one per seed and subset; an isolated edge-node model, **not** the paper's federated runs |
| Training | `AsymmetricRULLoss` (`--loss asymmetric`), 8 epochs, batch 64, `torch.set_num_threads(4)` for training |
| Train units | 80 % of training units via `split_units(units, 0.2, seed)`. FD001: 80 train / 20 held out. FD002: 208 / 52. |
| Scaler | `compute_scales_from_train(df_tr)` |
| Attribution | `baseline_x_gradient` (plus `gradient` as a comparison on FD001/FD002) |
| Environment | Python 3.11.4, torch 2.15.0, `machine: arm64`, `system: Darwin`, `processor: arm` |
| Whole-run figures | `elapsed_s: 675.2`, `process_peak_rss_bytes: 703,873,024` (whole validation process, training included) |

**Official test RMSE** (last window of every official test engine, predictions clipped to [0, 125]), mean ± sd over 5 seeds:

| FD001 | FD002 | FD003 | FD004 |
|---:|---:|---:|---:|
| 16.80 ± 1.42 | 23.74 ± 3.08 | 17.46 ± 2.68 | 26.44 ± 4.79 |

Per seed: FD001 = [17.18, 16.21, 15.49, 16.02, 19.09]; FD002 = [22.41, 23.19, 29.05, 23.03, 21.03].

### 4.2 Single-fault consistency on FD001 / FD002

**Population.** Held-out training units, which run to failure (`final_rul = 0`), plus all official test units. True RUL per window is `engine_true_rul = min(125, final_rul + last_cycle − cycle)`. Only windows with **true RUL < 30** (`LOW_RUL`) are scored. With the seed-7 split this gives 907 windows on FD001 and 2,586 on FD002.

#### 4.2.1 Headline rates (`baseline_x_gradient`, mean ± sd over 5 seeds)

| Metric | FD001 | FD002 | Chance (sensor count) |
|---|---:|---:|---:|
| Primary attribution subsystem = HPC | 26.6 ± 5.5 % | 59.6 ± 5.0 % | 35.7 % |
| Primary attribution subsystem ∈ core (HPC ∪ Turbine) | 99.4 ± 0.8 % | 100.0 ± 0.0 % | 71.4 % |
| Top-3 attributed sensors in core | 94.9 ± 5.1 % | 73.3 ± 1.7 % | 71.4 % |
| Top-3 drifting sensors in core (model-free) | 82.0 ± 1.7 % | 76.7 ± 0.6 % | 71.4 % |
| Primary *drift* subsystem = HPC (model-free, seed 7) | 51.4 % | 76.2 % | 35.7 % |

The JSON `note` reads: "drift metrics are model-independent (identical across seeds); chance rates assume attribution mass proportional to sensor count per subsystem." *(analysis)* The drift top-3 rate still has a small sd across seeds. That sd cannot come from the model. The likely source is the population: it includes each seed's held-out training units, and those differ from seed to seed.

Distribution of the primary attribution subsystem (mean over seeds):

- FD001: Turbine 72.8 %, HPC 26.6 %, Fan 0.6 %, LPC 0 %.
- FD002: HPC 59.6 %, Turbine 40.4 %, Fan 0 %, LPC 0 %.

Distribution of the primary *drift* subsystem (seed 7):

- FD001: HPC 51.4 %, Fan 42.0 %, Turbine 6.6 %, LPC 0 %.
- FD002: HPC 76.2 %, LPC 16.8 %, Turbine 4.0 %, Fan 3.0 %.

#### 4.2.2 Core dominance

Near end of life, almost all of the attribution mass sits in the core (HPC + combustor/turbine), against a 71.4 % chance rate. Fan and LPC are almost never primary.

Mean |z| per subsystem at RUL < 30 (seed 7) agrees with the physics:

| | Fan | LPC | HPC | Turbine |
|---|---:|---:|---:|---:|
| FD001 | 2.92 | 2.31 | **4.41** | 3.10 |
| FD002 | 2.54 | 2.06 | **3.70** | 1.88 |

#### 4.2.3 Why phi/s12 and P30/Ps30 decide the HPC-versus-Turbine split

Top-3 attribution frequency per sensor (seed 7, RUL < 30):

| Sensor | FD001 | FD002 |
|---|---:|---:|
| s12 phi (→ Turbine) | **81.8 %** | 0.0 % |
| s11 Ps30 (→ HPC) | **79.2 %** | **96.1 %** |
| s7 P30 (→ HPC) | 60.4 % | 0.0 % |
| s4 T50 (→ Turbine) | 9.8 % | **100.0 %** |
| s15 BPR (→ Fan) | 1.2 % | **78.8 %** |
| s14 NRc (→ HPC) | 7.5 % | 25.0 % |

**FD001.** The attribution mass is shared mainly between phi (s12), which is mapped to Turbine, and P30/Ps30 (s7/s11), which are mapped to HPC. The label-free FD001 end-of-life reference drift shows that **P30 and phi move together** in the HPC fault signature: s7 −4.37σ and s12 −4.98σ, both negative. Because phi is counted on the Turbine side, Turbine is primary in 72.8 % of windows and HPC alone in only 26.6 %, which is below chance. `docs/diagnostics.md`: "This depends on assigning phi to the combustor. Read the HPC-only rate with the mapping caveat above in mind."

**FD002.** The Turbine share comes from T50 (s4), not from phi. HPC wins 59.6 % of windows through Ps30 (and NRc).

**Drift agrees.** Ps30 (s11) is the most frequent top-3 *drifting* sensor on both datasets: 84.9 % on FD001 and 77.5 % on FD002 (seed 7).

#### 4.2.4 Trend across RUL buckets (seed 7, mean subsystem share %)

| | Bucket | Fan | LPC | HPC | Turbine |
|---|---|---:|---:|---:|---:|
| FD001 | RUL ≥ 100 | 16.9 | 5.3 | 30.3 | 47.6 |
| | 30 ≤ RUL < 100 | 17.9 | 5.4 | 32.2 | 44.5 |
| | RUL < 30 | 21.7 | 7.1 | 33.4 | 37.7 |
| FD002 | RUL ≥ 100 | 18.4 | 1.4 | 37.2 | 43.0 |
| | 30 ≤ RUL < 100 | 18.7 | 1.2 | 40.2 | 40.0 |
| | RUL < 30 | 18.8 | 1.0 | 41.8 | 38.4 |

The HPC share rises as RUL falls, but `docs/diagnostics.md` calls the trend small. The large signal is the core-versus-Fan/LPC split.

### 4.3 Unsupervised k = 2 fault-mode separation on FD003 / FD004

**Population.**

- FD003: all 100 training units plus the 20 test units with official RUL ≤ 30, 120 profiles in total.
- FD004: all 249 training units plus 53 such test units, 302 profiles in total.

Each unit's **profile** is the mean of its last `EOL_WINDOWS = 10` windows (`eol_profiles`). This gives three feature views:

- `drift`: 14 signed z values.
- `attribution`: 14 sensor shares.
- `combined`: the concatenation of both, 28 features.

**Clustering** (`cluster`). `StandardScaler` → `KMeans(n_clusters=2, n_init=10, random_state=seed)`. As an agreement check, `GaussianMixture(n_components=2, covariance_type="diag", n_init=3, random_state=seed)` is also fitted, and its agreement with KMeans is reported as ARI. The script also reports:

- silhouette;
- cross-seed ARI over the 10 seed pairs;
- cross-view ARI between drift and attribution clusters.

**Naming the clusters without labels** (`reference_signature` and `orient`).

1. Build a **label-free HPC reference**: the mean end-of-life signed drift of the *single-fault sibling* subset under the same operating conditions. `REFERENCE = {"FD003": "FD001", "FD004": "FD002"}`.
2. The cluster whose mean signed drift has the highest cosine similarity to that reference becomes `hpc_like` (label 0).
3. The other cluster becomes the `fan_candidate` (label 1).

FD003/FD004 labels are never used. The reference uses only the readme fact that FD001/FD002 contain HPC degradation alone.

#### 4.3.1 Separation metrics (mean ± sd over 5 seeds)

| View | Metric | FD003 | FD004 |
|---|---|---:|---:|
| drift | silhouette | 0.51 | 0.49 |
| | cross-seed ARI (10 pairs) | 1.00 | 0.99 ± 0.01 |
| | KMeans vs GMM ARI | 0.93 | 0.93 |
| | fan-candidate fraction | 45.8 % | 39.0 % |
| | cosine to HPC ref (hpc_like / candidate) | 0.98 / 0.02 | 1.00 / 0.43 |
| attribution | silhouette | 0.55 ± 0.08 | 0.39 ± 0.09 |
| | cross-seed ARI | 1.00 | 0.70 ± 0.31 |
| | KMeans vs GMM ARI | 1.00 | 0.91 ± 0.12 |
| combined | cross-seed ARI | 1.00 | 0.79 ± 0.14 |
| | KMeans vs GMM ARI | 1.00 | 0.71 ± 0.15 |
| drift vs attribution | cross-view ARI | 0.93 | 0.37 ± 0.11 |

Drift features do not depend on the model. On drift, cross-seed ARI therefore measures only KMeans initialisation stability (`docs/diagnostics.md`).

#### 4.3.2 Physical signature reversal (drift view, mean signed end-of-life z)

| Sensor | FD003 hpc_like | FD003 fan_candidate | FD004 hpc_like | FD004 fan_candidate |
|---|---:|---:|---:|---:|
| s7 P30 | −3.21 | **+21.89** | −1.30 | **+9.77** |
| s12 phi | −3.45 | **+26.86** | −1.41 | **+11.93** |
| s15 BPR | +3.15 | **−5.39** | +3.31 | **−6.47** |
| s20 W31 | −3.02 | **+4.10** | −1.38 | **+2.06** |
| s21 W32 | −3.12 | **+4.29** | −1.40 | **+2.06** |
| s8 Nf | +1.50 | +5.26 | +1.90 | +5.50 |
| s13 NRf | +1.50 | +5.27 | +1.90 | +5.52 |
| s9 Nc | +5.19 | +12.12 | +4.25 | +10.94 |
| s14 NRc | +4.30 | +12.13 | +3.83 | +12.22 |
| s11 Ps30 | +5.05 | +6.53 | +4.64 | +6.17 |

For comparison, the FD001 HPC reference has s7 −4.37, s12 −4.98, s15 +3.68, s20 −3.50, s21 −3.46. The FD002 reference has s7 −1.78, s12 −2.00, s15 +3.86, s20 −1.63, s21 −1.59.

Reading the table:

- **The hpc_like cluster matches the single-fault signature**, with cosine 0.98 on FD003 and 1.00 on FD004.
- **In the fan candidate, five sensors reverse direction.** P30, phi and the coolant bleeds W31/W32 flip sign, and the bypass ratio BPR falls.
- **Fan speed drifts about 2.5–3.5× more** in the fan candidate than in the hpc_like cluster, and core speed rises strongly.

`docs/diagnostics.md` calls this "consistent with a second, fan-side fault mode, but no ground truth exists to confirm the cluster-to-mode mapping."

#### 4.3.3 Why sensor-level drift direction separates modes better than subsystem share bars

Mean |z| per subsystem and mean attribution share (drift-view clusters):

| | Fan |z| | LPC |z| | HPC |z| | Turbine |z| | Fan share | LPC share | HPC share | Turbine share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FD003 hpc_like | 2.10 | 2.89 | 4.92 | 3.62 | 10.16 % | 7.65 % | 38.24 % | 43.95 % |
| FD003 fan_candidate | 5.31 | 3.00 | 11.27 | 8.87 | 11.36 % | 3.65 % | 43.15 % | 41.85 % |
| FD004 hpc_like | 2.60 | 2.34 | 3.86 | 2.13 | 6.38 % | 1.55 % | 49.97 % | 42.10 % |
| FD004 fan_candidate | 5.87 | 2.14 | 8.47 | 4.83 | 8.66 % | 1.52 % | 52.65 % | 37.17 % |

1. **The fan candidate drifts more everywhere, HPC included.** Its HPC |z| is driven by Nc/NRc and P30. Larger |z| in Fan alone therefore does not identify the mode.
2. **Share bars lose both sign and scale** *(analysis)*. `subsystem_shares` sums *absolute* attributions and renormalizes to 100 %. Two signatures with opposite drift directions and different overall magnitudes can produce nearly the same bar. Measured: FD003 Fan 11.4 % vs 10.2 %, HPC 43.1 % vs 38.2 %.
3. **The signed 14-dimensional drift vector keeps the reversal.** It retains the P30/phi/BPR/W31/W32 direction change, which is what drives the high silhouette and the near-perfect cross-seed stability.
4. **Practical guidance** (`docs/diagnostics.md`). On FD004, attribution-only clustering is not reliably seed-stable (ARI 0.70 ± 0.31) and agrees only weakly with drift clustering (0.37). Use drift profiles, or drift and attribution combined, to separate modes on multi-regime data.

> **⚠ Boundary.** The subsystem share bar in the dashboard **does not by itself distinguish the two fault modes**. Mode separation is an offline, fleet-level clustering result on end-of-life profiles. It is not a per-window label emitted by `maintenance_summary`.

### 4.4 Edge overhead: latency, memory and the socket audit

#### 4.4.1 Method (`latency`)

- `torch.set_num_threads(1)`. The JSON records `threads: 1` for every subset.
- The engine is the longest held-out training unit of the first seed (`lat_seed = seeds[0] = 7`).
- Warm-up: one full `diagnose_engine`, then 20 calls per variant.
- Timing: 3 repetitions over up to 300 windows. The reported figure is the median per-window time (`windows_timed: 300`).

The three timed variants are:

| Variant | What runs per window |
|---|---|
| `inference_only` | `model(x[i:i+1])` under `no_grad` |
| `inference_attribution_summary` | `attribute_windows` (forward and backward) + `sensor_shares` + `maintenance_summary`, with precomputed drift |
| `inference_attribution_drift_summary` | the previous row, plus drift **recomputed in pandas from the raw history** `frame.iloc[: i + 30]` |

#### 4.4.2 Results (Darwin arm64 development host, single thread)

| ms per window | FD001 | FD002 | FD003 | FD004 |
|---|---:|---:|---:|---:|
| inference only | 0.0513 | 0.0664 | 0.0602 | 0.0565 |
| + attribution + summary | 0.2193 | 0.3284 | 0.2743 | 0.2446 |
| + drift recomputed + attribution + summary | **0.4911** | **0.9319** | **0.6473** | **0.5895** |
| overhead ratio (full / inference) | 9.57× | 14.03× | 10.75× | 10.43× |

**Memory.** `autograd_saved_tensor_bytes_per_window = 117,368` bytes for all four subsets. This is measured with a `torch.autograd.graph.saved_tensors_hooks` pack hook that sums `numel × element_size` over one forward and backward pass. Comparison points:

- `input_window_bytes = 2,040` (30 × 17 × float32);
- model weights of 89,092 bytes (`model_info.tensor_payload_bytes`, from 22,273 float32 parameters).

**Interpretation** (`docs/diagnostics.md`). The relative overhead of about 10–14× comes from the backward pass and the pandas drift recomputation. The absolute cost is under 1 ms per window, and a C-MAPSS window advances once per engine cycle (one flight).

> **⚠ Boundary (hardware).**
> - **No Raspberry Pi 4 measurement was made.** `results/EXPERIMENTS.md` → *Edge measurement*: the Pi helpers "remained unreachable and exited with SSH status 255. No Raspberry Pi timing or memory measurement is claimed." Saying the overhead is acceptable on Pi-class hardware is an *inference* from sub-millisecond host figures and the per-cycle cadence. It is not a Pi result.
> - `docs/diagnostics.md` warns that the FD001 timing may overlap a concurrent dashboard-data generation run.
> - `process_peak_rss_bytes` (≈ 704 MB) is the peak of the whole validation process, including training, pandas and sklearn. It is not the memory footprint of the diagnostics.

#### 4.4.3 `SocketAudit`

`SocketAudit` is a context manager in `experiments/validate_diagnostics.py`:

- It monkey-patches `socket.socket.connect`, `connect_ex`, `send`, `sendall` and `sendto`, plus `socket.create_connection`.
- Each call increments `calls`.
- When the first positional argument is bytes-like, it adds `len()` to `bytes`.
- The original functions are restored on exit.

It wraps three phases:

1. the baseline fit (`baseline_fit_network_socket_calls`);
2. the diagnosis of every engine for every seed (`network_socket_calls` and `network_bytes_sent` per seed);
3. the latency timing (`latency.network_socket_calls`).

The result is `network_audit = {total_socket_calls_during_diagnostics: 0, raw_sensor_bytes_sent: 0}`. Every per-seed and per-subset counter is also 0.

> **⚠ Boundary.** The audit covers the five instrumented `socket.socket` methods and `socket.create_connection` in this process and code path only. Other I/O paths, such as `sendmsg` or subprocesses, are not instrumented. `raw_sensor_bytes_sent` sums the per-seed diagnosis counters.

---

## 5. Offline presentation dashboard architecture (`dashboard/`)

```
dashboard/
├── generate_demo_data.py   # offline: trains reference models, writes demo_data.json
├── demo_data.json          # committed data snapshot (674,295 bytes at 9182089)
├── server.py               # zero-dependency stdlib HTTP server
└── index.html              # self-contained vanilla HTML/CSS/SVG/JS page
```

### 5.1 `generate_demo_data.py`: training the reference models

`train_subset_models(subset, seed=42, rounds=5, local_epochs=2)`:

| Step | FD001 | FD002 |
|---|---|---|
| Scales | `compute_scales_from_train(train_FD001)` | `global_scale.json` (`load_scales`) |
| Clients | 2 clients: odd unit IDs → `"1"`, even → `"2"` | `partition.json` (`load_partition`) |
| Client train split | `split_units(units, 0.1, seed)`, train part only | same |

From the clients, three models are trained, matching the dashboard legends:

- **`Local MSE`**: an isolated model trained only on the first client (`"1"`, "Plant 1"), with MSE for `rounds × local_epochs = 10` epochs.
- **`FedAvg MSE`** and **`FedAvg Asym`**: an in-process FedAvg loop.
  - The global model and the per-client models are seeded with `set_seed(seed)` and `set_seed(seed + cid)`.
  - Each client keeps a persistent Adam optimizer (lr 1e-3).
  - In each of 5 rounds, every client loads the global state, trains for 2 local epochs, and the global model becomes the window-count-weighted mean of the client states.
  - This runs once with `get_loss_fn("mse")` and once with `get_loss_fn("asymmetric")`.

It also fits a healthy baseline: `fit_healthy_baseline(df_raw, N_REGIMES[subset])` on the subset's raw training frame.

### 5.2 Selecting 8 life-stage engines

`pick_representative_engines` considers test engines with at least 85 observed cycles. With `final_rul = min(125, RUL_FD00x[u])`, it fills four buckets in order. In each bucket it picks the **longest** unused engine:

| Stage | Rule on true final RUL |
|---|---|
| `critical` | ≤ 20 |
| `warning` | 22–48 |
| `midlife` | 52–85 |
| `healthy` | ≥ 95 |

The committed `demo_data.json` contains these 8 engines. The final-summary columns come from `diagnostics.final_summary`:

| id | stage | final true RUL | observed cycles | windows | final `primary_subsystem` | final `severity` | final `predicted_rul` |
|---|---|---:|---:|---:|---|---|---:|
| FD001-U81 | critical | 8 | 213 | 184 | Turbine | critical | 2.93 |
| FD001-U91 | warning | 38 | 234 | 205 | Turbine | warning | 26.67 |
| FD001-U93 | midlife | 85 | 244 | 215 | Turbine | warning | 40.24 |
| FD001-U12 | healthy | 124 | 217 | 188 | Turbine | healthy | 89.50 |
| FD002-U65 | critical | 11 | 367 | 338 | HPC | critical | −1.04 |
| FD002-U58 | warning | 26 | 247 | 218 | HPC | critical | 10.09 |
| FD002-U140 | midlife | 55 | 306 | 277 | Turbine | warning | 20.21 |
| FD002-U119 | healthy | 106 | 202 | 173 | Turbine | healthy | 87.04 |

Two schema subtleties are visible in this table:

- **`stage` and `severity` measure different things.** `stage` is the selection bucket, based on the *true* final RUL. `severity` is computed from the *predicted* RUL. They can differ, as for FD002-U58 and FD001-U93.
- **`final_summary.predicted_rul` is not clipped.** It is the raw model output from `diagnose_engine` (FD002-U65 shows −1.04). The plotted `pred_*` series are clipped to [0, 125].

### 5.3 `demo_data.json` schema

```
{
  "model_info": {architecture, window_cycles: 30, rul_cap_cycles: 125, sensor_channels: 17,
                 param_count: 22273, tensor_payload_bytes: 89092, round_exchange_bytes: 178184},
  "engines": [ {                                   # 8 entries
      id: "FD00x-U<unit>", subset, unit, stage, final_true_rul, total_observed_cycles,
      cycles[N], true_rul[N], pred_local_mse[N], pred_fedavg_mse[N], pred_fedavg_asym[N],
      sensors: {s2[N], s3[N], s4[N], s11[N]},     # min-max-scaled values (DEMO_SENSORS), 4 dp
      diagnostics: {
        method: "baseline_x_gradient",
        subsystem_share[N][4],                      # round_to_total(…, 1), Fan/LPC/HPC/Turbine
        top_contributors[N][3] = [sensor, share_pct],
        top_drifting[N][3]     = [sensor, drift_z],
        drift_index[N],                             # mean |z|
        final_summary: <maintenance_summary dict for the last window>
      } } ],
  "diagnostics_meta": {subsystems, subsystem_labels, subsystem_sensors, sensor_info,
                       excluded_channels: [setting1, setting2, setting3],
                       model: "FedAvg (Asymmetric) global model, evaluated locally on the edge node",
                       attribution_formula, drift_formula, privacy, evaluation_boundary},
  "cluster": {physical_5r: [ {seed, rounds: [{round, rmse, nasa, avg_train_s, avg_rss_mb}]} ×5 ],
              sim_15r_seed42: [ {round, rmse, nasa} ×15 ]},
  "benchmarks": {asymmetric: {FD001, FD002},      # from results/asymmetric_loss_comparison.json
                 reproducible: [228 rows]}         # from results/reproducible/summary.csv (scope global/client)
}
```

Notes:

- **Data sources.** `cluster.physical_5r` is read from `results/fd002/results_physical_seed*.json` in sorted file order (seeds 100, 2024, 42, 7, 777). `sim_15r_seed42` comes from `results/fd002/sim_fedavg_r15_e2.json` → `runs["42"]`. Their content is Part 1 material.
- **Diagnostics run on the edge-side model.** Per-engine diagnostics use the **FedAvg (Asymmetric)** global model on the engine's **raw** test frame (`engine_diagnostics`), exactly as an edge node holding the global model would.
- **The `sensors` series do contain sensor values.** They are min-max-scaled traces of 4 public C-MAPSS sensors, used to draw the telemetry chart. The `diagnostics` sub-tree contains only shares, z-scores and text.
- **Compact formatting.** After `json.dumps(indent=2)`, a regex inlines the innermost scalar arrays to keep the file "diff-friendly but compact".

> **⚠ Boundary (data snapshot).** `docs/diagnostics.md` says the existing RUL predictions "regenerate bit-identically". The gated test `GeneratorRegenerationTest` (§6.2) enforces this only when `CMAPSS_ZIP` is set.

### 5.4 `server.py`: a zero-dependency server

The server uses only the standard library: `argparse`, `functools`, `http.server`, `json`, `socketserver`, `threading`, `webbrowser` and `pathlib`.

- **`DemoHandler(SimpleHTTPRequestHandler)`**, bound to `DASHBOARD_DIR` via `functools.partial(..., directory=…)`.
  - `GET /api/health` returns `{"status": "ok", "has_data": <demo_data.json exists>}` with an explicit `Content-Type: application/json` and `Content-Length`.
  - `GET /` is rewritten to `/index.html`. Everything else uses the static file handler.
  - **Directory-listing fix (PR #10).** `list_directory()` is overridden to `send_error(404)`. Previously, URLs such as `/__pycache__/` returned a listing despite the docstring.
  - `log_message` is silenced.
- **CLI:** `--host` (default `127.0.0.1`), `--port` (default `8765`), `--no-browser`.
  - `TCPServer.allow_reuse_address = True`.
  - The browser opens after 0.6 s via `threading.Timer`.
  - Ctrl+C prints "Shutting down dashboard." and exits with code 0, as the smoke test checks.
- **`--port 0` fix (PR #10).** The printed URL now uses `httpd.server_address[1]`, the actual bound port. Before the fix it printed `http://127.0.0.1:0/`. The tests rely on this to launch on an ephemeral port.

> **⚠ Boundary (security scope).** Only directory *listings* are blocked. Individual files inside `dashboard/` (for example `server.py`) are still served if requested by name; `test_no_directory_listing` asserts that `/sub/secret.txt` returns 200. The default bind address is loopback.

### 5.5 `index.html`: a self-contained vanilla page

The page uses a single inline `<script>` with no external scripts, stylesheets or fonts, inline CSS variables and hand-built SVG charts. On load, `boot()` runs `fetch("/demo_data.json")`, so the page is meant to be served by `server.py`. It then calls `applyLabels()`, `onEngineChange()`, `initClusterRuns()` and `renderBusiness()`.

**The three tabs** (`switchTab` toggles `hidden` / `active` on the `tab-asset`, `tab-cluster` and `tab-business` sections):

**1. Live Asset RUL** (`#tab-asset`):

- `engineSelect`: the 8 engines, labelled `id — stage (End RUL: n)`.
- **`cycleSlider`.** Range `0 … N−1`. It starts at the last window on engine change. `oninput` calls `renderAsset()`.
- **Play/Pause** (`togglePlayAsset`). Advances the slider by one step every **90 ms** and restarts from 0 if already at the end.
- Four KPI cards: FedAvg Asym, FedAvg MSE, Local only, and status plus safety margin (`vsTrue`, `earlyMargin`, `lateRisk`).
- `drawRulChart`: true RUL plus the three predictions up to the cursor, and a critical band below 20.
- `drawSensorChart`: s2/s3/s4/s11 scaled traces.
- **Diagnostics card** (`#diagCard`), synced with the slider:

| Element | Function | Rendering |
|---|---|---|
| `#diagBar` | `drawDiagnostics` | stacked bar, width = share %. Text only if share ≥ 7 %. |
| `#diagLegend` | `drawDiagnostics` | per-subsystem % plus sensor list |
| `#diagPrimary` | `drawDiagnostics` | pill with the argmax subsystem |
| `#diagTop` | `drawDiagnostics` | top-3 contributors, meter width `min(100, pct × 2)` % |
| `#diagDrift` | `drawDiagnostics` | top-3 drifting: ▲/▼, signed σ, colour danger if |z| ≥ 3, warn if ≥ 2, meter `min(100, |z| × 12)` % |
| `#diagTrend` | `drawDiagTrend` | stacked area of subsystem shares from the first window to the cursor, with a cursor line |
| `#diagSummary` | `drawDiagnostics` | severity from `pSafe` (≤ 20 / ≤ 45) plus primary subsystem plus top drivers |
| `#diagBoundary` | `applyLabels` | evaluation-boundary text (EN/IT) |

If an engine has no `diagnostics` block, the card is hidden.

**2. Federated Cluster** (`#tab-cluster`):

- `clusterRunSelect`: the five physical 5-round FD002 runs plus the 15-round simulation.
- `roundSlider` with "Play Rounds" (650 ms per step).
- Node and coordinator cards.
- RMSE and NASA charts per round (`drawRoundChart`).

**3. Safety & Business Impact** (`#tab-business`):

- Three KPI cards.
- The MSE-versus-Asymmetric table from `benchmarks.asymmetric`.
- The FD002 engine-imbalance table filtered from `benchmarks.reproducible`.
- The engineering note.

**Hours conversion.** `#hoursSelect` offers 0 ("cycles only"), 0.5, **1 (default)**, 2 and 4 hours per cycle. `fmtHorizon(c)` renders `"<c> cycles (~<c·mult> h)"`. Changing the select re-renders tab 1.

> **⚠ Boundary.** The mapping from cycles to hours is a user-chosen presentation scale. C-MAPSS defines cycles, not hours.

**Bilingual toggle.**

- `LANG` starts as `"it"` (`<html lang="it">`).
- `toggleLang()` flips the language, sets the button text to the *other* language, and calls `applyLabels()` and `renderAll()`.
- All strings live in the `I18N.it` / `I18N.en` tables. Italian sensor names are in `I18N.it.sensorNames`. English falls back to `diagnostics_meta.sensor_info[...].name`.
- The engine dropdown is rebuilt and keeps the selected index.

**PR #10 bug fixes** (commit `79c4dd1`, merged as `e4242b5`):

1. `server.py`: `--port 0` printed `:0` instead of the bound port.
2. `server.py`: directory listings were served. They now return 404.
3. `index.html`: the hours-scale options stayed Italian ("1 ciclo = …") in EN. `applyLabels` now relabels **every** option.
4. `index.html`: `toggleLang()` threw on `DATA.engines` before the data had loaded. `applyLabels` now returns after the static labels when `!DATA`, and `renderAll` returns early when `!DATA`.

> **⚠ Boundary (hard-coded presentation strings).** Some figures are literals in `index.html`, not values read from `demo_data.json`:
> - the node card titles "(Raspberry Pi 4)";
> - the fallback metrics "~16.7k windows", "12.8s/round", "RSS 580/600 MB";
> - the RMSE "Baseline 43.51" line;
> - "38.13 → 22.21 RMSE", "−38.0 % … 2,963 to 430" and "50–64 % → 20–34 %".
>
> These are Part 1 (federated/benchmark) claims. They were **not** cross-checked in this part. The "(Raspberry Pi 4)" labels name the intended deployment target; they do not report a diagnostics measurement on Pi hardware (§4.4.2).

---

## 6. Testing and verification architecture (`tests/`)

`master` has exactly three test modules: `tests/test_diagnostics.py`, `tests/test_dashboard.py` and `tests/test_diagnostics_validation.py`. All are `unittest`-based and can be run with `python -m unittest` or pytest.

| Module | Test classes | Test methods | Gating |
|---|---|---:|---|
| `test_diagnostics.py` | 6 | 22 | `RealDataIntegrationTest` needs `cmapss_zip_path()` (`$CMAPSS_ZIP`, `./CMAPSSData.zip` or `~/CMAPSSData.zip`) |
| `test_dashboard.py` | 6 | 18 | `node` on PATH is **mandatory**. Regeneration needs `$CMAPSS_ZIP`. Browser tests are skipped if no Selenium driver starts. |
| `test_diagnostics_validation.py` | 5 | 16 | `CmapssSmokeTest` needs `$CMAPSS_ZIP` |

### 6.1 `tests/test_diagnostics.py`: unit and integration tests for `diagnostics.py`

**Synthetic fixture.** `synthetic_fleet(n_units=6, n_cycles=80, n_regimes, wild_settings)`:

- Sensors start flat. Each regime adds +5 to every sensor.
- From cycle 40, an HPC ramp is injected: s3 rises by up to +8 and s11 by up to +6.
- `wild_settings` adds σ = 50 noise to the settings.

| Class | Test | What it proves |
|---|---|---|
| `ChannelMapTest` | `test_fourteen_fault_sensors_no_settings` | 14 fault sensors; settings disjoint; `SENSOR_INFO` and `SUBSYSTEM_SENSORS` cover them exactly; `SUBSYSTEMS` order; Table 2 spot checks |
| `AttributionTest` | `test_shapes_and_determinism` | all 3 methods: shapes (7,), (7,17), (7,14), (7,4); `batch_size` 3 equals the default |
| | `test_input_x_gradient_formula` | equals `|x·∂ŷ/∂x|` averaged over t, computed independently |
| | `test_baseline_x_gradient_formula_nontrivial_baseline` | equals `|(x−b)·∂ŷ/∂x|` with a random b (rtol 1e-9); predictions match the forward pass |
| | `test_empty_windows` | `(0,)` / `(0,17)` float64 for every method and for `baseline=None`; unknown method raises `ValueError` |
| | `test_conservation_sum_100` | sensor and subsystem shares sum to 100; non-negative; all-zero input gives a uniform split; rounded rows sum to 100.0 |
| | `test_settings_excluded_even_when_wild` | settings with σ = 1e3 still get exactly 0 |
| | `test_baseline_equal_input_gives_uniform` | `x == b` gives 0 attribution and 100/14 shares |
| `RoundToTotalTest` | `test_valid_shares_sum_exactly` | `[33.333, 33.333, 33.334] → [33.3, 33.3, 33.4]`; 50 Dirichlet draws at 0/1/2 dp |
| | `test_largest_remainder_choice` | `[10.9, 20.5, 68.6] → [11, 20, 69]` and other exact cases |
| | `test_invalid_inputs_raise` | negative, NaN, inf, NaN/negative total, remainder > n, remainder < 0, empty |
| `BaselineDriftTest` | `test_single_regime_drift` | `z[:30]` mean 0; s3/s11 end of life > 5σ; s8 < 2σ |
| | `test_multi_regime_drift_removes_operating_condition` | 6 regimes: s8 |z| < 5; s3 > 5σ; regime-aware s3 > 3× the naive `k=1` value |
| | `test_regimes_1d_and_empty_inputs` | `(3,)` and `(T,3)` agree; empty arrays give int dtype; `(4,2)` raises; empty `standardize` |
| | `test_healthy_reference_settings_untouched` | reference settings equal the scaled observed settings |
| `EngineAndSummaryTest` | `test_engine_outputs_aligned` | all 7 output shapes; first cycle = 30; shares sum to 100; settings 0; deterministic |
| | `test_short_engine_padded` | 12 cycles give 1 window at cycle 12 |
| | `test_summary_structure` | all keys present; 3 + 3 entries; no settings; sorted shares; s3/s11 top drifting; `primary_drift_subsystem == "HPC"`; privacy zeros; EN and IT text; JSON-serializable |
| | `test_summarize_non_final_index` | index 0, −1 and N−1 are consistent |
| | `test_severity_thresholds` | 10 → critical, 30 → warning, 90 → healthy |
| `RealDataIntegrationTest` *(gated)* | `test_fd001_engine`, `test_fd002_multi_regime_engine` | first training unit of FD001 (k = 1) and FD002 (k = 6): finite outputs, conservation, settings 0, summaries at index 0/−1; FD002 engine visits more than one regime |

### 6.2 `tests/test_dashboard.py`: data, generator, server and UI

- `sys.dont_write_bytecode = True` keeps `__pycache__/` out of `dashboard/`, and the server subprocess gets `PYTHONDONTWRITEBYTECODE=1`.
- The generator and server modules are imported **by path** (`_load`), because the repository root also contains a Flower `server.py`.

| Class | Test | What it proves |
|---|---|---|
| `DemoDataInvariantsTest` | `test_top_level_and_engines` | 5 top-level keys; 8 engines, 4 per subset; unique IDs with the `FD00x-U` prefix; valid stages |
| | `test_diagnostics_meta` | subsystem order; excluded channels = settings; sensor sets consistent; privacy 0 |
| | `test_per_engine_alignment_and_conservation` | every per-cycle array has length N; cycles sorted and ending at `total_observed_cycles`; each share row sums to 100 (±1e-6) and is ≥ 0; top-3 entries sorted and resolvable; `final_summary` consistent with the last row |
| `GeneratorUnitTest` (synthetic) | `test_diagnostics_meta_matches_module_and_committed_json` | `diagnostics_meta()` equals the committed JSON block |
| | `test_engine_diagnostics_shapes` | method name; lengths; no settings; sorted; final cycle 70; s3/s11 top drifting |
| | `test_round_to_total_conservation` | each rounded row equals `round_to_total(exact)`, is within 0.1 of it and has 1 dp; adversarial thirds |
| | `test_engine_diagnostics_deterministic` | identical JSON on re-run |
| `GeneratorRegenerationTest` *(gated on `CMAPSS_ZIP`)* | `test_predictions_identical`, `test_file_bit_identical` | runs `G.main()` into a temp dir (patching `G.__file__`) and checks that `demo_data.json` is **byte-identical** |
| `ServerSmokeTest` | `test_live_endpoints_and_clean_shutdown` | spawns `server.py --port 0 --no-browser` and parses the bound port; `/api/health` gives `{"status":"ok","has_data":true}`; `/` equals `index.html`; `/demo_data.json` has 8 engines; unknown path → 404; SIGINT → exit 0 with "Shutting down" |
| | `test_no_directory_listing` | `/sub/` → 404 without leaking the filename; `/sub/secret.txt` → 200 |
| `IndexStaticTest` | `test_inline_script_node_check` | exactly one inline script; `node --check` passes, plus a negative control (appending `function (` must fail) |
| | `test_script_dom_ids_exist` | every `getElementById("…")` target exists; key IDs present (the three tabs, `cycleSlider`, `engineSelect`, `langBtn`, `diag*`) |
| `DashboardBrowserTest` (headless Selenium) | `test_tabs_switch` | the three tabs toggle visibility and `active`; cluster SVG paths and business table rows render |
| | `test_diag_card_renders_on_tab1` | card visible; IT title; 3 + 3 rows; summary present |
| | `test_scrub_slider_all_engines` | for **all 8 engines**: real `change` event; slider max/value = N−1; HOME/END keys; a JS scrub over every index compares bar widths, top-3 codes, 4 trend paths, IT severity, primary-subsystem text and readout against `demo_data.json`; 0 JS errors |
| | `test_lang_toggle` | IT → EN → IT → EN: button text; tab and diag titles; **hours option labels in both languages** (PR #10 fix 3); summary language; engine index and cycle position kept |
| | `test_lang_toggle_before_data_loaded` | Chrome only: blocks `*demo_data.json*` via CDP `Network.setBlockedURLs`, toggles language, expects 0 new JS errors (PR #10 fix 4) |

**Browser harness.**

- `launch_driver()` tries headless Chrome (`--headless=new --no-sandbox --disable-gpu --window-size=1400,1000`), then Chrome through Selenium Manager (for a stale chromedriver on PATH), then Safari.
- An `ERROR_HOOK` collects `window.__errs` from `error` and `unhandledrejection` events. On Chrome it is injected with `Page.addScriptToEvaluateOnNewDocument`.
- `tearDown` asserts there were no JS errors and no SEVERE browser-log entries, ignoring `favicon.ico`.

### 6.3 `tests/test_diagnostics_validation.py`: validation helpers and artifact integrity

| Class | Test | What it proves |
|---|---|---|
| `SocketAuditTest` | `test_counts_local_create_connection_connect_and_send` | with sandbox-safe shims and a `socketpair`: `create_connection` and its `connect` count 2; `send` counts 3 and records the payload byte count |
| | `test_baseline_and_diagnosis_have_no_socket_activity` | `fit_healthy_baseline` plus `diagnose_engine` on synthetic data make 0 calls and send 0 bytes |
| `ValidationHelpersTest` | `test_engine_true_rul_caps_and_aligns_cycles` | `[1, 30, 50]`, last 50, final 10 → `[59, 30, 10]`; cap at 125 |
| | `test_single_fault_metrics_aggregate_and_filter_multiple_units` | rates on hand-built HPC/Fan units, and with the `hpc_only_units` filter |
| | `test_eol_profiles_use_last_ten_windows_per_unit` | mean of the last 10 of 12 windows |
| | `test_orient_and_two_fault_metrics_separate_fault_modes` | `orient` relabels by cosine to the reference; all three views separate 4 + 4 units; silhouette > 0.95; cross-view ARI 1.0 |
| | `test_saved_tensor_bytes_counts_autograd_storage` | at least the weight bytes; deterministic |
| `ArtifactIntegrityTest` | `test_network_seeds_and_latency_constraints` | JSON network audit = 0; subset and seed order; `threads == 1`; every latency value > 0 |
| | `test_documented_rmse_and_single_fault_rates_match_json` | exact strings in `docs/diagnostics.md` and `results/EXPERIMENTS.md` are regenerated from the JSON |
| | `test_documented_clustering_metrics_match_json` | the same check for the silhouette / ARI table rows and the prose |
| | `test_documented_latency_and_autograd_bytes_match_json` | latency table rows (3 dp), "Autograd buffers add 117,368 bytes per window", and the EXPERIMENTS.md range sentence |
| | `test_protected_benchmark_artifacts_are_regular_and_unchanged` | no symlinks; SHA-256 pins (below) |
| `PaperTreeDigestTest` | `test_ignored_latex_artifacts_do_not_change_digest` | in a temp git repo: `*.aux`, `*.log` and `main.pdf` (gitignored) leave the digest unchanged |
| | `test_tracked_edits_and_untracked_sources_change_digest` | a new non-ignored file or an edit changes the digest; deleting the new file restores it |
| `CmapssSmokeTest` *(gated)* | `test_validate_diagnostics_smoke` | runs the script on FD001 + FD003, seed 7, 1 epoch, 1 thread (timeout 1200 s); checks the config echo, network audit 0 and latency threads 1 |
| | `test_reproducible_pipeline_quick_smoke_uses_temp_output` | `experiments/reproduce.py --quick --threads 1` into a temp dir (timeout 1800 s); checks the output files; protected SHA values are unchanged before and after |

**Protected-artifact pins** (`test_protected_benchmark_artifacts_are_regular_and_unchanged`):

| Artifact | SHA-256 |
|---|---|
| `experiments/reproduce.py` | `0301de4cab66c12c5249ac6521adada3ae82b7241575061fe39867bd18a8a848` |
| `results/reproducible/summary.csv` | `67fc4b14fc1660365b19976e6c0ed6598a2f16e928a8892f18c3a03dcd032e86` |
| `paper/` tree digest | `b39b3f15a5b22d5c65a5edbb5975963efce3b10f30b21c24b68697b54e55b365` |

**How the `paper/` tree digest is computed** (`paper_tree_sha256`, PR #11 / commit `d399113`):

1. List files with `git ls-files -z --cached --others --exclude-standard -- paper`, so gitignored LaTeX build artifacts are excluded.
2. For each file in sorted order, hash `relative_path + "\0" + sha256(content)` into the running digest.
3. Tracked files that have been deleted are skipped, so their absence changes the digest.

**Docs coupled to the JSON.** `ArtifactIntegrityTest` treats `docs/diagnostics.md` and `results/EXPERIMENTS.md` as artifacts that must quote the JSON exactly. Editing a number in those files without regenerating the JSON, or the other way round, fails the suite. This handbook (`docs/project_guide_part2.md`) is **not** checked by any test. Its numbers were copied by hand from the same JSON.

---

## Appendix A: symbol and file index

| Symbol | Location | Kind |
|---|---|---|
| `USEFUL_SENSORS`, `SEQ_LEN = 30`, `MAX_RUL = 125` | `common.py` | constants (17 channels) |
| `CmapssRegressor` | `common.py` | 1D-CNN RUL model |
| `cmapss_zip_path`, `load_cmapss_raw`, `load_rul`, `apply_scale`, `build_test_set` | `common.py` | data I/O |
| `get_loss_fn`, `AsymmetricRULLoss`, `train_epochs`, `make_xy`, `make_loader`, `split_units`, `predict`, `rmse`, `set_seed` | `common.py` | training helpers reused by validation and dashboard |
| `compute_scales_from_train` | `benchmark_asymmetric_loss.py` | healthy-regime (first 50 cycles per unit) min/max |
| `SETTINGS`, `FAULT_SENSORS`, `SUBSYSTEMS`, `SENSOR_INFO`, `SUBSYSTEM_LABELS`, `SUBSYSTEM_SENSORS`, `FAULT_IDX`, `SETTING_IDX`, `_SUB_MATRIX` | `diagnostics.py` | channel map |
| `N_REGIMES`, `SEVERITY_THRESHOLDS`, `ATTRIBUTION_METHODS`, `PRIVACY_METADATA`, `EVALUATION_BOUNDARY` | `diagnostics.py` | configuration constants |
| `scale_frame`, `sliding_windows` | `diagnostics.py` | preprocessing |
| `HealthyBaseline`, `fit_healthy_baseline` | `diagnostics.py` | regime-aware drift and baseline |
| `attribute_windows`, `sensor_shares`, `subsystem_shares`, `subsystem_drift`, `round_to_total` | `diagnostics.py` | attribution and aggregation |
| `severity_for`, `maintenance_summary`, `EdgeDiagnostics` | `diagnostics.py` | summary and driver |
| `SocketAudit`, `single_fault_metrics`, `eol_profiles`, `cluster`, `reference_signature`, `orient`, `signature`, `two_fault_metrics`, `saved_tensor_bytes`, `latency`, `run_subset` | `experiments/validate_diagnostics.py` | validation |
| `train_subset_models`, `engine_diagnostics`, `diagnostics_meta`, `pick_representative_engines`, `load_cluster_rounds`, `load_summary_highlights`, `DEMO_SENSORS` | `dashboard/generate_demo_data.py` | dashboard data generation |
| `DemoHandler`, `DASHBOARD_DIR` | `dashboard/server.py` | HTTP server |
| `I18N`, `applyLabels`, `toggleLang`, `switchTab`, `renderAsset`, `drawDiagnostics`, `drawDiagTrend`, `togglePlayAsset`, `fmtHorizon`, `boot` | `dashboard/index.html` | UI |

## Appendix B: consolidated list of boundaries and caveats

1. **No component labels.** C-MAPSS has no per-unit component labels, so no supervised component accuracy is claimed (§1.3).
2. **Measurement location, not fault location.** Subsystem shares reflect where the sensors are measured, not which module is faulty. The HPC-versus-Turbine split on FD001 depends on assigning phi (s12) to the combustor (§2.3, §4.2.3).
3. **`input_x_gradient` is unvalidated.** It is implemented and unit-tested but not in the validation JSON (§2.6).
4. **Naive drift failure is synthetic-only.** It is shown by a synthetic unit test, not by a real-data ablation (§3.1).
5. **Baseline fit includes held-out units.** In validation, the healthy baseline is fitted on all training units, held-out units included (§3.3).
6. **Mode naming is unconfirmed.** Fan-mode naming on FD003/FD004 is a physically consistent hypothesis without ground truth. Share bars alone do not separate modes, and FD004 attribution-only clustering is not seed-stable (§4.3).
7. **Latency is host-only.** Latency was measured on a Darwin arm64 development host, single thread, Python 3.11.4, torch 2.15.0. **No Raspberry Pi 4 measurement exists.** FD001 timing may overlap another job. Peak RSS is for the whole process (§4.4.2).
8. **Socket audit scope.** The audit covers specific `socket` entry points in one process (§4.4.3).
9. **Dashboard limits.**
   - `demo_data.json` contains scaled sensor traces for 4 sensors; only the `diagnostics` sub-tree is free of sensor values.
   - Hours are a presentation scale.
   - Several tab 2 and tab 3 figures are hard-coded literals not cross-checked here.
   - `server.py` blocks listings, not individual files (§5).
10. **Source discrepancies.** `readme.txt` swaps the FD004 train/test trajectory counts (248/249 versus 249/248 unit IDs in the files) and misnumbers the last column as "sensor measurement 26" (§1.3, §2.1).
11. **Guide numbers are not re-run.** All numbers in this guide come from committed artifacts at `9182089`. No experiment or test suite was re-executed to produce them.
