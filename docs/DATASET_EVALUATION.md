# Vehicle-fleet datasets for FedPdM-Edge

Evaluated 2026-11-19 for an OEM coordinating federated predictive maintenance
across customer vehicles of the same model. Recommendations below are engineering
judgments, not results measured on the external datasets.

## Decision and evidence boundary

1. **Immediate expansion: combined 709-unit C-MAPSS training fleet**, with
   **FD004 alone** as the harder, consistent-subset swarm. Preserve the official
   707-engine combined test split; never pool it into training clients.
2. **Next real-world automotive benchmark: Scania Component X.** It best matches
   manufacturer-to-customer fleet telemetry and maintenance outcomes. A
   specification-conditioned cohort is a useful model/variant proxy; neither
   actual customer ownership nor a single chassis model is established publicly.
3. **Supervised diagnostics benchmark: N-CMAPSS.** Use physical health truth to
   validate component attribution, separately from real automotive claims.

Only `/Users/eldemaster/orchestrator/CMAPSSData.zip` is locally present for this
work. Size: **12,425,978 bytes**; SHA-256:
`74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`.
Counts below were independently read from its text files. Scania, N-CMAPSS, APS
and battery data were **not downloaded or benchmarked**. CSV/HDF5 tests use
explicitly synthetic schema fixtures. A successful optimizer step on those
fixtures establishes software compatibility, not predictive accuracy.

## Decision matrix

Edge footprint means local input/model/storage burden, not measured latency.
Counts describe assets or records as specified, not interchangeable fleet sizes.
Primary sources and qualifications follow the matrix.

| Dataset | Data realism | Vehicle swarm size | Natural Non-IID axis | Temporal RUL support | Component failure labels | Edge footprint | Licensing / download |
|---|---|---|---|---|---|---|---|
| C-MAPSS FD001–004 / combined | Simulated same-type turbofans | 709 train / 707 test; per subset below | Subset, settings/regimes | Exact cycles; truncated test endpoints | Subset fault capabilities, no unit fault identity | Small archive; existing 17-channel CNN | Public NASA ZIP; portal license unspecified |
| Scania Component X | Actual commercial trucks, anonymized/perturbed | 23,550 / 5,046 / 5,045 train/validation/test | `Spec_0..Spec_7`; operational histories | Repair TTE for events; censoring otherwise | One anonymized component's repair outcome | 105 channels; irregular readouts; larger offline storage | CC BY 4.0; SND/Researchdata.se, IDA2024 |
| N-CMAPSS | Simulated damage under recorded commercial-flight envelopes | 128 units in published release | `Fc` flight class | Flight-cycle RUL | Physical degradation truth for five rotating components | High-rate, many-GB HDF5; downsample deliberately | CC0 1.0 dataset; NASA item 17 |
| Scania APS | Real truck summaries | 60,000 train / 16,000 test **rows** | Counter/histogram distributions; no customer axis | No trajectories or RUL | Binary APS-related failure | Small tabular input; no temporal CNN justification | UCI ~54 MB; license discrepancy below |
| NASA / CALCE / Stanford-MIT batteries | Primarily laboratory cells | Collection-dependent; Stanford-MIT 124 cells | Charge protocol, temperature, cell/batch | Capacity/cycle-life targets after defining EOL | Battery health, not whole-vehicle root cause | Small derived cycle features; larger raw waveforms | Collection-specific; links below |

## C-MAPSS: expand the controlled baseline

The NASA description supplies run-to-failure training trajectories, held-out RUL,
three settings and 21 sensors. FD002/004 introduce six regimes; FD003/004 allow
HPC and fan degradation. These are useful controlled stressors for heterogeneous
clients. Limits are simulated turbofans, anonymous unit indices rather than real
vehicle identities/specifications, no per-unit component-failure labels, and one
snapshot per cycle. Uniform cycle indexing is not realistic irregular customer
telemetry or a guarantee of uniform wall-clock intervals. See the
[NASA dataset description](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).

| Subset | Actual train units | Actual test units | Regimes | Possible fault modes |
|---|---:|---:|---:|---|
| FD001 | 100 | 100 | 1 | HPC |
| FD002 | 260 | 259 | 6 | HPC |
| FD003 | 100 | 100 | 1 | HPC, fan |
| FD004 | **249** | **248** | 6 | HPC, fan |
| Combined | **709** | **707** | Mixed subsets | Mixed subset capabilities |

**Source discrepancy:** the local archive README and NASA portal reverse
FD004's train/test counts (248/249). The actual `train_FD004.txt` contains 249
unique units, `test_FD004.txt` contains 248, and its RUL file has 248 labels.
The conditional integration test counts archive IDs independently of the adapter.

Use `FD002-train-U001` and `FD002-test-U001` as distinct identities. Combining
subsets enlarges client count, but mixes experimental populations; it is not
evidence of 709 real cars or customer-level partitions. Subset grouping is a
controlled Non-IID experiment. FD004's regime variation is also a useful
within-subset stressor, although regimes can change within an engine: do not
split its rows across clients. Group vehicles by training-prefix operating
profiles if finer partitions are needed.

## Scania Component X: strongest automotive fit

The 2024 release is cited as Lindgren et al.; the descriptor's first author is
Kharazian. The preprint/challenge is from 2024, while the Scientific Data article
is **12, 493 (24 March 2025)**. Actual onboard readouts, workshop records and
factory specifications connect by anonymous vehicle ID. Repairs proxy failure;
readout/repair frequencies and values were modified for confidentiality. A
Scania fleet is not proof of a single vehicle model. These limitations preclude
claims about actual failure rates or named components. See
[Kharazian et al., Scientific Data](https://www.nature.com/articles/s41597-025-04802-6).

### Public schema

The [2024 author preprint, Data Records](https://arxiv.org/html/2401.15199v1#S3)
describes 1,122,452 training readouts and 23,550 trucks; validation has 196,227
readouts / 5,046 trucks, test 198,140 / 5,045. The precise CSV contract is:

| File | Keys / columns | Meaning |
|---|---|---|
| `train_operational_readouts.csv` | `vehicle_id`, `time_step`, 105 numeric features | Irregular, vehicle-relative time; 107 columns total |
| Histogram features | `167_0..9`, `272_0..9`, `291_0..10`, `158_0..9`, `459_0..19`, `397_0..35` | 97 bins across six anonymized variables |
| Counter features | `171_0`, `666_0`, `427_0`, `837_0`, `309_0`, `835_0`, `370_0`, `100_0` | Eight cumulative counters |
| `train_tte.csv` | `vehicle_id`, `length_of_study_time_step`, `in_study_repair` | Study end; event indicator 1=repair, 0=right-censored |
| `train_specifications.csv` | `vehicle_id`, `Spec_0..Spec_7` | Eight categorical specs, e.g. `Cat0`; not sensor channels |
| `validation_*` / `test_*` | Same readout/specification layouts; `*_labels.csv`: `vehicle_id`, `class_label` | Endpoint classes 0–4, not point RUL |

The preprint's prose counts TTE outcome columns separately from its vehicle key.
Read joins by `vehicle_id`, never by row position. There are 2,272 observed
training repairs and 21,278 censored trucks.

### Targets, scoring and fleet mapping

For an observed repair at study end `E`, target a window ending at time `t` with
`min(cap, E - t)` in the source's anonymized time units. For non-events, `E - t`
is a **lower bound**, not a failure time. The adapter retains these vehicles and
their bounds but emits no ordinary regression windows for them. This complete
case regression subset is selection-biased; a later survival objective is needed
to exploit all trucks. Do not translate `time_step` into days or kilometers.

The challenge classes correspond to >48, 48–24, 24–12, 12–6 and 6–0 time units
before repair, respectively. Its asymmetric matrix charges 7–10 for unnecessary
alarms and 200–500 for missed/late failures. Evaluate the five-class maintenance
cost at each held-out vehicle's final readout, using the published boundaries and
matrix; bin-edge inclusivity needs an explicit protocol. `AsymmetricRULLoss` is
a regression surrogate and **does not reproduce** that challenge score. See the
[official challenge call](https://doris.snd.se//api/file/2024-34/1/documentation?filePath=2024_IDA_challenge_v2.pdf).

One truck maps naturally to one simulated edge client; an OEM aggregates model
updates. Group identical selected specification tuples (default `Spec_0`), or
attach operational-profile metadata derived from a training prefix. Full
eight-spec tuples may create tiny clients. These are variant/usage cohorts,
**not identified customers**. Split vehicles before windows and scaling; compare
within-spec cohorts to approximate a same-model deployment and hold out variants
to study generalization. Retain irregular timestamps; fixed readout-count windows
cover variable time spans. Time-delta features/resampling would require a
separate, justified experiment.

### Availability, license and storage

The [official SND / Researchdata.se catalogue](https://researchdata.se/en/catalogue/dataset/2024-34)
offers open CSVs under **CC BY 4.0**, with attribution/version DOI required.
Version 2 (2024-07-03, DOI `10.5878/jvb5-d390`) added test labels; version 3
(2025-04-09, `10.5878/bnh5-ka77`) updated documentation. The catalogue lists
11 files totaling 1.54 GiB, including two PDFs. Training readouts alone are about
1.14 GB. Budget additional pandas/window memory. Start from the
[IDA2024 challenge page](https://ida2024.blogs.dsv.su.se/industrial-challenge/)
or the catalogue. **No official Zenodo record was verified**; do not invent a
Zenodo download or equate an unverified mirror with the versioned release.

## N-CMAPSS: supervised component attribution

[Arias Chao et al. (2021)](https://www.mdpi.com/2306-5729/6/1/5)
uses recorded commercial-flight envelopes with **simulated** engine degradation:
128 units, eight datasets (DS08 split into a–e), roughly 90 million readings.
`Fc=1/2/3` denotes 1–3 h / 3–5 h / >5 h flights. This supports realistic usage
heterogeneity, not observed airline failures.

| HDF5 arrays (`*_dev`, `*_test`) | Role |
|---|---|
| `W` | Four conditions: altitude, Mach, throttle angle, inlet temperature |
| `X_s` | 14 measured sensor channels |
| `X_v` | Virtual sensors; privileged/model-derived telemetry |
| `T` | Physical health parameters for fan, LPC, HPC, HPT, LPT efficiency/flow |
| `Y` | Per-reading RUL, in flight cycles |
| `A` | `unit`, `cycle`, `Fc`, `hs` metadata |
| `W_var`, `X_s_var`, `X_v_var`, `T_var`, `A_var` | Column names |

Health traces enable supervised attribution validation. Fault-mode combinations
vary by subset; continuous health parameters are not ready-made one-hot labels.
Define/validate a component-degradation target before scoring attribution. Keep
`T` and `hs` out of prognostic inputs, and distinguish measured-only experiments
from virtual-sensor experiments. The paper specifies **CC0 1.0 for the dataset**
(distinct from article licensing).

Download [NASA PCoE repository item 17](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/),
“Turbofan Engine Degradation Simulation-2”, extract a `.h5`, and install optional
`h5py`. The full high-rate release needs many GB of storage/RAM. The adapter
reads bounded chunks but returns an eager DataFrame fleet; it is not an
out-of-core training system. Probe one file with a documented `row_stride`
first. Striding selects file rows, changes temporal resolution, and may drop
unit endpoints; `test_ruls` then describes the **last retained** sample. Use
`row_stride=1` for original endpoint evaluation and report subsampling effects.

## Other automotive and EV options

**Scania APS:** UCI provides 60,000 training and 16,000 test examples with
`class` (`pos`/`neg`) plus 170 anonymized features, including counters and
histograms. Training positives number 1,000. There are missing `na` values and
costs of 10 per false alarm versus 500 per missed APS fault. There is no
vehicle-linked temporal trajectory or RUL target. It tests imbalanced,
cost-sensitive tabular classification; imposing sliding windows would fabricate
time. This makes Component X the stronger temporal swarm benchmark.
UCI's current metadata says CC BY 4.0, while its embedded original description
says GPLv3-or-later: retain provenance and resolve the discrepancy before
redistribution. The listed download is approximately 54 MB. See
[UCI APS dataset and original description](https://archive.ics.uci.edu/dataset/421/aps%2Bfailure%2Bat%2Bscania%2Btrucks).

**EV batteries:** choose a battery-health task if that is the product target.
[NASA battery and randomized-usage collections](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
cover controlled cycling, temperature and usage variation.
[CALCE](https://calce.umd.edu/battery-data) provides chemistry/cell-specific aging
and operating-condition data. The
[Stanford-MIT / Toyota Research Institute release](https://data.matr.io/)
contains 124 commercial lithium-ion cells cycled to failure under fast charging,
listed as CC BY 4.0. These are lab-cell proxies, not 124 customer EVs. Define
capacity-based end-of-life and RUL consistently; partition by cell/protocol/batch,
never cycles shared across train/test. Charge policy and temperature offer
meaningful Non-IID axes, but pack thermal coupling, driving, BMS behavior and
whole-vehicle repairs remain unvalidated. NASA/CALCE terms, file sizes and formats
depend on the selected collection; verify download-specific terms rather than
assigning one blanket license. Battery adapters are outside this change.

## Adapter use and experiment contract

`vehicle_datasets.py` is an opt-in module; it does not wire these datasets into
the existing experiment runner or reinterpret existing benchmark results.

```python
from vehicle_datasets import (
    CmapssCombinedAdapter, ScaniaComponentXAdapter, NCMAPSSAdapter,
    build_flexible_cnn,
)

# common's ZIP lookup uses CMAPSS_ZIP, ./CMAPSSData.zip, then ~/CMAPSSData.zip.
train = CmapssCombinedAdapter().load()           # 709 training engines
test = CmapssCombinedAdapter().load(split="test")  # 707 held-out engines
subset_clients = train.partition_by("subset")
fd004 = CmapssCombinedAdapter().load("FD004")

# After explicitly downloading the external data:
scania = ScaniaComponentXAdapter("/data/scania")
trucks = scania.load()
clients = scania.partition_by_specifications(trucks, ["Spec_0", "Spec_1"])
client_ids = next(iter(clients.values()))
X, y = trucks.windows(client_ids, seq_len=20, stride=4, max_rul=48)
model = build_flexible_cnn(len(trucks.feature_cols), 20)  # input (N, W, C)

aviation = NCMAPSSAdapter("/data/N-CMAPSS_DS02-006.h5")
engines = aviation.load(split="dev", row_stride=10)
flight_clients = aviation.partition_by_flight_class(engines)
```

Each `VehicleFleetDataset` supplies `name`, `vehicles`, `feature_cols`,
`setting_cols`, `metadata` and optional `test_ruls`. Metadata captures subset,
specifications or flight class without placing labels in features. Raw vehicle
frames retain missing values; `windows()` calls `fed_norm.sanitize_telemetry_frame`
on each observed window in channel blocks, avoiding future/other-vehicle
imputation. Its stuck-channel fallback can turn constant windows into zero;
this inherited behavior deserves a dataset-specific ablation before benchmarking.

Windows are float32 `(N, W, C)` with a capped final-readout target `(N,)`.
Sliding mode omits short histories; `last_only=True` pads them on the left by
repeating the first sanitized reading. Empty/censored selections return empty
arrays. Train scalers only on training clients, using `feature_cols` explicitly
with existing normalization APIs. Do not transfer C-MAPSS's cycle cap blindly
to anonymized truck time or flight-cycle labels. C-MAPSS train targets derive
from each engine's final cycle; test targets use official endpoint RUL, and
official test evaluation should use `last_only=True`.

The flexible CNN reuses `common.CmapssRegressor`'s forward pass and changes only
the first convolution and pooled-length-dependent dense layer. At `(17,30)` it
has identical state-dict shapes and outputs with shared weights: **22,273
parameters**, about 87 KiB float32 weights. `(105,30)` has 30,721 parameters,
about 120 KiB. A `(30,105)` input is 12,600 bytes before activations/optimizer
state. These calculated sizes are not measured edge RAM/latency. All federated
clients need identical feature ordering, channel count and sequence length;
arbitrary-channel support does not make different schemas directly aggregatable.

## Verification

The focused suite checks baseline CNN parity and gradients at other dimensions;
synthetic Scania CSV ingestion, censoring, causal imputation, disjoint clients and
one asymmetric-loss optimizer step; synthetic official-layout HDF5 ingestion,
health-truth isolation, chunk/stride consistency, optional-dependency errors and
flight-class partitions; and conditional archive counts, official RUL alignment,
split isolation and subset grouping.

```sh
python3 -m unittest tests/test_vehicle_datasets.py -v
CMAPSS_ZIP=/Users/eldemaster/orchestrator/CMAPSSData.zip \
  python3 -m unittest tests/test_vehicle_datasets.py -v
python3 scripts/preflight_check.py --skip-unittest
```

Local verification uses the working Python 3.11 environment at
`/private/tmp/fedpdm-preflight-env/bin` at the front of `PATH`, with
`PYTHONDONTWRITEBYTECODE=1` and `OMP_NUM_THREADS=1`. Default system Python's pandas
extension has an incompatible architecture. Optional `h5py` was installed into
the temporary environment, without editing project dependency files.

Observed results: without `CMAPSS_ZIP`, 15 cases discovered, 14 passed and the
archive integration case skipped; with the local ZIP, all 15 passed, including
the HDF5 fixture tests. Preflight reported **OVERALL: PASS**, zero repository
symlinks, and canonical experiment/summary/paper hashes. An additional scope
audit confirmed all 35 protected files matched the original HEAD and only the
three requested files were added.
