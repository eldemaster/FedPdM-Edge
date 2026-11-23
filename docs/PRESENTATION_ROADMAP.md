# FedPdM-Edge pitch deck brief for Claude Design

Use this document as a production brief, not as loose inspiration. It is written so the complete **Ready-to-Paste Claude Design Master Prompt** at the end can be copied directly into Claude Design, with this file and `PROJECT_SPEC.md` attached when supported.

**Evidence snapshot:** `master` at commit `bb27348`, after PR #25 and PR #26. Every quantitative claim and repository path below was checked against the files named in the evidence ledger. Refresh the audit before reusing the deck on a later revision.

**Positioning:** FedPdM-Edge is a validated research proof of concept and a blueprint for a segregated, advisory-only Phase-1 industrial pilot. It is not a production maintenance system, safety controller, road-vehicle validation, or financial-ROI study. C-MAPSS remaining useful life (RUL) is measured in simulated engine cycles, not automatically in hours.

## Design system for the whole deck

- **Format:** 16:9 widescreen; 13 slides; approximately 20 minutes including a crisp live demo.
- **Visual language:** dark industrial-tech, spacious, diagram-led, and credible rather than futuristic. Use the dashboard palette: background `#0b111b`, primary cyan `#38bdf8`, positive green `#22c55e`, caution amber `#f59e0b`, and personalization/risk pink `#f472b6`. Use white for primary text and cool gray for caveats.
- **Typography:** a clean geometric sans for titles and body; a monospaced face only for paths, commands, method identifiers, and API routes. Titles should be short and assertive. Use one large claim per slide, no more than three short on-slide bullets, and move qualifications into speaker notes.
- **Hierarchy:** title 30–36 pt, key number 28–34 pt, body 18–22 pt, source/caveat footer 10–12 pt. Keep strong contrast on `#0b111b`; never encode meaning by color alone.
- **Evidence labels:** visibly badge each claim as **SIMULATION**, **PHYSICAL**, **LOCAL AUDIT**, **ADAPTER READY**, or **PROPOSED PILOT**. Never blend evidence classes.
- **Charts:** plot only the series supplied below. Preserve method and partition names. Show units on every axis. Use direct labels where possible, no decorative 3D, and no invented interpolation. Raw values are supplied at full artifact precision; display labels may be rounded as instructed.
- **Image policy:** favor vector diagrams, dashboard frames, restrained industrial silhouettes, and data graphics. Do not generate photographs that imply a real customer, connected truck fleet, factory deployment, or measured business outcome.
- **Notes policy:** on-slide copy stays punchy; speaker notes carry provenance, protocol, limitations, and transitions. Put a small source footer on every evidence slide.

## Audience and decision

The deck should secure agreement to scope an isolated, human-reviewed pilot: identify assets and telemetry owners, map tags and units, choose success and stop criteria, and assign OT, maintenance, IT, security, and ML owners.

| Audience | Decision lens | Message to land |
|---|---|---|
| Plant/OT and maintenance | Useful warning, nuisance inspections, workflow fit, safe operating boundaries | Earlier risk signals and local explanations can support a human decision; no automatic shutdown and no measured savings are claimed. |
| Security and IT | Data boundary, identity, network access, update leakage, rollback | Raw training windows are not serialized as model updates, but updates and shared statistics can still leak information; configured mTLS is not secure aggregation. |
| ML and data engineering | Generalization, Non-IID behavior, labels, drift, resource envelope | The repository now tests strictly vehicle-disjoint 10- and 20-client swarms and provides real-fleet adapters, while keeping external-dataset accuracy claims explicitly open. |

## Slide-by-slide production specification

### Slide 1 — Learn across fleets. Keep telemetry local. — 0:45

- **Layout archetype:** hero.
- **Visual and layout:** full-bleed `#0b111b`; a thin cyan line connects several customer-fleet gateway icons to an OEM coordinator. Sensor traces stop at each gateway; only small model-update packets cross the boundary. Keep the title left aligned and place a small **RESEARCH PoC → PILOT BLUEPRINT** badge beneath it.
- **On-slide copy:**
  - Privacy-Preserving Predictive Maintenance at the Industrial Edge
  - Vehicle-disjoint learning, local diagnostics, pilot-ready evidence boundaries
- **Chart data:** none; this is a conceptual hero, not evidence.
- **Speaker notes:** “Can an OEM improve a shared RUL model without pooling customer sensor histories? This PoC tests that architecture under controlled vehicle-swarm heterogeneity and gives us an honest path to an advisory pilot.”
- **Source footer:** `PROJECT_SPEC.md`; `docs/vehicle_swarm.md`.

### Slide 2 — The maintenance gap is both technical and organizational — 1:00

- **Layout archetype:** split.
- **Visual and layout:** left side shows isolated fleets with different duty profiles and data volumes; right side shows an RUL timeline with an early estimate, true endpoint, and late estimate. Use amber for operational uncertainty and pink only for the costly late-error path.
- **On-slide copy:**
  - Fleets differ in size, lifespan, operating regime, and degradation pattern
  - Late RUL estimates can postpone action; early estimates can trigger unnecessary work
  - A single score needs a reviewable local explanation
- **Chart data:** none; use labeled conceptual geometry only.
- **Speaker notes:** Explain that `AsymmetricRULLoss` penalizes RUL overestimation more heavily than underestimation. State plainly that no company-specific failure costs, savings, or alert outcomes were measured.
- **Source footer:** `common.py`; `results/asymmetric_loss_comparison.json`; `docs/vehicle_swarm.md`.

### Slide 3 — One architecture, two privacy boundaries — 1:30

- **Layout archetype:** architecture flow.
- **Visual and layout:** two horizontal lanes. Top: **OFFLINE FEDERATED TRAINING** from customer fleets through local windows, decentralized normalization, local training, model updates, and OEM aggregation. Bottom: **LOCAL EDGE INFERENCE** from stream to rolling window, RUL, diagnostics, debounced alert, and dashboard summary. Draw raw telemetry as locked inside the edge box. Use dashed outlines around components that still require integration.
- **On-slide copy:**
  - Training: model parameters and counts move; raw windows do not
  - Inference: RUL and diagnostics are computed locally
  - Transport controls exist; update privacy still needs a formal decision
- **Chart data:** none. Show these verified payload callouts: **22,273 float32 parameters**, **89,092 bytes one way**, **178,184 bytes download + upload per client per round**, excluding protocol and TLS overhead.
- **Speaker notes:** `server.py` and `client.py` implement configured TLS/mTLS; explicit `--insecure` is for local development. Optional client-delta clipping and Gaussian noise exist, but there is no epsilon/delta accountant or secure aggregation. The vehicle-swarm normalization server receives `LocalStats`, not raw arrays; aggregate statistics can still disclose information.
- **Source footer:** `results/edge/local-smoke-macos.json`; `fed_norm.py`; `server.py`; `client.py`; `edge_runtime.py`.

### Slide 4 — The new benchmark models customer fleets, not row shards — 1:30

- **Layout archetype:** 3-column KPI.
- **Visual and layout:** three large KPI cards over a faint fleet-partition map. Card 1: client scale; card 2: partial participation; card 3: normalization boundary. Under the cards, show three partition ribbons: IID, lifespan/quantity skew, regime/drift skew.
- **On-slide copy:**
  - **10 / 20 clients** — every vehicle belongs to exactly one customer fleet
  - **50% / 30% participation** — exactly 5 of 10 or 6 of 20 clients each round
  - **0 raw sensor rows shared** — normalization aggregates `LocalStats`
- **Chart data:** none; KPI values come directly from the benchmark configuration and audit fields.
- **Speaker notes:** The committed configurations are FD002 `K=10`, FD004 `K=10`, and FD004 `K=20`. `lifespan_quantity_skew` uses Dirichlet `alpha=0.5`. `regime_fault_skew` uses operating-regime profiles with Dirichlet `alpha=0.3`; on FD004 its second signal is a label-free end-of-life drift cluster, not a ground-truth component fault. The run uses five rounds and five seeds; it is an in-process simulation, not 10 or 20 networked sites.
- **Source footer:** `experiments/vehicle_swarm_benchmark.py`; `results/vehicle_swarm_benchmark.json`; `docs/vehicle_swarm.md`.

### Slide 5 — Non-IID stress is measured, not asserted — 1:20

- **Layout archetype:** chart + takeaway.
- **Visual and layout:** three aligned grouped bar charts for the FD004 `K=20`, `fraction_fit=0.3` configuration. Categories are IID, lifespan/quantity, and regime/drift. Panels are supervised-window count CV, lifespan pairwise JS, and drift-cluster pairwise JS. Cyan = IID, amber = lifespan/quantity, pink = regime/drift. Put the takeaway in a right-side callout.
- **On-slide copy:**
  - “Both intended Non-IID partitions exceed IID on the combined heterogeneity score in every committed configuration.”
  - Small footer: larger values mean more inter-client skew; metrics have different scales.
- **Exact chart data:** use the raw values below; show three-decimal labels.

| Partition | Count CV | Lifespan pairwise JS | Drift-cluster pairwise JS | Combined score |
|---|---:|---:|---:|---:|
| `iid_swarm` | 0.09934287033540237 | 0.1751300089234043 | 0.041063359435593604 | 0.399106069182159 |
| `lifespan_quantity_skew` | 1.1934111966410597 | 0.7863888999119484 | 0.21478617939836436 | 2.5007266183637693 |
| `regime_fault_skew` | 1.0584645082839426 | 0.39538933754460825 | 0.3718889501356717 | 2.00643451732452 |

- **Speaker notes:** Count CV uses supervised-window counts. Jensen-Shannon values compare client distributions. The combined score is a benchmark summary, not a universal heterogeneity unit. FD004 has no per-unit component label; use “drift cluster,” not “fault diagnosis.”
- **Source footer:** `results/vehicle_swarm_benchmark.json` → FD004 / `k=20` / each `heterogeneity_aggregate`; interpretation in `docs/vehicle_swarm.md`.

### Slide 6 — Optimize for the maintenance decision, then personalize carefully — 1:45

- **Layout archetype:** chart + takeaway.
- **Visual and layout:** three small-multiple horizontal bar panels for FD004 `K=20`, `regime_fault_skew`: RMSE, late-prediction rate, and NASA score per test vehicle. Compare four methods only. Use gray for local-only, cyan for FedAvg MSE, amber for FedAvg asymmetric, and pink for personalized asymmetric. Add a small “FedProx ≈ FedAvg here” annotation without plotting duplicate bars.
- **On-slide copy:**
  - MSE gives the lowest symmetric error
  - Asymmetric training sharply lowers late-risk metrics
  - Personalization helps on average, not for every customer
- **Exact chart data:** use these raw aggregate means; label RMSE and NASA to two decimals and late rate as one-decimal percent.

| Method | RMSE cycles | Late-prediction rate | NASA / test vehicle |
|---|---:|---:|---:|
| `local_only` | 46.80298195680963 | 0.5338709677419355 | 924.3840377945577 |
| `fedavg_mse` | 45.75365206396756 | 0.4588709677419355 | 362.75603482634654 |
| `fedavg_asym` | 49.676721538980495 | 0.3419354838709677 | 110.27362077219509 |
| `fedavg_asym_personalized` | 48.46673124683082 | 0.34838709677419355 | 97.33219311186204 |

- **Speaker notes:** This is five-seed simulation evidence under a five-round bounded protocol. `FedProx(mu=0.05)` does not materially change the displayed results. Across all nine committed configurations, personalization improves matched-client mean validation RMSE in eight, but some worst-client RMSE values worsen. Do not frame personalization as a universal fairness improvement.
- **Source footer:** `results/vehicle_swarm_benchmark.json` → FD004 / `k=20` / `regime_fault_skew` / `aggregate`; `docs/vehicle_swarm.md`.

### Slide 7 — The data path now extends beyond one benchmark schema — 1:30

- **Layout archetype:** 3-column KPI.
- **Visual and layout:** three vertical dataset cards linked to a shared `VehicleFleetDataset → windows → build_flexible_cnn` strip. Badge each card by evidence status. Use cyan for implemented adapter, green for locally verified C-MAPSS counts, amber for external validation still required.
- **On-slide copy:**
  - **Combined C-MAPSS — VERIFIED:** 709 training engines; collision-free subset/split IDs
  - **Scania Component X — ADAPTER READY:** real commercial heavy-truck schema; 105 operational features; repair/censoring semantics preserved
  - **N-CMAPSS — ADAPTER READY:** component-health truth isolated from prognostic inputs
- **Chart data:** none; this is a capability matrix.
- **Speaker notes:** The combined C-MAPSS test fleet has 707 official held-out engines. Scania Component X and N-CMAPSS were not downloaded or benchmarked in this work; their tests use synthetic schema fixtures. `build_flexible_cnn` matches `CmapssRegressor` at `(17, 30)` with 22,273 parameters and supports other channel/window dimensions, but all clients in one federation still require the same schema and feature order.
- **Source footer:** `vehicle_datasets.py`; `docs/DATASET_EVALUATION.md`; `tests/test_vehicle_datasets.py`.

### Slide 8 — Safety objective plus local explanations — 1:30

- **Layout archetype:** split.
- **Visual and layout:** left half is a paired dumbbell chart of MSE versus asymmetric FedAvg late rate for FD001 and FD002. Right half is a local diagnostics card with four subsystem shares—Fan, LPC, HPC, Turbine—and ranked sensor/drift lists. Use cyan for MSE, amber for asymmetric loss, and keep subsystem colors consistent with the dashboard.
- **On-slide copy:**
  - Late rate: **63.8% → 34.2%** on FD001
  - Late rate: **48.88% → 25.25%** on FD002
  - Attribution is an explanation, not a component diagnosis
- **Exact chart data:** dumbbell endpoints are `63.8` and `34.2` for FD001, `48.88030888030888` and `25.250965250965248` for FD002. Supporting RMSE values are FD001 `16.9834903717041` versus `16.25392303466797`; FD002 `22.334092330932616` versus `26.07202911376953`.
- **Speaker notes:** These values come from the separate five-seed `AsymmetricRULLoss` study, not the vehicle-swarm artifact. On FD002, lower late rate comes with higher RMSE. The diagnostics formula is absolute baseline difference times gradient, aggregated into subsystem shares. C-MAPSS has no per-unit component-fault labels. The instrumented diagnostics validation recorded zero socket calls and zero network bytes in its audited path; that is not a whole-system privacy guarantee.
- **Source footer:** `results/asymmetric_loss_comparison.json`; `results/diagnostics_validation.json`; `diagnostics.py`.

### Slide 9 — Hardening exists; pilot integration remains explicit — 1:15

- **Layout archetype:** architecture flow.
- **Visual and layout:** a four-layer stack: transport, update protection, data quality/normalization, runtime/verification. Put green checks on implemented components and amber connectors on integration or operational validation still required.
- **On-slide copy:**
  - Configured TLS/mTLS; explicit local-only plaintext option
  - Optional whole-delta clipping/noise; no formal privacy accountant or secure aggregation
  - Federated statistics and sanitation are tested; operational paths still need integration
- **Chart data:** none.
- **Speaker notes:** `fed_norm.py` is exercised by the vehicle-swarm benchmark, but `client.py`, `experiments/reproduce.py`, and `EdgeInferenceRunner` retain legacy scaling paths. `edge_runtime.py` supports JSONL, CSV replay, and an optional MQTT class; the current CLI does not expose MQTT. Target-gateway inference, diagnostics, thermals, energy, and encrypted wire traffic remain unprofiled.
- **Source footer:** `server.py`; `client.py`; `fed_norm.py`; `edge_runtime.py`; `PROJECT_SPEC.md`.

### Slide 10 — Live demo: asset → cluster → decision — 3:30

- **Layout archetype:** live demo frame.
- **Visual and layout:** show a clean browser frame with the real dashboard, a cyan focus ring around the current control, and a three-step rail: **Tab 1 Asset**, **Tab 2 Cluster**, **Tab 3 Safety**. Add a persistent **PoC Replay** badge so replay is never mistaken for a plant connection.
- **On-slide copy:**
  - Scrub a held-out asset and inspect RUL + local diagnostics
  - Replay physical and simulated federated rounds
  - Compare late-risk trade-offs and the smaller-client case
- **Chart data:** use the dashboard's committed `dashboard/demo_data.json` at runtime; do not redraw or invent series. It contains eight held-out engine replays, five physical FD002 runs, and one 15-round simulation series.
- **Speaker notes:** Follow the choreography below. Call out that true RUL is retrospective ground truth unavailable for a live operating asset. The replay is precomputed; it is not a training run or live plant stream.
- **Source footer:** `dashboard/demo_data.json`; `dashboard/index.html`; `dashboard/server.py`.

### Slide 11 — Evidence spans simulation, physical training, and local audits — 1:40

- **Layout archetype:** chart + takeaway.
- **Visual and layout:** left: grouped bar chart of five-seed IID RMSE by dataset and method. Right: two evidence cards, one physical and one smaller-client simulation. Use gray = centralized, cyan = local-only, green = FedAvg.
- **On-slide copy:**
  - Four controlled C-MAPSS subsets; paired five-seed simulation
  - Physical FD002 training: two Raspberry Pi 4 clients + Ubuntu hub
  - Smaller simulated client: federation improved RMSE in one imbalance study
- **Exact chart data:** plot the exact means below; show two-decimal labels.

| Dataset | Centralized | Local only | FedAvg |
|---|---:|---:|---:|
| FD001 | 14.849266052246094 | 19.80021584971879 | 19.269803619384767 |
| FD002 | 22.123116302490235 | 23.35265423905708 | 23.325714111328125 |
| FD003 | 18.00357780456543 | 21.976588462476734 | 21.59537582397461 |
| FD004 | 23.90577964782715 | 28.429855991860155 | 27.074039459228516 |

  Physical callout: FD002 five-round FedAvg RMSE **22.27 ± 0.44** across five seeds. Smaller-client simulation callout: FD002 imbalance client 0 **38.13 isolated → 22.21 federated RMSE**.
- **Speaker notes:** The physical and simulation protocols are separate; do not compare them as if controlled pairs. Five-seed paired tests are exploratory, with a minimum two-sided exact p-value of 0.0625. The imbalance result is potential technical value, not money saved or a guaranteed customer outcome.
- **Source footer:** `results/reproducible/summary.csv`; `dashboard/demo_data.json`; `results/EXPERIMENTS.md`.

### Slide 12 — What is proven, and what a pilot must prove — 1:30

- **Layout archetype:** split.
- **Visual and layout:** left column **PROVEN IN THIS REPOSITORY** with simulation/physical/audit badges; right column **PILOT EVIDENCE REQUIRED** with an amber checklist. Keep this slide candid and uncluttered.
- **On-slide copy:**
  - Proven: vehicle-disjoint simulation, compact model, local diagnostics, physical training traces, adapter contracts
  - Open: real-fleet predictive validity, gateway resource envelope, end-to-end security, alert calibration, operations and rollback
  - Advisory shadow mode; a human owns every maintenance decision
- **Chart data:** none.
- **Speaker notes:** C-MAPSS is simulated turbofan data. Scania and N-CMAPSS adapters are implemented but not externally benchmarked. No formal DP guarantee, secure aggregation, automatic shutdown, measured plant ROI, or 10/20-site physical deployment exists. Use the pilot to establish tag mapping, labels, data quality, security, resource budgets, alert thresholds, monitoring, and stop criteria.
- **Source footer:** `docs/DATASET_EVALUATION.md`; `docs/PILOT_RUNBOOK.md`; `PROJECT_SPEC.md`.

### Slide 13 — Scope the pilot around decisions, not technology theater — 0:55

- **Layout archetype:** hero.
- **Visual and layout:** five owner chips—OT, Maintenance, IT, Security, ML—surround one central decision card: **Approve a segregated, advisory shadow pilot?** Below it, show four gates: assets/data, acceptance metrics, security/resources, rollback/ownership.
- **On-slide copy:**
  - Nominate owners and representative assets
  - Agree success, stop, security, and resource thresholds before connection
  - Leave with a dated discovery workshop and evidence plan
- **Chart data:** none.
- **Speaker notes:** Ask for a scoped discovery commitment, not production approval. End on the maintenance decision and the evidence plan.
- **Source footer:** `docs/PILOT_RUNBOOK.md`; `PROJECT_SPEC.md`.

## Live demo choreography

### Before the meeting

1. From the repository root, confirm the intended revision and run `python3 scripts/preflight_check.py --skip-unittest`. Treat this as the fast artifact check; it does not replace the full unit suite. For C-MAPSS-gated checks, configure the archive and follow `docs/PILOT_RUNBOOK.md`.
2. Run exactly `python3 dashboard/server.py`. Keep the terminal open. If the browser does not open, navigate to `http://127.0.0.1:8765/`. Check `http://127.0.0.1:8765/api/health` before presenting.
3. The dashboard starts in Italian. Click **EN**. Select **PoC Replay** and **Cycles only**. Do not expose this unauthenticated demo server to a plant network.
4. If a mapped live adapter is available, validate `POST /api/edge_update` and `GET /api/live` locally before the meeting. Otherwise use replay throughout.

### Tab 1 — Live Asset RUL

1. Point to **PoC Replay**. Briefly select **Live Edge Stream** to show the empty waiting state, then return to replay. Mode selection is not evidence of a connected gateway.
2. Select `FD001-U12` for a healthy replay, then `FD001-U81` for a critical replay. `FD002-U119` and `FD002-U65` are the FD002 alternatives.
3. Drag the cycle slider backward, click **Play**, then pause on a useful point. Compare true RUL, FedAvg asymmetric, FedAvg MSE, and local MSE. A prediction below truth is early; above truth is late. The asymmetric model is not always early.
4. Scroll to the diagnostics card. Show Fan/LPC/HPC/Turbine shares, top contributors, top drifting sensors, and the share trend. Say “model attribution,” never “component fault probability.”
5. Click **IT**, then **EN**. Change the cadence from **Cycles only** to **1 cycle = 2.0 h**, then switch back. The hour value is a user-selected illustration, not a C-MAPSS measurement.

### Tab 2 — Federated Cluster

1. Open **2. Federated Cluster (2x RPi4 + Hub)**. Choose a physical FD002 run and scrub or play its rounds. The stored series include the initial point plus five training rounds.
2. Explain the calculated tensor payload: 89.1 decimal KB one way and 178.2 decimal KB download plus upload per client per round, excluding framing, metadata, protocol, and TLS overhead.
3. Select the 15-round simulation. Label it **SIMULATION**, not Pi evidence. Do not transfer its convergence or timing to the physical cluster.

### Tab 3 — Safety & Business Impact

1. Open **3. Safety & Business Impact**. Compare FD001 and FD002 FedAvg MSE versus asymmetric rows: late rate, RMSE, and NASA score. Late rate is an offline prediction metric, not field alert effectiveness.
2. Highlight FD002 imbalance client 0: 38.13 isolated versus 22.21 federated RMSE. Call it a controlled smaller-client example, not ROI.
3. Close on the engineering note and transition to Slide 12: what the pilot must measure.

### Optional live stream mapping and fallback

`edge_runtime.py` provides `WebhookDashboardSink`, but its CLI creates `JsonLinesAlertSink` and has no dashboard-URL option. The runtime emits only on debounced severity changes and requires `primary_subsystem`, `summary_text`, and `privacy`; `dashboard/server.py` instead permits `summary` and rejects unsupported top-level fields. Direct CLI-to-dashboard push is therefore not a working built-in command.

An integration adapter must map `summary_text` to `summary`, remove unsupported fields, and send only `asset_id`, `cycle`, `predicted_rul`, `severity`, `subsystem_shares_pct`, `top_contributors`, `top_drifting`, and `summary` to `http://127.0.0.1:8765/api/edge_update`. The server keeps only the latest accepted update per asset in memory; `GET /api/live` exposes that snapshot. The API checks are not authentication or a comprehensive privacy system.

If the adapter is not rehearsed, use **PoC Replay** throughout. The committed `dashboard/demo_data.json` works without network access, a C-MAPSS archive, or a new training run. Keep prepared screenshots as the final fallback and label them **REPLAY**.

## Evidence ledger and zero-hallucination rules

| Source | Permitted use in the deck | Boundary to preserve |
|---|---|---|
| `results/vehicle_swarm_benchmark.json` | Exact swarm configuration, partitions, participation, heterogeneity, per-round and aggregate metrics, personalization, normalization audit | In-process C-MAPSS simulation; not physical sites, real vehicles, transport testing, or convergence proof |
| `docs/vehicle_swarm.md` | Rounded tables, protocol explanation, interpretation, caveats | Do not promote drift clusters to component labels |
| `vehicle_datasets.py` | Implemented classes, schemas, window semantics, flexible model behavior | Opt-in adapters; no automatic integration or download |
| `docs/DATASET_EVALUATION.md` | Dataset provenance, counts, labels, availability, license notes, memory and target caveats | Scania and N-CMAPSS were not downloaded or benchmarked in this work |
| `results/asymmetric_loss_comparison.json` | Separate MSE/asymmetric RMSE, NASA, and late-rate comparison | Separate seed set/protocol from the swarm and paired suite |
| `results/diagnostics_validation.json` | Attribution/drift protocol and instrumented zero-network audit | No supervised component diagnosis or whole-system privacy proof |
| `results/reproducible/summary.csv` | Paired five-seed simulation means/std and exact p-values | Protocol-specific; exploratory seed count |
| `dashboard/demo_data.json` | Replay assets, cluster series, dashboard benchmark highlights, model payload facts | Precomputed replay; some sensor traces are bundled; not a plant stream |
| `tests/test_*.py` | Exact test inventory and verified behaviors | Static method count is not a claim that all gated tests ran in this session |

Never invent or imply revenue, avoided downtime, plant alert precision, road-vehicle accuracy, real-customer deployment, formal differential privacy, secure aggregation, encrypted historical runs, Pi inference/diagnostics performance, or statistical significance. If the source files disagree, stop and surface the conflict in speaker notes instead of resolving it by guesswork.

## Ready-to-Paste Claude Design Master Prompt

```text
Create a complete, polished 13-slide, 16:9 pitch deck titled “Privacy-Preserving Predictive Maintenance at the Industrial Edge” for plant/OT managers, maintenance leaders, security/IT engineers, and ML engineers. The goal is to secure agreement to scope a segregated, advisory-only Phase-1 pilot—not production approval.

Use the attached docs/PRESENTATION_ROADMAP.md as the authoritative production brief and PROJECT_SPEC.md as the technical reference. Follow the slide order, exact chart series, evidence labels, speaker-note boundaries, live-demo choreography, and source footers in the roadmap. If either attachment is unavailable, use the slide plan and facts embedded in this prompt. Do not browse for replacement numbers and do not fill gaps with plausible claims.

DESIGN SYSTEM
- 16:9 widescreen; approximately 20 minutes; dark industrial-tech rather than futuristic.
- Background #0b111b; primary cyan #38bdf8; positive green #22c55e; caution amber #f59e0b; personalization/risk pink #f472b6; white primary text and cool-gray caveats.
- Clean geometric sans typography; monospace only for code symbols, paths, methods, and API routes. Short assertive titles, one dominant claim per slide, no more than three short bullets, detailed speaker notes.
- Badge evidence as SIMULATION, PHYSICAL, LOCAL AUDIT, ADAPTER READY, or PROPOSED PILOT. Never merge evidence classes.
- Prefer vector flows, restrained fleet/gateway silhouettes, dashboard frames, and direct-labeled 2D charts. No decorative 3D and no generated image that implies a real customer or live plant deployment.
- Add a small source-path footer to every evidence slide. Use only repository paths named here or in the attached files.

POSITIONING AND NON-NEGOTIABLE BOUNDARIES
FedPdM-Edge is a research proof of concept and pilot blueprint. C-MAPSS is simulated turbofan data; RUL is in cycles, not automatically hours. It is not a production safety controller, road-vehicle validation, financial-ROI study, or automatic shutdown system. Raw training windows are not serialized as model updates, but model updates and aggregate statistics may leak information. Configured TLS/mTLS and optional whole-client-delta clipping/Gaussian noise exist; there is no formal epsilon/delta accountant and no secure aggregation.

SLIDE PLAN
1. HERO — “Learn across fleets. Keep telemetry local.” Show customer gateways connected to an OEM coordinator while telemetry remains local. Subtitle: “Vehicle-disjoint learning, local diagnostics, pilot-ready evidence boundaries.”
2. SPLIT — “The maintenance gap is both technical and organizational.” Show heterogeneous isolated fleets and an early/true/late RUL timeline. State that late estimates can postpone action, early estimates can trigger unnecessary work, and no company-specific costs were measured.
3. ARCHITECTURE FLOW — “One architecture, two privacy boundaries.” Two lanes: offline federated training and local edge inference. Show 22,273 float32 parameters, 89,092 bytes one way, and 178,184 bytes download plus upload per client per round, excluding protocol/TLS overhead.
4. THREE KPI COLUMNS — “The new benchmark models customer fleets, not row shards.” Show 10/20 strictly vehicle-disjoint clients; 50%/30% participation, exactly 5 of 10 or 6 of 20 each round; and 0 raw sensor rows shared to the normalization server. Explain IID, lifespan/quantity skew, and regime/drift skew. Label this SIMULATION.
5. CHART + TAKEAWAY — “Non-IID stress is measured, not asserted.” For FD004 K=20 fraction_fit=0.3, draw three grouped bar panels: count CV, lifespan pairwise JS, drift-cluster pairwise JS. Raw series by partition:
   iid_swarm = [0.09934287033540237, 0.1751300089234043, 0.041063359435593604], combined score 0.399106069182159.
   lifespan_quantity_skew = [1.1934111966410597, 0.7863888999119484, 0.21478617939836436], combined score 2.5007266183637693.
   regime_fault_skew = [1.0584645082839426, 0.39538933754460825, 0.3718889501356717], combined score 2.00643451732452.
   Display metric labels to three decimals. Say larger means more inter-client skew; do not call the drift cluster a component label.
6. SMALL-MULTIPLE BARS — “Optimize for the maintenance decision, then personalize carefully.” FD004 K=20 regime_fault_skew aggregate means; panels are RMSE cycles, late-prediction rate, NASA per test vehicle:
   local_only = [46.80298195680963, 0.5338709677419355, 924.3840377945577]
   fedavg_mse = [45.75365206396756, 0.4588709677419355, 362.75603482634654]
   fedavg_asym = [49.676721538980495, 0.3419354838709677, 110.27362077219509]
   fedavg_asym_personalized = [48.46673124683082, 0.34838709677419355, 97.33219311186204]
   Label RMSE/NASA to two decimals and late rate as one-decimal percent. Note that FedProx(mu=0.05) is effectively unchanged at displayed precision. Personalization improves matched-client mean validation RMSE in eight of nine configurations but is not uniformly fair.
7. THREE DATASET COLUMNS — “The data path now extends beyond one benchmark schema.” Combined C-MAPSS: 709 training engines, locally verified. Scania Component X: real commercial heavy-truck schema, 105 operational features, repair/censoring preserved, adapter ready but not benchmarked. N-CMAPSS: component-health truth isolated from prognostic inputs, adapter ready but not benchmarked. Connect all through VehicleFleetDataset → windows → build_flexible_cnn. State that one federation still requires the same schema and feature order.
8. SPLIT — “Safety objective plus local explanations.” Left dumbbell chart: FedAvg MSE vs asymmetric late rate, FD001 63.8 → 34.2 and FD002 48.88030888030888 → 25.250965250965248. Supporting RMSE: FD001 16.9834903717041 vs 16.25392303466797; FD002 22.334092330932616 vs 26.07202911376953. Right: Fan/LPC/HPC/Turbine attribution card. Explain that attribution is not component diagnosis. The diagnostics audit recorded zero socket calls and zero network bytes only in the instrumented path.
9. LAYERED FLOW — “Hardening exists; pilot integration remains explicit.” Show configured TLS/mTLS; optional clipping/noise; federated statistics/sanitation; streaming runtime and checks. Mark no privacy accountant/SecAgg, legacy-scale integration, and gateway profiling as open.
10. LIVE DEMO FRAME — “Live demo: asset → cluster → decision.” Show the real dashboard frame and a three-step rail for Tab 1 Asset, Tab 2 Cluster, Tab 3 Safety. Persistent PoC Replay badge. The committed replay contains eight held-out engine replays, five physical FD002 runs, and one 15-round simulation series. It is precomputed, not a live plant or new training run.
11. CHART + EVIDENCE CARDS — “Evidence spans simulation, physical training, and local audits.” Grouped IID RMSE means [centralized, local_only, FedAvg]:
   FD001 [14.849266052246094, 19.80021584971879, 19.269803619384767]
   FD002 [22.123116302490235, 23.35265423905708, 23.325714111328125]
   FD003 [18.00357780456543, 21.976588462476734, 21.59537582397461]
   FD004 [23.90577964782715, 28.429855991860155, 27.074039459228516]
   Show two-decimal labels. Separate PHYSICAL card: FD002 five-round FedAvg on two Raspberry Pi 4 clients and Ubuntu hub, RMSE 22.27 ± 0.44 across five seeds. Separate SIMULATION card: imbalance client 0, 38.13 isolated → 22.21 federated RMSE. Do not imply ROI or compare protocols as controlled pairs.
12. SPLIT — “What is proven, and what a pilot must prove.” Proven: vehicle-disjoint simulation, compact model, local diagnostics, physical training traces, adapter contracts. Open: real-fleet predictive validity, target-gateway resources, end-to-end security, alert calibration, monitoring, rollback. Human-reviewed advisory shadow mode only.
13. HERO CTA — “Scope the pilot around decisions, not technology theater.” Show OT, Maintenance, IT, Security, and ML owners around the question “Approve a segregated, advisory shadow pilot?” Ask for representative assets, acceptance/stop/security/resource thresholds, ownership, rollback, and a dated discovery workshop.

LIVE DEMO NOTES
- Run: python3 dashboard/server.py
- Open: http://127.0.0.1:8765/
- Click EN; use PoC Replay and Cycles only.
- Tab 1: FD001-U12 then FD001-U81; scrub/play; show RUL curves and local diagnostics; toggle IT/EN; briefly show “1 cycle = 2.0 h” then return to cycles and call it illustrative.
- Tab 2: replay a physical five-round FD002 run, then the 15-round simulation; never call the simulation physical.
- Tab 3: compare asymmetric-loss trade-offs, then the FD002 imbalance client 0 result.
- The edge runtime and dashboard payload schemas do not directly match. A mapped adapter must POST only asset_id, cycle, predicted_rul, severity, subsystem_shares_pct, top_contributors, top_drifting, and summary to /api/edge_update; GET /api/live exposes the latest in-memory snapshot. If this has not been rehearsed, remain in replay.

OUTPUT REQUIREMENTS
Deliver all 13 finished slides with final on-slide copy, composed layouts, exact charts, evidence badges, source footers, and detailed speaker notes. Keep caveats readable and specific. Do not invent revenue, avoided downtime, plant alert accuracy, road-vehicle accuracy, external-dataset benchmark results, networked 10/20-site evidence, Pi inference performance, security guarantees, or statistical significance. If a requested claim is unsupported or conflicts with an attached source, omit it and add a speaker-note warning rather than guessing.
```
