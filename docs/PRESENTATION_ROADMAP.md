# FedPdM-Edge presentation roadmap

**Positioning:** a validated research proof of concept (PoC) and a blueprint for a Phase-1 industrial pilot. It is not a production maintenance or safety system. The evidence below is tied to the repository at `origin/master` commit `9ce6367`; refresh and recheck before reuse. RUL means remaining useful life, measured in C-MAPSS cycles, not automatically in hours.

## Goal and audience

Secure agreement to scope a segregated, advisory-only pilot: identify assets and telemetry owners, map tags, agree on success and stop criteria, and assign OT, IT, security, maintenance and ML owners. Lead with the maintenance decision, then show the local-to-federated flow and the limits of the evidence.

| Audience | What matters | Message |
|---|---|---|
| Plant/OT managers and maintenance | Lead time, avoidable downtime, nuisance inspections, safety and integration with workflows | Show earlier warnings and local explanations; a human still decides when to inspect or shut down. No savings have been measured. |
| Security/IT engineers | Data boundaries, identity, network access, update leakage, operating and rollback controls | Show what leaves the gateway, the optional protections, and the evidence still needed for a plant connection. |
| ML engineers | Labels, generalization, drift, baseline comparisons, deployment cost | Show the evaluation protocols, trade-offs, limitations and a plan for plant-specific validation. |

## Slide-by-slide roadmap — 11 slides, approximately 18 minutes

Each visual should label simulated, physical, historical and proposed evidence separately. Numbers are illustrative benchmark results, not industrial performance promises.

### 1. Title & hook — 0:45

- **Objective:** Establish the opportunity: *Privacy-Preserving Predictive Maintenance at the Industrial Edge*.
- **Visual:** Two plant gateways connected to a coordinator, with local telemetry remaining at each site.
- **Fact:** The PoC predicts C-MAPSS engine RUL in cycles using a 30-cycle input window (`common.py`).
- **Speaker:** “Can plants learn from a shared model while keeping their sensor histories local? This PoC tests that idea and gives us a path to a controlled pilot.”

### 2. The industrial problem — 1:00

- **Objective:** Connect fragmented data and maintenance timing to the audience's decisions.
- **Visual:** Isolated sites; timeline with early inspection, failure and a late prediction; alert with a sensor explanation.
- **Fact:** An overestimated RUL can postpone maintenance beyond failure; an underestimate can trigger unnecessary early work (`common.py`, `AsymmetricRULLoss`). No company-specific failure costs were measured.
- **Speaker:** “Plants may be unable to pool raw history. A single score also conceals which signals drove it. We need useful lead time and a reviewable explanation.”

### 3. System architecture — 1:30

- **Objective:** Trace the data boundary and transport.
- **Visual:** Edge gateways → local 30-cycle windowing and diagnostics → Flower gRPC model updates → hub → updated model. Mark mTLS as configured deployment option.
- **Facts:** `server.py` and `client.py` implement TLS/mTLS configuration; explicit `--insecure` permits plaintext for local development. A 22,273-float32 model contains 89,092 bytes; one download plus one upload is 178,184 bytes (approximately 89.1 and 178.2 decimal KB) **per client per round**, excluding protocol/TLS overhead (`results/edge/local-smoke-macos.json`). Raw training windows are not serialized as model updates (`results/EXPERIMENTS.md`).
- **Speaker:** “The design shares model parameters, not raw training windows. Zero raw telemetry in that update path is a data-flow claim, not a packet capture or a guarantee against model-update leakage. Historical physical results did not validate this mTLS configuration.”

### 4. Core ML stack — 1:15

- **Objective:** Explain the compact temporal model and evidence behind its choice.
- **Visual:** 30 × 17 input → 1D convolution layers → RUL; compact model versus tested CNN-LSTM.
- **Facts:** PyTorch `CmapssRegressor` has 22,273 parameters (`common.py`). In the paired centralized regime experiment, FD002 CNN-LSTM RMSE was 43.14 versus CNN 21.46; FD004 was 43.00 versus 23.41 (five-seed means, `results/reproducible/summary.csv`). These are specific training settings, not a universal architecture ranking. Edge efficiency versus CNN-LSTM was not measured on a Pi.
- **Speaker:** “The CNN is a practical compact choice and outperformed the tested CNN-LSTM under this protocol. We have not established an edge-budget performance advantage.”

### 5. Safety-first objective — 1:30

- **Objective:** Make the cost asymmetry concrete.
- **Visual:** Two error arrows around true RUL, plus a small table with both late rate and RMSE.
- **Facts:** In the separate five-seed MSE vs `AsymmetricRULLoss` benchmark, FD001 FedAvg late-prediction rate fell 63.8% → 34.2%, RMSE 16.98 → 16.25; FD002 FedAvg fell 48.88% → 25.25%, RMSE rose 22.33 → 26.07. FD002 FedAvg NASA score standard deviation fell 2,962.80 → 429.82 (population SD). All six local/FedAvg/central late rates span 40.8–63.8% → 20.5–34.2%; these are prediction rates, not observed plant alerts. FD002 centralized NASA mean increased. Source: `results/asymmetric_loss_comparison.json`.
- **Speaker:** “We penalize overestimating life more heavily: defaults 2.5 versus 0.5. This shifts the trade-off, not a guarantee that every warning is early. Calibrate it with plant consequences.”

### 6. Local edge explainability — 1:15

- **Objective:** Show what an engineer can inspect alongside RUL.
- **Visual:** `Fan`, `LPC`, `HPC`, `Turbine` attribution bar plus ranked sensors and drift trend.
- **Facts:** `diagnostics.py` implements `baseline_x_gradient` and regime-aware `HealthyBaseline`; sensor attribution shares are normalized across the four subsystems. `experiments/validate_diagnostics.py`'s `SocketAudit` recorded 0 calls and 0 raw sensor bytes during the instrumented diagnostics path (`results/diagnostics_validation.json`). C-MAPSS lacks per-unit component fault labels; these are model explanations, not verified component diagnoses.
- **Speaker:** “The bar allocates attribution, not failure probabilities. Drift and model attribution are complementary leads for an engineer to investigate.”

### 7. Hardening & technology stack — 1:30

- **Objective:** Separate available building blocks from production readiness.
- **Visual:** Four layers: transport, update protection, normalization, runtime and checks.
- **Facts:** `client.py` optionally clips the concatenated client update L2 norm and adds Gaussian noise; defaults do not enable clipping/noise and there is no formal epsilon accountant or secure aggregation. `fed_norm.py` merges local statistics with Chan/Welford and offers sanitation/imputation; it is a separate utility, not proven integrated into the existing Flower training path. `edge_runtime.py` provides JSONL and CSV sources, optional MQTT adapter, rolling windows and debounced severity changes; its CLI supports JSONL or C-MAPSS replay. CI configuration and `scripts/preflight_check.py` provide checks.
- **Speaker:** “These are prototype hardening components. Certificates, integration, privacy accounting and operational validation remain pilot work.”

### 8. Live demo — 4:00

- **Objective:** Make the asset-to-cluster-to-business story tangible.
- **Visual:** The running dashboard, with replay mode visibly selected.
- **Facts:** `dashboard/index.html` and `dashboard/demo_data.json` contain eight held-out engine replays, five historical physical FD002 runs, and one 15-round simulation. Replay uses precomputed results; it is not a live plant or a new training run.
- **Speaker:** Follow the click-by-click script below; call out true RUL as retrospective ground truth unavailable for a real operating asset.

### 9. Quantitative evidence & physical cluster — 1:45

- **Objective:** Show the breadth and provenance of the evidence.
- **Visual:** Table with IID centralized/local/FedAvg RMSE; separate callouts for the physical and imbalance studies.
- **Facts:** Corrected five-seed IID means: FD001 14.85 / 19.80 / 19.27; FD002 22.12 / 23.35 / 23.33; FD003 18.00 / 21.98 / 21.60; FD004 23.91 / 28.43 / 27.07 cycles, respectively. Separate historical FD002 five-round FedAvg on two Pi 4 clients and Ubuntu hub: 22.27 ± 0.44 RMSE, five seeds. FD002 simulated engine-imbalance client 0: isolated 38.13 → FedAvg 22.21 RMSE, five-seed mean. Sources: `results/EXPERIMENTS.md`, `results/reproducible/summary.csv`, `results/fd002/results_physical_seed*.json`.
- **Speaker:** “Federation helped the small simulated client here, but this is neither monetary ROI nor proof for a real SME. Comparisons are protocol-specific; five-seed paired tests are exploratory, with minimum two-sided p = 0.0625.”

### 10. Limitations & Phase-1 roadmap — 1:30

- **Objective:** Earn trust and identify the work between PoC and pilot.
- **Visual:** C-MAPSS → lab mapping → segregated shadow pilot → acceptance decision, with human approval at every stage.
- **Facts:** C-MAPSS is simulated turbofan data, not the company's plant telemetry. Pi physical training traces exist; target-gateway inference, diagnostics, thermals, encrypted wire traffic and energy have not been profiled. No formal DP epsilon accounting or SecAgg exists. `docs/PILOT_RUNBOOK.md` specifies profiling and go/no-go evidence; its statements that TLS and federated normalization are absent predate the current code and should not be read as current implementation status.
- **Speaker:** “Use the runbook for pilot controls and profiling; verify the newer security and normalization code in the actual deployment. No automatic shutdowns.”

### 11. Call to action — 1:00

- **Objective:** Obtain a concrete discovery and pilot-scoping commitment.
- **Visual:** Owners and decisions: assets, data, success measures, security, resources.
- **Fact:** The runbook proposes at least two representative assets, advisory alerts and a fixed observation period with planned inspections or known maintenance events.
- **Speaker:** “Nominate OT, maintenance, IT, security and ML owners; select assets and an isolated data route; define lead-time, nuisance-alert, resource and security acceptance thresholds before shadow operation.”

## Live demo script

### Before the meeting

1. In a clean checkout, update from `origin/master` (for example `git switch master` then `git pull --ff-only origin master`); do not overwrite local work. Confirm the intended revision and review any changes to these claims. Install the repository's required dependencies for preflight. Run `python3 scripts/preflight_check.py --skip-unittest`; for the full pilot check use `python3 scripts/preflight_check.py --with-cmapss` with the dataset configured as described in `docs/PILOT_RUNBOOK.md`. Review failures; skipping tests is not full validation.
2. From the repository root run exactly `python3 dashboard/server.py`. Open `http://127.0.0.1:8765/` if the browser does not open; `/api/health` reports status and whether `demo_data.json` exists. Keep the terminal running. The dashboard server uses Python's standard library; replay does not need the dataset or model training.
3. The default language is Italian. Click **EN** for English. Select **PoC Replay** in Source and **Cycles only** in the cadence selector to avoid suggesting that simulated cycles are real hours. Rehearse at the intended screen size.
4. If planning live mode, prepare a local stream, trained model, scales, healthy baseline, dependencies and an adapter as described below. Check the dashboard's `/api/live` response before the meeting. Do not expose the unauthenticated demo server to a plant network.

### Tab 1 — Live Asset RUL

1. Point to the **PoC Replay** badge and source selector. Select **Live Edge Stream** briefly: with no updates it waits for the first update; switch back to **PoC Replay**. Explain that mode is selected by the presenter, not evidence of a connected gateway.
2. In **Asset (Held-out Test)** select `FD001-U12` (healthy replay), then `FD001-U81` (critical replay); alternatively use `FD002-U119` and `FD002-U65`. The stage label is based on the selected test engine's final true RUL, whereas status is based on the current safe prediction; they need not match.
3. Drag `cycleSlider` backwards, then click **Play**; click **Pause** at an interesting point. If at the end, Play restarts from the beginning. Point to true RUL and the FedAvg asymmetric, FedAvg MSE and local MSE curves and their cards. A prediction below true RUL is early; above it is late. The safe model is not always early. The dashboard's status margin refers to the asymmetric prediction; use the MSE card's comparison to discuss its late risk only when it actually shows one.
4. Scroll to `#diagCard`: as the slider moves, the four subsystem shares, top-three contributing sensors, top-three drifting sensors and share trend update together. Explain that shares describe attribution and that diagnostics are not component fault labels. The replay's displayed scaled sensor traces are included in the bundled demo data; do not describe the whole dashboard dataset as containing no sensor values.
5. Click **IT**, then **EN** to demonstrate language switching. Change the cadence selector `hoursSelect` from **Cycles only** to e.g. **1 cycle = 2.0 h** and back. The hour figure is a configurable illustrative conversion, not a C-MAPSS measurement or a plant-calibrated schedule.

### Tab 2 — Federated Cluster

1. Click **2. Federated Cluster (2x RPi4 + Hub)**. Use **Federated Run** to select a physical five-round FD002 seed; drag `roundSlider` or click **Play Rounds**. Point to the two Pi nodes, hub and RMSE/NASA curves.
2. Explain 89.1 KB one-way and 178.2 KB download plus upload per client per round as calculated model-tensor payload, excluding framing, TLS, metadata and other traffic. The raw training windows are not the model update; model updates can leak information.
3. Select the **15-round simulation** in the run selector and replay it. The 15-round result is a local simulation, not a physical Pi run; do not directly attribute its convergence or timing to the Pi cluster. In the historical FD002 comparison, physical five-round RMSE was 22.27 ± 0.44; simulated 15-round RMSE was 19.27 ± 1.30 under that separate setup (`results/EXPERIMENTS.md`).

### Tab 3 — Safety & Business Impact

1. Click **3. Safety & Business Impact**. In the loss table compare FD001 and FD002 FedAvg MSE and asymmetric rows, including late rate, RMSE and NASA score. Late rate means overestimated RUL on the test set, not field alert effectiveness.
2. In the imbalance table identify FD002 client 0: 38.13 isolated → 22.21 federated RMSE. This is an engine-count-imbalanced simulation, an example of potential value for a smaller site, not measured financial ROI or a guaranteed outcome.
3. End with the engineering note and a transition to pilot requirements. Some dashboard copy predates the current security implementations; explain the actual configured protections and remaining validation instead of treating that copy as current deployment evidence.

### Optional live stream and fallback

`edge_runtime.py` provides `WebhookDashboardSink`, but its CLI creates a `JsonLinesAlertSink` and has no dashboard URL option. The runtime emits only on debounced severity changes; its payload uses `summary_text`, `primary_subsystem` and `privacy`, whereas `dashboard/server.py` permits `summary` and rejects unsupported top-level fields. Therefore direct CLI-to-API push is **not** a working built-in command. An integration must use the Python API or an external adapter to map `summary_text` to `summary`, retain only the API's allowed fields (`asset_id`, `cycle`, `predicted_rul`, `severity`, `subsystem_shares_pct`, `top_contributors`, `top_drifting`, `summary`), and POST JSON to `http://127.0.0.1:8765/api/edge_update`. The server retains the latest update per asset in memory and `GET /api/live` exposes it; select **Live Edge Stream** and the asset. Check the full data flow locally before presenting. The API's field checks are not a comprehensive privacy or authentication system. For runtime replay, `--replay-cmapss` needs the local archive; supplying no `--model` uses an untrained demonstration model. JSONL mode uses a scale-derived baseline, not a locally fitted healthy baseline.

If no stream is ready, present **PoC Replay** throughout. The server serves the committed `dashboard/demo_data.json`; `index.html` also embeds fallback data. No network connection, C-MAPSS archive or live model is needed for the replay. If the server fails, use prepared screenshots and label them as replay.

## Technology spotlight and hard questions

| Technology | Plain-language explanation | Technical boundary |
|---|---|---|
| PyTorch 1D-CNN | Learns patterns across recent sensor history to estimate life remaining. | `common.CmapssRegressor` consumes 30 steps and 17 channels; validate mapping and performance for each asset. |
| Flower (`flwr`), gRPC, mTLS | Sites train locally and send model updates to a coordinator; certificates can authenticate the connection. | `server.py`/`client.py` implement configured TLS and mutual authentication; credential provisioning, rotation, compatibility and negative tests are pilot responsibilities. Historical traces do not prove encrypted deployment. |
| Client update clipping and noise | Limits the size of a site's update; optional noise can reduce information exposure. | `client.py` clips the concatenated delta and optionally adds Gaussian noise. No epsilon/delta accountant, SecAgg or formal end-to-end privacy claim; clipping alone is not DP. |
| Chan/Welford normalization | Combine site summaries to put sensor values on a consistent scale without collecting the original rows. | `fed_norm.py` merges counts, means and second moments; sanitation can impute missing/stuck values. Aggregates themselves can disclose information; the legacy model path uses min/max scales, so integration and training/inference parity must be established. |
| Gradient × healthy baseline | Highlights sensor differences that most influence this model's current output. | `diagnostics.py` uses absolute `(x − baseline) × gradient`, excludes settings from sensor attribution and normalizes subsystem shares; it is not causal proof. |
| K-Means regimes | Compares an asset with healthy behavior under a similar operating condition. | `fit_healthy_baseline` clusters settings for regime-aware drift; training data, regime coverage and transfer to new machines matter. |
| Standard-library HTTP and SVG | A portable, interactive dashboard with no frontend framework or server package. | `dashboard/server.py` serves local static assets and in-memory live summaries; `dashboard/index.html` draws SVG charts. It is a demo server, not a hardened multi-user service. |

**Security: Can the hub reconstruct data?** Raw windows are not sent as model updates, but updates and shared statistics may leak information. mTLS protects transport when configured; the hub still sees individual updates without SecAgg. Threat-model, test certificates and wire traffic, and make a formal privacy decision before connection (`server.py`, `client.py`, `docs/PILOT_RUNBOOK.md`).

**Scaling and drift: Will this work across many plants and new regimes?** Two clients were used in the physical study, and the regime experiment is a window-level operating-condition partition, not disjoint plant fleets (`results/EXPERIMENTS.md`). More sites, dropouts, network contention, distribution shift and unseen regimes require measurement. Monitor drift, data quality and site-level metrics; freeze and version normalization before any controlled change.

**Missing labels and sensors: How can we deploy?** C-MAPSS supplies unit RUL but no component labels; attribution is not component accuracy. Real assets need time-ordered, asset-separated maintenance/failure evidence and expert review. `fed_norm.py` offers imputation, but the runbook calls for explicit tag, unit and quality mapping and rejecting invalid inputs; never silently assume imputation is safe. Define gaps and label limitations in pilot acceptance.

**Pi 4 memory and latency?** Historical FD002 physical traces report training, not current inference profiling: `results/EXPERIMENTS.md` records seed-42 client training 12.8 and 17.3 seconds per round and 570–620 MB peak RSS. Diagnostics latency in the validation is from an Apple Silicon host, not Pi. Run `experiments/profile_edge.py` as in `docs/PILOT_RUNBOOK.md` on each target image and separately measure inference, diagnostics, thermal behavior and encrypted traffic.

**OPC-UA or MQTT?** `edge_runtime.py` has an optional `MqttSource` requiring `paho-mqtt`, plus JSONL and CSV source classes; MQTT is not exposed by the current CLI. There is no checked-in OPC-UA adapter in that runtime. Build and validate a read-only integration, tag mapping, units, sampling and failure handling; keep it out of the control path.

**Why not claim production ready?** The code implements important protections, but provisioning, threat review, model-update privacy, plant validation, gateway profiling, monitoring and rollback remain to be demonstrated together. The pilot should be advisory, isolated and human reviewed.

## Ready-to-paste slide-deck generation prompt

```text
Create an 11-slide, approximately 18-minute presentation for plant/OT managers, security/IT engineers and ML engineers, titled “Privacy-Preserving Predictive Maintenance at the Industrial Edge”. Produce clear slide titles, concise on-slide text, one visual suggestion and practical speaker notes per slide. Use a restrained industrial style, accessible diagrams, readable units and explicit evidence labels. Present FedPdM-Edge as a validated research PoC and Phase-1 advisory pilot blueprint, not a production system. Use the accompanying PROJECT_SPEC.md and the following verified facts; if a claim is not supported, omit it. Do not invent plant outcomes, financial ROI, statistical significance, latency, security guarantees or source citations.

Slides: 1 title and hook; 2 data silos, early versus late maintenance and explainability; 3 gateway → local windowing/diagnostics → Flower gRPC → hub, with configured mTLS; 4 compact PyTorch 1D-CNN; 5 asymmetric safety objective and accuracy trade-off; 6 local subsystem attribution and drift; 7 implemented hardening components and outstanding integration; 8 live dashboard walkthrough; 9 quantitative evidence; 10 honest limitations and Phase-1 pilot roadmap; 11 call to action.

Facts and boundaries: C-MAPSS is simulated turbofan data; RUL is in cycles. CmapssRegressor uses 30-cycle windows, 17 input channels and 22,273 float32 parameters. The tensor payload is 89,092 bytes one way and 178,184 bytes for one model download plus upload per client per round, excluding framing and TLS; raw training windows are not serialized as model updates, but updates may leak information. TLS/mTLS and optional client L2 delta clipping and Gaussian noise are implemented; historical physical experiments do not prove their use. There is no formal DP epsilon accountant or secure aggregation. Federated normalization and sanitation are separate utilities requiring integration and validation.

For five-seed IID simulations, centralized/local-only/FedAvg mean RMSE in cycles is FD001 14.85/19.80/19.27, FD002 22.12/23.35/23.33, FD003 18.00/21.98/21.60, FD004 23.91/28.43/27.07. In a separate historical FD002 five-seed physical five-round run with two Raspberry Pi 4 clients and an Ubuntu hub, FedAvg RMSE was 22.27 ± 0.44. A 15-round result is simulation, not a physical run. For FD002 simulated engine imbalance, client 0 isolated RMSE was 38.13 versus federated 22.21; this is not money saved. Five-seed paired tests are exploratory; minimum two-sided exact p is 0.0625.

In the separate five-seed AsymmetricRULLoss benchmark, FD001 FedAvg late-prediction rate was 63.8% with MSE and 34.2% with asymmetric loss, with RMSE 16.98 versus 16.25; FD002 was 48.88% versus 25.25%, with RMSE 22.33 versus 26.07. FD002 FedAvg NASA score population SD was 2,962.80 versus 429.82. These are test predictions, not plant alerts; FD002 centralized NASA mean worsened. Do not combine these results with a different asymmetric objective in the paired suite. The tested CNN-LSTM had higher error than the CNN in the specific centralized FD002/FD004 regime experiment; no comparative Pi resource advantage was measured.

Explain attribution as absolute baseline difference times gradient, aggregated into Fan/LPC/HPC/Turbine shares, not component failure probabilities or causal diagnoses. C-MAPSS has no per-unit component fault labels. SocketAudit measured zero calls and zero raw sensor bytes only in the instrumented diagnostics path. Dashboard replay uses precomputed held-out engines and historical/simulated results, with a user-selected mode and illustrative cycle-to-hour conversion; a live stream needs an integration adapter. Do not claim real-time plant validation.

For the pilot: name OT, maintenance, IT, security and ML owners; select at least two representative assets, map tags and units, establish labels and data quality, profile actual gateway resources, validate security and model-update privacy, calibrate alert thresholds, run in segregated human-reviewed shadow mode, and define acceptance and rollback criteria. Cite source file paths from PROJECT_SPEC.md or the facts above next to slide claims; flag any conflict or missing evidence instead of guessing.
```
