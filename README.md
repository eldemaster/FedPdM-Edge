# 🏭 FedPdM-Edge: Swarm Federated Predictive Maintenance for Industry 4.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-FL-green.svg)](https://flower.ai/)
[![Interreg Aurora](https://img.shields.io/badge/Project-ResilientEdge-003399.svg)](https://www.interregaurora.eu/)

**FedPdM-Edge** is a privacy-preserving, decentralized predictive maintenance framework developed for industrial manufacturing environments. Built under the **EU Interreg Aurora Project *ResilientEdge*** (Grant No. 20373282) by **Mid Sweden University** and the **University of Oulu**, it addresses the critical challenge of keeping raw machine telemetry strictly on-premise while collaboratively training robust Remaining Useful Life (RUL) regression models across distributed factory sites.

---

## 📌 Context & Problem Statement

In Industry 4.0 manufacturing plants, operational machinery telemetry (turbomachinery vibration, temperatures, pressure, actuator states) represents guarded intellectual property and operational secrets. Streaming high-frequency sensor readings to third-party cloud lakes introduces severe data exfiltration risks and regulatory barriers. 

However, isolated plants frequently encounter data scarcity: rare catastrophic mechanical failures happen infrequently at a single facility, causing locally-trained models to overfit or miss degradation anomalies. 

**FedPdM-Edge** solves this trade-off using **Swarm Intelligence and Federated Learning (Flower + PyTorch 1D-CNN)**:
- **Local Training Data:** The checked-in clients train on local telemetry and exchange model parameters. TLS, secure aggregation, and differential privacy are not configured; model updates may leak information and transport is not claimed to be encrypted.
- **Asymmetric Risk Optimization:** Evaluated against the exponential **NASA Asymmetric Scoring Function**, penalizing dangerous late failure predictions far more heavily than conservative early predictions.
- **Physical Edge Viability:** Tested and benchmarked across physical ARM single-board clusters (Raspberry Pi 4) and heterogeneous edge servers.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Tier1["Tier 1: Factory Edge Nodes (Physical Edge Clusters)"]
        subgraph PlantA["Factory Plant A (e.g. RPi 4)"]
            SensorsA["Machine Telemetry (17 Sensors)"] --> PrepA["Sliding Window (W=30)"]
            PrepA --> ModelA["Local 1D-CNN Regressor"]
        end
        subgraph PlantB["Factory Plant B (e.g. RPi 4)"]
            SensorsB["Machine Telemetry (17 Sensors)"] --> PrepB["Sliding Window (W=30)"]
            PrepB --> ModelB["Local 1D-CNN Regressor"]
        end
    end

    Bus["Flower gRPC transport<br/>[Model parameters: w_i <==> Global Model: w]"]

    ModelA <-->|"Local Weights w_1"| Bus
    ModelB <-->|"Local Weights w_2"| Bus

    subgraph Tier2["Tier 2: Swarm Coordinator Hub (Edge Gateway / Server)"]
        Bus <-->|"Global Model w"| Aggregator["Flower FedAvg Aggregator"]
        Aggregator --> Evaluator["NASA Asymmetric Risk Evaluator"]
        Evaluator --> Schedule["Predictive RUL Maintenance Schedule"]
    end
```

### Component Breakdown

| Component | Role | Function | Privacy Guarantee |
| :--- | :--- | :--- | :--- |
| **Plant Edge Node** | Raspberry Pi 4 / Edge Host | Runs local 1D-CNN training over sliding windows ($W=30$). | 100% On-Premise; raw data retained. |
| **Data Scaler** | Preprocessor | Min-Max sensor normalization across operating regimes. | Standardized local representations. |
| **1D-CNN Model** | PyTorch Regression Engine | 2 Conv1d layers + MaxPool1d + Fully Connected layers. | Low-latency inference on low-power ARM CPUs. |
| **Federated Bus** | Flower gRPC | Exchanges model parameters; TLS, secure aggregation, and differential privacy are not configured in the checked-in client/server. | Raw training telemetry stays on clients; model-update leakage is not mitigated. |
| **Coordinator Hub** | Swarm Hub (Flower) | Aggregates weights: $\mathbf{w}^{(t+1)} = \sum \frac{n_i}{N} \mathbf{w}_i^{(t)}$. | Global consensus model without centralization. |
| **Risk Engine** | Production Evaluator | NASA asymmetric penalty metric ($S = \sum e^{\alpha \Delta} - 1$). | Exponential penalty for late failure predictions. |

---

## 📊 Benchmark Results

Evaluated rigorously across multiple random seeds on the standard **NASA C-MAPSS** turbofan degradation benchmarks with official held-out test sets (`test_FD001.txt` and `test_FD002.txt`). Target Remaining Useful Life (RUL) is capped at 125 cycles following standard turbofan maintenance literature.

### 1. NASA C-MAPSS FD001 (Single Operating Condition)
*Evaluated over 5 independent random seeds ($n=5$)*:

| Architecture | Model Privacy | Test RMSE (Cycles) | NASA Asymmetric Score |
| :--- | :---: | :---: | :---: |
| **Constant Baseline** | N/A | $40.46 \pm 0.00$ | $17{,}301.19 \pm 0.00$ |
| **Isolated Local Edge** | 100% Local | $15.55 \pm 0.46$ | $464.43 \pm 74.67$ |
| **FedAvg Swarm (FedPdM)** | **100% Local** | **$14.92 \pm 0.42$** | **$394.85 \pm 45.73$** |
| *Centralized Upper Bound* | 0% (Exposed) | $14.75 \pm 0.69$ | $425.36 \pm 94.56$ |

> **Key Finding:** The federated swarm virtually matches the theoretical RMSE ceiling of centralized data pooling ($14.92$ vs $14.75$) while **outperforming centralized training on the NASA Risk Score** ($394.85$ vs $425.36$). The federated aggregation introduces implicit regularization that protects against local sensor overfitting.

### 2. NASA C-MAPSS FD002 (official test set, all 259 engines)
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

### 3. Asymmetric Risk-Aware Training Loss (`mse` vs. `asymmetric`)
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

---

## 📁 Repository Structure

```text
FedPdM-Edge/
├── client.py                 # Flower edge client running local training on plant data
├── server.py                 # Flower federated server aggregating global model weights
├── common.py                 # PyTorch 1D-CNN regressor, NASA scoring function, data loaders
├── compute_global_scale.py   # Global min-max scaling across operating conditions
├── final_benchmark.py        # Reproducible 4-way comparison (Constant, Local, FedAvg, Centralized)
├── run_fd002_federated.sh    # Multi-client local/remote test runner
├── setup_fd002.py            # Lifespan-sorted label-skew partition generator
├── sanity_check.py           # Verification gate for model convergence and data pipelines
├── plot_rul.py               # Generates RUL prediction vs ground truth curves
├── deploy.sh                 # Rsync deployment script to Raspberry Pi edge nodes
├── results/                  # Archived JSON result runs across seeds
│   ├── fd001/                # Official FD001 physical & simulation runs
│   └── fd002/                # Official FD002 heterogeneous operating condition runs
└── PDM_FIXES.md              # Experimental audit log and resolution notes
```

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Flower (`flwr`): `pip install flwr torch pandas numpy scikit-learn`
- NASA CMAPSS dataset (`CMAPSSData.zip` placed in home or data directory)

### 1. Verification & Sanity Check
Run the sanity verification to validate data loading, tensor windowing, and regression logic:
```bash
python sanity_check.py
```

### 2. Run Comprehensive Benchmark
Reproduce the 4-way comparative study across random seeds:
```bash
python final_benchmark.py
```

### 3. Launch Federated Swarm
Start the central aggregation server:
```bash
python server.py --rounds 10 --min-clients 2
```

In separate terminals (or remote edge nodes):
```bash
python client.py --client-id 1 --data-partition partition.json
python client.py --client-id 2 --data-partition partition.json
```

---

## 🤝 Citation & Attribution

This software is developed within the Interreg Aurora Project **ResilientEdge** (2026).
- **Authors:** Alessandro De Martini, Stefan Forsström (Mid Sweden University)
- **Partners:** Mid Sweden University (Sundsvall, Sweden), University of Oulu (Oulu, Finland)
- **Grant:** Interreg Aurora 20373282
