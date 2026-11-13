# Experiments (FD002, official test_FD002, all 259 engines, RUL cap 125)

Seeds: 42, 100, 2024, 7, 777. Data: CMAPSSData.zip (not committed).

## Real, on physical nodes (2026-11-12)
Server: Ubuntu PC (`eldemaster-ubuntu`, flwr 1.37). Clients: RPi `edgedataspace` (client 1), `edgedataspace2` (client 2), Cortex-A72, 3.7 GiB, torch 2.14 cpu.
`PROXIMAL_MU=0 ROUNDS=5 ./run_fd002_federated.sh` -> `results/fd002/results_physical_seed*.json`, logs `server_mu0_*.log`.
Result FedAvg: RMSE 22.27 ± 0.44, NASA 6037 ± 1030. Train time/round: RPi1 12.8 s, RPi2 17.3 s (seed 42 client logs); peak RSS 570-620 MB.

## Local (Mac, in-process, `fed_sim.py` / `final_benchmark.py --epochs 10`)
- `results/fd002/benchmark_e10.{json,csv}`: baseline 43.51; local 25.01 ± 0.54; centralized 20.39 ± 1.11.
- `sim_fedavg_r5_e2.json` 22.20 ± 0.42 (matches physical run); `sim_fedprox0.01_r5_e2.json` 22.20 ± 0.58; `sim_fedprox0.1_r5_e2.json` 22.86 ± 0.93; `sim_fedavg_r15_e2.json` 19.27 ± 1.30 NASA 2742.
- Commands: `python fed_sim.py --mu M --rounds R --out ...`, `python final_benchmark.py --epochs 10`.

## Started / NOT finished (no numbers claimed)
- Physical 15-round run (`ROUNDS=15 RESULTS_DIR=results/fd002_r15`) was launched on the PC; results not collected.
- `final_benchmark.py --epochs 30` and `fed_sim.py --rounds 30` were running locally; not collected.
- FD001 not re-run. FedProx not run on physical nodes. Early stopping by client val RMSE: no gain in sim (19.05 vs 19.27, from r15 history).
- Old pre-fix results kept in `results/fd002/legacy_pre_fix/` (evaluated 253 of 259 engines).
