# FL PdM (CMAPSS) — Fix List

Target: `pcubuntu:~/fedpdm/` (server) + `rpi:~/`, `rpi2:~/` (clients).
Audit round 6, 2026-09-18.

---

## P8 — CLOSED. The result is real and independently reproduced.

Switched to Option A: supervised RUL regression. 1D CNN, piecewise-linear RUL capped at
`MAX_RUL = 125` (standard CMAPSS practice). Federated over 2 Pis with FedAvg.

```
round 0 (untrained): RMSE 93.83   baseline 42.01
round 1:             24.36
round 2:             21.99
round 3:             19.83
round 4:             16.14
round 5:             14.77        baseline 42.01
```

**Verified independently, not taken on trust:**

- **Partition is clean.** 90 train units / 10 test units, `overlap train/test: 0`.
- **No scale leak.** `compute_global_scale.py` computes min/max over the 90 **train**
  units only, healthy cycles only. Correct.
- **The baseline is honest.** Their baseline (predict train-mean RUL, 80.2) gives
  RMSE 41.99. The *best possible constant predictor* gives 41.83. There is no weaker-
  baseline trick here:

  ```
  predict train mean (80.2)      RMSE 41.99   <- the one used
  predict train median (87.0)    RMSE 41.94
  predict cap (125.0)            RMSE 58.60
  best possible constant (84.0)  RMSE 41.83
  ```

- **Reproduced centrally with their architecture, split and scale:**

  ```
  epoch  0 (untrained) RMSE 93.83   <- matches their round 0 exactly
  epoch  4             RMSE 16.97
  epoch  8             RMSE 14.39
  epoch 10             RMSE 16.08
  reported federated:  14.77        <- sits in that band
  ```

Model 14.77 vs best-constant 41.83 — **2.8× better than any model-free predictor**, and
the untrained network (93.83) is correctly *worse* than the baseline. Training helps
monotonically. This is the first genuine result the project has produced.

Also fixed this round: **`CLIENT_ID` is now required** (`raise ValueError` if unset) —
P16 item closed. Pis are in sync (`client.py` 2897 B, `common.py` 2151 B,
`global_scale.json` 430 B on both).

`plot_rul.py` plots model RMSE against the baseline line — honest framing, P15 satisfied
for this script.

---

## P21 — `sanity_check.py` no longer asserts anything (NEW)

It computes `baseline_rmse` and `untrained_rmse` and **prints them**. There is no
`assert`. It cannot fail, so it cannot gate.

Round 5 is exactly why this matters: the lead-time regression went unnoticed because the
gate was broken. Now the gate is runnable but toothless.

**Fix — add assertions:**

```python
assert untrained_rmse > baseline_rmse, "untrained net should be worse than a constant"
assert trained_rmse < baseline_rmse * 0.75, "model must beat the constant baseline by 25%"
```

and run it over the seed distribution, not seed 42 alone. Current margins are wide
(93.83 / 41.99 / 14.77), so these will pass — that is the point: lock the win in before
the next refactor moves something.

---

## P19 — Dead files with broken imports (CARRIED, now worse)

`common.py` dropped `CmapssAutoencoder`, `extract_sequences` and `LEAD`. Two files still
import them and fail immediately:

```
fl_cmapss_fixed.py             from common import ... CmapssAutoencoder ... extract_sequences, LEAD
plot_explanatory_physical.py   from common import ... CmapssAutoencoder ... SEQ_LEN
```

Orphaned artefacts from the two abandoned approaches:

- `global_model.pt` (503 KB, autoencoder), `global_model.npz` (2 KB, Mahalanobis)
- `plot_physical.py`, `plot_fl_metrics.py`, `plot_engine_degradation.py`,
  `plot_explanatory.py` — all describe models that no longer exist
- `client_1_history.json`, `client_2_history.json` — autoencoder-era losses
- `physical_pdm_metrics.png`, `explanatory_physical.png` — figures of the dead models

**Delete them.** The repo currently contains three generations of model, two of which do
not run. Keep `common.py`, `compute_global_scale.py`, `client.py`, `server.py`,
`sanity_check.py`, `plot_rul.py`, `PDM_FIXES.md`. Anything worth keeping from the dead
branches is in git history — once P12 is done.

---

## P22 — Use the official test split and the NASA score (NEW)

The holdout is 10 units carved out of `train_FD001.txt`. That is methodologically fine,
but `test_FD001.txt` + `RUL_FD001.txt` are the published benchmark and they are still
unused. Without them the 14.77 cannot be compared to any paper.

- Evaluate on `test_FD001.txt`, scoring the **last window of each engine** against
  `RUL_FD001.txt` — that is how the benchmark is defined.
- Add the **NASA asymmetric score** from `Damage Propagation Modeling.pdf` alongside
  RMSE. It penalises late predictions harder than early ones, which is the whole point in
  maintenance. RMSE alone treats a dangerous late call the same as a harmless early one.

Published FD001 results sit around 12–16 RMSE, so 14.77 is plausible — but keep the
holdout number and the official number clearly labelled as different things.

---

## P10 — Baselines (partially answered, still not in the repo)

I measured centralised at ~14.4–16.1 RMSE; federated reports 14.77. **Federated ≈
centralised**, which is the expected and reportable result. Put it in the repo properly:

1. **Constant** — 41.99 (already computed per round).
2. **Local-only** — each Pi trains alone, no aggregation; evaluate both on the global
   holdout; mean ± std. This is the row that shows federation is worth the network.
3. **Centralised** — pooled 90 units.

Four rows, one table. Then **FD002 / FD004** (6 operating conditions) where clients are
genuinely non-IID and FedAvg vs FedProx becomes a real comparison. On FD001 everything is
IID and federation has nothing to prove.

---

## P11 — Multi-seed

Every number is seed 42. Run 5–10 seeds, report mean ± std. Cheap now — a run is 5 rounds.

---

## P12 — Commit (still one commit, now 17 untracked files)

Still only `eb4e7ed Initial commit`. Three model generations exist and none of them are
in history. Commit now, before P19 deletes anything.

- `.gitignore`: `results/*.png`, `*.pt`, `*.npz`, root `*.png` — keep the JSON.
- **`rpi:~/` and `rpi2:~/` are still not repositories.** `client.py`, `common.py` and
  `global_scale.json` are hand-copied. They are in sync right now; nothing enforces it.
  One repo plus a documented rsync step.

---

## P20 — Minor

- Client `evaluate()` returns a hardcoded `0.0`, so `losses_distributed` is a column of
  zeros. Return the local training RMSE instead, or drop the field.
- `server.py` recomputes the train-set RUL distribution on every startup just to get one
  scalar baseline. Cache it in `global_scale.json`.

---

## P16 — Deployment hardening (remaining)

- Write `results/partition.json` on the server; have clients read it instead of relying
  on numpy's frozen legacy RNG stream. (Works today, verified — but it is implicit.)
- `server_address="192.168.1.144:8081"` hardcoded in `client.py` — read from env.
- `start_server` / `start_numpy_client` deprecated in flwr 1.37 → `ServerApp`/`ClientApp`.

---

## P13 — Housekeeping

- Outputs still in the source root (`*.png`, `*.npz`, `*.pt`, `*_history.json`,
  `results_physical.json`). Move under `results/`.
- `cmapss.zip` (357 KB HTML landing page) still in `/Users/eldemaster/orchestrator/` on
  the Mac. Delete it.

---

## Order of work

**P12 first** (commit, so three generations of work stop being unversioned), then
**P21** (put teeth back in the gate), then **P19** (delete the dead branches), then
P10 → P22 → P11 → P20 → P16 → P13.

## How to verify

```bash
cd ~/fedpdm
python compute_global_scale.py
python sanity_check.py          # must ASSERT, not just print
# server: python server.py   |   pis: CLIENT_ID=1|2 python client.py
python plot_rul.py
```

Acceptance: four-row table (constant, local-only, federated, centralised), mean ± std
over ≥ 5 seeds, plus the official `test_FD001` RMSE and NASA score reported separately.
