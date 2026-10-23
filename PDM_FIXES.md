# FL PdM (CMAPSS) — Fix List

Target: `pcubuntu:~/fedpdm/` (server) + `rpi:~/fedpdm_client/`, `rpi2:~/fedpdm_client/`.
Audit round 8, 2026-09-18.

---

## Closed this round

- **P23 — data parity fixed.** `final_benchmark.py` now filters every split through the
  partition:
  ```python
  valid_units = set(units_c1 + units_c2)
  ...
  if u_int not in valid_units: continue
  ```
  Centralised, local-only and federated all train on the same 90 units. The comparison is
  fair now.
- **P24 — federated row is comparable.** Five physical runs
  (`results/results_physical_seed{42,100,2024,7,777}.json`), aggregated with mean ± std
  alongside the other three rows. Measured:
  ```
  seed   42: RMSE 15.20  NASA 398.3
  seed  100: RMSE 15.52  NASA 471.5
  seed 2024: RMSE 14.76  NASA 378.6
  seed    7: RMSE 14.82  NASA 396.6
  seed  777: RMSE 14.29  NASA 329.2
  -> federated  RMSE 14.92 ± 0.46   NASA 394.8 ± 51.1   (n=5)
  ```
  Tight spread. The result is stable, not a lucky seed.
- **P11 — 5 seeds** across all four rows.
- **P26 — capping documented.** The table prints
  `"Note: True RUL is capped at 125 cycles during evaluation as per standard CMAPSS practice."`
- **P25 (part)** — `__pycache__/` added to `.gitignore` and removed from the index;
  `plot_explanatory_physical.py` deleted.

---

## P21 — The sanity gate does not run, and its assertions are inert (STILL OPEN)

This is the third consecutive round in which a refactor has broken the gate. It is now
broken in three separate ways at once.

**1. It crashes on import of the data.** `load_cmapss_raw` gained a required `filename`
argument when the official test set was added. `sanity_check.py` was not updated:

```
File "/home/eldemaster/fedpdm/sanity_check.py", line 11, in run_sanity_check
    df_raw = load_cmapss_raw()
TypeError: load_cmapss_raw() missing 1 required positional argument: 'filename'
```

**2. The assertion block is duplicated and one line is a no-op:**

```python
assert untrained > baseline, "untrained net should be worse than a constant"
assert untrained > baseline, "untrained net should be worse than a constant"   # duplicate
assert trained_rmse < baseline * 0.75 if "trained_rmse" in locals() else True
```

`trained_rmse` is never computed in this file, so `"trained_rmse" in locals()` is `False`,
the ternary returns `True`, and the statement reduces to `assert True`. It always passes
and tests nothing. Verified. That is worse than having no assertion — it reads like a
gate in code review.

**3. It guards the wrong quantity.** It still evaluates on the old internal holdout:

```python
unique_units = df_raw['unit'].unique()
np.random.shuffle(unique_units)
global_test_units = unique_units[-10:]
```

while `server.py` and `final_benchmark.py` both report against the official
`test_FD001.txt`. The gate and the headline number now measure different things.

**Fix — rewrite the file against the official test set:**

```python
df_raw  = load_cmapss_raw('train_FD001.txt')
df_test = load_cmapss_raw('test_FD001.txt')   # + RUL_FD001.txt, via extract_last_sequence
...
trained_rmse = <train a model for 10 epochs, or load the last federated round>
assert untrained_rmse > baseline_rmse,        "untrained net must be worse than a constant"
assert trained_rmse   < baseline_rmse * 0.75, "model must beat the constant baseline by 25%"
```

Current margins are wide (untrained 84.58, baseline 40.46, trained 14.92), so both pass
easily. **That is the point** — lock the win in while it is safe to do so.

Then add `python sanity_check.py` to the top of whatever you run, so a broken gate fails
loudly instead of silently not existing.

---

## P28 — Commit discipline (NEW)

Still 4 commits, last one `9a3230d`. Uncommitted right now: 9 modified files, 1 deletion,
1 index deletion, and 6 untracked files including all five seed results.

Every round of this audit has ended with the round's work outside git. The fixes are good;
they are just not being saved. Commit P23, P24, P25 and P26 now, separately.

---

## P27 — Next experiment: FD002 / FD004

FD001 is a single operating condition, so the two clients are IID and FedAvg has nothing
to correct. Expect federated ≈ centralised on the table below — correct, but not a
finding.

The finding lives in FD002/FD004 (6 operating conditions, already in `CMAPSSData.zip`):
partition by operating condition, and the local-only vs federated gap becomes the point of
the project. FedProx vs FedAvg also becomes a real comparison there.

---

## P20 — Minor

- Client `evaluate()` returns a hardcoded `0.0`; `losses_distributed` is a column of
  zeros. Return local RMSE or drop the field.
- `start_server` / `start_numpy_client` deprecated in flwr 1.37 → `ServerApp` / `ClientApp`.

---

## Verified result (independent reproduction)

Official `test_FD001.txt`, last window per engine, RUL capped at 125:

```
federated (5 seeds)  RMSE 14.92 ± 0.46   NASA 394.8 ± 51.1
constant baseline    RMSE 40.46
```

Reproduced from scratch with the same architecture, scale and 90 client units:

```
epoch  0 (untrained) RMSE  84.58   <- matches the servers' round 0 exactly
epoch  5             RMSE  15.46   NASA 465.7
epoch  8             RMSE  15.01   NASA 395.8
```

Published FD001 results sit around 12–16 RMSE. This is competitive and the number is real.

---

## Order of work

**P21** (the gate — three rounds running) → **P28** (commit) → P27 → P20.

## How to verify

```bash
cd ~/fedpdm
python compute_global_scale.py
python sanity_check.py            # must RUN, and assert BOTH conditions
./deploy.sh
# server: python server.py  |  pis: cd ~/fedpdm_client && CLIENT_ID=1|2 python client.py
python final_benchmark.py
```

---

## Round 9 — 2026-10-23: FD002 (non-IID) result

P27 done. Pipeline moved to FD002 (6 operating conditions); clients partitioned by operating
condition. FD001 results kept apart in `results/fd001/`; FD002 in `results/fd002/`. The
benchmark was reading the FD001 federated files, which would have mixed datasets: fixed by
separating the folders. Server and client now take the seed from the `SEED` env variable (the
client seed used to be fixed at 42, so the old FD001 std only reflects the server seed).

Official `test_FD002.txt`, last window per engine, RUL capped at 125, 5 seeds, 5 rounds:

```
Constant baseline : RMSE 43.41          NASA   59296
Local-only (avg)  : RMSE 24.96 ± 0.99   NASA   10702 ± 2404
Federated (FedAvg): RMSE 21.98 ± 0.99   NASA    4878 ± 2291
Centralized       : RMSE 19.57 ± 0.86   NASA    2975 ± 1405
```

Federated beats local-only by 3.0 RMSE (12 %) and halves the NASA score; it stays 2.4 RMSE
behind centralised. This is the expected ordering and is the first real federated finding
(on FD001 the clients were IID and there was nothing to correct).

Closed: P21 (sanity gate, 15ad636), P27 (FD002 run), P28 (everything committed).

Open:
- FedProx vs FedAvg on FD002 (not run).
- Only 5 rounds, 2 clients; no round-count sweep. Federated may not have converged.
- P20: client `evaluate()` returns a hardcoded 0.0; `start_server` / `start_numpy_client`
  deprecated in flwr 1.37.
- `run_fd002_federated.sh` must launch clients with `ssh -f -n`, or the first ssh blocks.
