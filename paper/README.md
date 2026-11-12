# Paper skeleton (IEEEtran)

Federated PdM on edge nodes: C-MAPSS FD001/FD002, FedAvg, asymmetric NASA score.

## Build
```bash
cd paper
make            # regenerates tables/plot data from ../results, then runs latexmk
make clean
```
Needs a TeX Live with `IEEEtran`, `pgfplots`, `booktabs`, `multirow`, `cite`, and `latexmk`. `make_figures.py` uses only the Python standard library.

## Layout
- `main.tex`, `sections/*.tex`, `references.bib` (only real, checked references; verify before submission)
- `make_figures.py` parses `results/fd00*/benchmark_table.txt` and `results_physical_seed*.json` into `generated/` (tables, `macros.tex`, pgfplots data). Do not edit `generated/` by hand; rerun after `results/` changes.
- `figures/` for drawn/exported figures (empty so far; `results/` has no PNGs).

## Open items
Every pending value is a red `\todo{...}`; list them with `grep -rn 'todo{' sections main.tex`.
