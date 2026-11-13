# Paper

IEEEtran manuscript source for the paired FD001–FD004 study.

## Build

```bash
cd paper
make
```

The build requires TeX Live with `IEEEtran`, `pgfplots`, `booktabs`, `multirow`, `cite`, and `latexmk`. `make_figures.py` uses only the Python standard library. It validates and generates tables/macros from `results/reproducible/summary.csv`, `results/asymmetric_loss_comparison.json`, and `results/edge/local-smoke-macos.json`; incomplete or unexpected five-seed inputs fail the build instead of producing partial tables.

Do not edit `generated/` by hand. Run `python experiments/reproduce.py --data /path/to/CMAPSSData.zip` from the repository root, then run `python make_figures.py` from `paper/` after the matrix completes. Raw run JSON files are the source of truth.

## Open items

Remaining `\todo{...}` markers are editorial items requiring author input or verified literature citations. Results, privacy claims, and edge hardware limitations are documented in the manuscript and `results/EXPERIMENTS.md`.
