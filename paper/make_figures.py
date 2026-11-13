#!/usr/bin/env python3
"""Generate tables, macros and pgfplots data for the paper from results/.

Pure standard library (no numpy/matplotlib needed). Run from anywhere:
    python3 paper/make_figures.py
Outputs go to paper/generated/. Nothing is hard-coded: every number is parsed
from results/<ds>/benchmark_table.txt or results/<ds>/results_physical_seed*.json.
Missing inputs yield \\todo{} placeholders instead of invented values.
"""
import glob, json, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)
DATASETS = [("fd001", "FD001"), ("fd002", "FD002")]
ROWS = [("Constant Baseline", "Constant baseline", "n/a"),
        ("Local-Only", "Isolated local (avg.)", "local"),
        ("Federated", "FedAvg swarm (ours)", "local + updates shared"),
        ("Centralized", "Centralized reference", "pooled data")]
PAT = re.compile(r"^(\S.*?)\s*(?:\(.*?\))?\s*:\s*RMSE = ([\d.]+) ± ([\d.]+) \| NASA Score = ([\d.]+) ± ([\d.]+)")
TODO = r"\todo{pending}"


def fmt(x, nd=2):
    s = f"{x:,.{nd}f}"
    return s.replace(",", r"\,")


def parse_table(ds):
    p = ROOT / "results" / ds / "benchmark_table.txt"
    res = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = PAT.match(line.strip())
            if m:
                res[m.group(1).strip()] = tuple(float(m.group(i)) for i in range(2, 6))
        return res
    # Fallback to benchmark_e10.json + physical seed JSONs when benchmark_table.txt is absent
    ej = ROOT / "results" / ds / "benchmark_e10.json"
    if ej.exists():
        data = json.load(open(ej))
        key_map = {"baseline": "Constant Baseline", "local": "Local-Only", "central": "Centralized"}
        for row in data.get("summary", []):
            if row["method"] in key_map:
                res[key_map[row["method"]]] = (
                    row["rmse_mean"], row["rmse_std"], row["nasa_mean"], row["nasa_std"]
                )
    ds_seeds = load_seeds(ds)
    if ds_seeds:
        rm = [m["model_rmse"][-1][1] for m in ds_seeds.values()]
        ns = [m["nasa_score"][-1][1] for m in ds_seeds.values()]
        res["Federated"] = (st.mean(rm), st.pstdev(rm), st.mean(ns), st.pstdev(ns))
    return res


def load_seeds(ds):
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / ds / "results_physical_seed*.json"))):
        seed = int(re.search(r"seed(\d+)", f).group(1))
        m = json.load(open(f))["metrics_centralized"]
        out[seed] = m
    return out


macros, tex = [], []
tables = {ds: parse_table(ds) for ds, _ in DATASETS}
seeds = {ds: load_seeds(ds) for ds, _ in DATASETS}

# ---- main table -----------------------------------------------------------
tex.append(r"\begin{tabular}{llcrr}\toprule")
tex.append(r"Dataset & Method & Data placement & RMSE (cycles) & NASA score \\\midrule")
for i, (ds, label) in enumerate(DATASETS):
    t = tables[ds]
    for key, name, priv in ROWS:
        v = next((t[k] for k in t if k.startswith(key)), None)
        first = rf"\multirow{{4}}{{*}}{{{label}}}" if key == ROWS[0][0] else ""
        if v:
            fed = key == "Federated"
            r = rf"${fmt(v[0])}\pm{fmt(v[1])}$"; s = rf"${fmt(v[2])}\pm{fmt(v[3])}$"
            if fed: r, s = rf"\textbf{{{r}}}", rf"\textbf{{{s}}}"
        else:
            r = s = TODO
        tex.append(rf"{first} & {name} & {priv} & {r} & {s} \\")
    if i < len(DATASETS) - 1:
        tex.append(r"\midrule")
tex.append(r"\bottomrule\end{tabular}")
(OUT / "table_main.tex").write_text("\n".join(tex) + "\n")

# ---- macros + per-seed table + convergence data ---------------------------
tex = [r"\begin{tabular}{lrrrr}\toprule",
       r"Seed & \multicolumn{2}{c}{FD001 (round 5)} & \multicolumn{2}{c}{FD002 (round 5)} \\",
       r"& RMSE & NASA & RMSE & NASA \\\midrule"]
all_seeds = sorted(set().union(*[set(s) for s in seeds.values()]))
for sd in all_seeds:
    cells = []
    for ds, _ in DATASETS:
        m = seeds[ds].get(sd)
        cells += [fmt(m["model_rmse"][-1][1]), fmt(m["nasa_score"][-1][1], 1)] if m else [TODO, TODO]
    tex.append(f"{sd} & " + " & ".join(cells) + r" \\")
tex.append(r"\midrule")
for ds, label in DATASETS:
    s = seeds[ds]
    if not s:
        continue
    rm = [m["model_rmse"][-1][1] for m in s.values()]
    ns = [m["nasa_score"][-1][1] for m in s.values()]
    pre = "FdOne" if ds == "fd001" else "FdTwo"
    macros += [rf"\newcommand{{\{pre}FedRmseMeanJson}}{{{fmt(st.mean(rm))}}}",
               rf"\newcommand{{\{pre}FedRmseSampleStd}}{{{fmt(st.stdev(rm))}}}",
               rf"\newcommand{{\{pre}FedNasaSampleStd}}{{{fmt(st.stdev(ns),1)}}}",
               rf"\newcommand{{\{pre}NSeeds}}{{{len(s)}}}"]
for ds, _ in DATASETS:
    pre = "FdOne" if ds == "fd001" else "FdTwo"
    for key, nm in [("Constant", "Const"), ("Local", "Local"), ("Federated", "Fed"), ("Centralized", "Cent")]:
        v = next((tables[ds][k] for k in tables[ds] if k.startswith(key)), None)
        for j, q in enumerate(["Rmse", "RmseStd", "Nasa", "NasaStd"]):
            val = fmt(v[j], 2) if v else TODO
            macros.append(rf"\newcommand{{\{pre}{nm}{q}}}{{{val}}}")
tex.append(r"\bottomrule\end{tabular}")
(OUT / "table_seeds.tex").write_text("\n".join(tex) + "\n")

# ---- asymmetric loss comparison table + macros ----------------------------
asym_path = ROOT / "results" / "asymmetric_loss_comparison.json"
asym_tex = [
    r"\begin{tabular}{llcccrr}\toprule",
    r"Dataset & Setup & Loss & RMSE (cycles) & NASA score & $\Delta$ NASA & Late rate (\%) \\\midrule",
]
if asym_path.exists():
    asym_data = json.load(open(asym_path))
    setup_rows = [
        ("local", "Isolated local"),
        ("fedavg", "FedAvg (5r $\\times$ 2e)"),
        ("central", "Centralized (10e)"),
    ]
    for i, (ds_key, ds_label) in enumerate([("FD001", "FD001"), ("FD002", "FD002")]):
        pre = "AsymFdOne" if ds_key == "FD001" else "AsymFdTwo"
        sub = asym_data["subsets"][ds_key]
        losses = sub["losses"]
        for j, (s_key, s_label) in enumerate(setup_rows):
            s_macro = {"local": "Local", "fedavg": "Fed", "central": "Cent"}[s_key]
            mse_s = losses["mse"]["summary"][s_key]
            asy_s = losses["asymmetric"]["summary"][s_key]
            for loss_key, loss_label, cur, other in [
                ("mse", r"\texttt{mse}", mse_s, asy_s),
                ("asymmetric", r"\texttt{asymmetric}", asy_s, mse_s),
            ]:
                l_macro = "Mse" if loss_key == "mse" else "Asy"
                ds_col = rf"\multirow{{6}}{{*}}{{{ds_label}}}" if (j == 0 and loss_key == "mse") else ""
                setup_col = rf"\multirow{{2}}{{*}}{{{s_label}}}" if loss_key == "mse" else ""
                r_str = rf"${fmt(cur['rmse_mean'])}\pm{fmt(cur['rmse_std'])}$"
                if cur["rmse_mean"] <= other["rmse_mean"]:
                    r_str = rf"\textbf{{{r_str}}}"
                n_str = rf"${fmt(cur['nasa_score_mean'], 1)}\pm{fmt(cur['nasa_score_std'], 1)}$"
                if cur["nasa_score_mean"] <= other["nasa_score_mean"]:
                    n_str = rf"\textbf{{{n_str}}}"
                l_str = rf"${fmt(cur['late_rate_pct_mean'], 1)}\pm{fmt(cur['late_rate_pct_std'], 1)}$"
                if cur["late_rate_pct_mean"] <= other["late_rate_pct_mean"]:
                    l_str = rf"\textbf{{{l_str}}}"
                if loss_key == "mse":
                    d_str = "---"
                else:
                    pct = 100.0 * (cur["nasa_score_mean"] - other["nasa_score_mean"]) / other["nasa_score_mean"]
                    sign = "+" if pct > 0 else ""
                    d_str = rf"${sign}{pct:.1f}\%$"
                    macros.append(rf"\newcommand{{\{pre}{s_macro}NasaDelta}}{{{sign}{pct:.1f}\%}}")
                asym_tex.append(rf"{ds_col} & {setup_col} & {loss_label} & {r_str} & {n_str} & {d_str} & {l_str} \\")
                macros += [
                    rf"\newcommand{{\{pre}{s_macro}{l_macro}Rmse}}{{{fmt(cur['rmse_mean'])}}}",
                    rf"\newcommand{{\{pre}{s_macro}{l_macro}RmseStd}}{{{fmt(cur['rmse_std'])}}}",
                    rf"\newcommand{{\{pre}{s_macro}{l_macro}Nasa}}{{{fmt(cur['nasa_score_mean'], 1)}}}",
                    rf"\newcommand{{\{pre}{s_macro}{l_macro}NasaStd}}{{{fmt(cur['nasa_score_std'], 1)}}}",
                    rf"\newcommand{{\{pre}{s_macro}{l_macro}Late}}{{{fmt(cur['late_rate_pct_mean'], 1)}}}",
                    rf"\newcommand{{\{pre}{s_macro}{l_macro}LateStd}}{{{fmt(cur['late_rate_pct_std'], 1)}}}",
                ]
            if j < len(setup_rows) - 1:
                asym_tex.append(r"\cmidrule(lr){2-7}")
        if i == 0:
            asym_tex.append(r"\midrule")
asym_tex.append(r"\bottomrule\end{tabular}")
(OUT / "table_asymmetric.tex").write_text("\n".join(asym_tex) + "\n")

(OUT / "macros.tex").write_text("% auto-generated by make_figures.py\n" + "\n".join(macros) + "\n")

# convergence: global-model RMSE on official test set vs round, one column per seed
for ds, _ in DATASETS:
    s = seeds[ds]
    if not s:
        continue
    sd = sorted(s)
    rounds = [r for r, _ in s[sd[0]]["model_rmse"]]
    lines = ["round " + " ".join(f"s{x}" for x in sd) + " mean"]
    for i, r in enumerate(rounds):
        vals = [s[x]["model_rmse"][i][1] for x in sd]
        lines.append(f"{r} " + " ".join(f"{v:.4f}" for v in vals) + f" {st.mean(vals):.4f}")
    (OUT / f"conv_{ds}.dat").write_text("\n".join(lines) + "\n")
    base = s[sd[0]]["baseline_rmse"][0][1]
    (OUT / f"baseline_{ds}.tex").write_text(f"{base:.4f}")

# bar chart data (RMSE) for the 4-way comparison
lines = ["x label fd001 fd001e fd002 fd002e"]
for i, (key, name, _) in enumerate(ROWS):
    vals = []
    for ds, _l in DATASETS:
        v = next((tables[ds][k] for k in tables[ds] if k.startswith(key)), None)
        vals += [f"{v[0]:.2f}", f"{v[1]:.2f}"] if v else ["nan", "0"]
    lines.append(f"{i} {key.split('-')[0].split()[0]} " + " ".join(vals))
(OUT / "bars_rmse.dat").write_text("\n".join(lines) + "\n")
print("wrote", sorted(p.name for p in OUT.iterdir()))
