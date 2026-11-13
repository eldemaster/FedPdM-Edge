#!/usr/bin/env python3
"""Generate paper tables only from five-seed rows in results/reproducible."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'results' / 'reproducible' / 'summary.csv'
OUT = Path(__file__).resolve().parent / 'generated'
OUT.mkdir(exist_ok=True)


def fmt(value, decimals=2):
    return f'{float(value):,.{decimals}f}'.replace(',', r'\,')


def esc(value):
    return str(value).replace('_', r'\_')


def main():
    records = {}
    tests = []
    if SOURCE.exists():
        with SOURCE.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('comparison') and row.get('paired_sign_flip_p'):
                    tests.append(row)
                if (row.get('method') and row.get('metric') in ('rmse', 'nasa')
                        and row.get('scope', 'global') == 'global' and int(row['n']) >= 5):
                    records[(row['dataset'], row['partition'], row['method'], row['metric'])] = row
    clients = {}
    if SOURCE.exists():
        with SOURCE.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if (row.get('method') and row.get('scope') == 'client' and row.get('metric') in ('rmse','nasa')
                        and int(row['n']) >= 5):
                    clients[(row['dataset'],row['partition'],row['method'],int(row['client_id']),row['metric'])]=row

    def performance_table(filename, groups, methods):
        lines=[r'\begin{tabular}{lllrrr}',r'\toprule',
               r'Dataset & Partition & Method & $n$ & RMSE & NASA score \\',r'\midrule']
        count=0
        for ds,part in groups:
            for method in methods:
                r=records.get((ds,part,method,'rmse')); s=records.get((ds,part,method,'nasa'))
                if not r or not s: continue
                lines.append(f'{ds} & {esc(part)} & {esc(method)} & {r["n"]} & '
                             f'${fmt(r["mean"])}\\pm{fmt(r["std"])}$ & '
                             f'${fmt(s["mean"])}\\pm{fmt(s["std"])}$ \\\\')
                count+=1
        if not count: lines.append(r'\multicolumn{6}{c}{No complete five-seed rows available.} \\')
        lines += [r'\bottomrule',r'\end{tabular}']
        (OUT/filename).write_text('\n'.join(lines)+'\n',encoding='utf-8')

    performance_table('table_main.tex',[(f'FD00{i}','iid') for i in range(1,5)],('centralized','local_only','fedavg'))
    performance_table('table_algorithms.tex',[('FD002','regime'),('FD004','regime')],
                       ('centralized','local_only','fedavg','fedprox','fedadam','federated_asymmetric','asymmetric','cnn_lstm'))
    performance_table('table_partitions.tex',[(ds,p) for ds in ('FD002','FD004') for p in ('iid','regime','engine_imbalance')],
                       ('fedavg','fedprox','fedadam'))

    lines=[r'\begin{tabular}{llllrr}',r'\toprule',
           r'Dataset & Partition & Method & Client & RMSE & NASA score \\',r'\midrule']
    count=0
    for ds,part in (('FD002','regime'),('FD004','regime')):
        for method in ('centralized','local_only','fedavg','fedprox','fedadam','federated_asymmetric','asymmetric','cnn_lstm'):
            for cid in (0,1):
                r=clients.get((ds,part,method,cid,'rmse')); s=clients.get((ds,part,method,cid,'nasa'))
                if not r or not s: continue
                lines.append(f'{ds} & {esc(part)} & {esc(method)} & {cid+1} & '
                             f'${fmt(r["mean"])}\\pm{fmt(r["std"])}$ & '
                             f'${fmt(s["mean"])}\\pm{fmt(s["std"])}$ \\\\')
                count+=1
    if not count: lines.append(r'\multicolumn{6}{c}{No complete five-seed client rows available.} \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'table_clients.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    lines=[r'\begin{tabular}{llllr}',r'\toprule',r'Dataset & Partition & Comparison & Metric & paired $p$ \\',r'\midrule']
    count=0
    for row in tests:
        key_comparisons=('fedavg_vs_centralized','fedavg_vs_local_only')
        method_comparisons=key_comparisons+('fedavg_vs_fedprox','fedavg_vs_fedadam','fedavg_vs_federated_asymmetric',
                                           'centralized_vs_asymmetric','centralized_vs_cnn_lstm')
        primary_regime=(row['dataset'] in ('FD002','FD004') and row['partition']=='regime'
                        and row['comparison'] in method_comparisons)
        baseline_fd001=(row['dataset']=='FD001' and row['partition']=='iid' and row['comparison'] in key_comparisons)
        if ((primary_regime or baseline_fd001) and int(row['n'])>=5):
            lines.append(f'{row["dataset"]} & {esc(row["partition"])} & {esc(row["comparison"])} & {row["metric"]} & '
                         f'{float(row["paired_sign_flip_p"]):.4f} \\\\')
            count+=1
    if not count: lines.append(r'\multicolumn{5}{c}{No complete five-seed paired tests available.} \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (OUT/'table_significance.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

    # ---- asymmetric loss comparison table + macros ------------------------
    macros = []
    asym_path = ROOT / 'results' / 'asymmetric_loss_comparison.json'
    asym_tex = [
        r'\begin{tabular}{llcccrr}\toprule',
        r'Dataset & Setup & Loss & RMSE (cycles) & NASA score & $\Delta$ NASA & Late rate (\%) \\\midrule',
    ]
    if asym_path.exists():
        asym_data = json.loads(asym_path.read_text(encoding='utf-8'))
        setup_rows = [
            ('local', 'Isolated local'),
            ('fedavg', r'FedAvg (5r $\times$ 2e)'),
            ('central', 'Centralized (10e)'),
        ]
        for i, (ds_key, ds_label) in enumerate([('FD001', 'FD001'), ('FD002', 'FD002')]):
            pre = 'AsymFdOne' if ds_key == 'FD001' else 'AsymFdTwo'
            losses = asym_data['subsets'][ds_key]['losses']
            for j, (s_key, s_label) in enumerate(setup_rows):
                s_macro = {'local': 'Local', 'fedavg': 'Fed', 'central': 'Cent'}[s_key]
                mse_s = losses['mse']['summary'][s_key]
                asy_s = losses['asymmetric']['summary'][s_key]
                for loss_key, loss_label, cur, other in [
                    ('mse', r'\texttt{mse}', mse_s, asy_s),
                    ('asymmetric', r'\texttt{asymmetric}', asy_s, mse_s),
                ]:
                    l_macro = 'Mse' if loss_key == 'mse' else 'Asy'
                    ds_col = rf'\multirow{{6}}{{*}}{{{ds_label}}}' if (j == 0 and loss_key == 'mse') else ''
                    setup_col = rf'\multirow{{2}}{{*}}{{{s_label}}}' if loss_key == 'mse' else ''
                    r_str = rf"${fmt(cur['rmse_mean'])}\pm{fmt(cur['rmse_std'])}$"
                    if cur['rmse_mean'] <= other['rmse_mean']:
                        r_str = rf'\textbf{{{r_str}}}'
                    n_str = rf"${fmt(cur['nasa_score_mean'], 1)}\pm{fmt(cur['nasa_score_std'], 1)}$"
                    if cur['nasa_score_mean'] <= other['nasa_score_mean']:
                        n_str = rf'\textbf{{{n_str}}}'
                    l_str = rf"${fmt(cur['late_rate_pct_mean'], 1)}\pm{fmt(cur['late_rate_pct_std'], 1)}$"
                    if cur['late_rate_pct_mean'] <= other['late_rate_pct_mean']:
                        l_str = rf'\textbf{{{l_str}}}'
                    if loss_key == 'mse':
                        d_str = '---'
                    else:
                        pct = 100.0 * (cur['nasa_score_mean'] - other['nasa_score_mean']) / other['nasa_score_mean']
                        sign = '+' if pct > 0 else ''
                        d_str = rf'${sign}{pct:.1f}\%$'
                        macros.append(rf'\newcommand{{\{pre}{s_macro}NasaDelta}}{{{sign}{pct:.1f}\%}}')
                    asym_tex.append(rf'{ds_col} & {setup_col} & {loss_label} & {r_str} & {n_str} & {d_str} & {l_str} \\')
                    macros += [
                        rf"\newcommand{{\{pre}{s_macro}{l_macro}Rmse}}{{{fmt(cur['rmse_mean'])}}}",
                        rf"\newcommand{{\{pre}{s_macro}{l_macro}RmseStd}}{{{fmt(cur['rmse_std'])}}}",
                        rf"\newcommand{{\{pre}{s_macro}{l_macro}Nasa}}{{{fmt(cur['nasa_score_mean'], 1)}}}",
                        rf"\newcommand{{\{pre}{s_macro}{l_macro}NasaStd}}{{{fmt(cur['nasa_score_std'], 1)}}}",
                        rf"\newcommand{{\{pre}{s_macro}{l_macro}Late}}{{{fmt(cur['late_rate_pct_mean'], 1)}}}",
                        rf"\newcommand{{\{pre}{s_macro}{l_macro}LateStd}}{{{fmt(cur['late_rate_pct_std'], 1)}}}",
                    ]
                if j < len(setup_rows) - 1:
                    asym_tex.append(r'\cmidrule(lr){2-7}')
            if i == 0:
                asym_tex.append(r'\midrule')
    asym_tex.append(r'\bottomrule\end{tabular}')
    (OUT / 'table_asymmetric.tex').write_text('\n'.join(asym_tex) + '\n', encoding='utf-8')
    (OUT / 'macros.tex').write_text('% auto-generated by make_figures.py\n' + '\n'.join(macros) + '\n', encoding='utf-8')
    print(f'generated five-seed tables from {SOURCE if SOURCE.exists() else "no summary data"}')


if __name__=='__main__': main()
