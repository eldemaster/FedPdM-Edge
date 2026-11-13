#!/usr/bin/env python3
"""Generate paper tables only from five-seed rows in results/reproducible."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'results' / 'reproducible' / 'summary.csv'
OUT = Path(__file__).resolve().parent / 'generated'
OUT.mkdir(exist_ok=True)


def fmt(value):
    return f'{float(value):,.2f}'.replace(',', r'\,')


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
    (OUT/'macros.tex').write_text('% No scalar macros: all manuscript values are generated in tables.\n',encoding='utf-8')
    print(f'generated five-seed tables from {SOURCE if SOURCE.exists() else "no summary data"}')


if __name__=='__main__': main()
