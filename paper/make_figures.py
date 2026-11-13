#!/usr/bin/env python3
"""Generate paper tables and scalar macros from committed result artifacts."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'results' / 'reproducible' / 'summary.csv'
ASYM_SOURCE = ROOT / 'results' / 'asymmetric_loss_comparison.json'
EDGE_SOURCE = ROOT / 'results' / 'edge' / 'local-smoke-macos.json'
OUT = Path(__file__).resolve().parent / 'generated'
OUT.mkdir(exist_ok=True)

METHOD_LABELS = {
    'centralized': 'Centralized',
    'local_only': 'Local only',
    'fedavg': 'FedAvg',
    'fedprox': 'FedProx',
    'fedadam': 'FedAdam',
    'federated_asymmetric': r'FedAvg + exp.',
    'asymmetric': r'Central + exp.',
    'cnn_lstm': 'CNN-LSTM',
}


def fmt(value, decimals=2):
    return f'{float(value):,.{decimals}f}'.replace(',', r'\,')


def fmt_table(value, decimals=2):
    """Keep very large NASA-score values compact enough for IEEE tables."""
    value = float(value)
    if abs(value) >= 1_000_000:
        exponent = int(math.floor(math.log10(abs(value))))
        coefficient = value / (10 ** exponent)
        return rf'{coefficient:.{decimals}f}\times10^{{{exponent}}}'
    return fmt(value, decimals)


def esc(value):
    return str(value).replace('_', r'\_')


def method_label(method):
    return METHOD_LABELS.get(method, esc(method))


def require_keys(mapping, keys, description):
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise ValueError(f'{description} is incomplete; missing {missing}')


def main():
    for path, description in (
        (SOURCE, 'paired-suite summary'),
        (ASYM_SOURCE, 'asymmetric-loss comparison'),
        (EDGE_SOURCE, 'edge smoke profile'),
    ):
        if not path.exists():
            raise FileNotFoundError(f'missing {description}: {path}')

    records = {}
    clients = {}
    tests = []
    with SOURCE.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row.get('comparison') and row.get('paired_sign_flip_p'):
                tests.append(row)
            if not row.get('method') or row.get('metric') not in ('rmse', 'nasa'):
                continue
            if int(row['n']) < 5:
                continue
            if row.get('scope') == 'global':
                key = (row['dataset'], row['partition'], row['method'], row['metric'])
                records[key] = row
            elif row.get('scope') == 'client':
                key = (row['dataset'], row['partition'], row['method'],
                       int(row['client_id']), row['metric'])
                clients[key] = row

    main_groups = [(f'FD00{i}', 'iid') for i in range(1, 5)]
    main_methods = ('centralized', 'local_only', 'fedavg')
    algorithm_groups = [('FD002', 'regime'), ('FD004', 'regime')]
    algorithm_methods = (
        'centralized', 'local_only', 'fedavg', 'fedprox', 'fedadam',
        'federated_asymmetric', 'asymmetric', 'cnn_lstm',
    )
    partition_specs = []
    for dataset in ('FD002', 'FD004'):
        partition_specs.append((dataset, 'iid', 'fedavg'))
        for partition in ('regime', 'engine_imbalance'):
            partition_specs.extend(
                (dataset, partition, method)
                for method in ('fedavg', 'fedprox', 'fedadam')
            )

    required_records = {
        (dataset, partition, method, metric)
        for dataset, partition in main_groups
        for method in main_methods
        for metric in ('rmse', 'nasa')
    }
    required_records |= {
        (dataset, partition, method, metric)
        for dataset, partition in algorithm_groups
        for method in algorithm_methods
        for metric in ('rmse', 'nasa')
    }
    required_records |= {
        (dataset, partition, method, metric)
        for dataset, partition, method in partition_specs
        for metric in ('rmse', 'nasa')
    }
    require_keys(records, required_records,
                 'results/reproducible/summary.csv global rows')

    required_clients = {
        (dataset, 'regime', method, client_id, metric)
        for dataset in ('FD002', 'FD004')
        for method in algorithm_methods
        for client_id in (0, 1)
        for metric in ('rmse', 'nasa')
    }
    require_keys(clients, required_clients,
                 'results/reproducible/summary.csv client rows')

    asym_data = json.loads(ASYM_SOURCE.read_text(encoding='utf-8'))
    expected_asym_config = {
        'seeds': [42, 123, 456, 789, 1024],
        'subsets': ['FD001', 'FD002'],
        'epochs': 10,
        'rounds': 5,
        'local_epochs': 2,
        'alpha_late': 2.5,
        'alpha_early': 0.5,
    }
    if asym_data.get('config') != expected_asym_config:
        raise ValueError(
            f'unexpected asymmetric-loss config: {asym_data.get("config")}'
        )
    require_keys(asym_data.get('subsets', {}),
                 expected_asym_config['subsets'],
                 'results/asymmetric_loss_comparison.json subsets')
    for dataset in expected_asym_config['subsets']:
        losses = asym_data['subsets'][dataset].get('losses', {})
        require_keys(losses, ('mse', 'asymmetric'),
                     f'{dataset} asymmetric-loss variants')
        for loss_name in ('mse', 'asymmetric'):
            require_keys(losses[loss_name].get('summary', {}),
                         ('local', 'fedavg', 'central'),
                         f'{dataset}/{loss_name} summaries')
            actual_seeds = sorted(map(int, losses[loss_name].get('per_seed', {})))
            if actual_seeds != sorted(expected_asym_config['seeds']):
                raise ValueError(
                    f'{dataset}/{loss_name} has seeds {actual_seeds}'
                )

    edge_data = json.loads(EDGE_SOURCE.read_text(encoding='utf-8'))
    required_edge_fields = (
        'device', 'dataset', 'partition', 'client', 'rounds', 'local_epochs',
        'train_seconds_mean', 'peak_process_rss_bytes',
        'communication_tensor_payload_bytes_per_round',
    )
    require_keys(edge_data, required_edge_fields,
                 'results/edge/local-smoke-macos.json')

    def performance_table(filename, specs):
        lines = [
            r'\begin{tabular}{lllrrr}', r'\toprule',
            r'Dataset & Partition & Method & $n$ & RMSE & NASA score \\',
            r'\midrule',
        ]
        for dataset, partition, method in specs:
            rmse = records[(dataset, partition, method, 'rmse')]
            nasa = records[(dataset, partition, method, 'nasa')]
            lines.append(
                f'{dataset} & {esc(partition)} & {method_label(method)} & '
                f'{rmse["n"]} & '
                f'${fmt_table(rmse["mean"])}\\pm{fmt_table(rmse["std"])}$ & '
                f'${fmt_table(nasa["mean"])}\\pm{fmt_table(nasa["std"])}$ \\\\'
            )
        lines += [r'\bottomrule', r'\end{tabular}']
        (OUT / filename).write_text('\n'.join(lines) + '\n', encoding='utf-8')

    performance_table(
        'table_main.tex',
        [(dataset, partition, method)
         for dataset, partition in main_groups for method in main_methods],
    )
    performance_table(
        'table_algorithms.tex',
        [(dataset, partition, method)
         for dataset, partition in algorithm_groups
         for method in algorithm_methods],
    )
    performance_table('table_partitions.tex', partition_specs)

    lines = [
        r'\begin{tabular}{llllrr}', r'\toprule',
        r'Dataset & Partition & Method & Client & RMSE & NASA score \\',
        r'\midrule',
    ]
    for dataset, partition in algorithm_groups:
        for method in algorithm_methods:
            for client_id in (0, 1):
                rmse = clients[(dataset, partition, method, client_id, 'rmse')]
                nasa = clients[(dataset, partition, method, client_id, 'nasa')]
                lines.append(
                    f'{dataset} & {esc(partition)} & {method_label(method)} & '
                    f'{client_id + 1} & '
                    f'${fmt_table(rmse["mean"])}\\pm{fmt_table(rmse["std"])}$ & '
                    f'${fmt_table(nasa["mean"])}\\pm{fmt_table(nasa["std"])}$ \\\\'
                )
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'table_clients.tex').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )

    key_comparisons = ('fedavg_vs_centralized', 'fedavg_vs_local_only')
    method_comparisons = key_comparisons + (
        'fedavg_vs_fedprox', 'fedavg_vs_fedadam',
        'fedavg_vs_federated_asymmetric', 'centralized_vs_asymmetric',
        'centralized_vs_cnn_lstm',
    )
    selected_tests = []
    for row in tests:
        primary_regime = (
            row['dataset'] in ('FD002', 'FD004')
            and row['partition'] == 'regime'
            and row['comparison'] in method_comparisons
        )
        baseline_fd001 = (
            row['dataset'] == 'FD001'
            and row['partition'] == 'iid'
            and row['comparison'] in key_comparisons
        )
        if (primary_regime or baseline_fd001) and int(row['n']) >= 5:
            selected_tests.append(row)
    if len(selected_tests) != 32:
        raise ValueError(
            f'expected 32 selected paired-test rows, found {len(selected_tests)}'
        )

    lines = [
        r'\begin{tabular}{llllr}', r'\toprule',
        r'Dataset & Partition & Comparison & Metric & paired $p$ \\',
        r'\midrule',
    ]
    for row in selected_tests:
        left, right = row['comparison'].split('_vs_', 1)
        comparison = rf'{method_label(left)} vs. {method_label(right)}'
        lines.append(
            f'{row["dataset"]} & {esc(row["partition"])} & {comparison} & '
            f'{row["metric"].upper()} & '
            f'{float(row["paired_sign_flip_p"]):.4f} \\\\'
        )
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'table_significance.tex').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8'
    )

    central_iid_rmse = [
        float(records[(dataset, 'iid', 'centralized', 'rmse')]['mean'])
        for dataset in ('FD001', 'FD002', 'FD003', 'FD004')
    ]
    macros = [
        rf'\newcommand{{\SuiteCentralIidRmseMin}}{{{fmt(min(central_iid_rmse))}}}',
        rf'\newcommand{{\SuiteCentralIidRmseMax}}{{{fmt(max(central_iid_rmse))}}}',
        rf'\newcommand{{\SuiteFdTwoRegimeFedAdamRmse}}{{{fmt(records[("FD002", "regime", "fedadam", "rmse")]["mean"])}}}',
        rf'\newcommand{{\SuiteFdFourRegimeFedAdamRmse}}{{{fmt(records[("FD004", "regime", "fedadam", "rmse")]["mean"])}}}',
        rf'\newcommand{{\EdgeSmokeTrainSeconds}}{{{fmt(edge_data["train_seconds_mean"], 3)}}}',
        rf'\newcommand{{\EdgeSmokePeakRssMiB}}{{{fmt(edge_data["peak_process_rss_bytes"] / (1024 ** 2), 1)}}}',
        rf'\newcommand{{\EdgeSmokePayloadBytes}}{{{fmt(edge_data["communication_tensor_payload_bytes_per_round"], 0)}}}',
    ]

    asym_tex = [
        r'\begin{tabular}{llcccrr}\toprule',
        r'Dataset & Setup & Loss & RMSE (cycles) & NASA score & $\Delta$ NASA & Late rate (\%) \\\midrule',
    ]
    setup_rows = [
        ('local', 'Isolated local'),
        ('fedavg', r'FedAvg (5r $\times$ 2e)'),
        ('central', 'Centralized (10e)'),
    ]
    all_mse_late = []
    all_asym_late = []
    decentralized_nasa_reductions = []
    fd002_rmse_increases = []
    for dataset_index, dataset in enumerate(('FD001', 'FD002')):
        prefix = 'AsymFdOne' if dataset == 'FD001' else 'AsymFdTwo'
        losses = asym_data['subsets'][dataset]['losses']
        for setup_index, (setup, setup_label) in enumerate(setup_rows):
            setup_macro = {
                'local': 'Local', 'fedavg': 'Fed', 'central': 'Cent',
            }[setup]
            mse_summary = losses['mse']['summary'][setup]
            asym_summary = losses['asymmetric']['summary'][setup]
            all_mse_late.append(float(mse_summary['late_rate_pct_mean']))
            all_asym_late.append(float(asym_summary['late_rate_pct_mean']))
            if setup in ('local', 'fedavg'):
                decentralized_nasa_reductions.append(
                    100.0 * (
                        mse_summary['nasa_score_mean']
                        - asym_summary['nasa_score_mean']
                    ) / mse_summary['nasa_score_mean']
                )
            if dataset == 'FD002':
                fd002_rmse_increases.append(
                    100.0 * (
                        asym_summary['rmse_mean'] - mse_summary['rmse_mean']
                    ) / mse_summary['rmse_mean']
                )

            for loss_name, loss_label, current, other in (
                ('mse', r'\texttt{mse}', mse_summary, asym_summary),
                ('asymmetric', r'\texttt{asymmetric}', asym_summary, mse_summary),
            ):
                loss_macro = 'Mse' if loss_name == 'mse' else 'Asy'
                dataset_column = (
                    rf'\multirow{{6}}{{*}}{{{dataset}}}'
                    if setup_index == 0 and loss_name == 'mse' else ''
                )
                setup_column = (
                    rf'\multirow{{2}}{{*}}{{{setup_label}}}'
                    if loss_name == 'mse' else ''
                )
                rmse_text = (
                    rf"${fmt(current['rmse_mean'])}"
                    rf"\pm{fmt(current['rmse_std'])}$"
                )
                if current['rmse_mean'] <= other['rmse_mean']:
                    rmse_text = rf'\textbf{{{rmse_text}}}'
                nasa_text = (
                    rf"${fmt(current['nasa_score_mean'], 1)}"
                    rf"\pm{fmt(current['nasa_score_std'], 1)}$"
                )
                if current['nasa_score_mean'] <= other['nasa_score_mean']:
                    nasa_text = rf'\textbf{{{nasa_text}}}'
                late_text = (
                    rf"${fmt(current['late_rate_pct_mean'], 1)}"
                    rf"\pm{fmt(current['late_rate_pct_std'], 1)}$"
                )
                if current['late_rate_pct_mean'] <= other['late_rate_pct_mean']:
                    late_text = rf'\textbf{{{late_text}}}'
                if loss_name == 'mse':
                    delta_text = '---'
                else:
                    delta = 100.0 * (
                        current['nasa_score_mean'] - other['nasa_score_mean']
                    ) / other['nasa_score_mean']
                    sign = '+' if delta > 0 else ''
                    delta_text = rf'${sign}{delta:.1f}\%$'
                    macros.append(
                        rf'\newcommand{{\{prefix}{setup_macro}NasaDelta}}'
                        rf'{{{sign}{delta:.1f}\%}}'
                    )
                asym_tex.append(
                    f'{dataset_column} & {setup_column} & {loss_label} & '
                    f'{rmse_text} & {nasa_text} & {delta_text} & '
                    f'{late_text} \\\\'
                )
                macros += [
                    rf"\newcommand{{\{prefix}{setup_macro}{loss_macro}Rmse}}{{{fmt(current['rmse_mean'])}}}",
                    rf"\newcommand{{\{prefix}{setup_macro}{loss_macro}RmseStd}}{{{fmt(current['rmse_std'])}}}",
                    rf"\newcommand{{\{prefix}{setup_macro}{loss_macro}Nasa}}{{{fmt(current['nasa_score_mean'], 1)}}}",
                    rf"\newcommand{{\{prefix}{setup_macro}{loss_macro}NasaStd}}{{{fmt(current['nasa_score_std'], 1)}}}",
                    rf"\newcommand{{\{prefix}{setup_macro}{loss_macro}Late}}{{{fmt(current['late_rate_pct_mean'], 1)}}}",
                    rf"\newcommand{{\{prefix}{setup_macro}{loss_macro}LateStd}}{{{fmt(current['late_rate_pct_std'], 1)}}}",
                ]
            if setup_index < len(setup_rows) - 1:
                asym_tex.append(r'\cmidrule(lr){2-7}')
        if dataset_index == 0:
            asym_tex.append(r'\midrule')

    late_rate_reductions = [
        100.0 * (before - after) / before
        for before, after in zip(all_mse_late, all_asym_late)
    ]
    fd002_fed_mse_std = (
        asym_data['subsets']['FD002']['losses']['mse']['summary']
        ['fedavg']['nasa_score_std']
    )
    fd002_fed_asym_std = (
        asym_data['subsets']['FD002']['losses']['asymmetric']['summary']
        ['fedavg']['nasa_score_std']
    )
    macros += [
        rf'\newcommand{{\AsymMseLateMin}}{{{fmt(min(all_mse_late), 1)}}}',
        rf'\newcommand{{\AsymMseLateMax}}{{{fmt(max(all_mse_late), 1)}}}',
        rf'\newcommand{{\AsymLateMin}}{{{fmt(min(all_asym_late), 1)}}}',
        rf'\newcommand{{\AsymLateMax}}{{{fmt(max(all_asym_late), 1)}}}',
        rf'\newcommand{{\AsymLateReductionMin}}{{{fmt(min(late_rate_reductions), 1)}}}',
        rf'\newcommand{{\AsymLateReductionMax}}{{{fmt(max(late_rate_reductions), 1)}}}',
        rf'\newcommand{{\AsymDecentralNasaReductionMin}}{{{fmt(min(decentralized_nasa_reductions), 1)}}}',
        rf'\newcommand{{\AsymDecentralNasaReductionMax}}{{{fmt(max(decentralized_nasa_reductions), 1)}}}',
        rf'\newcommand{{\AsymFdTwoRmseIncreaseMin}}{{{fmt(min(fd002_rmse_increases), 1)}}}',
        rf'\newcommand{{\AsymFdTwoRmseIncreaseMax}}{{{fmt(max(fd002_rmse_increases), 1)}}}',
        rf'\newcommand{{\AsymFdTwoFedNasaStdRatio}}{{{fmt(fd002_fed_mse_std / fd002_fed_asym_std, 1)}}}',
    ]

    asym_tex.append(r'\bottomrule\end{tabular}')
    (OUT / 'table_asymmetric.tex').write_text(
        '\n'.join(asym_tex) + '\n', encoding='utf-8'
    )
    (OUT / 'macros.tex').write_text(
        '% auto-generated by make_figures.py\n' + '\n'.join(macros) + '\n',
        encoding='utf-8',
    )
    print(
        f'generated paper artifacts from {SOURCE}, {ASYM_SOURCE}, and '
        f'{EDGE_SOURCE}'
    )


if __name__ == '__main__':
    main()
