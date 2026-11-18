#!/usr/bin/env python3
"""Generate paper tables and scalar macros from committed result artifacts."""
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'results' / 'reproducible' / 'summary.csv'
ASYM_SOURCE = ROOT / 'results' / 'asymmetric_loss_comparison.json'
EDGE_SOURCE = ROOT / 'results' / 'edge' / 'local-smoke-macos.json'
DIAG_SOURCE = ROOT / 'results' / 'diagnostics_validation.json'
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


DIAG_SEEDS = [7, 42, 100, 777, 2024]
DIAG_SUBSETS = ['FD001', 'FD002', 'FD003', 'FD004']
DIAG_SINGLE = ('FD001', 'FD002')
DIAG_TWO = ('FD003', 'FD004')
DIAG_SUBSYSTEMS = ('Fan', 'LPC', 'HPC', 'Turbine')
DIAG_SUBSYSTEM_MAP = {
    'Fan': ['s8', 's13', 's15'],
    'LPC': ['s2'],
    'HPC': ['s3', 's7', 's9', 's11', 's14'],
    'Turbine': ['s4', 's12', 's17', 's20', 's21'],
}
DIAG_HPC_REFERENCE = {'FD003': 'FD001', 'FD004': 'FD002'}
DIAG_VIEWS = (('drift', 'Drift'), ('attribution', 'Attr'), ('combined', 'Comb'))
# Table 2 of Saxena et al. (2008); the same names as diagnostics.SENSOR_INFO.
DIAG_SENSOR_SYMBOLS = {
    's2': 'T24', 's3': 'T30', 's4': 'T50', 's7': 'P30', 's8': 'Nf',
    's9': 'Nc', 's11': 'Ps30', 's12': r'$\phi$', 's13': 'NRf', 's14': 'NRc',
    's15': 'BPR', 's17': 'htBleed', 's20': 'W31', 's21': 'W32',
}
DIAG_SIGNATURE_SENSORS = ('s7', 's12', 's15', 's20', 's21', 's8', 's13', 's9', 's14')
DATASET_MACRO = {
    'FD001': 'FdOne', 'FD002': 'FdTwo', 'FD003': 'FdThree', 'FD004': 'FdFour',
}


def mean_sd(values):
    values = [float(v) for v in values]
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values), sd


def check_summary(summary, values, description, tolerance=1e-3):
    """The committed summary must agree with the per-seed records it rounds."""
    mean, sd = mean_sd(values)
    if (abs(summary['mean'] - mean) > tolerance
            or abs(summary['std'] - sd) > tolerance):
        raise ValueError(
            f'{description}: summary {summary["mean"]}+-{summary["std"]} '
            f'disagrees with per-seed {mean}+-{sd}'
        )
    return mean, sd


def pm(mean, sd, decimals=1):
    """Math-mode mean +- sd; a zero sample sd (e.g. model-free) prints the mean only."""
    if round(sd, decimals) == 0:
        return f'${fmt(mean, decimals)}$'
    return f'${fmt(mean, decimals)}\\pm{fmt(sd, decimals)}$'


def signed(value, decimals=1):
    text = fmt(abs(value), decimals)
    if round(value, decimals) == 0:
        return f'${text}$'
    return f'${"+" if value > 0 else "-"}{text}$'


def diagnostics_artifacts(diag):
    """Validate results/diagnostics_validation.json and emit tables + macros."""
    config = diag.get('config', {})
    expected_config = {
        'seeds': DIAG_SEEDS,
        'subsets': DIAG_SUBSETS,
        'epochs': 8,
        'loss': 'asymmetric',
        'window': 30,
        'rul_cap': 125,
        'subsystem_map': DIAG_SUBSYSTEM_MAP,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(
                f'unexpected diagnostics config {key}: {config.get(key)}'
            )
    require_keys(diag, ('environment', 'network_audit', 'subsets',
                        'evaluation_boundary'),
                 'results/diagnostics_validation.json')
    network = diag['network_audit']
    if (network.get('total_socket_calls_during_diagnostics') != 0
            or network.get('raw_sensor_bytes_sent') != 0):
        raise ValueError(f'diagnostics network audit is not clean: {network}')
    subsets = diag['subsets']
    require_keys(subsets, DIAG_SUBSETS, 'diagnostics subsets')
    for name in DIAG_SUBSETS:
        result = subsets[name]
        require_keys(result, ('per_seed', 'latency', 'summary', 'test_rmse',
                              'baseline_fit_network_socket_calls'),
                     f'diagnostics {name}')
        seeds = [row['seed'] for row in result['per_seed']]
        if seeds != DIAG_SEEDS:
            raise ValueError(f'diagnostics {name} has seeds {seeds}')
        socket_calls = result['baseline_fit_network_socket_calls'] + sum(
            row['network_socket_calls'] for row in result['per_seed'])
        bytes_sent = sum(row['network_bytes_sent'] for row in result['per_seed'])
        if socket_calls or bytes_sent or result['latency']['network_socket_calls']:
            raise ValueError(f'diagnostics {name} recorded network activity')
        if result['latency']['threads'] != 1:
            raise ValueError(f'diagnostics {name} latency was not single-thread')

    n_sensors = sum(len(v) for v in DIAG_SUBSYSTEM_MAP.values())
    chance_hpc = len(DIAG_SUBSYSTEM_MAP['HPC']) / n_sensors
    chance_core = (len(DIAG_SUBSYSTEM_MAP['HPC'])
                   + len(DIAG_SUBSYSTEM_MAP['Turbine'])) / n_sensors
    for name in DIAG_SINGLE:
        chance = subsets[name]['summary']['chance_rates_by_sensor_count']
        if (abs(chance['primary_hpc'] - chance_hpc) > 1e-9
                or abs(chance['primary_core'] - chance_core) > 1e-9):
            raise ValueError(f'diagnostics {name} chance rates disagree')

    environment = diag['environment']
    macros = [
        r'\newcommand{\DiagSeeds}{' + ', '.join(map(str, DIAG_SEEDS)) + '}',
        rf'\newcommand{{\DiagNumSeeds}}{{{len(DIAG_SEEDS)}}}',
        rf'\newcommand{{\DiagEpochs}}{{{config["epochs"]}}}',
        rf'\newcommand{{\DiagMachine}}{{{esc(environment["machine"])}}}',
        rf'\newcommand{{\DiagSystem}}{{{esc(environment["system"])}}}',
        rf'\newcommand{{\DiagPython}}{{{esc(environment["python"])}}}',
        rf'\newcommand{{\DiagTorch}}{{{esc(environment["torch"])}}}',
        rf'\newcommand{{\DiagChanceHpc}}{{{fmt(100 * chance_hpc, 1)}}}',
        rf'\newcommand{{\DiagChanceCore}}{{{fmt(100 * chance_core, 1)}}}',
        rf'\newcommand{{\DiagSocketCalls}}{{{network["total_socket_calls_during_diagnostics"]}}}',
        rf'\newcommand{{\DiagBytesSent}}{{{network["raw_sensor_bytes_sent"]}}}',
    ]

    # ---------------------------------------------------------- single fault
    single = {}
    for name in DIAG_SINGLE:
        result = subsets[name]
        rows = [row['baseline_x_gradient'] for row in result['per_seed']]
        grads = [row['gradient'] for row in result['per_seed']]
        summary = result['summary']
        out = {}
        for metric in ('attr_primary_hpc_rate', 'attr_primary_core_rate',
                       'attr_top3_core_sensor_fraction',
                       'drift_top3_core_sensor_fraction'):
            out[metric] = check_summary(
                summary['baseline_x_gradient'][metric],
                [row[metric] for row in rows], f'{name} {metric}')
        for metric in ('attr_primary_hpc_rate', 'attr_primary_core_rate'):
            out['grad_' + metric] = check_summary(
                summary['gradient'][metric],
                [row[metric] for row in grads], f'{name} gradient {metric}')
        # The committed summary keeps only seed 7's drift-primary split; the
        # held-out population changes with the seed, so aggregate all seeds.
        for subsystem in DIAG_SUBSYSTEMS:
            out['drift_primary_' + subsystem] = mean_sd(
                [row['drift_primary_distribution'][subsystem] for row in rows])
            out['attr_primary_' + subsystem] = check_summary(
                summary['baseline_x_gradient']['attr_primary_distribution'][subsystem],
                [row['attr_primary_distribution'][subsystem] for row in rows],
                f'{name} attr primary {subsystem}')
            out['drift_abs_z_' + subsystem] = mean_sd(
                [row['drift_mean_abs_z_by_subsystem_rul_lt_30'][subsystem]
                 for row in rows])
            for bucket in ('rul_ge_100', 'rul_lt_30'):
                out[f'share_{bucket}_{subsystem}'] = mean_sd(
                    [row['attr_mean_subsystem_share_by_rul'][bucket][subsystem]
                     for row in rows])
        for sensor in ('s7', 's11', 's12'):
            out['attr_top3_' + sensor] = mean_sd(
                [row['attr_top3_sensor_frequency'][sensor] for row in rows])
        out['rmse'] = check_summary(
            result['test_rmse'],
            [row['test_rmse'] for row in result['per_seed']], f'{name} RMSE')
        windows = {row['n_windows_rul_lt_30'] for row in rows}
        if len(windows) != 1:
            raise ValueError(f'{name} low-RUL window count varies: {windows}')
        out['windows'] = windows.pop()
        heldout = {row['n_heldout_units'] for row in result['per_seed']}
        train_units = {row['n_train_units'] for row in result['per_seed']}
        if len(heldout) != 1 or len(train_units) != 1:
            raise ValueError(f'{name} unit split size varies')
        out['heldout'] = heldout.pop()
        out['train_units'] = train_units.pop()
        single[name] = out

        prefix = r'\Diag' + DATASET_MACRO[name]
        macro_values = {
            'AttrHpc': out['attr_primary_hpc_rate'],
            'AttrCore': out['attr_primary_core_rate'],
            'AttrTopCore': out['attr_top3_core_sensor_fraction'],
            'AttrTurbine': out['attr_primary_Turbine'],
            'GradHpc': out['grad_attr_primary_hpc_rate'],
            'GradCore': out['grad_attr_primary_core_rate'],
            'DriftHpc': out['drift_primary_HPC'],
            'DriftFan': out['drift_primary_Fan'],
            'DriftLpc': out['drift_primary_LPC'],
            'DriftTopCore': out['drift_top3_core_sensor_fraction'],
            'TopPhi': out['attr_top3_s12'],
            'TopPsThirty': out['attr_top3_s11'],
            'TopPThirty': out['attr_top3_s7'],
        }
        for suffix, (mean, sd) in macro_values.items():
            macros.append(rf'\newcommand{{{prefix}{suffix}}}{{{fmt(100 * mean, 1)}}}')
            macros.append(rf'\newcommand{{{prefix}{suffix}Std}}{{{fmt(100 * sd, 1)}}}')
        for subsystem in ('HPC', 'Fan', 'Turbine'):
            label = subsystem.capitalize() if subsystem != 'HPC' else 'Hpc'
            macros += [
                rf'\newcommand{{{prefix}AbsZ{label}}}{{{fmt(out["drift_abs_z_" + subsystem][0], 1)}}}',
                rf'\newcommand{{{prefix}Share{label}Early}}{{{fmt(out["share_rul_ge_100_" + subsystem][0], 1)}}}',
                rf'\newcommand{{{prefix}Share{label}Late}}{{{fmt(out["share_rul_lt_30_" + subsystem][0], 1)}}}',
            ]
        macros += [
            rf'\newcommand{{{prefix}Rmse}}{{{fmt(out["rmse"][0])}}}',
            rf'\newcommand{{{prefix}RmseStd}}{{{fmt(out["rmse"][1])}}}',
            rf'\newcommand{{{prefix}Windows}}{{{fmt(out["windows"], 0)}}}',
            rf'\newcommand{{{prefix}HeldoutUnits}}{{{out["heldout"]}}}',
            rf'\newcommand{{{prefix}TrainUnits}}{{{out["train_units"]}}}',
        ]

    def pct_cells(metric, chance=None):
        cells = [pm(100 * single[name][metric][0], 100 * single[name][metric][1])
                 for name in DIAG_SINGLE]
        cells.append('---' if chance is None else f'${fmt(100 * chance, 1)}$')
        return ' & '.join(cells)

    lines = [
        r'\begin{tabular}{@{}lrrr@{}}', r'\toprule',
        r'Low-RUL windows ($y<30$) & FD001 & FD002 & Chance \\',
        r'\midrule',
        'Windows per seed & ' + ' & '.join(
            f'${fmt(single[name]["windows"], 0)}$' for name in DIAG_SINGLE)
        + r' & --- \\',
        r'\multicolumn{4}{l}{\emph{Attribution, $(\mathbf{x}-\mathbf{b})\odot\nabla$ (\%)}} \\',
        r'\quad primary $=$ HPC & ' + pct_cells('attr_primary_hpc_rate', chance_hpc) + r' \\',
        r'\quad primary $\in$ core & ' + pct_cells('attr_primary_core_rate', chance_core) + r' \\',
        r'\quad top-3 sensors in core & ' + pct_cells('attr_top3_core_sensor_fraction', chance_core) + r' \\',
        r'\multicolumn{4}{l}{\emph{Plain saliency, $|\nabla|$ (\%)}} \\',
        r'\quad primary $=$ HPC & ' + pct_cells('grad_attr_primary_hpc_rate', chance_hpc) + r' \\',
        r'\quad primary $\in$ core & ' + pct_cells('grad_attr_primary_core_rate', chance_core) + r' \\',
        r'\multicolumn{4}{l}{\emph{Model-free drift (\%)}} \\',
        r'\quad primary $=$ HPC & ' + pct_cells('drift_primary_HPC', chance_hpc) + r' \\',
        r'\quad primary $=$ Fan & ' + pct_cells(
            'drift_primary_Fan', len(DIAG_SUBSYSTEM_MAP['Fan']) / n_sensors) + r' \\',
        r'\quad top-3 sensors in core & ' + pct_cells('drift_top3_core_sensor_fraction', chance_core) + r' \\',
        r'\midrule',
        'Test RMSE (cycles) & ' + ' & '.join(
            pm(*single[name]['rmse'], decimals=2) for name in DIAG_SINGLE)
        + r' & --- \\',
        r'\bottomrule', r'\end{tabular}',
    ]
    (OUT / 'table_diag_single.tex').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')

    # ---------------------------------------------------------- two faults
    two = {}
    for name in DIAG_TWO:
        result = subsets[name]
        summary = result['summary']
        if summary['hpc_reference']['subset'] != DIAG_HPC_REFERENCE[name]:
            raise ValueError(f'{name} HPC reference is not {DIAG_HPC_REFERENCE[name]}')
        out = {}
        for view, _ in DIAG_VIEWS:
            per = [row['clustering'][view] for row in result['per_seed']]
            view_summary = summary[view]
            out[(view, 'silhouette')] = check_summary(
                view_summary['silhouette'], [p['silhouette'] for p in per],
                f'{name} {view} silhouette')
            out[(view, 'gmm')] = check_summary(
                view_summary['kmeans_vs_gmm_ari'],
                [p['kmeans_vs_gmm_ari'] for p in per], f'{name} {view} GMM ARI')
            pairs = view_summary['cross_seed_ari']['per_seed']
            n_pairs = len(DIAG_SEEDS) * (len(DIAG_SEEDS) - 1) // 2
            if len(pairs) != n_pairs:
                raise ValueError(f'{name} {view} has {len(pairs)} seed pairs')
            out[(view, 'seed_ari')] = check_summary(
                view_summary['cross_seed_ari'], pairs, f'{name} {view} seed ARI')
            fractions = [
                p['fan_candidate']['n_units']
                / (p['fan_candidate']['n_units'] + p['hpc_like']['n_units'])
                for p in per
            ]
            out[(view, 'fan_fraction')] = check_summary(
                view_summary['fan_candidate_fraction'], fractions,
                f'{name} {view} fan fraction')
            for cluster in ('hpc_like', 'fan_candidate'):
                out[(view, cluster, 'cos')] = check_summary(
                    view_summary[cluster]['cosine_to_hpc_reference'],
                    [p[cluster]['cosine_to_hpc_reference'] for p in per],
                    f'{name} {view} {cluster} cosine')
                for sensor in DIAG_SENSOR_SYMBOLS:
                    out[(view, cluster, sensor)] = mean_sd(
                        [p[cluster]['drift_signed_z'][sensor] for p in per])
                for subsystem in DIAG_SUBSYSTEMS:
                    out[(view, cluster, 'share', subsystem)] = mean_sd(
                        [p[cluster]['attr_subsystem_share_pct'][subsystem]
                         for p in per])
                    out[(view, cluster, 'absz', subsystem)] = mean_sd(
                        [p[cluster]['drift_mean_abs_z'][subsystem] for p in per])
        out['cross_view'] = check_summary(
            summary['cross_view_ari_drift_vs_attribution'],
            [row['clustering']['cross_view_ari_drift_vs_attribution']
             for row in result['per_seed']], f'{name} cross-view ARI')
        out['reference'] = summary['hpc_reference']
        out['rmse'] = check_summary(
            result['test_rmse'],
            [row['test_rmse'] for row in result['per_seed']], f'{name} RMSE')
        counts = {(row['n_units_clustered']['train'],
                   row['n_units_clustered']['test_rul_le_30'])
                  for row in result['per_seed']}
        if len(counts) != 1:
            raise ValueError(f'{name} clustered population varies: {counts}')
        out['units'] = counts.pop()
        two[name] = out

        prefix = r'\Diag' + DATASET_MACRO[name]
        for view, label in DIAG_VIEWS:
            for key, suffix in (('silhouette', 'Sil'), ('gmm', 'GmmAri'),
                                ('seed_ari', 'SeedAri')):
                mean, sd = out[(view, key)]
                macros.append(rf'\newcommand{{{prefix}{label}{suffix}}}{{{fmt(mean)}}}')
                macros.append(rf'\newcommand{{{prefix}{label}{suffix}Std}}{{{fmt(sd)}}}')
            macros.append(
                rf'\newcommand{{{prefix}{label}FanFrac}}'
                rf'{{{fmt(100 * out[(view, "fan_fraction")][0], 1)}}}')
            macros.append(
                rf'\newcommand{{{prefix}{label}CosHpc}}'
                rf'{{{fmt(out[(view, "hpc_like", "cos")][0])}}}')
            macros.append(
                rf'\newcommand{{{prefix}{label}CosFan}}'
                rf'{{{fmt(out[(view, "fan_candidate", "cos")][0])}}}')
        fan_ratio = [
            out[('drift', 'fan_candidate', sensor)][0]
            / out[('drift', 'hpc_like', sensor)][0]
            for sensor in ('s8', 's13')
        ]
        macros += [
            rf'\newcommand{{{prefix}CrossViewAri}}{{{fmt(out["cross_view"][0])}}}',
            rf'\newcommand{{{prefix}CrossViewAriStd}}{{{fmt(out["cross_view"][1])}}}',
            rf'\newcommand{{{prefix}Rmse}}{{{fmt(out["rmse"][0])}}}',
            rf'\newcommand{{{prefix}RmseStd}}{{{fmt(out["rmse"][1])}}}',
            rf'\newcommand{{{prefix}TrainUnits}}{{{out["units"][0]}}}',
            rf'\newcommand{{{prefix}TestUnits}}{{{out["units"][1]}}}',
            rf'\newcommand{{{prefix}FanSpeedRatio}}{{{fmt(statistics.fmean(fan_ratio), 1)}}}',
        ]
        for cluster, label in (('hpc_like', 'Hpc'), ('fan_candidate', 'Cand')):
            for subsystem in ('Fan', 'HPC', 'Turbine'):
                sub_label = subsystem.capitalize() if subsystem != 'HPC' else 'Hpc'
                macros.append(
                    rf'\newcommand{{{prefix}{label}Share{sub_label}}}'
                    rf'{{{fmt(out[("drift", cluster, "share", subsystem)][0], 1)}}}')
                macros.append(
                    rf'\newcommand{{{prefix}{label}AbsZ{sub_label}}}'
                    rf'{{{fmt(out[("drift", cluster, "absz", subsystem)][0], 1)}}}')
            for sensor, sensor_label in (('s7', 'PThirty'), ('s12', 'Phi'),
                                         ('s15', 'Bpr')):
                macros.append(
                    rf'\newcommand{{{prefix}{label}Z{sensor_label}}}'
                    rf'{{{signed(out[("drift", cluster, sensor)][0])}}}')

    lines = [
        r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
        r'View & Metric & FD003 & FD004 \\',
        r'\midrule',
        'Units & train / test ($y\\le30$) & ' + ' & '.join(
            f'{two[name]["units"][0]} / {two[name]["units"][1]}'
            for name in DIAG_TWO) + r' \\',
        r'HPC ref. & label-free & ' + ' & '.join(
            two[name]['reference']['subset'] for name in DIAG_TWO) + r' \\',
    ]
    view_labels = {
        'drift': 'Drift', 'attribution': 'Attribution', 'combined': 'Combined',
    }
    for view, _ in DIAG_VIEWS:
        lines.append(r'\midrule')
        metric_rows = (
            ('silhouette', 'silhouette', 2, 1.0),
            ('seed_ari', 'cross-seed ARI', 2, 1.0),
            ('gmm', 'KMeans vs.\\ GMM ARI', 2, 1.0),
            ('fan_fraction', 'fan-cand.\\ fraction (\\%)', 1, 100.0),
        )
        for index, (key, label, decimals, scale) in enumerate(metric_rows):
            head = view_labels[view] if index == 0 else ''
            cells = ' & '.join(
                pm(scale * two[name][(view, key)][0],
                   scale * two[name][(view, key)][1], decimals)
                for name in DIAG_TWO)
            lines.append(f'{head} & {label} & {cells} \\\\')
        cells = ' & '.join(
            f'${fmt(two[name][(view, "hpc_like", "cos")][0])}$ / '
            f'${fmt(two[name][(view, "fan_candidate", "cos")][0])}$'
            for name in DIAG_TWO)
        lines.append(f' & cos.\\ to ref.\\ (HPC-l.\\,/\\,cand.) & {cells} \\\\')
    lines += [
        r'\midrule',
        r'Drift vs.\ attr. & cross-view ARI & ' + ' & '.join(
            pm(*two[name]['cross_view'], decimals=2) for name in DIAG_TWO) + r' \\',
        'Model & test RMSE (cycles) & ' + ' & '.join(
            pm(*two[name]['rmse'], decimals=2) for name in DIAG_TWO) + r' \\',
        r'\bottomrule', r'\end{tabular}',
    ]
    (OUT / 'table_diag_clusters.tex').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')

    lines = [
        r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
        r' & \multicolumn{3}{c}{FD003} & \multicolumn{3}{c}{FD004} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
        r'Sensor & ref. & HPC-l. & cand. & ref. & HPC-l. & cand. \\',
        r'\midrule',
    ]
    for sensor in DIAG_SIGNATURE_SENSORS:
        cells = []
        for name in DIAG_TWO:
            cells.append(signed(two[name]['reference']['eol_signed_drift_z'][sensor]))
            cells.append(signed(two[name][('drift', 'hpc_like', sensor)][0]))
            cells.append(signed(two[name][('drift', 'fan_candidate', sensor)][0]))
        lines.append(
            f'{DIAG_SENSOR_SYMBOLS[sensor]} ({sensor}) & ' + ' & '.join(cells) + r' \\')
    signature_max_sd = max(
        two[name][('drift', cluster, sensor)][1]
        for name in DIAG_TWO
        for cluster in ('hpc_like', 'fan_candidate')
        for sensor in DIAG_SIGNATURE_SENSORS
    )
    macros.append(
        rf'\newcommand{{\DiagSignatureMaxStd}}{{{fmt(signature_max_sd, 2)}}}')
    lines += [r'\midrule']
    for subsystem in ('Fan', 'HPC', 'Turbine'):
        cells = []
        for name in DIAG_TWO:
            cells.append('---')
            for cluster in ('hpc_like', 'fan_candidate'):
                cells.append(
                    f'${fmt(two[name][("drift", cluster, "share", subsystem)][0], 1)}$')
        lines.append(f'{subsystem} share (\\%) & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'table_diag_signatures.tex').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')

    # ---------------------------------------------------------- overhead
    latency = {name: subsets[name]['latency'] for name in DIAG_SUBSETS}
    saved = {latency[name]['autograd_saved_tensor_bytes_per_window']
             for name in DIAG_SUBSETS}
    window_bytes = {latency[name]['input_window_bytes'] for name in DIAG_SUBSETS}
    timed = {latency[name]['windows_timed'] for name in DIAG_SUBSETS}
    if len(saved) != 1 or len(window_bytes) != 1 or len(timed) != 1:
        raise ValueError('diagnostics latency byte/window counts vary by subset')
    saved, window_bytes, timed = saved.pop(), window_bytes.pop(), timed.pop()
    latency_rows = (
        ('inference_only_ms_per_window', 'Inference only'),
        ('inference_attribution_summary_ms_per_window', r'+ attribution + summary'),
        ('inference_attribution_drift_summary_ms_per_window',
         r'+ drift + attribution + summary'),
    )
    lines = [
        r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
        r'Per window (ms) & FD001 & FD002 & FD003 & FD004 \\',
        r'\midrule',
    ]
    for key, label in latency_rows:
        lines.append(f'{label} & ' + ' & '.join(
            f'${latency[name][key]:.3f}$' for name in DIAG_SUBSETS) + r' \\')
    lines.append(r'Full / inference ratio & ' + ' & '.join(
        f'${latency[name]["overhead_ratio"]:.1f}\\times$' for name in DIAG_SUBSETS)
        + r' \\')
    lines += [
        r'\midrule',
        r'Autograd saved bytes & \multicolumn{4}{r}{' + f'${fmt(saved, 0)}$' + r'} \\',
        r'Input window bytes & \multicolumn{4}{r}{' + f'${fmt(window_bytes, 0)}$' + r'} \\',
        r'Socket calls / raw bytes sent & \multicolumn{4}{r}{'
        + f'${network["total_socket_calls_during_diagnostics"]}$ / '
        + f'${network["raw_sensor_bytes_sent"]}$' + r'} \\',
        r'\bottomrule', r'\end{tabular}',
    ]
    (OUT / 'table_diag_latency.tex').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')

    inference = [latency[n]['inference_only_ms_per_window'] for n in DIAG_SUBSETS]
    attribution = [latency[n]['inference_attribution_summary_ms_per_window']
                   for n in DIAG_SUBSETS]
    full = [latency[n]['inference_attribution_drift_summary_ms_per_window']
            for n in DIAG_SUBSETS]
    ratio = [latency[n]['overhead_ratio'] for n in DIAG_SUBSETS]
    macros += [
        rf'\newcommand{{\DiagLatInferenceMin}}{{{min(inference):.3f}}}',
        rf'\newcommand{{\DiagLatInferenceMax}}{{{max(inference):.3f}}}',
        rf'\newcommand{{\DiagLatAttrMin}}{{{min(attribution):.3f}}}',
        rf'\newcommand{{\DiagLatAttrMax}}{{{max(attribution):.3f}}}',
        rf'\newcommand{{\DiagLatFullMin}}{{{min(full):.3f}}}',
        rf'\newcommand{{\DiagLatFullMax}}{{{max(full):.3f}}}',
        rf'\newcommand{{\DiagLatRatioMin}}{{{min(ratio):.1f}}}',
        rf'\newcommand{{\DiagLatRatioMax}}{{{max(ratio):.1f}}}',
        rf'\newcommand{{\DiagSavedBytes}}{{{fmt(saved, 0)}}}',
        rf'\newcommand{{\DiagWindowBytes}}{{{fmt(window_bytes, 0)}}}',
        rf'\newcommand{{\DiagSavedToWindowRatio}}{{{fmt(saved / window_bytes, 1)}}}',
        rf'\newcommand{{\DiagWindowsTimed}}{{{timed}}}',
    ]
    return macros


def main():
    for path, description in (
        (SOURCE, 'paired-suite summary'),
        (ASYM_SOURCE, 'asymmetric-loss comparison'),
        (EDGE_SOURCE, 'edge smoke profile'),
        (DIAG_SOURCE, 'edge diagnostics validation'),
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

    diag_data = json.loads(DIAG_SOURCE.read_text(encoding='utf-8'))
    macros += diagnostics_artifacts(diag_data)

    asym_tex.append(r'\bottomrule\end{tabular}')
    (OUT / 'table_asymmetric.tex').write_text(
        '\n'.join(asym_tex) + '\n', encoding='utf-8'
    )
    (OUT / 'macros.tex').write_text(
        '% auto-generated by make_figures.py\n' + '\n'.join(macros) + '\n',
        encoding='utf-8',
    )
    print(
        f'generated paper artifacts from {SOURCE}, {ASYM_SOURCE}, '
        f'{EDGE_SOURCE}, and {DIAG_SOURCE}'
    )


if __name__ == '__main__':
    main()
