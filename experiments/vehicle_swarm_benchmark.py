#!/usr/bin/env python3
"""Vehicle-disjoint, partially connected federated C-MAPSS benchmark.

The manufacturer receives model updates and channel-wise sufficient statistics;
raw telemetry and vehicle-level windows stay in the owning customer fleet.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    AsymmetricRULLoss,
    CmapssRegressor,
    MAX_RUL,
    SEQ_LEN,
    USEFUL_SENSORS,
    extract_last_sequence,
    extract_supervised_sequences,
    late_prediction_rate,
    load_cmapss_raw,
    load_rul,
    make_loader,
    nasa_score,
    predict,
    rmse,
    set_seed,
    train_epochs,
)
from diagnostics import FAULT_SENSORS, SETTINGS, fit_healthy_baseline  # noqa: E402
from fed_norm import (  # noqa: E402
    FederatedScaler,
    LocalStats,
    aggregate_federated_stats,
    compute_local_stats,
)

SEEDS = (7, 42, 100, 777, 2024)
PARTITION_SCHEMES = ("iid_swarm", "lifespan_quantity_skew", "regime_fault_skew")
METHODS = (
    "local_only",
    "fedavg_mse",
    "fedprox_mse",
    "fedavg_asym",
    "fedprox_asym",
    "fedavg_asym_personalized",
)
DEFAULT_CONFIGS = (
    {"dataset": "FD002", "k": 10, "fraction_fit": 0.5},
    {"dataset": "FD004", "k": 10, "fraction_fit": 0.5},
    {"dataset": "FD004", "k": 20, "fraction_fit": 0.3},
)


@dataclass(frozen=True)
class VehicleProfile:
    lifespan: int
    dominant_regime: int
    fault_cluster: int


@dataclass
class ClientData:
    client_id: int
    units: tuple[Hashable, ...]
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.train_y))


def _jsonable_id(value: Hashable) -> str | int:
    return int(value) if isinstance(value, (int, np.integer)) else str(value)


def assert_disjoint_partitions(partitions: Mapping[int, Sequence[Hashable]],
                               vehicle_ids: Iterable[Hashable]) -> None:
    """Prove each vehicle occurs exactly once across the K clients."""
    expected = set(vehicle_ids)
    seen: set[Hashable] = set()
    total = 0
    for client_id, units in partitions.items():
        current = set(units)
        if len(current) != len(units):
            raise AssertionError(f"client {client_id} contains duplicate vehicle IDs")
        overlap = seen.intersection(current)
        if overlap:
            raise AssertionError(f"vehicle overlap detected: {sorted(map(str, overlap))[:5]}")
        seen.update(current)
        total += len(units)
    if total != len(expected) or seen != expected:
        missing = expected.difference(seen)
        extra = seen.difference(expected)
        raise AssertionError(f"partition is not exhaustive (missing={missing}, extra={extra})")


def _dirichlet_counts(total: int, k: int, alpha: float,
                      rng: np.random.RandomState, minimum: int = 2) -> np.ndarray:
    if total < k * minimum:
        raise ValueError(f"need at least {minimum} vehicles per client")
    base = np.full(k, minimum, dtype=int)
    remaining = total - int(base.sum())
    if remaining:
        base += rng.multinomial(remaining, rng.dirichlet(np.full(k, alpha)))
    return base


def partition_vehicles(
    vehicle_ids: Sequence[Hashable],
    k: int,
    scheme: str,
    *,
    profiles: Mapping[Hashable, VehicleProfile],
    seed: int,
    alpha_quantity: float = 0.5,
    alpha_profile: float = 0.3,
) -> Dict[int, list[Hashable]]:
    """Create an exhaustive, pairwise-disjoint vehicle allocation."""
    if scheme not in PARTITION_SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    if k <= 0 or len(vehicle_ids) < k:
        raise ValueError("K must be positive and no greater than the number of vehicles")
    if set(vehicle_ids) != set(profiles):
        raise ValueError("profiles must describe exactly the supplied vehicle IDs")
    rng = np.random.RandomState(seed)
    result: Dict[int, list[Hashable]] = {i: [] for i in range(k)}

    if scheme == "iid_swarm":
        shuffled = np.asarray(vehicle_ids, dtype=object).copy()
        rng.shuffle(shuffled)
        for client_id, chunk in enumerate(np.array_split(shuffled, k)):
            result[client_id] = chunk.tolist()

    elif scheme == "lifespan_quantity_skew":
        # Dirichlet controls customer fleet size; contiguous lifespan bands make
        # customers systematically short-, medium-, or long-lived operators.
        ordered = sorted(vehicle_ids, key=lambda unit: (profiles[unit].lifespan, str(unit)))
        counts = _dirichlet_counts(len(ordered), k, alpha_quantity, rng)
        customer_order = rng.permutation(k)
        cursor = 0
        for client_id, count in zip(customer_order, counts):
            result[int(client_id)] = ordered[cursor:cursor + int(count)]
            cursor += int(count)

    else:
        # Each regime/drift profile independently draws a low-alpha allocation
        # over customer fleets. This is label-free: fault_cluster is an EOL drift
        # cluster, not a C-MAPSS component-fault label.
        buckets: dict[tuple[int, int], list[Hashable]] = {}
        for unit in vehicle_ids:
            profile = profiles[unit]
            buckets.setdefault((profile.dominant_regime, profile.fault_cluster), []).append(unit)
        for key in sorted(buckets):
            units = np.asarray(sorted(buckets[key], key=str), dtype=object)
            rng.shuffle(units)
            counts = rng.multinomial(len(units), rng.dirichlet(np.full(k, alpha_profile)))
            cursor = 0
            for client_id, count in enumerate(counts):
                result[client_id].extend(units[cursor:cursor + int(count)].tolist())
                cursor += int(count)
        # Low-alpha draws can leave a customer too small for a vehicle-held-out
        # validation split. Move, never copy, from the largest fleets.
        for small_id in [i for i in range(k) if len(result[i]) < 2]:
            while len(result[small_id]) < 2:
                donor = max(result, key=lambda i: len(result[i]))
                if len(result[donor]) <= 2:
                    raise AssertionError("cannot keep two vehicles in every client")
                result[small_id].append(result[donor].pop())

    assert_disjoint_partitions(result, vehicle_ids)
    return result


def _entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size else 0.0


def _js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64) + 1e-12
    b = np.asarray(b, dtype=np.float64) + 1e-12
    a /= a.sum()
    b /= b.sum()
    midpoint = 0.5 * (a + b)
    return float(0.5 * np.sum(a * np.log2(a / midpoint)) +
                 0.5 * np.sum(b * np.log2(b / midpoint)))


def _pairwise_mean_js(distributions: Sequence[np.ndarray]) -> float:
    values = [_js_divergence(distributions[i], distributions[j])
              for i in range(len(distributions)) for j in range(i + 1, len(distributions))]
    return float(np.mean(values)) if values else 0.0


def _gini(values: Sequence[float]) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if not len(x) or x.sum() == 0:
        return 0.0
    return float(np.sum((2 * np.arange(1, len(x) + 1) - len(x) - 1) * x) /
                 (len(x) * x.sum()))


def _categorical_distributions(partitions: Mapping[int, Sequence[Hashable]],
                               profiles: Mapping[Hashable, VehicleProfile],
                               attribute: str) -> tuple[list[np.ndarray], float]:
    labels = sorted({getattr(profile, attribute) for profile in profiles.values()})
    index = {label: i for i, label in enumerate(labels)}
    client_distributions = []
    for units in partitions.values():
        counts = np.zeros(len(labels), dtype=np.float64)
        for unit in units:
            counts[index[getattr(profiles[unit], attribute)]] += 1
        client_distributions.append(counts + 1e-9)
    allocation_entropies = []
    for label in labels:
        by_client = np.array([
            sum(getattr(profiles[unit], attribute) == label for unit in units)
            for units in partitions.values()
        ], dtype=np.float64)
        if by_client.sum():
            p = by_client / by_client.sum()
            allocation_entropies.append(_entropy(p) / math.log2(max(2, len(partitions))))
    return client_distributions, float(np.mean(allocation_entropies))


def partition_heterogeneity(partitions: Mapping[int, Sequence[Hashable]],
                            profiles: Mapping[Hashable, VehicleProfile]) -> dict:
    """Explicit count, lifespan, operating-regime, and drift-cluster skew."""
    assert_disjoint_partitions(partitions, profiles)
    sample_counts = np.array([
        sum(max(1, profiles[unit].lifespan - SEQ_LEN + 1) for unit in units)
        for units in partitions.values()
    ], dtype=np.float64)
    vehicle_counts = np.array([len(units) for units in partitions.values()], dtype=np.float64)
    all_lifespans = np.array([p.lifespan for p in profiles.values()], dtype=np.float64)
    quantiles = np.unique(np.quantile(all_lifespans, np.linspace(0, 1, 6)))
    if len(quantiles) < 2:
        quantiles = np.array([all_lifespans.min() - 1, all_lifespans.max() + 1])
    lifespan_hists, lifespan_means = [], []
    for units in partitions.values():
        current = np.array([profiles[unit].lifespan for unit in units], dtype=np.float64)
        lifespan_hists.append(np.histogram(current, bins=quantiles)[0] + 1e-9)
        lifespan_means.append(float(current.mean()))
    regime_dist, regime_allocation_entropy = _categorical_distributions(
        partitions, profiles, "dominant_regime")
    fault_dist, fault_allocation_entropy = _categorical_distributions(
        partitions, profiles, "fault_cluster")
    regime_js = _pairwise_mean_js(regime_dist)
    fault_js = _pairwise_mean_js(fault_dist)
    lifespan_js = _pairwise_mean_js(lifespan_hists)
    lifespan_spread = float(np.std(lifespan_means) / max(1.0, np.mean(all_lifespans)))
    count_cv = float(np.std(sample_counts) / max(1.0, np.mean(sample_counts)))
    return {
        "vehicle_count_min": int(vehicle_counts.min()),
        "vehicle_count_max": int(vehicle_counts.max()),
        "sample_count_cv": count_cv,
        "sample_count_gini": _gini(sample_counts),
        "lifespan_pairwise_js": lifespan_js,
        "lifespan_client_mean_spread": float(max(lifespan_means) - min(lifespan_means)),
        "lifespan_mean_cv": lifespan_spread,
        "dominant_regime_pairwise_js": regime_js,
        "dominant_regime_allocation_entropy": regime_allocation_entropy,
        "fault_cluster_pairwise_js": fault_js,
        "fault_cluster_allocation_entropy": fault_allocation_entropy,
        "heterogeneity_score": count_cv + lifespan_js + lifespan_spread + regime_js + fault_js,
    }


def select_round_clients(k: int, fraction_fit: float, seed: int, round_index: int) -> list[int]:
    """Deterministically sample exactly max(1, round(K*fraction_fit)) clients."""
    if k <= 0 or not 0 < fraction_fit <= 1:
        raise ValueError("K must be positive and fraction_fit must be in (0, 1]")
    count = max(1, round(k * fraction_fit))
    rng = np.random.RandomState(seed + 1009 * int(round_index))
    return sorted(int(i) for i in rng.choice(k, size=count, replace=False))


def build_decentralized_scaler(client_frames: Sequence[pd.DataFrame]) -> tuple[FederatedScaler, dict]:
    """Compute locally, then expose only LocalStats to the manufacturer aggregator."""
    local_stats: list[LocalStats] = []
    local_row_counts = []
    for frame in client_frames:
        raw_local_array = frame.loc[:, USEFUL_SENSORS].to_numpy(dtype=np.float64, copy=True)
        local_stats.append(compute_local_stats(raw_local_array))
        local_row_counts.append(int(len(raw_local_array)))
    scaler = aggregate_federated_stats(local_stats, mode="zscore", sensors=USEFUL_SENSORS)
    audit = {
        "aggregation_input_type": "LocalStats",
        "client_stat_objects": len(local_stats),
        "local_sensor_rows": local_row_counts,
        "raw_sensor_rows_shared": 0,
    }
    return scaler, audit


def _scale_frame(frame: pd.DataFrame, scaler: FederatedScaler) -> pd.DataFrame:
    result = frame.copy()
    scaled = scaler.transform(result.loc[:, USEFUL_SENSORS].to_numpy(dtype=np.float64))
    # Assign one column at a time so integer-valued settings in the raw files
    # are intentionally promoted to floating point on pandas 2.x/3.x.
    for index, sensor in enumerate(USEFUL_SENSORS):
        result[sensor] = scaled[:, index]
    return result


def _subset_frame(frame: pd.DataFrame, units: Sequence[Hashable]) -> pd.DataFrame:
    return frame[frame["unit"].isin(units)].copy()


def compute_vehicle_profiles(frame: pd.DataFrame, subset: str, seed: int = 0) -> dict[Hashable, VehicleProfile]:
    """Derive label-free operating-regime and EOL-drift profiles."""
    subset = subset.upper()
    n_regimes = 6 if subset in ("FD002", "FD004") else 1
    baseline = fit_healthy_baseline(frame, n_regimes=n_regimes, seed=seed)
    grouped = list(frame.sort_values(["unit", "cycle"]).groupby("unit", sort=True))
    regimes: dict[Hashable, int] = {}
    drift_features: list[np.ndarray] = []
    units = []
    for unit, group in grouped:
        row_regimes = baseline.regimes(group.loc[:, SETTINGS].to_numpy(dtype=np.float64))
        regimes[unit] = int(np.bincount(row_regimes, minlength=n_regimes).argmax())
        if subset == "FD004":
            end_drift = baseline.drift(group)[-min(SEQ_LEN, len(group)):].mean(axis=0)
            drift_features.append(np.nan_to_num(end_drift, nan=0.0, posinf=0.0, neginf=0.0))
            units.append(unit)
    fault_clusters = {unit: 0 for unit, _ in grouped}
    if subset == "FD004" and len(drift_features) >= 2:
        from sklearn.cluster import KMeans

        labels = KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(
            np.asarray(drift_features, dtype=np.float64))
        fault_clusters.update({unit: int(label) for unit, label in zip(units, labels)})
    return {
        unit: VehicleProfile(
            lifespan=int(group["cycle"].max()),
            dominant_regime=regimes[unit],
            fault_cluster=fault_clusters[unit],
        )
        for unit, group in grouped
    }


def load_fleet_data(dataset: str, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray,
                                                                 dict[Hashable, VehicleProfile]]:
    """Load one subset or the unique-ID 709-vehicle FD001--FD004 fleet."""
    dataset = dataset.upper()
    subsets = ("FD001", "FD002", "FD003", "FD004") if dataset == "COMBINED" else (dataset,)
    train_parts, test_parts, ruls, profiles = [], [], [], {}
    for subset in subsets:
        train = load_cmapss_raw(f"train_{subset}.txt")
        test = load_cmapss_raw(f"test_{subset}.txt")
        rul = np.asarray(load_rul(f"RUL_{subset}.txt"), dtype=np.float32)
        subset_profiles = compute_vehicle_profiles(train, subset, seed=seed)
        if dataset == "COMBINED":
            train["unit"] = train["unit"].map(lambda unit: f"{subset}:{int(unit)}")
            test["unit"] = test["unit"].map(lambda unit: f"{subset}:{int(unit)}")
            subset_profiles = {f"{subset}:{int(unit)}": profile
                               for unit, profile in subset_profiles.items()}
        train_parts.append(train)
        test_parts.append(test)
        ruls.extend(rul.tolist())
        profiles.update(subset_profiles)
    combined_train = pd.concat(train_parts, ignore_index=True)
    combined_test = pd.concat(test_parts, ignore_index=True)
    if len(profiles) != (709 if dataset == "COMBINED" else combined_train["unit"].nunique()):
        raise AssertionError("vehicle profile count does not match fleet size")
    return combined_train, combined_test, np.asarray(ruls, dtype=np.float32), profiles


def _unit_sequences(scaled_train: pd.DataFrame) -> dict[Hashable, tuple[np.ndarray, np.ndarray]]:
    cache = {}
    for unit, group in scaled_train.groupby("unit", sort=True):
        x, y = extract_supervised_sequences(group)
        cache[unit] = (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
    return cache


def _official_test_set(scaled_test: pd.DataFrame, ruls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for index, (_, group) in enumerate(scaled_test.groupby("unit", sort=False)):
        seq, target = extract_last_sequence(group, ruls[index])
        x.append(seq)
        y.append(target)
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _cap_arrays(x: np.ndarray, y: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or len(y) <= limit:
        return x, y
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(len(y), size=limit, replace=False))
    return x[keep], y[keep]


def _join_unit_sequences(cache: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
                         units: Sequence[Hashable]) -> tuple[np.ndarray, np.ndarray]:
    arrays = [cache[unit] for unit in units if len(cache[unit][1])]
    if not arrays:
        return (np.empty((0, SEQ_LEN, len(USEFUL_SENSORS)), dtype=np.float32),
                np.empty(0, dtype=np.float32))
    return np.concatenate([item[0] for item in arrays]), np.concatenate([item[1] for item in arrays])


def build_client_data(partitions: Mapping[int, Sequence[Hashable]],
                      sequence_cache: Mapping[Hashable, tuple[np.ndarray, np.ndarray]],
                      *, seed: int, max_train_sequences: int,
                      max_val_sequences: int) -> list[ClientData]:
    clients = []
    for client_id, units in sorted(partitions.items()):
        ordered = np.asarray(sorted(units, key=str), dtype=object)
        rng = np.random.RandomState(seed + 7919 * (client_id + 1))
        rng.shuffle(ordered)
        if len(ordered) < 2:
            raise ValueError(f"client {client_id} needs two vehicles for held-out validation")
        val_count = max(1, int(round(0.2 * len(ordered))))
        val_units = ordered[:val_count].tolist()
        train_units = ordered[val_count:].tolist()
        train_x, train_y = _join_unit_sequences(sequence_cache, train_units)
        val_x, val_y = _join_unit_sequences(sequence_cache, val_units)
        train_x, train_y = _cap_arrays(train_x, train_y, max_train_sequences,
                                       seed + 101 * client_id)
        val_x, val_y = _cap_arrays(val_x, val_y, max_val_sequences,
                                   seed + 103 * client_id)
        if not len(train_y) or not len(val_y):
            raise ValueError(f"client {client_id} has no train or validation windows")
        clients.append(ClientData(client_id, tuple(units), train_x, train_y, val_x, val_y))
    return clients


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def _average_states(states: Sequence[Mapping[str, torch.Tensor]], weights: Sequence[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    return {
        name: sum(state[name].float() * (weight / total) for state, weight in zip(states, weights))
        for name in states[0]
    }


def _loss(name: str) -> torch.nn.Module:
    return AsymmetricRULLoss(alpha_late=2.5, alpha_early=0.5) if name == "asym" else torch.nn.MSELoss()


def _fit_from_state(state: Mapping[str, torch.Tensor], client: ClientData, *, seed: int,
                    epochs: int, lr: float, mu: float, loss_name: str,
                    batch_size: int) -> dict[str, torch.Tensor]:
    set_seed(seed)
    model = CmapssRegressor()
    model.load_state_dict(state)
    loader = make_loader(client.train_x, client.train_y, batch_size=batch_size, seed=seed)
    train_epochs(model, loader, epochs, lr=lr, mu=mu,
                 global_params=list(model.parameters()), loss_fn=_loss(loss_name))
    return _clone_state(model.state_dict())


def _predictions(state: Mapping[str, torch.Tensor], x: np.ndarray) -> np.ndarray:
    model = CmapssRegressor()
    model.load_state_dict(state)
    return np.clip(predict(model, x), 0.0, float(MAX_RUL))


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    score = nasa_score(y, prediction)
    return {
        "rmse": rmse(y, prediction),
        "nasa_score": score,
        "nasa_score_per_sample": float(score / max(1, len(y))),
        "late_prediction_rate": late_prediction_rate(y, prediction),
    }


def _client_validation(state: Mapping[str, torch.Tensor], clients: Sequence[ClientData]) -> list[dict]:
    values = []
    for client in clients:
        metric = _metrics(client.val_y, _predictions(state, client.val_x))
        values.append({"client_id": client.client_id, "samples": int(len(client.val_y)), **metric})
    return values


def _summarize_client_metrics(values: Sequence[dict]) -> dict:
    errors = np.asarray([value["rmse"] for value in values], dtype=np.float64)
    scores = np.asarray([value["nasa_score_per_sample"] for value in values], dtype=np.float64)
    late = np.asarray([value["late_prediction_rate"] for value in values], dtype=np.float64)
    return {
        "client_mean_rmse": float(errors.mean()),
        "client_worst_rmse": float(errors.max()),
        "client_p90_rmse": float(np.percentile(errors, 90)),
        "client_mean_nasa_score_per_sample": float(scores.mean()),
        "client_mean_late_prediction_rate": float(late.mean()),
    }


def _with_client_metrics(global_metrics: dict, client_values: Sequence[dict]) -> dict:
    return {**global_metrics, **_summarize_client_metrics(client_values),
            "client_validation": list(client_values)}


def _run_local_only(initial_state: Mapping[str, torch.Tensor], clients: Sequence[ClientData],
                    test_x: np.ndarray, test_y: np.ndarray, *, seed: int, epochs: int,
                    lr: float, batch_size: int) -> tuple[dict, list[dict[str, torch.Tensor]]]:
    states, predictions, weights, client_values = [], [], [], []
    for client in clients:
        state = _fit_from_state(initial_state, client, seed=seed + 100 * client.client_id,
                                epochs=epochs, lr=lr, mu=0.0, loss_name="mse",
                                batch_size=batch_size)
        states.append(state)
        weights.append(client.n)
        predictions.append(_predictions(state, test_x))
        metric = _metrics(client.val_y, _predictions(state, client.val_x))
        client_values.append({"client_id": client.client_id, "samples": len(client.val_y), **metric})
    ensemble = np.average(np.vstack(predictions), axis=0, weights=np.asarray(weights))
    result = _with_client_metrics(_metrics(test_y, ensemble), client_values)
    result["round_history"] = []
    result["convergence"] = {"best_round": None, "rmse_improvement": 0.0}
    return result, states


def _run_federated(initial_state: Mapping[str, torch.Tensor], clients: Sequence[ClientData],
                   test_x: np.ndarray, test_y: np.ndarray, *, seed: int,
                   fraction_fit: float, rounds: int, local_epochs: int, lr: float,
                   mu: float, loss_name: str, batch_size: int) -> tuple[dict, dict[str, torch.Tensor]]:
    state = _clone_state(initial_state)
    history = []
    for round_index in range(1, rounds + 1):
        selected = select_round_clients(len(clients), fraction_fit, seed, round_index)
        states, weights = [], []
        for client_id in selected:
            client = clients[client_id]
            states.append(_fit_from_state(
                state, client, seed=seed + round_index * 1000 + client_id,
                epochs=local_epochs, lr=lr, mu=mu, loss_name=loss_name,
                batch_size=batch_size))
            weights.append(client.n)
        state = _average_states(states, weights)
        current = _metrics(test_y, _predictions(state, test_x))
        history.append({
            "round": round_index,
            "selected_clients": selected,
            "selected_count": len(selected),
            **current,
        })
    client_values = _client_validation(state, clients)
    result = _with_client_metrics(_metrics(test_y, _predictions(state, test_x)), client_values)
    rmse_history = [row["rmse"] for row in history]
    result["round_history"] = history
    result["convergence"] = {
        "best_round": int(np.argmin(rmse_history) + 1),
        "rmse_improvement": float(rmse_history[0] - rmse_history[-1]),
    }
    return result, state


def _run_personalization(global_state: Mapping[str, torch.Tensor], clients: Sequence[ClientData],
                         test_x: np.ndarray, test_y: np.ndarray, *, seed: int,
                         epochs: int, lr: float, batch_size: int,
                         base_history: Sequence[dict]) -> dict:
    before = _client_validation(global_state, clients)
    after, global_predictions, weights = [], [], []
    for client in clients:
        state = _fit_from_state(global_state, client, seed=seed + 50000 + client.client_id,
                                epochs=epochs, lr=lr, mu=0.0, loss_name="asym",
                                batch_size=batch_size)
        metric = _metrics(client.val_y, _predictions(state, client.val_x))
        after.append({"client_id": client.client_id, "samples": len(client.val_y), **metric})
        global_predictions.append(_predictions(state, test_x))
        weights.append(client.n)
    ensemble = np.average(np.vstack(global_predictions), axis=0, weights=np.asarray(weights))
    before_summary = _summarize_client_metrics(before)
    after_summary = _summarize_client_metrics(after)
    result = _with_client_metrics(_metrics(test_y, ensemble), after)
    result["round_history"] = list(base_history)
    result["convergence"] = {
        "best_round": int(np.argmin([row["rmse"] for row in base_history]) + 1),
        "rmse_improvement": float(base_history[0]["rmse"] - base_history[-1]["rmse"]),
    }
    result["personalization"] = {
        "before": before_summary,
        "after": after_summary,
        "client_mean_rmse_delta": float(
            after_summary["client_mean_rmse"] - before_summary["client_mean_rmse"]),
        "client_mean_nasa_delta": float(
            after_summary["client_mean_nasa_score_per_sample"] -
            before_summary["client_mean_nasa_score_per_sample"]),
    }
    return result


def run_all_methods(clients: Sequence[ClientData], test_x: np.ndarray, test_y: np.ndarray,
                    *, seed: int, fraction_fit: float, rounds: int = 3,
                    local_epochs: int = 1, personalization_epochs: int = 2,
                    lr: float = 1e-3, batch_size: int = 64,
                    fedprox_mu: float = 0.05) -> dict:
    """Run all six comparison methods from the same seeded initialization."""
    set_seed(seed)
    initial_state = _clone_state(CmapssRegressor().state_dict())
    results = {}
    results["local_only"], _ = _run_local_only(
        initial_state, clients, test_x, test_y, seed=seed,
        epochs=rounds * local_epochs, lr=lr, batch_size=batch_size)
    specifications = (
        ("fedavg_mse", 0.0, "mse"),
        ("fedprox_mse", fedprox_mu, "mse"),
        ("fedavg_asym", 0.0, "asym"),
        ("fedprox_asym", fedprox_mu, "asym"),
    )
    states = {}
    for method, mu, loss_name in specifications:
        results[method], states[method] = _run_federated(
            initial_state, clients, test_x, test_y, seed=seed,
            fraction_fit=fraction_fit, rounds=rounds, local_epochs=local_epochs,
            lr=lr, mu=mu, loss_name=loss_name, batch_size=batch_size)
    results["fedavg_asym_personalized"] = _run_personalization(
        states["fedavg_asym"], clients, test_x, test_y, seed=seed,
        epochs=personalization_epochs, lr=lr * 0.25, batch_size=batch_size,
        base_history=results["fedavg_asym"]["round_history"])
    return results


def _finite_method_metrics(result: Mapping[str, object]) -> bool:
    keys = ("rmse", "nasa_score", "late_prediction_rate", "client_worst_rmse", "client_p90_rmse")
    return all(np.isfinite(float(result[key])) for key in keys)


def make_synthetic_swarm(n_vehicles: int = 60, seed: int = 0) -> tuple[pd.DataFrame,
                                                                            dict[int, VehicleProfile]]:
    """Small six-regime/two-drift-cluster fleet for unit tests."""
    rng = np.random.RandomState(seed)
    rows, profiles = [], {}
    for unit in range(1, n_vehicles + 1):
        regime = (unit - 1) % 6
        fault_cluster = ((unit - 1) // 6) % 2
        lifespan = 42 + (unit * 7) % 38
        profiles[unit] = VehicleProfile(lifespan, regime, fault_cluster)
        for cycle in range(1, lifespan + 1):
            degradation = cycle / lifespan
            settings = np.array([regime * 0.7, regime % 2, regime * -0.2]) + rng.normal(0, 0.01, 3)
            sensors = rng.normal(0, 0.08, 14) + degradation * (0.2 + 0.25 * fault_cluster)
            values = np.concatenate([settings, sensors])
            row = {"unit": unit, "cycle": cycle}
            row.update(dict(zip(USEFUL_SENSORS, values)))
            rows.append(row)
    return pd.DataFrame(rows), profiles


def run_synthetic_smoke(seed: int = 7) -> dict:
    frame, profiles = make_synthetic_swarm(18, seed)
    partitions = partition_vehicles(list(profiles), 3, "regime_fault_skew",
                                    profiles=profiles, seed=seed)
    scaler, _ = build_decentralized_scaler([_subset_frame(frame, units)
                                             for units in partitions.values()])
    scaled = _scale_frame(frame, scaler)
    cache = _unit_sequences(scaled)
    clients = build_client_data(partitions, cache, seed=seed,
                                max_train_sequences=24, max_val_sequences=12)
    test_x = np.concatenate([client.val_x[:4] for client in clients])
    test_y = np.concatenate([client.val_y[:4] for client in clients])
    results = run_all_methods(clients, test_x, test_y, seed=seed, fraction_fit=2 / 3,
                              rounds=1, local_epochs=1, personalization_epochs=1,
                              batch_size=16)
    if not all(_finite_method_metrics(result) for result in results.values()):
        raise AssertionError("synthetic smoke produced a non-finite metric")
    return results


def _aggregate_scalar(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def aggregate_seed_results(seed_runs: Sequence[dict]) -> dict:
    scalar_keys = (
        "rmse", "nasa_score", "nasa_score_per_sample", "late_prediction_rate",
        "client_mean_rmse", "client_worst_rmse", "client_p90_rmse",
        "client_mean_nasa_score_per_sample", "client_mean_late_prediction_rate",
    )
    result = {}
    for method in METHODS:
        method_runs = [run["methods"][method] for run in seed_runs]
        summary = {key: _aggregate_scalar([float(run[key]) for run in method_runs])
                   for key in scalar_keys}
        convergence_runs = [run["convergence"] for run in method_runs
                            if run["convergence"]["best_round"] is not None]
        if convergence_runs:
            summary["convergence"] = {
                "best_round": _aggregate_scalar([run["best_round"] for run in convergence_runs]),
                "rmse_improvement": _aggregate_scalar(
                    [run["rmse_improvement"] for run in convergence_runs]),
            }
        histories = [run["round_history"] for run in method_runs if run["round_history"]]
        if histories:
            summary["round_rmse"] = [
                {"round": round_index + 1,
                 **_aggregate_scalar([history[round_index]["rmse"] for history in histories])}
                for round_index in range(min(map(len, histories)))
            ]
        if method == "fedavg_asym_personalized":
            summary["personalization"] = {
                key: _aggregate_scalar([run["personalization"][key] for run in method_runs])
                for key in ("client_mean_rmse_delta", "client_mean_nasa_delta")
            }
        result[method] = summary
    return result


def run_configuration(train: pd.DataFrame, test: pd.DataFrame, ruls: np.ndarray,
                      profiles: Mapping[Hashable, VehicleProfile], *, dataset: str,
                      k: int, fraction_fit: float, scheme: str, seeds: Sequence[int],
                      rounds: int, max_train_sequences: int,
                      max_val_sequences: int) -> dict:
    seed_runs = []
    for seed in seeds:
        partitions = partition_vehicles(list(profiles), k, scheme, profiles=profiles, seed=seed)
        client_frames = [_subset_frame(train, partitions[i]) for i in range(k)]
        scaler, privacy_audit = build_decentralized_scaler(client_frames)
        scaled_train = _scale_frame(train, scaler)
        scaled_test = _scale_frame(test, scaler)
        cache = _unit_sequences(scaled_train)
        clients = build_client_data(partitions, cache, seed=seed,
                                    max_train_sequences=max_train_sequences,
                                    max_val_sequences=max_val_sequences)
        test_x, test_y = _official_test_set(scaled_test, ruls)
        methods = run_all_methods(clients, test_x, test_y, seed=seed,
                                  fraction_fit=fraction_fit, rounds=rounds)
        seed_runs.append({
            "seed": seed,
            "partition": {str(client_id): [_jsonable_id(unit) for unit in units]
                          for client_id, units in partitions.items()},
            "heterogeneity": partition_heterogeneity(partitions, profiles),
            "normalization_privacy": privacy_audit,
            "methods": methods,
        })
        print(f"completed {dataset} K={k} {scheme} seed={seed}", flush=True)
    heterogeneity = {
        key: _aggregate_scalar([run["heterogeneity"][key] for run in seed_runs])
        for key in seed_runs[0]["heterogeneity"]
    }
    return {
        "dataset": dataset,
        "k": k,
        "fraction_fit": fraction_fit,
        "partition_scheme": scheme,
        "vehicle_count": len(profiles),
        "rounds": rounds,
        "per_seed": seed_runs,
        "aggregate": aggregate_seed_results(seed_runs),
        "heterogeneity_aggregate": heterogeneity,
    }


def run_benchmark(*, seeds: Sequence[int] = SEEDS, rounds: int = 5,
                  max_train_sequences: int = 512,
                  max_val_sequences: int = 128) -> dict:
    loaded = {}
    configurations = []
    for specification in DEFAULT_CONFIGS:
        dataset = specification["dataset"]
        if dataset not in loaded:
            loaded[dataset] = load_fleet_data(dataset)
        train, test, ruls, profiles = loaded[dataset]
        for scheme in PARTITION_SCHEMES:
            configurations.append(run_configuration(
                train, test, ruls, profiles, scheme=scheme, seeds=seeds,
                rounds=rounds, max_train_sequences=max_train_sequences,
                max_val_sequences=max_val_sequences, **specification))
    return {
        "benchmark": "non_iid_vehicle_swarm_federated_learning",
        "seeds": list(seeds),
        "partition_schemes": list(PARTITION_SCHEMES),
        "methods": list(METHODS),
        "training": {
            "rounds": rounds,
            "local_epochs": 1,
            "personalization_epochs": 2,
            "fedprox_mu": 0.05,
            "asymmetric_loss": {"alpha_late": 2.5, "alpha_early": 0.5},
            "max_train_sequences_per_client": max_train_sequences,
            "max_validation_sequences_per_client": max_val_sequences,
        },
        "privacy": {
            "partition_boundary": "vehicle ID",
            "raw_sensor_rows_shared": 0,
            "normalization_server_input": "LocalStats sufficient statistics only",
        },
        "combined_fleet_support": {
            "dataset_argument": "COMBINED",
            "vehicle_count": 709,
            "unique_unit_id_format": "FD00x:<unit>",
        },
        "configurations": configurations,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "vehicle_swarm_benchmark.json")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-train-sequences", type=int, default=512)
    parser.add_argument("--max-val-sequences", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--synthetic-smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.set_num_threads(1)
    if args.synthetic_smoke:
        print(json.dumps(run_synthetic_smoke(args.seeds[0]), indent=2))
        return 0
    benchmark = run_benchmark(
        seeds=args.seeds, rounds=args.rounds,
        max_train_sequences=args.max_train_sequences,
        max_val_sequences=args.max_val_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(benchmark, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
