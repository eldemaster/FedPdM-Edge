#!/usr/bin/env python3
"""Benchmark offline-first, service-bay federated C-MAPSS training."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import (AsymmetricRULLoss, CmapssRegressor, MAX_RUL, SEQ_LEN, USEFUL_SENSORS,
                    extract_last_sequence, extract_supervised_sequences, late_prediction_rate,
                    load_cmapss_raw, load_rul, make_loader, nasa_score, predict, rmse, set_seed,
                    train_epochs)
from fed_norm import aggregate_federated_stats, compute_local_stats
from service_bay_fl import (ServiceBayCoordinator, VehicleEdgeCheckpoint,
                            local_cosine_learning_rate, personalize_at_service_bay)
try:
    from vehicle_swarm_benchmark import (PARTITION_SCHEMES, _scale_frame, _subset_frame,
                                         assert_disjoint_partitions, compute_vehicle_profiles,
                                         partition_vehicles)
except ModuleNotFoundError:
    from experiments.vehicle_swarm_benchmark import (PARTITION_SCHEMES, _scale_frame, _subset_frame,
                                                     assert_disjoint_partitions,
                                                     compute_vehicle_profiles, partition_vehicles)

SEEDS = (7, 42, 100, 777, 2024)
CONFIGS = (("FD002", 10, 0.30), ("FD004", 10, 0.30), ("FD004", 20, 0.20))
METHODS = ("centralized_oracle", "centralized_budget_matched",
           "centralized_personalized_budget_matched", "local_only",
           "local_only_personalized_budget_matched",
           "fedavg_intermittent", "fedavgm_staleness",
           "scaffold_service_bay", "fedavgm_personalized", "fedavgm_asym_personalized")
FEDERATED_METHODS = ("fedavg_intermittent", "fedavgm_staleness", "scaffold_service_bay",
                     "fedavgm_personalized", "fedavgm_asym_personalized")


def _metric(y: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(np.asarray(pred, dtype=float), 0, MAX_RUL)
    score = nasa_score(y, pred)
    return {"rmse": rmse(y, pred), "nasa_score": score,
            "nasa_score_per_sample": score / max(1, len(y)),
            "late_prediction_rate": late_prediction_rate(y, pred)}


def _predict(state: Mapping[str, torch.Tensor], x: np.ndarray, device="cpu") -> np.ndarray:
    model = CmapssRegressor(); model.load_state_dict(state); model.to(device); model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=device)).detach().cpu().numpy()


def _clone(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _train(state, x, y, *, seed, epochs, lr, loss=None, batch_size=64, cosine=False,
           client_control=None, server_control=None, device="cpu", max_examples=None):
    set_seed(seed)
    if str(device).startswith("xpu") and hasattr(torch, "xpu"):
        torch.xpu.manual_seed_all(seed)
    model = CmapssRegressor(); model.load_state_dict(state); model.to(device)
    # The regressor returns (batch,), so keep scalar targets one-dimensional
    # too. (batch, 1) labels would broadcast MSE to a batch-by-batch matrix.
    loader = make_loader(x, np.asarray(y, dtype=np.float32).reshape(-1),
                         batch_size=batch_size, seed=seed)
    optimizer = (torch.optim.SGD(model.parameters(), lr=lr) if client_control is not None
                 else torch.optim.Adam(model.parameters(), lr=lr))
    model.train()
    examples_seen = 0
    for epoch in range(epochs):
        if cosine:
            for group in optimizer.param_groups:
                group["lr"] = local_cosine_learning_rate(lr, epoch, epochs)
        if client_control is None:
            criterion = loss or torch.nn.MSELoss()
            for bx, by in loader:
                if max_examples is not None:
                    remaining = max_examples - examples_seen
                    if remaining <= 0:
                        break
                    if len(bx) > remaining:
                        bx, by = bx[:remaining], by[:remaining]
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                criterion(model(bx), by).backward()
                optimizer.step()
                examples_seen += len(bx)
        else:
            criterion = loss or torch.nn.MSELoss()
            model.train()
            for bx, by in loader:
                if max_examples is not None:
                    remaining = max_examples - examples_seen
                    if remaining <= 0:
                        break
                    if len(bx) > remaining:
                        bx, by = bx[:remaining], by[:remaining]
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                criterion(model(bx), by).backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        parameter.grad.add_(server_control[name].to(parameter) - client_control[name].to(parameter))
                # Raw-RUL MSE can produce very large gradients. Sparse service
                # visits may contain several minibatches; clip the corrected
                # SCAFFOLD step so small smoke/test batches remain stable.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                examples_seen += len(bx)
    return _clone(model.state_dict())


def _unit_windows(frame) -> dict:
    result = {}
    for unit, group in frame.groupby("unit", sort=True):
        x, y = extract_supervised_sequences(group)
        result[unit] = (np.asarray(x, np.float32), np.asarray(y, np.float32))
    return result


def _official_test(frame, ruls):
    xs, ys = [], []
    for i, (_, group) in enumerate(frame.groupby("unit", sort=False)):
        x, y = extract_last_sequence(group, ruls[i]); xs.append(x); ys.append(y)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def _partition_client_data(partitions, cache, seed):
    clients = []
    for cid, units in sorted(partitions.items()):
        order = np.asarray(sorted(units, key=str), dtype=object)
        rng = np.random.RandomState(seed + 7919 * (cid + 1)); rng.shuffle(order)
        nval = max(1, round(0.2 * len(order)))
        val_units, train_units = order[:nval].tolist(), order[nval:].tolist()
        def join(selected):
            pieces = [cache[u] for u in selected if len(cache[u][1])]
            return np.concatenate([v[0] for v in pieces]), np.concatenate([v[1] for v in pieces])
        tx, ty = join(train_units); vx, vy = join(val_units)
        if not len(ty) or not len(vy):
            raise ValueError(f"client {cid} has no train or held-out vehicle windows")
        clients.append({"id": cid, "units": tuple(units), "train_x": tx, "train_y": ty,
                        "val_x": vx, "val_y": vy, "n": len(ty)})
    return clients


def _evaluate_global(state, test_x, test_y, clients, device="cpu"):
    result = _metric(test_y, _predict(state, test_x, device))
    local = [_metric(c["val_y"], _predict(state, c["val_x"], device)) for c in clients]
    result["customer_validation"] = {
        "mean_rmse": float(np.mean([m["rmse"] for m in local])),
        "mean_nasa_score_per_sample": float(np.mean([m["nasa_score_per_sample"] for m in local])),
        "mean_late_prediction_rate": float(np.mean([m["late_prediction_rate"] for m in local])),
        "per_customer": local,
    }
    return result


def _visit_schedule(k, participation, seed, cycles):
    count = max(1, round(k * participation))
    return [sorted(np.random.RandomState(seed + 1009 * cycle).choice(k, count, replace=False).tolist())
            for cycle in range(1, cycles + 1)]


def _training_window_budgets(clients, schedule, local_epochs):
    """Match training examples processed to each vehicle's realized visits."""
    visit_counts = {int(client["id"]): 0 for client in clients}
    for selected in schedule:
        for cid in selected:
            visit_counts[int(cid)] += 1
    per_vehicle = {
        int(client["id"]): int(visit_counts[int(client["id"])] * local_epochs * client["n"])
        for client in clients
    }
    return visit_counts, per_vehicle


def _schedule_staleness(schedule):
    """Return the exact elapsed service-cycle count for every check-in."""
    last_seen = {}
    trace = []
    for cycle, clients in enumerate(schedule, start=1):
        for client_id in clients:
            previous = last_seen.get(client_id, 0)
            trace.append({"cycle": cycle, "vehicle_id": int(client_id),
                          "last_synced_round": previous, "tau": cycle - previous})
            last_seen[client_id] = cycle
    return trace


def _run_federated(clients, test_x, test_y, *, seed, cycles, participation, method,
                   initial, local_epochs, batch_size, personalization_epochs, device="cpu",
                   schedule=None):
    scaffold = method == "scaffold_service_bay"
    personalized = method in ("fedavgm_personalized", "fedavgm_asym_personalized")
    asym = method == "fedavgm_asym_personalized"
    coordinator = ServiceBayCoordinator(initial, scaffold=scaffold)
    checkpoints = [VehicleEdgeCheckpoint(c["id"], 0, _clone(initial),
                                         {k: torch.zeros_like(v) for k, v in initial.items()})
                   for c in clients]
    visits = schedule or _visit_schedule(len(clients), participation, seed, cycles)
    history, visit_rows, client_personal = [], [], {}
    for cycle, selected in enumerate(visits, start=1):
        coordinator.round = cycle
        updates = []
        for cid in selected:
            c, checkpoint = clients[cid], checkpoints[cid]
            tau = cycle - checkpoint.last_synced_round
            old_local = _clone(checkpoint.model_weights)
            trained = _train(checkpoint.model_weights, c["train_x"], c["train_y"],
                             seed=seed + cycle * 1000 + cid,
                             epochs=local_epochs, lr=1e-3,
                             loss=AsymmetricRULLoss(2.5, 0.5) if asym else None,
                             batch_size=batch_size, cosine=True,
                             client_control=checkpoint.c_i if scaffold else None,
                             server_control=coordinator.global_c if scaffold else None,
                             device=device)
            if scaffold:
                steps_per_epoch = int(np.ceil(c["n"] / batch_size))
                effective_lr = float(np.mean([
                    local_cosine_learning_rate(1e-3, epoch, local_epochs)
                    for epoch in range(local_epochs)
                ]))
                coordinator.update_scaffold(checkpoint, old_local, trained,
                                            local_steps=local_epochs * steps_per_epoch,
                                            learning_rate=effective_lr)
            checkpoint.local_training_budget += local_epochs * len(c["train_y"])
            updates.append((checkpoint, trained, c["n"]))
            visit_rows.append({"cycle": cycle, "vehicle_id": cid,
                               "last_synced_round": checkpoint.last_synced_round,
                               "tau": tau, "training_windows": int(c["n"]),
                               "raw_sensor_rows_uploaded": 0})
        # FedAvg baseline uses the identical sparse schedule but no momentum/age decay.
        coordinator.aggregate(updates,
                              use_staleness=method != "fedavg_intermittent",
                              use_momentum=method != "fedavg_intermittent")
        for checkpoint, _, _ in updates:
            checkpoint.model_weights = _clone(coordinator.global_weights)
            checkpoint.last_synced_round = cycle
        if scaffold:
            coordinator.refresh_global_control(checkpoints)
        if personalized:
            for checkpoint, _, _ in updates:
                c = clients[int(checkpoint.vehicle_id)]
                checkpoint.model_weights = personalize_at_service_bay(
                    coordinator.global_weights, c["train_x"], c["train_y"],
                    seed=seed + cycle * 10000 + c["id"], epochs=personalization_epochs,
                    learning_rate=2.5e-4, batch_size=batch_size,
                    loss_fn=AsymmetricRULLoss(2.5, 0.5) if asym else None,
                    device=device)
        # Track global convergence each cycle; evaluate customer-held-out sets
        # once at the end so repeated cycles do not re-score every local window.
        current = _metric(test_y, _predict(coordinator.global_weights, test_x, device))
        history.append({"cycle": cycle, "selected_clients": selected, **current})
    global_result = _evaluate_global(coordinator.global_weights, test_x, test_y, clients, device)
    result = {"global": global_result, "round_history": history, "visits": visit_rows,
              "average_staleness_tau": float(np.mean([row["tau"] for row in visit_rows])) if visit_rows else 0.0,
              "max_staleness_tau": max((row["tau"] for row in visit_rows), default=0),
              "raw_sensor_rows_uploaded": 0}
    if personalized:
        # Evaluate each private model as it actually exists after its most recent
        # service visit. Offline customers are not given an extra sync here.
        client_personal = {c["id"]: checkpoints[c["id"]].model_weights for c in clients}
        values = [_metric(c["val_y"], _predict(client_personal[c["id"]], c["val_x"], device)) for c in clients]
        result["personalized_customer_validation"] = {
            "mean_rmse": float(np.mean([v["rmse"] for v in values])),
            "mean_nasa_score_per_sample": float(np.mean([v["nasa_score_per_sample"] for v in values])),
            "mean_late_prediction_rate": float(np.mean([v["late_prediction_rate"] for v in values])),
            "per_customer": values}
        # Apply each customer's private head model to the official test set as a
        # diagnostic ensemble; the globally shared parameters remain untouched.
        ensemble = np.mean([_predict(state, test_x, device) for state in client_personal.values()], axis=0)
        result["personalized_global_test"] = _metric(test_y, ensemble)
    return result


def run_seed(train, test, ruls, profiles, *, dataset, k, participation, scheme, seed,
             cycles=20, local_epochs=3, personalization_epochs=2, batch_size=64, device="cpu"):
    partitions = partition_vehicles(list(profiles), k, scheme, profiles=profiles, seed=seed)
    assert_disjoint_partitions(partitions, profiles)
    client_frames = [_subset_frame(train, partitions[i]) for i in range(k)]
    local_stats = [compute_local_stats(frame.loc[:, USEFUL_SENSORS].to_numpy()) for frame in client_frames]
    scaler = aggregate_federated_stats(local_stats, mode="zscore", sensors=USEFUL_SENSORS)
    scaled_train, scaled_test = _scale_frame(train, scaler), _scale_frame(test, scaler)
    clients = _partition_client_data(partitions, _unit_windows(scaled_train), seed)
    test_x, test_y = _official_test(scaled_test, ruls)
    schedule = _visit_schedule(k, participation, seed, cycles)
    visit_counts, per_vehicle_budget = _training_window_budgets(clients, schedule, local_epochs)
    federated_window_budget = int(sum(per_vehicle_budget.values()))
    _, personalized_vehicle_budget = _training_window_budgets(
        clients, schedule, local_epochs + personalization_epochs)
    personalized_window_budget = int(sum(personalized_vehicle_budget.values()))
    set_seed(seed)
    initial = _clone(CmapssRegressor().state_dict())
    pooled_x = np.concatenate([c["train_x"] for c in clients]); pooled_y = np.concatenate([c["train_y"] for c in clients])
    oracle = _train(initial, pooled_x, pooled_y, seed=seed, epochs=cycles * local_epochs,
                    lr=1e-3, batch_size=batch_size, cosine=True, device=device)
    matched_epochs = int(math.ceil(federated_window_budget / max(1, len(pooled_y))))
    matched_oracle = _train(initial, pooled_x, pooled_y, seed=seed, epochs=matched_epochs,
                            lr=1e-3, batch_size=batch_size, cosine=True, device=device,
                            max_examples=federated_window_budget)
    matched_personalized_epochs = int(math.ceil(personalized_window_budget / max(1, len(pooled_y))))
    matched_personalized_oracle = _train(
        initial, pooled_x, pooled_y, seed=seed, epochs=matched_personalized_epochs,
        lr=1e-3, batch_size=batch_size, cosine=True, device=device,
        max_examples=personalized_window_budget)
    methods = {"centralized_oracle": {"global": _evaluate_global(oracle, test_x, test_y, clients, device),
                                      "round_history": [], "average_staleness_tau": 0.0,
                                      "raw_sensor_rows_uploaded": 0,
                                      "training_windows_processed": int(cycles * local_epochs * len(pooled_y))},
               "centralized_budget_matched": {
                   "global": _evaluate_global(matched_oracle, test_x, test_y, clients, device),
                   "round_history": [], "average_staleness_tau": 0.0,
                   "raw_sensor_rows_uploaded": 0,
                   "training_windows_processed": federated_window_budget},
               "centralized_personalized_budget_matched": {
                   "global": _evaluate_global(matched_personalized_oracle, test_x, test_y,
                                               clients, device),
                   "round_history": [], "average_staleness_tau": 0.0,
                   "raw_sensor_rows_uploaded": 0,
                   "training_windows_processed": personalized_window_budget}}
    for baseline, budgets, epochs_per_visit in (
            ("local_only", per_vehicle_budget, local_epochs),
            ("local_only_personalized_budget_matched", personalized_vehicle_budget,
             local_epochs + personalization_epochs)):
        local_preds, local_customer_values = [], []
        for c in clients:
            cid = int(c["id"])
            local_epochs_matched = visit_counts[cid] * epochs_per_visit
            local = _train(initial, c["train_x"], c["train_y"], seed=seed + cid,
                           epochs=int(local_epochs_matched), lr=1e-3,
                           batch_size=batch_size, cosine=True, device=device,
                           max_examples=budgets[cid])
            local_preds.append(_predict(local, test_x, device))
            local_customer_values.append(_metric(c["val_y"], _predict(local, c["val_x"], device)))
        customer_validation = {"mean_rmse": float(np.mean([m["rmse"] for m in local_customer_values])),
                               "per_customer": local_customer_values}
        methods[baseline] = {
            "global": _metric(test_y, np.mean(local_preds, axis=0)),
            "round_history": [], "average_staleness_tau": 0.0,
            "raw_sensor_rows_uploaded": 0,
            "training_windows_processed": int(sum(budgets.values())),
            "customer_validation": customer_validation}
    for method in FEDERATED_METHODS:
        methods[method] = _run_federated(clients, test_x, test_y, seed=seed, cycles=cycles,
                                          participation=participation, method=method, initial=initial,
                                          local_epochs=local_epochs, batch_size=batch_size,
                                          personalization_epochs=personalization_epochs,
                                          device=device, schedule=schedule)
        methods[method]["training_windows_processed"] = (
            personalized_window_budget if method in
            ("fedavgm_personalized", "fedavgm_asym_personalized")
            else federated_window_budget)
    oracle_rmse = methods["centralized_oracle"]["global"]["rmse"]
    oracle_customer = methods["centralized_oracle"]["global"]["customer_validation"]["mean_rmse"]
    personal_budget_methods = {"centralized_personalized_budget_matched",
                               "local_only_personalized_budget_matched",
                               "fedavgm_personalized", "fedavgm_asym_personalized"}
    for method, record in methods.items():
        personalized_budget = method in personal_budget_methods
        local_reference = methods["local_only_personalized_budget_matched" if personalized_budget
                                 else "local_only"]
        central_reference = methods["centralized_personalized_budget_matched" if personalized_budget
                                   else "centralized_budget_matched"]
        local_rmse = local_reference["global"]["rmse"]
        matched_oracle_rmse = central_reference["global"]["rmse"]
        local_customer_rmse = local_reference["customer_validation"]["mean_rmse"]
        matched_oracle_customer = central_reference["global"]["customer_validation"]["mean_rmse"]
        global_metric = record.get("personalized_global_test", record["global"])
        customer_rmse = record.get("personalized_customer_validation", record.get("customer_validation", {})
                                  ).get("mean_rmse", record["global"].get("customer_validation", {})
                                        .get("mean_rmse", oracle_customer))
        global_denominator = local_rmse - oracle_rmse
        matched_global_denominator = local_rmse - matched_oracle_rmse
        customer_denominator = local_customer_rmse - oracle_customer
        matched_customer_denominator = local_customer_rmse - matched_oracle_customer
        record["comparison"] = {
            "gap_vs_centralized_rmse": float(global_metric["rmse"] - oracle_rmse),
            "gain_over_local_only_rmse": float(local_rmse - global_metric["rmse"]),
            "gap_closed_fraction": float((local_rmse - global_metric["rmse"]) / global_denominator)
            if abs(global_denominator) > 1e-12 else 0.0,
            "budget_matched_gap_closed_fraction": float(
                (local_rmse - global_metric["rmse"]) / matched_global_denominator)
            if abs(matched_global_denominator) > 1e-12 else 0.0,
            "gap_vs_centralized_customer_rmse": float(customer_rmse - oracle_customer),
            "gain_over_local_only_customer_rmse": float(local_customer_rmse - customer_rmse),
            "customer_gap_closed_fraction": float((local_customer_rmse - customer_rmse) /
                                                     customer_denominator)
            if abs(customer_denominator) > 1e-12 else 0.0,
            "budget_matched_customer_gap_closed_fraction": float(
                (local_customer_rmse - customer_rmse) / matched_customer_denominator)
            if abs(matched_customer_denominator) > 1e-12 else 0.0,
        }
    return {"seed": seed, "partition": {str(cid): list(units) for cid, units in partitions.items()},
            "vehicles_disjoint": True, "normalization": {"aggregated_type": "LocalStats",
            "raw_sensor_rows_uploaded": 0}, "training_budget": {
                "schedule_visit_count_by_vehicle": {str(cid): count for cid, count in visit_counts.items()},
                "local_training_windows_by_vehicle": {
                    str(cid): budget for cid, budget in per_vehicle_budget.items()},
                "personalized_training_windows_by_vehicle": {
                    str(cid): budget for cid, budget in personalized_vehicle_budget.items()},
                "federated_total_training_windows": federated_window_budget,
                "personalized_federated_total_training_windows": personalized_window_budget,
                "matched_local_only_total_training_windows": federated_window_budget,
                "matched_personalized_local_only_total_training_windows": personalized_window_budget,
                "matched_centralized_total_training_windows": federated_window_budget,
                "matched_personalized_centralized_total_training_windows": personalized_window_budget,
                "full_centralized_oracle_training_windows": int(cycles * local_epochs * len(pooled_y))
            }, "methods": methods}


def _summarize(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=0))}


def _aggregate(seed_runs):
    output = {}
    for method in METHODS:
        records = [run["methods"][method] for run in seed_runs]
        paths = ["global"] if method not in ("fedavgm_personalized", "fedavgm_asym_personalized") else ["global", "personalized_global_test", "personalized_customer_validation"]
        summary = {}
        for path in paths:
            rows = [r[path] for r in records] if path != "global" else [r["global"] for r in records]
            keys = ("rmse", "nasa_score", "nasa_score_per_sample", "late_prediction_rate") if path != "personalized_customer_validation" else ("mean_rmse", "mean_nasa_score_per_sample", "mean_late_prediction_rate")
            summary[path] = {key: _summarize([row[key] for row in rows]) for key in keys}
        summary["average_staleness_tau"] = _summarize([r["average_staleness_tau"] for r in records])
        summary["comparison"] = {key: _summarize([r["comparison"][key] for r in records])
                                 for key in ("gap_vs_centralized_rmse", "gain_over_local_only_rmse",
                                             "gap_closed_fraction",
                                             "budget_matched_gap_closed_fraction",
                                             "gap_vs_centralized_customer_rmse",
                                             "gain_over_local_only_customer_rmse",
                                             "customer_gap_closed_fraction",
                                             "budget_matched_customer_gap_closed_fraction")}
        histories = [r["round_history"] for r in records if r["round_history"]]
        if histories:
            summary["round_rmse"] = [{"cycle": i + 1, **_summarize([h[i]["rmse"] for h in histories])}
                                      for i in range(min(map(len, histories)))]
        output[method] = summary
    return output


def run_benchmark(*, seeds: Sequence[int] = SEEDS, cycles: int = 20, local_epochs: int = 3,
                  personalization_epochs: int = 2, datasets=None, schemes=PARTITION_SCHEMES,
                  configs=None, batch_size=64, device="cpu"):
    if device.startswith("xpu") and (not hasattr(torch, "xpu") or not torch.xpu.is_available()):
        raise RuntimeError("PyTorch XPU was requested but is unavailable")
    configs = configs or CONFIGS
    loaded, configurations = {}, []
    for dataset, k, participation in configs:
        if dataset not in loaded:
            train, test, ruls = (load_cmapss_raw(f"train_{dataset}.txt"),
                                 load_cmapss_raw(f"test_{dataset}.txt"),
                                 np.asarray(load_rul(f"RUL_{dataset}.txt"), np.float32))
            loaded[dataset] = (train, test, ruls, compute_vehicle_profiles(train, dataset))
        train, test, ruls, profiles = loaded[dataset]
        for scheme in schemes:
            seed_runs = []
            for seed in seeds:
                seed_runs.append(run_seed(train, test, ruls, profiles, dataset=dataset, k=k,
                                          participation=participation, scheme=scheme, seed=seed,
                                          cycles=cycles, local_epochs=local_epochs,
                                          personalization_epochs=personalization_epochs,
                                          batch_size=batch_size, device=device))
                print(f"completed {dataset} K={k} {scheme} seed={seed}", flush=True)
            configurations.append({"dataset": dataset, "k": k, "participation": participation,
                                   "cycles": cycles, "partition_scheme": scheme,
                                   "per_seed": seed_runs, "aggregate": _aggregate(seed_runs)})
    return {"benchmark": "service_bay_intermittent_connectivity_federated_learning",
            "seeds": list(seeds), "methods": list(METHODS), "configurations": configurations,
            "training": {"local_epochs_per_visit": local_epochs, "full_local_history": True,
                         "max_train_windows_per_client": None, "cosine_learning_rate": True,
                         "budget_matching": {
                             "unit": "training windows processed",
                             "local_only_matches_each_vehicles_realized_visit_budget": True,
                             "centralized_budget_matched_matches_total_federated_windows": True,
                             "personalized_baselines_include_head_finetuning_windows": True,
                             "centralized_oracle_is_full_budget_upper_bound": True},
                         "device": str(device),
                         "momentum_beta": 0.9, "staleness_gamma": 0.5,
                         "asymmetric_loss": {"alpha_late": 2.5, "alpha_early": 0.5}},
            "privacy": {"raw_sensor_rows_uploaded": 0,
                        "normalization_server_input": "LocalStats sufficient statistics only"},
            "staleness_protocol": "Each vehicle trains from its last synchronized checkpoint; tau is cycle minus last_synced_round; arriving stale deltas are applied to current server weights with age weighting, then the current global model is synced back."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/service_bay_benchmark.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--personalization-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("cpu", "xpu"), default="cpu")
    parser.add_argument("--configs", nargs="*", choices=["FD002-K10", "FD004-K10", "FD004-K20"])
    parser.add_argument("--schemes", nargs="+", choices=PARTITION_SCHEMES, default=list(PARTITION_SCHEMES))
    args = parser.parse_args(argv); torch.set_num_threads(1)
    config_map = {"FD002-K10": CONFIGS[0], "FD004-K10": CONFIGS[1], "FD004-K20": CONFIGS[2]}
    configs = [config_map[name] for name in args.configs] if args.configs else CONFIGS
    result = run_benchmark(seeds=args.seeds, cycles=args.cycles,
                           local_epochs=args.local_epochs,
                           personalization_epochs=args.personalization_epochs,
                           configs=configs, schemes=args.schemes, batch_size=args.batch_size,
                           device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
