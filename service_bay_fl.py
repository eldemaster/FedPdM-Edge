"""Offline-first federated learning primitives for vehicle service-bay visits.

Only model parameters, persistent control variates, and LocalStats are stored in
edge checkpoints. Sensor windows remain on the vehicle/customer fleet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import copy

import numpy as np
import torch
from torch import nn

from common import CmapssRegressor, make_loader, set_seed, train_epochs
from fed_norm import LocalStats


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _stats_to_dict(stats: LocalStats | None) -> dict | None:
    if stats is None:
        return None
    return {key: np.asarray(getattr(stats, key)).copy() for key in ("count", "mean", "M2", "min", "max")}


def _stats_from_dict(value: dict | None) -> LocalStats | None:
    if value is None:
        return None
    return LocalStats(**{key: np.asarray(value[key]) for key in ("count", "mean", "M2", "min", "max")})


@dataclass
class VehicleEdgeCheckpoint:
    """Persisted model/control state for a vehicle between service visits."""

    vehicle_id: str | int
    last_synced_round: int
    model_weights: dict[str, torch.Tensor]
    c_i: dict[str, torch.Tensor]
    local_stats: LocalStats | None = None
    local_training_budget: int = 0

    def state_dict(self) -> dict:
        return {"vehicle_id": self.vehicle_id,
                "last_synced_round": int(self.last_synced_round),
                "model_weights": _clone_state(self.model_weights),
                "c_i": _clone_state(self.c_i),
                "local_stats": _stats_to_dict(self.local_stats),
                "local_training_budget": int(self.local_training_budget),
                "raw_sensor_arrays": 0}

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def from_state_dict(cls, payload: dict) -> "VehicleEdgeCheckpoint":
        if payload.get("raw_sensor_arrays", 0) != 0:
            raise ValueError("edge checkpoint must not contain raw sensor arrays")
        return cls(payload["vehicle_id"], int(payload["last_synced_round"]),
                   _clone_state(payload["model_weights"]), _clone_state(payload["c_i"]),
                   _stats_from_dict(payload.get("local_stats")),
                   int(payload.get("local_training_budget", 0)))

    @classmethod
    def load(cls, path: str | Path) -> "VehicleEdgeCheckpoint":
        try:
            payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch before weights_only was introduced
            payload = torch.load(Path(path), map_location="cpu")
        return cls.from_state_dict(payload)


def staleness_weight(sample_count: int, tau: int, gamma: float = 0.5) -> float:
    if sample_count < 0 or tau < 0 or gamma < 0:
        raise ValueError("sample_count, tau, and gamma must be non-negative")
    return float(sample_count * (1.0 + tau) ** (-gamma))


class ServiceBayCoordinator:
    """Stateful FedAvgM/SCAFFOLD coordinator accepting sparse check-ins."""

    def __init__(self, initial_weights: Mapping[str, torch.Tensor], *, beta: float = 0.9,
                 gamma: float = 0.5, scaffold: bool = False):
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        self.global_weights = _clone_state(initial_weights)
        self.beta, self.gamma, self.scaffold = float(beta), float(gamma), bool(scaffold)
        self.momentum = {key: torch.zeros_like(value, dtype=torch.float32)
                         for key, value in self.global_weights.items()}
        self.global_c = {key: torch.zeros_like(value, dtype=torch.float32)
                         for key, value in self.global_weights.items()}
        self.round = 0

    def aggregate(self, updates: Sequence[tuple[VehicleEdgeCheckpoint, Mapping[str, torch.Tensor], int]],
                  *, use_staleness: bool = True, use_momentum: bool = True) -> dict[str, int | float]:
        """Apply weighted client deltas; tuple is (checkpoint, trained weights, n)."""
        if not updates:
            return {"participants": 0, "average_tau": 0.0, "total_weight": 0.0}
        weights = [staleness_weight(n, max(0, self.round - checkpoint.last_synced_round), self.gamma)
                   if use_staleness else float(n) for checkpoint, _, n in updates]
        total = sum(weights)
        if total <= 0:
            raise ValueError("updates must have positive total sample weight")
        # Apply the update produced since the vehicle's own last sync point to
        # today's server model. Subtracting today's global weights here would
        # count all intervening server movement as a client update.
        delta = {key: sum((trained[key].detach().cpu().float() - checkpoint.model_weights[key].float()) *
                          (weight / total)
                          for (checkpoint, trained, _), weight in zip(updates, weights))
                 for key in self.global_weights}
        for key in self.global_weights:
            self.momentum[key] = self.beta * self.momentum[key] + delta[key]
            step = self.momentum[key] if use_momentum else delta[key]
            self.global_weights[key] = (self.global_weights[key].float() + step).to(
                dtype=self.global_weights[key].dtype)
        return {"participants": len(updates),
                "average_tau": float(np.average([max(0, self.round - item[0].last_synced_round)
                                                  for item in updates], weights=weights)),
                "total_weight": float(total)}

    def update_scaffold(self, checkpoint: VehicleEdgeCheckpoint,
                        old_weights: Mapping[str, torch.Tensor],
                        new_weights: Mapping[str, torch.Tensor], *, local_steps: int,
                        learning_rate: float) -> None:
        """Update persistent c_i using the standard SCAFFOLD client delta rule."""
        if local_steps <= 0 or learning_rate <= 0:
            raise ValueError("local_steps and learning_rate must be positive")
        old_ci = checkpoint.c_i
        new_ci = {}
        for key in self.global_weights:
            change = (old_weights[key].float() - new_weights[key].float()) / (local_steps * learning_rate)
            new_ci[key] = old_ci[key].float() - self.global_c[key] + change
        checkpoint.c_i = new_ci

    def refresh_global_control(self, checkpoints: Sequence[VehicleEdgeCheckpoint]) -> None:
        """Recompute the fleet mean control variate after a check-in cycle."""
        if not checkpoints:
            raise ValueError("at least one vehicle checkpoint is required")
        for key in self.global_c:
            self.global_c[key] = sum(cp.c_i[key].float() for cp in checkpoints) / len(checkpoints)


def personalize_at_service_bay(global_weights: Mapping[str, torch.Tensor], train_x: np.ndarray,
                               train_y: np.ndarray, *, seed: int = 0, epochs: int = 2,
                               learning_rate: float = 2.5e-4, batch_size: int = 64,
                               loss_fn: nn.Module | None = None,
                               freeze_backbone: bool = True, device="cpu") -> dict[str, torch.Tensor]:
    """Fine-tune a private head copy after sync; never edits coordinator weights."""
    set_seed(seed)
    model = CmapssRegressor()
    model.load_state_dict(global_weights); model.to(device)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("fc1") or name.startswith("fc2"))
    loader = make_loader(train_x, train_y, batch_size=batch_size, seed=seed)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = loss_fn or nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()
    return _clone_state(model.state_dict())


def local_cosine_learning_rate(base_lr: float, epoch: int, epochs: int, minimum_ratio: float = 0.1) -> float:
    if base_lr <= 0 or epochs <= 0 or not 0 <= epoch < epochs or not 0 <= minimum_ratio <= 1:
        raise ValueError("invalid cosine schedule arguments")
    progress = epoch / max(1, epochs - 1)
    return base_lr * (minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + np.cos(np.pi * progress)))

