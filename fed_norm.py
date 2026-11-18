"""Privacy-preserving aggregation and robust preprocessing for sensor data.

Each client computes channel-wise sufficient statistics locally. Only those
statistics are needed by :func:`aggregate_federated_stats`; raw samples never
enter that function.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from common import CmapssRegressor, SEQ_LEN, USEFUL_SENSORS


@dataclass(frozen=True)
class LocalStats:
    """Per-channel sufficient statistics computed on one edge node."""

    count: np.ndarray
    mean: np.ndarray
    M2: np.ndarray
    min: np.ndarray
    max: np.ndarray

    @property
    def variance(self) -> np.ndarray:
        return np.divide(self.M2, self.count, out=np.zeros_like(self.M2, dtype=np.float64),
                         where=self.count > 0)


@dataclass(frozen=True)
class FederatedScaler:
    """Global channel statistics and a transform suitable for edge inference."""

    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    mode: str = "zscore"
    clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0)
    sensors: Tuple[str, ...] = tuple(USEFUL_SENSORS)

    def __post_init__(self):
        if self.mode not in ("zscore", "minmax"):
            raise ValueError("mode must be 'zscore' or 'minmax'")
        if self.clip_range is not None and self.clip_range[0] > self.clip_range[1]:
            raise ValueError("clip_range lower bound must not exceed upper bound")

    def transform(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        """Scale the last axis and return float32 values."""
        x = np.asarray(values, dtype=np.float64)
        if x.shape[-1] != len(self.mean):
            raise ValueError(f"expected {len(self.mean)} channels, got {x.shape[-1]}")
        if self.mode == "zscore":
            denom = np.where(self.std > 0.0, self.std, 1.0)
            scaled = (x - self.mean) / denom
        else:
            span = np.where(self.max > self.min, self.max - self.min, 1.0)
            scaled = (x - self.min) / span
        if clip and self.clip_range is not None:
            scaled = np.clip(scaled, *self.clip_range)
        return scaled.astype(np.float32)

    def as_dict(self) -> dict:
        """Return a serialization-friendly representation."""
        return {
            "mode": self.mode,
            "clip_range": self.clip_range,
            "mean": dict(zip(self.sensors, self.mean.tolist())),
            "std": dict(zip(self.sensors, self.std.tolist())),
            "min": dict(zip(self.sensors, self.min.tolist())),
            "max": dict(zip(self.sensors, self.max.tolist())),
        }


def compute_local_stats(training_array: np.ndarray) -> LocalStats:
    """Compute Welford statistics locally over all leading sample dimensions.

    The final dimension is the channel dimension. Non-finite observations are
    excluded independently per channel; callers can use
    :func:`sanitize_telemetry_frame` first when imputations are desired.
    """
    x = np.asarray(training_array, dtype=np.float64)
    if x.ndim < 2:
        raise ValueError("training_array must have a sample axis and a channel axis")
    channels = x.shape[-1]
    x = x.reshape(-1, channels)
    finite = np.isfinite(x)
    count = finite.sum(axis=0).astype(np.int64)
    safe = np.where(finite, x, 0.0)
    mean = np.divide(safe.sum(axis=0), count, out=np.zeros(channels), where=count > 0)
    centered = np.where(finite, x - mean, 0.0)
    M2 = np.sum(centered * centered, axis=0, dtype=np.float64)
    low = np.min(np.where(finite, x, np.inf), axis=0)
    high = np.max(np.where(finite, x, -np.inf), axis=0)
    low[count == 0] = np.nan
    high[count == 0] = np.nan
    return LocalStats(count=count, mean=mean, M2=M2, min=low, max=high)


def aggregate_federated_stats(
    client_stats: Sequence[LocalStats],
    *,
    mode: str = "zscore",
    clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
    sensors: Optional[Sequence[str]] = None,
) -> FederatedScaler:
    """Merge client aggregates with Chan's parallel variance formula.

    The input contract accepts only ``LocalStats`` objects, never raw arrays.
    """
    if not client_stats:
        raise ValueError("at least one client statistic is required")
    if any(not isinstance(stat, LocalStats) for stat in client_stats):
        raise TypeError("client_stats must contain LocalStats only; raw samples are never accepted")
    n_channels = len(client_stats[0].count)
    count = np.zeros(n_channels, dtype=np.int64)
    mean = np.zeros(n_channels, dtype=np.float64)
    M2 = np.zeros(n_channels, dtype=np.float64)
    low = np.full(n_channels, np.inf, dtype=np.float64)
    high = np.full(n_channels, -np.inf, dtype=np.float64)

    for stat in client_stats:
        arrays = (stat.count, stat.mean, stat.M2, stat.min, stat.max)
        if any(np.asarray(a).shape != (n_channels,) for a in arrays):
            raise ValueError("all client statistics must have the same channel count")
        n_b = np.asarray(stat.count, dtype=np.int64)
        m_b = np.asarray(stat.mean, dtype=np.float64)
        m2_b = np.asarray(stat.M2, dtype=np.float64)
        nonempty = n_b > 0
        total = count + n_b
        delta = m_b - mean
        ratio = np.divide(n_b, total, out=np.zeros(n_channels, dtype=np.float64), where=total > 0)
        mean = np.where(nonempty, mean + delta * ratio, mean)
        cross = np.divide(count * n_b, total, out=np.zeros(n_channels, dtype=np.float64), where=total > 0)
        M2 = M2 + m2_b + delta * delta * cross
        count = total
        low = np.minimum(low, np.where(nonempty, stat.min, np.inf))
        high = np.maximum(high, np.where(nonempty, stat.max, -np.inf))

    if np.any(count == 0):
        raise ValueError("global statistics have channels with no finite observations")
    std = np.sqrt(np.maximum(M2 / count, 0.0))
    sensor_names = tuple(sensors if sensors is not None else USEFUL_SENSORS[:n_channels])
    if len(sensor_names) != n_channels:
        raise ValueError("sensors must contain one name per channel")
    return FederatedScaler(mean=mean, std=std, min=low, max=high, mode=mode,
                           clip_range=clip_range, sensors=sensor_names)


def sanitize_telemetry_frame(
    frame: pd.DataFrame,
    *,
    sensors: Sequence[str] = USEFUL_SENSORS,
    healthy_means: Optional[Mapping[str, float]] = None,
    clip_bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    stuck_tolerance: float = 0.0,
) -> Tuple[np.ndarray, dict]:
    """Impute missing, dropped, and stuck sensors and clip known raw bounds.

    Forward/back filling is performed independently per ``unit`` when that
    column exists. Missing channels use ``healthy_means`` when provided,
    otherwise their available finite mean, and finally zero.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    names = tuple(sensors)
    if not names:
        raise ValueError("sensors must not be empty")
    out = pd.DataFrame(index=frame.index)
    originally_missing = np.zeros((len(frame), len(names)), dtype=bool)
    imputed = []

    for j, name in enumerate(names):
        if name not in frame.columns:
            values = pd.Series(np.nan, index=frame.index, dtype=np.float64)
            originally_missing[:, j] = True
            imputed.append(name)
        else:
            values = pd.to_numeric(frame[name], errors="coerce").astype(np.float64)
            values = values.replace([np.inf, -np.inf], np.nan)
            originally_missing[:, j] = values.isna().to_numpy()
            unit_ids = frame["unit"].to_numpy() if "unit" in frame.columns else np.zeros(len(frame))
            for unit_id in pd.unique(unit_ids):
                positions = np.flatnonzero(pd.isna(unit_ids) if pd.isna(unit_id) else unit_ids == unit_id)
                unit_values = values.iloc[positions].dropna().to_numpy()
                if unit_values.size > 1 and np.ptp(unit_values) <= stuck_tolerance:
                    values.iloc[positions] = np.nan
                    originally_missing[positions, j] = True
                    imputed.append(name)
        if name in frame.columns and originally_missing[:, j].any() and name not in imputed:
            imputed.append(name)
        if "unit" in frame.columns:
            filled = values.groupby(frame["unit"], sort=False).transform(lambda s: s.ffill().bfill())
        else:
            filled = values.ffill().bfill()
        fallback_values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        fallback = None
        if healthy_means is not None and name in healthy_means:
            fallback = float(healthy_means[name])
        elif fallback_values.size:
            fallback = float(np.mean(fallback_values))
        if fallback is None or not np.isfinite(fallback):
            fallback = 0.0
        out[name] = filled.fillna(fallback).astype(np.float64)

    arr = out.loc[:, names].to_numpy(dtype=np.float64, copy=True)
    clipped_mask = np.zeros_like(arr, dtype=bool)
    if clip_bounds:
        for j, name in enumerate(names):
            if name not in clip_bounds:
                continue
            lo, hi = clip_bounds[name]
            if lo > hi:
                raise ValueError(f"invalid clipping bounds for {name}")
            clipped = np.clip(arr[:, j], lo, hi)
            clipped_mask[:, j] = clipped != arr[:, j]
            arr[:, j] = clipped
    report = {
        "missing_fraction": float(originally_missing.mean()) if arr.size else 0.0,
        "clipped_fraction": float(clipped_mask.mean()) if arr.size else 0.0,
        "imputed_channels": sorted(set(imputed)),
    }
    return arr.astype(np.float32), report


def compare_federated_vs_centralized(
    client_arrays: Sequence[np.ndarray],
    *,
    mode: str = "zscore",
    clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
    evaluate_cmapss: bool = False,
    cmapss_subset: str = "FD001",
    epochs: int = 2,
) -> dict:
    """Compare merged client statistics to concatenated centralized statistics.

    Set ``evaluate_cmapss`` to additionally train/evaluate a small local model
    parity probe on a C-MAPSS subset. Raw arrays are consumed here only to
    calculate client-side aggregates; they are not passed to the aggregator.
    """
    if not client_arrays:
        raise ValueError("at least one client array is required")
    stats = [compute_local_stats(a) for a in client_arrays]
    scaler = aggregate_federated_stats(stats, mode=mode, clip_range=clip_range)
    merged = np.concatenate([np.asarray(a).reshape(-1, np.asarray(a).shape[-1]) for a in client_arrays], axis=0)
    finite = np.isfinite(merged)
    central_values = np.where(finite, merged, np.nan)
    central = {
        "mean": np.nanmean(central_values, axis=0),
        "std": np.nanstd(central_values, axis=0),
        "min": np.nanmin(central_values, axis=0),
        "max": np.nanmax(central_values, axis=0),
    }
    result = {"scaler": scaler, "centralized": central,
              "equivalent": all(np.allclose(getattr(scaler, k), central[k], atol=1e-6, rtol=0)
                                 for k in central)}
    if evaluate_cmapss:
        result["cmapss_rmse"] = _cmapss_parity_probe(scaler, cmapss_subset, epochs)
    return result


def _cmapss_parity_probe(scaler: FederatedScaler, subset: str, epochs: int) -> float:
    """Small deterministic smoke evaluation; imports training helpers lazily."""
    import torch
    from common import (apply_scale, build_test_set, load_cmapss_raw, make_loader,
                        make_xy, rmse, set_seed, train_epochs)
    from benchmark_asymmetric_loss import compute_scales_from_train

    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("subset must be FD001, FD002, FD003, or FD004")
    train = load_cmapss_raw(f"train_{subset}.txt")
    units = sorted(train.unit.unique())
    # An equal unit split keeps training data at the edge; only LocalStats merge.
    mode = scaler.mode
    if mode == "minmax":
        scales = compute_scales_from_train(train)
        frame = apply_scale(train, scales)
        x_test, y_test = build_test_set(scales, subset)
    else:
        raw = train.groupby("unit", sort=False).head(50)[list(scaler.sensors)].to_numpy()
        probe_scaler = aggregate_federated_stats([compute_local_stats(raw)], mode="zscore")
        train = train.copy()
        train.loc[:, scaler.sensors] = probe_scaler.transform(train[list(scaler.sensors)].to_numpy(), clip=False)
        frame = train
        raw_test = load_cmapss_raw(f"test_{subset}.txt")
        x_test, y_test = [], []
        rul = __import__("common").load_rul(f"RUL_{subset}.txt")
        for unit, group in raw_test.groupby("unit"):
            values = probe_scaler.transform(group[list(scaler.sensors)].to_numpy(), clip=False)
            if len(values) < SEQ_LEN:
                values = np.concatenate([np.repeat(values[:1], SEQ_LEN-len(values), axis=0), values])
            x_test.append(values[-SEQ_LEN:])
            y_test.append(min(125, rul[int(unit)-1]))
        x_test, y_test = np.asarray(x_test, np.float32), np.asarray(y_test, np.float32)
    x_train, y_train = make_xy(frame, units)
    set_seed(42)
    model = train_epochs(CmapssRegressor(), make_loader(x_train, y_train, batch_size=64, seed=42),
                         epochs=epochs)
    with torch.no_grad():
        pred = model(torch.as_tensor(x_test)).numpy()
    return rmse(y_test, pred)
