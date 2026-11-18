"""Streaming edge inference and debounced maintenance alerting.

The runtime deliberately keeps raw telemetry on the edge.  Network-facing
alert sinks receive only the aggregate, allow-listed fields in
``ALERT_PAYLOAD_FIELDS``; model windows and individual sensor readings are
never included.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, Mapping, Optional, TextIO, Union
from urllib import request

import numpy as np
import pandas as pd
import torch

from common import (
    MAX_RUL,
    SEQ_LEN,
    USEFUL_SENSORS,
    CmapssRegressor,
    load_cmapss_raw,
    load_scales,
)
from diagnostics import FAULT_SENSORS, EdgeDiagnostics, HealthyBaseline, fit_healthy_baseline, severity_for


PathLike = Union[str, Path]
ALERT_PAYLOAD_FIELDS = frozenset(
    {
        "asset_id",
        "cycle",
        "predicted_rul",
        "severity",
        "primary_subsystem",
        "subsystem_shares_pct",
        "top_contributors",
        "top_drifting",
        "summary_text",
        "privacy",
    }
)
_SEVERITIES = ("healthy", "warning", "critical")


def _timestamp_number(value: object) -> float:
    """Normalize numeric or ISO-8601 timestamps for monotonic comparisons."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    elif isinstance(value, str):
        text = value.strip()
        try:
            result = float(text)
        except ValueError:
            try:
                result = datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
            except ValueError as exc:
                raise ValueError(f"invalid timestamp {value!r}; expected epoch or ISO-8601") from exc
    else:
        raise ValueError(f"invalid timestamp {value!r}; expected epoch or ISO-8601")
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result


@dataclass(frozen=True)
class TelemetrySample:
    asset_id: str
    cycle: int
    timestamp: float
    values: Dict[str, float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "TelemetrySample":
        asset = payload.get("asset_id", payload.get("unit"))
        if asset is None or str(asset).strip() == "":
            raise ValueError("telemetry payload requires asset_id")
        if "cycle" not in payload:
            raise ValueError("telemetry payload requires cycle")
        try:
            cycle = int(payload["cycle"])
        except (TypeError, ValueError) as exc:
            raise ValueError("cycle must be an integer") from exc
        if cycle < 0:
            raise ValueError("cycle must be non-negative")

        raw_values = payload.get("values")
        if raw_values is None:
            raw_values = {name: payload[name] for name in USEFUL_SENSORS if name in payload}
        if not isinstance(raw_values, Mapping):
            raise ValueError("values must be a JSON object")
        values: Dict[str, float] = {}
        for key, value in raw_values.items():
            if key not in USEFUL_SENSORS or value is None or value == "":
                continue
            try:
                values[str(key)] = float(value)
            except (TypeError, ValueError):
                # Treat an unparseable channel exactly like a missing channel;
                # the rolling buffer will use last-observation-carried-forward.
                continue
        timestamp = _timestamp_number(payload.get("timestamp", cycle))
        return cls(str(asset), cycle, timestamp, values)


class TelemetrySource(ABC):
    """Iterable adapter for PLC, file replay, or broker telemetry."""

    @abstractmethod
    def __iter__(self) -> Iterator[TelemetrySample]:
        raise NotImplementedError


class JsonLinesStreamSource(TelemetrySource):
    """Read one ``TelemetrySample`` JSON object per non-empty line."""

    def __init__(self, source: Union[PathLike, TextIO, Iterable[str]]):
        self.source = source

    @staticmethod
    def parse_line(line: Union[str, bytes]) -> TelemetrySample:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError("telemetry JSON line must contain an object")
        return TelemetrySample.from_mapping(payload)

    def __iter__(self) -> Iterator[TelemetrySample]:
        owned: Optional[TextIO] = None
        if isinstance(self.source, (str, Path)):
            owned = open(self.source, "r", encoding="utf-8")
            lines: Iterable[str] = owned
        else:
            lines = self.source
        try:
            for line_number, line in enumerate(lines, 1):
                if isinstance(line, bytes):
                    nonempty = bool(line.strip())
                else:
                    nonempty = bool(str(line).strip())
                if not nonempty:
                    continue
                try:
                    yield self.parse_line(line)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid telemetry JSON at line {line_number}: {exc}") from exc
        finally:
            if owned is not None:
                owned.close()


class CsvReplaySource(TelemetrySource):
    """Replay CSV rows with ``asset_id`` (or ``unit``), cycle, and sensor columns."""

    def __init__(self, source: Union[PathLike, TextIO], asset_column: Optional[str] = None):
        self.source = source
        self.asset_column = asset_column

    def __iter__(self) -> Iterator[TelemetrySample]:
        owned: Optional[TextIO] = None
        if isinstance(self.source, (str, Path)):
            owned = open(self.source, "r", encoding="utf-8", newline="")
            stream: TextIO = owned
        else:
            stream = self.source
        try:
            for line_number, row in enumerate(csv.DictReader(stream), 2):
                asset_column = self.asset_column or ("asset_id" if "asset_id" in row else "unit")
                payload: Dict[str, object] = {
                    "asset_id": row.get(asset_column),
                    "cycle": row.get("cycle"),
                    "timestamp": row.get("timestamp") or row.get("cycle"),
                    "values": {name: row[name] for name in USEFUL_SENSORS if row.get(name) not in (None, "")},
                }
                try:
                    yield TelemetrySample.from_mapping(payload)
                except ValueError as exc:
                    raise ValueError(f"invalid telemetry CSV at line {line_number}: {exc}") from exc
        finally:
            if owned is not None:
                owned.close()


class MqttSource(TelemetrySource):
    """Optional MQTT adapter with no import-time dependency on paho-mqtt.

    The MQTT client is injectable so parsing and callback behavior can be unit
    tested without a broker.  Network activity starts only when iteration
    starts, not in the constructor.
    """

    _STOP = object()

    def __init__(self, topic: str, host: str = "localhost", port: int = 1883, qos: int = 0,
                 client: object = None):
        if client is None:
            try:
                import paho.mqtt.client as mqtt  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "MqttSource requires the optional 'paho-mqtt' package; "
                    "install it with: pip install paho-mqtt"
                ) from exc
            client = mqtt.Client()
        self.topic, self.host, self.port, self.qos = topic, host, int(port), int(qos)
        self.client = client
        self._messages: "queue.Queue[object]" = queue.Queue()
        self._started = False
        self._closed = False
        setattr(self.client, "on_message", self._on_message)

    @staticmethod
    def parse_message(message: object) -> TelemetrySample:
        if not hasattr(message, "payload"):
            raise ValueError("MQTT message has no payload")
        return JsonLinesStreamSource.parse_line(getattr(message, "payload"))

    def _on_message(self, _client: object, _userdata: object, message: object, *_args: object) -> None:
        try:
            self._messages.put(self.parse_message(message))
        except Exception as exc:  # surface malformed broker messages to the consumer
            self._messages.put(exc)

    def __iter__(self) -> Iterator[TelemetrySample]:
        if self._closed:
            return
        if not self._started:
            self.client.connect(self.host, self.port)
            self.client.subscribe(self.topic, qos=self.qos)
            self.client.loop_start()
            self._started = True
        while not self._closed:
            item = self._messages.get()
            if item is self._STOP:
                break
            if isinstance(item, Exception):
                raise ValueError(f"invalid MQTT telemetry: {item}") from item
            yield item  # type: ignore[misc]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._messages.put(self._STOP)
        if self._started:
            self.client.loop_stop()
            self.client.disconnect()


@dataclass
class _AssetHistory:
    rows: Deque[tuple]
    last_values: np.ndarray
    last_cycle: Optional[int] = None
    last_timestamp: Optional[float] = None


class RollingWindowBuffer:
    """Thread-safe, memory-bounded rolling windows with per-channel LOCF."""

    def __init__(self, window_size: int = SEQ_LEN, max_assets: int = 1024, fill_value: float = 0.0):
        if window_size <= 0 or max_assets <= 0:
            raise ValueError("window_size and max_assets must be positive")
        if not math.isfinite(fill_value):
            raise ValueError("fill_value must be finite")
        self.window_size = int(window_size)
        self.max_assets = int(max_assets)
        self.fill_value = float(fill_value)
        self._assets: "OrderedDict[str, _AssetHistory]" = OrderedDict()
        self._lock = threading.RLock()

    def _history(self, asset_id: str) -> _AssetHistory:
        history = self._assets.get(asset_id)
        if history is None:
            if len(self._assets) >= self.max_assets:
                self._assets.popitem(last=False)
            history = _AssetHistory(
                deque(maxlen=self.window_size),
                np.full(len(USEFUL_SENSORS), self.fill_value, dtype=np.float64),
            )
            self._assets[asset_id] = history
        else:
            self._assets.move_to_end(asset_id)
        return history

    def append(self, sample: TelemetrySample) -> np.ndarray:
        """Add a sample and return its finite, imputed sensor vector."""
        with self._lock:
            timestamp = _timestamp_number(sample.timestamp)
            if sample.cycle < 0:
                raise ValueError("cycle must be non-negative")
            history = self._history(sample.asset_id)
            if history.last_cycle is not None and sample.cycle <= history.last_cycle:
                raise ValueError(
                    f"non-monotonic cycle for {sample.asset_id}: {sample.cycle} <= {history.last_cycle}"
                )
            if history.last_timestamp is not None and timestamp <= history.last_timestamp:
                raise ValueError(
                    f"non-monotonic timestamp for {sample.asset_id}: {timestamp} <= {history.last_timestamp}"
                )

            row = history.last_values.copy()
            for index, sensor in enumerate(USEFUL_SENSORS):
                raw = sample.values.get(sensor)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    row[index] = value
            history.last_values = row
            history.last_cycle = int(sample.cycle)
            history.last_timestamp = timestamp
            history.rows.append((int(sample.cycle), timestamp, row.copy()))
            return row.copy()

    def window(self, asset_id: str) -> np.ndarray:
        """Return a W x C window, left-padded by repeating the first reading."""
        with self._lock:
            history = self._assets.get(asset_id)
            if history is None or not history.rows:
                raise KeyError(f"no telemetry buffered for asset {asset_id!r}")
            self._assets.move_to_end(asset_id)
            rows = np.stack([entry[2] for entry in history.rows])
            if len(rows) < self.window_size:
                rows = np.concatenate(
                    [np.repeat(rows[:1], self.window_size - len(rows), axis=0), rows], axis=0
                )
            return rows.copy()

    def frame(self, asset_id: str) -> pd.DataFrame:
        """Return the padded raw-unit frame expected by ``EdgeDiagnostics``."""
        with self._lock:
            history = self._assets.get(asset_id)
            if history is None or not history.rows:
                raise KeyError(f"no telemetry buffered for asset {asset_id!r}")
            self._assets.move_to_end(asset_id)
            entries = list(history.rows)
            if len(entries) < self.window_size:
                entries = [entries[0]] * (self.window_size - len(entries)) + entries
            data = np.stack([entry[2] for entry in entries])
            frame = pd.DataFrame(data, columns=USEFUL_SENSORS)
            frame.insert(0, "cycle", [entry[0] for entry in entries])
            return frame

    def stored_samples(self, asset_id: Optional[str] = None) -> int:
        with self._lock:
            if asset_id is not None:
                history = self._assets.get(asset_id)
                return 0 if history is None else len(history.rows)
            return sum(len(history.rows) for history in self._assets.values())

    @property
    def asset_count(self) -> int:
        with self._lock:
            return len(self._assets)

    @property
    def asset_ids(self) -> tuple:
        with self._lock:
            return tuple(self._assets)


@dataclass(frozen=True)
class AlertDecision:
    observed_severity: str
    severity: str
    changed: bool
    pending_cycles: int


@dataclass
class _DebounceState:
    current: str = "healthy"
    pending: Optional[str] = None
    count: int = 0


class AlertManager:
    """Consecutive-cycle debounce for escalation and recovery transitions."""

    def __init__(self, confirm_cycles: int = 3):
        if confirm_cycles <= 0:
            raise ValueError("confirm_cycles must be positive")
        self.confirm_cycles = int(confirm_cycles)
        self._states: Dict[str, _DebounceState] = {}
        self._lock = threading.Lock()

    def update(self, asset_id: str, severity: str) -> AlertDecision:
        if severity not in _SEVERITIES:
            raise ValueError(f"unknown severity {severity!r}")
        with self._lock:
            state = self._states.setdefault(asset_id, _DebounceState())
            if severity == state.current:
                state.pending, state.count = None, 0
                return AlertDecision(severity, state.current, False, 0)
            if severity == state.pending:
                state.count += 1
            else:
                state.pending, state.count = severity, 1
            if state.count < self.confirm_cycles:
                return AlertDecision(severity, state.current, False, state.count)
            state.current = severity
            state.pending, state.count = None, 0
            return AlertDecision(severity, state.current, True, 0)

    def current(self, asset_id: str) -> str:
        with self._lock:
            return self._states.get(asset_id, _DebounceState()).current

    def clear(self, asset_id: str) -> None:
        with self._lock:
            self._states.pop(asset_id, None)


def _validate_alert_payload(payload: Mapping[str, object]) -> Dict[str, object]:
    if set(payload) != ALERT_PAYLOAD_FIELDS:
        missing = ALERT_PAYLOAD_FIELDS - set(payload)
        extra = set(payload) - ALERT_PAYLOAD_FIELDS
        raise AssertionError(f"unsafe alert payload fields; missing={sorted(missing)}, extra={sorted(extra)}")
    privacy = payload.get("privacy")
    if not isinstance(privacy, Mapping):
        raise AssertionError("alert privacy metadata must be an object")
    if privacy.get("contains_raw_sensor_values") is not False:
        raise AssertionError("alert privacy metadata must declare zero raw sensor values")
    if privacy.get("raw_sensor_bytes_transmitted") != 0:
        raise AssertionError("alert must transmit zero raw sensor bytes")

    forbidden_keys = set(USEFUL_SENSORS) | {"values", "window", "raw_values", "sensor_values"}

    def check(value: object) -> None:
        if isinstance(value, Mapping):
            forbidden = forbidden_keys.intersection(value)
            if forbidden:
                raise AssertionError(f"raw telemetry field(s) in alert: {sorted(forbidden)}")
            for nested in value.values():
                check(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                check(nested)

    check(payload)
    # JSON round-trip creates a detached, plain-data object for sinks.
    return json.loads(json.dumps(dict(payload), allow_nan=False))


class AlertSink(ABC):
    @abstractmethod
    def emit(self, payload: Mapping[str, object]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        return None


class JsonLinesAlertSink(AlertSink):
    def __init__(self, destination: Union[PathLike, TextIO] = sys.stdout):
        self._owned = isinstance(destination, (str, Path))
        self._stream = (
            open(destination, "a", encoding="utf-8") if self._owned else destination
        )
        self._lock = threading.Lock()

    def emit(self, payload: Mapping[str, object]) -> None:
        safe = _validate_alert_payload(payload)
        with self._lock:
            self._stream.write(json.dumps(safe, sort_keys=True, allow_nan=False) + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self._owned and not self._stream.closed:
            self._stream.close()


class WebhookDashboardSink(AlertSink):
    """POST aggregate alerts to a dashboard using only the standard library."""

    def __init__(self, url: str, timeout: float = 5.0, opener=None):
        self.url, self.timeout = url, float(timeout)
        self._opener = opener or request.urlopen

    def emit(self, payload: Mapping[str, object]) -> None:
        safe = _validate_alert_payload(payload)
        body = json.dumps(safe, allow_nan=False).encode("utf-8")
        req = request.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        response = self._opener(req, timeout=self.timeout)
        if hasattr(response, "__enter__"):
            with response as opened:
                opened.read()
        elif hasattr(response, "read"):
            response.read()


class EdgeInferenceRunner:
    """One-pass window inference, diagnostics, debounce, and safe alert emission."""

    def __init__(self, diagnostics: EdgeDiagnostics, alert_sink: Optional[AlertSink] = None,
                 buffer: Optional[RollingWindowBuffer] = None,
                 alert_manager: Optional[AlertManager] = None):
        self.diagnostics = diagnostics
        self.alert_sink = alert_sink
        self.buffer = buffer or RollingWindowBuffer()
        self.alert_manager = alert_manager or AlertManager()

    def process_sample(self, sample: TelemetrySample) -> dict:
        started = time.perf_counter()
        self.buffer.append(sample)
        frame = self.buffer.frame(sample.asset_id)
        diagnosed = self.diagnostics.diagnose_engine(frame, sample.asset_id)

        # A regressor is unconstrained; the operational prediction is bounded by
        # the same target cap used during C-MAPSS training.
        predicted_rul = float(np.clip(diagnosed["pred_rul"][-1], 0.0, float(MAX_RUL)))
        diagnosed["pred_rul"] = diagnosed["pred_rul"].copy()
        diagnosed["pred_rul"][-1] = predicted_rul
        summary = self.diagnostics.summarize(diagnosed)
        observed = severity_for(predicted_rul)
        decision = self.alert_manager.update(sample.asset_id, observed)

        payload: Dict[str, object] = {
            "asset_id": sample.asset_id,
            "cycle": int(sample.cycle),
            "predicted_rul": round(predicted_rul, 2),
            "severity": decision.severity,
            "primary_subsystem": summary["primary_subsystem"],
            "subsystem_shares_pct": summary["subsystem_shares_pct"],
            "top_contributors": summary["top_contributors"],
            "top_drifting": summary["top_drifting"],
            "summary_text": summary["text"]["en"],
            "privacy": summary["privacy"],
        }
        safe_payload = _validate_alert_payload(payload)
        emitted = False
        if decision.changed and self.alert_sink is not None:
            self.alert_sink.emit(safe_payload)
            emitted = True
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "asset_id": sample.asset_id,
            "cycle": int(sample.cycle),
            "predicted_rul": round(predicted_rul, 2),
            "observed_severity": observed,
            "severity": decision.severity,
            "pending_cycles": decision.pending_cycles,
            "alert_changed": decision.changed,
            "alert_emitted": emitted,
            "latency_ms": latency_ms,
            "alert_payload": safe_payload,
        }

    def run(self, source: TelemetrySource, max_samples: Optional[int] = None,
            interval_ms: float = 0.0) -> Iterator[dict]:
        for index, sample in enumerate(source):
            if max_samples is not None and index >= max_samples:
                break
            yield self.process_sample(sample)
            if interval_ms > 0:
                time.sleep(interval_ms / 1000.0)


def _neutral_baseline(scales: Mapping[str, Mapping[str, float]]) -> HealthyBaseline:
    """Finite CLI fallback; deployments should fit and persist their local baseline."""
    minimum = np.array([scales["min"][name] for name in USEFUL_SENSORS], dtype=float)
    maximum = np.array([scales["max"][name] for name in USEFUL_SENSORS], dtype=float)
    midpoint = (minimum + maximum) / 2.0
    spread = np.maximum(maximum - minimum, 1.0)
    setting_index = [USEFUL_SENSORS.index(name) for name in ("setting1", "setting2", "setting3")]
    fault_index = [USEFUL_SENSORS.index(name) for name in FAULT_SENSORS]
    return HealthyBaseline(
        centroids=np.zeros((1, 3)),
        set_mean=midpoint[setting_index],
        set_std=spread[setting_index],
        regime_mean=midpoint[fault_index][None, :],
        regime_std=spread[fault_index][None, :],
        healthy_sigma=np.ones(len(FAULT_SENSORS)),
    )


def _load_model(path: Optional[str]) -> CmapssRegressor:
    model = CmapssRegressor()
    if path:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility
            state = torch.load(path, map_location="cpu")
        if isinstance(state, Mapping) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    model.eval()
    return model


def _cmapss_samples(frame: pd.DataFrame, subset: str) -> Iterator[TelemetrySample]:
    for unit, group in frame.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        for row in group.itertuples(index=False):
            values = {sensor: float(getattr(row, sensor)) for sensor in USEFUL_SENSORS}
            cycle = int(row.cycle)
            yield TelemetrySample(f"{subset}-U{int(unit):03d}", cycle, float(cycle), values)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Streaming local RUL diagnostics and debounced alerts")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--jsonl", metavar="PATH", help="newline-delimited telemetry JSON")
    source_group.add_argument("--replay-cmapss", choices=("FD001", "FD002", "FD003", "FD004"))
    parser.add_argument("--model", help="CmapssRegressor state_dict/checkpoint")
    parser.add_argument("--scales", default="global_scale.json", help="model min/max scale JSON")
    parser.add_argument("--alerts", default="-", help="alert JSONL output path (default: stdout)")
    parser.add_argument("--confirm-cycles", type=int, default=3)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--interval-ms", type=float, default=0.0, help="replay delay per sample")
    args = parser.parse_args(list(argv) if argv is not None else None)

    scales = load_scales(args.scales)
    model = _load_model(args.model)
    if not args.model:
        print("warning: no --model supplied; using an untrained demonstration model", file=sys.stderr)

    if args.replay_cmapss:
        train = load_cmapss_raw(f"train_{args.replay_cmapss}.txt")
        baseline = fit_healthy_baseline(train, n_regimes=1 if args.replay_cmapss in ("FD001", "FD003") else 6)
        source: TelemetrySource = _IterableTelemetrySource(
            _cmapss_samples(load_cmapss_raw(f"test_{args.replay_cmapss}.txt"), args.replay_cmapss)
        )
    else:
        baseline = _neutral_baseline(scales)
        print(
            "warning: JSONL mode is using a scale-derived baseline; fit HealthyBaseline on local healthy history "
            "for production drift scores",
            file=sys.stderr,
        )
        source = JsonLinesStreamSource(args.jsonl)

    destination: Union[str, TextIO] = sys.stdout if args.alerts == "-" else args.alerts
    sink = JsonLinesAlertSink(destination)
    runner = EdgeInferenceRunner(
        EdgeDiagnostics(model, scales, baseline),
        alert_sink=sink,
        alert_manager=AlertManager(args.confirm_cycles),
    )
    try:
        for _ in runner.run(source, max_samples=args.max_samples, interval_ms=args.interval_ms):
            pass
    finally:
        sink.close()
    return 0


class _IterableTelemetrySource(TelemetrySource):
    """Internal adapter used by the C-MAPSS CLI replay generator."""

    def __init__(self, samples: Iterable[TelemetrySample]):
        self.samples = samples

    def __iter__(self) -> Iterator[TelemetrySample]:
        return iter(self.samples)


if __name__ == "__main__":
    raise SystemExit(main())
