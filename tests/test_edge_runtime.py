"""Focused tests for the streaming edge runtime; no broker or C-MAPSS ZIP required."""
import io
import json
import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import SEQ_LEN, USEFUL_SENSORS  # noqa: E402
from diagnostics import FAULT_SENSORS, EdgeDiagnostics, HealthyBaseline  # noqa: E402
from edge_runtime import (  # noqa: E402
    ALERT_PAYLOAD_FIELDS,
    AlertManager,
    EdgeInferenceRunner,
    JsonLinesAlertSink,
    JsonLinesStreamSource,
    MqttSource,
    RollingWindowBuffer,
    TelemetrySample,
)


def values(base=10.0):
    return {sensor: base + index for index, sensor in enumerate(USEFUL_SENSORS)}


def sample(asset, cycle, sensor_values=None):
    return TelemetrySample(asset, cycle, float(cycle), values() if sensor_values is None else sensor_values)


class RollingWindowBufferTest(unittest.TestCase):
    def test_cold_start_repeats_first_reading(self):
        buffer = RollingWindowBuffer()
        first = values(1.0)
        second = values(2.0)
        buffer.append(sample("A", 1, first))
        buffer.append(sample("A", 2, second))
        window = buffer.window("A")
        self.assertEqual(window.shape, (SEQ_LEN, len(USEFUL_SENSORS)))
        expected_first = np.array([first[name] for name in USEFUL_SENSORS])
        expected_second = np.array([second[name] for name in USEFUL_SENSORS])
        np.testing.assert_allclose(window[:-1], np.repeat(expected_first[None, :], SEQ_LEN - 1, axis=0))
        np.testing.assert_allclose(window[-1], expected_second)

    def test_steady_state_slides_at_fixed_size(self):
        buffer = RollingWindowBuffer()
        for cycle in range(1, SEQ_LEN + 8):
            buffer.append(sample("A", cycle, values(float(cycle))))
        window = buffer.window("A")
        self.assertEqual(buffer.stored_samples("A"), SEQ_LEN)
        self.assertEqual(window[0, 0], 8.0)
        self.assertEqual(window[-1, 0], float(SEQ_LEN + 7))

    def test_nan_and_missing_channels_use_last_observation(self):
        buffer = RollingWindowBuffer()
        buffer.append(sample("A", 1, values(5.0)))
        changed = {"setting1": float("nan"), "s2": 999.0}
        imputed = buffer.append(sample("A", 2, changed))
        self.assertTrue(np.isfinite(imputed).all())
        self.assertEqual(imputed[USEFUL_SENSORS.index("setting1")], 5.0)
        self.assertEqual(imputed[USEFUL_SENSORS.index("s2")], 999.0)
        self.assertEqual(imputed[USEFUL_SENSORS.index("s3")], values(5.0)["s3"])

        fresh = RollingWindowBuffer()
        first = fresh.append(sample("B", 1, {"s2": math.nan}))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_allclose(first, 0.0)

    def test_multiple_assets_are_independent_and_bounded(self):
        buffer = RollingWindowBuffer(max_assets=3)
        for asset_index, asset in enumerate(("A", "B", "C")):
            for cycle in range(1, SEQ_LEN + 6):
                buffer.append(sample(asset, cycle, values(100.0 * asset_index + cycle)))
        self.assertEqual(buffer.asset_count, 3)
        self.assertEqual(buffer.stored_samples(), 3 * SEQ_LEN)
        self.assertEqual(buffer.window("A")[-1, 0], SEQ_LEN + 5)
        self.assertEqual(buffer.window("C")[-1, 0], 200 + SEQ_LEN + 5)

        buffer.append(sample("D", 1))
        self.assertEqual(buffer.asset_count, 3)
        self.assertNotIn("B", buffer.asset_ids)  # A was just touched, so B is least-recently-used.
        self.assertLessEqual(buffer.stored_samples(), 3 * SEQ_LEN)

    def test_non_monotonic_cycle_or_timestamp_is_rejected(self):
        buffer = RollingWindowBuffer()
        buffer.append(sample("A", 2))
        with self.assertRaisesRegex(ValueError, "non-monotonic cycle"):
            buffer.append(sample("A", 2))
        with self.assertRaisesRegex(ValueError, "non-monotonic timestamp"):
            buffer.append(TelemetrySample("A", 3, 1.0, values()))


class AlertManagerTest(unittest.TestCase):
    def test_escalation_and_recovery_require_confirmation(self):
        manager = AlertManager(confirm_cycles=3)
        first = manager.update("A", "critical")
        second = manager.update("A", "critical")
        self.assertEqual((first.severity, second.severity), ("healthy", "healthy"))
        self.assertFalse(first.changed)
        self.assertFalse(second.changed)

        third = manager.update("A", "critical")
        self.assertTrue(third.changed)
        self.assertEqual(third.severity, "critical")

        for _ in range(2):
            pending = manager.update("A", "healthy")
            self.assertFalse(pending.changed)
            self.assertEqual(pending.severity, "critical")
        cleared = manager.update("A", "healthy")
        self.assertTrue(cleared.changed)
        self.assertEqual(cleared.severity, "healthy")

    def test_candidate_change_restarts_consecutive_count(self):
        manager = AlertManager(confirm_cycles=3)
        manager.update("A", "critical")
        manager.update("A", "critical")
        reset = manager.update("A", "warning")
        self.assertEqual(reset.pending_cycles, 1)
        self.assertEqual(reset.severity, "healthy")


class ConstantRulModel(torch.nn.Module):
    def __init__(self, rul):
        super().__init__()
        self.register_buffer("rul", torch.tensor(float(rul)))

    def forward(self, x):
        # Retain an input gradient so the real EdgeDiagnostics attribution path runs.
        return x.sum(dim=(1, 2)) * 0.0 + self.rul


def synthetic_diagnostics(rul=10.0):
    scales = {
        "min": {name: 0.0 for name in USEFUL_SENSORS},
        "max": {name: 100.0 for name in USEFUL_SENSORS},
    }
    baseline = HealthyBaseline(
        centroids=np.zeros((1, 3)),
        set_mean=np.zeros(3),
        set_std=np.ones(3),
        regime_mean=np.full((1, len(FAULT_SENSORS)), 10.0),
        regime_std=np.ones((1, len(FAULT_SENSORS))),
        healthy_sigma=np.ones(len(FAULT_SENSORS)),
    )
    return EdgeDiagnostics(ConstantRulModel(rul), scales, baseline)


class EdgeInferenceEndToEndTest(unittest.TestCase):
    def test_json_stream_to_json_alert_has_no_raw_telemetry(self):
        stream = io.StringIO()
        for cycle in range(1, 4):
            stream.write(
                json.dumps(
                    {
                        "asset_id": "PLC-7",
                        "cycle": cycle,
                        "timestamp": f"2026-01-01T00:00:0{cycle}Z",
                        "values": values(float(cycle)),
                    }
                )
                + "\n"
            )
        stream.seek(0)
        alert_output = io.StringIO()
        runner = EdgeInferenceRunner(
            synthetic_diagnostics(10.0),
            alert_sink=JsonLinesAlertSink(alert_output),
            alert_manager=AlertManager(confirm_cycles=3),
        )

        results = list(runner.run(JsonLinesStreamSource(stream)))
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result["latency_ms"] >= 0 for result in results))
        self.assertEqual([r["alert_emitted"] for r in results], [False, False, True])

        lines = alert_output.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        emitted = json.loads(lines[0])
        self.assertEqual(set(emitted), ALERT_PAYLOAD_FIELDS)
        self.assertEqual(emitted["severity"], "critical")
        self.assertIsInstance(emitted["summary_text"], str)
        self.assertAlmostEqual(sum(emitted["subsystem_shares_pct"].values()), 100.0, places=7)
        self.assertFalse(emitted["privacy"]["contains_raw_sensor_values"])
        self.assertEqual(emitted["privacy"]["raw_sensor_bytes_transmitted"], 0)
        serialized = json.dumps(emitted)
        self.assertNotIn('"values"', serialized)
        for sensor in USEFUL_SENSORS:
            self.assertNotIn(f'"{sensor}":', serialized)


class MqttSourceTest(unittest.TestCase):
    def test_message_parsing_with_injected_client(self):
        class FakeClient:
            on_message = None

        class FakeMessage:
            payload = json.dumps(
                {"asset_id": "MQTT-1", "cycle": 9, "timestamp": 123.5, "values": {"s2": 42.0}}
            ).encode("utf-8")
            topic = "factory/telemetry"

        source = MqttSource("factory/telemetry", client=FakeClient())
        parsed = source.parse_message(FakeMessage())
        self.assertEqual(parsed.asset_id, "MQTT-1")
        self.assertEqual(parsed.cycle, 9)
        self.assertEqual(parsed.timestamp, 123.5)
        self.assertEqual(parsed.values, {"s2": 42.0})


if __name__ == "__main__":
    unittest.main()
