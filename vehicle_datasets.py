"""Opt-in fleet adapters; no downloads or changes to the baseline pipeline.

Frames retain raw telemetry, time order and uncapped ``rul`` labels. Only
``feature_cols`` enter a CNN. See docs/DATASET_EVALUATION.md for provenance,
censoring, units, download instructions and memory limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from torch import nn

from common import CmapssRegressor, USEFUL_SENSORS, load_cmapss_raw, load_rul
from fed_norm import sanitize_telemetry_frame


@dataclass
class VehicleFleetDataset:
    """One ordered frame and metadata record per anonymous vehicle.

    ``rul`` is NaN when no exact target is known (notably censored trucks).
    ``test_ruls`` holds uncapped endpoint labels, when supplied by the source.
    Partition vehicles BEFORE constructing windows or fitting training scalers.
    """

    name: str
    vehicles: dict[str, pd.DataFrame]
    feature_cols: list[str]
    setting_cols: list[str]
    metadata: dict[str, dict]
    test_ruls: dict[str, float] | None = None

    def partition_by(self, key: str) -> dict[str, list[str]]:
        """Group complete vehicles by a metadata field, never individual rows."""
        clients: dict[str, list[str]] = {}
        for vehicle in sorted(self.vehicles):
            value = self.metadata[vehicle][key]  # fail on absent metadata
            clients.setdefault(str(value), []).append(vehicle)
        return clients

    def windows(
        self, vehicle_ids: Iterable[str] | None = None, *, seq_len: int = 30,
        max_rul: float = 125.0, stride: int = 1, last_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return float32 ``(N, W, C), (N,)`` arrays for selected vehicles.

        Sliding targets belong to the LAST readout of each window. Unknown
        targets are omitted, including all censored Scania vehicles. Short
        trajectories are omitted in sliding mode; last_only repeats the first
        reading on the left, matching common.extract_last_sequence.

        The shared sanitizer runs separately inside each observed window, so
        backfill/means cannot use future readouts or another vehicle. This is
        deliberately not a scaler: fit normalization on training clients only.
        Materializes windows; select clients/stride for large external files.
        """
        _positive_integer(seq_len, "seq_len")
        _positive_integer(stride, "stride")
        if not np.isfinite(max_rul) or max_rul <= 0:
            raise ValueError("max_rul must be finite and positive")
        ids = list(self.vehicles if vehicle_ids is None else vehicle_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("vehicle_ids must be unique")
        xs, ys = [], []
        for vehicle in ids:
            frame = self.vehicles[vehicle]
            if frame.empty:
                continue
            ends = [len(frame)] if last_only else range(seq_len, len(frame) + 1, stride)
            for end in ends:
                target = float(frame["rul"].iloc[end - 1])
                if not np.isfinite(target):
                    continue
                if target < 0:
                    raise ValueError(f"negative RUL for {vehicle}")
                observed = frame.iloc[max(0, end - seq_len):end]
                # The existing sanitizer inserts columns individually. Bounded
                # channel blocks avoid pandas fragmentation warnings for 105+
                # sensors, without changing its independent per-channel logic.
                clean = np.concatenate([
                    sanitize_telemetry_frame(observed, sensors=self.feature_cols[i:i + 64])[0]
                    for i in range(0, len(self.feature_cols), 64)
                ], axis=1)
                if len(clean) < seq_len:
                    clean = np.concatenate([np.repeat(clean[:1], seq_len - len(clean), axis=0), clean])
                xs.append(clean)
                ys.append(min(target, max_rul))
        shape = (0, seq_len, len(self.feature_cols))
        return (np.stack(xs).astype(np.float32) if xs else np.empty(shape, dtype=np.float32),
                np.asarray(ys, dtype=np.float32))


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def build_flexible_cnn(in_channels: int, seq_len: int) -> nn.Module:
    """The baseline Conv32/Conv64/FC32 CNN for input ``(batch, W, C)``.

    At (17, 30), parameter names, shapes and forward computation match
    common.CmapssRegressor exactly (22,273 parameters). Two max pools require
    W >= 4. Each federated run must use the SAME feature order and W everywhere.
    """
    _positive_integer(in_channels, "in_channels")
    _positive_integer(seq_len, "seq_len")
    if seq_len < 4:
        raise ValueError("seq_len must be at least 4 for two max pools")
    model = CmapssRegressor()
    if in_channels != len(USEFUL_SENSORS):
        model.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
    if seq_len // 4 != 7:
        model.fc1 = nn.Linear(64 * (seq_len // 4), 32)
    return model


class CmapssCombinedAdapter:
    """Reuse common's ZIP resolution ($CMAPSS_ZIP) and raw/RUL readers.

    ``load()`` loads all four training subsets, or select e.g. ``load('FD004')``.
    ``split='both'`` includes held-out engines but keeps them separately marked;
    never train on those rows. IDs always include split, even for a single split.
    """

    SUBSETS = {"FD001": (1, 1), "FD002": (6, 1), "FD003": (1, 2), "FD004": (6, 2)}

    def load(self, subsets: str | Iterable[str] | None = None, *, split: str = "train") -> VehicleFleetDataset:
        selected = list(self.SUBSETS) if subsets is None else ([subsets] if isinstance(subsets, str) else list(subsets))
        if not selected or len(set(selected)) != len(selected) or any(s not in self.SUBSETS for s in selected):
            raise ValueError("subsets must be unique members of FD001..FD004")
        if split not in ("train", "test", "both"):
            raise ValueError("split must be train, test or both")
        vehicles, metadata, test_ruls = {}, {}, {}
        for subset in selected:
            conditions, faults = self.SUBSETS[subset]
            for part in (("train", "test") if split == "both" else (split,)):
                raw = load_cmapss_raw(f"{part}_{subset}.txt")
                ruls = load_rul(f"RUL_{subset}.txt") if part == "test" else None
                if ruls is not None and len(ruls) != raw.unit.nunique():
                    raise ValueError(f"{subset} test units and RUL labels do not align")
                for unit, group in raw.groupby("unit", sort=True):
                    vehicle = f"{subset}-{part}-U{int(unit):03d}"
                    frame = group.sort_values("cycle").reset_index(drop=True).copy()
                    endpoint = float(ruls[int(unit) - 1]) if ruls is not None else 0.0
                    frame["rul"] = frame.cycle.max() - frame.cycle + endpoint
                    vehicles[vehicle] = frame
                    metadata[vehicle] = {
                        "subset": subset, "split": part, "source_unit": int(unit),
                        "operating_conditions": conditions, "fault_mode_count": faults,
                        # Subset-level capability, NOT a per-engine fault label.
                        "condition_group": f"{conditions}-regime/{faults}-mode",
                        "rul_unit": "cycles",
                    }
                    if ruls is not None:
                        test_ruls[vehicle] = endpoint
        return VehicleFleetDataset("C-MAPSS-" + "+".join(selected), vehicles,
                                   list(USEFUL_SENSORS), [f"setting{i}" for i in range(1, 4)],
                                   metadata, test_ruls or None)


SCANIA_FEATURE_COLS = [f"{variable}_{i}" for variable, bins in
                       ((167, 10), (272, 10), (291, 11), (158, 10), (459, 20), (397, 36))
                       for i in range(bins)] + [f"{v}_0" for v in (171, 666, 427, 837, 309, 835, 370, 100)]
SCANIA_SPEC_COLS = [f"Spec_{i}" for i in range(8)]
SCANIA_DOWNLOAD = "https://researchdata.se/en/catalogue/dataset/2024-34"
NCMAPSS_DOWNLOAD = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


class ScaniaComponentXAdapter:
    """Read the published CSV schema. Censoring is never converted to failure.

    Train needs *_tte.csv. Validation/test instead have *_labels.csv, with
    five interval classes, NOT exact RUL. Those frames retain class_label in
    metadata and NaN rul; windows() cannot invent point targets for them.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _read(self, filename: str) -> pd.DataFrame:
        path = self.directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Download Scania Component X CSVs from {SCANIA_DOWNLOAD}")
        frame = pd.read_csv(path, dtype={"vehicle_id": str}, na_values=["na", "null"])
        _require_columns(frame, ["vehicle_id"], filename)
        if frame.vehicle_id.isna().any():
            raise ValueError(f"{filename}: missing vehicle_id")
        return frame

    def load(self, *, split: str = "train") -> VehicleFleetDataset:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        readouts = self._read(f"{split}_operational_readouts.csv")
        specs = self._read(f"{split}_specifications.csv")
        outcomes = self._read(f"{split}_tte.csv" if split == "train" else f"{split}_labels.csv")
        _require_columns(readouts, ["time_step", *SCANIA_FEATURE_COLS], "operational readouts")
        _require_columns(specs, SCANIA_SPEC_COLS, "specifications")
        outcome_cols = ["length_of_study_time_step", "in_study_repair"] if split == "train" else ["class_label"]
        _require_columns(outcomes, outcome_cols, "outcomes")
        for table in (specs, outcomes):
            if table.vehicle_id.duplicated().any():
                raise ValueError("specifications/outcomes must have exactly one row per vehicle_id")
            if set(table.vehicle_id) != set(readouts.vehicle_id):
                raise ValueError("readouts/specifications/outcomes vehicle IDs must match")
        specs, outcomes = specs.set_index("vehicle_id"), outcomes.set_index("vehicle_id")
        time = pd.to_numeric(readouts.time_step, errors="coerce")
        if not np.isfinite(time).all() or (time < 0).any():
            raise ValueError("time_step must be finite and nonnegative")
        readouts["time_step"] = time
        if readouts.duplicated(["vehicle_id", "time_step"]).any():
            raise ValueError("duplicate vehicle_id/time_step readouts")
        vehicles, metadata = {}, {}
        for unit, group in readouts.groupby("vehicle_id", sort=True):
            vehicle = f"Scania-{split}-{unit}"
            frame = group.sort_values("time_step").reset_index(drop=True).copy()
            # Restrict inputs to the public sensor schema; outcomes/specs never enter features.
            frame = frame[["vehicle_id", "time_step", *SCANIA_FEATURE_COLS]]
            meta = {"source_unit": unit, "split": split, "rul_unit": "anonymized time_step",
                    **{col: ("<missing>" if pd.isna(specs.loc[unit, col]) else str(specs.loc[unit, col]))
                       for col in SCANIA_SPEC_COLS}}
            frame["rul"] = np.nan
            if split == "train":
                event = float(outcomes.loc[unit, "in_study_repair"])
                end = float(outcomes.loc[unit, "length_of_study_time_step"])
                if event not in (0.0, 1.0) or not np.isfinite(end) or end < frame.time_step.max():
                    raise ValueError(f"invalid repair indicator/study end for vehicle {unit}")
                frame["tte_lower_bound"] = end - frame.time_step
                meta.update(event_observed=bool(event), study_end=end)
                if event:
                    frame["rul"] = frame["tte_lower_bound"]
            else:
                label = float(outcomes.loc[unit, "class_label"])
                if label not in range(5):
                    raise ValueError(f"invalid class_label for vehicle {unit}")
                meta["class_label"] = int(label)
            vehicles[vehicle], metadata[vehicle] = frame, meta
        return VehicleFleetDataset("Scania-Component-X-" + split, vehicles,
                                   list(SCANIA_FEATURE_COLS), [], metadata)

    @staticmethod
    def partition_by_specifications(
        fleet: VehicleFleetDataset, spec_cols: Iterable[str] = ("Spec_0",),
    ) -> dict[str, list[str]]:
        """Exact categorical specification tuples; no learned/full-fleet clustering.

        Choose a small subset to avoid singleton clients. Alternatively attach
        an operational_profile from a predefined training prefix and call
        fleet.partition_by('operational_profile'). Customer IDs are unavailable.
        """
        cols = list(spec_cols)
        if not cols or len(set(cols)) != len(cols) or any(c not in SCANIA_SPEC_COLS for c in cols):
            raise ValueError("spec_cols must be unique Spec_0..Spec_7 names")
        clients: dict[str, list[str]] = {}
        for vehicle in sorted(fleet.vehicles):
            # repr(tuple) avoids ambiguous delimiter concatenation.
            key = repr(tuple((c, fleet.metadata[vehicle][c]) for c in cols))
            clients.setdefault(key, []).append(vehicle)
        return clients


def make_synthetic_scania_fleet(
    directory: str | Path, *, n_vehicles: int = 6, min_readouts: int = 5,
    max_readouts: int = 10, seed: int = 42,
) -> Path:
    """Write SYNTHETIC train-schema fixtures ONLY for tests/schema verification.

    Not sampled from Scania; not a real fleet and not benchmark evidence.
    Includes unequal counts, irregular times, sensor NaNs and censored trucks.
    """
    for name, value in (("n_vehicles", n_vehicles), ("min_readouts", min_readouts), ("max_readouts", max_readouts)):
        _positive_integer(value, name)
    if max_readouts < min_readouts:
        raise ValueError("max_readouts must be >= min_readouts")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    readouts, specs, outcomes = [], [], []
    for i in range(n_vehicles):
        count = min_readouts + i % (max_readouts - min_readouts + 1)
        times = np.cumsum(rng.integers(1, 5, count)).astype(float)
        sensors = np.cumsum(rng.uniform(0.1, 2.0, (count, len(SCANIA_FEATURE_COLS))), axis=0)
        sensors[0, 0] = np.nan
        for t, row in zip(times, sensors):
            readouts.append({"vehicle_id": i, "time_step": t, **dict(zip(SCANIA_FEATURE_COLS, row))})
        specs.append({"vehicle_id": i, **{s: f"Cat{(i + j) % 2}" for j, s in enumerate(SCANIA_SPEC_COLS)}})
        outcomes.append({"vehicle_id": i, "length_of_study_time_step": times[-1] + 8,
                         "in_study_repair": int(i % 3 != 2)})
    for suffix, rows in (("operational_readouts", readouts), ("specifications", specs), ("tte", outcomes)):
        pd.DataFrame(rows).to_csv(directory / f"train_{suffix}.csv", index=False)
    return directory


class NCMAPSSAdapter:
    """Load official W/X_s/X_v/T/Y/A_{dev,test} arrays and *_var names.

    h5py is optional and imported only by load(). Reads in chunks, but returns
    an in-memory fleet: select one file and row_stride for manageable probes.
    T health truth and A health state NEVER enter feature_cols. X_v is retained
    as virtual telemetry and only enters features with explicit opt-in.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, *, split: str = "dev", include_virtual: bool = False,
             row_stride: int = 1, chunk_rows: int = 100_000) -> VehicleFleetDataset:
        if split not in ("dev", "test"):
            raise ValueError("split must be dev or test")
        _positive_integer(row_stride, "row_stride")
        _positive_integer(chunk_rows, "chunk_rows")
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing N-CMAPSS HDF5 file {self.path}. Download NASA repository item 17: {NCMAPSS_DOWNLOAD}; extract a .h5 file, then install h5py.")
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("NCMAPSSAdapter requires optional h5py; install with python3 -m pip install h5py") from exc
        pieces: dict[int, list[pd.DataFrame]] = {}
        groups = ("W", "X_s", "X_v", "T", "A", "Y")
        with h5py.File(self.path, "r") as handle:
            required = [f"{g}_{split}" for g in groups] + [f"{g}_var" for g in groups if g != "Y"]
            missing = set(required) - set(handle.keys())
            if missing:
                raise ValueError(f"N-CMAPSS missing HDF5 datasets: {sorted(missing)}")
            names = {}
            nrows = handle[f"Y_{split}"].shape[0]
            for group in groups:
                array = handle[f"{group}_{split}"]
                if array.ndim != 2 or array.shape[0] != nrows:
                    raise ValueError(f"N-CMAPSS {group}_{split}: expected aligned 2D arrays")
                if group == "Y":
                    if array.shape[1] != 1:
                        raise ValueError("N-CMAPSS Y must have one RUL column")
                    names[group] = ["rul"]
                else:
                    raw_names = np.asarray(handle[f"{group}_var"]).ravel()
                    decoded = [n.decode("utf-8").strip() if isinstance(n, bytes) else str(n).strip() for n in raw_names]
                    if len(decoded) != array.shape[1] or len(set(decoded)) != len(decoded):
                        raise ValueError(f"N-CMAPSS {group}_var names do not match columns")
                    names[group] = decoded if group == "A" else [f"{group}:{n}" for n in decoded]
            if not {"unit", "cycle", "Fc", "hs"}.issubset(names["A"]):
                raise ValueError("N-CMAPSS A_var must include unit, cycle, Fc, hs")
            all_names = [n for g in groups for n in names[g]]
            if len(set(all_names)) != len(all_names):
                raise ValueError("N-CMAPSS column names collide")
            # Each chunk contains chunk_rows retained readings; stride is global.
            for start in range(0, nrows, chunk_rows * row_stride):
                selection = slice(start, min(nrows, start + chunk_rows * row_stride), row_stride)
                frame = pd.DataFrame(np.concatenate([handle[f"{g}_{split}"][selection] for g in groups], axis=1), columns=all_names)
                if (not np.isfinite(frame[["unit", "cycle", "Fc", "rul"]]).all().all()
                        or (frame.rul < 0).any() or (frame.cycle < 0).any()
                        or not frame.Fc.isin([1, 2, 3]).all()
                        or (frame.unit < 1).any() or (frame.unit % 1 != 0).any()):
                    raise ValueError("N-CMAPSS invalid unit/cycle/Fc/RUL values")
                for unit, part in frame.groupby("unit", sort=False):
                    pieces.setdefault(int(unit), []).append(part)
        vehicles, metadata, test_ruls = {}, {}, {}
        for unit, parts in pieces.items():
            # Preserve file/sample order within repeated flight cycle numbers.
            frame = pd.concat(parts, ignore_index=True)
            if not frame.cycle.is_monotonic_increasing or frame.Fc.nunique() != 1:
                raise ValueError(f"N-CMAPSS unit {unit}: cycles must be ordered and flight class constant")
            vehicle = f"{self.path.stem}-{split}-U{unit:03d}"
            fc = int(frame.Fc.iloc[0])
            vehicles[vehicle] = frame
            metadata[vehicle] = {"source_unit": unit, "source_file": self.path.name, "split": split,
                                 "Fc": fc, "flight_class": {1: "short", 2: "medium", 3: "long"}[fc],
                                 "health_parameter_cols": names["T"], "virtual_cols": names["X_v"],
                                 "row_stride": row_stride, "rul_unit": "flight cycles"}
            if split == "test":
                test_ruls[vehicle] = float(frame.rul.iloc[-1])
        features = names["W"] + names["X_s"] + (names["X_v"] if include_virtual else [])
        return VehicleFleetDataset("N-CMAPSS-" + self.path.stem + "-" + split, vehicles,
                                   features, names["W"], metadata, test_ruls or None)

    @staticmethod
    def partition_by_flight_class(fleet: VehicleFleetDataset) -> dict[str, list[str]]:
        """Fc=1/2/3 are short/medium/long flights; clients own disjoint units."""
        return fleet.partition_by("Fc")
