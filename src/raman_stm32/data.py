"""Dataset inspection, leakage-safe splitting, and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_dirs, project_path
from .constants import CLASS_NAMES, CLASS_TO_INDEX, RAW_FILES, SPECTRAL_LENGTH


@dataclass(frozen=True)
class RawDataset:
    spectra: np.ndarray
    metadata: pd.DataFrame
    raman_shift_cm1: np.ndarray


def _fraction_text(value: float) -> str:
    return np.format_float_positional(float(value), trim="-")


def load_raw_dataset(raw_dir: str | Path) -> RawDataset:
    """Load only the three experimental workbooks; never load the simulation."""
    raw_path = Path(raw_dir)
    spectra_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    reference_axis: np.ndarray | None = None

    for class_name in CLASS_NAMES:
        workbook = raw_path / RAW_FILES[class_name]
        if not workbook.is_file():
            raise FileNotFoundError(
                f"Missing {workbook}. Download dataset version 1 from "
                "https://data.mendeley.com/datasets/jtk7rv77td/1"
            )
        frame = pd.read_excel(workbook, engine="openpyxl")
        if list(frame.columns[:2]) != ["Concentration", "Number"]:
            raise ValueError(f"Unexpected metadata columns in {workbook}: {list(frame.columns[:2])}")
        if frame.shape[1] - 2 != SPECTRAL_LENGTH:
            raise ValueError(
                f"Expected {SPECTRAL_LENGTH} spectral variables in {workbook}, "
                f"found {frame.shape[1] - 2}"
            )

        axis = np.asarray([float(value) for value in frame.columns[2:]], dtype=np.float64)
        if reference_axis is None:
            reference_axis = axis
        elif not np.allclose(axis, reference_axis, rtol=0.0, atol=1e-9):
            raise ValueError(f"Raman-shift axis differs in {workbook}")

        values = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"NaN or infinite spectral values found in {workbook}")
        concentrations = frame["Concentration"].to_numpy(dtype=np.float64)
        numbers = frame["Number"].to_numpy(dtype=np.int64)
        metadata_parts.append(
            pd.DataFrame(
                {
                    "sample_id": [f"{class_name}-{number:04d}" for number in numbers],
                    "compound": class_name,
                    "label": CLASS_TO_INDEX[class_name],
                    "number": numbers,
                    "concentration_fraction": concentrations,
                    "concentration_percent": concentrations * 100.0,
                    "group_id": [f"concentration-{_fraction_text(value)}" for value in concentrations],
                    "source_file": RAW_FILES[class_name],
                }
            )
        )
        spectra_parts.append(values)

    assert reference_axis is not None
    spectra = np.concatenate(spectra_parts, axis=0)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    return RawDataset(spectra=spectra, metadata=metadata, raman_shift_cm1=reference_axis)


def inspect_dataset(dataset: RawDataset) -> dict[str, Any]:
    per_compound: dict[str, Any] = {}
    for compound in CLASS_NAMES:
        class_meta = dataset.metadata[dataset.metadata["compound"] == compound]
        numbers = set(int(value) for value in class_meta["number"])
        expected = set(range(min(numbers), max(numbers) + 1))
        counts = class_meta.groupby("concentration_fraction", sort=True).size()
        per_compound[compound] = {
            "spectra": int(len(class_meta)),
            "number_min": min(numbers),
            "number_max": max(numbers),
            "missing_numbers": sorted(expected - numbers),
            "concentration_counts": {_fraction_text(k): int(v) for k, v in counts.items()},
        }

    duplicate_rows = int(
        pd.DataFrame(dataset.spectra).duplicated(keep=False).sum()
    )
    return {
        "included_files": [RAW_FILES[name] for name in CLASS_NAMES],
        "excluded_main_task": "Linear_simulated_dataset.xlsx",
        "total_spectra": int(dataset.spectra.shape[0]),
        "spectral_variables": int(dataset.spectra.shape[1]),
        "raman_shift_cm1": {
            "minimum": float(dataset.raman_shift_cm1.min()),
            "maximum": float(dataset.raman_shift_cm1.max()),
            "strictly_increasing": bool(np.all(np.diff(dataset.raman_shift_cm1) > 0)),
        },
        "finite_values": bool(np.isfinite(dataset.spectra).all()),
        "exact_duplicate_spectra_rows_including_all_occurrences": duplicate_rows,
        "concentration_values_fraction": sorted(
            float(value) for value in dataset.metadata["concentration_fraction"].unique()
        ),
        "per_compound": per_compound,
        "grouping_limitation": (
            "The files contain concentration and a row index, but no acquisition/session ID. "
            "Concentration is therefore used as a conservative global experimental-group proxy."
        ),
    }


def _candidate_tiebreak(seed: int, validation: tuple[float, ...], test: tuple[float, ...]) -> int:
    text = f"{seed}|{validation}|{test}".encode("utf-8")
    return int.from_bytes(sha256(text).digest()[:8], "big")


def choose_group_splits(metadata: pd.DataFrame, seed: int) -> dict[str, list[float]]:
    """Choose 7/2/2 concentration groups using metadata counts only.

    Each validation and test set must contain at least one low (<=10%) and one
    high (>=25%) concentration. All compounds at a concentration stay together.
    """
    groups = tuple(sorted(float(x) for x in metadata["concentration_fraction"].unique()))
    if len(groups) != 11:
        raise ValueError(f"Expected 11 concentration groups, found {len(groups)}")
    counts = metadata.groupby(["concentration_fraction", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=groups, columns=range(len(CLASS_NAMES)), fill_value=0)
    totals_by_class = counts.to_numpy().sum(axis=0)
    total = int(totals_by_class.sum())
    targets = {"train": 0.70, "validation": 0.15, "test": 0.15}

    def has_range_coverage(candidate: tuple[float, ...]) -> bool:
        return any(x <= 0.10 for x in candidate) and any(x >= 0.25 for x in candidate)

    best: tuple[float, int, tuple[float, ...], tuple[float, ...]] | None = None
    for validation in combinations(groups, 2):
        if not has_range_coverage(validation):
            continue
        remaining = tuple(x for x in groups if x not in validation)
        for test in combinations(remaining, 2):
            if not has_range_coverage(test):
                continue
            train = tuple(x for x in remaining if x not in test)
            split_groups = {"train": train, "validation": validation, "test": test}
            score = 0.0
            for name, selected in split_groups.items():
                selected_counts = counts.loc[list(selected)].to_numpy().sum(axis=0)
                expected_class = totals_by_class * targets[name]
                score += float(np.mean(((selected_counts - expected_class) / totals_by_class) ** 2))
                score += float(((selected_counts.sum() - total * targets[name]) / total) ** 2)
            candidate = (score, _candidate_tiebreak(seed, validation, test), validation, test)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("No valid concentration-group split was found")
    validation, test = best[2], best[3]
    train = tuple(x for x in groups if x not in validation and x not in test)
    return {"train": list(train), "validation": list(validation), "test": list(test)}


def assign_splits(metadata: pd.DataFrame, groups: dict[str, list[float]]) -> pd.Series:
    lookup = {
        _fraction_text(concentration): split
        for split, concentrations in groups.items()
        for concentration in concentrations
    }
    assigned = metadata["concentration_fraction"].map(lambda value: lookup[_fraction_text(value)])
    if assigned.isna().any():
        raise RuntimeError("At least one sample was not assigned to a split")
    return assigned


def snv_clip_scale(
    spectra: np.ndarray,
    clip_standard_deviations: float = 8.0,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Per-spectrum standard-normal-variate preprocessing scaled to [-1, 1]."""
    values = np.asarray(spectra, dtype=np.float32)
    means = values.mean(axis=1, keepdims=True, dtype=np.float64)
    stds = values.std(axis=1, keepdims=True, dtype=np.float64)
    if np.any(stds < epsilon):
        raise ValueError("A spectrum has near-zero standard deviation")
    normalized = (values - means) / stds
    normalized = np.clip(normalized, -clip_standard_deviations, clip_standard_deviations)
    return (normalized / clip_standard_deviations).astype(np.float32)


def _save_model_zoo_csv(path: Path, features: np.ndarray, labels: np.ndarray | None) -> None:
    flat = features.reshape(features.shape[0], -1)
    if labels is not None:
        flat = np.column_stack([flat, labels.astype(np.int32)])
    pd.DataFrame(flat).to_csv(path, index=False, header=False, float_format="%.8g")


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    raw_dir = project_path(config, config["paths"]["raw_dir"])
    processed_dir = project_path(config, config["paths"]["processed_dir"])
    reports_dir = project_path(config, config["paths"]["reports_dir"])
    dataset = load_raw_dataset(raw_dir)
    split_groups = choose_group_splits(dataset.metadata, int(config["seed"]))
    metadata = dataset.metadata.copy()
    metadata["split"] = assign_splits(metadata, split_groups)

    pre = config["data"]["preprocessing"]
    features = snv_clip_scale(
        dataset.spectra,
        clip_standard_deviations=float(pre["clip_standard_deviations"]),
        epsilon=float(pre["epsilon"]),
    )
    features = features[:, np.newaxis, :, np.newaxis]

    payload: dict[str, np.ndarray] = {"raman_shift_cm1": dataset.raman_shift_cm1.astype(np.float32)}
    split_summary: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        mask = metadata["split"].to_numpy() == split
        split_meta = metadata.loc[mask].reset_index(drop=True)
        x = features[mask]
        y = split_meta["label"].to_numpy(dtype=np.int32)
        payload[f"x_{split}"] = x
        payload[f"y_{split}"] = y
        payload[f"sample_id_{split}"] = split_meta["sample_id"].to_numpy(dtype=str)
        payload[f"concentration_fraction_{split}"] = split_meta["concentration_fraction"].to_numpy(np.float32)
        payload[f"concentration_percent_{split}"] = split_meta["concentration_percent"].to_numpy(np.float32)
        _save_model_zoo_csv(processed_dir / f"{split}.csv", x, y)
        split_summary[split] = {
            "samples": int(len(split_meta)),
            "concentrations_fraction": split_groups[split],
            "class_counts": {
                name: int((split_meta["compound"] == name).sum()) for name in CLASS_NAMES
            },
        }

    # Prediction input has no label column. It is deliberately selected from test only.
    prediction_rows = []
    for label in range(len(CLASS_NAMES)):
        for concentration in np.unique(payload["concentration_fraction_test"]):
            indices = np.flatnonzero(
                (payload["y_test"] == label)
                & np.isclose(payload["concentration_fraction_test"], concentration)
            )
            prediction_rows.append(int(indices[0]))
    prediction_rows = np.asarray(prediction_rows, dtype=np.int64)
    _save_model_zoo_csv(processed_dir / "prediction.csv", payload["x_test"][prediction_rows], None)
    prediction_labels = payload["y_test"][prediction_rows]
    pd.Series(
        [CLASS_NAMES[int(label)] for label in prediction_labels],
        name="ground_truth",
    ).to_csv(
        processed_dir / "prediction_ground_truth.csv",
        index=False,
        header=False,
    )

    np.savez_compressed(processed_dir / "dataset.npz", **payload)
    metadata.to_csv(processed_dir / "metadata.csv", index=False)
    pd.DataFrame({"raman_shift_cm1": dataset.raman_shift_cm1}).to_csv(
        processed_dir / "raman_shift_axis.csv", index=False
    )

    inspection = inspect_dataset(dataset)
    spectral_hashes = pd.util.hash_pandas_object(pd.DataFrame(dataset.spectra), index=False).to_numpy()
    duplicate_frame = metadata.assign(spectral_hash=spectral_hashes)
    duplicate_frame = duplicate_frame[duplicate_frame.duplicated("spectral_hash", keep=False)]
    cross_split_duplicates = int(
        sum(group["split"].nunique() > 1 for _, group in duplicate_frame.groupby("spectral_hash"))
    )
    if cross_split_duplicates:
        raise RuntimeError(f"Found {cross_split_duplicates} exact duplicate patterns crossing splits")
    report = {
        "dataset_inspection": inspection,
        "split_policy": {
            "seed": int(config["seed"]),
            "group_key": "concentration_fraction (global across compounds)",
            "feature_values_used_to_choose_split": False,
            "groups": split_groups,
            "disjoint": len({x for values in split_groups.values() for x in values}) == 11,
            "exact_duplicate_audit": {
                "duplicate_occurrences": int(len(duplicate_frame)),
                "duplicate_patterns": int(duplicate_frame["spectral_hash"].nunique()),
                "patterns_crossing_splits": cross_split_duplicates,
            },
        },
        "splits": split_summary,
        "preprocessing": pre,
        "model_input_shape": [1, 1, SPECTRAL_LENGTH, 1],
    }
    with (reports_dir / "dataset_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report


def load_prepared(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = project_path(config, config["paths"]["processed_dir"]) / "dataset.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found at {path}; run `raman-stm32 prepare` first")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
