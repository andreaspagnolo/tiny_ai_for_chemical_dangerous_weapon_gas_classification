#!/usr/bin/env python3
"""Compare deterministic host and physical STM32N6 prediction results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HOST_RESULTS = (
    ROOT / "artifacts/model_zoo/linear_softmax/prediction_host/prediction_results.json"
)
N6_RESULTS = (
    ROOT
    / "artifacts/model_zoo/linear_softmax/prediction_stm32n6/prediction_results.json"
)
EXPECTED_CLASSES = ["TEP", "TEP", "DIMP", "DIMP", "DMMP", "DMMP"]
CLASS_NAMES = ["TEP", "DIMP", "DMMP"]
# Four output quantization steps. The model's output scale is exactly 1/256.
MAX_SCORE_DELTA = 4.0 / 256.0


def _load_predictions(path: Path, expected_target: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target") != expected_target:
        raise AssertionError(
            f"{path}: expected target {expected_target!r}, obtained {payload.get('target')!r}"
        )
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(EXPECTED_CLASSES):
        raise AssertionError(
            f"{path}: expected {len(EXPECTED_CLASSES)} predictions, obtained "
            f"{len(predictions) if isinstance(predictions, list) else 'invalid JSON'}"
        )
    return predictions


def verify_stm32n6_prediction(
    host_path: Path = HOST_RESULTS,
    n6_path: Path = N6_RESULTS,
) -> float:
    host = _load_predictions(host_path, "host")
    n6 = _load_predictions(n6_path, "stedgeai_n6")

    host_classes = [item.get("predicted_class") for item in host]
    n6_classes = [item.get("predicted_class") for item in n6]
    if host_classes != EXPECTED_CLASSES:
        raise AssertionError(
            f"Host predictions differ from the expected classes: {host_classes}"
        )
    if n6_classes != host_classes:
        raise AssertionError(
            f"STM32N6 predictions {n6_classes} differ from host {host_classes}"
        )

    host_scores = np.asarray(
        [[item["scores"][name] for name in CLASS_NAMES] for item in host],
        dtype=np.float64,
    )
    n6_scores = np.asarray(
        [[item["scores"][name] for name in CLASS_NAMES] for item in n6],
        dtype=np.float64,
    )
    max_delta = float(np.max(np.abs(host_scores - n6_scores)))
    if max_delta > MAX_SCORE_DELTA:
        raise AssertionError(
            "STM32N6 score delta exceeds four INT8 output steps: "
            f"{max_delta:.8f} > {MAX_SCORE_DELTA:.8f}"
        )
    return max_delta


def main() -> int:
    max_delta = verify_stm32n6_prediction()
    print("STM32N6 physical inference verified: 6/6 predictions match the host")
    print(f"Maximum score delta: {max_delta:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
