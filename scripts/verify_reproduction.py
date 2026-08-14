#!/usr/bin/env python3
"""Verify the final local, Model Zoo, and Developer Cloud results."""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
import re
import sys

import numpy as np

from raman_stm32.config import load_config
from raman_stm32.data import load_prepared
from raman_stm32.evaluation import _classification_metrics, predict_tflite
from raman_stm32.modeling import tflite_details


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ispu_validation import verify_ispu_results
from scripts.verify_stm32n6_prediction import MAX_SCORE_DELTA, verify_stm32n6_prediction


LOCAL_ACCURACY = 0.9969879518072289
LOCAL_MACRO_F1 = 0.9970209513356721
TRAINING_SAMPLES = 1445
MODEL_PARAMETERS = 1539
EXPECTED_BOARDS = {
    "stm32n6": "STM32N6570-DK",
    "stm32u5": "B-U585I-IOT02A",
    "stm32f4": "NUCLEO-F401RE",
    "st_ispu": "LSM6DSO16IS",
}


def _assert_close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"{label}: expected {expected}, obtained {actual}")


def verify_local() -> None:
    report_path = ROOT / "artifacts/reports/linear_softmax_evaluation_metrics.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for model_name in (
        "linear_softmax_float32_keras",
        "linear_softmax_float32_tflite",
        "linear_softmax_int8",
    ):
        metrics = report["metrics"][model_name]
        _assert_close(metrics["accuracy"], LOCAL_ACCURACY, f"local {model_name} accuracy")
        _assert_close(metrics["macro_f1"], LOCAL_MACRO_F1, f"local {model_name} macro F1")
    _assert_close(
        report["linear_float_int8"]["prediction_agreement"],
        1.0,
        "local FP32/INT8 prediction agreement",
    )


def _extract_log_metric(log: str, labels: tuple[str, ...], unit: str) -> str:
    label_pattern = "(?:" + "|".join(re.escape(label) for label in labels) + ")"
    number_pattern = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = (
        rf"{label_pattern}\s*:\s*(?P<value>{number_pattern})"
        rf"\s*\(?{re.escape(unit)}\)?"
    )
    match = re.search(pattern, log, flags=re.IGNORECASE)
    if match is None:
        raise AssertionError(
            f"Developer Cloud log does not report {labels[0]} in {unit}"
        )
    return match.group("value")


def _extract_board(log: str) -> str:
    patterns = (
        r"Benchmarking board\s*:\s*(?P<board>[^\s,]+)",
        r"Starting the model benchmark on target\s+(?P<board>[^\s,]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, log, flags=re.IGNORECASE)
        if match is not None:
            return match.group("board")
    raise AssertionError("Developer Cloud log does not report the benchmark target")


def _assert_positive_log_value(actual: str, label: str) -> None:
    if not Decimal(actual).is_finite() or Decimal(actual) <= 0:
        raise AssertionError(f"Developer Cloud log reports invalid {label} = {actual}")


def verify_board_logs(root: Path = ROOT) -> dict[str, dict[str, str]]:
    actual_results = {}
    for family, expected_board in EXPECTED_BOARDS.items():
        log_path = (
            root
            / "artifacts/model_zoo/linear_softmax"
            / f"benchmarking_{family}/stm32ai_main.log"
        )
        log = log_path.read_text(encoding="utf-8")
        actual = {
            "board": _extract_board(log),
            "cycles": _extract_log_metric(log, ("Cycles", "Number of cycles"), "M"),
            "inference_ms": _extract_log_metric(
                log, ("Inference_time", "Inference Time"), "ms"
            ),
            "ram_kib": _extract_log_metric(log, ("Total RAM",), "KiB"),
            "flash_kib": _extract_log_metric(log, ("Total Flash",), "KiB"),
        }
        if actual["board"] != expected_board:
            raise AssertionError(
                f"Developer Cloud log {log_path} reports board {actual['board']}, "
                f"expected {expected_board}"
            )
        for key in ("cycles", "inference_ms", "ram_kib", "flash_kib"):
            _assert_positive_log_value(actual[key], key)
        if "Benchmark complete." not in log and "operation finished: benchmarking" not in log:
            raise AssertionError(f"Developer Cloud benchmark did not complete in {log_path}")
        actual_results[family] = actual
    return actual_results


def verify_model_zoo_candidate() -> tuple[
    float, dict[str, dict[str, str]], dict[str, float | int]
]:
    import tensorflow as tf

    config = load_config(ROOT / "configs/project/pipeline.yaml")
    data = load_prepared(config)
    x_test = data["x_test"]
    y_test = data["y_test"]
    float_path = (
        ROOT / "artifacts/model_zoo/linear_softmax/training/saved_models/best_model.keras"
    )
    int8_path = (
        ROOT
        / "artifacts/model_zoo/linear_softmax/quantization/quantized_models/quantized_model.tflite"
    )

    float_model = tf.keras.models.load_model(float_path, compile=False)
    if x_test.shape[0] != 332:
        raise AssertionError(f"Expected 332 test samples, obtained {x_test.shape[0]}")
    if data["x_train"].shape[0] != TRAINING_SAMPLES:
        raise AssertionError(
            f"Expected {TRAINING_SAMPLES} training samples, obtained {data['x_train'].shape[0]}"
        )
    if float_model.count_params() != MODEL_PARAMETERS:
        raise AssertionError(
            f"Expected {MODEL_PARAMETERS} model parameters, obtained {float_model.count_params()}"
        )
    float_probabilities = float_model.predict(x_test, batch_size=32, verbose=0)
    int8_probabilities = predict_tflite(int8_path, x_test)
    float_metrics = _classification_metrics(y_test, float_probabilities)
    int8_metrics = _classification_metrics(y_test, int8_probabilities)
    for name, metrics in (("FP32", float_metrics), ("INT8", int8_metrics)):
        _assert_close(metrics["accuracy"], 1.0, f"Model Zoo {name} accuracy")
        _assert_close(metrics["macro_f1"], 1.0, f"Model Zoo {name} macro F1")
    _assert_close(
        float(np.mean(np.argmax(float_probabilities, axis=1) == np.argmax(int8_probabilities, axis=1))),
        1.0,
        "Model Zoo FP32/INT8 prediction agreement",
    )

    details = tflite_details(int8_path)
    assert details["input"]["shape_signature"] == [1, 1, 512, 1], details
    assert details["fully_integer_io"], details
    assert details["operators"] == ["RESHAPE", "FULLY_CONNECTED", "SOFTMAX"], details
    board_results = verify_board_logs()
    return verify_stm32n6_prediction(), board_results, verify_ispu_results()


def main() -> int:
    verify_local()
    n6_delta, board_results, ispu_result = verify_model_zoo_candidate()
    print("Reproduction verified successfully")
    print("Local Linear Softmax: accuracy 99.70%, macro F1 99.70%")
    print("Model Zoo Linear Softmax: FP32/INT8 accuracy 100.00%, macro F1 100.00%")
    print("Developer Cloud benchmark observations (parsed from logs; not pass/fail limits):")
    for result in board_results.values():
        print(
            f"  {result['board']}: {result['inference_ms']} ms, "
            f"{result['cycles']} M cycles, {result['ram_kib']} KiB RAM, "
            f"{result['flash_kib']} KiB Flash"
        )
    print(
        "Physical STM32N6: 6/6 predictions match host; maximum score delta "
        f"{n6_delta:.8f} (limit {MAX_SCORE_DELTA:.8f})"
    )
    print(
        "Physical LSM6DSO16IS: "
        f"{ispu_result['correct']}/{ispu_result['total']} predictions match host "
        f"and ground truth; maximum score delta "
        f"{ispu_result['max_score_delta']:.8f}; observed duration at "
        f"{ispu_result['clock_mhz']} MHz: {ispu_result['duration_ms']:.3f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
