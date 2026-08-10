#!/usr/bin/env python3
"""Verify the final local, Model Zoo, and Developer Cloud results."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

from raman_stm32.config import load_config
from raman_stm32.data import load_prepared
from raman_stm32.evaluation import _classification_metrics, predict_tflite
from raman_stm32.modeling import tflite_details


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ACCURACY = 0.9969879518072289
LOCAL_MACRO_F1 = 0.9970209513356721
TRAINING_SAMPLES = 1445
MODEL_PARAMETERS = 1539
BOARD_RESULTS = {
    "stm32n6": {
        "board": "STM32N6570-DK",
        "cycles": "0.017",
        "inference_ms": "0.02",
        "ram_kib": "0.53",
        "flash_kib": "24.35",
    },
    "stm32u5": {
        "board": "B-U585I-IOT02A",
        "cycles": "0.01",
        "inference_ms": "0.06",
        "ram_kib": "2.03",
        "flash_kib": "7.94",
    },
    "stm32f4": {
        "board": "NUCLEO-F401RE",
        "cycles": "0.009",
        "inference_ms": "0.11",
        "ram_kib": "2.03",
        "flash_kib": "7.92",
    },
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


def _assert_log_metric(log: str, labels: tuple[str, ...], value: str, unit: str) -> None:
    label_pattern = "(?:" + "|".join(re.escape(label) for label in labels) + ")"
    pattern = rf"{label_pattern}\s*:\s*{re.escape(value)}\s*\(?{re.escape(unit)}\)?"
    if re.search(pattern, log, flags=re.IGNORECASE) is None:
        raise AssertionError(
            f"Developer Cloud log does not report {labels[0]} = {value} {unit}"
        )


def verify_board_logs() -> None:
    for family, expected in BOARD_RESULTS.items():
        log_path = (
            ROOT
            / "artifacts/model_zoo/linear_softmax"
            / f"benchmarking_{family}/stm32ai_main.log"
        )
        log = log_path.read_text(encoding="utf-8")
        if expected["board"] not in log:
            raise AssertionError(
                f"Developer Cloud log {log_path} does not name {expected['board']}"
            )
        _assert_log_metric(log, ("Cycles", "Number of cycles"), expected["cycles"], "M")
        _assert_log_metric(
            log,
            ("Inference_time", "Inference Time"),
            expected["inference_ms"],
            "ms",
        )
        _assert_log_metric(log, ("Total RAM",), expected["ram_kib"], "KiB")
        _assert_log_metric(log, ("Total Flash",), expected["flash_kib"], "KiB")
        if "Benchmark complete." not in log and "operation finished: benchmarking" not in log:
            raise AssertionError(f"Developer Cloud benchmark did not complete in {log_path}")


def verify_model_zoo_candidate() -> None:
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
    verify_board_logs()


def main() -> int:
    verify_local()
    verify_model_zoo_candidate()
    print("Reproduction verified successfully")
    print("Local Linear Softmax: accuracy 99.70%, macro F1 99.70%")
    print("Model Zoo Linear Softmax: FP32/INT8 accuracy 100.00%, macro F1 100.00%")
    print("Training samples / parameters: 1445 / 1539 = 0.939")
    print("Developer Cloud: N6 0.02 ms; U5 0.06 ms; F4 0.11 ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
