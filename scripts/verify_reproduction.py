#!/usr/bin/env python3
"""Verify the final local and STM32N6 workflow results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from raman_stm32.config import load_config
from raman_stm32.data import load_prepared
from raman_stm32.evaluation import _classification_metrics, predict_tflite
from raman_stm32.modeling import tflite_details


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ACCURACY = 0.9969879518072289
LOCAL_MACRO_F1 = 0.9970209513356721


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


def verify_n6_candidate() -> None:
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

    log_path = (
        ROOT
        / "artifacts/model_zoo/linear_softmax/benchmarking_stm32n6/stm32ai_main.log"
    )
    log = log_path.read_text(encoding="utf-8")
    expected_lines = (
        "Benchmarking board : STM32N6570-DK",
        "Cycles : 0.017 M",
        "Inference_time : 0.02 ms",
        "Total RAM : 0.53 KiB",
        "Total Flash : 24.35 KiB",
        "operation finished: benchmarking",
    )
    for line in expected_lines:
        if line not in log:
            raise AssertionError(f"Developer Cloud log does not contain {line!r}")


def main() -> int:
    verify_local()
    verify_n6_candidate()
    print("Reproduction verified successfully")
    print("Local Linear Softmax: accuracy 99.70%, macro F1 99.70%")
    print("STM32N6 Model Zoo candidate: FP32/INT8 accuracy 100.00%, macro F1 100.00%")
    print("Developer Cloud STM32N6570-DK: 0.02 ms, 0.017 M cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
