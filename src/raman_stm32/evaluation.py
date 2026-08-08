"""Consistent evaluation for baseline, Keras, float TFLite, and INT8 TFLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .config import ensure_output_dirs, project_path
from .constants import CLASS_NAMES
from .data import load_prepared, snv_clip_scale
from .modeling import tflite_details


def predict_tflite(model_path: str | Path, features: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    predictions: list[np.ndarray] = []
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]
    for sample in features:
        batch = sample[np.newaxis, ...].astype(np.float32)
        if input_detail["dtype"] == np.int8:
            batch = np.clip(np.rint(batch / in_scale + in_zero), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], batch)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if output_detail["dtype"] == np.int8:
            output = (output.astype(np.float32) - out_zero) * out_scale
        predictions.append(output[0].astype(np.float32))
    return np.stack(predictions)


def _classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(len(CLASS_NAMES)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(
            labels, predictions, labels=np.arange(len(CLASS_NAMES))
        ).astype(int).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def evaluate_models(config: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf

    ensure_output_dirs(config)
    data = load_prepared(config)
    models_dir = project_path(config, config["paths"]["models_dir"])
    reports_dir = project_path(config, config["paths"]["reports_dir"])
    x_test, y_test = data["x_test"], data["y_test"]

    baseline = joblib.load(models_dir / "logistic_regression_baseline.joblib")
    baseline_probs = baseline.predict_proba(x_test.reshape(len(x_test), -1)).astype(np.float32)
    keras_model = tf.keras.models.load_model(models_dir / "raman_tiny_cnn_float32.keras", compile=False)
    keras_probs = keras_model.predict(x_test, batch_size=1, verbose=0)
    float_tflite_path = models_dir / "raman_tiny_cnn_float32.tflite"
    int8_tflite_path = models_dir / "raman_tiny_cnn_int8.tflite"
    float_tflite_probs = predict_tflite(float_tflite_path, x_test)
    int8_probs = predict_tflite(int8_tflite_path, x_test)

    results = {
        "split": "held-out test concentrations",
        "samples": int(len(y_test)),
        "class_names": list(CLASS_NAMES),
        "baseline_logistic_regression": _classification_metrics(y_test, baseline_probs),
        "cnn_float32_keras": _classification_metrics(y_test, keras_probs),
        "cnn_float32_tflite": _classification_metrics(y_test, float_tflite_probs),
        "cnn_int8_tflite": _classification_metrics(y_test, int8_probs),
        "float_int8": {
            "accuracy_delta_int8_minus_float": float(
                accuracy_score(y_test, np.argmax(int8_probs, axis=1))
                - accuracy_score(y_test, np.argmax(float_tflite_probs, axis=1))
            ),
            "prediction_agreement": float(
                np.mean(np.argmax(int8_probs, axis=1) == np.argmax(float_tflite_probs, axis=1))
            ),
            "mean_absolute_probability_delta": float(np.mean(np.abs(int8_probs - float_tflite_probs))),
        },
        "model_files": {
            "float32_tflite": tflite_details(float_tflite_path),
            "int8_tflite": tflite_details(int8_tflite_path),
        },
    }

    concentration_rows: list[dict[str, Any]] = []
    concentrations = data["concentration_percent_test"]
    for value in sorted(float(x) for x in np.unique(concentrations)):
        mask = np.isclose(concentrations, value)
        for model_name, probabilities in (
            ("baseline_logistic_regression", baseline_probs),
            ("cnn_float32_tflite", float_tflite_probs),
            ("cnn_int8_tflite", int8_probs),
        ):
            concentration_rows.append(
                {
                    "concentration_percent": value,
                    "model": model_name,
                    "samples": int(mask.sum()),
                    "accuracy": float(accuracy_score(y_test[mask], np.argmax(probabilities[mask], axis=1))),
                }
            )
    pd.DataFrame(concentration_rows).to_csv(reports_dir / "metrics_by_concentration.csv", index=False)
    results["by_concentration"] = concentration_rows

    with (reports_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    return results


def predict_csv(
    config: dict[str, Any], csv_path: str | Path, already_preprocessed: bool = False
) -> list[dict[str, Any]]:
    path = Path(csv_path)
    values = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
    if values.shape[1] != 512:
        raise ValueError(f"Expected 512 already-preprocessed values per row, found {values.shape[1]}")
    if not already_preprocessed:
        preprocessing = config["data"]["preprocessing"]
        values = snv_clip_scale(
            values,
            clip_standard_deviations=float(preprocessing["clip_standard_deviations"]),
            epsilon=float(preprocessing["epsilon"]),
        )
    features = values[:, np.newaxis, :, np.newaxis]
    model_path = project_path(config, config["paths"]["models_dir"]) / "raman_tiny_cnn_int8.tflite"
    probabilities = predict_tflite(model_path, features)
    rows = []
    for index, probs in enumerate(probabilities):
        predicted = int(np.argmax(probs))
        rows.append(
            {
                "row": index,
                "predicted_index": predicted,
                "predicted_class": CLASS_NAMES[predicted],
                "scores": {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)},
            }
        )
    return rows
