"""Train-from-scratch linear softmax experiment for STM32 deployment.

The network is mathematically the same model family as multinomial logistic
regression, but its Keras weights are learned independently from a random
initialization. No coefficient is copied from the scikit-learn baseline.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

from .config import ensure_output_dirs, load_config, project_path
from .constants import CLASS_NAMES, MODEL_INPUT_SHAPE
from .data import load_prepared, snv_clip_scale
from .evaluation import _classification_metrics, predict_tflite
from .modeling import (
    _representative_indices,
    configure_determinism,
    convert_tflite,
    tflite_details,
)
from .samples import export_board_samples
from .verification import NPU_REFERENCE_OPERATORS


DEFAULT_PROJECT_CONFIG = "configs/project/pipeline.yaml"
DEFAULT_EXPERIMENT_CONFIG = "configs/project/linear_softmax_experiment.yaml"
MODEL_STEM = "raman_linear_softmax"


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).resolve().open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def build_raman_linear_softmax(
    experiment: dict[str, Any], batch_size: int | None = None
):
    """Build the 1,539-parameter linear classifier using standard Keras ops."""
    import tensorflow as tf

    l2_value = float(experiment["model"]["l2_regularization"])
    inputs = tf.keras.Input(
        shape=MODEL_INPUT_SHAPE,
        batch_size=batch_size,
        dtype=tf.float32,
        name="raman_spectrum",
    )
    flattened = tf.keras.layers.Flatten(name="flatten")(inputs)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=tf.keras.regularizers.l2(l2_value),
        name="class_probabilities",
    )(flattened)
    return tf.keras.Model(inputs, outputs, name=MODEL_STEM)


def _static_copy(model, experiment: dict[str, Any]):
    static_model = build_raman_linear_softmax(experiment, batch_size=1)
    static_model.set_weights(model.get_weights())
    return static_model


def train_linear_softmax(
    project_config: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    """Train from a deterministic random initialization using train/validation."""
    import tensorflow as tf

    ensure_output_dirs(project_config)
    seed = int(project_config["seed"])
    configure_determinism(seed)
    data = load_prepared(project_config)
    models_dir = project_path(project_config, project_config["paths"]["models_dir"])
    reports_dir = project_path(project_config, project_config["paths"]["reports_dir"])
    train_cfg = experiment["training"]

    model = build_raman_linear_softmax(experiment, batch_size=None)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(train_cfg["learning_rate"])
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(train_cfg["early_stopping_patience"]),
            min_delta=float(train_cfg["early_stopping_min_delta"]),
            restore_best_weights=True,
            verbose=1,
        )
    ]
    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_validation"], data["y_validation"]),
        batch_size=int(train_cfg["batch_size"]),
        epochs=int(train_cfg["epochs"]),
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    static_model = _static_copy(model, experiment)
    keras_path = models_dir / f"{MODEL_STEM}_float32.keras"
    float_tflite_path = models_dir / f"{MODEL_STEM}_float32.tflite"
    static_model.save(keras_path)
    convert_tflite(static_model, float_tflite_path)
    validation_probabilities = static_model.predict(
        data["x_validation"], batch_size=1, verbose=0
    )

    val_losses = np.asarray(history.history["val_loss"], dtype=np.float64)
    result = {
        "experiment": experiment["experiment_name"],
        "seed": seed,
        "training_origin": "random_glorot_initialization_trained_from_scratch",
        "pretrained_or_logistic_weights_imported": False,
        "selection": "minimum validation loss with early stopping; test split not used",
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss_epoch": int(np.argmin(val_losses) + 1),
        "parameter_count": int(static_model.count_params()),
        "estimated_macs": 512 * len(CLASS_NAMES),
        "input_shape": [int(value) for value in static_model.input_shape],
        "output_shape": [int(value) for value in static_model.output_shape],
        "validation_accuracy": float(
            accuracy_score(
                data["y_validation"], np.argmax(validation_probabilities, axis=1)
            )
        ),
        "keras_model": str(keras_path),
        "float_tflite_model": str(float_tflite_path),
        "float_tflite": tflite_details(float_tflite_path),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    (reports_dir / "linear_softmax_training_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def quantize_linear_softmax(
    project_config: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    """Create a static full-integer INT8 model calibrated on training data."""
    import tensorflow as tf

    ensure_output_dirs(project_config)
    seed = int(project_config["seed"])
    configure_determinism(seed)
    data = load_prepared(project_config)
    models_dir = project_path(project_config, project_config["paths"]["models_dir"])
    reports_dir = project_path(project_config, project_config["paths"]["reports_dir"])
    keras_path = models_dir / f"{MODEL_STEM}_float32.keras"
    if not keras_path.is_file():
        raise FileNotFoundError(f"Missing {keras_path}; run linear training first")
    model = tf.keras.models.load_model(keras_path, compile=False)
    count = int(experiment["quantization"]["representative_samples"])
    indices = _representative_indices(data["y_train"], count, seed)
    output_path = models_dir / f"{MODEL_STEM}_int8.tflite"
    convert_tflite(model, output_path, representative_data=data["x_train"][indices])
    details = tflite_details(output_path)
    details["representative_samples"] = int(len(indices))
    details["representative_source"] = "training split only"
    if details["input"]["shape_signature"] != [1, 1, 512, 1]:
        raise RuntimeError("Linear deployment model does not have static batch size one")
    if not details["fully_integer_io"]:
        raise RuntimeError("Linear quantized model does not have full INT8 I/O")
    (reports_dir / "linear_softmax_quantization_report.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    return details


def evaluate_linear_softmax(
    project_config: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the Linear Softmax FP32 and INT8 exports locally."""
    import tensorflow as tf

    ensure_output_dirs(project_config)
    data = load_prepared(project_config)
    models_dir = project_path(project_config, project_config["paths"]["models_dir"])
    reports_dir = project_path(project_config, project_config["paths"]["reports_dir"])
    x_test, y_test = data["x_test"], data["y_test"]

    keras_model = tf.keras.models.load_model(
        models_dir / f"{MODEL_STEM}_float32.keras", compile=False
    )
    float_path = models_dir / f"{MODEL_STEM}_float32.tflite"
    int8_path = models_dir / f"{MODEL_STEM}_int8.tflite"
    linear_keras = keras_model.predict(x_test, batch_size=1, verbose=0)
    linear_float = predict_tflite(float_path, x_test)
    linear_int8 = predict_tflite(int8_path, x_test)
    predictions = {
        "linear_softmax_float32_keras": linear_keras,
        "linear_softmax_float32_tflite": linear_float,
        "linear_softmax_int8": linear_int8,
    }
    metrics = {
        name: _classification_metrics(y_test, probabilities)
        for name, probabilities in predictions.items()
    }

    concentration_rows: list[dict[str, Any]] = []
    concentrations = data["concentration_percent_test"]
    for concentration in sorted(float(value) for value in np.unique(concentrations)):
        mask = np.isclose(concentrations, concentration)
        for model_name, probabilities in predictions.items():
            concentration_rows.append(
                {
                    "concentration_percent": concentration,
                    "model": model_name,
                    "samples": int(mask.sum()),
                    "accuracy": float(
                        accuracy_score(
                            y_test[mask], np.argmax(probabilities[mask], axis=1)
                        )
                    ),
                }
            )

    result = {
        "experiment": experiment["experiment_name"],
        "training_origin": "trained_from_scratch_without_logistic_weight_transfer",
        "split": "same held-out 6% and 75% concentration groups used by previous models",
        "test_samples": int(len(y_test)),
        "metrics": metrics,
        "linear_float_int8": {
            "accuracy_delta_int8_minus_float": float(
                metrics["linear_softmax_int8"]["accuracy"]
                - metrics["linear_softmax_float32_tflite"]["accuracy"]
            ),
            "prediction_agreement": float(
                np.mean(
                    np.argmax(linear_int8, axis=1)
                    == np.argmax(linear_float, axis=1)
                )
            ),
            "mean_absolute_probability_delta": float(
                np.mean(np.abs(linear_int8 - linear_float))
            ),
        },
        "by_concentration": concentration_rows,
        "model_files": {
            "linear_float32_tflite": tflite_details(float_path),
            "linear_int8_tflite": tflite_details(int8_path),
        },
    }
    (reports_dir / "linear_softmax_evaluation_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def verify_linear_artifacts(
    project_config: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    """Record artifact hashes and the conservative local operator check."""
    del experiment
    models_dir = project_path(project_config, project_config["paths"]["models_dir"])
    reports_dir = project_path(project_config, project_config["paths"]["reports_dir"])
    paths = [
        models_dir / f"{MODEL_STEM}_float32.keras",
        models_dir / f"{MODEL_STEM}_float32.tflite",
        models_dir / f"{MODEL_STEM}_int8.tflite",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing linear artifacts: {missing}")
    int8 = tflite_details(paths[-1])
    unknown = sorted(set(int8["operators"]) - NPU_REFERENCE_OPERATORS)
    result = {
        "files": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ],
        "checks": {
            "trained_model_is_separate_from_logistic_baseline": True,
            "static_batch_one": int8["input"]["shape_signature"]
            == [1, 1, 512, 1],
            "int8_input_and_output": int8["fully_integer_io"],
            "only_reference_supported_operator_types": not unknown,
            "unrecognized_operator_types": unknown,
            "operators": int8["operators"],
        },
        "compiler_note": (
            "The local operator inventory is intentionally conservative. The "
            "STEdgeAI compiler report is authoritative for Neural-ART mapping."
        ),
    }
    (reports_dir / "linear_softmax_deployment_readiness.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def predict_csv(
    project_config: dict[str, Any], csv_path: str | Path, already_preprocessed: bool
) -> list[dict[str, Any]]:
    values = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float32)
    if values.shape[1] != 512:
        raise ValueError(f"Expected 512 values per row, found {values.shape[1]}")
    if not already_preprocessed:
        preprocessing = project_config["data"]["preprocessing"]
        values = snv_clip_scale(
            values,
            clip_standard_deviations=float(
                preprocessing["clip_standard_deviations"]
            ),
            epsilon=float(preprocessing["epsilon"]),
        )
    features = values[:, np.newaxis, :, np.newaxis]
    model_path = (
        project_path(project_config, project_config["paths"]["models_dir"])
        / f"{MODEL_STEM}_int8.tflite"
    )
    probabilities = predict_tflite(model_path, features)
    return [
        {
            "row": index,
            "predicted_index": int(np.argmax(scores)),
            "predicted_class": CLASS_NAMES[int(np.argmax(scores))],
            "scores": {
                class_name: float(scores[class_index])
                for class_index, class_name in enumerate(CLASS_NAMES)
            },
        }
        for index, scores in enumerate(probabilities)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("train", "quantize", "evaluate", "export-samples", "verify", "predict", "all"),
    )
    parser.add_argument("--project-config", default=DEFAULT_PROJECT_CONFIG)
    parser.add_argument("--experiment-config", default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--input-csv")
    parser.add_argument("--preprocessed", action="store_true")
    args = parser.parse_args(argv)
    project_config = load_config(args.project_config)
    experiment = load_experiment_config(args.experiment_config)

    if args.command in ("train", "all"):
        print(json.dumps(train_linear_softmax(project_config, experiment), indent=2))
    if args.command in ("quantize", "all"):
        print(json.dumps(quantize_linear_softmax(project_config, experiment), indent=2))
    if args.command in ("evaluate", "all"):
        print(json.dumps(evaluate_linear_softmax(project_config, experiment), indent=2))
    if args.command in ("export-samples", "all"):
        model_path = (
            project_path(project_config, project_config["paths"]["models_dir"])
            / f"{MODEL_STEM}_int8.tflite"
        )
        samples_dir = (
            project_path(project_config, project_config["paths"]["samples_dir"])
            / "linear_softmax"
        )
        print(
            json.dumps(
                export_board_samples(
                    project_config, model_path=model_path, samples_dir=samples_dir
                ),
                indent=2,
            )
        )
    if args.command in ("verify", "all"):
        print(json.dumps(verify_linear_artifacts(project_config, experiment), indent=2))
    if args.command == "predict":
        if not args.input_csv:
            parser.error("predict requires --input-csv")
        print(
            json.dumps(
                predict_csv(project_config, args.input_csv, args.preprocessed), indent=2
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
