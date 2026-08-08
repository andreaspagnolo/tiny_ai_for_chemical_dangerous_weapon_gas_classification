"""Small NPU-friendly CNN, training, and TFLite conversion."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .config import ensure_output_dirs, project_path
from .constants import CLASS_NAMES, MODEL_INPUT_SHAPE
from .data import load_prepared


def configure_determinism(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_raman_tiny_cnn(batch_size: int | None = None):
    """Build a 10,899-parameter spectral CNN using Neural-ART-friendly ops.

    The height is one, so Conv2D kernels (1, K) are semantically 1-D spectral
    convolutions while avoiding a Conv1D conversion ambiguity.
    """
    import tensorflow as tf

    inputs = tf.keras.Input(
        shape=MODEL_INPUT_SHAPE,
        batch_size=batch_size,
        dtype=tf.float32,
        name="raman_spectrum",
    )
    x = tf.keras.layers.Conv2D(
        8, (1, 9), strides=(1, 2), padding="same", activation="relu", name="conv_spectral_1"
    )(inputs)
    x = tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2), name="pool_1")(x)
    x = tf.keras.layers.Conv2D(16, (1, 7), padding="same", activation="relu", name="conv_spectral_2")(x)
    x = tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2), name="pool_2")(x)
    x = tf.keras.layers.Conv2D(24, (1, 5), padding="same", activation="relu", name="conv_spectral_3")(x)
    x = tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2), name="pool_3")(x)
    x = tf.keras.layers.Conv2D(24, (1, 3), padding="same", activation="relu", name="conv_spectral_4")(x)
    x = tf.keras.layers.MaxPooling2D((1, 2), strides=(1, 2), name="pool_4")(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(16, activation="relu", name="spectral_bottleneck")(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="class_probabilities")(x)
    return tf.keras.Model(inputs, outputs, name="raman_tiny_cnn")


def _copy_as_static_model(trained_model):
    static_model = build_raman_tiny_cnn(batch_size=1)
    static_model.set_weights(trained_model.get_weights())
    return static_model


def _representative_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_class = max(1, count // len(CLASS_NAMES))
    for label in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(labels == label)
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < min(count, len(labels)):
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        take = min(count - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def convert_tflite(model, output_path: Path, representative_data: np.ndarray | None = None) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if representative_data is not None:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        def generator() -> Iterable[list[np.ndarray]]:
            for sample in representative_data:
                yield [sample[np.newaxis, ...].astype(np.float32)]

        converter.representative_dataset = generator
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    output_path.write_bytes(converter.convert())


def tflite_details(model_path: str | Path) -> dict[str, Any]:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    # XNNPACK adds DELEGATE pseudo-ops at runtime; they are not in the flatbuffer.
    ops = [entry["op_name"] for entry in interpreter._get_ops_details() if entry["op_name"] != "DELEGATE"]
    return {
        "input": {
            "name": inp["name"],
            "shape": inp["shape"].astype(int).tolist(),
            "shape_signature": inp["shape_signature"].astype(int).tolist(),
            "dtype": np.dtype(inp["dtype"]).name,
            "scale": float(inp["quantization"][0]),
            "zero_point": int(inp["quantization"][1]),
        },
        "output": {
            "name": out["name"],
            "shape": out["shape"].astype(int).tolist(),
            "shape_signature": out["shape_signature"].astype(int).tolist(),
            "dtype": np.dtype(out["dtype"]).name,
            "scale": float(out["quantization"][0]),
            "zero_point": int(out["quantization"][1]),
        },
        "operators": ops,
        "fully_integer_io": bool(inp["dtype"] == np.int8 and out["dtype"] == np.int8),
        "bytes": int(Path(model_path).stat().st_size),
    }


def train_models(config: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf

    ensure_output_dirs(config)
    seed = int(config["seed"])
    configure_determinism(seed)
    data = load_prepared(config)
    models_dir = project_path(config, config["paths"]["models_dir"])
    reports_dir = project_path(config, config["paths"]["reports_dir"])

    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_validation"], data["y_validation"]

    baseline = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    baseline.fit(x_train.reshape(len(x_train), -1), y_train)
    baseline_path = models_dir / "logistic_regression_baseline.joblib"
    joblib.dump(baseline, baseline_path)
    baseline_val_accuracy = accuracy_score(y_val, baseline.predict(x_val.reshape(len(x_val), -1)))

    model = build_raman_tiny_cnn(batch_size=None)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(config["training"]["learning_rate"])),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=int(config["training"]["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(config["training"]["reduce_lr_patience"]),
            factor=float(config["training"]["reduce_lr_factor"]),
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=int(config["training"]["batch_size"]),
        epochs=int(config["training"]["epochs"]),
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    static_model = _copy_as_static_model(model)
    keras_path = models_dir / "raman_tiny_cnn_float32.keras"
    float_tflite_path = models_dir / "raman_tiny_cnn_float32.tflite"
    static_model.save(keras_path)
    convert_tflite(static_model, float_tflite_path)

    val_probs = static_model.predict(x_val, batch_size=1, verbose=0)
    result = {
        "seed": seed,
        "model_selection": "maximum validation accuracy with early stopping",
        "epochs_completed": len(history.history["loss"]),
        "parameter_count": int(static_model.count_params()),
        "model_input_shape": [int(value) for value in static_model.input_shape],
        "model_output_shape": [int(value) for value in static_model.output_shape],
        "baseline_validation_accuracy": float(baseline_val_accuracy),
        "cnn_validation_accuracy": float(accuracy_score(y_val, np.argmax(val_probs, axis=1))),
        "keras_model": str(keras_path),
        "float_tflite_model": str(float_tflite_path),
        "float_tflite": tflite_details(float_tflite_path),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    with (reports_dir / "training_report.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def quantize_model(config: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf

    ensure_output_dirs(config)
    seed = int(config["seed"])
    configure_determinism(seed)
    data = load_prepared(config)
    models_dir = project_path(config, config["paths"]["models_dir"])
    reports_dir = project_path(config, config["paths"]["reports_dir"])
    keras_path = models_dir / "raman_tiny_cnn_float32.keras"
    if not keras_path.is_file():
        raise FileNotFoundError(f"Missing {keras_path}; run training first")
    model = tf.keras.models.load_model(keras_path, compile=False)
    count = int(config["quantization"]["representative_samples"])
    indices = _representative_indices(data["y_train"], count, seed)
    representative = data["x_train"][indices]
    output_path = models_dir / "raman_tiny_cnn_int8.tflite"
    convert_tflite(model, output_path, representative_data=representative)
    details = tflite_details(output_path)
    details["representative_samples"] = int(len(indices))
    details["representative_source"] = "training split only"
    if details["input"]["shape_signature"] != [1, 1, 512, 1]:
        raise RuntimeError(f"Deployment model is not static batch 1: {details['input']['shape_signature']}")
    if not details["fully_integer_io"]:
        raise RuntimeError("Quantized model does not have INT8 input and output")
    with (reports_dir / "quantization_report.json").open("w", encoding="utf-8") as stream:
        json.dump(details, stream, indent=2)
    return details
