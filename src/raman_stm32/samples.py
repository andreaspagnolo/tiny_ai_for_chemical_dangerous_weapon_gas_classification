"""Export exact host and board input tensors from the held-out test split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ensure_output_dirs, project_path
from .constants import CLASS_NAMES
from .data import load_prepared
from .modeling import tflite_details


def export_board_samples(
    config: dict[str, Any],
    model_path: str | Path | None = None,
    samples_dir: str | Path | None = None,
) -> dict[str, Any]:
    ensure_output_dirs(config)
    data = load_prepared(config)
    if model_path is None:
        model_path = (
            project_path(config, config["paths"]["models_dir"])
            / "raman_tiny_cnn_int8.tflite"
        )
    else:
        model_path = project_path(config, model_path)
    if samples_dir is None:
        samples_dir = project_path(config, config["paths"]["samples_dir"])
    else:
        samples_dir = project_path(config, samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    details = tflite_details(model_path)
    scale = float(details["input"]["scale"])
    zero_point = int(details["input"]["zero_point"])
    per_class = int(config["samples"]["per_class"])
    manifest: dict[str, Any] = {
        "model": str(model_path),
        "tensor_shape_nhwc": [1, 1, 512, 1],
        "tensor_dtype": "int8",
        "quantization": {"scale": scale, "zero_point": zero_point},
        "byte_order": "512 contiguous signed int8 values in NHWC order",
        "preprocessing": "SNV, clip to +/-8 standard deviations, divide by 8",
        "source_split": "test only",
        "samples": [],
    }
    all_float: list[np.ndarray] = []
    all_int8: list[np.ndarray] = []
    for label, class_name in enumerate(CLASS_NAMES):
        indices_list: list[int] = []
        for concentration in np.unique(data["concentration_fraction_test"]):
            candidates = np.flatnonzero(
                (data["y_test"] == label)
                & np.isclose(data["concentration_fraction_test"], concentration)
            )
            if len(candidates):
                indices_list.append(int(candidates[0]))
            if len(indices_list) == per_class:
                break
        if len(indices_list) < per_class:
            remaining = [
                int(index)
                for index in np.flatnonzero(data["y_test"] == label)
                if int(index) not in indices_list
            ]
            indices_list.extend(remaining[: per_class - len(indices_list)])
        indices = np.asarray(indices_list, dtype=np.int64)
        for ordinal, index in enumerate(indices, start=1):
            tensor = data["x_test"][index : index + 1].astype(np.float32)
            quantized = np.clip(np.rint(tensor / scale + zero_point), -128, 127).astype(np.int8)
            stem = f"{class_name.lower()}_{ordinal}"
            np.save(samples_dir / f"{stem}_float32.npy", tensor, allow_pickle=False)
            np.save(samples_dir / f"{stem}_int8.npy", quantized, allow_pickle=False)
            quantized.tofile(samples_dir / f"{stem}_int8.bin")
            np.savetxt(samples_dir / f"{stem}_preprocessed.csv", tensor.reshape(1, -1), delimiter=",", fmt="%.8g")
            all_float.append(tensor)
            all_int8.append(quantized)
            manifest["samples"].append(
                {
                    "name": stem,
                    "sample_id": str(data["sample_id_test"][index]),
                    "true_class_index": label,
                    "true_class": class_name,
                    "concentration_fraction": float(data["concentration_fraction_test"][index]),
                    "concentration_percent": float(data["concentration_percent_test"][index]),
                    "float32_npy": f"{stem}_float32.npy",
                    "int8_npy": f"{stem}_int8.npy",
                    "int8_bin": f"{stem}_int8.bin",
                    "int8_bytes": int(quantized.nbytes),
                }
            )
    float_batch = np.concatenate(all_float, axis=0)
    int8_batch = np.concatenate(all_int8, axis=0)
    from .evaluation import predict_tflite

    host_probabilities = predict_tflite(model_path, float_batch)
    for sample_record, probabilities in zip(manifest["samples"], host_probabilities, strict=True):
        predicted_index = int(np.argmax(probabilities))
        sample_record["expected_host_predicted_index"] = predicted_index
        sample_record["expected_host_predicted_class"] = CLASS_NAMES[predicted_index]
        sample_record["expected_host_scores_dequantized"] = {
            class_name: float(probabilities[index])
            for index, class_name in enumerate(CLASS_NAMES)
        }

    np.savez_compressed(
        samples_dir / "board_samples.npz",
        float32=float_batch,
        int8=int8_batch,
    )
    np.save(
        samples_dir / "board_inputs_float32.npy",
        float_batch,
        allow_pickle=False,
    )
    np.save(
        samples_dir / "board_inputs_int8.npy",
        int8_batch,
        allow_pickle=False,
    )
    with (samples_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest
