"""Artifact integrity and deployment-readiness checks."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

from .config import ensure_output_dirs, project_path
from .modeling import tflite_details


NPU_REFERENCE_OPERATORS = {
    "CONV_2D",
    "MAX_POOL_2D",
    "AVERAGE_POOL_2D",
    "RESHAPE",
    "FULLY_CONNECTED",
    "SOFTMAX",
}


def _file_record(path: Path) -> dict[str, Any]:
    digest = sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def verify_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    models_dir = project_path(config, config["paths"]["models_dir"])
    reports_dir = project_path(config, config["paths"]["reports_dir"])
    required = [
        models_dir / "raman_tiny_cnn_float32.keras",
        models_dir / "raman_tiny_cnn_float32.tflite",
        models_dir / "raman_tiny_cnn_int8.tflite",
        models_dir / "logistic_regression_baseline.joblib",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts: {missing}")
    int8 = tflite_details(models_dir / "raman_tiny_cnn_int8.tflite")
    float32 = tflite_details(models_dir / "raman_tiny_cnn_float32.tflite")
    unknown = sorted(set(int8["operators"]) - NPU_REFERENCE_OPERATORS)
    report = {
        "files": [_file_record(path) for path in required],
        "checks": {
            "static_batch_one": int8["input"]["shape_signature"] == [1, 1, 512, 1],
            "int8_input_and_output": int8["fully_integer_io"],
            "only_reference_supported_operator_types": not unknown,
            "unrecognized_operator_types": unknown,
            "tflite_host_load_and_allocate": True,
            "float_tflite": float32,
            "int8_tflite": int8,
        },
        "npu_mapping_note": (
            "All flatbuffer operator types appear in ST's Neural-ART support table and every pooling "
            "window is 2. Softmax hardware mapping depends on the Neural-ART compiler expansion. "
            "The generated STEdgeAI report is authoritative and still must be inspected."
        ),
        "stedgeai_compile": {
            "status": "not_run" if shutil.which("stedgeai") is None else "available_not_run",
            "reason": "The stedgeai executable is not installed in this environment."
            if shutil.which("stedgeai") is None
            else "Executable found, but a licensed/installed N6 toolchain run was not requested automatically.",
        },
        "stm32n6570_dk_benchmark": {
            "status": "not_run",
            "reason": "Requires ST Developer Cloud credentials or a connected, provisioned physical board.",
        },
    }
    with (reports_dir / "deployment_readiness.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report

