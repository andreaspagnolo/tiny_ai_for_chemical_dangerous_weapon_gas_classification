#!/usr/bin/env python3
"""Prepare, run, and verify physical LSM6DSO16IS validation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raman_stm32.evaluation import predict_tflite
from raman_stm32.modeling import tflite_details


CLASS_NAMES = ("TEP", "DIMP", "DMMP")
EXPECTED_CLASSES = ("TEP", "TEP", "DIMP", "DIMP", "DMMP", "DMMP")
ISPU_DEVICE = "imu_22"
EXPECTED_ISPU_CLOCK_MHZ = 10
MODEL_PATH = (
    ROOT
    / "artifacts/model_zoo/linear_softmax/quantization/quantized_models/quantized_model.tflite"
)
PREDICTION_PATH = ROOT / "data/processed/prediction.csv"
GROUND_TRUTH_PATH = ROOT / "data/processed/prediction_ground_truth.csv"
OUTPUT_ROOT = ROOT / "artifacts/model_zoo/linear_softmax/prediction_st_ispu"
VALIDATION_DATA_PATH = OUTPUT_ROOT / "validation_data.npz"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
REFERENCE_PATH = OUTPUT_ROOT / "tflite_reference.json"
HOST_OUTPUT_DIR = OUTPUT_ROOT / "host"
TARGET_OUTPUT_DIR = OUTPUT_ROOT / "target"
ISPU_REPOSITORY_REVISION = "03149889edc95f0a94177be127b8cb64f7d1716a"
BRIDGE_BINARIES = {
    "nucleo-f401re": "nucleo_f401re_ispu_stedgeai_validate.bin",
    "nucleo-u575zi-q": "nucleo_u575ziq_ispu_stedgeai_validate.bin",
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_fingerprint(inputs: np.ndarray, outputs: np.ndarray) -> str:
    digest = sha256()
    for array in (inputs, outputs):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_validation_arrays(
    prediction_path: Path = PREDICTION_PATH,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing {prediction_path}; run `raman-stm32 prepare` first")
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Missing {ground_truth_path}; run `raman-stm32 prepare` first"
        )

    features = np.loadtxt(prediction_path, delimiter=",", dtype=np.float32, ndmin=2)
    labels_array = np.loadtxt(
        ground_truth_path, delimiter=",", dtype=str, ndmin=1
    )
    labels = tuple(str(value) for value in labels_array.tolist())
    if features.shape != (len(EXPECTED_CLASSES), 512):
        raise ValueError(
            f"Expected six 512-value prediction rows, obtained {features.shape}"
        )
    if labels != EXPECTED_CLASSES:
        raise ValueError(f"Expected ground truth {EXPECTED_CLASSES}, obtained {labels}")
    if not np.isfinite(features).all():
        raise ValueError("ISPU validation input contains NaN or infinite values")

    inputs = features.reshape(len(features), 1, 512, 1).astype(np.float32)
    label_indices = np.asarray([CLASS_NAMES.index(label) for label in labels])
    outputs = np.eye(len(CLASS_NAMES), dtype=np.float32)[label_indices]
    return inputs, outputs, labels


def _prediction_records(scores: np.ndarray, labels: tuple[str, ...]) -> list[dict]:
    probabilities = np.asarray(scores, dtype=np.float32).reshape(len(labels), -1)
    if probabilities.shape != (len(labels), len(CLASS_NAMES)):
        raise ValueError(
            f"Expected output shape ({len(labels)}, {len(CLASS_NAMES)}), "
            f"obtained {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("ISPU output contains NaN or infinite values")

    records = []
    for row, (score_row, ground_truth) in enumerate(zip(probabilities, labels)):
        predicted_index = int(np.argmax(score_row))
        ground_truth_index = CLASS_NAMES.index(ground_truth)
        records.append(
            {
                "row": row,
                "ground_truth_index": ground_truth_index,
                "ground_truth_class": ground_truth,
                "predicted_index": predicted_index,
                "predicted_class": CLASS_NAMES[predicted_index],
                "correct": predicted_index == ground_truth_index,
                "scores": {
                    name: float(score_row[index])
                    for index, name in enumerate(CLASS_NAMES)
                },
            }
        )
    return records


def _result_payload(
    target: str,
    scores: np.ndarray,
    labels: tuple[str, ...],
    model_sha256: str,
    validation_fingerprint: str,
    command: list[str] | None = None,
    source_npz: Path | None = None,
) -> dict:
    records = _prediction_records(scores, labels)
    correct = sum(record["correct"] for record in records)
    payload = {
        "target": target,
        "model": str(MODEL_PATH),
        "model_sha256": model_sha256,
        "validation_fingerprint": validation_fingerprint,
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "predictions": records,
    }
    if command is not None:
        payload["command"] = command
    if source_npz is not None:
        payload["source_npz"] = str(source_npz)
        payload["source_npz_sha256"] = _sha256_file(source_npz)
    return payload


def _print_prediction_summary(payload: dict) -> None:
    print("Row  Ground truth  Prediction  Correct  Scores [TEP, DIMP, DMMP]")
    for record in payload["predictions"]:
        scores = [record["scores"][name] for name in CLASS_NAMES]
        print(
            f"{record['row'] + 1:>3}  {record['ground_truth_class']:<12}  "
            f"{record['predicted_class']:<10}  "
            f"{'yes' if record['correct'] else 'no':<7}  "
            f"[{scores[0]:.8f}, {scores[1]:.8f}, {scores[2]:.8f}]"
        )


def prepare_validation() -> dict:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}; complete Model Zoo quantization first"
        )
    inputs, outputs, labels = build_validation_arrays()
    details = tflite_details(MODEL_PATH)
    if details["input"]["shape_signature"] != [1, 1, 512, 1]:
        raise ValueError(f"Unexpected model input: {details['input']}")
    if details["output"]["shape_signature"] != [1, 3]:
        raise ValueError(f"Unexpected model output: {details['output']}")
    if not details["fully_integer_io"]:
        raise ValueError("The ISPU validation model must have full INT8 I/O")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VALIDATION_DATA_PATH, m_inputs=inputs, m_outputs=outputs)
    model_hash = _sha256_file(MODEL_PATH)
    fingerprint = _validation_fingerprint(inputs, outputs)
    reference_scores = predict_tflite(MODEL_PATH, inputs)
    reference = _result_payload(
        "tflite_host", reference_scores, labels, model_hash, fingerprint
    )
    manifest = {
        "model": str(MODEL_PATH),
        "model_sha256": model_hash,
        "validation_data": str(VALIDATION_DATA_PATH),
        "validation_fingerprint": fingerprint,
        "class_names": list(CLASS_NAMES),
        "expected_classes": list(labels),
        "input_shape": list(inputs.shape),
        "output_shape": list(outputs.shape),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    REFERENCE_PATH.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    print(f"ISPU validation data written to {VALIDATION_DATA_PATH}")
    _print_prediction_summary(reference)
    print(
        "TFLite reference accuracy: "
        f"{reference['correct']}/{reference['total']} ({reference['accuracy']:.2%})"
    )
    return manifest


def _load_manifest() -> tuple[dict, np.ndarray, np.ndarray, tuple[str, ...]]:
    if not MANIFEST_PATH.is_file() or not VALIDATION_DATA_PATH.is_file():
        raise FileNotFoundError("Missing ISPU validation data; run the `prepare` command first")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    with np.load(VALIDATION_DATA_PATH, allow_pickle=False) as data:
        inputs = np.asarray(data["m_inputs"], dtype=np.float32)
        outputs = np.asarray(data["m_outputs"], dtype=np.float32)
    labels = tuple(manifest["expected_classes"])
    if _sha256_file(MODEL_PATH) != manifest["model_sha256"]:
        raise ValueError("The model differs from the model used to prepare ISPU validation")
    if _validation_fingerprint(inputs, outputs) != manifest["validation_fingerprint"]:
        raise ValueError("ISPU validation data differs from its manifest")
    if labels != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected manifest ground truth: {labels}")
    return manifest, inputs, outputs, labels


def _resolve_executable(environment_name: str, candidates: tuple[str, ...]) -> str:
    configured = os.environ.get(environment_name)
    if configured:
        resolved = shutil.which(configured) or (
            str(Path(configured).resolve()) if Path(configured).is_file() else None
        )
        if resolved:
            return resolved
        raise FileNotFoundError(
            f"{environment_name} does not identify an executable: {configured}"
        )
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise FileNotFoundError(
        f"Set {environment_name} or add one of {candidates} to PATH"
    )


def _run_and_log(command: list[str], log_path: Path, environment: dict | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    rendered = shlex.join(command)
    log_text = (
        f"$ {rendered}\n\n{completed.stdout}"
        + (f"\n[stderr]\n{completed.stderr}" if completed.stderr else "")
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log_text, encoding="utf-8")
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path}"
        )


def _generated_outputs(output_dir: Path) -> tuple[Path, np.ndarray]:
    candidates = sorted(output_dir.glob("*_val_io.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one *_val_io.npz in {output_dir}, found {len(candidates)}"
        )
    path = candidates[0]
    with np.load(path, allow_pickle=False) as data:
        keys = [key for key in data.files if key == "c_outputs" or key.startswith("c_outputs_")]
        if len(keys) != 1:
            raise ValueError(f"Expected one generated-model output in {path}, found {keys}")
        scores = np.asarray(data[keys[0]])
    if np.issubdtype(scores.dtype, np.integer):
        output = tflite_details(MODEL_PATH)["output"]
        scores = (
            scores.astype(np.float32) - int(output["zero_point"])
        ) * float(output["scale"])
    return path, scores.astype(np.float32)


def _set_ispu_clock(conf_path: Path, clock_mhz: int) -> None:
    if clock_mhz not in (5, 10):
        raise ValueError(f"ISPU clock must be 5 or 10 MHz, obtained {clock_mhz}")
    text = conf_path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"(?m)^ispu_clock[ \t]+\d+[ \t]*$",
        f"ispu_clock {clock_mhz}",
        text,
    )
    if count != 1:
        raise ValueError(f"Expected one ispu_clock setting in {conf_path}, found {count}")
    conf_path.write_text(updated, encoding="utf-8")


def _prepare_ispu_configuration(
    executable: str,
    environment: dict,
    ispu_repository: Path,
    clock_mhz: int,
) -> Path:
    repository = ispu_repository.resolve()
    revision = _ispu_repository_revision(repository)
    if revision != ISPU_REPOSITORY_REVISION:
        raise ValueError(
            f"Expected st-mems-ispu {ISPU_REPOSITORY_REVISION}, found {revision}"
        )
    template = (
        repository
        / "examples/ism330is_lsm6dso16is/template_stedgeai_validate/ispu"
    )
    if not template.is_dir():
        raise FileNotFoundError(f"Missing ISPU validation template: {template}")

    application_dir = OUTPUT_ROOT / f"ispu_application_{clock_mhz}mhz"
    if application_dir.exists():
        shutil.rmtree(application_dir)
    shutil.copytree(template, application_dir)
    _set_ispu_clock(application_dir / "conf.txt", clock_mhz)

    generate_command = [
        executable,
        "generate",
        "--target",
        "ispu",
        "--device",
        ISPU_DEVICE,
        "--model",
        str(MODEL_PATH),
        "--input-data-type",
        "float32",
        "--output-data-type",
        "float32",
        "--no-workspace",
        "--no-report",
        "--output",
        str(application_dir),
    ]
    _run_and_log(
        generate_command,
        OUTPUT_ROOT / f"generate_{clock_mhz}mhz.log",
        environment,
    )
    make_executable = shutil.which("make", path=environment.get("PATH", ""))
    if make_executable is None:
        raise FileNotFoundError("GNU Make is required to build the ISPU configuration")
    _run_and_log(
        [make_executable, "-C", str(application_dir / "make")],
        OUTPUT_ROOT / f"build_{clock_mhz}mhz.log",
        environment,
    )
    configuration = application_dir / "make/bin/ispu.json"
    if not configuration.is_file():
        raise FileNotFoundError(f"ISPU build did not create {configuration}")
    return configuration


def _target_duration_ms(log_path: Path) -> float:
    log = log_path.read_text(encoding="utf-8")
    match = re.search(
        r"duration\s*:\s*(?P<duration>\d+(?:\.\d+)?)\s*ms\s+by\s+sample",
        log,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise AssertionError(f"ISPU target log does not report duration: {log_path}")
    duration = float(match.group("duration"))
    if not np.isfinite(duration) or duration <= 0:
        raise AssertionError(f"ISPU target log reports invalid duration: {duration}")
    return duration


def run_validation(
    mode: str,
    ispu_repository: Path | None = None,
    clock_mhz: int = EXPECTED_ISPU_CLOCK_MHZ,
) -> dict:
    if mode not in ("host", "target"):
        raise ValueError(f"Unsupported ISPU validation mode: {mode}")
    manifest, _, _, labels = _load_manifest()
    executable = _resolve_executable("STEDGEAI_PATH", ("stedgeai", "stedgeai.exe"))
    environment = os.environ.copy()
    toolchain_bin = environment.get("ISPU_TOOLCHAIN_BIN")
    if toolchain_bin:
        environment["PATH"] = (
            str(Path(toolchain_bin).resolve())
            + os.pathsep
            + environment.get("PATH", "")
        )
    if mode == "target":
        missing_tools = [
            executable_name
            for executable_name in ("stred-gcc", "make")
            if shutil.which(executable_name, path=environment.get("PATH", "")) is None
        ]
        if missing_tools:
            raise FileNotFoundError(
                "Target validation requires "
                + ", ".join(missing_tools)
                + "; add the ISPU Toolchain bin directory to PATH or set "
                "ISPU_TOOLCHAIN_BIN, and install GNU Make"
            )

    ispu_configuration = None
    if mode == "target":
        repository = ispu_repository or ROOT.parent / "st-mems-ispu"
        ispu_configuration = _prepare_ispu_configuration(
            executable, environment, repository, clock_mhz
        )

    output_dir = HOST_OUTPUT_DIR if mode == "host" else TARGET_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "prediction_results.json"
    if result_path.exists():
        result_path.unlink()
    for old_npz in output_dir.glob("*_val_io.npz"):
        old_npz.unlink()

    command = [
        executable,
        "validate",
        "--target",
        "ispu",
        "--device",
        ISPU_DEVICE,
        "--mode",
        mode,
        "--model",
        str(MODEL_PATH),
        "--input-data-type",
        "float32",
        "--output-data-type",
        "float32",
        "--no-workspace",
        "--output",
        str(output_dir),
        "--valinput",
        str(VALIDATION_DATA_PATH),
    ]
    if ispu_configuration is not None:
        command.extend(["--ispu-conf", str(ispu_configuration)])
    log_path = output_dir / "validation.log"
    _run_and_log(command, log_path, environment)
    source_npz, scores = _generated_outputs(output_dir)
    payload = _result_payload(
        f"stedgeai_ispu_{mode}",
        scores,
        labels,
        manifest["model_sha256"],
        manifest["validation_fingerprint"],
        command=command,
        source_npz=source_npz,
    )
    if mode == "target":
        payload["ispu_clock_mhz"] = clock_mhz
        payload["ispu_configuration"] = str(ispu_configuration)
        payload["ispu_configuration_sha256"] = _sha256_file(ispu_configuration)
        payload["duration_ms"] = _target_duration_ms(log_path)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_prediction_summary(payload)
    print(
        f"ISPU {mode} accuracy against ground truth: "
        f"{payload['correct']}/{payload['total']} ({payload['accuracy']:.2%})"
    )
    if mode == "target":
        print(
            f"ISPU target duration at {clock_mhz} MHz: "
            f"{payload['duration_ms']:.3f} ms by sample (observed, not a pass/fail limit)"
        )
    print(f"Prediction results written to {result_path}")
    return payload


def _ispu_repository_revision(ispu_repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ispu_repository,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def flash_bridge(
    board: str,
    ispu_repository: Path,
    probe_index: int = 0,
) -> None:
    if board not in BRIDGE_BINARIES:
        raise ValueError(
            f"Unsupported bridge board {board}; choose one of {tuple(BRIDGE_BINARIES)}"
        )
    repository = ispu_repository.resolve()
    revision = _ispu_repository_revision(repository)
    if revision != ISPU_REPOSITORY_REVISION:
        raise ValueError(
            f"Expected st-mems-ispu {ISPU_REPOSITORY_REVISION}, found {revision}"
        )
    binary = (
        repository
        / "host_firmware/nucleo_ispu_stedgeai_validate/binary"
        / BRIDGE_BINARIES[board]
    )
    if not binary.is_file():
        raise FileNotFoundError(f"Missing bridge firmware: {binary}")
    programmer = _resolve_executable(
        "STM32_PROGRAMMER_CLI", ("STM32_Programmer_CLI", "STM32_Programmer_CLI.exe")
    )
    command = [
        programmer,
        "-c",
        "port=SWD",
        f"index={probe_index}",
        "-w",
        str(binary),
        "0x08000000",
        "-s",
        "0x08000000",
    ]
    _run_and_log(command, OUTPUT_ROOT / "bridge_flash.log")
    print(f"ISPU validation bridge flashed on {board}")


def _load_result(path: Path, expected_target: str, manifest: dict) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing ISPU result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("target") != expected_target:
        raise AssertionError(
            f"{path}: expected target {expected_target}, obtained {payload.get('target')}"
        )
    if payload.get("model_sha256") != manifest["model_sha256"]:
        raise AssertionError(f"{path}: result was generated with a different model")
    if payload.get("validation_fingerprint") != manifest["validation_fingerprint"]:
        raise AssertionError(f"{path}: result used different validation samples")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(EXPECTED_CLASSES):
        raise AssertionError(f"{path}: expected six predictions")
    ground_truth = tuple(item.get("ground_truth_class") for item in predictions)
    if ground_truth != EXPECTED_CLASSES:
        raise AssertionError(f"{path}: unexpected ground truth {ground_truth}")
    for item in predictions:
        expected_correct = item.get("predicted_class") == item.get("ground_truth_class")
        if item.get("correct") != expected_correct:
            raise AssertionError(f"{path}: invalid correctness flag in row {item.get('row')}")
    return predictions


def verify_ispu_results(
    host_path: Path = HOST_OUTPUT_DIR / "prediction_results.json",
    target_path: Path = TARGET_OUTPUT_DIR / "prediction_results.json",
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, float | int]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing ISPU manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    host = _load_result(host_path, "stedgeai_ispu_host", manifest)
    target = _load_result(target_path, "stedgeai_ispu_target", manifest)
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    if target_payload.get("ispu_clock_mhz") != EXPECTED_ISPU_CLOCK_MHZ:
        raise AssertionError(
            f"Physical ISPU result must use {EXPECTED_ISPU_CLOCK_MHZ} MHz, "
            f"obtained {target_payload.get('ispu_clock_mhz')}"
        )
    target_duration_ms = float(target_payload.get("duration_ms", 0.0))
    if not np.isfinite(target_duration_ms) or target_duration_ms <= 0:
        raise AssertionError(
            f"Physical ISPU result has invalid observed duration {target_duration_ms}"
        )
    host_classes = [item["predicted_class"] for item in host]
    target_classes = [item["predicted_class"] for item in target]
    if tuple(host_classes) != EXPECTED_CLASSES:
        raise AssertionError(f"ISPU host predictions are incorrect: {host_classes}")
    if target_classes != host_classes:
        raise AssertionError(
            f"Physical ISPU predictions {target_classes} differ from host {host_classes}"
        )

    host_scores = np.asarray(
        [[item["scores"][name] for name in CLASS_NAMES] for item in host],
        dtype=np.float64,
    )
    target_scores = np.asarray(
        [[item["scores"][name] for name in CLASS_NAMES] for item in target],
        dtype=np.float64,
    )
    for name, scores in (("host", host_scores), ("target", target_scores)):
        if not np.isfinite(scores).all():
            raise AssertionError(f"ISPU {name} scores contain non-finite values")
        if np.any(scores < -1e-6) or np.any(scores > 1.0 + 1e-6):
            raise AssertionError(f"ISPU {name} scores are outside the probability range")
        if np.any(np.abs(scores.sum(axis=1) - 1.0) > 0.05):
            raise AssertionError(f"ISPU {name} scores do not form probability vectors")
    return {
        "correct": len(target_classes),
        "total": len(EXPECTED_CLASSES),
        "max_score_delta": float(np.max(np.abs(host_scores - target_scores))),
        "clock_mhz": EXPECTED_ISPU_CLOCK_MHZ,
        "duration_ms": target_duration_ms,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="create deterministic ISPU validation data")
    subparsers.add_parser("host", help="validate generated ISPU code on the host")
    target_parser = subparsers.add_parser(
        "target", help="validate the model on a physical ISPU"
    )
    target_parser.add_argument(
        "--ispu-repository",
        type=Path,
        default=ROOT.parent / "st-mems-ispu",
    )
    target_parser.add_argument(
        "--clock-mhz",
        type=int,
        choices=(5, 10),
        default=EXPECTED_ISPU_CLOCK_MHZ,
    )
    flash_parser = subparsers.add_parser("flash", help="flash the Nucleo bridge firmware")
    flash_parser.add_argument("--board", choices=tuple(BRIDGE_BINARIES), required=True)
    flash_parser.add_argument("--ispu-repository", type=Path, required=True)
    flash_parser.add_argument("--probe-index", type=int, default=0)
    subparsers.add_parser("verify", help="compare host and physical ISPU predictions")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "prepare":
        prepare_validation()
    elif args.command == "host":
        run_validation(args.command)
    elif args.command == "target":
        run_validation(args.command, args.ispu_repository, args.clock_mhz)
    elif args.command == "flash":
        flash_bridge(args.board, args.ispu_repository, args.probe_index)
    elif args.command == "verify":
        result = verify_ispu_results()
        print(
            "Physical ISPU inference verified: "
            f"{result['correct']}/{result['total']} predictions match host and ground truth"
        )
        print(f"Maximum host/ISPU score delta: {result['max_score_delta']:.8f}")
        print(
            f"Observed ISPU duration at {result['clock_mhz']} MHz: "
            f"{result['duration_ms']:.3f} ms by sample"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
