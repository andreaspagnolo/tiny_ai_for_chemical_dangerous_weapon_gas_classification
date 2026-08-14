from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.ispu_validation as ispu_validation
from scripts.ispu_validation import (
    CLASS_NAMES,
    EXPECTED_ISPU_CLOCK_MHZ,
    EXPECTED_CLASSES,
    ISPU_REPOSITORY_REVISION,
    ROOT,
    _generated_outputs,
    _prepare_ispu_configuration,
    _result_payload,
    _target_duration_ms,
    _validation_fingerprint,
    build_validation_arrays,
    verify_ispu_results,
)


def test_validation_arrays_use_the_six_prediction_samples(tmp_path):
    prediction_path = tmp_path / "prediction.csv"
    ground_truth_path = tmp_path / "prediction_ground_truth.csv"
    features = np.arange(6 * 512, dtype=np.float32).reshape(6, 512) / 4096.0
    np.savetxt(prediction_path, features, delimiter=",")
    ground_truth_path.write_text("\n".join(EXPECTED_CLASSES) + "\n", encoding="utf-8")

    inputs, outputs, labels = build_validation_arrays(
        prediction_path, ground_truth_path
    )

    assert inputs.shape == (6, 1, 512, 1)
    assert inputs.dtype == np.float32
    assert outputs.shape == (6, 3)
    assert outputs.dtype == np.float32
    assert tuple(CLASS_NAMES[index] for index in np.argmax(outputs, axis=1)) == labels
    assert labels == EXPECTED_CLASSES


def test_generated_outputs_reads_the_c_model_tensor(tmp_path):
    scores = np.eye(3, dtype=np.float32)[[0, 0, 1, 1, 2, 2]]
    np.savez(tmp_path / "network_val_io.npz", m_outputs_1=scores, c_outputs_1=scores)

    path, actual = _generated_outputs(tmp_path)

    assert path.name == "network_val_io.npz"
    np.testing.assert_array_equal(actual, scores)


def test_host_validation_uses_st_cli_and_writes_normalized_results(
    tmp_path, monkeypatch, capsys
):
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"model")
    inputs = np.zeros((6, 1, 512, 1), dtype=np.float32)
    outputs = np.eye(3, dtype=np.float32)[[0, 0, 1, 1, 2, 2]]
    validation_path = tmp_path / "validation_data.npz"
    np.savez(validation_path, m_inputs=inputs, m_outputs=outputs)
    fingerprint = _validation_fingerprint(inputs, outputs)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_sha256": ispu_validation._sha256_file(model_path),
                "validation_fingerprint": fingerprint,
                "expected_classes": list(EXPECTED_CLASSES),
            }
        ),
        encoding="utf-8",
    )
    host_output = tmp_path / "host"
    captured_command = []

    def fake_run(command, log_path, environment):
        captured_command.extend(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("host validation", encoding="utf-8")
        np.savez(
            host_output / "network_val_io.npz",
            m_outputs=outputs,
            c_outputs=outputs,
        )

    monkeypatch.setattr(ispu_validation, "MODEL_PATH", model_path)
    monkeypatch.setattr(ispu_validation, "VALIDATION_DATA_PATH", validation_path)
    monkeypatch.setattr(ispu_validation, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(ispu_validation, "HOST_OUTPUT_DIR", host_output)
    monkeypatch.setattr(
        ispu_validation, "_resolve_executable", lambda *args: "stedgeai"
    )
    monkeypatch.setattr(ispu_validation, "_run_and_log", fake_run)

    result = ispu_validation.run_validation("host")

    assert result["correct"] == 6
    assert captured_command[:7] == [
        "stedgeai",
        "validate",
        "--target",
        "ispu",
        "--device",
        "imu_22",
        "--mode",
    ]
    assert captured_command[captured_command.index("--mode") + 1] == "host"
    assert (host_output / "prediction_results.json").is_file()
    terminal_output = capsys.readouterr().out
    assert "Ground truth  Prediction  Correct" in terminal_output
    assert "ISPU host accuracy against ground truth: 6/6 (100.00%)" in terminal_output


def test_flash_uses_pinned_prebuilt_bridge(tmp_path, monkeypatch):
    repository = tmp_path / "st-mems-ispu"
    binary = (
        repository
        / "host_firmware/nucleo_ispu_stedgeai_validate/binary"
        / "nucleo_f401re_ispu_stedgeai_validate.bin"
    )
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"firmware")
    captured_command = []

    monkeypatch.setattr(
        ispu_validation,
        "_ispu_repository_revision",
        lambda path: ISPU_REPOSITORY_REVISION,
    )
    monkeypatch.setattr(
        ispu_validation, "_resolve_executable", lambda *args: "STM32_Programmer_CLI"
    )
    monkeypatch.setattr(
        ispu_validation,
        "_run_and_log",
        lambda command, log_path: captured_command.extend(command),
    )

    ispu_validation.flash_bridge("nucleo-f401re", repository)

    assert captured_command == [
        "STM32_Programmer_CLI",
        "-c",
        "port=SWD",
        "index=0",
        "-w",
        str(binary),
        "0x08000000",
        "-s",
        "0x08000000",
    ]


def test_target_configuration_is_generated_with_explicit_10mhz_clock(
    tmp_path, monkeypatch
):
    repository = tmp_path / "st-mems-ispu"
    template = (
        repository
        / "examples/ism330is_lsm6dso16is/template_stedgeai_validate/ispu"
    )
    (template / "make").mkdir(parents=True)
    (template / "conf.txt").write_text(
        "acc_odr 6667\n\nispu_clock 5\n", encoding="utf-8"
    )
    output_root = tmp_path / "output"
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"model")
    commands = []

    def fake_run(command, log_path, environment):
        commands.append(command)
        if command[0] == "make":
            configuration = output_root / "ispu_application_10mhz/make/bin/ispu.json"
            configuration.parent.mkdir(parents=True)
            configuration.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(ispu_validation, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(ispu_validation, "MODEL_PATH", model_path)
    monkeypatch.setattr(
        ispu_validation,
        "_ispu_repository_revision",
        lambda path: ISPU_REPOSITORY_REVISION,
    )
    monkeypatch.setattr(ispu_validation, "_run_and_log", fake_run)
    monkeypatch.setattr(
        ispu_validation.shutil,
        "which",
        lambda executable, path=None: "make" if executable == "make" else None,
    )

    configuration = _prepare_ispu_configuration(
        "stedgeai", {"PATH": "tools"}, repository, EXPECTED_ISPU_CLOCK_MHZ
    )

    assert configuration.is_file()
    assert (
        output_root / "ispu_application_10mhz/conf.txt"
    ).read_text(encoding="utf-8").endswith("ispu_clock 10\n")
    assert commands[0][0:6] == [
        "stedgeai",
        "generate",
        "--target",
        "ispu",
        "--device",
        "imu_22",
    ]
    assert commands[1][0] == "make"


def test_target_duration_is_parsed_from_real_st_log_format(tmp_path):
    log_path = tmp_path / "validation.log"
    log_path.write_text(
        "duration       :   23.994 ms by sample (23.261/24.248/0.373)\n",
        encoding="utf-8",
    )

    assert _target_duration_ms(log_path) == pytest.approx(23.994)


def _write_results(
    tmp_path: Path,
    last_target_class: str | None = None,
    clock_mhz: int = EXPECTED_ISPU_CLOCK_MHZ,
):
    labels = EXPECTED_CLASSES
    host_scores = np.eye(3, dtype=np.float32)[[0, 0, 1, 1, 2, 2]]
    target_scores = host_scores * 0.98 + (1.0 - host_scores) * 0.01
    if last_target_class is not None:
        target_scores[-1] = 0.0
        target_scores[-1, CLASS_NAMES.index(last_target_class)] = 1.0
    fingerprint = _validation_fingerprint(
        np.zeros((6, 1, 512, 1), dtype=np.float32),
        np.eye(3, dtype=np.float32)[[0, 0, 1, 1, 2, 2]],
    )
    manifest = {
        "model_sha256": "model-hash",
        "validation_fingerprint": fingerprint,
        "expected_classes": list(labels),
    }
    manifest_path = tmp_path / "manifest.json"
    host_path = tmp_path / "host.json"
    target_path = tmp_path / "target.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    host_path.write_text(
        json.dumps(
            _result_payload(
                "stedgeai_ispu_host",
                host_scores,
                labels,
                "model-hash",
                fingerprint,
            )
        ),
        encoding="utf-8",
    )
    target_path.write_text(
        json.dumps(
            {
                **_result_payload(
                    "stedgeai_ispu_target",
                    target_scores,
                    labels,
                    "model-hash",
                    fingerprint,
                ),
                "ispu_clock_mhz": clock_mhz,
                "duration_ms": 12.345,
            }
        ),
        encoding="utf-8",
    )
    return host_path, target_path, manifest_path


def test_ispu_verifier_accepts_target_predictions_matching_ground_truth(tmp_path):
    host_path, target_path, manifest_path = _write_results(tmp_path)

    result = verify_ispu_results(host_path, target_path, manifest_path)

    assert result["correct"] == 6
    assert result["total"] == 6
    assert result["max_score_delta"] == pytest.approx(0.02)
    assert result["clock_mhz"] == 10
    assert result["duration_ms"] == pytest.approx(12.345)


def test_ispu_verifier_rejects_physical_class_disagreement(tmp_path):
    host_path, target_path, manifest_path = _write_results(
        tmp_path, last_target_class="TEP"
    )

    with pytest.raises(AssertionError, match="differ from host"):
        verify_ispu_results(host_path, target_path, manifest_path)


def test_ispu_verifier_rejects_ambiguous_5mhz_target_run(tmp_path):
    host_path, target_path, manifest_path = _write_results(tmp_path, clock_mhz=5)

    with pytest.raises(AssertionError, match="must use 10 MHz"):
        verify_ispu_results(host_path, target_path, manifest_path)


def test_readme_contains_complete_physical_ispu_workflow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert ISPU_REPOSITORY_REVISION in readme
    assert "target --clock-mhz 10 --ispu-repository ../st-mems-ispu" in readme
    for command in (
        "python scripts/ispu_validation.py prepare",
        "python scripts/ispu_validation.py host",
        "python scripts/ispu_validation.py flash --board nucleo-f401re",
        "python scripts/ispu_validation.py target",
        "python scripts/ispu_validation.py verify",
    ):
        assert command in readme
