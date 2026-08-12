from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/stm32_model_zoo"


def test_board_configs_have_independent_targets_and_output_directories():
    expected = {
        "stm32n6": ("raman_linear_softmax_stm32n6", "STM32N6570-DK"),
        "stm32u5": ("raman_linear_softmax_stm32u5", "B-U585I-IOT02A"),
        "stm32f4": ("raman_linear_softmax_stm32f4", "NUCLEO-F401RE"),
    }
    output_dirs = set()
    for family, (project_name, board) in expected.items():
        path = CONFIG_DIR / f"linear_softmax_benchmarking_{family}_config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["general"]["project_name"] == project_name
        assert config["benchmarking"]["board"] == board
        output_dir = config["hydra"]["run"]["dir"]
        assert output_dir.endswith(f"benchmarking_{family}")
        output_dirs.add(output_dir)
    assert len(output_dirs) == len(expected)


def test_readme_contains_the_complete_model_zoo_reproduction_sequence():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    expected_configs = (
        "linear_softmax_training_config.yaml",
        "linear_softmax_evaluation_float_config.yaml",
        "linear_softmax_quantization_config.yaml",
        "linear_softmax_evaluation_int8_config.yaml",
        "linear_softmax_benchmarking_stm32n6_config.yaml",
        "linear_softmax_benchmarking_stm32u5_config.yaml",
        "linear_softmax_benchmarking_stm32f4_config.yaml",
        "linear_softmax_prediction_host_config.yaml",
        "linear_softmax_prediction_stm32n6_config.yaml",
    )
    for config_name in expected_configs:
        assert config_name in text
        assert f"--config-name {config_name}" in text


def test_overlay_contains_the_required_cross_platform_and_privacy_fixes():
    text = (ROOT / "scripts/install_model_zoo_overlay.py").read_text(encoding="utf-8")
    assert "string.replace(match, var_value, 1)" in text
    assert "Path(tracking_uri).resolve().as_uri()" in text
    assert 'startswith("raman_linear_softmax")' in text
    assert "clearml_connection_new" in text
    assert "gen_load_val_predict(cfg=configs, model=model)" in text
    assert "self.ai_runner = ai_runner_interp" in text
    assert "parse_prediction_section(cfg.prediction)" in text
    assert "Prediction accuracy against ground truth" in text
    assert '"ground_truth_class":' in text
    assert "prediction_ground_truth.csv" in text


def test_prediction_configs_share_model_input_and_preprocessing():
    host = yaml.safe_load(
        (CONFIG_DIR / "linear_softmax_prediction_host_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    n6 = yaml.safe_load(
        (CONFIG_DIR / "linear_softmax_prediction_stm32n6_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert host["model"]["model_path"] == n6["model"]["model_path"]
    assert host["dataset"]["prediction_path"] == n6["dataset"]["prediction_path"]
    assert host["preprocessing"] == n6["preprocessing"]
    assert host["prediction"]["target"] == "host"
    assert n6["prediction"]["target"] == "stedgeai_n6"
    assert n6["prediction"]["input_type"] == "float32"
    assert n6["prediction"]["output_type"] == "float32"
    assert n6["tools"]["stedgeai"]["on_cloud"] is False
