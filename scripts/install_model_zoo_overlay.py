#!/usr/bin/env python3
"""Clone, validate, and patch Model Zoo Services v4.1.1 for this project."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


EXPECTED_REVISION = "0f6210ed5156126b782e1c43249063a477484b20"
MODEL_ZOO_TAG = "v4.1.1"
MODEL_ZOO_URL = "https://github.com/STMicroelectronics/stm32ai-modelzoo-services.git"
EXPECTED_FILES = (
    "arc_fault_detection/tf/src/models/__init__.py",
    "arc_fault_detection/tf/wrappers/models/custom_models/models.py",
    "arc_fault_detection/stm32ai_main.py",
    "arc_fault_detection/tf/src/quantization/tflite_quantizer.py",
    "arc_fault_detection/tf/src/evaluation/tflite_evaluator.py",
    "common/utils/cfg_utils.py",
    "common/utils/logs_utils.py",
)


def clone_if_missing(model_zoo_dir: Path) -> None:
    if model_zoo_dir.exists():
        return
    model_zoo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            MODEL_ZOO_TAG,
            "--depth",
            "1",
            MODEL_ZOO_URL,
            str(model_zoo_dir),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", EXPECTED_REVISION],
        cwd=model_zoo_dir,
        check=True,
    )


def _require_expected_checkout(model_zoo_dir: Path) -> None:
    missing = [relative for relative in EXPECTED_FILES if not (model_zoo_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Not a compatible Model Zoo Services clone; missing: {missing}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=model_zoo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"Expected Model Zoo Services {EXPECTED_REVISION}, found {revision}. "
            f"Run: git -C {model_zoo_dir} checkout {EXPECTED_REVISION}"
        )


def install(project_root: Path, model_zoo_dir: Path) -> None:
    _require_expected_checkout(model_zoo_dir)
    model_source = (
        project_root
        / "integration/stm32ai-modelzoo-services/arc_fault_detection/tf/src/models/raman_linear_softmax.py"
    )
    model_target = (
        model_zoo_dir
        / "arc_fault_detection/tf/src/models/raman_linear_softmax.py"
    )
    init_path = model_zoo_dir / "arc_fault_detection/tf/src/models/__init__.py"
    registry_path = (
        model_zoo_dir
        / "arc_fault_detection/tf/wrappers/models/custom_models/models.py"
    )
    quantizer_path = (
        model_zoo_dir / "arc_fault_detection/tf/src/quantization/tflite_quantizer.py"
    )
    evaluator_path = (
        model_zoo_dir / "arc_fault_detection/tf/src/evaluation/tflite_evaluator.py"
    )
    main_path = model_zoo_dir / "arc_fault_detection/stm32ai_main.py"
    config_utils_path = model_zoo_dir / "common/utils/cfg_utils.py"
    logs_utils_path = model_zoo_dir / "common/utils/logs_utils.py"

    init_text = init_path.read_text(encoding="utf-8")
    registry_text = registry_path.read_text(encoding="utf-8")
    quantizer_text = quantizer_path.read_text(encoding="utf-8")
    evaluator_text = evaluator_path.read_text(encoding="utf-8")
    main_text = main_path.read_text(encoding="utf-8")
    config_utils_text = config_utils_path.read_text(encoding="utf-8")
    logs_utils_text = logs_utils_path.read_text(encoding="utf-8")

    model_import = "from .raman_linear_softmax import get_raman_linear_softmax"
    if model_import not in init_text:
        init_text = init_text.rstrip() + f"\n{model_import}\n"

    registry_entry = "    'raman_linear_softmax': (get_raman_linear_softmax, {}),\n"
    if registry_entry.strip() not in registry_text:
        marker = "TF_MODEL_FNS = {\n"
        if marker not in registry_text:
            raise RuntimeError(f"Unexpected registry structure in {registry_path}")
        registry_text = registry_text.replace(marker, marker + registry_entry, 1)

    # AFD 4.1.1 reads these values from the wrong configuration level.
    replacements = {
        "configs.quantization_input_type": "configs.quantization.quantization_input_type",
        "configs.quantization_output_type": "configs.quantization.quantization_output_type",
        "configs.quantization_granularity": "configs.quantization.granularity",
    }
    for old, new in replacements.items():
        quantizer_text = quantizer_text.replace(old, new)

    static_batch_guard = """    # Train with variable batches and deploy with a static batch of one.
    if model.name == "raman_linear_softmax" and model.input_shape[0] is None:
        static_input = tf.keras.Input(
            batch_shape=(1, *model.input_shape[1:]),
            dtype=model.input.dtype,
            name=model.input.name.split(":")[0],
        )
        static_model = tf.keras.models.clone_model(model, input_tensors=static_input)
        static_model.set_weights(model.get_weights())
        model = static_model

"""
    converter_marker = "    # Create the TFLite converter\n"
    if "Train with variable batches and deploy with a static batch of one" not in quantizer_text:
        if converter_marker not in quantizer_text:
            raise RuntimeError(f"Unexpected quantizer structure in {quantizer_path}")
        quantizer_text = quantizer_text.replace(
            converter_marker, static_batch_guard + converter_marker, 1
        )

    int8_guard = """    if (
        configs.quantization.quantization_input_type == 'int8'
        and configs.quantization.quantization_output_type == 'int8'
    ):
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

"""
    optimization_marker = "    # Set the optimizations and representative dataset generator\n"
    if "converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]" not in quantizer_text:
        if optimization_marker not in quantizer_text:
            raise RuntimeError(f"Unexpected quantizer structure in {quantizer_path}")
        quantizer_text = quantizer_text.replace(
            optimization_marker, int8_guard + optimization_marker, 1
        )

    static_eval_old = '''        expected_shape = list(input_detail["shape"])
        expected_shape[0] = features.shape[0]
        # Get shape of a batch
        interpreter.resize_tensor_input(input_detail["index"], expected_shape)
        interpreter.allocate_tensors()

        tf.print(f"[INFO] : Quantization input details : {input_detail['quantization']}")
        tf.print(f"[INFO] : Dtype input details : {input_detail['dtype']}")

        x_proc = self._quantize_input(features, input_detail)
        interpreter.set_tensor(input_detail["index"], x_proc)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_detail["index"])
'''
    static_eval_new = '''        tf.print(f"[INFO] : Quantization input details : {input_detail['quantization']}")
        tf.print(f"[INFO] : Dtype input details : {input_detail['dtype']}")

        x_proc = self._quantize_input(features, input_detail)
        shape_signature = input_detail.get("shape_signature", input_detail["shape"])
        if int(shape_signature[0]) == 1:
            interpreter.allocate_tensors()
            raw_outputs = []
            for sample in x_proc:
                interpreter.set_tensor(input_detail["index"], sample[np.newaxis, ...])
                interpreter.invoke()
                raw_outputs.append(interpreter.get_tensor(output_detail["index"]).copy())
            raw_out = np.concatenate(raw_outputs, axis=0)
        else:
            expected_shape = list(input_detail["shape"])
            expected_shape[0] = features.shape[0]
            interpreter.resize_tensor_input(input_detail["index"], expected_shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_detail["index"], x_proc)
            interpreter.invoke()
            raw_out = interpreter.get_tensor(output_detail["index"])
'''
    if "shape_signature = input_detail.get" not in evaluator_text:
        if static_eval_old not in evaluator_text:
            raise RuntimeError(f"Unexpected evaluator structure in {evaluator_path}")
        evaluator_text = evaluator_text.replace(static_eval_old, static_eval_new, 1)

    # Upstream 4.1.1 uses a regex replacement, so Windows backslashes are
    # interpreted as escapes (for example, ``\S``). A literal replacement is
    # correct for environment-variable expansion on every platform.
    windows_env_old = '''        match = "\\\\" + match
        string = re.sub(match, var_value, string, count=1)
'''
    windows_env_new = '''        string = string.replace(match, var_value, 1)
'''
    if "string = string.replace(match, var_value, 1)" not in config_utils_text:
        if windows_env_old not in config_utils_text:
            raise RuntimeError(f"Unexpected environment expansion in {config_utils_path}")
        config_utils_text = config_utils_text.replace(
            windows_env_old, windows_env_new, 1
        )

    # MLflow treats a Windows drive letter as a URI scheme. Convert an absolute
    # local path such as C:/project/mlruns to a standards-compliant file URI.
    if "from pathlib import Path" not in logs_utils_text:
        logs_utils_text = logs_utils_text.replace(
            "import os\n", "import os\nfrom pathlib import Path\n", 1
        )
    mlflow_uri_old = "    mlflow.set_tracking_uri(cfg['mlflow']['uri'])\n"
    mlflow_uri_new = '''    tracking_uri = cfg['mlflow']['uri']
    if os.name == "nt" and os.path.isabs(tracking_uri):
        tracking_uri = Path(tracking_uri).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
'''
    if "tracking_uri = Path(tracking_uri).resolve().as_uri()" not in logs_utils_text:
        if mlflow_uri_old not in logs_utils_text:
            raise RuntimeError(f"Unexpected MLflow initialization in {logs_utils_path}")
        logs_utils_text = logs_utils_text.replace(
            mlflow_uri_old, mlflow_uri_new, 1
        )

    # ClearML is optional upstream and is unrelated to this reproduction. Do
    # not let a user's global ~/.clearml.conf upload this project's artifacts.
    clearml_old = "    if get_active_config_file() is not None:\n"
    clearml_new = '''    if (
        not str(cfg.general.project_name).startswith("raman_linear_softmax")
        and get_active_config_file() is not None
    ):
'''
    if 'not str(cfg.general.project_name).startswith("raman_linear_softmax")' not in main_text:
        if clearml_old not in main_text:
            raise RuntimeError(f"Unexpected ClearML initialization in {main_path}")
        main_text = main_text.replace(clearml_old, clearml_new, 1)

    clearml_connection_old = "    if get_active_config_file() is not None: \n"
    clearml_connection_new = '''    if (
        not str(configs.general.project_name).startswith("raman_linear_softmax")
        and get_active_config_file() is not None
    ):
'''
    if 'not str(configs.general.project_name).startswith("raman_linear_softmax")' not in main_text:
        if clearml_connection_old not in main_text:
            raise RuntimeError(f"Unexpected ClearML task connection in {main_path}")
        main_text = main_text.replace(
            clearml_connection_old, clearml_connection_new, 1
        )

    shutil.copy2(model_source, model_target)
    init_path.write_text(init_text, encoding="utf-8")
    registry_path.write_text(registry_text, encoding="utf-8")
    quantizer_path.write_text(quantizer_text, encoding="utf-8")
    evaluator_path.write_text(evaluator_text, encoding="utf-8")
    main_path.write_text(main_text, encoding="utf-8")
    config_utils_path.write_text(config_utils_text, encoding="utf-8")
    logs_utils_path.write_text(logs_utils_text, encoding="utf-8")
    print(f"Installed and registered raman_linear_softmax in {model_zoo_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zoo-dir", required=True, type=Path)
    parser.add_argument(
        "--clone",
        action="store_true",
        help="clone the pinned Model Zoo release when the target directory is absent",
    )
    args = parser.parse_args()
    model_zoo_dir = args.model_zoo_dir.resolve()
    if args.clone:
        clone_if_missing(model_zoo_dir)
    install(Path(__file__).resolve().parents[1], model_zoo_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
