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
    "arc_fault_detection/tf/src/prediction/tflite_predictor.py",
    "arc_fault_detection/tf/src/utils/parse_config.py",
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
    predictor_path = (
        model_zoo_dir / "arc_fault_detection/tf/src/prediction/tflite_predictor.py"
    )
    parse_config_path = (
        model_zoo_dir / "arc_fault_detection/tf/src/utils/parse_config.py"
    )
    main_path = model_zoo_dir / "arc_fault_detection/stm32ai_main.py"
    config_utils_path = model_zoo_dir / "common/utils/cfg_utils.py"
    logs_utils_path = model_zoo_dir / "common/utils/logs_utils.py"

    init_text = init_path.read_text(encoding="utf-8")
    registry_text = registry_path.read_text(encoding="utf-8")
    quantizer_text = quantizer_path.read_text(encoding="utf-8")
    evaluator_text = evaluator_path.read_text(encoding="utf-8")
    predictor_text = predictor_path.read_text(encoding="utf-8")
    parse_config_text = parse_config_path.read_text(encoding="utf-8")
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

    predictor_import_old = "from tabulate import tabulate\n\nfrom common.model_utils.tf_model_loader import load_model_from_path\n"
    predictor_import_new = """from tabulate import tabulate
import json
from pathlib import Path

from common.model_utils.tf_model_loader import load_model_from_path
from common.utils import ai_runner_interp
"""
    if "from common.utils import ai_runner_interp" not in predictor_text:
        if predictor_import_old not in predictor_text:
            raise RuntimeError(f"Unexpected predictor imports in {predictor_path}")
        predictor_text = predictor_text.replace(
            predictor_import_old, predictor_import_new, 1
        )

    predictor_init_old = """        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
"""
    predictor_init_new = """        self.cfg = cfg
        self.model = model
        self.dataloaders = dataloaders
        self.target = getattr(cfg.prediction, "target", "host")
        self.ai_runner = None
        if self.target in ("stedgeai_host", "stedgeai_n6", "stedgeai_h7p"):
            self.ai_runner = ai_runner_interp(self.target, Path(model.model_path).name)
"""
    if "self.ai_runner = ai_runner_interp" not in predictor_text:
        if predictor_init_old not in predictor_text:
            raise RuntimeError(f"Unexpected predictor initialization in {predictor_path}")
        predictor_text = predictor_text.replace(
            predictor_init_old, predictor_init_new, 1
        )

    predictor_host_old = """        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        expected_shape = list(input_detail["shape"])
        expected_shape[0] = x.shape[0]
        interpreter.resize_tensor_input(input_detail["index"], expected_shape)
        interpreter.allocate_tensors()
        x_proc = self._quantize_input(x, input_detail)
        interpreter.set_tensor(input_detail["index"], x_proc)
        interpreter.invoke()
        raw_out = interpreter.get_tensor(output_detail["index"])
        return self._dequantize_output(raw_out, output_detail)
"""
    predictor_host_new = """        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        x_proc = self._quantize_input(x, input_detail)
        shape_signature = input_detail.get("shape_signature", input_detail["shape"])
        if int(shape_signature[0]) == 1:
            interpreter.allocate_tensors()
            outputs = []
            for sample in x_proc:
                interpreter.set_tensor(input_detail["index"], sample[np.newaxis, ...])
                interpreter.invoke()
                outputs.append(interpreter.get_tensor(output_detail["index"]).copy())
            raw_out = np.concatenate(outputs, axis=0)
        else:
            expected_shape = list(input_detail["shape"])
            expected_shape[0] = x.shape[0]
            interpreter.resize_tensor_input(input_detail["index"], expected_shape)
            interpreter.allocate_tensors()
            interpreter.set_tensor(input_detail["index"], x_proc)
            interpreter.invoke()
            raw_out = interpreter.get_tensor(output_detail["index"])
        return self._dequantize_output(raw_out, output_detail)
"""
    if "shape_signature = input_detail.get" not in predictor_text:
        if predictor_host_old not in predictor_text:
            raise RuntimeError(f"Unexpected host prediction in {predictor_path}")
        predictor_text = predictor_text.replace(
            predictor_host_old, predictor_host_new, 1
        )

    target_methods = """    def _get_target_probs(self, x: np.ndarray) -> np.ndarray:
        input_detail = self.ai_runner.get_inputs()[0]
        output_detail = self.ai_runner.get_outputs()[0]
        input_scale = float(np.asarray(input_detail.scale).reshape(-1)[0])
        input_zero_point = int(np.asarray(input_detail.zero_point).reshape(-1)[0])
        output_scale = float(np.asarray(output_detail.scale).reshape(-1)[0])
        output_zero_point = int(np.asarray(output_detail.zero_point).reshape(-1)[0])
        outputs = []
        for sample in x:
            sample_batch = sample[np.newaxis, ...]
            if np.issubdtype(input_detail.dtype, np.integer):
                sample_batch = np.rint(sample_batch / input_scale + input_zero_point)
                limits = np.iinfo(input_detail.dtype)
                sample_batch = np.clip(sample_batch, limits.min, limits.max)
            sample_batch = sample_batch.astype(input_detail.dtype)
            raw_outputs, _ = self.ai_runner.invoke(sample_batch)
            raw_output = np.asarray(raw_outputs[0])
            if np.issubdtype(raw_output.dtype, np.integer) and output_scale > 0:
                raw_output = (raw_output.astype(np.float32) - output_zero_point) * output_scale
            outputs.append(raw_output.astype(np.float32).reshape(1, -1))
        return np.concatenate(outputs, axis=0)

    def _write_results(self, probs: np.ndarray, class_names) -> Path:
        records = []
        for row, scores in enumerate(probs.reshape(probs.shape[0], -1)):
            predicted_index = int(np.argmax(scores))
            records.append({
                "row": row,
                "predicted_index": predicted_index,
                "predicted_class": class_names[predicted_index],
                "scores": {
                    class_name: float(scores[index])
                    for index, class_name in enumerate(class_names)
                },
            })
        output_path = Path(self.cfg.output_dir) / "prediction_results.json"
        output_path.write_text(
            json.dumps({
                "target": self.target,
                "model": str(self.model.model_path),
                "prediction_path": str(self.cfg.dataset.prediction_path),
                "predictions": records,
            }, indent=2),
            encoding="utf-8",
        )
        return output_path

"""
    predictor_method_marker = "    def _format_prediction_table(self, probs: np.ndarray, class_names):\n"
    if "def _get_target_probs" not in predictor_text:
        if predictor_method_marker not in predictor_text:
            raise RuntimeError(f"Unexpected predictor methods in {predictor_path}")
        predictor_text = predictor_text.replace(
            predictor_method_marker, target_methods + predictor_method_marker, 1
        )

    ground_truth_methods = '''    def _load_ground_truth(self, class_names) -> np.ndarray:
        labels_path = Path(self.cfg.dataset.prediction_path).with_name(
            "prediction_ground_truth.csv"
        )
        if not labels_path.is_file():
            raise FileNotFoundError(
                f"Missing prediction ground truth: {labels_path}. "
                "Run `raman-stm32 prepare` again."
            )
        labels = np.loadtxt(labels_path, dtype=str, delimiter=",", ndmin=1)
        try:
            ground_truth = np.asarray(
                [class_names.index(str(label)) for label in labels], dtype=np.int32
            )
        except ValueError as error:
            raise ValueError(
                f"Unknown class in prediction ground truth {labels_path}: {error}"
            ) from error
        return ground_truth

    def _format_ground_truth_table(
        self, probs: np.ndarray, class_names, ground_truth: np.ndarray
    ) -> str:
        scores_by_sample = probs.reshape(probs.shape[0], -1)
        rows = []
        for row, scores in enumerate(scores_by_sample):
            predicted_index = int(np.argmax(scores))
            truth_index = int(ground_truth[row])
            rows.append([
                row + 1,
                class_names[truth_index],
                class_names[predicted_index],
                "yes" if predicted_index == truth_index else "no",
                np.round(scores, 2),
            ])
        return tabulate(
            rows,
            headers=["Sample", "Ground truth", "Prediction", "Correct", "Scores"],
            tablefmt="grid",
            showindex=False,
        )

'''
    ground_truth_marker = "    def _get_target_probs(self, x: np.ndarray) -> np.ndarray:\n"
    if "def _load_ground_truth" not in predictor_text:
        if ground_truth_marker not in predictor_text:
            raise RuntimeError(f"Unexpected target predictor methods in {predictor_path}")
        predictor_text = predictor_text.replace(
            ground_truth_marker, ground_truth_methods + ground_truth_marker, 1
        )

    write_signature_old = "    def _write_results(self, probs: np.ndarray, class_names) -> Path:\n"
    write_signature_new = (
        "    def _write_results(\n"
        "        self, probs: np.ndarray, class_names, ground_truth: np.ndarray\n"
        "    ) -> Path:\n"
    )
    if write_signature_old in predictor_text:
        predictor_text = predictor_text.replace(
            write_signature_old, write_signature_new, 1
        )

    write_record_old = '''        for row, scores in enumerate(probs.reshape(probs.shape[0], -1)):
            predicted_index = int(np.argmax(scores))
            records.append({
                "row": row,
                "predicted_index": predicted_index,
                "predicted_class": class_names[predicted_index],
                "scores": {
'''
    write_record_new = '''        for row, scores in enumerate(probs.reshape(probs.shape[0], -1)):
            predicted_index = int(np.argmax(scores))
            ground_truth_index = int(ground_truth[row])
            records.append({
                "row": row,
                "ground_truth_index": ground_truth_index,
                "ground_truth_class": class_names[ground_truth_index],
                "predicted_index": predicted_index,
                "predicted_class": class_names[predicted_index],
                "correct": predicted_index == ground_truth_index,
                "scores": {
'''
    if '"ground_truth_class":' not in predictor_text:
        if write_record_old not in predictor_text:
            raise RuntimeError(f"Unexpected prediction JSON records in {predictor_path}")
        predictor_text = predictor_text.replace(write_record_old, write_record_new, 1)

    write_summary_old = '''                "prediction_path": str(self.cfg.dataset.prediction_path),
                "predictions": records,
'''
    write_summary_new = '''                "prediction_path": str(self.cfg.dataset.prediction_path),
                "correct": int(sum(record["correct"] for record in records)),
                "total": len(records),
                "accuracy": float(np.mean([record["correct"] for record in records])),
                "predictions": records,
'''
    if '"accuracy": float(np.mean' not in predictor_text:
        if write_summary_old not in predictor_text:
            raise RuntimeError(f"Unexpected prediction JSON summary in {predictor_path}")
        predictor_text = predictor_text.replace(write_summary_old, write_summary_new, 1)

    predictor_run_old = """        interpreter = self.model
        x = self.dataloaders["predict"]
        class_names = list(self.cfg.dataset.class_names)
        probs = self._get_probs(interpreter, x)
        print(self._format_prediction_table(probs, class_names))
"""
    predictor_run_new = """        x = self.dataloaders["predict"]
        class_names = list(self.cfg.dataset.class_names)
        if self.target == "host":
            probs = self._get_probs(self.model, x)
        elif self.ai_runner is not None:
            probs = self._get_target_probs(x)
        else:
            raise ValueError(f"Unsupported prediction target: {self.target}")
        print(self._format_prediction_table(probs, class_names))
        output_path = self._write_results(probs, class_names)
        print(f"[INFO] : Prediction results saved to {output_path}")
"""
    if "Prediction results saved to" not in predictor_text:
        if predictor_run_old not in predictor_text:
            raise RuntimeError(f"Unexpected prediction entry point in {predictor_path}")
        predictor_text = predictor_text.replace(
            predictor_run_old, predictor_run_new, 1
        )

    predictor_run_ground_truth = """        x = self.dataloaders["predict"]
        class_names = list(self.cfg.dataset.class_names)
        ground_truth = self._load_ground_truth(class_names)
        if len(ground_truth) != len(x):
            raise ValueError(
                f"Prediction samples/ground-truth mismatch: {len(x)} != {len(ground_truth)}"
            )
        if self.target == "host":
            probs = self._get_probs(self.model, x)
        elif self.ai_runner is not None:
            probs = self._get_target_probs(x)
        else:
            raise ValueError(f"Unsupported prediction target: {self.target}")
        print(self._format_ground_truth_table(probs, class_names, ground_truth))
        predicted = np.argmax(probs.reshape(probs.shape[0], -1), axis=1)
        correct = int(np.sum(predicted == ground_truth))
        accuracy = correct / len(ground_truth)
        print(
            f"[INFO] : Prediction accuracy against ground truth: "
            f"{correct}/{len(ground_truth)} ({accuracy:.2%})"
        )
        output_path = self._write_results(probs, class_names, ground_truth)
        print(f"[INFO] : Prediction results saved to {output_path}")
"""
    if "Prediction accuracy against ground truth" not in predictor_text:
        if predictor_run_new not in predictor_text:
            raise RuntimeError(f"Unexpected patched prediction entry point in {predictor_path}")
        predictor_text = predictor_text.replace(
            predictor_run_new, predictor_run_ground_truth, 1
        )

    parse_import_old = (
        "parse_top_level, parse_general_section, parse_training_section, "
        "\\\n                         check_hardware_type, parse_model_section\n"
    )
    parse_import_new = (
        "parse_top_level, parse_general_section, parse_training_section, "
        "parse_prediction_section, \\\n                         check_hardware_type, parse_model_section\n"
    )
    if "parse_training_section, parse_prediction_section" not in parse_config_text:
        if parse_import_old not in parse_config_text:
            raise RuntimeError(f"Unexpected configuration imports in {parse_config_path}")
        parse_config_text = parse_config_text.replace(
            parse_import_old, parse_import_new, 1
        )

    parse_prediction_block = """    # Prediction section parsing
    if cfg.operation_mode in mode_groups.prediction:
        if not cfg.prediction:
            cfg.prediction = DefaultMunch.fromDict({})
        parse_prediction_section(cfg.prediction)

"""
    parse_tools_marker = "    # Tools section parsing\n"
    if "# Prediction section parsing" not in parse_config_text:
        if parse_tools_marker not in parse_config_text:
            raise RuntimeError(f"Unexpected configuration structure in {parse_config_path}")
        parse_config_text = parse_config_text.replace(
            parse_tools_marker, parse_prediction_block + parse_tools_marker, 1
        )

    parse_tools_old = (
        "    if cfg.operation_mode in (mode_groups.benchmarking):\n"
        "        parse_tools_section(cfg.tools," " \n"
        "                            cfg.operation_mode,\n"
        "                            cfg.hardware_type)\n"
    )
    parse_tools_new = """    if (
        cfg.operation_mode in mode_groups.benchmarking
        or (
            cfg.operation_mode in mode_groups.prediction
            and cfg.prediction.target != "host"
        )
    ):
        parse_tools_section(cfg.tools,
                            cfg.operation_mode,
                            cfg.hardware_type)
"""
    if "cfg.prediction.target != \"host\"" not in parse_config_text:
        if parse_tools_old not in parse_config_text:
            raise RuntimeError(f"Unexpected tools parsing in {parse_config_path}")
        parse_config_text = parse_config_text.replace(
            parse_tools_old, parse_tools_new, 1
        )

    main_prediction_import_old = "from common.benchmarking import benchmark, cloud_connect\n"
    main_prediction_import_new = """from common.benchmarking import benchmark, cloud_connect
from common.prediction import gen_load_val_predict
"""
    if "from common.prediction import gen_load_val_predict" not in main_text:
        if main_prediction_import_old not in main_text:
            raise RuntimeError(f"Unexpected main imports in {main_path}")
        main_text = main_text.replace(
            main_prediction_import_old, main_prediction_import_new, 1
        )

    main_prediction_old = """    elif mode == "prediction":
        predictor = get_predictor(cfg=configs,
                                  model=model,
                                  dataloaders=dataloaders)
"""
    main_prediction_new = """    elif mode == "prediction":
        if configs.prediction.target != "host":
            gen_load_val_predict(cfg=configs, model=model)
            os.chdir(SCRIPT_DIR)
        predictor = get_predictor(cfg=configs,
                                  model=model,
                                  dataloaders=dataloaders)
"""
    if "gen_load_val_predict(cfg=configs, model=model)" not in main_text:
        if main_prediction_old not in main_text:
            raise RuntimeError(f"Unexpected prediction mode in {main_path}")
        main_text = main_text.replace(
            main_prediction_old, main_prediction_new, 1
        )

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
    predictor_path.write_text(predictor_text, encoding="utf-8")
    parse_config_path.write_text(parse_config_text, encoding="utf-8")
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
