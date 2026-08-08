#!/usr/bin/env python3
"""Install the Linear Softmax model into Model Zoo Services v4.1.1."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


EXPECTED_REVISION = "0f6210ed5156126b782e1c43249063a477484b20"
EXPECTED_FILES = (
    "arc_fault_detection/tf/src/models/__init__.py",
    "arc_fault_detection/tf/wrappers/models/custom_models/models.py",
    "arc_fault_detection/stm32ai_main.py",
    "arc_fault_detection/tf/src/quantization/tflite_quantizer.py",
    "arc_fault_detection/tf/src/evaluation/tflite_evaluator.py",
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

    init_text = init_path.read_text(encoding="utf-8")
    registry_text = registry_path.read_text(encoding="utf-8")
    quantizer_text = quantizer_path.read_text(encoding="utf-8")
    evaluator_text = evaluator_path.read_text(encoding="utf-8")

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

    shutil.copy2(model_source, model_target)
    init_path.write_text(init_text, encoding="utf-8")
    registry_path.write_text(registry_text, encoding="utf-8")
    quantizer_path.write_text(quantizer_text, encoding="utf-8")
    evaluator_path.write_text(evaluator_text, encoding="utf-8")
    print(f"Installed and registered raman_linear_softmax in {model_zoo_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-zoo-dir", required=True, type=Path)
    args = parser.parse_args()
    install(Path(__file__).resolve().parents[1], args.model_zoo_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
