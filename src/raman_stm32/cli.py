"""Command-line entry point for the complete reproducible pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import ensure_output_dirs, load_config, project_path
from .data import inspect_dataset, load_raw_dataset, prepare_dataset


DEFAULT_CONFIG = "configs/project/pipeline.yaml"


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2))


def _inspect(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    raw_dir = project_path(config, config["paths"]["raw_dir"])
    report = inspect_dataset(load_raw_dataset(raw_dir))
    output = project_path(config, config["paths"]["reports_dir"]) / "dataset_inspection.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Project YAML configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare", "train", "quantize", "evaluate", "verify", "all"):
        subparsers.add_parser(name)
    export_samples = subparsers.add_parser("export-samples")
    export_samples.add_argument(
        "--model-path",
        help="INT8 TFLite model whose input quantization parameters should be used",
    )
    export_samples.add_argument(
        "--output-dir",
        help="Destination directory (default: paths.samples_dir from the project config)",
    )
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input-csv", required=True, help="Headerless CSV with 512 values per row")
    predict.add_argument(
        "--preprocessed",
        action="store_true",
        help="Treat values as already SNV-preprocessed (default: preprocess raw intensities)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "inspect":
        _print_json(_inspect(config))
    elif args.command == "prepare":
        _print_json(prepare_dataset(config))
    elif args.command == "train":
        from .modeling import train_models

        _print_json(train_models(config))
    elif args.command == "quantize":
        from .modeling import quantize_model

        _print_json(quantize_model(config))
    elif args.command == "evaluate":
        from .evaluation import evaluate_models

        _print_json(evaluate_models(config))
    elif args.command == "export-samples":
        from .samples import export_board_samples

        _print_json(
            export_board_samples(
                config,
                model_path=args.model_path,
                samples_dir=args.output_dir,
            )
        )
    elif args.command == "verify":
        from .verification import verify_artifacts

        _print_json(verify_artifacts(config))
    elif args.command == "predict":
        from .evaluation import predict_csv

        _print_json(predict_csv(config, Path(args.input_csv), already_preprocessed=args.preprocessed))
    elif args.command == "all":
        from .evaluation import evaluate_models
        from .modeling import quantize_model, train_models
        from .samples import export_board_samples
        from .verification import verify_artifacts

        _inspect(config)
        prepare_dataset(config)
        train_models(config)
        quantize_model(config)
        result = evaluate_models(config)
        export_board_samples(config)
        verify_artifacts(config)
        _print_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
