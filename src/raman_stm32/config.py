"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parents[2])
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ("processed_dir", "models_dir", "reports_dir", "samples_dir"):
        project_path(config, config["paths"][key]).mkdir(parents=True, exist_ok=True)

