"""JSON configuration loading and path resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union


PathLike = Union[str, Path]


class ConfigError(ValueError):
    """Raised when the experiment configuration is incomplete or invalid."""


def load_config(path: PathLike) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError("Configuration file does not exist: {}".format(config_path)) from error
    except json.JSONDecodeError as error:
        raise ConfigError("Invalid JSON in {}: {}".format(config_path, error)) from error
    if not isinstance(value, dict):
        raise ConfigError("The configuration root must be a JSON object")
    for section in ("dataset", "endo3r", "evaluation"):
        if not isinstance(value.get(section), dict):
            raise ConfigError("Missing configuration object: {}".format(section))
    if "output_root" not in value:
        raise ConfigError("Missing configuration value: output_root")
    value["_config_path"] = str(config_path)
    value["_project_root"] = str(config_path.parent.parent.resolve())
    return value


def project_path(value: PathLike, config: Mapping[str, Any]) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not expanded.is_absolute():
        expanded = Path(str(config["_project_root"])) / expanded
    return expanded.resolve()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
