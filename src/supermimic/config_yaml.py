"""Translate a plain YAML training preset into Tyro CLI arguments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


def _flatten_config(mapping: dict[str, Any], prefix: str = "") -> list[str]:
    args: list[str] = []
    for key, value in mapping.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError(f"invalid YAML config key: {key!r}")
        name = f"{prefix}.{key.replace('_', '-')}" if prefix else key.replace("_", "-")
        if isinstance(value, dict):
            args.extend(_flatten_config(value, name))
        elif isinstance(value, bool):
            args.append(f"--{name}" if value else f"--{prefix + '.' if prefix else ''}no-{key.replace('_', '-')}")
        elif isinstance(value, list):
            if not value or any(isinstance(item, (dict, list, bool)) for item in value):
                raise ValueError(f"YAML config {name!r} needs a nonempty list of scalar values")
            args.extend([f"--{name}", *("None" if item is None else str(item) for item in value)])
        elif value is None or isinstance(value, (str, int, float)):
            args.extend([f"--{name}", "None" if value is None else str(value)])
        else:
            raise ValueError(f"unsupported YAML value for {name!r}: {value!r}")
    return args


def expand_config_args(args: Sequence[str]) -> list[str]:
    """Expand --config first, then pass explicit CLI options through as overrides."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parsed, remaining = parser.parse_known_args(args)
    if parsed.config is None:
        return remaining
    with parsed.config.open(encoding="utf-8") as stream:
        mapping = yaml.safe_load(stream)
    if not isinstance(mapping, dict):
        raise TypeError(f"YAML config must be a mapping: {parsed.config}")
    return _flatten_config(mapping) + remaining
