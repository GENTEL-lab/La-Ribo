from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


_PATTERN = re.compile(r"\$\{([^}]+)\}")
_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _get_by_path(root: Any, dotted_path: str) -> Any:
    current = root
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(f"Cannot resolve path: {dotted_path}")
    return current


def _resolve_interpolations(obj: Any, root: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _resolve_interpolations(value, root) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_resolve_interpolations(value, root) for value in obj]
    if not isinstance(obj, str):
        return obj

    match = _PATTERN.fullmatch(obj)
    if match:
        return _resolve_interpolations(_get_by_path(root, match.group(1)), root)

    def replacer(found: re.Match[str]) -> str:
        resolved = _resolve_interpolations(_get_by_path(root, found.group(1)), root)
        return str(resolved)

    return _PATTERN.sub(replacer, obj)


def _coerce_scalar_strings(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _coerce_scalar_strings(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_coerce_scalar_strings(value) for value in obj]
    if isinstance(obj, str) and _NUMBER_PATTERN.fullmatch(obj):
        return int(obj) if _INT_PATTERN.fullmatch(obj) else float(obj)
    return obj


def to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{key: to_namespace(value) for key, value in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(value) for value in obj]
    return obj


def namespace_to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {key: namespace_to_dict(value) for key, value in vars(obj).items()}
    if isinstance(obj, dict):
        return {key: namespace_to_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [namespace_to_dict(value) for value in obj]
    if isinstance(obj, tuple):
        return [namespace_to_dict(value) for value in obj]
    return obj


def load_yaml_config(path: str | Path) -> SimpleNamespace:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    data = _resolve_interpolations(data, data)
    data = _coerce_scalar_strings(data)
    return to_namespace(data)
