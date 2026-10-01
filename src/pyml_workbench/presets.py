"""Strict, atomic JSON persistence for ``ExperimentConfig`` presets.

This helper module adds only standard-library imports; its save/load functions
resolve ``ExperimentConfig`` lazily. Importing it through the package still
runs the package's existing core API initialization. Presets contain
configuration and path references only; they never read or copy the referenced
dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ExperimentConfig


PRESET_SCHEMA_VERSION = 1
MAX_PRESET_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WRAPPER_REQUIRED = {"schema", "config", "sha256"}
_WRAPPER_ALLOWED = _WRAPPER_REQUIRED | {"name"}
_CONFIG_REQUIRED = {"dataset", "task", "model_id", "parameters", "split", "output_dir"}
_CONFIG_ALLOWED = _CONFIG_REQUIRED | {"sequence"}
_DATASET_FIELDS = {"source_path", "sheet_name", "target_column", "feature_columns"}
_SPLIT_FIELDS = {"seed", "train_fraction", "validation_fraction", "test_fraction"}


class PresetError(ValueError):
    """Raised when a preset is invalid, unsafe, or cannot be written."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PresetError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PresetError(f"non-finite JSON number is not allowed: {value}")


def _validate_json_tree(value: Any, where: str = "config") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PresetError(f"{where} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PresetError(f"{where} contains a non-string object key")
            _validate_json_tree(item, f"{where}.{key}")
        return
    raise PresetError(f"{where} contains an unsupported value type")


def _canonical_json(value: Any) -> bytes:
    _validate_json_tree(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PresetError(f"configuration cannot be encoded as strict JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _config_type() -> type[Any]:
    # Lazy import keeps importing this persistence helper independent of the
    # experiment, GUI, and optional model stack.
    from .config import ExperimentConfig

    return ExperimentConfig


def _validate_config_shape(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PresetError("preset config must be a JSON object")
    keys = set(payload)
    if not _CONFIG_REQUIRED.issubset(keys) or keys - _CONFIG_ALLOWED:
        raise PresetError("preset config has missing or unknown fields")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or set(dataset) != _DATASET_FIELDS:
        raise PresetError("preset dataset fields are incomplete or unknown")
    if not isinstance(dataset.get("source_path"), str):
        raise PresetError("dataset.source_path must be a string path")
    for key in ("sheet_name", "target_column"):
        if dataset.get(key) is not None and not isinstance(dataset.get(key), str):
            raise PresetError(f"dataset.{key} must be a string or null")
    features = dataset.get("feature_columns")
    if features is not None and (
        not isinstance(features, list) or any(not isinstance(item, str) for item in features)
    ):
        raise PresetError("dataset.feature_columns must be a string list or null")

    if not isinstance(payload.get("task"), str) or not isinstance(payload.get("model_id"), str):
        raise PresetError("task and model_id must be strings")
    if not isinstance(payload.get("parameters"), dict):
        raise PresetError("parameters must be a JSON object")
    split = payload.get("split")
    if not isinstance(split, dict) or set(split) != _SPLIT_FIELDS:
        raise PresetError("split fields are incomplete or unknown")
    seed = split.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PresetError("split.seed must be an integer")
    for key in ("train_fraction", "validation_fraction", "test_fraction"):
        value = split.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PresetError(f"split.{key} must be a finite number")
    if payload.get("output_dir") is not None and not isinstance(payload.get("output_dir"), str):
        raise PresetError("output_dir must be a string path or null")
    if "sequence" in payload and payload["sequence"] is not None and not isinstance(payload["sequence"], dict):
        raise PresetError("sequence must be a JSON object or null")
    _validate_json_tree(payload)
    return payload


def _same_path(left: Path, right: Path) -> bool:
    try:
        left_key = os.path.normcase(os.path.abspath(os.fspath(left.resolve(strict=False))))
        right_key = os.path.normcase(os.path.abspath(os.fspath(right.resolve(strict=False))))
    except (OSError, RuntimeError):
        left_key = os.path.normcase(os.path.abspath(os.fspath(left)))
        right_key = os.path.normcase(os.path.abspath(os.fspath(right)))
    return left_key == right_key


def _is_within(candidate: Path, directory: Path) -> bool:
    try:
        candidate_key = os.path.normcase(os.path.abspath(os.fspath(candidate.resolve(strict=False))))
        directory_key = os.path.normcase(os.path.abspath(os.fspath(directory.resolve(strict=False))))
        return os.path.commonpath([candidate_key, directory_key]) == directory_key
    except (OSError, RuntimeError, ValueError):
        return False


def _reject_source_or_output_conflict(target: Path, config: Any) -> None:
    source = Path(config.dataset.source_path).expanduser()
    if _same_path(target, source):
        raise PresetError("preset destination cannot overwrite the configured dataset source")
    if config.output_dir:
        output_dir = Path(config.output_dir).expanduser()
        if _is_within(target, output_dir):
            raise PresetError("preset destination cannot be inside the experiment output directory")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, data: bytes) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise PresetError("preset destination directory does not exist")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise PresetError("preset destination must be a regular file")
    if not path.exists() and os.path.lexists(path):
        raise PresetError("preset destination cannot be a symlink or reparse point")
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
        temp_path = Path(temp_name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists() and (path.is_symlink() or not path.is_file()):
            raise PresetError("preset destination changed during the atomic write")
        os.replace(temp_path, path)
        _fsync_directory(parent)
    except PresetError:
        raise
    except OSError as exc:
        raise PresetError(f"could not atomically write preset: {exc}") from exc
    finally:
        if temp_path is not None and os.path.lexists(temp_path):
            try:
                if not temp_path.is_symlink() and temp_path.is_file():
                    temp_path.unlink()
            except OSError:
                pass


def save_preset(
    path: os.PathLike[str] | str,
    config: "ExperimentConfig",
    *,
    name: str | None = None,
) -> str:
    """Atomically save a strict JSON wrapper and return its config SHA-256."""
    config_type = _config_type()
    if not isinstance(config, config_type):
        raise PresetError("config must be an ExperimentConfig")
    try:
        config.validate()
        canonical_config = config.to_dict()
    except Exception as exc:
        raise PresetError(f"experiment configuration is invalid: {exc}") from exc
    _validate_config_shape(canonical_config)
    digest = _sha256(canonical_config)
    wrapper: dict[str, Any] = {
        "schema": PRESET_SCHEMA_VERSION,
        "config": canonical_config,
        "sha256": digest,
    }
    if name is not None:
        if not isinstance(name, str) or not name.strip() or len(name) > 128 or any(ord(ch) < 32 for ch in name):
            raise PresetError("preset name must be a non-empty printable string of at most 128 characters")
        wrapper["name"] = name.strip()
    encoded = json.dumps(
        wrapper,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    if len(encoded) > MAX_PRESET_BYTES:
        raise PresetError("preset exceeds the 1 MiB size limit")
    target = Path(path).expanduser()
    _reject_source_or_output_conflict(target, config)
    _atomic_write(target, encoded)
    return digest


def _decode_wrapper(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except PresetError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise PresetError(f"preset is not valid strict UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PresetError("preset wrapper must be a JSON object")
    keys = set(payload)
    if not _WRAPPER_REQUIRED.issubset(keys) or keys - _WRAPPER_ALLOWED:
        raise PresetError("preset wrapper has missing or unknown fields")
    if type(payload.get("schema")) is not int or payload["schema"] != PRESET_SCHEMA_VERSION:
        raise PresetError("unsupported preset schema")
    digest = payload.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise PresetError("preset SHA-256 field is invalid")
    if "name" in payload and (
        not isinstance(payload["name"], str)
        or not payload["name"].strip()
        or len(payload["name"]) > 128
        or any(ord(ch) < 32 for ch in payload["name"])
    ):
        raise PresetError("preset name is invalid")
    config_payload = _validate_config_shape(payload.get("config"))
    if _sha256(config_payload) != digest:
        raise PresetError("preset SHA-256 does not match its config")
    return payload


def load_preset(path: os.PathLike[str] | str) -> "ExperimentConfig":
    """Load and validate a preset without reading data or running an experiment."""
    target = Path(path).expanduser()
    try:
        if target.is_symlink() or (os.path.lexists(target) and not target.is_file()):
            raise PresetError("preset source must be a regular file")
        size = target.stat().st_size
        if size > MAX_PRESET_BYTES:
            raise PresetError("preset exceeds the 1 MiB size limit")
        with target.open("rb") as stream:
            raw = stream.read(MAX_PRESET_BYTES + 1)
    except PresetError:
        raise
    except OSError as exc:
        raise PresetError(f"could not read preset: {exc}") from exc
    if len(raw) > MAX_PRESET_BYTES:
        raise PresetError("preset exceeds the 1 MiB size limit")
    wrapper = _decode_wrapper(raw)
    config_type = _config_type()
    try:
        config = config_type.from_dict(wrapper["config"])
        config.validate()
        roundtrip = config.to_dict()
    except Exception as exc:
        raise PresetError(f"preset config failed ExperimentConfig validation: {exc}") from exc
    if _canonical_json(roundtrip) != _canonical_json(wrapper["config"]):
        raise PresetError("preset config does not match the canonical ExperimentConfig representation")
    return config


__all__ = [
    "MAX_PRESET_BYTES",
    "PRESET_SCHEMA_VERSION",
    "PresetError",
    "load_preset",
    "save_preset",
]
