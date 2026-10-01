"""Strict, local JSON storage for application preferences.

The store is deliberately independent from pyml_workbench.config.
Preferences must never change an experiment's canonical configuration or hash.
"""
from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
MAX_PREFERENCE_BYTES = 64 * 1024
MAX_RECENT_FILES = 10
MAX_PATH_LENGTH = 32_767
_SCHEMA_KEYS = frozenset({"schema_version", "log_directory", "recent_files"})


class PreferenceError(ValueError):
    """Raised when a preference file or update does not satisfy its contract."""


@dataclass(frozen=True, slots=True)
class RecentFile:
    """A stored absolute path with existence checked when preferences are read."""

    path: str
    exists: bool


@dataclass(frozen=True, slots=True)
class Preferences:
    """The current application-owned settings, independent of experiment config."""

    log_directory: str | None = None
    recent_files: tuple[RecentFile, ...] = ()


def _path_string(value: str | os.PathLike[str], *, field: str) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise PreferenceError(f"{field} must be a filesystem path") from exc
    if not isinstance(raw, str):
        raise PreferenceError(f"{field} must be a text path")
    if not raw or not raw.strip():
        raise PreferenceError(f"{field} cannot be empty")
    if "\x00" in raw:
        raise PreferenceError(f"{field} cannot contain a NUL character")
    if len(raw) > MAX_PATH_LENGTH:
        raise PreferenceError(f"{field} exceeds the supported path length")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PreferenceError(f"{field} must be valid UTF-8 text") from exc
    return raw


def _absolute_path(value: str | os.PathLike[str], *, field: str) -> str:
    raw = _path_string(value, field=field)
    try:
        expanded = os.path.expanduser(raw)
        normalized = os.path.normpath(os.path.abspath(expanded))
    except (OSError, ValueError) as exc:
        raise PreferenceError(f"{field} is not a usable filesystem path") from exc
    if not os.path.isabs(normalized):
        raise PreferenceError(f"{field} must resolve to an absolute path")
    if len(normalized) > MAX_PATH_LENGTH:
        raise PreferenceError(f"{field} exceeds the supported path length")
    return normalized


def _stored_absolute_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise PreferenceError(f"{field} must be a string path")
    raw = _path_string(value, field=field)
    if not os.path.isabs(raw):
        raise PreferenceError(f"{field} must be an absolute path")
    normalized = os.path.normpath(raw)
    if normalized != raw:
        raise PreferenceError(f"{field} is not normalized")
    return raw


def _path_key(path: str) -> str:
    """Use the platform's path identity rules (case-insensitive on Windows)."""
    return os.path.normcase(path)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreferenceError(f"Duplicate preference JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PreferenceError(f"Non-finite JSON number is not allowed: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PreferenceError("Non-finite JSON numbers are not allowed")
    return parsed


class PreferenceStore:
    """Read and atomically update a versioned app preference JSON file.

    The application chooses an independent settings path and passes it in.
    Construction and import are side-effect free. Call record_recent_file only
    after the GUI has successfully loaded the corresponding dataset; this class
    never opens or reads dataset contents.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(_absolute_path(path, field="preference file"))

    def load(self) -> Preferences:
        """Load preferences, returning empty defaults when the file is absent."""
        log_directory, recent_paths = self._read_state()
        recent_files = tuple(
            RecentFile(path=value, exists=self._is_file(value))
            for value in recent_paths
        )
        return Preferences(log_directory=log_directory, recent_files=recent_files)

    def set_log_directory(
        self, path: str | os.PathLike[str] | None
    ) -> Preferences:
        """Store a normalized log directory path, or clear it with None."""
        log_directory = (
            None if path is None else _absolute_path(path, field="log_directory")
        )
        _current_log_directory, recent_paths = self._read_state()
        self._write_state(log_directory=log_directory, recent_paths=recent_paths)
        return self.load()

    def record_recent_file(self, path: str | os.PathLike[str]) -> Preferences:
        """Remember a successfully opened file, newest first, keeping at most 10.

        This method intentionally does not open the file. The GUI calls it only
        after load_dataset succeeds; a later load reports whether it still exists.
        """
        normalized = _absolute_path(path, field="recent file")
        log_directory, recent_paths = self._read_state()
        key = _path_key(normalized)
        updated = [normalized]
        updated.extend(item for item in recent_paths if _path_key(item) != key)
        self._write_state(
            log_directory=log_directory,
            recent_paths=updated[:MAX_RECENT_FILES],
        )
        return self.load()

    def clear_recent_files(self) -> Preferences:
        """Clear only the remembered paths; source files are never removed."""
        log_directory, _recent_paths = self._read_state()
        self._write_state(log_directory=log_directory, recent_paths=[])
        return self.load()

    def _read_state(self) -> tuple[str | None, list[str]]:
        try:
            file_stat = self.path.stat()
        except FileNotFoundError:
            return None, []
        except OSError as exc:
            raise PreferenceError(f"Could not inspect preference file: {exc}") from exc
        if not stat.S_ISREG(file_stat.st_mode):
            raise PreferenceError("Preference path must refer to a regular file")
        try:
            with self.path.open("rb") as stream:
                raw = stream.read(MAX_PREFERENCE_BYTES + 1)
        except OSError as exc:
            raise PreferenceError(f"Could not read preference file: {exc}") from exc
        if len(raw) > MAX_PREFERENCE_BYTES:
            raise PreferenceError(
                f"Preference file exceeds the {MAX_PREFERENCE_BYTES}-byte limit"
            )
        try:
            text = raw.decode("utf-8")
            payload = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except PreferenceError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise PreferenceError(
                f"Preference file is not valid UTF-8 JSON: {exc}"
            ) from exc
        return self._validate_payload(payload)

    @staticmethod
    def _validate_payload(payload: Any) -> tuple[str | None, list[str]]:
        if not isinstance(payload, dict):
            raise PreferenceError("Preference JSON must be an object")
        if frozenset(payload) != _SCHEMA_KEYS:
            unknown = sorted(set(payload).difference(_SCHEMA_KEYS))
            missing = sorted(_SCHEMA_KEYS.difference(payload))
            details = []
            if unknown:
                details.append(f"unknown fields: {unknown}")
            if missing:
                details.append(f"missing fields: {missing}")
            raise PreferenceError(
                "Invalid preference schema (" + "; ".join(details) + ")"
            )
        version = payload["schema_version"]
        if type(version) is not int or version != SCHEMA_VERSION:
            raise PreferenceError(f"Unsupported preference schema version: {version!r}")

        log_directory_value = payload["log_directory"]
        if log_directory_value is None:
            log_directory = None
        else:
            log_directory = _stored_absolute_path(
                log_directory_value, field="log_directory"
            )

        recent_value = payload["recent_files"]
        if not isinstance(recent_value, list):
            raise PreferenceError("recent_files must be a JSON array")
        if len(recent_value) > MAX_RECENT_FILES:
            raise PreferenceError(
                f"recent_files may contain at most {MAX_RECENT_FILES} paths"
            )
        recent_paths: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(recent_value):
            path = _stored_absolute_path(item, field=f"recent_files[{index}]")
            key = _path_key(path)
            if key in seen:
                raise PreferenceError("recent_files contains duplicate paths")
            seen.add(key)
            recent_paths.append(path)
        return log_directory, recent_paths

    @staticmethod
    def _is_file(path: str) -> bool:
        try:
            return Path(path).is_file()
        except OSError:
            return False

    def _write_state(
        self, *, log_directory: str | None, recent_paths: list[str]
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "log_directory": log_directory,
            "recent_files": recent_paths,
        }
        try:
            encoded = (
                json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise PreferenceError(
                f"Preferences are not valid finite JSON: {exc}"
            ) from exc
        if len(encoded) > MAX_PREFERENCE_BYTES:
            raise PreferenceError(
                f"Preference update exceeds the {MAX_PREFERENCE_BYTES}-byte limit"
            )

        temporary_path: str | None = None
        descriptor: int | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        except OSError as exc:
            raise PreferenceError(f"Could not write preference file: {exc}") from exc
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass


__all__ = [
    "MAX_PREFERENCE_BYTES",
    "MAX_RECENT_FILES",
    "PreferenceError",
    "PreferenceStore",
    "Preferences",
    "RecentFile",
    "SCHEMA_VERSION",
]
