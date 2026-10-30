"""Inspect optional external plugins without importing or executing their code."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping

MANIFEST_NAME = "pyml-plugin.json"
MAX_JSON_BYTES = 64 * 1024
PLUGIN_SCHEMA_VERSION = 2
LEGACY_PLUGIN_SCHEMA_VERSION = 1
HYPEREEG_SOURCE = "HyperEEG"
PYFNIRS_SOURCE = "PYfNIRs"


class PluginError(ValueError):
    """An invalid plugin manifest or local plugin setting."""


def _read_json(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise PluginError("插件配置文件过大。")
        payload = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise PluginError(f"无法读取插件配置：{exc}") from exc
    if not isinstance(payload, dict):
        raise PluginError("插件配置必须是 JSON 对象。")
    return payload


def _absolute_setting(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PluginError("插件设置中的路径无效。")
    try:
        value.encode("utf-8")
        return str(Path(value).expanduser().resolve())
    except (OSError, ValueError, UnicodeError) as exc:
        raise PluginError(f"插件路径无效：{exc}") from exc


@dataclass(frozen=True)
class PluginSettings:
    directory: str | None = None
    python_executable: str | None = None


class PluginSettingsStore:
    """Store plugin settings under stable source keys and migrate v1 files."""

    def __init__(
        self,
        path: Path,
        default_settings: PluginSettings | None = None,
        *,
        legacy_paths: Mapping[str, Path] | None = None,
    ):
        self.path = Path(path)
        self.default_settings = default_settings or PluginSettings()
        self.legacy_paths = {
            self._source_key(source): Path(legacy_path)
            for source, legacy_path in (legacy_paths or {}).items()
        }

    @staticmethod
    def _source_key(source_project: str) -> str:
        if (
            not isinstance(source_project, str)
            or not source_project.strip()
            or "\x00" in source_project
        ):
            raise PluginError("插件来源标识无效。")
        return source_project.strip()

    @staticmethod
    def _normalise_settings(settings: PluginSettings) -> PluginSettings:
        if not isinstance(settings, PluginSettings):
            raise PluginError("插件设置格式无效。")
        return PluginSettings(
            _absolute_setting(settings.directory),
            _absolute_setting(settings.python_executable),
        )

    @staticmethod
    def _decode_settings(value: object) -> PluginSettings:
        if not isinstance(value, dict) or set(value) != {
            "directory", "python_executable"
        }:
            raise PluginError("插件来源记录的字段不受支持。")
        return PluginSettings(
            _absolute_setting(value["directory"]),
            _absolute_setting(value["python_executable"]),
        )

    @classmethod
    def _decode_legacy_settings(cls, payload: dict) -> PluginSettings:
        if (
            set(payload) != {"schema_version", "directory", "python_executable"}
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != LEGACY_PLUGIN_SCHEMA_VERSION
        ):
            raise PluginError("旧版插件设置版本或字段不受支持。")
        return cls._decode_settings(
            {
                "directory": payload["directory"],
                "python_executable": payload["python_executable"],
            }
        )

    def _decode_registry(self, payload: dict) -> dict[str, PluginSettings]:
        if (
            set(payload) != {"schema_version", "plugins"}
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != PLUGIN_SCHEMA_VERSION
            or not isinstance(payload["plugins"], dict)
        ):
            raise PluginError("插件注册表版本或字段不受支持。")
        registry: dict[str, PluginSettings] = {}
        for source, raw_settings in payload["plugins"].items():
            key = self._source_key(source)
            registry[key] = self._decode_settings(raw_settings)
        return registry

    def _read_legacy_file(self, path: Path) -> PluginSettings:
        return self._decode_legacy_settings(_read_json(path))

    def _load_legacy_registry(self, primary_payload: dict | None = None) -> dict[str, PluginSettings]:
        registry: dict[str, PluginSettings] = {}
        if primary_payload is not None:
            # The original plugins.json represented the generic HyperEEG entry.
            registry[HYPEREEG_SOURCE] = self._decode_legacy_settings(primary_payload)
        for source, path in self.legacy_paths.items():
            if not path.exists() or path.resolve() == self.path.resolve():
                continue
            registry.setdefault(source, self._read_legacy_file(path))
        return registry

    def load_all(self) -> dict[str, PluginSettings]:
        """Return all configured source entries, migrating old files once."""
        if self.path.exists():
            payload = _read_json(self.path)
            if payload.get("schema_version") == PLUGIN_SCHEMA_VERSION:
                return self._decode_registry(payload)
            if (
                type(payload.get("schema_version")) is int
                and payload["schema_version"] == LEGACY_PLUGIN_SCHEMA_VERSION
            ):
                registry = self._load_legacy_registry(payload)
                self._write_registry(registry)
                return registry
            raise PluginError("插件注册表版本或字段不受支持。")

        registry = self._load_legacy_registry()
        if registry:
            self._write_registry(registry)
        return registry

    def load(self, source_project: str = HYPEREEG_SOURCE) -> PluginSettings:
        """Load one source entry, returning an empty setting when it is absent."""
        key = self._source_key(source_project)
        return self.load_all().get(key, self.default_settings)

    def save(
        self,
        settings: PluginSettings,
        source_project: str = HYPEREEG_SOURCE,
    ) -> PluginSettings:
        """Update one source entry without replacing the other plugin settings."""
        key = self._source_key(source_project)
        normalised = self._normalise_settings(settings)
        self.update_sources({key: normalised})
        return normalised

    def remove(self, source_project: str) -> None:
        self.update_sources({self._source_key(source_project): None})

    def update_sources(
        self,
        updates: Mapping[str, PluginSettings | None],
    ) -> dict[str, PluginSettings]:
        """Merge source updates and persist the complete registry atomically."""
        registry = self.load_all()
        for source, settings in updates.items():
            key = self._source_key(source)
            if settings is None:
                registry.pop(key, None)
                continue
            normalised = self._normalise_settings(settings)
            if not normalised.directory and not normalised.python_executable:
                registry.pop(key, None)
            else:
                registry[key] = normalised
        self._write_registry(registry)
        return registry

    def _write_registry(self, registry: Mapping[str, PluginSettings]) -> None:
        plugins: dict[str, dict[str, str | None]] = {}
        for source, settings in registry.items():
            key = self._source_key(source)
            normalised = self._normalise_settings(settings)
            plugins[key] = {
                "directory": normalised.directory,
                "python_executable": normalised.python_executable,
            }
        payload = {"schema_version": PLUGIN_SCHEMA_VERSION, "plugins": plugins}
        raw = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        if len(raw) > MAX_JSON_BYTES:
            raise PluginError("插件注册表过大。")
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise PluginError(f"无法保存插件设置：{exc}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class DetectedPlugin:
    directory: Path
    name: str
    version: str
    program: str
    arguments: tuple[str, ...]


def discover_bundled_conversion(source_project: str) -> PluginSettings:
    """Find a valid sibling support release for optional legacy callers."""
    folder = {HYPEREEG_SOURCE: "hypereeg", PYFNIRS_SOURCE: "pyfnirs"}[source_project]
    anchors = (Path(sys.executable).resolve().parent, Path(__file__).resolve().parent)
    for anchor in anchors:
        for parent in (anchor, *anchor.parents):
            candidate = parent / "PY-ML-support" / folder
            settings = PluginSettings(str(candidate))
            try:
                _detect_data_conversion(
                    settings,
                    required_source_project=source_project,
                    allow_legacy_unmarked=source_project == HYPEREEG_SOURCE,
                )
            except PluginError:
                continue
            return settings
    return PluginSettings()


def _plugin_file(directory: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PluginError("插件启动文件路径无效。")
    relative = Path(value)
    if relative.is_absolute() or relative.drive:
        raise PluginError("插件启动文件必须使用插件目录内的相对路径。")
    resolved = (directory / relative).resolve()
    if not resolved.is_relative_to(directory) or not resolved.is_file():
        raise PluginError("插件启动文件缺失，或位于插件目录之外。")
    return resolved


def detect_data_conversion(settings: PluginSettings) -> DetectedPlugin:
    """Detect a generic data-conversion plugin without running its code."""
    return _detect_data_conversion(settings)


def detect_hypereeg_data_conversion(settings: PluginSettings) -> DetectedPlugin:
    """Detect HyperEEG releases, including older manifests without a source key."""
    return _detect_data_conversion(
        settings,
        required_source_project=HYPEREEG_SOURCE,
        allow_legacy_unmarked=True,
    )


def detect_pyfnirs_data_conversion(settings: PluginSettings) -> DetectedPlugin:
    """Detect only the PYfNIRs source-marked data-conversion release."""
    return _detect_data_conversion(settings, required_source_project=PYFNIRS_SOURCE)


def _detect_data_conversion(
    settings: PluginSettings,
    *,
    required_source_project: str | None = None,
    allow_legacy_unmarked: bool = False,
) -> DetectedPlugin:
    """Validate metadata and launch files only; no plugin code runs during detection."""
    if not settings.directory:
        if required_source_project == PYFNIRS_SOURCE:
            raise PluginError(
                "未配置 PYfNIRs 插件目录，请在“插件 → 设置…”中选择目录。"
            )
        if required_source_project == HYPEREEG_SOURCE:
            raise PluginError(
                "未配置 HyperEEG 插件目录，请在“插件 → 设置…”中选择目录。"
            )
        raise PluginError("未配置插件目录，请在“插件 → 设置…”中选择目录。")
    try:
        directory = Path(settings.directory).expanduser().resolve()
        if not directory.is_dir():
            raise PluginError("插件目录不存在。")
        manifest = _read_json(directory / MANIFEST_NAME)
        if (type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != 1
                or manifest.get("id") != "data-conversion"):
            raise PluginError("未检测到兼容的数据转换插件。")
        if required_source_project is not None:
            source = manifest.get("source_project")
            matches_legacy_hypereeg = (
                required_source_project == HYPEREEG_SOURCE
                and allow_legacy_unmarked
                and source is None
            )
            if source != required_source_project and not matches_legacy_hypereeg:
                raise PluginError(f"插件来源标识不匹配；需要 {required_source_project}。")
        name, version = manifest.get("name"), manifest.get("version")
        if not all(isinstance(item, str) and item.strip() for item in (name, version)):
            raise PluginError("插件名称或版本缺失。")
        entry = manifest.get("entrypoint")
        if not isinstance(entry, dict):
            raise PluginError("插件缺少启动入口。")
        path = _plugin_file(directory, entry.get("path"))
        if entry.get("kind") == "executable":
            if path.suffix.lower() != ".exe":
                raise PluginError("独立插件入口必须是 EXE 文件。")
            program, arguments = str(path), ()
        elif entry.get("kind") == "python":
            if path.suffix.lower() != ".py":
                raise PluginError("Python 插件入口必须是 .py 文件。")
            interpreter = settings.python_executable
            if not interpreter and not getattr(sys, "frozen", False):
                interpreter = sys.executable
            if not interpreter:
                raise PluginError("此插件需要 Python；请在设置中选择插件环境的 Python 程序。")
            interpreter_path = Path(interpreter).expanduser().resolve()
            if not interpreter_path.is_file():
                raise PluginError("设置的 Python 程序不存在。")
            if os.name == "nt" and interpreter_path.name.casefold() not in {"python.exe", "pythonw.exe"}:
                raise PluginError("请选择插件环境的 python.exe 或 pythonw.exe。")
            program = str(interpreter_path)
            arguments = ("-X", "utf8", str(path))
        else:
            raise PluginError("插件启动方式不受支持。")
        return DetectedPlugin(directory, name, version, program, arguments)
    except PluginError:
        raise
    except (OSError, ValueError) as exc:
        raise PluginError(f"无法检测插件：{exc}") from exc
