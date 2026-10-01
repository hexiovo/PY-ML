"""Integrity-checked persistence for snapshots owned by durable jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping
from typing import Any

import joblib

from .config import ExperimentConfig
from .experiment import DatasetSnapshot, _json_digest


_RECEIPT_SCHEMA_VERSION = 1
_RECEIPT_KIND = "pyml_workbench.owned_dataset_snapshot"
_RECEIPT_FIELDS = {
    "schema_version",
    "kind",
    "relative_path",
    "file_sha256",
    "manifest_sha256",
    "manifest",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def save_owned_snapshot(snapshot: DatasetSnapshot, path: str | Path) -> dict[str, Any]:
    """Atomically persist a validated snapshot and return its JSON receipt.

    The path is supplied by the owning job; this helper never reads or writes
    ``snapshot.source_path``.
    """
    if not isinstance(snapshot, DatasetSnapshot):
        raise ValueError("Owned snapshot must be a DatasetSnapshot")

    snapshot.verify()
    manifest = snapshot.to_dict()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(snapshot, temporary, compress=3)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)

    return {
        "schema_version": _RECEIPT_SCHEMA_VERSION,
        "kind": _RECEIPT_KIND,
        "relative_path": target.name,
        "file_sha256": _file_sha256(target),
        "manifest_sha256": _json_digest(manifest),
        "manifest": manifest,
    }


def load_owned_snapshot(
    path: str | Path,
    receipt: Mapping[str, object],
    *,
    expected_manifest: Mapping[str, object] | None = None,
    config: ExperimentConfig | None = None,
) -> DatasetSnapshot:
    """Load an owned snapshot only when its file and metadata match receipts."""
    target = Path(path).expanduser().resolve()
    receipt_manifest = _validate_receipt(receipt, target)

    try:
        observed_file_sha256 = _file_sha256(target)
    except FileNotFoundError as exc:
        raise ValueError(f"Owned snapshot file is missing: {target}") from exc
    if observed_file_sha256 != receipt["file_sha256"]:
        raise ValueError("Owned snapshot file checksum does not match its receipt")

    try:
        snapshot = joblib.load(target)
    except Exception as exc:
        raise ValueError(f"Owned snapshot file could not be deserialized: {target}") from exc
    if not isinstance(snapshot, DatasetSnapshot):
        raise ValueError("Owned snapshot artifact has an unexpected type")

    # Reuse DatasetSnapshot's data/split/config identity checks without looking
    # at its source path; that source may no longer exist after enqueue.
    snapshot.verify(config)
    manifest = snapshot.to_dict()
    if manifest != receipt_manifest:
        raise ValueError("Owned snapshot metadata does not match its receipt")
    if expected_manifest is not None:
        expected = _json_object(expected_manifest, "expected manifest")
        if manifest != expected:
            raise ValueError("Owned snapshot metadata does not match the expected manifest")
    return snapshot


def _validate_receipt(receipt: Mapping[str, object], target: Path) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("Owned snapshot receipt must be a JSON object")
    if set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("Owned snapshot receipt has an invalid shape")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != _RECEIPT_SCHEMA_VERSION:
        raise ValueError("Unsupported owned snapshot receipt schema version")
    if receipt["kind"] != _RECEIPT_KIND:
        raise ValueError("Owned snapshot receipt has an unexpected kind")

    relative_path = receipt["relative_path"]
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path in {".", ".."}
        or "/" in relative_path
        or "\\" in relative_path
        or relative_path != target.name
    ):
        raise ValueError("Owned snapshot receipt path does not match the artifact filename")

    for name in ("file_sha256", "manifest_sha256"):
        digest = receipt[name]
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"Owned snapshot receipt has an invalid {name}")

    manifest = _json_object(receipt["manifest"], "receipt manifest")
    if _json_digest(manifest) != receipt["manifest_sha256"]:
        raise ValueError("Owned snapshot manifest checksum does not match its receipt")
    return manifest


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Owned snapshot {label} must be a JSON object")
    try:
        result = json.loads(
            json.dumps(dict(value), ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Owned snapshot {label} is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Owned snapshot {label} must be a JSON object")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
