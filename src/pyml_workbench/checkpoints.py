"""Atomic, integrity-checked JSON checkpoints for local application state."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be safely read or written."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"Duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CheckpointError(f"Non-finite checkpoint number is not allowed: {value}")


def _canonical_payload(scope: str, payload: Mapping[str, Any]) -> bytes:
    if not isinstance(scope, str) or not scope.strip() or len(scope) > 80:
        raise CheckpointError("Checkpoint scope must be short non-empty text")
    if not isinstance(payload, Mapping):
        raise CheckpointError("Checkpoint payload must be a JSON object")
    try:
        encoded = json.dumps(
            {"scope": scope, "payload": dict(payload)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CheckpointError(f"Checkpoint payload is not finite UTF-8 JSON: {exc}") from exc
    return encoded


def save_checkpoint(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    scope: str,
) -> None:
    """Atomically save a scoped JSON object with a SHA-256 integrity digest."""
    target = Path(path).expanduser().absolute()
    canonical = _canonical_payload(scope, payload)
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "payload": dict(payload),
        "payload_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    try:
        encoded = (json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise CheckpointError(f"Checkpoint could not be encoded: {exc}") from exc
    if len(encoded) > MAX_CHECKPOINT_BYTES:
        raise CheckpointError(f"Checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes")

    temporary: str | None = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise CheckpointError(f"Could not write checkpoint {target}: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    scope: str,
) -> dict[str, Any] | None:
    """Load a checkpoint, returning ``None`` only when the file is absent."""
    target = Path(path).expanduser().absolute()
    try:
        info = target.stat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CheckpointError(f"Could not inspect checkpoint {target}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointError("Checkpoint path must refer to a regular file")
    if info.st_size > MAX_CHECKPOINT_BYTES:
        raise CheckpointError(f"Checkpoint exceeds {MAX_CHECKPOINT_BYTES} bytes")
    try:
        with target.open("rb") as stream:
            raw = stream.read(MAX_CHECKPOINT_BYTES + 1)
        envelope = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CheckpointError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CheckpointError(f"Checkpoint is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(envelope, dict) or set(envelope) != {
        "schema_version", "scope", "payload", "payload_sha256"
    }:
        raise CheckpointError("Checkpoint envelope has an unsupported shape")
    if type(envelope["schema_version"]) is not int or envelope["schema_version"] != SCHEMA_VERSION:
        raise CheckpointError(f"Unsupported checkpoint schema: {envelope['schema_version']!r}")
    if envelope["scope"] != scope:
        raise CheckpointError("Checkpoint scope does not match the requested state")
    payload = envelope["payload"]
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a JSON object")
    digest = envelope["payload_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise CheckpointError("Checkpoint digest is malformed")
    expected = hashlib.sha256(_canonical_payload(scope, payload)).hexdigest()
    if not hmac.compare_digest(digest, expected):
        raise CheckpointError("Checkpoint integrity digest does not match")
    return payload
