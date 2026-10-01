"""JSON-safe summaries for owned sequence plans and their split coverage."""
from __future__ import annotations

from typing import Any, Mapping


def sequence_plan_summary(plan: Any) -> dict[str, Any] | None:
    """Summarize actual source rows, groups, and windows in each plan split."""
    if plan is None:
        return None
    counts: dict[str, dict[str, int]] = {}
    for name in ("train", "validation", "test"):
        partition = getattr(plan, name)
        targets = getattr(partition, "window_target_positions", None)
        counts[name] = {
            "source_rows": int(len(partition.row_positions)),
            "groups": int(len(partition.group_ids)),
            "windows": 0 if targets is None else int(len(targets)),
        }
    return _with_proportions(counts, plan_sha256=getattr(plan, "plan_sha256", None))


def sequence_plan_summary_from_manifest(manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Build the same summary from the JSON manifest persisted with a batch job."""
    if manifest is None:
        return None
    if not isinstance(manifest, Mapping):
        raise ValueError("Sequence plan manifest must be a JSON object")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping):
        raise ValueError("Sequence plan manifest has no partition map")
    counts: dict[str, dict[str, int]] = {}
    for name in ("train", "validation", "test"):
        partition = partitions.get(name)
        if not isinstance(partition, Mapping):
            raise ValueError(f"Sequence plan manifest is missing {name!r}")
        rows = partition.get("row_positions")
        groups = partition.get("group_ids")
        targets = partition.get("window_target_positions")
        if not isinstance(rows, list) or not isinstance(groups, list):
            raise ValueError(f"Sequence plan manifest {name!r} counts are invalid")
        if targets is not None and not isinstance(targets, list):
            raise ValueError(f"Sequence plan manifest {name!r} windows are invalid")
        counts[name] = {
            "source_rows": len(rows),
            "groups": len(groups),
            "windows": 0 if targets is None else len(targets),
        }
    return _with_proportions(counts, plan_sha256=manifest.get("plan_sha256"))


def _with_proportions(counts: dict[str, dict[str, int]], *, plan_sha256: Any) -> dict[str, Any]:
    totals = {field: sum(split[field] for split in counts.values()) for field in ("source_rows", "groups", "windows")}
    splits: dict[str, dict[str, Any]] = {}
    for name, raw in counts.items():
        entry: dict[str, Any] = dict(raw)
        for field, total in totals.items():
            entry[f"{field}_proportion"] = raw[field] / total if total else None
        splits[name] = entry
    return {
        "schema_version": 1,
        "plan_sha256": plan_sha256,
        "totals": totals,
        "splits": splits,
    }
