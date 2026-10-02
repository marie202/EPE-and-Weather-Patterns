from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    """Recursively convert Path and other non-JSON types for manifest export."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass
class FieldManifest:
    field_key: str
    status: str
    reuse_tier: str
    source_path: str | None = None
    output_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    smoothing: dict[str, Any] | None = None
    retained_modes: int | None = None


def write_manifest(
    path: Path,
    *,
    config: dict[str, Any],
    fields: list[FieldManifest],
    outputs: dict[str, str],
    regime_set_label: str | None = None,
    regime_set_id: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(
        {
            "config": config,
            "regime_set_label": regime_set_label,
            "regime_set_id": regime_set_id,
            "fields": [asdict(entry) for entry in fields],
            "outputs": outputs,
        }
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
