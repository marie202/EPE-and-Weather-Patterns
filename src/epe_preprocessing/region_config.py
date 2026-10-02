from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore[assignment]


@dataclass
class RegionConfig:
    region_id: str
    label: str
    era5_hourly_dir: str
    regimes_dir: str
    bbox: dict[str, float] | None = None
    ref_point: dict[str, float] | None = None
    timezone: str = "UTC"


def resolve_step_root(path: Path | str) -> Path:
    """Convert nested 1_Detect_Weather_Patterns path back to detect_weather root.

    If path already contains src/epe_preprocessing/regions.yaml, return it as-is.
    Otherwise, traverse upward to find the parent directory that contains it.
    """
    path = Path(path).resolve()

    # Check if this is already the step root
    if (path / "src" / "epe_preprocessing" / "regions.yaml").is_file():
        return path

    # Traverse upward looking for the step root
    for parent in path.parents:
        if (parent / "src" / "epe_preprocessing" / "regions.yaml").is_file():
            return parent

    raise FileNotFoundError(
        f"Could not find src/epe_preprocessing/regions.yaml starting from {path}"
    )


def find_region_yaml_key(step_root: Path, study_area: str) -> str:
    """Find the yaml key (e.g. 'germany', 'ontario') for a given study_area.

    If study_area is already a yaml key, return it.
    Otherwise, search for a matching region_id field.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required — pip install pyyaml")

    regions_yaml = Path(step_root) / "src" / "epe_preprocessing" / "regions.yaml"
    raw: dict[str, Any] = yaml.safe_load(regions_yaml.read_text(encoding="utf-8"))

    # Direct key match
    if study_area in raw:
        return study_area

    # Match by region_id field
    for key, val in raw.items():
        if val.get("region_id") == study_area:
            return key

    available = {k: v.get("region_id", k) for k, v in raw.items()}
    raise KeyError(
        f"Study area {study_area!r} not found in {regions_yaml}. "
        f"Available keys/region_ids: {available}"
    )


def load_region_config(*, step_root: Path, region_id: str) -> RegionConfig:
    """Load a region entry from src/epe_preprocessing/regions.yaml.

    Looks up by the yaml key (e.g. 'ontario', 'germany') or by the
    nested ``region_id`` field (e.g. 'ontario_south').
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required — pip install pyyaml")

    regions_yaml = Path(step_root) / "src" / "epe_preprocessing" / "regions.yaml"
    raw: dict[str, Any] = yaml.safe_load(regions_yaml.read_text(encoding="utf-8"))

    # Match by yaml key or by nested region_id value
    entry: dict[str, Any] | None = None
    for key, val in raw.items():
        if key == region_id or val.get("region_id") == region_id:
            entry = dict(val)
            if "region_id" not in entry:
                entry["region_id"] = key
            break

    if entry is None:
        available = {k: v.get("region_id", k) for k, v in raw.items()}
        raise KeyError(
            f"Region {region_id!r} not found in {regions_yaml}. "
            f"Available keys/region_ids: {available}"
        )

    return RegionConfig(
        region_id=str(entry.get("region_id", region_id)),
        label=str(entry.get("label", region_id)),
        era5_hourly_dir=str(entry.get("era5_hourly_dir", "")),
        regimes_dir=str(entry.get("regimes_dir", "")),
        bbox=entry.get("bbox"),
        ref_point=entry.get("ref_point"),
        timezone=str(entry.get("timezone", "UTC")),
    )
