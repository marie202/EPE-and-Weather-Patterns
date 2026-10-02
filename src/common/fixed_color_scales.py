"""Shared "one fixed color scale per field type, per region" YAML, read by every
composite-plotting script so a given field (e.g. Z500 mean anomaly) maps to the same color
across every bbox step -- but Europe and North America each get their own independently-pooled
vmin/vmax (since a single shared range makes North America's, typically weaker, anomalies wash
out relative to Europe's), and EPE flag-frequency additionally gets a SEPARATE scale per
pipeline (``EPE_FLAG_FREQ_Z500`` vs ``EPE_FLAG_FREQ_RR``) since RR's per-sub-regime-bin
frequencies run much hotter than the Z500-only pipeline's per-whole-regime frequencies -- see
those constants' docstring below. ``EPE_FLAG_FREQ_HIGHRISK`` is the one exception to the "each
region gets its own vmin/vmax" rule: both regions' YAML sections carry the SAME value for this
key, because it's pooled together across regions by ``scripts/compute_fixed_composite_scales.py``
for direct Europe/North-America comparability -- see that script's module docstring.
``EPE_FLAG_FREQ_RR`` used to share that same cross-region pooling but now gets its own per-region
vmin/vmax like every other field (a Germany-dominated shared vmax washed out Germany's own
panels against Ontario's typically higher frequencies).

Same fallback convention as ``k_overrides_<region>_custom.yaml`` (see
``src/common/resolver.py::load_k_overrides``): if the YAML doesn't exist yet, or has
no entries for the requested region, callers get an empty mapping and fall back to today's
per-call auto-scaling. Populated by ``scripts/compute_fixed_composite_scales.py`` from real
pooled data — see that script's docstring for how the numbers are derived.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Keys shared by every plotting script that consumes this file.
Z500_MEAN_ANOM_GPM = "z500_mean_anom_gpm"
Z500_STD_GPM = "z500_std_gpm"
# EPE flag-frequency, pooled SEPARATELY per pipeline (not into one shared range): the Z500-only
# pipeline's panels show "% of days flagged" per WHOLE Z500 regime, while the RR pipeline's
# panels show it per (much smaller, deliberately extreme-day-concentrated) probability sub-bin --
# RR's peaks run far hotter, so a shared scale left the Z500-only panels' real (lower, but still
# meaningful) signal barely visible under a range dominated by RR.
EPE_FLAG_FREQ_Z500 = "epe_flag_freq_z500"
EPE_FLAG_FREQ_RR = "epe_flag_freq_rr"
# Same field as EPE_FLAG_FREQ_RR, but pooled ONLY from each Z500 regime's own highest-probability
# ("high-risk") bin, instead of every bin -- gives the high-risk-bin composite panels headroom to
# show their real spatial variation instead of saturating against a scale dominated by low-risk
# bins (see scripts/compute_fixed_composite_scales.py and residual_regimes/plot_composites.py's
# epe_vmax_fixed_highrisk plumbing).
EPE_FLAG_FREQ_HIGHRISK = "epe_flag_freq_highrisk"

ALL_KEYS = (
    Z500_MEAN_ANOM_GPM, Z500_STD_GPM, EPE_FLAG_FREQ_Z500, EPE_FLAG_FREQ_RR, EPE_FLAG_FREQ_HIGHRISK,
)


def default_fixed_scales_path(module_root: Path) -> Path:
    return Path(module_root) / "src" / "common" / "fixed_color_scales.yaml"


# Key under which the generating EPE-definition/cluster-method fingerprint is stored, so
# consumers (ensure_fixed_color_scales.sh) can tell a stale YAML (produced under a different
# EXTREMES_*/DOMAIN_*/CLUSTER_METHOD combo) from a current one without re-deriving the numbers.
META_KEY = "_meta"


def load_fixed_color_scales(path: Path | None, region: str) -> dict[str, tuple[float, float]]:
    """Return ``{key: (vmin, vmax)}`` for whichever of ``ALL_KEYS`` are present in ``region``'s
    section of the YAML at ``path`` (e.g. ``"europe"``, ``"northamerica"`` -- each region is
    pooled and scaled independently, see ``scripts/compute_fixed_composite_scales.py``).
    Returns ``{}`` if ``path`` is ``None``, the file doesn't exist, or ``region`` has no section
    (safe default -- callers should fall back to their existing auto-scaling in that case).
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required for fixed_color_scales (pip install pyyaml).")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {p}")

    region_data = data.get(region)
    if not isinstance(region_data, dict):
        return {}

    out: dict[str, tuple[float, float]] = {}
    for key in ALL_KEYS:
        entry = region_data.get(key)
        if not isinstance(entry, dict) or "vmin" not in entry or "vmax" not in entry:
            continue
        out[key] = (float(entry["vmin"]), float(entry["vmax"]))
    return out


def load_fixed_color_scales_tag(path: Path | None) -> str | None:
    """Return the ``definition_tag`` stashed under ``_meta`` by :func:`save_fixed_color_scales`,
    or ``None`` if the file/section/tag is absent. Used by
    ``scripts/ensure_fixed_color_scales.sh`` to decide whether the YAML on disk still matches the
    EPE definition/cluster methods the caller is about to plot with, without re-pooling data.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file() or yaml is None:
        return None
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return None
    meta = data.get(META_KEY)
    if not isinstance(meta, dict):
        return None
    tag = meta.get("definition_tag")
    return str(tag) if tag is not None else None


def save_fixed_color_scales(
    values_by_region: dict[str, dict[str, tuple[float, float]]],
    path: Path,
    definition_tag: str | None = None,
) -> None:
    """Write ``values_by_region`` (``{region: {key: (vmin, vmax)}}``) to ``path`` in the format
    :func:`load_fixed_color_scales` reads back. ``definition_tag``, if given, is stashed under
    the top-level ``_meta.definition_tag`` (shared across regions) so
    :func:`load_fixed_color_scales_tag` can later detect staleness.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required for fixed_color_scales (pip install pyyaml).")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# Fixed composite color scales -- pinned so a given field (Z500 mean anomaly, Z500\n"
        "# anomaly std-dev) maps to the same color across every bbox step and both the Z500-only\n"
        "# and RR pipelines, while EPE flag-frequency gets its own scale PER PIPELINE\n"
        "# (epe_flag_freq_z500 vs epe_flag_freq_rr/epe_flag_freq_highrisk) since RR's per-bin\n"
        "# frequencies run much hotter than the Z500-only pipeline's per-whole-regime ones.\n"
        "# Europe and North America are pooled and scaled INDEPENDENTLY (separate top-level\n"
        "# sections below) so North America's typically weaker anomalies don't wash out under a\n"
        "# range dominated by Europe -- epe_flag_freq_highrisk is the one exception, pooled\n"
        "# TOGETHER across both regions for direct side-by-side comparability. Computed by\n"
        "# scripts/compute_fixed_composite_scales.py from real pooled composite data. Every\n"
        "# z_04/z_04b/rr_03/rr_03b composite job sources scripts/ensure_fixed_color_scales.sh,\n"
        "# which checks _meta.definition_tag below against the EPE definition/cluster methods in\n"
        "# effect for that run and, on a mismatch, WARNS and falls back to per-call auto-scaling\n"
        "# for that run only -- it does NOT auto-regenerate this file; rerun\n"
        "# sbatch compute_fixed_composite_scales_tu_hpc.sh deliberately when the pinned production\n"
        "# EPE definition actually changes. Delete this file (or point --fixed-scales-yaml\n"
        "# elsewhere) to fall back to today's per-call auto-scaling.\n\n"
    )
    body: dict[str, Any] = {}
    for region, values in values_by_region.items():
        body[region] = {
            key: {"vmin": float(vmin), "vmax": float(vmax)}
            for key, (vmin, vmax) in values.items()
            if key in ALL_KEYS
        }
    if definition_tag is not None:
        body[META_KEY] = {"definition_tag": str(definition_tag)}
    with p.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(body, f, sort_keys=False, default_flow_style=False)
