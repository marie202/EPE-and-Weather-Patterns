"""Filename/path helpers for rr_01 feature-relevance outputs.

All rr_01 artefacts live under a mode subfolder — ``<region>/bbox/`` or
``<region>/regime/`` — so bbox and regime runs are physically separated and
can never overwrite or leak into each other, on top of also being tagged by
mode in the filename itself:

    <region>/bbox/feature_relevance_sequential_{region}_{bbox}_bbox_optionB.csv
    <region>/regime/feature_relevance_bootstrap_{region}_{bbox}_regime_stability.csv
    ...
"""
from __future__ import annotations

from pathlib import Path

RR01_MODES = ("bbox", "regime")

# How every non-Z500 field is scored in rr_01 (see feature_relevance.py's module
# docstring): 'domain_mean' is production (each candidate field scored on its
# spatially-averaged residual value, avoiding confounding the field-discriminative-
# power ranking with the per-field PCA mode-count disparity) or 'pca_block' (each
# field's full multi-mode PCA block). ALWAYS explicitly tagged in every rr_01
# filename (see rr01_tag) — the two representations must never be able to silently
# overwrite each other's CSVs at the same region/bbox/mode path.
RR01_FIELD_REPRESENTATIONS = ("domain_mean", "pca_block")
_FIELD_REPRESENTATION_TAG = {"domain_mean": "domainmean", "pca_block": "pcablock"}


def _field_representation_tag(field_representation: str) -> str:
    if field_representation not in RR01_FIELD_REPRESENTATIONS:
        raise ValueError(
            f"field_representation must be one of {RR01_FIELD_REPRESENTATIONS}, "
            f"got {field_representation!r}"
        )
    return _FIELD_REPRESENTATION_TAG[field_representation]


def rr01_tag(
    region: str,
    bbox_step: str,
    mode: str = "bbox",
    tag_suffix: str = "",
    *,
    z500_cluster_method: str = "kmeans",
    field_representation: str = "domain_mean",
) -> str:
    """Canonical tag embedded in rr_01 output filenames.

    ``z500_cluster_method``: which Z500 parent-regime source this run's extreme labels
    were sliced against in MODE="regime" (ignored in MODE="bbox", which never touches
    Z500 regimes) — "kmeans" (default; also covers the Ward-seeded K-means hybrid, which
    keeps the untagged path) or "ward" (pure Ward HRC, z_03's CLUSTER_MODE=ward; appends
    "_zward", the same tag ``stage2_residual.paths.experiment_id`` uses for Stage-2 RR
    outputs) — so a ward MODE="regime" run never overwrites an existing kmeans one at the
    same region/bbox, and vice versa.

    ``field_representation``: 'domain_mean' (production default) or 'pca_block' — see
    ``RR01_FIELD_REPRESENTATIONS``. ALWAYS appended ("_domainmean"/"_pcablock", never
    left implicit), so a run under one representation can never overwrite the other's
    files at the same region/bbox/mode — the two used to share one untagged filename,
    which meant whichever representation ran most recently silently became "the"
    selection every downstream consumer (write_field_selection.py, rr_02/rr_022's
    --build-regime-map) read, regardless of which one was actually intended as
    production.

    ``tag_suffix`` is for one-off diagnostics (e.g. evaluating a bbox against
    another bbox's regime labels) — it is appended after mode so results land
    in distinct files without colliding with the canonical bbox/regime outputs
    that ``resolve_rr01_csv`` (used by write_field_selection.py and rr_02) reads.
    """
    if mode not in RR01_MODES:
        raise ValueError(f"mode must be one of {RR01_MODES}, got {mode!r}")
    tag = f"{region}_{bbox_step}_{mode}"
    if z500_cluster_method == "ward":
        tag = f"{tag}_zward"
    tag = f"{tag}_{_field_representation_tag(field_representation)}"
    if tag_suffix:
        tag = f"{tag}_{tag_suffix}"
    return tag


def rr01_mode_dir(results_dir: Path, region: str, mode: str = "bbox") -> Path:
    """Directory holding all rr_01 artefacts for this region/mode combination.

    bbox and regime results live in physically separate subfolders so a
    regime-mode run can never overwrite or be mistaken for a bbox-mode one.
    """
    if mode not in RR01_MODES:
        raise ValueError(f"mode must be one of {RR01_MODES}, got {mode!r}")
    return Path(results_dir) / region / mode


def resolve_rr01_csv(
    results_dir: Path,
    region: str,
    bbox_step: str,
    kind: str,
    *,
    mode: str = "bbox",
    z500_cluster_method: str = "kmeans",
    tag_suffix: str = "",
    field_representation: str = "domain_mean",
) -> Path | None:
    """Return the rr_01 CSV for *kind* if it exists, or None.

    ``tag_suffix`` must match the value rr_01 was run with (e.g. a one-off diagnostic
    like a different EPE extremes definition or ``area_weight_tag``) — defaults to ""
    (the canonical/production file), same convention as ``resolve_rr01_per_regime_csv``.
    Production callers (``write_field_selection.write_yaml``) must never pass a
    tag_suffix here, or rr_02's field-selection YAML would silently pick up a
    diagnostic run's results instead of the canonical one.

    ``field_representation``: must match the value rr_01 was actually run with for
    this region/bbox/mode ('domain_mean', production default, or 'pca_block') — see
    ``rr01_tag``. Passing the wrong one resolves a real but different-representation
    CSV rather than failing loudly, since both now always exist as distinct files.
    """
    mode_dir = rr01_mode_dir(results_dir, region, mode)
    specs = {
        "sequential_optionB": ("feature_relevance_sequential", "_optionB.csv"),
        "residual_optionA": ("feature_relevance_residual", "_optionA.csv"),
        "hierarchical": ("feature_relevance_hierarchical", ".csv"),
        "bootstrap_stability": ("feature_relevance_bootstrap", "_stability.csv"),
    }
    if kind not in specs:
        raise ValueError(f"unknown kind {kind!r}; expected one of {sorted(specs)}")

    stem, suffix = specs[kind]
    # mode="bbox" never touches the Z500 regime file (global selection across all days), so
    # its filenames are never "_zward"-tagged regardless of what's requested here — mirrors
    # feature_relevance.run_feature_relevance's matching write-side exemption.
    _method = z500_cluster_method if mode == "regime" else "kmeans"
    tag = rr01_tag(region, bbox_step, mode, tag_suffix=tag_suffix, z500_cluster_method=_method,
                   field_representation=field_representation)
    path = mode_dir / f"{stem}_{tag}{suffix}"
    return path if path.exists() else None


def per_regime_dir(results_dir: Path, region: str, *, tag_suffix: str = "") -> Path:
    """Directory holding rr_01 MODE=regime per-Z500-regime CSVs.

    Mirrors the OUT_DIR convention in rr_01_feature_relevance_tu_hpc.sh: canonical
    runs land in ``<region>/regime/per_regime/``, while cross-bbox diagnostics
    submitted with a custom ``OUT_DIR=.../regime/<tag>_diagnostic`` land in
    ``<region>/regime/<tag>_diagnostic/per_regime/`` instead.
    """
    base = rr01_mode_dir(results_dir, region, "regime")
    if tag_suffix:
        diagnostic_dir = base / f"{tag_suffix}_diagnostic" / "per_regime"
        if diagnostic_dir.is_dir():
            return diagnostic_dir
    return base / "per_regime"


def resolve_rr01_per_regime_csv(
    results_dir: Path,
    region: str,
    bbox_step: str,
    z_regime_id: int,
    *,
    tag_suffix: str = "",
    z500_cluster_method: str = "kmeans",
    field_representation: str = "domain_mean",
) -> Path | None:
    """Return the rr_01 MODE=regime sequential-optionB CSV for one Z500 regime, or None.

    ``field_representation``: must match the value rr_01 MODE=regime was actually run
    with for this region/bbox ('domain_mean', production default, or 'pca_block') —
    see ``rr01_tag``.
    """
    ward_part = "_zward" if z500_cluster_method == "ward" else ""
    rep_part = f"_{_field_representation_tag(field_representation)}"
    tag_part = f"_{tag_suffix}" if tag_suffix else ""
    d = per_regime_dir(results_dir, region, tag_suffix=tag_suffix)
    path = d / f"feature_relevance_sequential_{region}_{bbox_step}_regime{ward_part}{rep_part}{tag_part}_R{z_regime_id}_optionB.csv"
    return path if path.exists() else None


def rr01_output_path(
    out_dir: Path,
    kind: str,
    region: str,
    bbox_step: str,
    mode: str = "bbox",
    *,
    z500_cluster_method: str = "kmeans",
    field_representation: str = "domain_mean",
) -> Path:
    """Canonical output path for a new rr_01 artefact (writers use this).

    ``out_dir`` is expected to already be the mode subfolder (i.e. what
    ``rr01_mode_dir`` returns) — callers set ``--out-dir`` that way.
    """
    tag = rr01_tag(region, bbox_step, mode, z500_cluster_method=z500_cluster_method,
                   field_representation=field_representation)
    templates = {
        "sequential_optionB": f"feature_relevance_sequential_{tag}_optionB.csv",
        "sequential_optionB_png": f"feature_relevance_sequential_{tag}_optionB.png",
        "residual_optionA": f"feature_relevance_residual_{tag}_optionA.csv",
        "residual_optionA_png": f"feature_relevance_residual_{tag}_optionA.png",
        "hierarchical": f"feature_relevance_hierarchical_{tag}.csv",
        "hierarchical_optionC_png": f"feature_relevance_hierarchical_{tag}_optionC.png",
        "bootstrap_stability": f"feature_relevance_bootstrap_{tag}_stability.csv",
        "sequential_all_regimes_png": f"feature_relevance_sequential_{tag}_all_regimes.png",
    }
    if kind not in templates:
        raise ValueError(f"unknown kind {kind!r}")
    return Path(out_dir) / templates[kind]
