from __future__ import annotations

"""Orchestration for residual regimes two-stage clustering pipeline.

Stage 1: Cluster Z500 to define circulation regimes.
         Reuses existing legacy Z500 regime files if available.
         Warns and recomputes if legacy file not found (e.g. different k).
Bridge: Compute residual anomalies (anomaly - regime-conditioned mean).
Stage 2: Cluster residual PCs with k-means to find sub-regimes.
"""

from dataclasses import asdict
from pathlib import Path
import time
import warnings

import xarray as xr

from stage2_residual.fields.cluster import run_kmeans_regimes, run_ward_hierarchical_clustering, write_regime_characterisation
from stage2_residual.fields.config import MultivariateRunConfig
from stage2_residual.fields.field_specs import get_field_spec
from stage2_residual.fields.joint_features import build_joint_feature_dataset
from stage2_residual.fields.per_field_pca import run_per_field_pca
from stage2_residual.fields.preprocess import run_field_preprocessing, explicit_raw_nc_for_field
from stage2_residual.fields.reuse import resolve_legacy_z500_artifacts
from stage2_residual.fields.provenance import FieldManifest, write_manifest

from .config import ResidualRunConfig, build_arg_parser, config_from_args
from .paths import ResidualWorkflowPaths, experiment_id, resolve_mf_pca_path
from .per_regime_stage2 import merge_per_regime_outputs, run_stage2_for_z500_regime
from .postprocess import remap_rare_regimes_per_z500

__all__ = [
    "ResidualRunConfig",
    "ResidualWorkflowPaths",
    "build_arg_parser",
    "config_from_args",
    "experiment_id",
    "main",
    "run_pipeline",
]


# ---------------------------------------------------------------------------
# Legacy Z500 regime lookup
# ---------------------------------------------------------------------------

def _detect_weather_root(config: ResidualRunConfig) -> Path:
    """Resolve the detect_weather root (one level above module_root if needed).

    module_root may be .../detect_weather or .../detect_weather/1_Detect_Weather_Patterns.
    Legacy clustering and preprocessed files always live under detect_weather/.
    """
    module_root = config.module_root
    # If clustering dir exists directly under module_root, use it
    if (module_root / "results" / "clustering").is_dir():
        return module_root
    # Otherwise try one level up (module_root = .../1_Detect_Weather_Patterns)
    parent = module_root.parent
    if (parent / "results" / "clustering").is_dir():
        return parent
    # Fall back to module_root and let callers handle missing dirs
    return module_root


def _find_legacy_z500_regimes(config: ResidualRunConfig) -> Path | None:
    """Search for existing Z500 regime file from the original z500 pipeline.

    Looks in results/clustering/{region}/ for a file matching the requested
    z500_k, smooth_window, z500_epsilon, var_threshold, year_span, and bbox_step.

    Returns None if not found — caller should warn and recompute.
    """
    dw_root = _detect_weather_root(config)
    legacy_dir = dw_root / "results" / "clustering" / config.region
    if not legacy_dir.is_dir():
        return None

    k_str   = f"k{config.z500_k:02d}"
    w_str   = f"w{config.smooth_window:02d}"
    eps_str = f"eps{int(round(config.z500_epsilon)):02d}"
    var_str = f"var{int(round(config.var_threshold * 100)):02d}"
    method  = getattr(config, "z500_cluster_method", None) or "kmeans"

    # Pattern: 09_Z500_regimes_{kmeans|ward}_k05_w11_eps20_var90[_yj]_1979-2025_stepB5_*.nc
    # The wildcard right after var_str absorbs an optional "_yj" Yeo-Johnson tag
    # some bbox/region Stage-1 clustering runs were produced with (not every
    # legacy Z500 file has one, and which variant exists is inconsistent across
    # regions/bboxes) — without it, a region/bbox that only has the "_yj_"
    # variant on disk (e.g. northamerica B5 k04) matches nothing and this
    # function wrongly reports the file as missing.
    pattern = (
        f"09_Z500_regimes_{method}_{k_str}_{w_str}_{eps_str}_{var_str}*"
        f"_{config.year_span}_step{config.bbox_step}_*.nc"
    )
    matches = sorted(legacy_dir.glob(pattern))

    if not matches:
        return None
    if len(matches) > 1:
        # Ascending sort puts a bare "..._var90_1979-..." before "..._var90_yj_1979-...",
        # so matches[-1] naturally prefers the "_yj" variant when both exist —
        # consistent with the rest of the pipeline defaulting to Yeo-Johnson.
        print(
            f"  [rr_pipeline] WARNING: Multiple legacy Z500 files matched pattern, "
            f"using most recent: {matches[-1].name}"
        )
    return matches[-1]


def _exclude_rr_tagged(paths: list[Path], *, tag: str) -> list[Path]:
    """Drop field-set-specific artifacts left over from a PRIOR residual-regime run
    (filenames containing e.g. ``_anomaly_rr_z500k5_z500vo850+cape_...`` or
    ``_preprocessed_rr_z500k5_z500vo850+cape_...``).

    Without this, picking `sorted(matches)[-1]` (as both callers below do) can
    silently prefer a stale ``rr_z500k*``-tagged file over the correct
    general-purpose one purely because ``'r' > 'm'`` alphabetically (e.g.
    ``..._anomaly_rr_z500k5_..._stepB4...`` sorts after
    ``..._anomaly_mf_cape_..._stepB4...``). Mirrors the exclusion
    ``feature_relevance.py::_find_anomaly_file`` already applies for the same
    reason when resolving rr_01's own anomaly file.

    Deliberately does NOT fall back to the stale files if exclusion empties the
    list — an empty result here is a genuine "run rr_00 preprocessing for this
    field/bbox" signal, not something to paper over with combo-specific data.
    """
    return [
        p for p in paths
        if f"_{tag}_rr_z500k" not in p.name and f"_{tag}_rr_z500" not in p.name
    ]


def _check_anomaly_files(
    config: ResidualRunConfig,
    paths: ResidualWorkflowPaths,
    stamp,
) -> dict[str, Path]:
    """Check that anomaly files exist for all residual fields. Returns found paths."""
    found: dict[str, Path] = {}
    missing: list[str] = []

    dw_root = _detect_weather_root(config)

    for field_key in config.residual_fields:
        if field_key == "z500":
            # z500 anomaly lives under detect_weather/data/preprocessed/{region}/
            z500_dir = dw_root / "data" / "preprocessed" / config.region
            if z500_dir.is_dir():
                matches = sorted(z500_dir.glob(f"*z500*anom*step{config.bbox_step}*.nc"))
                if matches:
                    # Prefer file matching config.smooth_window (e.g. w11)
                    w_tag = f"_w{config.smooth_window}_"
                    preferred = [m for m in matches if w_tag in m.name]
                    found["z500"] = preferred[-1] if preferred else matches[-1]
                    continue
            missing.append("z500")
        else:
            # Other fields under detect_weather/data/multivariate/preprocessed/{region}/{field}/
            prep_dir = (
                dw_root / "data" / "multivariate" / "preprocessed"
                / config.region / field_key
            )
            if prep_dir.is_dir():
                matches = sorted(prep_dir.glob(f"03_MF_{field_key}_anomaly_*step{config.bbox_step}*.nc"))
                matches = _exclude_rr_tagged(matches, tag="anomaly")
                if matches:
                    found[field_key] = matches[-1]
                    continue
            missing.append(field_key)

    if missing:
        stamp(f"  MISSING anomaly files for: {', '.join(missing)}")
        stamp(f"  → Run rr_00 preprocessing first for these fields")

    # Check for smooth-window mismatches across files
    import re
    windows = {}
    for field_key, path in found.items():
        # Match smooth window that is followed by _eps or _detrdm (not a field name like w700)
        m = re.search(r"_w(\d+)_(?:eps|detrdm)", path.name)
        if m:
            windows[field_key] = int(m.group(1))

    if windows:
        unique_windows = set(windows.values())
        for field_key, w in windows.items():
            stamp(f"  smooth_window {field_key}: w{w}")
        if len(unique_windows) > 1:
            stamp(f"  WARNING: smooth_window mismatch across fields: {windows}")
            stamp(f"  → This may cause time-dimension issues. Ensure all fields use the same smooth_window.")

    return found


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Execute the full two-stage residual regimes pipeline."""
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    paths = ResidualWorkflowPaths(config)

    t0 = time.perf_counter()

    def stamp(msg: str) -> None:
        dt = time.perf_counter() - t0
        print(f"[residual_regimes | +{dt:7.1f}s] {msg}")

    stamp(f"Regime set: {config.regime_set_label}")
    stamp(f"Region: {config.region}  Bbox: {config.bbox_step}  Year span: {config.year_span}")
    stamp(f"Stage 1: Z500 circulation regimes (K={config.z500_k})")
    if config.per_regime_mode:
        resid_tag = "residualized (Option 2)" if config.residualize else "NOT residualized (raw anomaly PCs)"
        stamp(f"Stage 2: Residual sub-clustering [per-regime, {resid_tag}] "
              f"(fields union={'+'.join(config.residual_fields)}, per-Z500 field/k selection below)")
    else:
        stamp(f"Stage 2: Residual sub-clustering (K={config.final_k}, fields={'+'.join(config.residual_fields)})")

    if config.dry_run:
        stamp("=" * 60)
        stamp("DRY RUN — checking inputs, no computation will run")
        stamp("=" * 60)

    # ------------------------------------------------------------------
    # Check legacy Z500 regime file
    # ------------------------------------------------------------------
    if config.z500_regimes_override is not None:
        # Cross-bbox diagnostic: classify THIS bbox's residual fields using ANOTHER
        # bbox's Z500 circulation regimes, instead of this bbox's own. Bypasses the
        # bbox_step-based glob search entirely — config.z500_k must already match
        # the override file's actual regime count (not verified against the file).
        if not config.z500_regimes_override.is_file():
            raise FileNotFoundError(f"z500_regimes_override not found: {config.z500_regimes_override}")
        legacy_z500_regimes = config.z500_regimes_override
        stamp(f"Stage 1 [OVERRIDE]: Using explicit Z500 regimes file (cross-bbox diagnostic): "
              f"{legacy_z500_regimes.name}")
    else:
        legacy_z500_regimes = _find_legacy_z500_regimes(config)

        if legacy_z500_regimes is not None:
            stamp(f"Stage 1 [REUSE]: Found legacy Z500 regimes: {legacy_z500_regimes.name}")
        else:
            msg = (
                f"No legacy Z500 regime file found for "
                f"K={config.z500_k}, bbox={config.bbox_step}, region={config.region}, "
                f"year_span={config.year_span}.\n"
                f"  Expected pattern: results/clustering/{config.region}/"
                f"09_Z500_regimes_kmeans_k{config.z500_k:02d}_w{config.smooth_window:02d}_"
                f"eps{int(config.z500_epsilon):02d}_var{int(config.var_threshold*100):02d}_"
                f"{config.year_span}_step{config.bbox_step}_*.nc\n"
                f"  → Z500 Stage 1 clustering will be RECOMPUTED (adds ~1-2 hours)."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            stamp(f"Stage 1 [WARNING]: {msg}")

    # ------------------------------------------------------------------
    # Check anomaly files for all fields
    # ------------------------------------------------------------------
    stamp("Checking anomaly files for all residual fields...")
    anomaly_paths = _check_anomaly_files(config, paths, stamp)

    missing_fields = [f for f in config.residual_fields if f not in anomaly_paths]
    if missing_fields:
        stamp(f"ERROR: Missing preprocessed anomaly files for: {', '.join(missing_fields)}")
        if config.dry_run:
            stamp("DRY RUN complete — fix missing files before running.")
            return
        raise RuntimeError(
            f"Missing preprocessed anomaly files for fields: {missing_fields}. "
            f"Run rr_00 preprocessing first."
        )

    for field_key, path in anomaly_paths.items():
        stamp(f"  ✓ {field_key}: {path.name}")

    if config.dry_run:
        stamp("=" * 60)
        stamp("DRY RUN SUMMARY")
        stamp("=" * 60)
        stamp(f"  Stage 1 Z500 regimes: {'REUSE ' + legacy_z500_regimes.name if legacy_z500_regimes else 'RECOMPUTE (not found)'}")
        stamp(f"  Anomaly files ready:  {list(anomaly_paths.keys())}")
        stamp(f"  Missing fields:       {missing_fields or 'none'}")
        stamp(f"  Output will go to:    {paths.clustering_root}")
        stamp("DRY RUN complete — re-run without --dry-run to execute.")
        return

    # ------------------------------------------------------------------
    # STAGE 1: Z500 circulation regimes
    # ------------------------------------------------------------------
    manifest_fields: list[FieldManifest] = []
    pca_paths_by_field: dict[str, Path] = {}
    outputs: dict[str, str] = {}

    if legacy_z500_regimes is not None:
        stamp(f"Stage 1: Reusing legacy Z500 regimes — skipping z500 clustering entirely.")
        z500_regimes = legacy_z500_regimes
        outputs["z500_stage1_regimes"] = str(z500_regimes)
    else:
        stamp("Stage 1: No legacy file found — recomputing Z500 clustering...")
        legacy = resolve_legacy_z500_artifacts(_config_to_multivariate(config))

        z500_pca_result = run_per_field_pca(
            field_key="z500",
            in_preprocessed_path=anomaly_paths["z500"],
            out_pca_path=paths.field_pca_path("z500"),
            var_threshold=config.var_threshold,
            max_modes=config.max_modes_per_field,
            overwrite=config.overwrite,
        )
        pca_paths_by_field["z500"] = z500_pca_result.pca_path

        z500_joint_path = paths.residuals_dir / f"08_RR_z500_joint_{experiment_id(config)}.nc"
        z500_joint_path.parent.mkdir(parents=True, exist_ok=True)
        z500_joint = build_joint_feature_dataset(
            pca_paths_by_field={"z500": z500_pca_result.pca_path},
            out_path=z500_joint_path,
            overwrite=config.overwrite,
        )
        z500_regimes = run_kmeans_regimes(
            joint_features_path=z500_joint,
            out_regimes_path=paths.z500_regimes_stage1_path,
            final_k=config.z500_k,
            random_state=config.random_state,
            overwrite=config.overwrite,
            ward_hc_path=None,
        )
        stamp(f"Stage 1: Z500 regimes computed (K={config.z500_k}): {z500_regimes}")
        outputs["z500_stage1_regimes"] = str(z500_regimes)

    # ------------------------------------------------------------------
    # STAGE 2: reuse rr_00's per-field PCA (07_MF_PCA) + joint features + Ward HC
    # + k-means. No residual-specific PCA is recomputed here — see module
    # docstring: the old "07_RR_PCA"/scalar-sigma recompute is legacy.
    # ------------------------------------------------------------------
    stamp("Stage 2: Reusing rr_00's per-field PCA (07_MF_PCA)...")
    for field_key in config.residual_fields:
        if field_key == "z500":
            continue
        mf_pca_path = resolve_mf_pca_path(config, field_key)
        if mf_pca_path is None:
            raise RuntimeError(
                f"Stage 2 PCA for {field_key}: no rr_00 07_MF_PCA file found for "
                f"bbox={config.bbox_step}, region={config.region}. "
                f"Run rr_00 preprocessing (--stop-after-step 7) first for this field/bbox, "
                f"or check --sigma-mode/--yeo-johnson match how rr_00 was run."
            )
        with xr.open_dataset(mf_pca_path, chunks=None) as ds:
            retained = int(ds.attrs.get("k", ds["PCs_reduced"].sizes.get("mode", 0)))
        pca_paths_by_field[field_key] = mf_pca_path
        manifest_fields.append(FieldManifest(
            field_key=field_key,
            status="reused",
            reuse_tier="mf_pca",
            output_paths=[str(mf_pca_path)],
            retained_modes=retained,
        ))
        stamp(f"  Stage 2 PCA {field_key}: reusing {mf_pca_path.name} ({retained} modes)")

    if config.per_regime_mode:
        # ------------------------------------------------------------------
        # STAGE 2 [per-regime]: independent joint features + Ward HC + k-means
        # PER Z500 regime, each with its own field subset and own k (explicit
        # or auto via silhouette sweep on that regime's own day-subset), then
        # flattened back into one contiguous 1..K_total regime_labels.
        # ------------------------------------------------------------------
        if config.stop_after_step <= 8:
            stamp("NOTE: --stop-after-step <= 8 has no effect in per_regime_mode — "
                  "running the full per-regime sweep+kmeans+merge+filter+characterisation in one pass.")

        stamp(f"Stage 2 [per-regime]: {len(config.per_regime_field_map)} Z500 regimes, "
              f"final_k_range={config.final_k_range}")
        per_regime_results = {}
        for z, fields_for_regime in sorted(config.per_regime_field_map.items()):
            k_spec = (config.per_regime_k_spec or {}).get(z, "auto")
            regime_pca_paths = {
                f: pca_paths_by_field[f] for f in fields_for_regime if f != "z500"
            }
            result = run_stage2_for_z500_regime(
                z_regime_id=z,
                z500_regimes_path=z500_regimes,
                pca_paths_by_field=regime_pca_paths,
                k_spec=k_spec,
                final_k_range=config.final_k_range,
                joint_features_alldays_path=paths.per_region_joint_features_alldays_path(z),
                joint_features_subset_path=paths.per_region_joint_features_path(z),
                ward_hc_path=paths.per_region_ward_hc_path(z),
                local_regimes_path=paths.per_region_local_regimes_path(z),
                random_state=config.random_state,
                overwrite=config.overwrite,
                residualize=config.residualize,
            )
            per_regime_results[z] = result
            stamp(f"  Z500-{z}: fields={'+'.join(fields_for_regime)} k={result.k_used} "
                  f"({result.k_selection_method})")

        raw_merged_path = merge_per_regime_outputs(
            per_regime_results,
            out_path=paths.regimes_path_per_regime,
            overwrite=config.overwrite,
        )
        outputs["stage2_regimes_per_regime_raw"] = str(raw_merged_path)

        stamp(f"Post-processing: per-Z500 rare-regime filter (< {10} days within each Z500 regime)...")
        filtered_path, active_rrs_per_z500 = remap_rare_regimes_per_z500(
            regimes_path=raw_merged_path,
            z500_regimes_path=z500_regimes,
            out_path=paths.regimes_filtered_path_per_regime,
            min_days=10,
            overwrite=config.overwrite,
            restrict_fallback_to_same_regime=True,
        )
        for z_r, active in sorted(active_rrs_per_z500.items()):
            stamp(f"  Z500-{z_r}: active subregimes = SR{{{', '.join(str(r) for r in active)}}}")
        outputs["stage2_regimes_filtered"] = str(filtered_path)

        characterisation_path = write_regime_characterisation(
            regimes_path=filtered_path,
            anomaly_paths_by_field=anomaly_paths,
            out_path=paths.regime_characterisation_path_per_regime,
            overwrite=config.overwrite,
        )
        outputs["regime_characterisation"] = str(characterisation_path)
    else:
        # ------------------------------------------------------------------
        # STAGE 2 [global]: one joint feature space + Ward HC + k-means across
        # ALL days, ignoring which Z500 regime each day belongs to.
        # ------------------------------------------------------------------
        stamp("Stage 2: Building joint features and Ward hierarchy...")
        joint_path = build_joint_feature_dataset(
            pca_paths_by_field=pca_paths_by_field,
            out_path=paths.joint_features_path,
            overwrite=config.overwrite,
            regime_set_label=f"Residual Stage 2 (K={config.final_k})",
            regime_set_id=f"rr_stage2_k{config.final_k}",
        )

        hc_path = run_ward_hierarchical_clustering(
            joint_features_path=joint_path,
            out_hc_path=paths.ward_hc_path,
            overwrite=config.overwrite,
        )
        outputs["ward_hc"] = str(hc_path)

        if config.stop_after_step <= 8:
            stamp(f"Stopped after step 8 (Ward HC) — re-run with --final-k <k> to complete k-means.")
            return

        stamp(f"Stage 2: Running k-means with K={config.final_k}...")
        regimes_path = run_kmeans_regimes(
            joint_features_path=joint_path,
            out_regimes_path=paths.regimes_path,
            final_k=config.final_k,
            random_state=config.random_state,
            overwrite=config.overwrite,
            regime_set_label=config.regime_set_label,
            regime_set_id=config.regime_set_id,
            ward_hc_path=hc_path,
        )
        stamp(f"Stage 2: Final regimes (K={config.final_k}): {regimes_path.name}")
        outputs["stage2_regimes"] = str(regimes_path)

        # ------------------------------------------------------------------
        # POST-PROCESSING: merge rare regimes (< 10 days) into nearest valid
        # ------------------------------------------------------------------
        stamp(f"Post-processing: per-Z500 rare-regime filter (< {10} days within each Z500 regime)...")
        filtered_path, active_rrs_per_z500 = remap_rare_regimes_per_z500(
            regimes_path=regimes_path,
            z500_regimes_path=z500_regimes,
            out_path=paths.regimes_filtered_path,
            min_days=10,
            overwrite=config.overwrite,
        )
        for z_r, active in sorted(active_rrs_per_z500.items()):
            stamp(f"  Z500-{z_r}: active subregimes = SR{{{', '.join(str(r) for r in active)}}}")
        outputs["stage2_regimes_filtered"] = str(filtered_path)

        characterisation_path = write_regime_characterisation(
            regimes_path=filtered_path,
            anomaly_paths_by_field=anomaly_paths,
            out_path=paths.regime_characterisation_path,
            overwrite=config.overwrite,
        )
        outputs["regime_characterisation"] = str(characterisation_path)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    outputs["manifest"] = str(paths.manifest_path)
    config_payload = asdict(config)
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        paths.manifest_path,
        config=config_payload,
        fields=manifest_fields,
        outputs=outputs,
        regime_set_label=config.regime_set_label,
        regime_set_id=config.regime_set_id,
    )

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    try:
        import numpy as np
        ds_out = xr.open_dataset(filtered_path)
        ds_z500 = xr.open_dataset(z500_regimes)
        rr_labels = ds_out["regime_labels"].values
        z500_labels = ds_z500["regime_labels"].values
        n_days = len(rr_labels)

        stamp("=" * 60)
        stamp("RESULTS SUMMARY")
        stamp("=" * 60)
        stamp(f"  Output: {filtered_path.name}")
        stamp(f"  Total days classified: {n_days}")

        # Overall residual regime distribution
        stamp(f"")
        import json
        active_per_z500 = json.loads(ds_out.attrs.get("active_rrs_per_z500", "{}"))
        effective_k = int(ds_out.attrs.get("effective_k", int(rr_labels.max())))
        stamp(f"  Stage 2 residual regimes (K={effective_k}) — overall:")
        unique_rr, counts_rr = np.unique(rr_labels, return_counts=True)
        for regime, count in zip(unique_rr, counts_rr):
            pct = 100 * count / n_days
            stamp(f"    SR{int(regime)}: {count:5d} days ({pct:5.1f}%)")

        # Cross-tabulation: z500 regime × residual sub-regime
        stamp(f"")
        stamp(f"  Sub-regime breakdown within each Z500 regime (K={config.z500_k}):")
        unique_z500 = np.unique(z500_labels)
        for z_regime in unique_z500:
            z_mask = z500_labels == z_regime
            z_count = z_mask.sum()
            z_pct = 100 * z_count / n_days
            stamp(f"    Z500-{int(z_regime)} ({z_count} days, {z_pct:.1f}%):")
            rr_within = rr_labels[z_mask]
            unique_sub, counts_sub = np.unique(rr_within, return_counts=True)
            for sub, sub_count in zip(unique_sub, counts_sub):
                sub_pct_of_z = 100 * sub_count / z_count
                sub_pct_total = 100 * sub_count / n_days
                stamp(f"      → SR{int(sub)}: {sub_count:5d} days ({sub_pct_of_z:.1f}% of Z500-{int(z_regime)}, {sub_pct_total:.1f}% total)")

        ds_out.close()
        ds_z500.close()
        stamp("=" * 60)

        # Warn if degenerate
        if any(c < 10 for c in counts_rr):
            stamp("  WARNING: Degenerate clustering detected (some regimes have < 10 days).")
            stamp("  Consider re-running with --overwrite to reset Ward HC and k-means.")
    except Exception as e:
        stamp(f"  (Could not print results summary: {e})")

    stamp("Residual regimes pipeline completed successfully.")


def _config_to_multivariate(config: ResidualRunConfig) -> MultivariateRunConfig:
    """Convert ResidualRunConfig to MultivariateRunConfig for reusing preprocessing."""
    return MultivariateRunConfig(
        module_root=config.module_root,
        region=config.region,
        bbox_step=config.bbox_step,
        bbox_config=config.bbox_config,
        bbox=config.bbox,
        raw_years_span=config.raw_years_span,
        year_span=config.year_span,
        smooth_window=config.smooth_window,
        detrend_method=config.detrend_method,
        climatology_subtract=config.climatology_subtract,
        sigma_mode=config.sigma_mode,
        var_threshold=config.var_threshold,
        max_modes_per_field=config.max_modes_per_field,
        final_k=config.final_k,
        random_state=config.random_state,
        stop_after_step=config.stop_after_step,
        z500_epsilon=config.z500_epsilon,
        vo850_epsilon=config.vo850_epsilon,
        tcwv_epsilon=config.tcwv_epsilon,
        deg0l_epsilon=config.deg0l_epsilon,
        vo850_smoothing_method=config.vo850_smoothing_method,
        vo850_gaussian_sigma=config.vo850_gaussian_sigma,
        vo850_coarsen_factor=config.vo850_coarsen_factor,
        source_fields=config.residual_fields,
        overwrite=config.overwrite,
        z500_raw_nc=config.z500_raw_nc,
        vo850_raw_nc=config.vo850_raw_nc,
        tcwv_raw_nc=config.tcwv_raw_nc,
        deg0l_raw_nc=config.deg0l_raw_nc,
    )


def main() -> None:
    """Entry point for CLI."""
    run_pipeline()
