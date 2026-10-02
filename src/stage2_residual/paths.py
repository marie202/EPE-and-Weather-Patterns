from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from common.preprocessing_helpers import bbox_step_filename_suffix

from .config import ResidualRunConfig


def _sigma_label(sigma_mode: str) -> str:
    return "sigsc" if str(sigma_mode) == "scalar" else "sigf"


def _detrend_label(detrend_method: str) -> str:
    return "dm" if str(detrend_method) == "domain_mean" else "gp"


def _clim_label(clim_subtract: str) -> str:
    return "clims" if str(clim_subtract) == "smooth" else "climr"


def option_label(config: ResidualRunConfig) -> str:
    base = f"detr{_detrend_label(config.detrend_method)}_{_clim_label(config.climatology_subtract)}_{_sigma_label(config.sigma_mode)}"
    if config.yeo_johnson:
        base = f"{base}_yj"
    if config.bbox:
        return f"{base}_stepcustom"
    return base + bbox_step_filename_suffix(config.module_root, config.region, config.bbox_step)


def field_list_slug(config: ResidualRunConfig) -> str:
    return "-".join(config.residual_fields)


def experiment_id(config: ResidualRunConfig) -> str:
    """Experiment ID for residual regimes pipeline."""
    # "ward" (pure Ward HRC Z500 parent regimes) gets its own tag so it never collides with
    # existing "kmeans"-parent-regime RR outputs at the same z500_k/final_k — the ward-seeded
    # K-means hybrid (z500_cluster_method default "kmeans") keeps the existing untagged path.
    z500_method = getattr(config, "z500_cluster_method", None) or "kmeans"
    z500_method_tag = "_zward" if z500_method == "ward" else ""
    fields_joined = "+".join(f for f in config.residual_fields if f != "z500")
    # A per-regime "alldays" cache is built from the UNION of fields across every Z500 regime
    # (not just this one regime's own selection), so a bbox with more Z500 regimes (higher k)
    # can accumulate a much longer field list than any single regime ever clusters on. Left
    # unbounded, this pushes filenames built from experiment_id() (which embed it directly)
    # past the 255-byte filesystem NAME_MAX -- surfaced downstream as a confusing
    # PermissionError from netCDF4/HDF5, not a clean "name too long" error. Hash it down past
    # a safe threshold; every field list short enough to have ever worked before stays
    # byte-identical. The threshold was originally 80, calibrated against fields_joined alone
    # -- it didn't account for how much the REST of a caller's filename (prefix + bbox
    # coordinate string + tag_suffix) varies. rr_021's coefstability filename for
    # northamerica/B1/domainQ950 hit fields_joined=76 (under 80, left unhashed) but totaled
    # 260 bytes with that bbox's overhead, crashing on write (OSError: File name too long).
    # 50 leaves real margin (that case needed <=71) against longer bbox strings elsewhere.
    if len(fields_joined) > 50:
        fields_joined = "h" + hashlib.sha1(fields_joined.encode()).hexdigest()[:10]
    return (
        f"rr_z500k{config.z500_k}_z500{fields_joined}"
        f"_w{config.smooth_window:02d}_eps{int(round(config.z500_epsilon)):02d}"
        f"_var{int(round(config.var_threshold * 100)):02d}_{option_label(config)}{z500_method_tag}"
        f"_{config.year_span}_k{config.final_k:02d}"
    )


def _mf_option_tag(config: ResidualRunConfig) -> str:
    """Sigma/Yeo-Johnson tag fragment used in rr_00's 07_MF_PCA filenames, so Stage 2
    finds the correct preprocessing variant (field-mode + Yeo-Johnson, per
    run_configs.yaml) instead of a stale/legacy one."""
    sigma_tag = "sigsc" if str(config.sigma_mode) == "scalar" else "sigf"
    return f"{sigma_tag}_yj" if config.yeo_johnson else sigma_tag


def resolve_mf_pca_path(config: ResidualRunConfig, field_key: str) -> Path | None:
    """Locate rr_00's existing per-field PCA (``07_MF_PCA_...``) for ``field_key``.

    Stage 2 reuses this file directly rather than recomputing a residual-specific
    PCA (see ``residual_regimes/auxiliary.py`` module docstring) — the "07_RR_PCA"
    /scalar-sigma path is legacy. Uses a loose glob (ignores exact eps/var/vo_tag/
    final_k numerics) rather than reconstructing multivariate_regimes' exact
    per-field experiment id, since ``ResidualRunConfig`` doesn't track every MF
    tuning knob bit-for-bit and a glob is more forgiving of that drift. The one
    exception is ``config.area_weight_tag``, which the glob DOES pin exactly (see
    below) so a rerun after a preprocessing fix can't be picked up silently.
    """
    mf_field_dir = (
        config.module_root / "data" / "multivariate" / "processed"
        / config.region / "per_field" / field_key
    )
    if not mf_field_dir.is_dir():
        return None
    tag = _mf_option_tag(config)
    aw_tag = getattr(config, "area_weight_tag", "") or ""
    # area_weight_tag lands at the very end of the filename, right after the k{final_k}
    # component (per_field_experiment_id in multivariate_regimes/paths.py appends it
    # there). Pin the tail exactly rather than a trailing "*.nc" wildcard: with aw_tag=""
    # the pattern must reject files carrying SOME tag (an already-tagged rerun) just as
    # much as it accepts untagged ones, or an unset tag would silently match either.
    tail = f"_k[0-9][0-9]_{aw_tag}.nc" if aw_tag else "_k[0-9][0-9].nc"
    matches = sorted(mf_field_dir.glob(f"07_MF_PCA_{field_key}_mf_{field_key}_*{tag}_step{config.bbox_step}*{tail}"))
    if not matches:
        return None
    w_tag = f"_w{config.smooth_window}_"
    preferred = [m for m in matches if w_tag in m.name]
    return preferred[-1] if preferred else matches[-1]


def resolve_mf_preprocessed_path(config: ResidualRunConfig, field_key: str) -> Path | None:
    """Locate rr_00's per-field pre-PCA preprocessed file (``04_05_06_MF_..._X_matrix``)
    for ``field_key`` — the flattened (n_days, n_space) matrix ``run_per_field_pca`` fits
    its SVD on to produce the ``07_MF_PCA_...`` file ``resolve_mf_pca_path`` resolves.

    Added for ``feature_relevance.py``'s per-fold PCA refit (2026-08-18): scoring a
    field's PCA block honestly under cross-validation means fitting PCA fresh inside
    each fold's training rows only, which needs this raw pre-PCA matrix rather than the
    single record-wide fit baked into ``PCs_reduced``. Mirrors ``resolve_mf_pca_path``'s
    loose-glob/exact-area-weight-tag approach exactly, for the same reason: a glob is
    more forgiving of ``ResidualRunConfig`` not tracking every MF tuning knob bit-for-bit,
    except ``area_weight_tag``, pinned exactly so a rerun after a preprocessing fix can't
    be picked up silently.
    """
    mf_field_dir = (
        config.module_root / "data" / "multivariate" / "preprocessed"
        / config.region / field_key
    )
    if not mf_field_dir.is_dir():
        return None
    tag = _mf_option_tag(config)
    aw_tag = getattr(config, "area_weight_tag", "") or ""
    tail = f"_k[0-9][0-9]_{aw_tag}.nc" if aw_tag else "_k[0-9][0-9].nc"
    matches = sorted(mf_field_dir.glob(f"04_05_06_MF_{field_key}_preprocessed_mf_{field_key}_*{tag}_step{config.bbox_step}*{tail}"))
    if not matches:
        return None
    w_tag = f"_w{config.smooth_window}_"
    preferred = [m for m in matches if w_tag in m.name]
    return preferred[-1] if preferred else matches[-1]


def resolve_mf_anomaly_path(config: ResidualRunConfig, field_key: str) -> Path | None:
    """Locate rr_00's per-field pre-flatten anomaly file (``03_MF_..._anomaly``) for
    ``field_key`` — the still-gridded (lat/lon) anomaly ``run_field_preprocessing``
    produces BEFORE step 4-6's area-weight-and-flatten into ``X_matrix``.

    Added for ``feature_relevance.py``'s ``--field-representation domain_mean`` A/B test
    (2026-08-19): a fair domain-mean comparison for non-Z500 fields needs to run the SAME
    ``_domain_mean_anomaly`` area-weighted-spatial-mean Z500 already uses, on a real
    lat/lon field — not on ``X_matrix``, whose values are already area-weighted for PCA
    (``z_weighted`` in ``compute_step6_flatten``) and would double up / mis-weight a plain
    mean over its flattened space dimension. Mirrors ``resolve_mf_pca_path``'s
    loose-glob/exact-area-weight-tag approach.
    """
    mf_field_dir = (
        config.module_root / "data" / "multivariate" / "preprocessed"
        / config.region / field_key
    )
    if not mf_field_dir.is_dir():
        return None
    tag = _mf_option_tag(config)
    aw_tag = getattr(config, "area_weight_tag", "") or ""
    tail = f"_k[0-9][0-9]_{aw_tag}.nc" if aw_tag else "_k[0-9][0-9].nc"
    matches = sorted(mf_field_dir.glob(f"03_MF_{field_key}_anomaly_mf_{field_key}_*{tag}_step{config.bbox_step}*{tail}"))
    if not matches:
        return None
    w_tag = f"_w{config.smooth_window}_"
    preferred = [m for m in matches if w_tag in m.name]
    return preferred[-1] if preferred else matches[-1]


def per_regime_fields_signature(config: ResidualRunConfig) -> str:
    """Short hash of the per-regime field map only (no k spec), so two different
    per-regime configurations sharing the same residual_fields union (and hence the
    same `experiment_id()`) never collide on disk. Empty string when per_regime_mode
    is off (global-mode paths are untouched by this).

    Used for the pre-k Stage-2 artifacts (joint features, Ward linkage) that depend
    only on which fields/days go into them, not on how many clusters we later cut
    them into — so a k-selection sweep (many k_specs over the same fields) and the
    eventual production run (one explicit k_spec) share these files instead of each
    recomputing Ward HC from scratch. See `per_regime_signature` for the k-dependent
    counterpart used for the final labeled-regime artifacts.
    """
    if not config.per_regime_mode:
        return ""
    payload = json.dumps(
        {
            "fields": {str(k): list(v) for k, v in sorted((config.per_regime_field_map or {}).items())},
            # Distinguishes cross-bbox diagnostics (config.z500_regimes_override) that
            # happen to share z500_k with some other run — experiment_id() alone only
            # differentiates by z500_k, which is not unique to a given override source.
            "z500_regimes_override": str(config.z500_regimes_override or ""),
            "z500_cluster_method": getattr(config, "z500_cluster_method", None) or "kmeans",
            # Residualized (Option 2, regime-mean-subtracted) and non-residualized joint
            # features are NOT the same data — must never share a cache entry, or a run with
            # --residualize would silently reuse a stale pre-fix (or vice versa) file.
            "residualize": bool(getattr(config, "residualize", False)),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def per_regime_signature(config: ResidualRunConfig) -> str:
    """Short hash of the per-regime field map + k spec + final_k_range, so two
    different per-regime configurations sharing the same residual_fields union
    (and hence the same `experiment_id()`) never collide on disk. Empty string
    when per_regime_mode is off (global-mode paths are untouched by this).

    Used for the k-dependent final-regime artifacts. Pre-k artifacts (joint
    features, Ward linkage) use `per_regime_fields_signature` instead, so they
    stay shared across different k_specs over the same fields.
    """
    if not config.per_regime_mode:
        return ""
    payload = json.dumps(
        {
            "fields": {str(k): list(v) for k, v in sorted((config.per_regime_field_map or {}).items())},
            "k": {str(k): v for k, v in sorted((config.per_regime_k_spec or {}).items())},
            "final_k_range": list(config.final_k_range),
            # Distinguishes cross-bbox diagnostics (config.z500_regimes_override) that
            # happen to share z500_k with some other run — experiment_id() alone only
            # differentiates by z500_k, which is not unique to a given override source.
            "z500_regimes_override": str(config.z500_regimes_override or ""),
            "z500_cluster_method": getattr(config, "z500_cluster_method", None) or "kmeans",
            # See per_regime_fields_signature — residualized and non-residualized final-regime
            # artifacts must never collide either.
            "residualize": bool(getattr(config, "residualize", False)),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class ResidualWorkflowPaths:
    """Directory structure for residual regimes pipeline.

    Uses residual_regimes/ output directories instead of multivariate/.
    Reuses preprocessed fields from multivariate/preprocessed/.
    """

    config: ResidualRunConfig

    @property
    def raw_root(self) -> Path:
        """Raw ERA5 data (shared with multivariate)."""
        return self.config.module_root / "data" / "multivariate" / "raw" / "era5"

    @property
    def preprocessed_root(self) -> Path:
        """Preprocessed anomalies (shared with multivariate)."""
        return self.config.module_root / "data" / "multivariate" / "preprocessed" / self.config.region

    @property
    def processed_root(self) -> Path:
        """PCA and residuals (residual-specific)."""
        return self.config.module_root / "data" / "stage2_residual" / "processed" / self.config.region

    @property
    def residuals_dir(self) -> Path:
        """Stage 1 Z500 regimes and computed residuals."""
        return self.processed_root / "residuals"

    @property
    def joint_features_dir(self) -> Path:
        """Joint features for Stage 2."""
        return self.processed_root / "joint_features_stage2"

    @property
    def clustering_root(self) -> Path:
        """Stage 2 clustering results."""
        return self.config.module_root / "results" / "stage2_residual" / "clustering" / self.config.region

    @property
    def regimes_root(self) -> Path:
        """Stage 2 regime characterisation."""
        return self.config.module_root / "results" / "stage2_residual" / "regimes" / self.config.region

    @property
    def diagnostics_root(self) -> Path:
        """Manifests and diagnostics."""
        return (
            self.config.module_root
            / "results"
            / "stage2_residual"
            / "diagnostics"
            / self.config.region
        )

    @property
    def figures_root(self) -> Path:
        return self.config.module_root / "figures" / "stage2_residual" / self.config.region

    @property
    def logs_root(self) -> Path:
        return self.config.module_root / "logs" / "stage2_residual"

    def raw_field_dir(self, field_key: str) -> Path:
        return self.raw_root / field_key / self.config.region

    def preprocessed_field_dir(self, field_key: str) -> Path:
        return self.preprocessed_root / field_key

    def processed_field_dir(self, field_key: str) -> Path:
        return self.processed_root / "per_field" / field_key

    def field_anomaly_path(self, field_key: str) -> Path:
        """Anomaly file from multivariate preprocessing."""
        return self.preprocessed_field_dir(field_key) / f"03_MF_{field_key}_anomaly_{experiment_id(self.config)}.nc"

    def field_preprocessed_path(self, field_key: str) -> Path:
        """Preprocessed file from multivariate (normalized, detrended)."""
        return self.preprocessed_field_dir(field_key) / f"04_05_06_MF_{field_key}_preprocessed_{experiment_id(self.config)}.nc"

    def field_pca_path(self, field_key: str) -> Path:
        """PCA components for this field."""
        return self.processed_field_dir(field_key) / f"07_RR_PCA_{field_key}_{experiment_id(self.config)}.nc"

    @property
    def z500_regimes_stage1_path(self) -> Path:
        """Stage 1: Z500 circulation regimes (k-means)."""
        return self.residuals_dir / f"08_RR_z500_regimes_k{self.config.z500_k}_{experiment_id(self.config)}.nc"

    @property
    def residuals_path(self) -> Path:
        """Residual anomalies for all fields after Stage 1."""
        return self.residuals_dir / f"08a_RR_residuals_all_fields_{experiment_id(self.config)}.nc"

    @property
    def joint_features_path(self) -> Path:
        """Joint features from residuals for Stage 2 clustering."""
        return self.joint_features_dir / f"08b_RR_joint_features_stage2_{experiment_id(self.config)}.nc"

    @property
    def ward_hc_path(self) -> Path:
        """Ward hierarchical clustering on residual joint features."""
        return self.joint_features_dir / f"08b_RR_HC_Ward_{experiment_id(self.config)}.nc"

    @property
    def regimes_path(self) -> Path:
        """Stage 2: Final k-means clustering of residuals."""
        return self.clustering_root / f"09_RR_regimes_kmeans_k{self.config.final_k:02d}_{experiment_id(self.config)}.nc"

    @property
    def regimes_filtered_path(self) -> Path:
        """Stage 2 labels after rare-regime merging (< min_days reassigned)."""
        return self.clustering_root / f"09b_RR_regimes_filtered_k{self.config.final_k:02d}_{experiment_id(self.config)}.nc"

    @property
    def regime_characterisation_path(self) -> Path:
        """Regime composites and statistics."""
        return self.regimes_root / (
            f"10_11_RR_regime_reconstruction_characterisation_kmeans_k{self.config.final_k:02d}_{experiment_id(self.config)}.nc"
        )

    @property
    def manifest_path(self) -> Path:
        return self.diagnostics_root / f"manifest_{experiment_id(self.config)}.json"

    # -- Per-Z500-regime Stage 2 (per_regime_mode) --------------------------------
    # All new filenames, suffixed `_perregime_Z{z}_{sig}_{experiment_id}` — never
    # collide with the global-mode paths above regardless of config.

    def per_region_joint_features_alldays_path(self, z: int) -> Path:
        """Full day-range joint features for Z500 regime z's own field subset
        (day-subsetting happens after this, in per_region_joint_features_path).

        Keyed by `per_regime_fields_signature` (fields only, not k_spec) so a
        k-selection sweep and the eventual production run share this file."""
        sig = per_regime_fields_signature(self.config)
        return self.joint_features_dir / (
            f"08b_RR_joint_features_perregime_Z{z}_alldays_{sig}_{experiment_id(self.config)}.nc"
        )

    def per_region_joint_features_path(self, z: int) -> Path:
        """Joint features for Z500 regime z, sliced to that regime's own days.

        Keyed by `per_regime_fields_signature` (fields only, not k_spec) so a
        k-selection sweep and the eventual production run share this file."""
        sig = per_regime_fields_signature(self.config)
        return self.joint_features_dir / (
            f"08b_RR_joint_features_perregime_Z{z}_{sig}_{experiment_id(self.config)}.nc"
        )

    def per_region_ward_hc_path(self, z: int) -> Path:
        """Ward hierarchical clustering for Z500 regime z's own residual subset.

        Keyed by `per_regime_fields_signature` (fields only, not k_spec) so a
        k-selection sweep and the eventual production run share this file."""
        sig = per_regime_fields_signature(self.config)
        return self.joint_features_dir / (
            f"08b_RR_HC_Ward_perregime_Z{z}_{sig}_{experiment_id(self.config)}.nc"
        )

    def per_region_local_regimes_path(self, z: int) -> Path:
        """Raw local k-means output for Z500 regime z (before merge across regimes)."""
        sig = per_regime_signature(self.config)
        return self.clustering_root / (
            f"09_RR_regimes_kmeans_perregime_Z{z}_{sig}_{experiment_id(self.config)}.nc"
        )

    # -- rr_021 subregime-k selection diagnostics ---------------------------------

    def per_region_k_sweep_scratch_path(self, z: int, k: int) -> Path:
        """Scratch k-means output for one (Z500 regime, candidate k) pair explored
        by the rr_021 subregime-k sweep. Never `per_region_local_regimes_path` —
        exploring many candidate k's must never risk colliding with or overwriting
        a chosen production fit. Keyed by fields only (not k_spec), matching the
        other pre-decision per-regime artifacts."""
        sig = per_regime_fields_signature(self.config)
        return self.diagnostics_root / "rr021_k_sweep" / (
            f"rr021_kmeans_scratch_Z{z}_k{k:02d}_{sig}_{experiment_id(self.config)}.nc"
        )

    def per_region_k_sweep_csv_path(self, z: int) -> Path:
        """Per-regime k-sweep diagnostic CSV (k, silhouette, inertia, min_cluster_frac, ...)."""
        sig = per_regime_fields_signature(self.config)
        return self.diagnostics_root / f"rr021_k_sweep_Z{z}_{sig}_{experiment_id(self.config)}.csv"

    @property
    def k_sweep_figure_path(self) -> Path:
        """Combined silhouette/inertia-vs-k figure, faceted by Z500 regime."""
        sig = per_regime_fields_signature(self.config)
        return self.figures_root / f"rr021_k_sweep_{sig}_{experiment_id(self.config)}.png"

    @property
    def k_sweep_full_csv_path(self) -> Path:
        """All candidate k's for all Z500 regimes in one table (unlike
        `per_region_k_sweep_csv_path`, one file per regime) — for comparing how
        close the silhouette/ARI runner-ups were to the suggested k, across
        regimes, without opening every per-regime CSV separately."""
        sig = per_regime_fields_signature(self.config)
        return self.diagnostics_root / f"rr021_k_sweep_full_{sig}_{experiment_id(self.config)}.csv"

    @property
    def k_sweep_local_maxima_csv_path(self) -> Path:
        """One row per (Z500 regime, local silhouette maximum k) — every
        candidate 'second peak' k, not just the auto-suggested one, with its
        already-computed ARI stability check. Complements
        `k_sweep_summary_csv_path` (suggested k only)."""
        sig = per_regime_fields_signature(self.config)
        return self.diagnostics_root / f"rr021_k_sweep_local_maxima_{sig}_{experiment_id(self.config)}.csv"

    @property
    def k_sweep_summary_csv_path(self) -> Path:
        """One-row-per-Z500-regime summary table (fields, suggested k, silhouette,
        inertia, mean_ari, verdict) — the finalist row from each regime's k-sweep,
        for a quick cross-regime comparison without opening every per-regime CSV."""
        sig = per_regime_fields_signature(self.config)
        return self.diagnostics_root / f"rr021_k_sweep_summary_{sig}_{experiment_id(self.config)}.csv"

    @property
    def k_overrides_yaml_path(self) -> Path:
        """Human-reviewed per-regime subregime-k YAML written by rr_021, consumed by
        rr_022 in place of the old runtime "auto" silhouette sweep.

        Residualize-tagged (``_resid`` suffix when ``config.residualize`` is set) so a
        residualized k-selection run never overwrites (or gets silently read in place
        of) the non-residualized one — mirrors per_regime_fields_signature's reasoning.
        The rr_021/rr_022 shell launchers must reconstruct this exact suffix themselves
        (they read/write the path by hardcoded pattern, not via this property).
        """
        resid_tag = "_resid" if getattr(self.config, "residualize", False) else ""
        return (
            self.config.module_root / "src" / "common"
            / f"rr_k_overrides_{self.config.region}_{self.config.bbox_id}{resid_tag}.yaml"
        )

    @property
    def regimes_path_per_regime(self) -> Path:
        """Merged (contiguous 1..K_total) per-regime Stage 2 output."""
        sig = per_regime_signature(self.config)
        return self.clustering_root / f"09_RR_regimes_perregime_{sig}_{experiment_id(self.config)}.nc"

    @property
    def regimes_filtered_path_per_regime(self) -> Path:
        """Merged per-regime output after rare-regime merging (< min_days reassigned)."""
        sig = per_regime_signature(self.config)
        return self.clustering_root / f"09b_RR_regimes_filtered_perregime_{sig}_{experiment_id(self.config)}.nc"

    @property
    def regime_characterisation_path_per_regime(self) -> Path:
        """Regime composites and statistics for the merged per-regime output."""
        sig = per_regime_signature(self.config)
        return self.regimes_root / (
            f"10_11_RR_regime_reconstruction_characterisation_perregime_{sig}_{experiment_id(self.config)}.nc"
        )

    # -- Per-Z500-regime Stage 2, LOGISTIC production path (see stage2_logistic.py) --
    # Continuous per-day predicted-extreme-probability index, replacing the discrete
    # k-means regime_labels above (README_ResidualRegimes.md "Main Finding
    # (2026-07-29)"). Keyed by `per_regime_fields_signature` only (no k_spec — there
    # is no cluster count here) plus an opt-in `output_tag`, since the result also
    # depends on which extreme-labels file/var was used (e.g. a domain_pctl_q
    # sensitivity sweep), which the signature does not encode.

    def per_region_logistic_index_path(self, z: int, *, output_tag: str = "") -> Path:
        """Raw single-Z500-regime logistic fit (before merge across regimes)."""
        sig = per_regime_fields_signature(self.config)
        tag_suffix = f"_{output_tag}" if output_tag else ""
        return self.clustering_root / (
            f"09_RR_logistic_perregime_Z{z}_{sig}_{experiment_id(self.config)}{tag_suffix}.nc"
        )

    def logistic_index_path_per_regime(self, *, output_tag: str = "") -> Path:
        """Merged (all Z500 regimes) continuous predicted-extreme-probability index."""
        sig = per_regime_fields_signature(self.config)
        tag_suffix = f"_{output_tag}" if output_tag else ""
        return self.clustering_root / (
            f"09_RR_logistic_perregime_merged_{sig}_{experiment_id(self.config)}{tag_suffix}.nc"
        )

    def logistic_index_summary_csv_path(self, *, output_tag: str = "") -> Path:
        """One-row-per-Z500-regime summary (n_days, n_extreme, base_rate, fields)."""
        sig = per_regime_fields_signature(self.config)
        tag_suffix = f"_{output_tag}" if output_tag else ""
        return self.diagnostics_root / (
            f"rr_logistic_stage2_summary_{sig}_{experiment_id(self.config)}{tag_suffix}.csv"
        )
