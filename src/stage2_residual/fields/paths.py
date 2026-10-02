from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.preprocessing_helpers import bbox_step_filename_suffix

from .config import MultivariateRunConfig
from .field_specs import is_standard_multivariate_field_set


def _sigma_label(sigma_mode: str) -> str:
    return "sigsc" if str(sigma_mode) == "scalar" else "sigf"


def _detrend_label(detrend_method: str) -> str:
    return "dm" if str(detrend_method) == "domain_mean" else "gp"


def _clim_label(clim_subtract: str) -> str:
    return "clims" if str(clim_subtract) == "smooth" else "climr"


def option_label(config: MultivariateRunConfig) -> str:
    base = f"detr{_detrend_label(config.detrend_method)}_{_clim_label(config.climatology_subtract)}_{_sigma_label(config.sigma_mode)}"
    if config.yeo_johnson:
        base = f"{base}_yj"
    if config.bbox:
        return f"{base}_stepcustom"
    return base + bbox_step_filename_suffix(config.module_root, config.region, config.bbox_step)


def field_list_slug(config: MultivariateRunConfig) -> str:
    return "-".join(config.source_fields)


def experiment_id(config: MultivariateRunConfig) -> str:
    """Full experiment ID including the source field list. Used for joint features,
    clustering, and regime outputs that are specific to a particular field combination."""
    vo_tag = (
        f"vo850g{config.vo850_gaussian_sigma:g}"
        if "vo850" in config.source_fields and config.vo850_smoothing_method == "gaussian"
        else f"vo850c{config.vo850_coarsen_factor:d}"
        if "vo850" in config.source_fields and config.vo850_smoothing_method == "coarsen"
        else "vo850none"
        if "vo850" in config.source_fields
        else "novo850"
    )
    regime_tag = "z500vo850tcwv" if is_standard_multivariate_field_set(config.source_fields) else field_list_slug(config)
    return (
        f"mf_{regime_tag}_w{config.smooth_window:02d}_eps{int(round(config.z500_epsilon)):02d}"
        f"_var{int(round(config.var_threshold * 100)):02d}_{option_label(config)}_{config.year_span}_{vo_tag}_k{config.final_k:02d}"
    )


def per_field_experiment_id(config: MultivariateRunConfig, field_key: str) -> str:
    """Per-field preprocessing ID — excludes the source field list so that anomaly,
    preprocessed, and PCA files are reused across different field-set experiments.
    Adding new fields to SOURCE_FIELDS no longer forces reprocessing of existing fields.
    Matches the naming of single-field legacy files (mf_<field>_w11_...).

    Shared by field_anomaly_path (03), field_preprocessed_path (04_05_06), and
    field_pca_path (07) below, so config.area_weight_tag flows into all three
    filenames from this one spot."""
    vo_tag = (
        f"vo850g{config.vo850_gaussian_sigma:g}"
        if config.vo850_smoothing_method == "gaussian"
        else f"vo850c{config.vo850_coarsen_factor:d}"
        if config.vo850_smoothing_method == "coarsen"
        else "vo850none"
    )
    tag_suffix = f"_{config.area_weight_tag}" if config.area_weight_tag else ""
    return (
        f"mf_{field_key}_w{config.smooth_window:02d}_eps{int(round(config.z500_epsilon)):02d}"
        f"_var{int(round(config.var_threshold * 100)):02d}_{option_label(config)}_{config.year_span}_{vo_tag}_k{config.final_k:02d}{tag_suffix}"
    )


@dataclass(frozen=True)
class WorkflowPaths:
    config: MultivariateRunConfig

    @property
    def raw_root(self) -> Path:
        return self.config.module_root / "data" / "multivariate" / "raw" / "era5"

    @property
    def preprocessed_root(self) -> Path:
        return self.config.module_root / "data" / "multivariate" / "preprocessed" / self.config.region

    @property
    def processed_root(self) -> Path:
        return self.config.module_root / "data" / "multivariate" / "processed" / self.config.region

    @property
    def joint_features_dir(self) -> Path:
        return self.processed_root / "joint_features"

    @property
    def clustering_root(self) -> Path:
        return self.config.module_root / "results" / "multivariate" / "clustering" / self.config.region

    @property
    def regimes_root(self) -> Path:
        return self.config.module_root / "results" / "multivariate" / "regimes" / self.config.region

    @property
    def diagnostics_root(self) -> Path:
        """Manifests, baseline compare, and Alpine VO850 check outputs."""
        return (
            self.config.module_root
            / "results"
            / "summary_reports_multivariate"
            / "diagnostics"
            / self.config.region
        )

    @property
    def figures_root(self) -> Path:
        return self.config.module_root / "figures" / "multivariate" / self.config.region

    @property
    def logs_root(self) -> Path:
        return self.config.module_root / "logs" / "multivariate"

    def raw_field_dir(self, field_key: str) -> Path:
        return self.raw_root / field_key / self.config.region

    def preprocessed_field_dir(self, field_key: str) -> Path:
        return self.preprocessed_root / field_key

    def processed_field_dir(self, field_key: str) -> Path:
        return self.processed_root / "per_field" / field_key

    def field_anomaly_path(self, field_key: str) -> Path:
        return self.preprocessed_field_dir(field_key) / f"03_MF_{field_key}_anomaly_{per_field_experiment_id(self.config, field_key)}.nc"

    def field_preprocessed_path(self, field_key: str) -> Path:
        return self.preprocessed_field_dir(field_key) / f"04_05_06_MF_{field_key}_preprocessed_{per_field_experiment_id(self.config, field_key)}.nc"

    def field_pca_path(self, field_key: str) -> Path:
        return self.processed_field_dir(field_key) / f"07_MF_PCA_{field_key}_{per_field_experiment_id(self.config, field_key)}.nc"

    @property
    def joint_features_path(self) -> Path:
        return self.joint_features_dir / f"08_MF_joint_features_{experiment_id(self.config)}.nc"

    @property
    def ward_hc_path(self) -> Path:
        return self.joint_features_dir / f"08_MF_HC_Ward_{experiment_id(self.config)}.nc"

    @property
    def regimes_path(self) -> Path:
        return self.clustering_root / f"09_MF_regimes_kmeans_k{self.config.final_k:02d}_{experiment_id(self.config)}.nc"

    @property
    def regime_characterisation_path(self) -> Path:
        return self.regimes_root / (
            f"10_11_MF_regime_reconstruction_characterisation_kmeans_k{self.config.final_k:02d}_{experiment_id(self.config)}.nc"
        )

    @property
    def baseline_compare_path(self) -> Path:
        return self.diagnostics_root / f"12_MF_baseline_compare_{experiment_id(self.config)}.nc"

    @property
    def manifest_path(self) -> Path:
        return self.diagnostics_root / f"manifest_{experiment_id(self.config)}.json"
