from __future__ import annotations

"""One-stop helper module for the parallel multivariate workflow.

This file intentionally aggregates the main functions and dataclasses used by
the multifield pipeline so the workflow can be followed from a single module.
The implementation is still delegated to the specialized modules, but this
module acts as the main auxiliary entrypoint for imports and orchestration.
"""

from dataclasses import asdict
from pathlib import Path
import time

from .cluster import run_kmeans_regimes, run_ward_hierarchical_clustering, write_regime_characterisation
from .compare_baseline import write_baseline_comparison
from .config import MultivariateRunConfig, build_arg_parser, config_from_args
from .download import build_request, main as download_main, parse_years, resolve_out_dir, years_filename_slug
from .field_specs import (
    ERA5_TCWV,
    ERA5_VO500,
    ERA5_VO850,
    ERA5_Z500,
    FieldSpec,
    STANDARD_MULTIVARIATE_REGIME_SET,
    default_field_order,
    get_field_spec,
)
from .joint_features import build_joint_feature_dataset
from .paths import WorkflowPaths, experiment_id, field_list_slug, option_label
from .per_field_pca import PerFieldPcaResult, run_per_field_pca
from .preprocess import (
    FieldPreprocessResult,
    build_step4_6_from_anomaly,
    explicit_raw_nc_for_field,
    run_field_preprocessing,
    run_step4_6_from_existing_anomaly,
    select_field_raw_paths,
)
from .provenance import FieldManifest, write_manifest
from .reuse import LegacyZ500Artifacts, resolve_legacy_z500_artifacts
from .smoothing import apply_spatial_smoothing

__all__ = [
    "ERA5_TCWV",
    "ERA5_VO500",
    "ERA5_VO850",
    "ERA5_Z500",
    "STANDARD_MULTIVARIATE_REGIME_SET",
    "FieldManifest",
    "FieldPreprocessResult",
    "FieldSpec",
    "LegacyZ500Artifacts",
    "MultivariateRunConfig",
    "PerFieldPcaResult",
    "WorkflowPaths",
    "apply_spatial_smoothing",
    "build_arg_parser",
    "build_joint_feature_dataset",
    "build_request",
    "build_step4_6_from_anomaly",
    "config_from_args",
    "default_field_order",
    "download_main",
    "experiment_id",
    "field_list_slug",
    "get_field_spec",
    "main",
    "option_label",
    "parse_years",
    "resolve_legacy_z500_artifacts",
    "resolve_out_dir",
    "run_field_preprocessing",
    "run_kmeans_regimes",
    "run_per_field_pca",
    "run_pipeline",
    "run_step4_6_from_existing_anomaly",
    "run_ward_hierarchical_clustering",
    "select_field_raw_paths",
    "write_baseline_comparison",
    "write_manifest",
    "write_regime_characterisation",
    "years_filename_slug",
]


def run_pipeline() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    paths = WorkflowPaths(config)

    t0 = time.perf_counter()

    def stamp(msg: str) -> None:
        dt = time.perf_counter() - t0
        print(f"[multifield | +{dt:7.1f}s] {msg}")

    manifest_fields: list[FieldManifest] = []
    anomaly_paths_by_field: dict[str, str] = {}
    pca_paths_by_field: dict[str, Path] = {}
    outputs: dict[str, str] = {}

    legacy = resolve_legacy_z500_artifacts(config)
    stamp(f"Regime set: {config.regime_set_label}")
    stamp(f"Legacy Z500 reuse tier: {legacy.tier}")

    if legacy.tier == "step7" and legacy.pca_path is not None:
        pca_paths_by_field["z500"] = legacy.pca_path
        if legacy.anomaly_path is None:
            stamp("Legacy Step 7 PCA matched but no anomaly path was validated; recomputing Z500 anomaly locally.")
            z500_result = run_field_preprocessing(
                config,
                field_key="z500",
                paths=paths,
                raw_nc=config.z500_raw_nc,
            )
            anomaly_paths_by_field["z500"] = str(z500_result.anomaly_path)
        else:
            anomaly_paths_by_field["z500"] = str(legacy.anomaly_path)
        manifest_fields.append(
            FieldManifest(
                field_key="z500",
                status="reused",
                reuse_tier="step7",
                source_path=str(legacy.pca_path),
                output_paths=[str(legacy.pca_path)],
                notes=list(legacy.notes),
                retained_modes=None,
            )
        )
    elif legacy.tier == "step4_6" and legacy.preprocessed_path is not None:
        stamp(f"Running multivariate Z500 PCA from legacy Step 4-6 file {legacy.preprocessed_path.name}")
        pca_result = run_per_field_pca(
            field_key="z500",
            in_preprocessed_path=legacy.preprocessed_path,
            out_pca_path=paths.field_pca_path("z500"),
            var_threshold=config.var_threshold,
            max_modes=config.max_modes_per_field,
            overwrite=config.overwrite,
        )
        pca_paths_by_field["z500"] = pca_result.pca_path
        if legacy.anomaly_path is None:
            z500_result = run_field_preprocessing(
                config,
                field_key="z500",
                paths=paths,
                raw_nc=config.z500_raw_nc,
            )
            anomaly_paths_by_field["z500"] = str(z500_result.anomaly_path)
        else:
            anomaly_paths_by_field["z500"] = str(legacy.anomaly_path)
        manifest_fields.append(
            FieldManifest(
                field_key="z500",
                status="reused",
                reuse_tier="step4_6",
                source_path=str(legacy.preprocessed_path),
                output_paths=[str(pca_result.pca_path)],
                notes=list(legacy.notes),
                retained_modes=pca_result.retained_modes,
            )
        )
    elif legacy.tier == "step3" and legacy.anomaly_path is not None:
        stamp(f"Building multivariate Z500 Step 4-7 from legacy Step 3 anomaly {legacy.anomaly_path.name}")
        z500_result = run_step4_6_from_existing_anomaly(
            config,
            field_key="z500",
            in_anomaly_path=legacy.anomaly_path,
            paths=paths,
        )
        pca_result = run_per_field_pca(
            field_key="z500",
            in_preprocessed_path=z500_result.preprocessed_path,
            out_pca_path=paths.field_pca_path("z500"),
            var_threshold=config.var_threshold,
            max_modes=config.max_modes_per_field,
            overwrite=config.overwrite,
        )
        pca_paths_by_field["z500"] = pca_result.pca_path
        anomaly_paths_by_field["z500"] = str(z500_result.anomaly_path)
        manifest_fields.append(
            FieldManifest(
                field_key="z500",
                status="reused",
                reuse_tier="step3",
                source_path=str(legacy.anomaly_path),
                output_paths=[str(z500_result.preprocessed_path), str(pca_result.pca_path)],
                notes=list(legacy.notes),
                retained_modes=pca_result.retained_modes,
            )
        )
    else:
        stamp("No compatible legacy Z500 artifact found; running Z500 preprocessing from raw.")
        z500_result = run_field_preprocessing(
            config,
            field_key="z500",
            paths=paths,
            raw_nc=config.z500_raw_nc,
        )
        pca_result = run_per_field_pca(
            field_key="z500",
            in_preprocessed_path=z500_result.preprocessed_path,
            out_pca_path=paths.field_pca_path("z500"),
            var_threshold=config.var_threshold,
            max_modes=config.max_modes_per_field,
            overwrite=config.overwrite,
        )
        pca_paths_by_field["z500"] = pca_result.pca_path
        anomaly_paths_by_field["z500"] = str(z500_result.anomaly_path)
        manifest_fields.append(
            FieldManifest(
                field_key="z500",
                status="recomputed",
                reuse_tier="none",
                output_paths=[str(z500_result.anomaly_path), str(z500_result.preprocessed_path), str(pca_result.pca_path)],
                notes=list(legacy.notes),
                retained_modes=pca_result.retained_modes,
            )
        )

    for field_key in config.source_fields:
        if field_key == "z500":
            continue
        field_spec = get_field_spec(field_key)
        stamp(f"Running {field_spec.output_prefix} preprocessing and per-field PCA.")
        field_result = run_field_preprocessing(
            config,
            field_key=field_key,
            paths=paths,
            raw_nc=explicit_raw_nc_for_field(config, field_key),
        )
        field_pca = run_per_field_pca(
            field_key=field_key,
            in_preprocessed_path=field_result.preprocessed_path,
            out_pca_path=paths.field_pca_path(field_key),
            var_threshold=config.var_threshold,
            max_modes=config.max_modes_per_field,
            overwrite=config.overwrite,
        )
        pca_paths_by_field[field_key] = field_pca.pca_path
        anomaly_paths_by_field[field_key] = str(field_result.anomaly_path)
        smoothing = None
        if field_key == "vo850":
            smoothing = {
                "method": config.vo850_smoothing_method,
                "gaussian_sigma": config.vo850_gaussian_sigma,
                "coarsen_factor": config.vo850_coarsen_factor,
            }
        manifest_fields.append(
            FieldManifest(
                field_key=field_key,
                status="reused" if field_result.reused else "recomputed",
                reuse_tier="none",
                output_paths=[
                    str(field_result.anomaly_path),
                    str(field_result.preprocessed_path),
                    str(field_pca.pca_path),
                ],
                smoothing=smoothing,
                retained_modes=field_pca.retained_modes,
            )
        )

    if config.stop_after_step >= 8:
        stamp("Building joint standardized PC features and Ward hierarchy.")
        joint_path = build_joint_feature_dataset(
            pca_paths_by_field=pca_paths_by_field,
            out_path=paths.joint_features_path,
            overwrite=config.overwrite,
            regime_set_label=config.regime_set_label,
            regime_set_id=config.regime_set_id,
        )
        hc_path = run_ward_hierarchical_clustering(
            joint_features_path=joint_path,
            out_hc_path=paths.ward_hc_path,
            overwrite=config.overwrite,
            regime_set_label=config.regime_set_label,
            regime_set_id=config.regime_set_id,
        )
        outputs["joint_features"] = str(joint_path)
        outputs["ward_hc"] = str(hc_path)

    if config.stop_after_step >= 9:
        stamp(f"Running final k-means clustering with k={config.final_k}.")
        regimes_path = run_kmeans_regimes(
            joint_features_path=paths.joint_features_path,
            out_regimes_path=paths.regimes_path,
            final_k=config.final_k,
            random_state=config.random_state,
            overwrite=config.overwrite,
            regime_set_label=config.regime_set_label,
            regime_set_id=config.regime_set_id,
            ward_hc_path=paths.ward_hc_path,
        )
        characterisation_path = write_regime_characterisation(
            regimes_path=regimes_path,
            anomaly_paths_by_field={k: Path(v) for k, v in anomaly_paths_by_field.items()},
            out_path=paths.regime_characterisation_path,
            overwrite=config.overwrite,
        )
        outputs["regimes"] = str(regimes_path)
        outputs["regime_characterisation"] = str(characterisation_path)

    if config.stop_after_step >= 10:
        stamp("Comparing multifield run against legacy Z500 baseline outputs.")
        compare_path = write_baseline_comparison(
            multifield_regimes_path=paths.regimes_path,
            baseline_regimes_path=legacy.regimes_path,
            multifield_joint_features_path=paths.joint_features_path,
            baseline_pca_path=legacy.pca_path,
            out_path=paths.baseline_compare_path,
            overwrite=config.overwrite,
        )
        if compare_path is not None:
            outputs["baseline_compare"] = str(compare_path)

    outputs["manifest"] = str(paths.manifest_path)
    config_payload = asdict(config)
    config_payload["regime_set_label"] = config.regime_set_label
    config_payload["regime_set_id"] = config.regime_set_id
    write_manifest(
        paths.manifest_path,
        config=config_payload,
        fields=manifest_fields,
        outputs=outputs,
        regime_set_label=config.regime_set_label,
        regime_set_id=config.regime_set_id,
    )
    stamp("Multivariate workflow completed.")


def main() -> None:
    run_pipeline()
