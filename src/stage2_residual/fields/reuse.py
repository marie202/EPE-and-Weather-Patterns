from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import xarray as xr

from common.preprocessing_helpers import bbox_step_filename_suffix

from .config import MultivariateRunConfig


@dataclass(frozen=True)
class LegacyZ500Artifacts:
    tier: str
    anomaly_path: Path | None = None
    preprocessed_path: Path | None = None
    pca_path: Path | None = None
    hc_path: Path | None = None
    regimes_path: Path | None = None
    step1011_path: Path | None = None
    notes: tuple[str, ...] = ()


def _opt_label(config: MultivariateRunConfig) -> str:
    detr_label = "dm" if config.detrend_method == "domain_mean" else "gp"
    clim_label = "clims" if config.climatology_subtract == "smooth" else "climr"
    sigma_label = "sigsc" if config.sigma_mode == "scalar" else "sigf"
    return (
        f"detr{detr_label}_{clim_label}_{sigma_label}"
        + bbox_step_filename_suffix(config.module_root, config.region, config.bbox_step)
    )


def _legacy_expected(config: MultivariateRunConfig) -> dict[str, str]:
    return {
        "region": config.region,
        "bbox_step": config.bbox_step,
        "year_span": config.year_span,
        "detrend_method": config.detrend_method,
        "climatology_subtract": config.climatology_subtract,
        "sigma_mode": config.sigma_mode,
    }


def _readable_and_matching(path: Path, *, expected_attrs: dict[str, str]) -> bool:
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path, chunks=None) as ds:
            attrs = ds.attrs
            for key, expected in expected_attrs.items():
                if key in attrs and str(attrs[key]) != str(expected):
                    return False
    except Exception:
        return False
    return True


def _legacy_preprocessed_dir(config: MultivariateRunConfig) -> Path:
    return config.module_root / "data" / "preprocessed" / config.region


def _legacy_processed_dir(config: MultivariateRunConfig) -> Path:
    return config.module_root / "data" / "processed" / config.region


def _legacy_results_dir(config: MultivariateRunConfig, family: str) -> Path:
    return config.module_root / "results" / family / config.region


def _legacy_bbox_tag(config: MultivariateRunConfig) -> str:
    return bbox_step_filename_suffix(config.module_root, config.region, config.bbox_step)


def resolve_legacy_z500_artifacts(config: MultivariateRunConfig) -> LegacyZ500Artifacts:
    notes: list[str] = []
    expected = _legacy_expected(config)
    opt_label = _opt_label(config)
    w = int(config.smooth_window)
    eps = int(round(float(config.z500_epsilon)))
    var = int(round(float(config.var_threshold) * 100))
    years = str(config.year_span)
    bbox_tag = _legacy_bbox_tag(config)

    anom = _legacy_preprocessed_dir(config) / f"03_Z500_anomaly_w{w:02d}_{opt_label}_{years}.nc"
    pre = _legacy_preprocessed_dir(config) / (
        f"04_05_06_Z500_preprocessed_w{w:02d}_eps{eps:02d}_{opt_label}_{years}.nc"
    )
    pca = _legacy_processed_dir(config) / (
        f"07_Z500_PCA_w{w:02d}_eps{eps:02d}_var{var:02d}_{opt_label}_{years}.nc"
    )
    hc = _legacy_processed_dir(config) / (
        f"08_Z500_HC_Ward_w{w:02d}_eps{eps:02d}_var{var:02d}_{opt_label}_{years}.nc"
    )
    reg = _legacy_results_dir(config, "clustering") / (
        f"09_Z500_regimes_kmeans_k{config.final_k:02d}_w{w:02d}_eps{eps:02d}_var{var:02d}_{years}{bbox_tag}.nc"
    )
    s1011 = _legacy_results_dir(config, "regimes") / (
        f"10_11_Z500_regime_reconstruction_characterisation_kmeans_k{config.final_k:02d}_w{w:02d}_"
        f"eps{eps:02d}_var{var:02d}_{years}{bbox_tag}.nc"
    )

    pca_ok = _readable_and_matching(pca, expected_attrs=expected)
    pre_ok = _readable_and_matching(pre, expected_attrs=expected)
    anom_ok = _readable_and_matching(anom, expected_attrs=expected)
    hc_ok = _readable_and_matching(hc, expected_attrs=expected)
    reg_ok = _readable_and_matching(reg, expected_attrs={"year_span": years})
    s1011_ok = _readable_and_matching(s1011, expected_attrs={"year_span": years})

    if pca_ok:
        notes.append(f"Reusing legacy Step 7 PCA: {pca.name}")
        return LegacyZ500Artifacts(
            tier="step7",
            anomaly_path=anom if anom_ok else None,
            preprocessed_path=pre if pre_ok else None,
            pca_path=pca,
            hc_path=hc if hc_ok else None,
            regimes_path=reg if reg_ok else None,
            step1011_path=s1011 if s1011_ok else None,
            notes=tuple(notes),
        )
    if pre_ok:
        notes.append(f"Reusing legacy Step 4-6 preprocessed Z500: {pre.name}")
        return LegacyZ500Artifacts(
            tier="step4_6",
            anomaly_path=anom if anom_ok else None,
            preprocessed_path=pre,
            hc_path=hc if hc_ok else None,
            regimes_path=reg if reg_ok else None,
            step1011_path=s1011 if s1011_ok else None,
            notes=tuple(notes),
        )
    if anom_ok:
        notes.append(f"Reusing legacy Step 3 anomaly Z500: {anom.name}")
        return LegacyZ500Artifacts(
            tier="step3",
            anomaly_path=anom,
            hc_path=hc if hc_ok else None,
            regimes_path=reg if reg_ok else None,
            step1011_path=s1011 if s1011_ok else None,
            notes=tuple(notes),
        )

    notes.append("No compatible legacy Z500 artifacts found; recompute in multivariate namespace.")
    return LegacyZ500Artifacts(
        tier="none",
        hc_path=hc if hc_ok else None,
        regimes_path=reg if reg_ok else None,
        step1011_path=s1011 if s1011_ok else None,
        notes=tuple(notes),
    )
