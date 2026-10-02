from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from common.preprocessing_helpers import (
    PreprocessConfig,
    _load_bbox_steps_yaml,
    apply_yeo_johnson,
    compute_step4_sigma_and_norm,
    compute_step5_area_weighting,
    compute_step6_flatten,
    cyclic_running_mean,
    detrend_domain_mean,
    detrend_linear,
    geopotential_to_height_if_needed,
    infer_core_dims,
    mjjas_season_day_climatology_mean_unstack,
    pick_time_dim_from_dataset,
    select_pressure_level,
    select_raw_era5_nc_files,
    sort_raw_era5_paths_by_time_lo,
    subtract_mjjas_season_day_climatology,
)

from .config import MultivariateRunConfig
from .field_specs import FieldSpec, get_field_spec
from .paths import WorkflowPaths
from .smoothing import apply_spatial_smoothing
from .vorticity import open_vo850_or_compute_from_uv


@dataclass(frozen=True)
class FieldPreprocessResult:
    field_key: str
    anomaly_path: Path
    preprocessed_path: Path
    source_paths: tuple[Path, ...]
    reused: bool = False


def _multivariate_raw_dir(config: MultivariateRunConfig, field_key: str) -> Path:
    return config.module_root / "data" / "multivariate" / "raw" / "era5" / field_key / config.region


def _legacy_z500_raw_dir(config: MultivariateRunConfig) -> Path:
    return config.module_root / "data" / "raw" / "era5" / "z500" / config.region


def select_field_raw_paths(
    config: MultivariateRunConfig,
    *,
    field_spec: FieldSpec,
    explicit_raw_nc: Path | None = None,
) -> list[Path]:
    raw_dir = _multivariate_raw_dir(config, field_spec.key)
    if explicit_raw_nc is not None:
        return select_raw_era5_nc_files(raw_dir, years_span=config.raw_years_span, raw_nc=explicit_raw_nc)

    if field_spec.key == "z500" and not raw_dir.exists():
        raw_dir = _legacy_z500_raw_dir(config)
    return select_raw_era5_nc_files(raw_dir, years_span=config.raw_years_span, raw_nc=None)


def _resolve_bbox(config: MultivariateRunConfig) -> dict[str, float] | None:
    if config.bbox:
        toks = [t.strip() for t in str(config.bbox).split(",")]
        if len(toks) != 4:
            raise ValueError(f'Invalid bbox override {config.bbox!r}; expected "N,W,S,E".')
        n, w, s, e = (float(t) for t in toks)
        return {"N": n, "W": w, "S": s, "E": e}

    steps = _load_bbox_steps_yaml(config.bbox_config_path)
    if config.bbox_step not in steps:
        raise KeyError(
            f"Unknown bbox_step={config.bbox_step!r}. Available in {config.bbox_config_path}: {sorted(steps)}"
        )
    entry = steps[config.bbox_step]
    bbox = entry.get("bbox")
    if not isinstance(bbox, dict):
        raise KeyError(f"bbox_step={config.bbox_step!r} has no bbox entry in {config.bbox_config_path}")
    return {k: float(bbox[k]) for k in ("N", "W", "S", "E")}


def _infer_field_var(ds: xr.Dataset, field_spec: FieldSpec) -> str:
    for candidate in field_spec.candidate_var_names:
        if candidate in ds.data_vars:
            return candidate
    raise KeyError(
        f"Could not infer variable for field {field_spec.key!r}. "
        f"Candidates={field_spec.candidate_var_names}, available={list(ds.data_vars)}"
    )


def _epsilon_for_field(config: MultivariateRunConfig, field_key: str) -> float:
    if config.yeo_johnson:
        # After YJ the field is in ~N(0,1) space; physical epsilons (e.g. 20 m for
        # Z500) would always trigger the sigma floor. Use a small value that only
        # guards against truly degenerate near-zero sigma cells.
        return 0.01
    if field_key == "z500":
        return float(config.z500_epsilon)
    if field_key == "vo850":
        return float(config.vo850_epsilon)
    if field_key == "tcwv":
        return float(config.tcwv_epsilon)
    if field_key == "deg0l":
        return float(config.deg0l_epsilon)
    return float(get_field_spec(field_key).default_epsilon)


def _open_field_to_mjjas(
    paths: list[Path],
    *,
    field_spec: FieldSpec,
    config: MultivariateRunConfig,
) -> xr.DataArray:
    cfg = PreprocessConfig(
        target_pressure_hpa=int(field_spec.pressure_level_hpa or 0),
        detrend_method=config.detrend_method,
        climatology_subtract=config.climatology_subtract,
        epsilon_m=_epsilon_for_field(config, field_spec.key),
        sigma_mode=config.sigma_mode,
        smooth_windows=(config.smooth_window,),
    )
    sorted_paths = sort_raw_era5_paths_by_time_lo(paths, time_dim_candidates=cfg.time_dim_candidates)
    if len(sorted_paths) == 1:
        ds = xr.open_dataset(sorted_paths[0], decode_times=True, chunks=None)
    else:
        ds = xr.open_mfdataset(sorted_paths, combine="by_coords", decode_times=True, chunks=None)

    time_dim = pick_time_dim_from_dataset(ds, time_dim_candidates=cfg.time_dim_candidates)
    if field_spec.key == "vo850":
        probe_name = next(
            (name for name in (*field_spec.candidate_var_names, "u", "v", "u_component_of_wind") if name in ds),
            next(iter(ds.data_vars)),
        )
        dims_probe = infer_core_dims(ds[probe_name], cfg=cfg)
        da = open_vo850_or_compute_from_uv(
            ds,
            lat_dim=dims_probe["lat"],
            lon_dim=dims_probe["lon"],
            field_spec_candidates=field_spec.candidate_var_names,
        )
    else:
        field_var = _infer_field_var(ds, field_spec)
        da = ds[field_var]
    if field_spec.pressure_level_hpa is not None:
        da = select_pressure_level(da, cfg=cfg)
    if field_spec.convert_geopotential_to_height:
        da = geopotential_to_height_if_needed(da, cfg=cfg)
    da = da.rename(field_spec.key)

    dims = infer_core_dims(da, cfg=cfg)
    lat_dim = dims["lat"]
    lon_dim = dims["lon"]
    bbox = _resolve_bbox(config)
    if bbox is not None:
        lat_vals = np.asarray(da[lat_dim].values)
        lat_desc = bool(lat_vals.size >= 2 and lat_vals[0] > lat_vals[-1])
        lat_slice = slice(bbox["N"], bbox["S"]) if lat_desc else slice(bbox["S"], bbox["N"])
        lon_slice = slice(bbox["W"], bbox["E"])
        da = da.sel({lat_dim: lat_slice, lon_dim: lon_slice})

    da = da.sortby(time_dim)
    daily = da.resample({time_dim: "1D"}).mean(skipna=True)
    daily = daily.where(daily[time_dim].dt.month.isin((5, 6, 7, 8, 9)), drop=True)
    daily = daily.astype("float32")
    daily.attrs = dict(da.attrs)
    daily.attrs["field_key"] = field_spec.key
    daily.attrs["raw_input_paths"] = "|".join(str(p) for p in sorted_paths)
    if config.yeo_johnson:
        daily = apply_yeo_johnson(daily)
    return daily


def _detrend_field(da: xr.DataArray, *, config: MultivariateRunConfig) -> xr.DataArray:
    cfg = PreprocessConfig(
        detrend_method=config.detrend_method,
        climatology_subtract=config.climatology_subtract,
        epsilon_m=config.z500_epsilon,
        sigma_mode=config.sigma_mode,
        smooth_windows=(config.smooth_window,),
    )
    dims = infer_core_dims(da, cfg=cfg)
    if config.detrend_method == "domain_mean":
        out = detrend_domain_mean(da, time_dim=dims["time"], lat_dim=dims["lat"], lon_dim=dims["lon"])
    else:
        out = detrend_linear(da, time_dim=dims["time"])
    return out.rename(f"{da.name}_det")


def _build_anomaly(
    da_det: xr.DataArray,
    *,
    config: MultivariateRunConfig,
    field_key: str,
) -> xr.DataArray:
    cfg = PreprocessConfig(
        detrend_method=config.detrend_method,
        climatology_subtract=config.climatology_subtract,
        epsilon_m=config.z500_epsilon,
        sigma_mode=config.sigma_mode,
        smooth_windows=(config.smooth_window,),
    )
    dims = infer_core_dims(da_det, cfg=cfg)
    time_dim = dims["time"]
    clim_raw = mjjas_season_day_climatology_mean_unstack(da_det, time_dim=time_dim).rename("field_clim_raw")
    clim_smooth = cyclic_running_mean(clim_raw, dim="season_day", window=int(config.smooth_window)).rename(
        "field_clim_smooth"
    )
    clim_for_sub = clim_smooth if config.climatology_subtract == "smooth" else clim_raw
    anom = subtract_mjjas_season_day_climatology(da_det, clim_for_sub, time_dim=time_dim).rename("field_anom")
    anom.attrs = dict(da_det.attrs)
    anom.attrs["field_key"] = field_key
    anom.attrs["climatology_subtract"] = config.climatology_subtract
    anom.attrs["window_days"] = int(config.smooth_window)
    return anom


def _maybe_smooth_anomaly(
    anom: xr.DataArray,
    *,
    config: MultivariateRunConfig,
    field_key: str,
) -> xr.DataArray:
    if field_key != "vo850":
        return anom
    out = apply_spatial_smoothing(
        anom,
        method=config.vo850_smoothing_method,
        gaussian_sigma=config.vo850_gaussian_sigma,
        coarsen_factor=config.vo850_coarsen_factor,
    )
    out.attrs = dict(anom.attrs)
    out.attrs["field_key"] = field_key
    out.attrs["smoothing_method"] = config.vo850_smoothing_method
    out.attrs["gaussian_sigma_gridcells"] = float(config.vo850_gaussian_sigma)
    out.attrs["coarsen_factor"] = int(config.vo850_coarsen_factor)
    return out.rename("field_anom")


def build_step4_6_from_anomaly(
    anomaly: xr.DataArray,
    *,
    field_key: str,
    config: MultivariateRunConfig,
) -> xr.Dataset:
    """Standardise anomalies, apply sqrt(cos phi) area weighting, and flatten for PCA input.

    Area weighting corrects for lat/lon grid-cell area convergence toward the poles — a
    property of the shared grid, not of the physical field — so it is applied uniformly
    to every field.
    """
    cfg = PreprocessConfig(
        detrend_method=config.detrend_method,
        climatology_subtract=config.climatology_subtract,
        epsilon_m=_epsilon_for_field(config, field_key),
        sigma_mode=config.sigma_mode,
        smooth_windows=(config.smooth_window,),
    )
    sigma, season_day_of_time, field_norm = compute_step4_sigma_and_norm(
        z_anom=anomaly.rename("field_anom"),
        window_days=int(config.smooth_window),
        epsilon_m=_epsilon_for_field(config, field_key),
        cfg=cfg,
    )
    w_lat, weighted_field = compute_step5_area_weighting(z_norm=field_norm, cfg=cfg)
    weighted_field = weighted_field.rename("weighted_field")
    # Ocean-only fields (SST) have NaN over land. Zero-fill the standardised, weighted
    # anomaly so those cells enter the flatten as zero-variance columns (no covariance
    # contribution, PCs match a masked ocean EOF) while preserving the full grid for
    # flatten/reshape.
    if get_field_spec(field_key).fill_nonfinite_with_zero:
        weighted_field = weighted_field.fillna(0.0)
    X_matrix, latitude_space, longitude_space = compute_step6_flatten(z_weighted=weighted_field, cfg=cfg)

    field_norm_out = (
        field_norm.reset_coords("season_day", drop=True) if "season_day" in field_norm.coords else field_norm
    )
    ds = xr.Dataset(
        data_vars={
            "field_norm": field_norm_out.astype("float32"),
            "sigma_season_day": sigma.astype("float32"),
            "weighted_field": weighted_field.astype("float32"),
            "w_lat": w_lat.astype("float32"),
            "season_day_of_time": season_day_of_time.astype("int16"),
            "X_matrix": X_matrix,
            "latitude_space": latitude_space.astype("float32"),
            "longitude_space": longitude_space.astype("float32"),
        },
        attrs=dict(anomaly.attrs),
    )
    ds.attrs.update(
        {
            "step": "04_05_06_preprocess_standardise_weighting_flattening",
            "window_days": int(config.smooth_window),
            "epsilon_m": float(_epsilon_for_field(config, field_key)),
            "area_weighting": "sqrt(cos(phi)) applied after standardisation and before flatten (all fields)",
            "definition_step4": "field_norm(time,lat,lon) = field_anom(time,lat,lon) / sigma(season_day,lat,lon)",
            "definition_step5": "weighted_field(time,lat,lon) = field_norm(time,lat,lon) * sqrt(cos(lat))",
            "workflow_family": "multifield",
            "field_key": field_key,
        }
    )
    return ds


def explicit_raw_nc_for_field(config: MultivariateRunConfig, field_key: str) -> Path | None:
    if field_key == "z500":
        return config.z500_raw_nc
    if field_key == "vo850":
        return config.vo850_raw_nc
    if field_key == "tcwv":
        return config.tcwv_raw_nc
    if field_key == "deg0l":
        return config.deg0l_raw_nc
    return None


def run_field_preprocessing(
    config: MultivariateRunConfig,
    *,
    field_key: str,
    paths: WorkflowPaths,
    raw_nc: Path | None = None,
) -> FieldPreprocessResult:
    field_spec = get_field_spec(field_key)
    anom_path = paths.field_anomaly_path(field_key)
    prepro_path = paths.field_preprocessed_path(field_key)
    if (
        not config.overwrite
        and anom_path.is_file()
        and prepro_path.is_file()
    ):
        return FieldPreprocessResult(
            field_key=field_key,
            anomaly_path=anom_path,
            preprocessed_path=prepro_path,
            source_paths=(),
            reused=True,
        )

    in_files = select_field_raw_paths(config, field_spec=field_spec, explicit_raw_nc=raw_nc)
    field = _open_field_to_mjjas(in_files, field_spec=field_spec, config=config)
    det = _detrend_field(field, config=config)
    anom = _build_anomaly(det, config=config, field_key=field_key)
    anom = _maybe_smooth_anomaly(anom, config=config, field_key=field_key)
    ds_pre = build_step4_6_from_anomaly(anom, field_key=field_key, config=config)

    anom_path.parent.mkdir(parents=True, exist_ok=True)
    prepro_path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"field_anom": anom.astype("float32")}, attrs=dict(anom.attrs)).to_netcdf(anom_path)
    ds_pre.to_netcdf(prepro_path)
    return FieldPreprocessResult(
        field_key=field_key,
        anomaly_path=anom_path,
        preprocessed_path=prepro_path,
        source_paths=tuple(in_files),
        reused=False,
    )


def run_step4_6_from_existing_anomaly(
    config: MultivariateRunConfig,
    *,
    field_key: str,
    in_anomaly_path: Path,
    paths: WorkflowPaths,
) -> FieldPreprocessResult:
    prepro_path = paths.field_preprocessed_path(field_key)
    anom_path = paths.field_anomaly_path(field_key)
    if not config.overwrite and prepro_path.is_file():
        return FieldPreprocessResult(
            field_key=field_key,
            anomaly_path=anom_path if anom_path.is_file() else in_anomaly_path,
            preprocessed_path=prepro_path,
            source_paths=(in_anomaly_path,),
            reused=True,
        )
    with xr.open_dataset(in_anomaly_path, chunks=None) as ds:
        var = "field_anom" if "field_anom" in ds.data_vars else next(v for v in ds.data_vars if "anom" in v.lower())
        anomaly = ds[var].load().rename("field_anom")
    ds_pre = build_step4_6_from_anomaly(anomaly, field_key=field_key, config=config)
    prepro_path.parent.mkdir(parents=True, exist_ok=True)
    ds_pre.to_netcdf(prepro_path)
    if not anom_path.exists():
        anom_path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset({"field_anom": anomaly.astype("float32")}, attrs=dict(anomaly.attrs)).to_netcdf(anom_path)
    return FieldPreprocessResult(
        field_key=field_key,
        anomaly_path=anom_path if anom_path.exists() else in_anomaly_path,
        preprocessed_path=prepro_path,
        source_paths=(in_anomaly_path,),
        reused=False,
    )
