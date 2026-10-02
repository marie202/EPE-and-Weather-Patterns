from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr


def cape_convective_ok(cape: xr.DataArray, cape_threshold: float = 0.0) -> xr.DataArray:
    """True where CAPE passes the convective filter (non-finite → False).

    CAPE exactly 0 never counts. With ``cape_threshold <= 0`` (default), require
    ``CAPE > 0``. With ``cape_threshold > 0``, require ``CAPE > cape_threshold``.
    """
    thr = float(cape_threshold)
    if thr < 0.0:
        raise ValueError("cape_threshold must be >= 0")
    min_cape = thr if thr > 0.0 else 0.0
    return (cape > min_cape) & np.isfinite(cape)


def cape_filter_description(cape_threshold: float) -> str:
    thr = float(cape_threshold)
    if thr <= 0.0:
        return "CAPE>0"
    return f"CAPE>{thr:g}"


def netcdf_attr_bool(value: bool) -> int:
    """NetCDF4 global attrs do not accept numpy/python bool; use 0/1."""
    return 1 if value else 0


# -------------------------
# Paths + file autodetection
# -------------------------


def resolve_paper_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "1_Detect_Weather_Patterns").is_dir():
            return p / "1_Detect_Weather_Patterns"
    raise FileNotFoundError("Could not locate 1_Detect_Weather_Patterns/ from current working directory")


def resolve_step_root(start: Path) -> Path:
    # Backwards-compatible alias used in some notebooks.
    return resolve_paper_root(start)


def newest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No files matched")
    return max(paths, key=lambda p: p.stat().st_mtime)


def newest_extremes_file(step_root: Path, region_label: str | None = None) -> Path:
    if region_label is None:
        candidates = list(step_root.glob("data/processed/*/extreme_precip_events/extremes_*.nc"))
    else:
        candidates = list((step_root / "data" / "processed" / region_label / "extreme_precip_events").glob("extremes_*.nc"))
    if not candidates:
        raise FileNotFoundError(
            "No extremes files found. Expected something like:\n"
            "  1_Detect_Weather_Patterns/data/processed/<region>/extreme_precip_events/extremes_*.nc\n"
            "Run scripts/epe_01a_daily_aggregate_detect_extremes.sh first (or set EXTREMES_PATH explicitly)."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def newest_duration_file(step_root: Path, region_label: str | None = None) -> Path:
    if region_label is None:
        candidates = list(step_root.glob("data/processed/*/extreme_precip_events/convective_duration_*.nc"))
    else:
        candidates = list(
            (step_root / "data" / "processed" / region_label / "extreme_precip_events").glob("convective_duration_*.nc")
        )
    if not candidates:
        raise FileNotFoundError(
            "No convective-duration files found. Expected something like:\n"
            "  1_Detect_Weather_Patterns/data/processed/<region>/extreme_precip_events/convective_duration_*.nc\n"
            "Run scripts/epe_01b_hourly_convective_duration.sh first (or set DURATION_PATH explicitly)."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_duration_thresholds_from_name(path: Path) -> dict[str, float]:
    """Extract qXX and capeY from convective_duration filename.

    Examples: ..._q99_cape0_...  -> {"q": 99.0, "cape_threshold": 0.0}
              ..._q999_sampleAll_cape0_... -> {"q": 99.9, "quantile_sample": "All", ...}
              ..._q99_sampleWetmm01_cape0_... -> wet-hours sample
    """
    name = Path(path).name
    m = re.search(
        r"_q(?P<q>\d{2,3})_(?:sample(?P<sample>[A-Za-z0-9]+)_)?cape(?P<cape>-?\d+(?:\.\d+)?)_",
        name,
    )
    if not m:
        return {}
    q_raw = m.group("q")
    q = float(q_raw) / 10.0 if len(q_raw) == 3 else float(q_raw)
    cape = float(m.group("cape"))
    out: dict[str, float | str] = {"q": q, "cape_threshold": cape}
    sample = m.group("sample")
    if sample:
        out["quantile_sample_tag"] = sample
        if sample.startswith("Wetmm"):
            out["quantile_sample"] = "wet_hours"
            out["wet_threshold_mm"] = float(sample[len("Wetmm") :]) / 10.0
        elif sample == "All":
            out["quantile_sample"] = "all_hours"
    return out


# -------------------------
# Coordinate/dimension helpers
# -------------------------


def guess_time_name(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.coords:
            return cand
    raise KeyError(f"No time coordinate found; coords={list(ds.coords)}")


def guess_lat_lon(obj: xr.Dataset | xr.DataArray) -> tuple[str, str]:
    coords = obj.coords
    lat = "latitude" if "latitude" in coords else "lat" if "lat" in coords else None
    lon = "longitude" if "longitude" in coords else "lon" if "lon" in coords else None
    if lat is None or lon is None:
        raise KeyError(f"Could not find lat/lon coords in {list(coords)}")
    return lat, lon


def dataset_rename_time(ds: xr.Dataset) -> xr.Dataset:
    tname = guess_time_name(ds)
    return ds.rename({tname: "time"}) if tname != "time" else ds


def days_coord_ns(times: Any) -> np.ndarray:
    """Calendar-day coords at nanosecond dtype (quiet xarray / pandas coercion warnings)."""
    return np.asarray(times).astype("datetime64[D]").astype("datetime64[ns]")


def regime_labels_by_day(reg: xr.Dataset) -> xr.Dataset:
    """Return a dataset with coords day (datetime64[ns] at UTC midnight per calendar day)."""
    reg = dataset_rename_time(reg)
    if "regime_labels" not in reg:
        raise KeyError(f"Expected reg['regime_labels'], found vars={list(reg.data_vars)}")
    reg_days = days_coord_ns(reg["time"].values)
    reg_day_ds = xr.Dataset(
        {"regime_labels": ("day", reg["regime_labels"].values.astype(int))},
        coords={"day": reg_days},
    )
    # If duplicate days exist, keep first occurrence
    _, first_idx = np.unique(reg_day_ds["day"].values, return_index=True)
    return reg_day_ds.isel(day=np.sort(first_idx))


def subset_bbox(ds_in: xr.Dataset | None, *, bbox: dict[str, float]) -> xr.Dataset | None:
    """Subset any dataset to a lat/lon bbox.

    bbox keys: N,W,S,E (CDS order). Works with ascending or descending lat.
    """
    if ds_in is None:
        return None

    # Datasets with no vars (time-only placeholders) still might have coords.
    if len(ds_in.data_vars) == 0:
        lat_name = "latitude" if "latitude" in ds_in.coords else ("lat" if "lat" in ds_in.coords else None)
        lon_name = "longitude" if "longitude" in ds_in.coords else ("lon" if "lon" in ds_in.coords else None)
        if lat_name is None or lon_name is None:
            warnings.warn("Dataset has no vars and no lat/lon coords; cannot subset by bbox. Returning unchanged.")
            return ds_in
    else:
        lat_name, lon_name = guess_lat_lon(ds_in)

    lat = ds_in[lat_name]
    lon = ds_in[lon_name]
    lat_desc = bool((lat.size >= 2) and (float(lat[0]) > float(lat[-1])))
    lat_slice = slice(bbox["N"], bbox["S"]) if lat_desc else slice(bbox["S"], bbox["N"])
    lon_slice = slice(bbox["W"], bbox["E"])
    return ds_in.sel({lat_name: lat_slice, lon_name: lon_slice})


# -------------------------
# EPE-derived conversions
# -------------------------


def extremes_placeholder_from_duration(dur: xr.Dataset) -> xr.Dataset:
    """Minimal stand-in when no extremes_*.nc exists but convective_duration was loaded."""
    if "time" not in dur.coords:
        raise ValueError("duration dataset must have a 'time' coordinate")
    out = xr.Dataset(coords={"time": dur["time"].copy()})
    for k in (
        "timezone",
        "day_start_hour_local",
        "region_label",
        "q",
        "cape_threshold",
        "cp_frac_min",
        "months_tag",
        "years_spec",
    ):
        if k in dur.attrs:
            out.attrs[k] = dur.attrs[k]
    out.attrs["extremes_source"] = "placeholder_from_convective_duration"
    return out


def synthetic_extremes_from_duration(dur: xr.Dataset) -> xr.Dataset:
    """Build extremes-like daily fields from ``convective_duration_*.nc`` (epe_01b)."""
    if "n_conv_hours_grid" not in dur:
        raise KeyError(f"Missing 'n_conv_hours_grid'. Vars={list(dur.data_vars)}")
    if "P_conv_total_grid_day" not in dur:
        raise KeyError(f"Missing 'P_conv_total_grid_day'. Vars={list(dur.data_vars)}")

    E_conv = (dur["n_conv_hours_grid"] > 0).astype("i1")
    P = dur["P_conv_total_grid_day"].astype("float32")

    if "flagged_domain_pctl" in dur:
        # Domain-aggregated extreme-day definition (epe_01b_..._test.py TEST variant, e.g.
        # domain_pctl_q=0.95 + domain_agg="lexmax") — takes priority over the plain per-cell-any
        # definition below when present, since it is the more specific/intended label.
        domain_extreme = dur["flagged_domain_pctl"].astype("i1")
    elif "flagged_any_gridcell" in dur:
        domain_extreme = dur["flagged_any_gridcell"].astype("i1")
    elif "n_conv_hours_anywhere" in dur:
        domain_extreme = (dur["n_conv_hours_anywhere"] > 0).astype("i1")
    else:
        raise KeyError("Duration file needs 'flagged_any_gridcell' or 'n_conv_hours_anywhere'.")

    coords: dict[str, Any] = {"time": dur.coords["time"]}
    for k in ("latitude", "longitude", "lat", "lon"):
        if k in dur.coords:
            coords[k] = dur.coords[k]

    out = xr.Dataset({"E_conv": E_conv, "P": P, "domain_extreme": domain_extreme}, coords=coords)
    for k in ("q", "cape_threshold", "cp_frac_min", "timezone", "day_start_hour_local", "region_label", "years_spec"):
        if k in dur.attrs:
            out.attrs[k] = dur.attrs[k]
    out.attrs["precip_metric"] = "P_conv_total_grid_day"
    out.attrs["extremes_source"] = "synthesized_from_convective_duration"
    return out


# -------------------------
# Units + basic summaries
# -------------------------


MM_PER_M = 1000.0


def to_mm(values, units: str | None) -> Any:
    u = (units or "").strip().lower()
    if u in {"m", "meter", "meters", "metre", "metres"}:
        return values * MM_PER_M
    if "mm" in u:
        return values
    # Default assumption in this workflow: ERA5 precip is meters
    return values * MM_PER_M


def _duration_has_time_hour(duration_ds: xr.Dataset) -> bool:
    if "time_hour" in duration_ds.coords or "time_hour" in duration_ds.dims:
        return True
    return any("time_hour" in duration_ds[v].dims for v in duration_ds.data_vars)


def _analysis_day_per_hour(duration_ds: xr.Dataset, time_hour_values) -> np.ndarray:
    """Map each UTC hour to its analysis-day start (matches daily ``time`` coordinate)."""
    tz_name = str(duration_ds.attrs.get("timezone", "UTC"))
    start_local = int(
        duration_ds.attrs.get(
            "day_start_hour_local",
            duration_ds.attrs.get("day_start_hour_utc", 6),
        )
    )
    t = pd.DatetimeIndex(pd.to_datetime(time_hour_values, utc=True))
    if t.tz is None:
        t = t.tz_localize("UTC")
    local = t.tz_convert(tz_name)
    shifted = local - pd.Timedelta(hours=start_local)
    day_start = shifted.normalize() + pd.Timedelta(hours=start_local)
    return day_start.tz_convert("UTC").to_numpy(dtype="datetime64[ns]")


def _hour_index_lists_per_daily_time(duration_ds: xr.Dataset) -> list[np.ndarray]:
    """For each daily ``time`` value, ``time_hour`` indices in that analysis day."""
    day_per_hour = _analysis_day_per_hour(duration_ds, duration_ds["time_hour"].values)
    daily = pd.to_datetime(duration_ds["time"].values, utc=True)
    if getattr(daily, "tz", None) is None:
        daily = daily.tz_localize("UTC")
    daily_ns = daily.to_numpy(dtype="datetime64[ns]")
    return [np.where(day_per_hour == d)[0] for d in daily_ns]


def _flagged_daily_intensity_proxy_max(
    duration_ds: xr.Dataset,
    *,
    agg: str = "mean",
) -> tuple[xr.DataArray | None, str]:
    """Proxy intensity from P_conv_total/n_conv on wet flagged cells (mm/h) per day.

    ``agg`` selects the spatial aggregation over wet cells: ``"mean"`` or ``"median"``.
    Wet-cell conditioning (rate > 0) avoids diluting precip statistics with dry cells.
    """
    if "P_conv_total_grid_day" not in duration_ds.data_vars or "n_conv_hours_grid" not in duration_ds.data_vars:
        return None, "missing P_conv_total_grid_day or n_conv_hours_grid"
    lat_name, lon_name = guess_lat_lon(duration_ds)
    spatial = (lat_name, lon_name)
    n = duration_ds["n_conv_hours_grid"].astype(float)
    p = to_mm(
        duration_ds["P_conv_total_grid_day"].astype(float),
        str(duration_ds["P_conv_total_grid_day"].attrs.get("units", "")),
    )
    rate = p / n.where(n > 0)
    rate_wet = rate.where(rate > 0)  # wet cells only — appropriate for precip
    if agg == "median":
        daily_agg = rate_wet.median(dim=spatial, skipna=True)
        metric = "proxy_P_conv_total_over_n_conv_wet_median_mm_per_h"
    else:
        daily_agg = rate_wet.mean(dim=spatial, skipna=True)
        metric = "proxy_P_conv_total_over_n_conv_wet_mean_mm_per_h"
    return daily_agg, metric


def _embedded_p_hour_from_duration(duration_ds: xr.Dataset) -> tuple[xr.DataArray | None, str]:
    """Precip on ``time_hour`` already stored by epe_01b (``--hourly-precip conv|all|both``)."""
    if "P_hour" in duration_ds.data_vars:
        return duration_ds["P_hour"], "P_hour_in_duration_nc"
    if "P_conv_hour" in duration_ds.data_vars:
        return duration_ds["P_conv_hour"], "P_conv_hour_in_duration_nc"
    return None, ""


def _intensity_has_finite_values(da: xr.DataArray | None) -> bool:
    if da is None:
        return False
    vals = np.asarray(da.values, dtype=float)
    return bool(vals.size and np.any(np.isfinite(vals)))


def flagged_hourly_precip_max_per_day(
    duration_ds: xr.Dataset,
    *,
    hourly_precip_path: Path | None = None,
    era5_hourly_dir: Path | None = None,
    hourly_var: str = "tp_hourly",
    allow_daily_mean_proxy: bool = True,
    agg: str = "mean",
) -> tuple[xr.DataArray | None, str]:
    """Daily mean (or median) hourly precip on wet convective-flagged cells (mm/h).

    ``agg`` controls spatial aggregation over wet flagged cells (P > 0) per hour:
    ``"mean"`` or ``"median"``.  Wet-cell conditioning is appropriate for precipitation
    since it avoids diluting the rate with dry boundary cells.  The temporal aggregation
    over convective hours within a day is always the mean.

    Uses embedded ``P_hour`` / ``P_conv_hour``, an external hourly file, raw ERA5 under
    ``era5_hourly_dir``, or (when ``allow_daily_mean_proxy``) a proxy from the duration file.
    """
    def _daily_proxy_or_none(reason: str) -> tuple[xr.DataArray | None, str]:
        if not allow_daily_mean_proxy:
            return None, reason
        proxy = _flagged_daily_intensity_proxy_max(duration_ds, agg=agg)
        if proxy[0] is not None:
            return proxy
        return None, reason

    has_hourly_precip = "P_hour" in duration_ds.data_vars or "P_conv_hour" in duration_ds.data_vars
    has_hourly_mask = "E_conv_hour" in duration_ds.data_vars
    if not has_hourly_precip and not has_hourly_mask:
        return _daily_proxy_or_none(
            "no P_hour/P_conv_hour and no E_conv_hour in duration file"
        )

    if not _duration_has_time_hour(duration_ds):
        return _daily_proxy_or_none("duration file lacks time_hour axis")

    p_hour_m, metric = _embedded_p_hour_from_duration(duration_ds)
    if p_hour_m is None:
        try:
            from epe_preprocessing.epe_object_catalog import _load_p_hour_meters

            p_hour_m, metric = _load_p_hour_meters(
                duration_ds,
                hourly_precip_path=hourly_precip_path,
                hourly_var=hourly_var,
                era5_hourly_dir=era5_hourly_dir,
            )
        except Exception as e:
            return _daily_proxy_or_none(f"hourly precip not loaded: {e}")

    if p_hour_m is None:
        return _daily_proxy_or_none("hourly precip fields missing")

    if "E_conv_hour" in duration_ds:
        flagged = duration_ds["E_conv_hour"].astype(int) > 0
        metric_flag = f"{metric}+E_conv_hour"
    elif "P_conv_hour" in duration_ds:
        flagged = duration_ds["P_conv_hour"].notnull() & (duration_ds["P_conv_hour"] > 0)
        metric_flag = f"{metric}+P_conv_hour_mask"
    else:
        return _daily_proxy_or_none(
            "E_conv_hour missing (re-run epe_01b with --hourly-mask conv); "
            f"vars={sorted(duration_ds.data_vars)}"
        )

    lat_name, lon_name = guess_lat_lon(duration_ds)
    spatial = (lat_name, lon_name)
    p_mm = to_mm(p_hour_m.astype(float), str(p_hour_m.attrs.get("units", "")))
    # Wet-cell mask: flagged by E_conv_hour AND precipitating (P > 0).
    # Conditioning on P > 0 is appropriate for precipitation — avoids diluting
    # the mean/median with dry boundary cells inside the convective mask.
    p_flagged_wet = p_mm.where(flagged & (p_mm > 0))
    if agg == "median":
        p_hour_agg = p_flagged_wet.median(dim=spatial, skipna=True)
        da_name = "P_flagged_hourly_median_day"
        metric_flag = f"{metric_flag}_median_wet"
    else:
        p_hour_agg = p_flagged_wet.mean(dim=spatial, skipna=True)
        da_name = "P_flagged_hourly_mean_day"
        metric_flag = f"{metric_flag}_mean_wet"

    # Temporal aggregation: mean over convective hours within each day.
    n_days = int(duration_ds.sizes.get("time", p_hour_agg.sizes.get("time", 0)))
    out = np.full((n_days,), np.nan, dtype=float)
    for ti, hrs in enumerate(_hour_index_lists_per_daily_time(duration_ds)):
        if hrs.size == 0:
            continue
        vals = np.asarray(p_hour_agg.isel(time_hour=hrs).values, dtype=float)
        wet = vals[np.isfinite(vals) & (vals > 0)]
        if wet.size:
            out[ti] = float(np.nanmean(wet))

    da = xr.DataArray(
        out.astype(float),
        coords={"time": duration_ds["time"].values},
        dims=("time",),
        name=da_name,
        attrs={"units": "mm/h", "metric": metric_flag},
    )
    if _intensity_has_finite_values(da):
        return da, metric_flag
    return _daily_proxy_or_none(
        f"hourly intensity had no finite values ({metric_flag}); "
        "falling back to proxy"
    )


@dataclass(frozen=True)
class ExtremeQuickStats:
    n_days_total: int
    n_days_extreme: int
    mean_unique_hours_extreme_day: float | None
    mean_flagged_cells_extreme_day: float | None


def _as_bool_mask(da: xr.DataArray) -> xr.DataArray:
    if da.dtype == bool:
        return da
    return da.astype(int) > 0


def _daily_extreme_mask(*, extremes_ds: xr.Dataset | None, duration_ds: xr.Dataset | None) -> xr.DataArray | None:
    if extremes_ds is not None and "domain_extreme" in extremes_ds:
        return _as_bool_mask(extremes_ds["domain_extreme"])
    if duration_ds is not None:
        if "flagged_domain_pctl" in duration_ds:
            # Domain-aggregated extreme-day definition (epe_01b_..._test.py TEST variant, e.g.
            # domain_pctl_q=0.95 + domain_agg="lexmax") — takes priority over the plain per-cell-any
            # definition below when present, same precedence as synthetic_extremes_from_duration().
            return _as_bool_mask(duration_ds["flagged_domain_pctl"])
        if "flagged_any_gridcell" in duration_ds:
            return _as_bool_mask(duration_ds["flagged_any_gridcell"])
        if "n_conv_hours_anywhere" in duration_ds:
            return _as_bool_mask(duration_ds["n_conv_hours_anywhere"] > 0)
    return None


def compute_extreme_quick_stats(
    *,
    extremes_ds: xr.Dataset | None = None,
    duration_ds: xr.Dataset | None = None,
) -> ExtremeQuickStats:
    """Compute the short statistics you requested (day-based).

    - total days: length of the daily time coordinate
    - extreme days: domain-level flag (prefers extremes_ds['domain_extreme'], else duration flags)
    - avg unique hours flagged per extreme day: prefers duration_ds['n_conv_hours_anywhere']
    - avg flagged cells per extreme day: prefers E_conv>0 count, else duration n_conv_gridcells_any_hour_day
    """
    base = extremes_ds if extremes_ds is not None else duration_ds
    if base is None or "time" not in base.coords:
        raise KeyError("Need a dataset with a daily 'time' coordinate to compute stats.")

    n_days_total = int(base["time"].size)

    day_mask = _daily_extreme_mask(extremes_ds=extremes_ds, duration_ds=duration_ds)
    if day_mask is None:
        n_days_extreme = 0
    else:
        n_days_extreme = int(day_mask.sum().item())

    # Mean unique hours on extreme days
    mean_unique_hours: float | None = None
    if duration_ds is not None and "n_conv_hours_anywhere" in duration_ds and day_mask is not None:
        a = duration_ds["n_conv_hours_anywhere"].astype(float)
        mean_unique_hours = float(a.where(day_mask).mean(skipna=True).item())

    # Mean flagged cells per extreme day
    mean_cells: float | None = None
    if day_mask is not None:
        if extremes_ds is not None and "E_conv" in extremes_ds:
            E = extremes_ds["E_conv"]
            lat_name, lon_name = guess_lat_lon(E)
            n_cells_day = (E.astype(int) > 0).sum(dim=(lat_name, lon_name)).astype(float)
            mean_cells = float(n_cells_day.where(day_mask).mean(skipna=True).item())
        elif duration_ds is not None and "n_conv_gridcells_any_hour_day" in duration_ds:
            mean_cells = float(duration_ds["n_conv_gridcells_any_hour_day"].astype(float).where(day_mask).mean().item())
        elif duration_ds is not None and "n_conv_hours_grid" in duration_ds:
            H = duration_ds["n_conv_hours_grid"]
            lat_name, lon_name = guess_lat_lon(H)
            n_cells_day = (H.astype(float) > 0).sum(dim=(lat_name, lon_name)).astype(float)
            mean_cells = float(n_cells_day.where(day_mask).mean(skipna=True).item())

    return ExtremeQuickStats(
        n_days_total=n_days_total,
        n_days_extreme=n_days_extreme,
        mean_unique_hours_extreme_day=mean_unique_hours,
        mean_flagged_cells_extreme_day=mean_cells,
    )


def print_extreme_quick_stats(
    *,
    extremes_ds: xr.Dataset | None = None,
    duration_ds: xr.Dataset | None = None,
    label: str = "",
) -> ExtremeQuickStats:
    stats = compute_extreme_quick_stats(extremes_ds=extremes_ds, duration_ds=duration_ds)
    prefix = (label.strip() + " ") if label.strip() else ""

    print(f"{prefix}total days: {stats.n_days_total}")
    print(f"{prefix}extreme days (domain flag): {stats.n_days_extreme}")
    if stats.mean_unique_hours_extreme_day is None:
        print(f"{prefix}avg unique extreme hours per extreme day: (not available)")
    else:
        print(f"{prefix}avg unique extreme hours per extreme day: {stats.mean_unique_hours_extreme_day:.2f}")
    if stats.mean_flagged_cells_extreme_day is None:
        print(f"{prefix}avg flagged cells per extreme day: (not available)")
    else:
        print(f"{prefix}avg flagged cells per extreme day: {stats.mean_flagged_cells_extreme_day:.2f}")

    return stats


# -------------------------
# Step-12 study AOIs (Germany tight + ontario_south)
# -------------------------

# ERA5 single-level grid spacing used in Step-12 CDS downloads (degrees).
ERA5_GRID_STEP_DEG = 0.25

# Southernmost point of Ontario (Middle Island, Lake Erie), degrees north.
ONTARIO_PROVINCE_SOUTHERNMOST_LAT_DEG = 41.676

# Longitude centre for the ontario_south tight box (``regions.yaml`` ref_point lon).
ONTARIO_SOUTH_LON_CENTER_DEG = -79.0


def bbox_latlon_from_nwse(bb: dict[str, float]) -> dict[str, float]:
    """Convert ``regions.yaml``-style N/W/S/E to lat/lon min/max keys."""
    return {
        "lat_min": float(bb["S"]),
        "lat_max": float(bb["N"]),
        "lon_min": float(bb["W"]),
        "lon_max": float(bb["E"]),
    }


def ontario_south_study_bbox(
    *,
    germany_bbox: dict[str, float] | None = None,
    southern_lat_deg: float = ONTARIO_PROVINCE_SOUTHERNMOST_LAT_DEG,
    lon_center_deg: float | None = None,
) -> dict[str, float]:
    """Tight ``ontario_south`` window: same Δlat/Δlon as Germany, south at Ontario's tip.

    Matches ``regions.yaml`` → ``ontario.bbox`` when Germany uses the tight DEU box.
    At ``ERA5_GRID_STEP_DEG`` spacing this yields the same grid-cell count as Germany.
    """
    if germany_bbox is None:
        germany_bbox = {
            "lat_min": 47.2,
            "lat_max": 55.1,
            "lon_min": 5.9,
            "lon_max": 15.0,
        }
    dlat = germany_bbox["lat_max"] - germany_bbox["lat_min"]
    dlon = germany_bbox["lon_max"] - germany_bbox["lon_min"]
    lat_min = float(southern_lat_deg)
    lat_max = lat_min + dlat
    lon_c = float(ONTARIO_SOUTH_LON_CENTER_DEG if lon_center_deg is None else lon_center_deg)
    lon_min = lon_c - dlon / 2.0
    lon_max = lon_c + dlon / 2.0
    return {
        "name": "Southern Ontario (ontario_south; same ERA5 spans as Germany tight)",
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }


def era5_grid_cell_count(bbox: dict[str, float], *, step_deg: float = ERA5_GRID_STEP_DEG) -> tuple[int, int]:
    """Approximate (nlat, nlon) cell counts for a lat/lon box at fixed ERA5 spacing."""
    nlat = int(round((bbox["lat_max"] - bbox["lat_min"]) / step_deg)) + 1
    nlon = int(round((bbox["lon_max"] - bbox["lon_min"]) / step_deg)) + 1
    return nlat, nlon


# -------------------------
# Optional: cartopy overlays
# -------------------------


def _try_import_cartopy():
    try:
        import cartopy.crs as ccrs  # type: ignore
        from cartopy.io import shapereader  # type: ignore

        return True, ccrs, shapereader, None
    except Exception as e:  # pragma: no cover
        return False, None, None, e


def _map_region_key(region: str) -> str:
    """Normalize on-disk region ids to cartopy overlay keys."""
    r = (region or "germany").strip().lower()
    if r in {"ontario", "ontario_south"}:
        return "ontario"
    return r


def _region_label_from_module() -> str:
    """Read ``REGION_LABEL`` set on this module (e.g. ``s12f.REGION_LABEL = 'ontario'``)."""
    import sys

    mod = sys.modules[__name__]
    return str(getattr(mod, "REGION_LABEL", "germany"))


def _ne_record_attrs(rec) -> dict:
    return dict(getattr(rec, "attributes", {}) or {})


def _ne_adm0_code(attrs: dict) -> str:
    """ISO-like country code from Natural Earth attributes (handles field-name drift)."""
    for key in ("adm0_a3", "ADM0_A3", "sov_a3", "SOV_A3", "iso_a3", "ISO_A3", "BRK_A3"):
        raw = attrs.get(key)
        if raw is None:
            continue
        code = str(raw).strip().upper()
        if code and code not in {"-99", "-1", "NAN"}:
            if code in {"US1", "US"}:
                return "USA"
            return code
    iso2 = str(attrs.get("iso_a2") or attrs.get("ISO_A2") or "").strip().upper()
    if iso2 in {"US", "UM"}:
        return "USA"
    if iso2 == "CA":
        return "CAN"
    admin = str(attrs.get("admin") or attrs.get("ADMIN") or "").strip().lower()
    if "united states" in admin:
        return "USA"
    if admin == "canada":
        return "CAN"
    return ""


def _ne_adm1_name(attrs: dict) -> str:
    return str(
        attrs.get("name")
        or attrs.get("name_en")
        or attrs.get("name_alt")
        or attrs.get("NAME")
        or ""
    ).strip().lower()


def _ne_is_usa_or_canada(adm0: str) -> bool:
    return adm0 in {"USA", "CAN"}


def _ne_is_ontario(attrs: dict) -> bool:
    if _ne_adm0_code(attrs) != "CAN":
        return False
    name = _ne_adm1_name(attrs)
    return name == "ontario" or name.startswith("ontario ")


def _ne_geometries_from_shapefile(
    shapereader,
    shape_name: str,
    *,
    predicate,
) -> list:
    shp = shapereader.natural_earth(resolution="10m", category="cultural", name=shape_name)
    geoms = []
    for rec in shapereader.Reader(shp).records():
        if predicate(_ne_record_attrs(rec)):
            geoms.append(rec.geometry)
    return geoms


def _add_line_geometries(ax, ccrs, geoms: list, *, edgecolor: str, linewidth: float, zorder: float) -> None:
    if not geoms:
        return
    ax.add_geometries(
        geoms,
        crs=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )


def _add_ontario_study_admin_boundaries(
    ax,
    ccrs,
    shapereader,
    *,
    lw: float = 0.5,
    color: str = "0.35",
) -> None:
    """US states, CA provinces, Ontario counties, and a bold Ontario outline."""
    import cartopy.feature as cfeature

    # Sub-provincial (Ontario census divisions / counties)
    def _in_ontario_province(attrs: dict) -> bool:
        if _ne_adm0_code(attrs) != "CAN":
            return False
        prov = str(attrs.get("adm1_name") or attrs.get("name_1") or attrs.get("NAME_1") or "").lower()
        return prov == "ontario" or prov.startswith("ontario ")

    admin2_on = _ne_geometries_from_shapefile(
        shapereader,
        "admin_2_counties",
        predicate=_in_ontario_province,
    )
    _add_line_geometries(ax, ccrs, admin2_on, edgecolor="0.55", linewidth=max(0.25, lw * 0.7), zorder=4.5)

    admin1_na = _ne_geometries_from_shapefile(
        shapereader,
        "admin_1_states_provinces_lines",
        predicate=lambda a: _ne_is_usa_or_canada(_ne_adm0_code(a)),
    )
    if admin1_na:
        _add_line_geometries(ax, ccrs, admin1_na, edgecolor=color, linewidth=lw, zorder=5.0)
    else:
        for scale in ("10m", "50m"):
            try:
                ax.add_feature(
                    cfeature.NaturalEarthFeature(
                        "cultural",
                        "admin_1_states_provinces_lines",
                        scale,
                        facecolor="none",
                    ),
                    edgecolor=color,
                    linewidth=lw,
                    zorder=5.0,
                )
                break
            except Exception:
                continue

    ontario_prov = _ne_geometries_from_shapefile(
        shapereader,
        "admin_1_states_provinces_lines",
        predicate=_ne_is_ontario,
    )
    _add_line_geometries(ax, ccrs, ontario_prov, edgecolor="black", linewidth=max(1.1, lw + 0.6), zorder=6.0)


def _add_ontario_study_country_outlines(
    ax,
    ccrs,
    shapereader,
    *,
    lw: float = 1.2,
    color: str = "black",
) -> None:
    """USA + Canada country borders and coastlines on top of data layers."""
    import cartopy.feature as cfeature

    countries = _ne_geometries_from_shapefile(
        shapereader,
        "admin_0_countries",
        predicate=lambda a: _ne_is_usa_or_canada(_ne_adm0_code(a)),
    )
    if countries:
        _add_line_geometries(ax, ccrs, countries, edgecolor=color, linewidth=lw, zorder=7.0)
    else:
        for scale in ("10m", "50m"):
            try:
                ax.add_feature(
                    cfeature.BORDERS.with_scale(scale),
                    edgecolor=color,
                    linewidth=lw,
                    facecolor="none",
                    zorder=7.0,
                )
                break
            except Exception:
                continue

    for scale in ("10m", "50m"):
        try:
            ax.coastlines(resolution=scale, linewidth=0.55, edgecolor="0.4", zorder=8.0)
            break
        except Exception:
            continue


def add_germany_state_boundaries(ax, *, lw: float = 0.6, color: str = "black"):
    """Overlay admin boundaries for the current ``REGION_LABEL`` on this module."""
    ok, ccrs, shapereader, err = _try_import_cartopy()
    if not ok:
        warnings.warn(f"Cartopy not available; skipping admin boundaries. Import error: {err}")
        return

    region = _map_region_key(_region_label_from_module())

    if region == "ontario":
        _add_ontario_study_admin_boundaries(ax, ccrs, shapereader, lw=lw, color=color)
        return

    geoms = _ne_geometries_from_shapefile(
        shapereader,
        "admin_1_states_provinces_lines",
        predicate=lambda a: _ne_adm0_code(a) == "DEU",
    )
    if not geoms:
        warnings.warn(f"No admin-1 boundary geometries found for region={region!r}; skipping overlay.")
        return

    _add_line_geometries(ax, ccrs, geoms, edgecolor=color, linewidth=lw, zorder=5.0)


def add_germany_country_outline(ax, *, lw: float = 1.2, color: str = "black"):
    """Overlay country outlines for the current ``REGION_LABEL`` on this module."""
    ok, ccrs, shapereader, _ = _try_import_cartopy()
    if not ok:
        return

    region = _map_region_key(_region_label_from_module())

    if region == "ontario":
        _add_ontario_study_country_outlines(ax, ccrs, shapereader, lw=lw, color=color)
        return

    geoms = _ne_geometries_from_shapefile(
        shapereader,
        "admin_0_countries",
        predicate=lambda a: _ne_adm0_code(a) == "DEU",
    )
    if not geoms:
        return

    _add_line_geometries(ax, ccrs, geoms, edgecolor=color, linewidth=lw, zorder=6.0)

