"""Object-based convective extreme events from hourly ERA5 + epe_01b masks.

Builds an event catalogue: hourly scores H1–H3, convective-hour score C(t),
3D connected components on ``E_conv_hour``, per-event metrics, A/B/C classification,
ranking ``S``, and representative analysis-day (shared notebook + CLI).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from epe_preprocessing.epe_functions import guess_lat_lon, to_mm
from epe_preprocessing.detect_convective_duration import (
    parse_years,
    years_filename_slug,
)


def _structure_8neighbour_xy_samecell_time() -> np.ndarray:
    """3×3×3 binary connectivity: 8-neighbours in (y,x) at same t; ±1 h only at same (y,x)."""
    s = np.zeros((3, 3, 3), dtype=bool)
    s[1, :, :] = True
    s[0, 1, 1] = True
    s[2, 1, 1] = True
    s[1, 1, 1] = True
    return s


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - med) / (1.4826 * mad + 1e-12)


def _hourly_h3_amax_over_atotal(e_slice: np.ndarray) -> float:
    from scipy import ndimage

    if not np.any(e_slice):
        return np.nan
    lbl, n = ndimage.label(e_slice, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.nan
    sizes = np.bincount(lbl.ravel())[1:]
    atot = int(e_slice.sum())
    amax = int(sizes.max()) if sizes.size else 0
    return float(amax / atot) if atot > 0 else np.nan


def _compactness_edge_metric(mask_yx: np.ndarray) -> float:
    from scipy import ndimage

    m = mask_yx.astype(bool)
    if m.sum() == 0:
        return np.nan
    eroded = ndimage.binary_erosion(m, structure=np.ones((3, 3), dtype=bool))
    boundary = m & ~eroded
    p = float(max(boundary.sum(), 1))
    a = float(m.sum())
    return float(4.0 * np.pi * a / (p * p))


def _centroid_latlon(mask_yx: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask_yx)
    if yy.size == 0:
        return (np.nan, np.nan)
    return (float(lat2d[yy, xx].mean()), float(lon2d[yy, xx].mean()))


def _cell_area_m2(lat1d: np.ndarray, lon1d: np.ndarray) -> np.ndarray:
    re = 6_371_000.0
    lat1d = np.asarray(lat1d, dtype=np.float64)
    lon1d = np.asarray(lon1d, dtype=np.float64)
    dlat = np.abs(np.diff(lat1d).mean()) * np.pi / 180.0
    dlon = np.abs(np.diff(lon1d).mean()) * np.pi / 180.0
    lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing="ij")
    phi = lat2d * np.pi / 180.0
    return (re * re) * dlat * dlon * np.cos(phi).astype(np.float32)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return float(2 * r * np.arcsin(min(1.0, np.sqrt(a))))


def _analysis_day_floor(
    time_hour: pd.Timestamp | np.datetime64,
    *,
    day_start_hour_utc: int,
) -> pd.Timestamp:
    ts = pd.Timestamp(time_hour)
    ad = (ts - pd.Timedelta(hours=int(day_start_hour_utc))).floor("D")
    return pd.Timestamp(ad.normalize())


def guess_time_name_for_da(da: xr.DataArray) -> str:
    for cand in ("time", "valid_time", "time_hour"):
        if cand in da.dims or cand in da.coords:
            return cand
    raise KeyError(f"No time dim on DataArray; dims={da.dims}")


def _guess_time_dim_ds(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"No time coordinate in hourly ERA5 dataset; coords={list(ds.coords)}")


def _months_tokens_from_tag(months_tag: str) -> list[str]:
    """e.g. '0506070809' -> ['05','06',…]; fallback MJJAS."""
    t = str(months_tag).strip()
    if len(t) >= 10 and len(t) % 2 == 0 and re.fullmatch(r"\d+", t):
        return [t[i : i + 2] for i in range(0, len(t), 2)]
    return ["05", "06", "07", "08", "09"]


def _open_era5_hourly_tp_cp_cape(
    in_dir: Path,
    *,
    filename_token: str,
    years: list[str],
    months_tag: str,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset:
    """Same file discovery as epe_01b when the raw directory is known (absolute path)."""
    if chunks is None:
        chunks = {"time": 24 * 31, "latitude": 64, "longitude": 64}
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"ERA5 hourly directory not found: {in_dir}")

    years_slug = years_filename_slug(years)
    months = _months_tokens_from_tag(months_tag)

    legacy = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_{years_slug}_m{months_tag}_hourly.nc"
    if legacy.exists():
        return xr.open_dataset(legacy, chunks=chunks)

    per_year: list[Path] = []
    for y in years:
        p = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m{months_tag}_hourly.nc"
        if not p.exists():
            per_year = []
            break
        per_year.append(p)
    if per_year:
        return xr.open_mfdataset(per_year, chunks=chunks, combine="by_coords")

    per_year_any: list[Path] = []
    for y in years:
        matches = sorted(in_dir.glob(f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m*_hourly.nc"))
        pref = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m{months_tag}_hourly.nc"
        if pref.exists() and pref in matches:
            per_year_any.append(pref)
            continue
        if len(matches) == 1:
            per_year_any.append(matches[0])
            continue
        per_year_any = []
        break
    if per_year_any:
        return xr.open_mfdataset(per_year_any, chunks=chunks, combine="by_coords")

    paths: list[Path] = []
    for y in years:
        for m in months:
            p = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m{m}_hourly.nc"
            if not p.exists():
                raise FileNotFoundError(
                    f"No hourly ERA5 file at {p}. Expected legacy, per-year, or year-month "
                    f"…_tp-cp-cape_{filename_token}_… under {in_dir} (same layout as epe_00/epe_01b)."
                )
            paths.append(p)
    return xr.open_mfdataset(paths, chunks=chunks, combine="by_coords")


def _era5_precip_var_from_duration_attrs(duration_ds: xr.Dataset) -> str:
    metric = str(duration_ds.attrs.get("hourly_precip_metric", "tp_hourly")).lower()
    if metric.startswith("cp"):
        return "cp"
    return "tp"


def _proxy_hourly_precip_from_duration(duration_ds: xr.Dataset) -> tuple[xr.DataArray, str] | None:
    required = {"time_hour", "time", "P_conv_total_grid_day", "n_conv_hours_grid"}
    if not required.issubset(set(duration_ds.variables) | set(duration_ds.coords)):
        return None

    day_start_hour_utc = int(duration_ds.attrs.get("day_start_hour_utc", 6))
    time_hour = pd.to_datetime(duration_ds["time_hour"].values)
    analysis_day = (time_hour - pd.to_timedelta(day_start_hour_utc, unit="h")).floor("D")
    analysis_day_start = analysis_day + pd.to_timedelta(day_start_hour_utc, unit="h")
    day_of_hour = xr.DataArray(
        pd.to_datetime(analysis_day_start).to_numpy(dtype="datetime64[ns]"),
        dims=("time_hour",),
        coords={"time_hour": duration_ds["time_hour"]},
        name="analysis_day_start",
    )

    # Daily labels may be stored on a shifted UTC axis; nearest daily match is sufficient.
    p_day = duration_ds["P_conv_total_grid_day"].sel(
        time=day_of_hour,
        method="nearest",
        tolerance=np.timedelta64(12, "h"),
    )
    n_day = duration_ds["n_conv_hours_grid"].sel(
        time=day_of_hour,
        method="nearest",
        tolerance=np.timedelta64(12, "h"),
    )
    proxy = (p_day / n_day.where(n_day > 0)).astype("float32")
    if "extreme_grid_any" in duration_ds:
        proxy = proxy.where(duration_ds["extreme_grid_any"] > 0)
    proxy.attrs.update(duration_ds["P_conv_total_grid_day"].attrs)
    proxy.name = "P_hour_proxy"
    return proxy, "proxy_from_duration"


def _load_p_hour_meters(
    duration_ds: xr.Dataset,
    *,
    hourly_precip_path: Path | None,
    hourly_var: str,
    era5_hourly_dir: Path | None = None,
) -> tuple[xr.DataArray, str]:
    if "P_hour" in duration_ds:
        return duration_ds["P_hour"], "P_hour_in_duration_nc"
    if "P_conv_hour" in duration_ds:
        return duration_ds["P_conv_hour"], "P_conv_hour_in_duration_nc"

    def _align_to_duration_grid(tp: xr.DataArray, *, label: str) -> tuple[xr.DataArray, str]:
        tdim = "time" if "time" in tp.dims else guess_time_name_for_da(tp)
        tp = tp.rename({tdim: "time_hour"})
        th = duration_ds["time_hour"]
        tp = tp.sel(time_hour=th.values, method="nearest", tolerance=np.timedelta64(3, "h"))
        lat_name, lon_name = guess_lat_lon(duration_ds)
        tp = tp.sel(
            **{lat_name: duration_ds[lat_name], lon_name: duration_ds[lon_name]},
            method="nearest",
        )
        return tp.astype("float32"), label

    if hourly_precip_path is not None:
        pth = Path(hourly_precip_path)
        if not pth.exists():
            raise FileNotFoundError(str(pth))
        raw = xr.open_dataset(pth)
        if hourly_var not in raw:
            raise KeyError(f"Variable {hourly_var!r} not in {pth}; vars={list(raw.data_vars)}")
        tp = raw[hourly_var]
        return _align_to_duration_grid(tp, label=f"external:{pth.name}")

    raw_error: Exception | None = None
    if era5_hourly_dir is not None:
        try:
            years_spec = str(duration_ds.attrs.get("years_spec", "")).strip()
            if not years_spec:
                raise ValueError(
                    "duration_ds.attrs missing 'years_spec'; cannot auto-open raw ERA5. "
                    "Set HOURLY_SOURCE_PATH or regenerate duration with epe_01b."
                )
            years = parse_years(years_spec)
            months_tag = str(duration_ds.attrs.get("months_tag", "0506070809"))
            filename_token = str(duration_ds.attrs.get("hourly_filename_token", "")).strip()
            if not filename_token:
                filename_token = str(duration_ds.attrs.get("region_label", "germany"))
            raw = _open_era5_hourly_tp_cp_cape(
                Path(era5_hourly_dir),
                filename_token=filename_token,
                years=years,
                months_tag=months_tag,
            )
            tdim = _guess_time_dim_ds(raw)
            if tdim != "time":
                raw = raw.rename({tdim: "time"})
            raw = raw.sel(time=raw["time"].dt.month.isin([5, 6, 7, 8, 9]))
            era_var = _era5_precip_var_from_duration_attrs(duration_ds)
            if era_var not in raw:
                raise KeyError(f"Expected '{era_var}' in raw hourly ERA5; vars={list(raw.data_vars)}")
            tp = raw[era_var].astype("float32")
            return _align_to_duration_grid(
                tp,
                label=f"era5_dir:{Path(era5_hourly_dir).name}:{era_var}",
            )
        except (FileNotFoundError, KeyError, ValueError) as e:
            raw_error = e

    proxy = _proxy_hourly_precip_from_duration(duration_ds)
    if proxy is not None:
        return proxy

    msg = (
        "Duration file has no 'P_hour'. Set HOURLY_SOURCE_PATH to an hourly NetCDF, "
        "pass era5_hourly_dir (e.g. cfg.era5_hourly_dir) to load raw ERA5 tp/cp like epe_01b, "
        "or regenerate convective_duration with --hourly-precip all|both. "
        "A duration-based proxy is used automatically when P_conv_total_grid_day and "
        "n_conv_hours_grid are available."
    )
    if raw_error is not None:
        msg = f"{msg} Raw ERA5 fallback failed: {raw_error}"
    raise ValueError(msg)


@dataclass(frozen=True)
class EventClassificationConfig:
    use_propagation_for_c: bool = True


def _classify_events(df: pd.DataFrame, cfg: EventClassificationConfig) -> pd.Series:
    if len(df) == 0:
        return pd.Series(dtype=object)
    ah = df["area_hours_m2h"].astype(float)
    dur = df["duration_h"].astype(float)
    prop = df["propagation_km"].astype(float)
    q33_ah, q66_ah = ah.quantile(0.33), ah.quantile(0.66)
    q33_d, q66_d = dur.quantile(0.33), dur.quantile(0.66)
    q66_p = prop.quantile(0.66)
    typ: list[str] = []
    for i in range(len(df)):
        a, d, pr = ah.iloc[i], dur.iloc[i], prop.iloc[i]
        big = a >= q66_ah and (d >= q66_d or (cfg.use_propagation_for_c and pr >= q66_p))
        small = a <= q33_ah and d <= q33_d
        if big:
            typ.append("C")
        elif small:
            typ.append("A")
        else:
            typ.append("B")
    return pd.Series(typ, index=df.index, dtype="object")


def _contiguous_runs(sorted_idx: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start, end) index runs on the time_hour axis."""
    if sorted_idx.size == 0:
        return []
    u = np.sort(np.unique(sorted_idx.astype(np.int64)))
    runs: list[tuple[int, int]] = []
    a = int(u[0])
    b = a
    for v in u[1:]:
        v = int(v)
        if v == b + 1:
            b = v
        else:
            runs.append((a, b + 1))
            a = b = v
    runs.append((a, b + 1))
    return runs


def _compute_hourly_scores(
    p_blk: np.ndarray,
    e_blk: np.ndarray,
    *,
    h1_percentile: float,
    wet_threshold_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tlen, _, _ = e_blk.shape
    h1 = np.full(tlen, np.nan, dtype=np.float64)
    h2 = np.full(tlen, np.nan, dtype=np.float64)
    h3 = np.full(tlen, np.nan, dtype=np.float64)
    for t in range(tlen):
        pt = p_blk[t]
        h1[t] = float(np.nanpercentile(pt, h1_percentile))
        wet = pt > wet_threshold_m
        if np.any(wet):
            w = pt[wet]
            p50 = float(np.percentile(w, 50))
            h2[t] = float(np.percentile(w, 99) / max(p50, wet_threshold_m))
        h3[t] = _hourly_h3_amax_over_atotal(e_blk[t])
    return h1, h2, h3


def _event_row_from_label(
    lab_vol: np.ndarray,
    lid: int,
    p_vol_m: np.ndarray,
    e_vol: np.ndarray,
    cell_area: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    times: np.ndarray,
    c_t: np.ndarray | None,
    *,
    day_start_hour_utc: int,
    rep_day_mode: str,
    precip_units: str | None,
    bbox: tuple[slice, slice, slice] | None = None,
) -> dict[str, Any] | None:
    # ``bbox`` (from ``ndimage.find_objects``) restricts the search to this
    # label's bounding box instead of rescanning the whole (potentially
    # decades-long) volume — needed for tractable runtime with many events.
    if bbox is not None:
        local = np.argwhere(lab_vol[bbox] == lid)
        if local.size == 0:
            return None
        t0 = bbox[0].start or 0
        y0 = bbox[1].start or 0
        x0 = bbox[2].start or 0
        tt = local[:, 0] + t0
        yy = local[:, 1] + y0
        xx = local[:, 2] + x0
    else:
        idx = np.argwhere(lab_vol == lid)
        if idx.size == 0:
            return None
        tt = idx[:, 0]
        yy = idx[:, 1]
        xx = idx[:, 2]
    ut = np.sort(np.unique(tt))
    duration_h = int(ut.size)

    areas_t: dict[int, float] = {}
    ah_contrib: dict[int, float] = {}
    i_t: dict[int, float] = {}
    compact_at_max_area = np.nan
    max_area = 0.0
    best_t_for_area = int(ut[0])
    c_vals: list[float] = []

    peak_val_m = -np.inf
    peak_lat = np.nan
    peak_lon = np.nan
    peak_t = int(ut[0])

    for t in ut:
        t = int(t)
        sel = tt == t
        ys = yy[sel]
        xs = xx[sel]
        m = np.zeros_like(e_vol[t], dtype=bool)
        m[ys, xs] = True
        m = m & e_vol[t].astype(bool)
        a_m2 = float((m * cell_area).sum())
        areas_t[t] = a_m2
        ah_contrib[t] = a_m2
        if m.sum() > 0:
            vals = p_vol_m[t][m]
            i_t[t] = float(np.nanpercentile(vals, 99.0))
            if c_t is not None and t < len(c_t):
                c_vals.append(float(c_t[t]))
            ys_m, xs_m = np.nonzero(m)
            if vals.size > 0 and np.any(np.isfinite(vals)):
                j = int(np.nanargmax(vals))
                v = float(vals[j])
                if v > peak_val_m:
                    peak_val_m = v
                    peak_lat = float(lat2d[ys_m[j], xs_m[j]])
                    peak_lon = float(lon2d[ys_m[j], xs_m[j]])
                    peak_t = t
        else:
            i_t[t] = np.nan
        if a_m2 > max_area:
            max_area = a_m2
            best_t_for_area = t
            compact_at_max_area = _compactness_edge_metric(m)

    area_hours = float(sum(ah_contrib.values()))
    a_max = float(max(areas_t.values())) if areas_t else 0.0
    i_max_m = float(np.nanmax(list(i_t.values()))) if i_t else np.nan
    i_max_mm = float(to_mm(i_max_m, precip_units))

    prev_lat: float | None = None
    prev_lon: float | None = None
    prop_km = 0.0
    for t in ut:
        t = int(t)
        sel = tt == t
        ys = yy[sel]
        xs = xx[sel]
        m = np.zeros_like(e_vol[t], dtype=bool)
        m[ys, xs] = True
        m = m & e_vol[t].astype(bool)
        plat, plon = _centroid_latlon(m, lat2d, lon2d)
        if prev_lat is not None and np.isfinite(plat):
            prop_km += _haversine_km(prev_lat, prev_lon, plat, plon)
        if np.isfinite(plat):
            prev_lat, prev_lon = plat, plon

    t_start = pd.Timestamp(times[ut[0]])
    t_end = pd.Timestamp(times[ut[-1]])

    if rep_day_mode == "peak_intensity":
        finite = {k: v for k, v in i_t.items() if np.isfinite(v)}
        if finite:
            best_t = int(max(finite, key=lambda k: finite[k]))
        else:
            best_t = int(max(ah_contrib, key=lambda k: ah_contrib[k]))
    else:
        best_t = int(max(ah_contrib, key=lambda k: ah_contrib[k]))
    rep_ts = pd.Timestamp(times[best_t])
    rep_day = _analysis_day_floor(rep_ts, day_start_hour_utc=day_start_hour_utc)

    s_score = float(max(area_hours, 1e-30) ** 0.7 * max(i_max_mm, 1e-6) ** 0.3)

    peak_time = pd.Timestamp(times[peak_t]) if np.isfinite(peak_val_m) else pd.NaT
    peak_mm = float(to_mm(peak_val_m, precip_units)) if np.isfinite(peak_val_m) else np.nan

    return {
        "duration_h": duration_h,
        "A_max_m2": a_max,
        "area_hours_m2h": area_hours,
        "I_max_mm": i_max_mm,
        "I_peak_mm": peak_mm,
        "lat_peak": peak_lat,
        "lon_peak": peak_lon,
        "peak_time_hour": peak_time,
        "propagation_km": prop_km,
        "compactness": compact_at_max_area,
        "mean_C_in_event": float(np.mean(c_vals)) if c_vals else np.nan,
        "t_start": t_start,
        "t_end": t_end,
        "rep_day": rep_day,
        "rep_time_hour": rep_ts,
        "best_t_area": best_t_for_area,
        "S_score": s_score,
    }


def build_object_event_catalog(
    duration_ds: xr.Dataset,
    *,
    hourly_precip_path: Path | None = None,
    hourly_var: str = "tp_hourly",
    era5_hourly_dir: Path | str | None = None,
    time_start: pd.Timestamp | str | None = None,
    time_end: pd.Timestamp | str | None = None,
    months: tuple[int, ...] | None = (5, 6, 7, 8, 9),
    h1_percentile: float = 99.0,
    wet_threshold_m: float = 1e-8,
    score_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    rep_day_mode: str = "max_area_hour",
    rank_by: str = "S_score",
    top_n: int | None = None,
    reg_map: pd.Series | None = None,
    classification_config: EventClassificationConfig | None = None,
) -> pd.DataFrame:
    """One row per 3D connected component on ``E_conv_hour`` (see module docstring).

    Hourly precipitation for H1/H2 is taken from, in order: embedded ``P_hour`` /
    ``P_conv_hour`` in the duration file; ``hourly_precip_path``; raw ERA5 under
    ``era5_hourly_dir`` (same file naming as epe_00/epe_01b, using ``years_spec`` /
    ``hourly_filename_token`` / ``months_tag`` from ``duration_ds.attrs``); or a
    duration-derived proxy from ``P_conv_total_grid_day / n_conv_hours_grid``.
    """
    if "E_conv_hour" not in duration_ds:
        raise KeyError("duration_ds must contain 'E_conv_hour' (epe_01b --hourly-mask conv).")
    if rep_day_mode not in {"max_area_hour", "peak_intensity"}:
        raise ValueError("rep_day_mode must be 'max_area_hour' or 'peak_intensity'")
    rank_by = str(rank_by).strip()
    if rank_by not in {"S_score", "I_max_mm", "I_peak_mm"}:
        raise ValueError("rank_by must be 'S_score', 'I_max_mm', or 'I_peak_mm'")

    day_start_hour_utc = int(duration_ds.attrs.get("day_start_hour_utc", 6))
    era_dir = Path(era5_hourly_dir) if era5_hourly_dir is not None else None
    p_hour, p_source = _load_p_hour_meters(
        duration_ds,
        hourly_precip_path=hourly_precip_path,
        hourly_var=hourly_var,
        era5_hourly_dir=era_dir,
    )
    precip_units = p_hour.attrs.get("units")

    lat_name, lon_name = guess_lat_lon(duration_ds)
    lat1d = duration_ds[lat_name].values.astype(np.float64)
    lon1d = duration_ds[lon_name].values.astype(np.float64)
    lat2d, lon2d = np.meshgrid(lat1d, lon1d, indexing="ij")
    cell_area = _cell_area_m2(lat1d, lon1d).astype(np.float32)

    th = duration_ds["time_hour"]
    tvals = pd.to_datetime(th.values)
    msel = np.ones(len(tvals), dtype=bool)
    if months is not None:
        msel &= np.isin(pd.Series(tvals).dt.month.values, list(months))
    if time_start is not None:
        msel &= tvals >= pd.Timestamp(time_start)
    if time_end is not None:
        msel &= tvals <= pd.Timestamp(time_end)
    sel_idx = np.sort(np.nonzero(msel)[0].astype(np.int64))
    if sel_idx.size == 0:
        return pd.DataFrame()

    sel_set = set(sel_idx.tolist())
    runs = _contiguous_runs(sel_idx)

    h1_sel: list[float] = []
    h2_sel: list[float] = []
    h3_sel: list[float] = []
    for start, end in runs:
        sl = slice(start, end)
        e_blk = duration_ds["E_conv_hour"].isel(time_hour=sl).values.astype(bool)
        p_blk = p_hour.isel(time_hour=sl).values.astype(np.float32)
        h1, h2, h3 = _compute_hourly_scores(
            p_blk, e_blk, h1_percentile=h1_percentile, wet_threshold_m=wet_threshold_m
        )
        for tix in range(start, end):
            if tix not in sel_set:
                continue
            j = tix - start
            h1_sel.append(float(h1[j]))
            h2_sel.append(float(h2[j]))
            h3_sel.append(float(h3[j]))

    h1_cat = np.asarray(h1_sel, dtype=np.float64)
    h2_cat = np.asarray(h2_sel, dtype=np.float64)
    h3_cat = np.asarray(h3_sel, dtype=np.float64)
    z1 = _robust_z(h1_cat)
    z2 = _robust_z(h2_cat)
    z3 = _robust_z(h3_cat)
    w1, w2, w3 = score_weights
    c_cat = w1 * z1 + w2 * z2 + w3 * z3

    struct = _structure_8neighbour_xy_samecell_time()
    rows: list[dict[str, Any]] = []
    eid = 0
    c_cursor = 0

    from scipy import ndimage

    for start, end in runs:
        sl = slice(start, end)
        e_blk = duration_ds["E_conv_hour"].isel(time_hour=sl).values.astype(bool)
        p_blk = p_hour.isel(time_hour=sl).values.astype(np.float32)
        times_blk = tvals[sl]
        c_t_blk = np.full(e_blk.shape[0], np.nan, dtype=np.float64)
        for tix in range(start, end):
            if tix not in sel_set:
                continue
            j = tix - start
            c_t_blk[j] = c_cat[c_cursor]
            c_cursor += 1

        lab, nfeat = ndimage.label(e_blk, structure=struct)
        bboxes = ndimage.find_objects(lab)
        for lid in range(1, nfeat + 1):
            m = _event_row_from_label(
                lab,
                lid,
                p_blk,
                e_blk,
                cell_area,
                lat2d,
                lon2d,
                times_blk,
                c_t_blk,
                day_start_hour_utc=day_start_hour_utc,
                rep_day_mode=rep_day_mode,
                precip_units=precip_units,
                bbox=bboxes[lid - 1],
            )
            if m is None:
                continue
            eid += 1
            rows.append(
                {
                    "event_id": eid,
                    "time_hour_run_start": pd.Timestamp(times_blk[0]),
                    "duration_h": m["duration_h"],
                    "A_max_m2": m["A_max_m2"],
                    "area_hours_m2h": m["area_hours_m2h"],
                    "I_max_mm": m["I_max_mm"],
                    "I_peak_mm": m["I_peak_mm"],
                    "lat_peak": m["lat_peak"],
                    "lon_peak": m["lon_peak"],
                    "peak_time_hour": m["peak_time_hour"],
                    "propagation_km": m["propagation_km"],
                    "compactness": m["compactness"],
                    "mean_C_in_event": m["mean_C_in_event"],
                    "t_start": m["t_start"],
                    "t_end": m["t_end"],
                    "rep_day": m["rep_day"],
                    "rep_time_hour": m["rep_time_hour"],
                    "S_score": m["S_score"],
                    "P_hour_source": p_source,
                    "day_start_hour_utc": day_start_hour_utc,
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    cfg = classification_config or EventClassificationConfig()
    df = df.sort_values(rank_by, ascending=False).reset_index(drop=True)
    df["event_type"] = _classify_events(df, cfg)
    if reg_map is not None:
        rd = pd.to_datetime(df["rep_day"]).dt.floor("D")
        df["wp_z500_regime"] = rd.map(reg_map)

    if top_n is not None:
        df = df.head(int(top_n)).reset_index(drop=True)
    return df


__all__ = [
    "build_object_event_catalog",
    "EventClassificationConfig",
]
