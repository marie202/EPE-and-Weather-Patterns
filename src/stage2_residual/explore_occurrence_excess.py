#!/usr/bin/env python3
"""
Excess-over-threshold (POT) severity metric for the Z500 EPE statistics CSV pipeline.

Companion to ``common.epe_04_boxplot_render.compute_epe_04_core_metrics``: raw precip
amount conditioned on a fixed global percentile threshold converges across regimes at high
percentiles (every regime's "p99.9 day" sits close to the same threshold by construction).
Excess *over* that threshold -- ``precip - P_q`` -- does not have this problem, so it actually
separates regimes by how far into the tail they push events.

Lives here rather than in ``epe_04_boxplot_render.py`` since it was developed and validated
separately; ``steps/stage1/04_export_statistics_csv.py`` merges
``compute_z500_excess_core_metrics``'s output into that module's ``core_metrics`` dict, so it
flows through the same CSV/notebook path (``epe_boxplot_values_<region>.csv``,
``metric="excess_mean"``/``"excess_median"``) as every other Z500 boxplot metric -- see
``2_3_Results_Z500_Stats.ipynb``.

The RR-level analogue of this metric uses the same ``precip - P_q_hourly`` exceedance recipe
but lives directly in ``stage2_residual.plot_rr_statistics._excess_spatial_precip_per_day``
(RR already re-opens its own duration file there for every other metric); nothing here is
RR-specific, though the duration-file fallback below reuses that same function.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from epe_preprocessing.epe_functions import (
    _daily_extreme_mask,
    _intensity_has_finite_values,
    dataset_rename_time,
    guess_lat_lon,
    to_mm,
)

from .plot_rr_statistics import _excess_spatial_precip_per_day, _extreme_day_mask, _open_duration_dataset
from .plot_subregimes import _norm_dates


def _z500_excess_spatial_stats(
    ext_c: xr.Dataset,
) -> tuple[xr.DataArray | None, xr.DataArray | None, xr.DataArray | None]:
    """Spatial mean/median/max of (P - P_q) on E_conv-flagged cells, per day (mm).

    ``P_q`` loses its ``units`` attr in epe_02 (``quantile(..., keep_attrs=False)``) but is
    computed directly from ``P``, so ``P``'s units string applies to both. Returns
    ``(None, None, None)`` when ``ext_c`` doesn't carry real per-cell ``P_q`` -- e.g. the
    synthetic stand-in ``epe.epe_functions.synthetic_extremes_from_duration`` builds when no
    ``extremes_*.nc`` exists locally, only ``convective_duration_*.nc`` (the common case for
    this project: epe_02 is rarely run, epe_01b almost always is) -- so the caller can fall
    back to ``_z500_excess_from_duration_fallback``.
    """
    if not {"P", "P_q", "E_conv"}.issubset(ext_c.data_vars):
        return None, None, None
    lat_e, lon_e = guess_lat_lon(ext_c)
    spatial = (lat_e, lon_e)
    units = str(ext_c["P"].attrs.get("units", ""))
    excess = to_mm(ext_c["P"].astype(float), units) - to_mm(ext_c["P_q"].astype(float), units)
    e_day = ext_c["E_conv"].astype(int) > 0
    excess_masked = excess.where(e_day)
    mean_da = excess_masked.mean(dim=spatial, skipna=True)
    median_da = excess_masked.median(dim=spatial, skipna=True)
    max_da = excess_masked.max(dim=spatial, skipna=True)
    if not _intensity_has_finite_values(mean_da):
        return None, None, None
    return mean_da, median_da, max_da


def _z500_excess_from_duration_fallback(
    duration_audit_path: Path | None, reg_day_ds: xr.Dataset,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, np.ndarray | None]]:
    """Recompute excess straight from the duration file's ``P_q_hourly`` when ``ext_c`` has no
    real ``P_q``, reusing the RR level's ``_excess_spatial_precip_per_day``, and align it to the
    Z500 regime day index the same way ``compute_epe_04_core_metrics`` aligns duration/
    connectivity metrics. Returns
    ``(labels_arr, flag_arr, {"excess_mean": ..., "excess_median": ..., "excess_max": ...})``,
    or ``(None, None, {})`` if no duration file is available or no days overlap.
    """
    if duration_audit_path is None or not Path(duration_audit_path).is_file():
        return None, None, {}
    dur = dataset_rename_time(_open_duration_dataset(duration_audit_path)).load()
    dur_days = _norm_dates(dur["time"].values)
    dur = dur.assign_coords(day=("time", dur_days.values.astype("datetime64[ns]")))
    common = np.intersect1d(dur_days.values.astype("datetime64[ns]"), reg_day_ds["day"].values)
    if common.size == 0:
        return None, None, {}
    dur = dur.sel(time=dur["day"].isin(common))
    labels_arr = reg_day_ds["regime_labels"].sel(day=dur["day"].values).values.astype(float)
    flag_arr = np.asarray(_extreme_day_mask(dur), dtype=bool)
    excess_arrays = {
        "excess_mean": _excess_spatial_precip_per_day(dur, agg="mean"),
        "excess_median": _excess_spatial_precip_per_day(dur, agg="median"),
        "excess_max": _excess_spatial_precip_per_day(dur, agg="max"),
    }
    return labels_arr, flag_arr, excess_arrays


def compute_z500_excess_core_metrics(
    ext_c: xr.Dataset, labels_ext: xr.DataArray, regimes: np.ndarray, *,
    reg_day_ds: xr.Dataset, duration_audit_path: Path | None = None,
) -> dict[str, dict]:
    """Same ``{"data", "ticks", "regime_ids", "title", "ylabel"}`` shape per metric as
    ``epe_04_boxplot_render.compute_epe_04_core_metrics`` -- merged into that dict by
    ``export_regime_z500_statistics_csv.py`` so excess-over-threshold flows into the same
    ``epe_boxplot_values``/``epe_boxplot_meta`` CSVs and notebook figure loop as every other
    Z500 boxplot metric. Best-effort: returns ``{}`` if neither a real ``P_q`` nor a duration
    file with ``P_q_hourly`` is available, rather than raising -- consistent with how
    ``compute_epe_04_core_metrics`` treats an unavailable metric as simply absent.
    """
    mean_da, median_da, max_da = _z500_excess_spatial_stats(ext_c)
    day_flag_da = _daily_extreme_mask(extremes_ds=ext_c, duration_ds=None) if mean_da is not None else None

    if mean_da is not None and day_flag_da is not None:
        labels_arr = np.asarray(labels_ext.values, dtype=float)
        flag_arr = np.asarray(day_flag_da.values, dtype=bool)
        excess_arrays = {
            "excess_mean": np.asarray(mean_da.values, dtype=float),
            "excess_median": np.asarray(median_da.values, dtype=float),
            "excess_max": np.asarray(max_da.values, dtype=float),
        }
    else:
        labels_arr, flag_arr, excess_arrays = _z500_excess_from_duration_fallback(
            duration_audit_path, reg_day_ds,
        )
        if labels_arr is None:
            return {}

    stat_labels = {"excess_mean": "mean", "excess_median": "median", "excess_max": "max"}
    out: dict[str, dict] = {}
    for key, values in excess_arrays.items():
        if values is None:
            continue
        stat_label = stat_labels.get(key, key)
        data, ticks, reg_ids = [], [], []
        for r in regimes:
            sel = (labels_arr == int(r)) & flag_arr
            vals = np.asarray(values)[sel]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            ticks.append(f"{int(r)}\n(n={vals.size})")
            reg_ids.append(int(r))
        if data:
            out[key] = dict(
                data=data, ticks=ticks, regime_ids=reg_ids,
                title=f"Excess over threshold (P-P_q), {stat_label} by regime (extreme days only)",
                ylabel=f"{stat_label} excess over P_q on flagged cells (mm)",
            )
    return out
