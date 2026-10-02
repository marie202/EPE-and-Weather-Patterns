#!/usr/bin/env python3
"""
Z500 mean vs. std-dev vs. RGB-composite, per RR sub-regime/bin.

Standalone sibling of ``plot_subregimes.py``: reuses the same RR/Z500 regime-file resolution
and the same per-(Z500 parent regime, RR sub-regime) day-grouping logic (>= MIN_DAYS). One
standalone PNG (no bundled PDF) is written per Z500 parent regime:

  Figure (one per Z500 parent regime). Rows = the RR sub-regimes/bins that occur within that
  Z500 regime (>= MIN_DAYS). Columns = [Z500 anomaly MEAN (sub-group days) | Z500 anomaly
  STD-DEV (sub-group days) | additive-RGB composite of the impact fields for that same
  sub-group, same recipe as plot_rgb_overview.py]. A ternary colour-key legend for the RGB
  column sits on the outer right of every figure. Black isolines on the first two columns show
  the mean raw Z500 height (gpm) for that row's days, same convention as plot_subregimes.py.

The RGB column reuses plot_rgb_overview.py's additive-RGB machinery unchanged (same global,
pooled-across-every-group 95th-percentile per-field scale, so colours are comparable across
every row/figure of a run) rather than re-deriving it here.

``--mode logistic`` reads the Stage-2 [LOGISTIC] continuous ``predicted_extreme_prob`` index
(rr_02_stage2_logistic.py's output) instead of a discrete k-means RR regimes file, same as
``stage2_residual.plot_composites --mode logistic`` — this is the production RR path
(``HOW_TO_RUN_Jobs.md`` always runs rr_02/rr_022/rr_03 with ``MODE=logistic``); rows are then
labelled "Bin<b>" instead of "RR<rr>". The discrete ``bbox``/``regime`` modes are appendix-only
k-means paths and may not exist on disk for a given region/bbox.

Usage:
    python -m stage2_residual.plot_subregimes_variance \
        --region europe --bbox-step B5 --module-root /path/to/detect_weather
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from common.colorbar_discretize import discretize_diverging, discretize_sequential_from_floor, style_discrete_ticks

from .plot_results import (
    _dw_root,
    _find_anomaly,
    _find_raw_z500,
    _find_rr_regimes,
    _find_z500_regimes,
    _load_labels,
)
from .plot_rgb_overview import (
    _CHANNELS,
    _LEGEND_PAD,
    _PANEL_SIZE,
    _add_rgb_legend,
    _draw_rgb_panel,
    _get_regime_alignment,
    _global_field_scale,
    _render_rgb,
)
from . import rr_geo
import common.regime_alignment as regime_alignment
from common.physics import G0, z500_anomaly_units_look_physical
from common.fixed_color_scales import (
    Z500_MEAN_ANOM_GPM,
    Z500_STD_GPM,
    default_fixed_scales_path,
    load_fixed_color_scales,
)

MIN_DAYS = 10
DEFAULT_N_BINS = 2
_G = 9.80665  # m/s² — divides raw ERA5 geopotential (m²/s²) to give geopotential metres (gpm)


# ---------------------------------------------------------------------------
# Data containers + loaders (copied from plot_subregimes.py — kept local so this
# script stands alone; see module docstring)
# ---------------------------------------------------------------------------

@dataclass
class FieldData:
    data: np.ndarray          # (time, lat, lon)
    dates: pd.DatetimeIndex   # normalised to day
    lats: np.ndarray
    lons: np.ndarray
    units: str = ""           # native physical units, read from the netCDF variable attrs


def _norm_dates(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(times))).normalize()


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    lat = next((d for d in da.dims if "lat" in d.lower()), None)
    lon = next((d for d in da.dims if "lon" in d.lower()), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not find lat/lon dims in {da.dims}")
    return lat, lon


def _load_field(path: Path) -> FieldData | None:
    try:
        ds = xr.open_dataset(path)
        var = next((v for v in ds.data_vars if "anom" in v.lower()), list(ds.data_vars)[0])
        da = ds[var].load()
        ds.close()
    except Exception as e:
        print(f"[rr_subregimes_variance]   WARNING: could not load {path.name}: {e}")
        return None
    lat_dim, lon_dim = _spatial_dims(da)
    time_dim = next((d for d in da.dims if d not in (lat_dim, lon_dim)), da.dims[0])
    da = da.transpose(time_dim, lat_dim, lon_dim)
    return FieldData(
        data=da.values,
        dates=_norm_dates(da[time_dim].values),
        lats=da[lat_dim].values,
        lons=da[lon_dim].values,
        units=str(da.attrs.get("units", "")).strip(),
    )


def _load_raw_z500_field(path: Path) -> FieldData | None:
    """Load raw ERA5 Z500 geopotential and convert to geopotential metres (gpm)."""
    try:
        ds = xr.open_dataset(path)
        var = "z" if "z" in ds.data_vars else next(iter(ds.data_vars))
        da = ds[var].load()
        ds.close()
    except Exception as e:
        print(f"[rr_subregimes_variance] WARNING: could not load raw Z500 {path.name}: {e}")
        return None
    if "pressure_level" in da.dims:
        da = da.isel(pressure_level=0, drop=True)
    lat_dim, lon_dim = _spatial_dims(da)
    time_dim = next((d for d in da.dims if d not in (lat_dim, lon_dim)), da.dims[0])
    da = da.transpose(time_dim, lat_dim, lon_dim)
    return FieldData(
        data=da.values.astype(float) / _G,
        dates=_norm_dates(da[time_dim].values),
        lats=da[lat_dim].values,
        lons=da[lon_dim].values,
        units="gpm",
    )


def _crop_field_to_bbox(fd: FieldData | None, bbox: dict[str, float] | None) -> FieldData | None:
    """Restrict a FieldData's grid (and data) to a bbox's N/S/W/E window.

    Raw Z500 is loaded once per REGION (not per bbox step) from a domain-wide file; left
    uncropped, contour levels derived from the whole domain's min/max end up poorly spaced
    across whatever small bbox is actually being plotted (see plot_subregimes.py's identical
    helper for the full rationale).
    """
    if fd is None or bbox is None:
        return fd
    lat_mask = (fd.lats >= bbox["S"]) & (fd.lats <= bbox["N"])
    lon_mask = (fd.lons >= bbox["W"]) & (fd.lons <= bbox["E"])
    if not lat_mask.any() or not lon_mask.any():
        return fd
    return FieldData(
        data=fd.data[:, lat_mask, :][:, :, lon_mask],
        dates=fd.dates,
        lats=fd.lats[lat_mask],
        lons=fd.lons[lon_mask],
        units=fd.units,
    )


_CONTOUR_STEP_CANDIDATES_GPM = (40.0, 20.0, 10.0, 5.0, 2.0, 1.0)


def _z500_raw_contour_levels(mean_gpm: np.ndarray, step: float = 40.0, min_levels: int = 3) -> np.ndarray:
    """Evenly-spaced contour levels for a raw Z500 field in gpm (typically 5000-5900).

    Starts at ``step`` gpm spacing and falls back to finer steps (20/10/5/2/1) if the
    field's own value range is too narrow for a fixed 40 gpm step to produce at least
    ``min_levels`` isolines -- see ``plot_subregimes._z500_raw_contour_levels`` (this is a
    copy, kept in sync with it).
    """
    finite = mean_gpm[np.isfinite(mean_gpm)]
    if finite.size == 0:
        return np.arange(5100.0, 5900.0, step)
    data_min, data_max = float(finite.min()), float(finite.max())
    candidates = [step] + [s for s in _CONTOUR_STEP_CANDIDATES_GPM if s < step]
    chosen = candidates[-1]
    for cand in candidates:
        # Count multiples of cand that actually fall inside [data_min, data_max] --
        # a contour level outside the field's real range never gets drawn, so padding
        # the array with floor/ceil-derived levels beyond that range doesn't help.
        n_inside = int(np.floor(data_max / cand) - np.ceil(data_min / cand)) + 1
        chosen = cand
        if n_inside >= min_levels:
            break
    lo = np.floor(data_min / chosen) * chosen
    hi = np.ceil(data_max / chosen) * chosen + chosen
    return np.arange(lo, hi, chosen)


# ---------------------------------------------------------------------------
# Composite primitives over an arbitrary day-mask
# ---------------------------------------------------------------------------

def _composite_over_dates(fd: FieldData, group_dates: pd.DatetimeIndex) -> np.ndarray | None:
    sel = fd.dates.isin(group_dates)
    if sel.sum() == 0:
        return None
    return np.nanmean(fd.data[sel], axis=0)


def _composite_over_dates_std(fd: FieldData, group_dates: pd.DatetimeIndex) -> np.ndarray | None:
    """Per-cell standard deviation across a day-group's days — variability, not the mean pattern."""
    sel = fd.dates.isin(group_dates)
    if sel.sum() == 0:
        return None
    return np.nanstd(fd.data[sel], axis=0)


# ---------------------------------------------------------------------------
# Drawing helpers (copied from plot_subregimes.py)
# ---------------------------------------------------------------------------

def _draw_map(ax, is_geo, lons, lats, data, *, cmap, norm=None, vmin=None, vmax=None):
    tr = rr_geo.data_transform() if is_geo else None
    kw = dict(cmap=cmap, shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if tr is not None:
        kw["transform"] = tr
    return ax.pcolormesh(lons, lats, data, **kw)


def _overlay_z500_isolines(ax, is_geo, lons, lats, field, *, levels=7, label_every: int = 2) -> None:
    """Overlay labelled Z500 (mean) contour lines on an existing map panel."""
    if field is None:
        return
    tr = rr_geo.data_transform() if is_geo else None
    ckw = dict(colors="k", linewidths=0.6, levels=levels, zorder=15)
    if tr is not None:
        ckw["transform"] = tr
    try:
        cs = ax.contour(lons, lats, field, **ckw)
        label_lvls = cs.levels[::label_every] if label_every > 1 else cs.levels
        ax.clabel(cs, levels=label_lvls, inline=True, fontsize=8.5, fmt="%.0f")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Figure: Z500 mean | Z500 std-dev | RGB composite, per RR sub-regime/bin row
# ---------------------------------------------------------------------------

def _page_z500_variance(
    z500_val, rr_present, group_dates_by_rr, fd_z500, fields_data, impact_fields, channel_of,
    global_scales, ref_shape, region, bbox, bbox_dict, module_root,
    z500_raw_fd: "FieldData | None" = None,
    row_label_prefix: str = "SR",
    mean_vmin: float | None = None, mean_vmax: float | None = None,
    std_vmin: float | None = None, std_vmax: float | None = None,
):
    """One figure per Z500 parent regime. Columns: [Z500 mean | Z500 std-dev | RGB composite],
    rows: RR sub-regimes/bins. Ternary legend for the RGB column on the outer right.
    """
    ncol = 3
    nrow = len(rr_present)
    if nrow == 0:
        return None

    panels_w = _PANEL_SIZE * ncol
    fig_w = panels_w + _LEGEND_PAD
    fig_h = _PANEL_SIZE * nrow + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    panels_frac = panels_w / fig_w

    z_lats, z_lons = fd_z500.lats, fd_z500.lons
    ref_fd = fields_data[impact_fields[0]]

    # Magnitude-aware gpm/normalized detection (common.physics.
    # z500_anomaly_units_look_physical), same guard the Z500-only pipeline applies -- a raw
    # units-string check alone isn't enough: a Yeo-Johnson-transformed (~N(0,1)) anomaly file can
    # carry a stale "m**2 s**-2"-looking units attr forward from before the transform, which
    # would otherwise get divided by g0 a second time (see preprocessing_helpers.
    # apply_yeo_johnson's docstring).
    z500_unit_scale = (
        1.0 / G0 if z500_anomaly_units_look_physical(xr.DataArray(fd_z500.data), fd_z500.units) else 1.0
    )

    mean_sub: dict[int, np.ndarray] = {}
    std_sub: dict[int, np.ndarray] = {}
    for rr in rr_present:
        gdates = group_dates_by_rr[rr]
        m = _composite_over_dates(fd_z500, gdates)
        s = _composite_over_dates_std(fd_z500, gdates)
        if m is not None:
            mean_sub[rr] = m * z500_unit_scale if z500_unit_scale != 1.0 else m
        if s is not None:
            # std-dev scales linearly too (std(k*x) = |k|*std(x)).
            std_sub[rr] = s * z500_unit_scale if z500_unit_scale != 1.0 else s

    mean_stack = list(mean_sub.values())
    mean_cmap = mean_norm = mean_extend = None
    if mean_vmin is not None and mean_vmax is not None:
        mv = max(abs(float(mean_vmin)), abs(float(mean_vmax)), 1e-6)
        mean_cmap, mean_norm, mean_extend = discretize_diverging(-mv, mv, "RdBu_r")
    elif mean_stack:
        mv = max(float(np.nanpercentile(np.abs(np.stack(mean_stack)), 98)), 1e-6)
        mean_cmap, mean_norm, mean_extend = discretize_diverging(-mv, mv, "RdBu_r")

    std_stack = list(std_sub.values())
    std_cmap = std_norm = std_extend = None
    if std_vmax is not None:
        sv = max(float(std_vmax), 1e-6)
        std_cmap, std_norm, std_extend = discretize_sequential_from_floor(sv, "YlGnBu", vmin=0.0)
    elif std_stack:
        sv = max(float(np.nanpercentile(np.stack(std_stack), 98)), 1e-6)
        std_cmap, std_norm, std_extend = discretize_sequential_from_floor(sv, "YlGnBu", vmin=0.0)

    # Raw Z500 mean per sub-group, for the isoline overlay (one set of levels per row).
    z500_raw_sub: dict[int, np.ndarray] = {}
    z_lats_raw = z_lons_raw = None
    if z500_raw_fd is not None:
        z_lats_raw, z_lons_raw = z500_raw_fd.lats, z500_raw_fd.lons
        for rr in rr_present:
            c = _composite_over_dates(z500_raw_fd, group_dates_by_rr[rr])
            if c is not None:
                z500_raw_sub[rr] = c

    z_extent = rr_geo.map_extent(bbox_dict, z_lats, z_lons)
    rgb_extent = rr_geo.map_extent(bbox_dict, ref_fd.lats, ref_fd.lons)

    for row, rr in enumerate(rr_present):
        n_days = len(group_dates_by_rr[rr])
        z_raw_rr = z500_raw_sub.get(rr)
        z_iso_levels = _z500_raw_contour_levels(z_raw_rr) if z_raw_rr is not None else 7

        # Col 0: Z500 anomaly mean.
        ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + 1), extent=z_extent)
        mc = mean_sub.get(rr)
        if mc is not None and mean_norm is not None:
            im = _draw_map(ax, is_geo, z_lons, z_lats, mc, cmap=mean_cmap, norm=mean_norm)
            style_discrete_ticks(plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=mean_extend))
        _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
        rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
        ax.set_ylabel(f"{row_label_prefix}{rr}\n{n_days} d", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 1: Z500 anomaly std-dev.
        ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + 2), extent=z_extent)
        sc = std_sub.get(rr)
        if sc is not None and std_norm is not None:
            im = _draw_map(ax, is_geo, z_lons, z_lats, sc, cmap=std_cmap, norm=std_norm)
            style_discrete_ticks(plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=std_extend))
        _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
        rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 2: additive-RGB composite of the impact fields, same sub-group days.
        ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + 3), extent=rgb_extent)
        comp_by_field = {
            f: _composite_over_dates(fields_data[f], group_dates_by_rr[rr]) for f in impact_fields
        }
        rgb_img = _render_rgb(comp_by_field, impact_fields, global_scales, ref_shape)
        overlay_lons, overlay_lats, overlay_field = (
            (z_lons_raw, z_lats_raw, z_raw_rr) if z_raw_rr is not None else (z_lons, z_lats, mc)
        )
        _draw_rgb_panel(
            ax, is_geo, ref_fd.lats, ref_fd.lons, rgb_img,
            bbox_dict=bbox_dict, module_root=module_root, region=region, bbox_step=bbox,
            z_lons=overlay_lons, z_lats=overlay_lats, z500_mean=overlay_field, levels=z_iso_levels,
        )
        ax.text(0.02, 0.02, f"n={n_days}", transform=ax.transAxes, fontsize=6, color="black",
                bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor="none"))
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=(0.0, 0.03, panels_frac, 0.96))
    _add_rgb_legend(fig, impact_fields, channel_of, panels_frac)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_subregime_variance_plots(
    module_root, region, bbox_step, smooth_window=11, out_dir=None, mode=None,
    z500_cluster_method="ward", area_weight_tag: str = "",
    n_bins: int = DEFAULT_N_BINS, output_tag: str = "", bin_overrides_path: Path | None = None,
    formats: tuple[str, ...] = ("png",),
    mean_vmin: float | None = None, mean_vmax: float | None = None,
    std_vmin: float | None = None, std_vmax: float | None = None,
    composite_out: Path | None = None, no_render: bool = False,
) -> list[Path]:
    """``mode="logistic"`` reads the Stage-2 [LOGISTIC] continuous ``predicted_extreme_prob``
    index (rr_02_stage2_logistic.py's output) instead of a discrete k-means RR regimes file,
    quantile-binning each Z500 regime's own index into ``n_bins`` groups at plot time (bin 1 =
    lowest probability, bin ``n_bins`` = highest) — see
    ``stage2_residual.plot_composites.load_composite_inputs_logistic`` for the identical
    recipe this mirrors. Rows are then labelled "Bin<b>" instead of "RR<rr>". Any other
    ``mode`` (``"bbox"``/``"regime"``/``None``) reads the discrete k-means RR regimes file, same
    as before.

    Writes one PNG (and/or PDF, per ``formats``) per Z500 parent regime — no bundled multi-page
    report — named ``..._Z<z>.<ext>``, where ``<z>`` is the canonical (physical) regime id from
    ``regime_alignment_<region>.yaml`` when ``z500_cluster_method="ward"`` (so the same physical
    regime keeps the same filename across bbox steps/reruns), falling back to the raw
    per-bbox-step cluster id for "kmeans". Returns the list of paths written.
    """
    dw = _dw_root(module_root)
    is_logistic = mode == "logistic"
    per_z500_fields: dict[int, list[str]] | None = None

    if is_logistic:
        from .plot_results import _find_rr_logistic_merged
        from .stage2_logistic import bin_predicted_prob_per_regime, load_bin_overrides_yaml

        rr_path = _find_rr_logistic_merged(dw, region, bbox_step, output_tag=output_tag,
                                            z500_cluster_method=z500_cluster_method)
        if rr_path is None:
            raise FileNotFoundError(
                f"No merged Stage-2 [LOGISTIC] index file for {region} {bbox_step} "
                f"(z500_cluster_method={z500_cluster_method}) — run stage2_logistic.py first."
            )
        print(f"[rr_subregimes_variance] RR logistic index: {rr_path.name}")

        with xr.open_dataset(rr_path) as ds:
            time_dim = ds["predicted_extreme_prob"].dims[0]
            rr_times = pd.DatetimeIndex(ds[time_dim].values)
            predicted_prob = np.asarray(ds["predicted_extreme_prob"].values, dtype=np.float64)
            z500_regime = np.asarray(ds["z500_regime"].values, dtype=int)
            per_regime_fields_attr = ds.attrs.get("per_regime_fields")
        rr_dates = _norm_dates(rr_times)

        bin_edges_by_regime = {}
        if bin_overrides_path is not None:
            bin_edges_by_regime = load_bin_overrides_yaml(bin_overrides_path, bbox_step)
            print(f"[rr_subregimes_variance] Bin overrides: {bin_overrides_path.name} -> "
                  f"{ {z: len(e) + 1 for z, e in bin_edges_by_regime.items()} } bins by regime "
                  f"(others kept as a single undivided group -- see stage2_logistic."
              f"bin_predicted_prob_per_regime)")

        rr_labels = bin_predicted_prob_per_regime(
            predicted_prob, z500_regime, n_bins=n_bins, bin_edges_by_regime=bin_edges_by_regime,
        )
        n_rr = int(rr_labels.max()) if len(rr_labels) else int(n_bins)
        z500_on_rr = z500_regime
        z500_vals_raw = sorted({int(v) for v in z500_on_rr if np.isfinite(v) and v > 0})
        if z500_cluster_method == "ward":
            _, _groups = _get_regime_alignment(region)
            z500_vals = regime_alignment.canonical_order(z500_vals_raw, _groups, bbox_step)
            z500_canonical_map = regime_alignment.relabel_ids(z500_vals_raw, _groups, bbox_step)
        else:
            z500_vals = z500_vals_raw
            z500_canonical_map = {}
        row_label_prefix = "SR"
        print(f"[rr_subregimes_variance] {n_rr} bins/regime, Z500 regimes: {z500_vals}, "
              f"{len(rr_dates)} days, n_bins={n_bins}")

        if per_regime_fields_attr:
            per_z500_fields = {int(k): v for k, v in json.loads(per_regime_fields_attr).items()}
            fields = sorted({f for fs in per_z500_fields.values() for f in fs} | {"z500"})
        else:
            fields = ["z500"]
    else:
        rr_path = _find_rr_regimes(dw, region, bbox_step, module_root=module_root, mode=mode,
                                    z500_cluster_method=z500_cluster_method)
        if rr_path is None:
            raise FileNotFoundError(
                f"No RR regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})"
            )
        print(f"[rr_subregimes_variance] RR regimes:   {rr_path.name}")

        rr_labels, rr_times = _load_labels(rr_path)
        rr_labels = rr_labels.astype(int)
        rr_dates = _norm_dates(rr_times)
        n_rr = int(rr_labels.max())

        z500_path = _find_z500_regimes(dw, region, bbox_step,
                                        cluster_method=z500_cluster_method or "kmeans")
        if z500_path is None:
            raise FileNotFoundError(
                f"No Z500 regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})"
            )
        print(f"[rr_subregimes_variance] Z500 regimes: {z500_path.name}")
        z500_labels_raw, z500_times = _load_labels(z500_path)
        z500_series = pd.Series(z500_labels_raw.astype(int), index=_norm_dates(z500_times))
        z500_on_rr = z500_series.reindex(rr_dates).values  # aligned to rr timeline
        z500_vals_raw = sorted({int(v) for v in z500_on_rr if np.isfinite(v) and v > 0})
        if z500_cluster_method == "ward":
            _, _groups = _get_regime_alignment(region)
            z500_vals = regime_alignment.canonical_order(z500_vals_raw, _groups, bbox_step)
            z500_canonical_map = regime_alignment.relabel_ids(z500_vals_raw, _groups, bbox_step)
        else:
            z500_vals = z500_vals_raw
            z500_canonical_map = {}
        row_label_prefix = "SR"
        print(f"[rr_subregimes_variance] {n_rr} RR regimes, Z500 regimes: {z500_vals}, {len(rr_dates)} days")

        m = re.search(r"z500([^_]*)_w\d+", rr_path.name)
        fields = ["z500"] + ([f for f in m.group(1).split("+") if f] if (m and m.group(1)) else [])

    # Load every field (Z500 + impact fields) needed for the mean/std columns and the RGB column.
    fields_data: dict[str, FieldData] = {}
    for field in fields:
        p = _find_anomaly(dw, region, field, bbox_step, smooth_window, area_weight_tag=area_weight_tag)
        fd = _load_field(p) if p else None
        if fd is not None:
            fields_data[field] = fd
            print(f"[rr_subregimes_variance]   ✓ {field}: {p.name}")
        else:
            print(f"[rr_subregimes_variance]   ✗ {field}: anomaly not found — skipping")
    if "z500" not in fields_data:
        raise RuntimeError("No Z500 anomaly found — cannot build variance page.")
    fd_z500 = fields_data["z500"]

    # Full universe of impact fields loaded across EVERY Z500 regime (per_z500_fields, when
    # available, gives each regime its own subset of these — see the per-regime selection below;
    # this uncapped list is only used to size the pooled colour scale and as the non-logistic
    # fallback, where there's no per-regime field map to select from).
    all_impact_fields = [f for f in fields_data if f != "z500"]
    if not all_impact_fields:
        raise RuntimeError("No impact fields (non-Z500) available for the RGB composite column.")

    # Raw Z500 geopotential (optional — isolines are skipped if missing).
    raw_z500_path = _find_raw_z500(dw, region)
    z500_raw_fd = None
    if raw_z500_path is not None:
        print(f"[rr_subregimes_variance] Raw Z500: {raw_z500_path.name}")
        z500_raw_fd = _load_raw_z500_field(raw_z500_path)
    else:
        print("[rr_subregimes_variance] WARNING: raw Z500 not found; isolines will be skipped")

    bbox_dict = rr_geo.load_bbox_for_step(module_root, region, bbox_step)
    z500_raw_fd = _crop_field_to_bbox(z500_raw_fd, bbox_dict)

    # Build every (Z500, sub-group) day-group across the WHOLE run first — needed so the RGB
    # column's colour scale is pooled globally (matches plot_rgb_overview.py's convention: one
    # shared 95th-percentile-per-field scale across every panel of the run, not per-figure).
    group_dates: dict[int, dict[int, pd.DatetimeIndex | None]] = {}
    for z in z500_vals:
        z_mask = z500_on_rr == z
        group_dates[z] = {}
        for rr in range(1, n_rr + 1):
            g = z_mask & (rr_labels == rr)
            group_dates[z][rr] = rr_dates[g] if int(g.sum()) >= MIN_DAYS else None

    all_group_dates_flat = [
        group_dates[z][rr] for z in z500_vals for rr in range(1, n_rr + 1)
        if group_dates[z][rr] is not None
    ]

    if composite_out is not None:
        # Pool the Z500 mean/std composites over every (Z500, sub-group) day-group -- same gpm
        # conversion _page_z500_variance applies -- for scripts/compute_fixed_composite_scales.py.
        z500_unit_scale = (
            1.0 / G0 if z500_anomaly_units_look_physical(xr.DataArray(fd_z500.data), fd_z500.units) else 1.0
        )
        mean_vals: list[np.ndarray] = []
        std_vals: list[np.ndarray] = []
        for dates in all_group_dates_flat:
            m = _composite_over_dates(fd_z500, dates)
            if m is not None:
                mean_vals.append((m * z500_unit_scale).ravel())
            s = _composite_over_dates_std(fd_z500, dates)
            if s is not None:
                std_vals.append((s * z500_unit_scale).ravel())
        pooled = {}
        if mean_vals:
            pooled["z500_anom_gpm"] = np.concatenate(mean_vals)
        if std_vals:
            pooled["z500_std_gpm"] = np.concatenate(std_vals)
        comp_out = Path(composite_out).expanduser().resolve()
        comp_out.parent.mkdir(parents=True, exist_ok=True)
        ds_out = xr.Dataset({name: (("sample",), vals) for name, vals in pooled.items()})
        ds_out.attrs.update({"region": region, "bbox_step": bbox_step, "mode": mode or ""})
        ds_out.to_netcdf(comp_out)
        print(f"[rr_subregimes_variance] wrote composites NetCDF {comp_out}")

    if no_render:
        print("[rr_subregimes_variance] --no-render set: skipping figure output")
        return []

    global_scales = _global_field_scale(all_impact_fields, fields_data, all_group_dates_flat)
    print(f"[rr_subregimes_variance] Global RGB scales: "
          f"{ {f: f'{s:.4g}' for f, s in global_scales.items()} }")

    ref_fd = fields_data[all_impact_fields[0]]
    ref_shape = (len(ref_fd.lats), len(ref_fd.lons))

    out_dir = out_dir or (dw / "results" / "stage2_residual" / "figures" / region)
    out_dir.mkdir(parents=True, exist_ok=True)
    ward_tag = "_zward" if z500_cluster_method == "ward" else ""
    if is_logistic:
        logistic_tag = "_logisticadaptivebin" if bin_overrides_path is not None else f"_logistic{n_bins}bin"
    else:
        logistic_tag = ""
    base_stem = f"rr_subregimes_z500_variance_{region}_{bbox_step}{ward_tag}{logistic_tag}"

    written: list[Path] = []
    for z in z500_vals:
        rr_present = [rr for rr in range(1, n_rr + 1) if group_dates[z][rr] is not None]
        if not rr_present:
            print(f"[rr_subregimes_variance] Z500-{z}: no sub-group with >= {MIN_DAYS} days — skipping")
            continue
        gdb = {rr: group_dates[z][rr] for rr in rr_present}

        # Each Z500 regime gets its OWN field set — per_z500_fields (from the logistic merged
        # file's per_regime_fields attr) records exactly which fields rr_02's logistic fit
        # selected for THIS regime, which can differ regime to regime. Falls back to the
        # uncapped, run-wide field list only when no per-regime map exists (non-logistic modes,
        # or a logistic file without the attr) — same 3 fields everywhere in that case, since
        # there's nothing regime-specific to select from.
        if per_z500_fields is not None:
            z_fields = [f for f in per_z500_fields.get(z, []) if f in fields_data]
            if not z_fields:
                z_fields = all_impact_fields
        else:
            z_fields = all_impact_fields
        impact_fields_z = z_fields[:3]
        channel_of_z = {f: _CHANNELS[i] for i, f in enumerate(impact_fields_z)}
        print(f"[rr_subregimes_variance] Z500-{z}: sub-groups {rr_present}, RGB fields {impact_fields_z}")

        fig = _page_z500_variance(
            z, rr_present, gdb, fd_z500, fields_data, impact_fields_z, channel_of_z,
            global_scales, ref_shape, region, bbox_step, bbox_dict, module_root,
            z500_raw_fd=z500_raw_fd, row_label_prefix=row_label_prefix,
            mean_vmin=mean_vmin, mean_vmax=mean_vmax, std_vmin=std_vmin, std_vmax=std_vmax,
        )
        if fig is None:
            continue

        # Filename uses the canonical (physical) regime id when available (ward; see
        # regime_alignment_<region>.yaml) rather than the raw per-bbox-step cluster id, so the
        # same physical regime gets the same "_Z<n>" filename across bbox steps/reruns -- same
        # convention as stage2_residual.plot_composites.run_composite_plots.
        z_file_id = z500_canonical_map.get(z, z)
        for fmt in formats:
            out_path = out_dir / f"{base_stem}_Z{z_file_id}.{fmt}"
            fig.savefig(out_path, bbox_inches="tight", dpi=200)
            written.append(out_path)
            print(f"[rr_subregimes_variance] Saved: {out_path}")
        plt.close(fig)

    return written


def main(argv=None) -> int:
    import sys
    p = argparse.ArgumentParser(
        description="Z500 mean / std-dev / RGB-composite figures per RR sub-regime, one PNG per Z500 regime.")
    p.add_argument("--region", required=True, choices=("europe", "northamerica"))
    p.add_argument("--bbox-step", required=True)
    p.add_argument("--module-root", type=Path, default=None)
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--formats", default="png",
        help="Comma-separated output formats: png (default) and/or pdf. One file per Z500 "
             "regime per format — never a single bundled multi-page report.",
    )
    p.add_argument("--mode", choices=("bbox", "regime", "logistic"), default=None,
                   help="Force which pipeline's Stage-2 output to plot: a discrete k-means RR "
                        "regimes file (rr_02 bbox-mode vs rr_022 per-regime-mode), or the "
                        "continuous predicted_extreme_prob index quantile-binned at plot time "
                        "('logistic' -- see --n-bins; matches production rr_03 usage). "
                        "Default: auto-detect between bbox/regime (prefers bbox-mode); never "
                        "auto-selects 'logistic'.")
    p.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="ward",
                   help="Restrict to RR/Z500 regime files built on this Z500 parent-regime "
                        "source ('ward' reads '_zward'-tagged files; also required for the "
                        "canonical Z<n> filename ids, see regime_alignment_<region>.yaml). "
                        "Default: ward (pinned production, matches plot_composites.py/"
                        "plot_rgb_overview.py).")
    p.add_argument("--area-weight-tag", default="",
                   help="Must match the rr_00 run's --area-weight-tag (e.g. 'aw2') to resolve "
                        "that rerun's per-field anomaly files exactly (see plot_results._find_anomaly).")
    p.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS,
                   help="--mode logistic only: quantile bins per Z500 regime for the continuous "
                        "predicted_extreme_prob index (default: 2, 'non-extreme-like' vs "
                        "'extreme-like').")
    p.add_argument("--output-tag", default="",
                   help="--mode logistic only: select among several logistic runs that differ "
                        "only by extreme-label definition (matches stage2_logistic.py's "
                        "--output-tag). Leave unset for the untagged production run.")
    p.add_argument("--bin-overrides-path", type=Path, default=None,
                   help="--mode logistic only: YAML written by select_subregime_bins.py giving "
                        "per-regime data-driven probability edges, overriding --n-bins for the "
                        "regimes it covers.")
    p.add_argument("--mean-vmin", type=float, default=None, help="Override Z500 mean-anomaly fill vmin (gpm).")
    p.add_argument("--mean-vmax", type=float, default=None, help="Override Z500 mean-anomaly fill vmax (gpm).")
    p.add_argument("--std-vmax", type=float, default=None, help="Override Z500 std-dev fill vmax (gpm). vmin is always 0.")
    p.add_argument(
        "--fixed-scales-yaml", type=Path, default=os.environ.get("FIXED_SCALES_YAML") or None,
        help=(
            "src/common/fixed_color_scales.yaml (or equivalent) supplying default "
            "mean/std vmin/vmax when the flags above aren't passed explicitly. Auto-discovered "
            "at the default repo path if unset; missing file falls back to today's per-call "
            "auto-scaling. See scripts/compute_fixed_composite_scales.py."
        ),
    )
    p.add_argument(
        "--composite-out", type=Path, default=None,
        help=(
            "Persist pooled composites (z500_anom_gpm, z500_std_gpm) as a small NetCDF here. "
            "Enables a separate orchestrator to compute global vmin/vmax across bboxes/regions "
            "-- see scripts/compute_fixed_composite_scales.py."
        ),
    )
    p.add_argument(
        "--no-render", action="store_true",
        help="Skip PNG/PDF rendering (compute composites only). Use with --composite-out.",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.module_root is None:
        args.module_root = Path(__file__).resolve().parents[2]
    module_root = args.module_root.expanduser().resolve()

    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())

    scales_path = args.fixed_scales_yaml or default_fixed_scales_path(module_root)
    scales = load_fixed_color_scales(scales_path, args.region)
    mean_vmin, mean_vmax = args.mean_vmin, args.mean_vmax
    if Z500_MEAN_ANOM_GPM in scales:
        _fixed_vmin, _fixed_vmax = scales[Z500_MEAN_ANOM_GPM]
        mean_vmin = _fixed_vmin if mean_vmin is None else mean_vmin
        mean_vmax = _fixed_vmax if mean_vmax is None else mean_vmax
    std_vmax = args.std_vmax
    if std_vmax is None and Z500_STD_GPM in scales:
        std_vmax = scales[Z500_STD_GPM][1]

    run_subregime_variance_plots(
        module_root=module_root,
        region=args.region, bbox_step=args.bbox_step,
        smooth_window=args.smooth_window, out_dir=args.out_dir,
        mode=args.mode, z500_cluster_method=args.z500_cluster_method,
        area_weight_tag=args.area_weight_tag,
        n_bins=args.n_bins, output_tag=args.output_tag, bin_overrides_path=args.bin_overrides_path,
        formats=formats,
        mean_vmin=mean_vmin, mean_vmax=mean_vmax, std_vmin=None, std_vmax=std_vmax,
        composite_out=args.composite_out, no_render=args.no_render,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
