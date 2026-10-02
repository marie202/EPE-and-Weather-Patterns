#!/usr/bin/env python3
"""
Summary-statistics PDF for RR sub-regimes (mirrors the legacy z500 epe_04 stats).

Saved as rr_statistics_{region}_{bbox}.pdf. For every (Z500 regime × RR
sub-regime) group with >= MIN_DAYS, computes — from the convective_duration_*.nc
product — the same extreme-event diagnostics as the legacy Z500 pipeline
(``common.epe_04_boxplot_render``):

  - Occurrence of EPE   : % of the group's days that are convective-extreme days
                          (bar chart, strict domain-level extreme-day mask).
  - Duration            : convective hours per flagged grid cell per day, over
                          every day assigned to the group with >=1 flagged cell
                          (boxplot; days with zero flagged cells are NaN and
                          drop out, but the sample is NOT restricted to
                          domain-level extreme days -- see note below).
  - Intensity           : hourly-max precip on flagged cells per day, mean and
                          median, same day sample as Duration (boxplot).
  - Spatial mean/median : spatial mean/median of precip accumulated during
    precip on flagged     convective hours, over flagged cells, same day
    cells                 sample as Duration (boxplot). RR analogue of the
                          legacy pipeline's (b1) boxplots — see
                          ``_spatial_precip_per_day``.
  - Spatial clustering  : daily number of 8-neighbour connected components on
                          the flagged grid, same day sample as Duration
                          (boxplot).

Note on the boxplot day sample: unlike the occurrence bar chart and the EPE
frequency composites (which both use the strict domain-level `extreme` day
mask, see ``_daily_extreme_mask``), the boxplot metrics above are computed
over EVERY day in the (Z500 x RR) group -- gating on `extreme` as well would
collapse most low-probability RR sub-regimes to an empty or near-empty sample
(since the classifier the RR bins are built from concentrates almost all
domain-extreme days into the top bin by construction), hiding the
within-Z500-regime severity gradient across RR sub-regimes that the RR
binning exists to reveal.

Every x-axis tick names the parent Z500 regime and the RR sub-regime, e.g.
"Z1-SR1", "Z1-SR2", "Z2-SR1", ... -- "SR" in every mode, including the logistic
probability-bin path (user decision 2026-09-28: the statistics figures label Stage-2
sub-regimes SR<n>, matching the composite pages; see ``label_prefix`` below). Boxes/bars are coloured by parent Z500 regime using the same
region-hued, canonical-id-keyed colour map as the Z500 EPE.4 PDF (Purples for europe, Greens
for northamerica; see ``common.boxplot_style.regime_region_colors`` and
``common.regime_alignment``), and use the same notch/whisker/font conventions
(``common.boxplot_style.style_boxplot``/``style_bar``).

Reuses the proven metric helpers from common.epe_figures /
epe.epe_functions; only the grouping (by Z500×RR) and labelling are new.

Usage:
    python -m stage2_residual.plot_rr_statistics \
        --region europe --bbox-step B5 --module-root /path/to/detect_weather
"""
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

# Must be set before netCDF4/HDF5 is loaded (i.e. before the xarray import below) — see the same
# guard in plot_composites.py for why: reopening a convective_duration_*.nc file for a second
# metric segfaults inside HDF5's file-locking path on this Nextcloud-synced project folder.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

import common.regime_alignment as regime_alignment
from common.epe_figures import (
    _precip_factor_from_units,
    build_extent_mean_patch_size_from_duration,
)
from common.boxplot_style import (
    PDF_FS_PANEL_ANNOT,
    PDF_FS_PANEL_AXIS,
    PDF_FS_PANEL_LEGEND,
    PDF_FS_PANEL_SUPTITLE,
    PDF_FS_PANEL_TICK_X,
    PDF_FS_PANEL_TICK_Y,
    PDF_FS_PANEL_TITLE,
    regime_region_colors,
    style_bar,
    style_boxplot,
)

_ALIGNMENT_YAML_DIR = Path(__file__).resolve().parents[2] / "src" / "common"
_ALIGNMENT_CACHE: dict[str, tuple[str, list[dict]]] = {}


def _get_regime_alignment(region: str) -> tuple[str, list[dict]]:
    """Cached ``regime_alignment.load_regime_alignment`` for ``region`` -- the same
    ``regime_alignment_<region>.yaml`` the Z500 notebook (``2_3_Results_Z500_Stats.ipynb``)
    uses, so a Z500 parent regime gets the same canonical id (and therefore the same
    region-hued color) here as it does there.
    """
    if region not in _ALIGNMENT_CACHE:
        path = _ALIGNMENT_YAML_DIR / f"regime_alignment_{region}.yaml"
        _ALIGNMENT_CACHE[region] = regime_alignment.load_regime_alignment(path)
    return _ALIGNMENT_CACHE[region]

from .plot_results import _dw_root, _find_rr_regimes, _find_z500_regimes, _load_labels
from .plot_subregimes import MIN_DAYS, _norm_dates, find_duration_file, resolve_epe_label


# ---------------------------------------------------------------------------
# Robust duration-file opener
# ---------------------------------------------------------------------------

def _open_duration_dataset(path: Path) -> xr.Dataset:
    """Open a convective-duration NetCDF, working around xarray/pandas int64 overflow.

    Newer xarray (2025+) inspects the raw int64 fill value (-9223372036854775806)
    from 'hours since ...' time coordinates before masking, which overflows
    pandas Timedelta.  Fall back to decode_times=False and decode manually.
    """
    import re as _re

    try:
        return xr.open_dataset(path, chunks=None)
    except Exception:
        pass

    ds = xr.open_dataset(path, chunks=None, decode_times=False)
    for coord in list(ds.coords):
        units = str(ds[coord].attrs.get("units", ""))
        m = _re.match(r"(\w+)\s+since\s+(.+)", units.strip())
        if not m:
            continue
        try:
            unit_str, ref_str = m.group(1), m.group(2).strip()
            ref = pd.Timestamp(ref_str)
            raw = ds[coord].values.astype(np.float64)
            raw[np.abs(raw) > 1e15] = np.nan  # mask NetCDF INT64 fill values
            times = (ref + pd.to_timedelta(raw, unit=unit_str)).values.astype("datetime64[ns]")
            ds = ds.assign_coords({coord: xr.Variable(ds[coord].dims, times, {})})
        except Exception as _e:
            print(f"[rr_stats] WARNING: could not decode time coord '{coord}': {_e}")
    return ds


# ---------------------------------------------------------------------------
# Metric extraction (all returned as per-day numpy arrays on the duration axis)
# ---------------------------------------------------------------------------

def _spatial_dims_da(da: xr.DataArray) -> tuple[str, ...]:
    return tuple(d for d in da.dims if d != "time")


def _extreme_day_mask(dur: xr.Dataset) -> np.ndarray:
    try:
        from epe_preprocessing.epe_functions import _daily_extreme_mask
        m = _daily_extreme_mask(extremes_ds=None, duration_ds=dur)
        if m is not None:
            return np.asarray(m.values, dtype=bool)
    except Exception:
        pass
    if "flagged_any_gridcell" in dur:
        return np.asarray(dur["flagged_any_gridcell"].values).astype(int) > 0
    if "n_conv_hours_grid" in dur:
        n = dur["n_conv_hours_grid"].astype(float)
        return np.asarray((n.where(n > 0).count(dim=_spatial_dims_da(n)) > 0).values, dtype=bool)
    return np.zeros(dur.sizes["time"], dtype=bool)


def _duration_per_day(dur: xr.Dataset) -> np.ndarray | None:
    """Mean convective hours per flagged cell, per day."""
    if "n_conv_hours_grid" not in dur:
        return None
    n = dur["n_conv_hours_grid"].astype(float)
    mean_h = n.where(n > 0).mean(dim=_spatial_dims_da(n), skipna=True)
    return np.asarray(mean_h.values, dtype=float)


def _spatial_precip_per_day(dur: xr.Dataset, agg: str = "mean") -> np.ndarray | None:
    """Spatial mean/median/max of precip accumulated during convective hours, over flagged cells.
    ``agg``: "mean", "median", or "max".

    RR analogue of the legacy Z500 pipeline's "(b1) spatial mean/median precip on
    flagged cells" boxplot. The Z500 version uses the full daily precip field ``P``
    over ``E_conv``-flagged cells from the extremes NetCDF; RR does not load that file,
    but the ``convective_duration_*.nc`` product it already reads carries
    ``P_conv_total_grid_day`` (precip accumulated only during convective-flagged hours,
    per cell per day) alongside ``n_conv_hours_grid`` (the same flag mask used for
    duration/connectivity), so the same "spatial stat over flagged cells" question can
    be answered without an extra data source. The two metrics are not numerically
    identical (convective-hours-only accumulation vs. full-day total).
    """
    if "P_conv_total_grid_day" not in dur or "n_conv_hours_grid" not in dur:
        return None
    p = dur["P_conv_total_grid_day"].astype(float)
    n = dur["n_conv_hours_grid"].astype(float)
    p_flagged = p.where(n > 0)
    sp = _spatial_dims_da(p)
    if agg == "mean":
        stat = p_flagged.mean(dim=sp, skipna=True)
    elif agg == "median":
        stat = p_flagged.median(dim=sp, skipna=True)
    else:
        stat = p_flagged.max(dim=sp, skipna=True)
    return np.asarray(stat.values, dtype=float)


def _excess_spatial_precip_per_day(dur: xr.Dataset, agg: str = "mean") -> np.ndarray | None:
    """Spatial mean/median/max of the exceedance (precip - P_q_hourly) accumulated over
    E_conv-flagged hours, on flagged cells -- the severity analogue of
    ``_spatial_precip_per_day`` that stays comparable across regimes at high percentiles
    (raw amount converges toward the same conditioning threshold; the excess above it does
    not, by construction). ``agg``: "mean", "median", or "max".

    ``P_conv_total_grid_day`` is the sum of hourly precip over flagged hours, and every
    flagged hour satisfies ``P_hour > P_q_hourly`` by construction (epe_01b), so
    ``P_conv_total_grid_day - n_conv_hours_grid * P_q_hourly`` recovers the per-cell/day total
    exceedance without needing the raw hourly file. Unlike ``_spatial_precip_per_day``, this
    converts to mm (both source vars are raw ERA5 units, typically m).
    """
    if "P_conv_total_grid_day" not in dur or "n_conv_hours_grid" not in dur or "P_q_hourly" not in dur:
        return None
    p = dur["P_conv_total_grid_day"].astype(float)
    n = dur["n_conv_hours_grid"].astype(float)
    pq = dur["P_q_hourly"].astype(float)
    factor = _precip_factor_from_units(str(p.attrs.get("units", "")))
    excess = (p - n * pq) * factor
    excess_flagged = excess.where(n > 0)
    sp = _spatial_dims_da(excess_flagged)
    if agg == "mean":
        stat = excess_flagged.mean(dim=sp, skipna=True)
    elif agg == "median":
        stat = excess_flagged.median(dim=sp, skipna=True)
    else:
        stat = excess_flagged.max(dim=sp, skipna=True)
    return np.asarray(stat.values, dtype=float)


def _intensity_per_day(
    dur: xr.Dataset, module_root: Path, study_area: str, agg: str = "mean"
) -> tuple[np.ndarray | None, str]:
    try:
        from common.epe_figures import _resolve_hourly_intensity_for_summary
        da, metric = _resolve_hourly_intensity_for_summary(
            dur, module_root=module_root, study_area=study_area, agg=agg
        )
        if da is None:
            return None, metric
        da = da.reindex(time=dur["time"].values)
        return np.asarray(da.values, dtype=float), metric
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _connectivity_per_day(dur: xr.Dataset) -> np.ndarray | None:
    try:
        from common.epe_figures import build_connectivity_n_components_from_duration
        da = build_connectivity_n_components_from_duration(dur)
        if da is None:
            return None
        da = da.reindex(time=dur["time"].values)
        return np.asarray(da.values, dtype=float)
    except Exception:
        return None


def _extent_per_day(dur: xr.Dataset) -> np.ndarray | None:
    """Mean size (# grid cells) of each 8-connected flagged patch, per day -- the typical EPE
    spatial-footprint size (as opposed to ``_connectivity_per_day``'s patch *count*, or a plain
    domain-wide flagged-cell count that would conflate one big patch with many small ones)."""
    try:
        da = build_extent_mean_patch_size_from_duration(dur)
        if da is None:
            return None
        da = da.reindex(time=dur["time"].values)
        return np.asarray(da.values, dtype=float)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _z_colors(region: str, bbox_step: str, zvals: list[int]):
    """Parent-Z500-regime colour map, keyed by this bbox step's RAW Z500 cluster id but colored
    by its CANONICAL (cross-bbox-aligned) id via ``regime_alignment_<region>.yaml`` -- the same
    region-hued (Purples for europe, Greens for northamerica) shade the Z500 notebook's
    per-bbox-step boxplots and alluvial diagrams use for the same physical regime, so RR and
    Z500 report families are visually cross-referenceable regime-for-regime, not just by number.
    """
    _, groups = _get_regime_alignment(region)
    raw_to_canon = regime_alignment.relabel_ids(zvals, groups, bbox_step)
    canon_colors = regime_region_colors(region, raw_to_canon.values(), n_total=len(groups))
    return {z: canon_colors[raw_to_canon[z]] for z in zvals}


def _grouped_boxplot(
    ax, per_group_values, labels, zvals, *, region: str, bbox_step: str, title, ylabel, tier: str = "panel",
) -> bool:
    """Z500xRR boxplot using the shared Z500/RR box style (notch, whiskers, per-Z500 colour)."""
    data, ticks, colors, zc = [], [], [], _z_colors(region, bbox_step, zvals)
    for vals, lab, z in zip(per_group_values, labels, zvals):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        data.append(v)
        ticks.append(f"{lab}\n(n={v.size})")
        colors.append(zc[z])
    if not data:
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center", fontsize=9)
        ax.axis("off")
        return False
    style_boxplot(
        ax,
        data,
        ticks,
        title=title,
        ylabel=ylabel,
        xlabel="Z500 x RR sub-regime",
        colors=colors,
        tier=tier,
    )
    return True


def _grouped_bar(
    ax, values, labels, zvals, *, region: str, bbox_step: str, title, ylabel, counts=None, tier: str = "panel",
) -> bool:
    """Z500xRR bar chart using the shared Z500/RR colour scheme (per-parent-Z500 colour)."""
    zc = _z_colors(region, bbox_step, zvals)
    colors = [zc[z] for z in zvals]
    ticks = [f"{lab}\n(n={counts[i]})" if counts is not None else lab for i, lab in enumerate(labels)]
    style_bar(ax, values, ticks, colors=colors, title=title, ylabel=ylabel, tier=tier)
    return True


# ---------------------------------------------------------------------------
# Summary text file
# ---------------------------------------------------------------------------

def _write_summary_txt(
    out_path: Path,
    *,
    region: str,
    bbox_step: str,
    rr_path: Path,
    z500_path: Path,
    dur_path: Path,
    total_days: int,
    groups: list[tuple[int, int]],
    labels: list[str],
    totals: list[int],
    occ_pct: list[float],
    n_extreme: list[int],
    inten_metric: str,
) -> None:
    _, _groups_align = _get_regime_alignment(region)
    z_unique = regime_alignment.canonical_order({z for z, _ in groups}, _groups_align, bbox_step)
    lines = [
        f"rr_statistics summary — {region}  {bbox_step}",
        f"Generated: {datetime.date.today()}",
        "",
        "Input files:",
        f"  RR regimes:   {rr_path.name}",
        f"  Z500 regimes: {z500_path.name}",
        f"  Duration:     {dur_path.name}",
        f"  Intensity:    {inten_metric}",
        "",
        f"Total days in regime files: {total_days}",
        f"Groups (Z500 × RR) with >= {MIN_DAYS} days: {len(groups)}",
        "",
        f"{'Group':<12} {'N days':>8} {'% total':>9} {'N extreme':>10} {'EPE occ %':>10}",
        "-" * 55,
    ]
    for lab, (z, r), n, pct, n_ext in zip(labels, groups, totals, occ_pct, n_extreme):
        pct_total = 100.0 * n / total_days if total_days else float("nan")
        lines.append(
            f"{lab:<12} {n:>8d} {pct_total:>8.1f}% {n_ext:>10d} {pct:>9.1f}%"
        )

    lines += ["", "Balance within each Z500 regime (% of Z500 days in each RR):"]
    for z in z_unique:
        z_groups = [(lab, n) for lab, (zz, _), n in zip(labels, groups, totals) if zz == z]
        z_total = sum(n for _, n in z_groups)
        parts = "  ".join(f"{lab}={100*n/z_total:.1f}%" for lab, n in z_groups)
        lines.append(f"  Z{z} ({z_total} d): {parts}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[rr_stats] Summary text: {out_path}")


# ---------------------------------------------------------------------------
# Regime frequency / balance plot page
# ---------------------------------------------------------------------------

def build_regime_distribution_figure(
    groups: list[tuple[int, int]],
    labels: list[str],
    zvals: list[int],
    totals: list[int],
    total_days: int,
    region: str,
    bbox_step: str,
    label_prefix: str = "SR",
) -> "plt.Figure":
    """Build (without saving) the page: (A) absolute % per Z500×RR group; (B) balance within each Z500 regime."""
    pct_all = [100.0 * n / total_days for n in totals]
    _, _groups_align = _get_regime_alignment(region)
    z_unique = regime_alignment.canonical_order({z for z, _ in groups}, _groups_align, bbox_step)
    zc = _z_colors(region, bbox_step, z_unique)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(10, 1.4 * len(labels)), 10),
                                          facecolor="white")
    fig.suptitle(
        f"Sub-regime frequency and balance — {region}  {bbox_step}",
        fontsize=PDF_FS_PANEL_SUPTITLE, fontweight="bold",
    )

    # (A) Absolute frequency
    colors_top = [zc[z] for z in zvals]
    tick_labels = [f"{lab}\n(n={n})" for lab, n in zip(labels, totals)]
    bars = ax_top.bar(range(len(pct_all)), pct_all, color=colors_top, edgecolor="white")
    ax_top.bar_label(bars, fmt="%.1f%%", fontsize=PDF_FS_PANEL_ANNOT, padding=2)
    ax_top.set_xticks(range(len(labels)))
    ax_top.set_xticklabels(tick_labels, fontsize=PDF_FS_PANEL_TICK_X)
    ax_top.set_ylabel("% of all days", fontsize=PDF_FS_PANEL_AXIS)
    ax_top.set_title("(A) Sub-regime frequency (% of total days)", fontsize=PDF_FS_PANEL_TITLE)
    ax_top.tick_params(axis="y", labelsize=PDF_FS_PANEL_TICK_Y)
    ax_top.grid(axis="y", alpha=0.25)
    # Z500 regime legend patches
    from matplotlib.patches import Patch
    ax_top.legend(
        handles=[Patch(facecolor=zc[z], label=f"Z{z}") for z in z_unique],
        fontsize=PDF_FS_PANEL_LEGEND, loc="upper right",
    )

    # (B) Stacked bar: for each Z500 regime, fraction in each RR
    rr_unique = sorted({r for _, r in groups})
    rr_cmap = plt.cm.Set2(np.linspace(0, 1, max(len(rr_unique), 1)))
    rr_colors = {r: rr_cmap[i] for i, r in enumerate(rr_unique)}

    x_pos = range(len(z_unique))
    bottoms = np.zeros(len(z_unique))
    for rr in rr_unique:
        heights = []
        for z in z_unique:
            z_total = sum(n for (zz, _), n in zip(groups, totals) if zz == z)
            n_zr = next((n for (zz, r), n in zip(groups, totals) if zz == z and r == rr), 0)
            heights.append(100.0 * n_zr / z_total if z_total else 0.0)
        ax_bot.bar(x_pos, heights, bottom=bottoms, color=rr_colors[rr],
                   label=f"{label_prefix}{rr}", edgecolor="white")
        # Add % labels where segment is large enough to read
        for xi, (h, b) in enumerate(zip(heights, bottoms)):
            if h >= 5.0:
                ax_bot.text(xi, b + h / 2, f"{h:.0f}%", ha="center", va="center",
                            fontsize=PDF_FS_PANEL_ANNOT, color="black")
        bottoms += np.array(heights)

    z_labels = [f"Z{z}\n({sum(n for (zz,_),n in zip(groups,totals) if zz==z)} d)"
                for z in z_unique]
    ax_bot.set_xticks(list(x_pos))
    ax_bot.set_xticklabels(z_labels, fontsize=PDF_FS_PANEL_TICK_X)
    ax_bot.set_ylabel("% of Z500 regime days", fontsize=PDF_FS_PANEL_AXIS)
    ax_bot.set_ylim(0, 105)
    ax_bot.set_title("(B) Balance: RR sub-regime fraction within each Z500 regime", fontsize=PDF_FS_PANEL_TITLE)
    ax_bot.tick_params(axis="y", labelsize=PDF_FS_PANEL_TICK_Y)
    ax_bot.legend(fontsize=PDF_FS_PANEL_LEGEND, loc="upper right")
    ax_bot.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def build_overview_figure(
    labels: list[str], zvals: list[int], region: str, bbox_step: str, dur_path_name: str,
    occ_pct: list[float], totals: list[int],
    dur_vals: list[np.ndarray], int_mean_vals: list[np.ndarray], conn_vals: list[np.ndarray],
    inten_mean_label: str,
) -> "plt.Figure":
    """Build (without saving) the 2x2 EPE-statistics overview page (intensity panel uses mean)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    fig.suptitle(
        f"RR sub-regime extreme-event statistics — {region} {bbox_step}\n"
        f"(source: {dur_path_name})", fontsize=PDF_FS_PANEL_SUPTITLE, fontweight="bold",
    )
    _grouped_bar(axes[0, 0], occ_pct, labels, zvals, region=region, bbox_step=bbox_step,
                 title="(A) EPE occurrence", ylabel="% of group days that are extreme",
                 counts=totals)
    _grouped_boxplot(axes[0, 1], dur_vals, labels, zvals, region=region, bbox_step=bbox_step,
                     title="(B) Duration (flagged days)", ylabel="hours per flagged cell / day")
    _grouped_boxplot(axes[1, 0], int_mean_vals, labels, zvals, region=region, bbox_step=bbox_step,
                     title="(C) Intensity — mean (flagged days)", ylabel=inten_mean_label)
    _grouped_boxplot(axes[1, 1], conn_vals, labels, zvals, region=region, bbox_step=bbox_step,
                     title="(D) Spatial clustering (flagged days)", ylabel="# connected components / day")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def build_full_bar_figure(
    values: list[float], labels: list[str], zvals: list[int], region: str, bbox_step: str,
    title: str, ylabel: str, counts: list[int] | None = None,
) -> "plt.Figure":
    """Build (without saving) a full-size bar page for one metric across all Z500×RR groups."""
    fig, ax = plt.subplots(figsize=(max(10.0, 1.4 * len(labels)), 5.2), facecolor="white")
    _grouped_bar(ax, values, labels, zvals, region=region, bbox_step=bbox_step,
                 title=f"{title} — {region} {bbox_step}", ylabel=ylabel, counts=counts, tier="full")
    fig.tight_layout(pad=1.5, h_pad=1.2)
    return fig


def build_full_box_figure(
    per_group: list[np.ndarray], labels: list[str], zvals: list[int], region: str, bbox_step: str,
    title: str, ylabel: str,
) -> "plt.Figure":
    """Build (without saving) a full-size boxplot page for one metric across all Z500×RR groups."""
    fig, ax = plt.subplots(figsize=(max(10.0, 1.4 * len(labels)), 5.2), facecolor="white")
    _grouped_boxplot(ax, per_group, labels, zvals, region=region, bbox_step=bbox_step,
                     title=f"{title} — {region} {bbox_step}", ylabel=ylabel, tier="full")
    fig.tight_layout(pad=1.5, h_pad=1.2)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_rr_statistics_inputs(module_root, region, bbox_step, smooth_window=11,
                              epe_region_label=None, mode=None,
                              z500_cluster_method: str = "kmeans",
                              n_bins: int | None = None, output_tag: str = "",
                              bin_overrides_path=None) -> dict:
    """Resolve, load and compute every input the statistics pages need.

    Separated from ``run_rr_statistics`` so a notebook can load once (the
    slow part — opening the regime-label and convective-duration NetCDFs and
    computing per-day metrics) and then call the ``build_*_figure`` helpers
    repeatedly while tuning figure styling.

    `mode`: "bbox" or "regime" to force which pipeline's RR regimes file to use
    (rr_02 vs rr_022); "logistic" to summarize the Stage-2 [LOGISTIC] continuous
    `predicted_extreme_prob` index instead, quantile-binned into `n_bins` groups
    per Z500 regime at load time (same recipe as `plot_composites.py`'s
    `mode="logistic"` — default `n_bins=2`; `output_tag` selects among logistic
    runs that differ only by extreme-label definition). None auto-detects
    between bbox/regime (see `_find_rr_regimes`); never auto-selects "logistic".

    `bin_overrides_path`, "logistic" mode only: a `select_subregime_bins.py`-written
    YAML (see `stage2_logistic.load_bin_overrides_yaml`) of per-Z500-regime
    data-driven probability edges — overrides the uniform `n_bins` quantile split
    for regimes it covers, same recipe as `plot_composites.py`'s
    `load_composite_inputs_logistic`, so the statistics groups line up with the
    adaptive bins shown in the composite maps. Regimes absent from the file still
    fall back to the plain `n_bins` split. Ignored outside "logistic" mode.

    `z500_cluster_method`: "kmeans" (default; also covers the Ward-seeded K-means
    hybrid) or "ward" (pure Ward HRC) — which Z500 parent-regime source both the RR
    regimes file and the Z500 regimes file must have been built from, so the
    Z500-regime x RR-subregime pairing shown in the statistics is consistent.
    """
    dw = _dw_root(module_root)
    label = resolve_epe_label(epe_region_label)
    is_logistic = mode == "logistic"
    bin_edges_by_regime: dict[int, list[float]] = {}

    if is_logistic:
        from .plot_results import _find_rr_logistic_merged
        from .stage2_logistic import bin_predicted_prob_per_regime, load_bin_overrides_yaml

        n_bins = n_bins or 2
        rr_path = _find_rr_logistic_merged(dw, region, bbox_step, output_tag=output_tag,
                                            z500_cluster_method=z500_cluster_method)
        if rr_path is None:
            raise FileNotFoundError(
                f"No merged Stage-2 [LOGISTIC] index file for {region} {bbox_step} "
                f"(z500_cluster_method={z500_cluster_method}) — run stage2_logistic.py first."
            )
        z500_path = None
        print(f"[rr_stats] RR logistic index: {rr_path.name}")

        with xr.open_dataset(rr_path) as ds:
            time_dim = ds["predicted_extreme_prob"].dims[0]
            rr_times = pd.DatetimeIndex(ds[time_dim].values)
            predicted_prob = np.asarray(ds["predicted_extreme_prob"].values, dtype=np.float64)
            z500_regime = np.asarray(ds["z500_regime"].values, dtype=int)
            # Provenance: which EXTREME_LABELS_PATH stage2_logistic.py's fit was supervised on
            # (see merge_per_regime_logistic_outputs) -- "" for merged files written before this
            # attr existed (pre-2026-08-06). Surfaced to callers (e.g.
            # export_regime_rr_statistics_csv.py's PROVENANCE.json) so a stale merged file isn't
            # silently mistaken for one built from the current pinned domain_pctl_q.
            rr_extreme_labels_path = str(ds.attrs.get("extreme_labels_path", ""))
            rr_domain_pctl_q_tag = str(ds.attrs.get("domain_pctl_q_tag", ""))
        rr_dates = _norm_dates(rr_times)

        bin_edges_by_regime = {}
        if bin_overrides_path is not None:
            bin_edges_by_regime = load_bin_overrides_yaml(bin_overrides_path, bbox_step)
            print(f"[rr_stats] Bin overrides: {Path(bin_overrides_path).name} -> "
                  f"{ {z: len(e) + 1 for z, e in bin_edges_by_regime.items()} } bins by regime "
                  f"(others kept as a single undivided group -- see stage2_logistic."
              f"bin_predicted_prob_per_regime)")

        rr_labels = bin_predicted_prob_per_regime(
            predicted_prob, z500_regime, n_bins=n_bins, bin_edges_by_regime=bin_edges_by_regime,
        )
        z_on_rr = pd.Series(z500_regime, index=rr_dates)
        # "SR", not "Bin", even though these are probability quantile bins rather than discrete
        # k-means sub-clusters: the paper labels Stage-2 sub-regimes SR<n> everywhere (same
        # decision as the composite pages' row_label_prefix in plot_composites.py).
        label_prefix = "SR"
    else:
        rr_path = _find_rr_regimes(dw, region, bbox_step, module_root=module_root, mode=mode,
                                    z500_cluster_method=z500_cluster_method)
        if rr_path is None:
            raise FileNotFoundError(f"No RR regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})")
        z500_path = _find_z500_regimes(dw, region, bbox_step, smooth_window=smooth_window,
                                        cluster_method=z500_cluster_method)
        if z500_path is None:
            raise FileNotFoundError(f"No Z500 regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})")
        print(f"[rr_stats] RR regimes:   {rr_path.name}")
        print(f"[rr_stats] Z500 regimes: {z500_path.name}")

        rr_labels, rr_times = _load_labels(rr_path)
        rr_labels = rr_labels.astype(int)
        rr_dates = _norm_dates(rr_times)
        z_raw, z_times = _load_labels(z500_path)
        z_series = pd.Series(z_raw.astype(int), index=_norm_dates(z_times))
        z_on_rr = z_series.reindex(rr_dates)
        label_prefix = "SR"
        # Discrete bbox/regime mode has no per-file extreme-labels provenance today (see
        # _find_rr_regimes) -- left blank rather than guessed.
        rr_extreme_labels_path = ""
        rr_domain_pctl_q_tag = ""

    rr_by_day = pd.Series(rr_labels, index=rr_dates)
    z_by_day = pd.Series(z_on_rr.values, index=rr_dates)

    dur_path = find_duration_file(dw, label, module_root)
    if dur_path is None:
        raise FileNotFoundError(
            f"No convective_duration_*.nc found (label={label}). "
            f"Set EPE_FILE / EPE_REGION_LABEL or run epe duration detection."
        )
    print(f"[rr_stats] Duration file: {dur_path}")

    from epe_preprocessing.epe_functions import dataset_rename_time
    dur = dataset_rename_time(_open_duration_dataset(dur_path)).load()
    dur_days = _norm_dates(dur["time"].values)
    dur = dur.assign_coords(day=("time", dur_days.values.astype("datetime64[ns]")))

    # Per duration-day Z500 / RR labels (NaN where the regime files lack the day).
    z_of_day = z_by_day.reindex(dur_days).to_numpy(dtype=float)
    rr_of_day = rr_by_day.reindex(dur_days).to_numpy(dtype=float)

    # All-days precip amount (mean/median) -- the non-circular baseline: dur_days already spans
    # EVERY MJJAS day (not just flagged/extreme ones), so this just attaches the fixed-AOI daily
    # total precip field (written by epe_01_aggregate_daily.py *before* any extreme-day
    # selection) to that same day range. See common.alldays_precip's module docstring.
    from common.alldays_precip import load_daily_tp_alldays

    tp_alldays = load_daily_tp_alldays(module_root, region)
    tp_mean_of_day = tp_alldays["tp_mean_mm"].reindex(dur_days).to_numpy(dtype=float)
    tp_median_of_day = tp_alldays["tp_median_mm"].reindex(dur_days).to_numpy(dtype=float)

    # Metrics on the duration timeline.
    extreme = _extreme_day_mask(dur)
    dur_h = _duration_per_day(dur)
    inten_mean, inten_metric_mean = _intensity_per_day(dur, module_root, label, agg="mean")
    inten_median, inten_metric_median = _intensity_per_day(dur, module_root, label, agg="median")
    nconn = _connectivity_per_day(dur)
    extent = _extent_per_day(dur)
    spat_mean = _spatial_precip_per_day(dur, agg="mean")
    spat_median = _spatial_precip_per_day(dur, agg="median")
    spat_max = _spatial_precip_per_day(dur, agg="max")
    excess_mean = _excess_spatial_precip_per_day(dur, agg="mean")
    excess_median = _excess_spatial_precip_per_day(dur, agg="median")
    excess_max = _excess_spatial_precip_per_day(dur, agg="max")
    print(f"[rr_stats] intensity mean metric: {inten_metric_mean}")
    print(f"[rr_stats] intensity median metric: {inten_metric_median}")
    print(f"[rr_stats] duration available: {dur_h is not None}, connectivity: {nconn is not None}, extent: {extent is not None}")
    print(f"[rr_stats] spatial precip on flagged cells available: {spat_mean is not None}")
    print(f"[rr_stats] excess-over-threshold on flagged cells available: {excess_mean is not None}")

    # Groups (Z500 × RR) with >= MIN_DAYS total days, ordered by (canonical Z500 id, rr) --
    # not raw z id -- so "Z1-..." groups always sort/plot first, "Z2-..." second, etc.,
    # matching the canonical order used by the Z500 boxplots/alluvial diagrams (see
    # common.regime_alignment.canonical_order), regardless of this bbox step's own
    # raw Z500 cluster numbering.
    valid = np.isfinite(z_of_day) & np.isfinite(rr_of_day)
    raw_z_rr = {(int(z), int(r)) for z, r in zip(z_of_day[valid], rr_of_day[valid])}
    _, _groups_for_order = _get_regime_alignment(region)
    _z_order = regime_alignment.canonical_order(
        {z for z, _ in raw_z_rr}, _groups_for_order, bbox_step,
    )
    _z_rank = {z: i for i, z in enumerate(_z_order)}
    combos = sorted(raw_z_rr, key=lambda zr: (_z_rank[zr[0]], zr[1]))
    groups, labels, zvals, masks, totals = [], [], [], [], []
    for (z, r) in combos:
        m = valid & (z_of_day == z) & (rr_of_day == r)
        n = int(m.sum())
        if n < MIN_DAYS:
            continue
        groups.append((z, r)); labels.append(f"Z{z}-{label_prefix}{r}")
        zvals.append(z); masks.append(m); totals.append(n)
    if not groups:
        raise RuntimeError("No (Z500×RR) group with >= MIN_DAYS days.")
    print(f"[rr_stats] groups: {labels}")

    # Per-group occurrence (%, still gated on the strict domain-level `extreme` day mask -- this
    # is what the "days assigned"/"EPE occurrence" bar panels and EPE frequency composites use)
    # and per-group metric samples over EVERY day assigned to the group (not just domain-extreme
    # days). The severity metrics themselves (duration/intensity/spatial precip/excess) are each
    # already computed `.where(n_conv_hours_grid > 0)` per day, i.e. they resolve to NaN on days
    # with zero flagged cells; `_rr_entry_boxplot` filters non-finite values before plotting, so
    # widening `sel` to `m` doesn't pull in "nothing happened" days -- it just stops throwing away
    # the (often many) days with flagged-but-not-domain-extreme convection, which is what lets the
    # boxplots show a within-Z500-regime / across-RR-sub-regime severity gradient instead of
    # collapsing every low-probability bin to an empty (or near-empty, domain-extreme-only) sample.
    occ_pct = []
    n_extreme = []
    dur_vals, int_mean_vals, int_median_vals, conn_vals, extent_vals = [], [], [], [], []
    spat_mean_vals, spat_median_vals, spat_max_vals = [], [], []
    excess_mean_vals, excess_median_vals, excess_max_vals = [], [], []
    tp_alldays_mean_vals, tp_alldays_median_vals = [], []
    for m in masks:
        n_tot = int(m.sum())
        n_ext = int((m & extreme).sum())
        occ_pct.append(100.0 * n_ext / n_tot if n_tot else np.nan)
        n_extreme.append(n_ext)
        sel = m
        dur_vals.append(dur_h[sel] if dur_h is not None else np.array([]))
        int_mean_vals.append(inten_mean[sel] if inten_mean is not None else np.array([]))
        int_median_vals.append(inten_median[sel] if inten_median is not None else np.array([]))
        conn_vals.append(nconn[sel] if nconn is not None else np.array([]))
        extent_vals.append(extent[sel] if extent is not None else np.array([]))
        spat_mean_vals.append(spat_mean[sel] if spat_mean is not None else np.array([]))
        spat_median_vals.append(spat_median[sel] if spat_median is not None else np.array([]))
        spat_max_vals.append(spat_max[sel] if spat_max is not None else np.array([]))
        excess_mean_vals.append(excess_mean[sel] if excess_mean is not None else np.array([]))
        excess_median_vals.append(excess_median[sel] if excess_median is not None else np.array([]))
        excess_max_vals.append(excess_max[sel] if excess_max is not None else np.array([]))
        # Unlike the metrics above, NOT `.where`-gated on flagged cells -- every day in `sel` has
        # a real value here (no NaN-on-quiet-days), since this is the unconditioned all-days
        # baseline (see the load-time comment near `tp_alldays` above).
        tp_alldays_mean_vals.append(tp_mean_of_day[sel])
        tp_alldays_median_vals.append(tp_median_of_day[sel])

    # MJJAS monthly extreme-day counts per (Z500×RR) group — same shape as the Z500
    # epe_occurrence_monthly export (pct_of_all_days among extreme days that fall in a kept group).
    _mjjas = (5, 6, 7, 8, 9)
    _month_names = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
    months = pd.DatetimeIndex(dur_days.values.astype("datetime64[ns]")).month.to_numpy()
    total_ext_kept = int(sum(n_extreme))
    occ_monthly_rows: list[dict] = []
    for (z, r), lab, m, n_ext in zip(groups, labels, masks, n_extreme, strict=False):
        for month in _mjjas:
            n_m = int((m & extreme & (months == month)).sum())
            occ_monthly_rows.append({
                "z": int(z), "rr": int(r), "label": lab, "month": int(month),
                "month_name": _month_names[month], "n_days": n_m,
                "pct_within_group": (100.0 * n_m / n_ext) if n_ext else 0.0,
                "pct_of_all_days": (100.0 * n_m / total_ext_kept) if total_ext_kept else 0.0,
            })
    occ_monthly = pd.DataFrame(occ_monthly_rows)

    def _intensity_ylabel(metric: str, agg: str) -> str:
        if "proxy" in str(metric).lower():
            return f"precip rate proxy — wet-cell {agg} (mm/h)"
        return f"hourly precip — wet-cell {agg} (mm/h)"

    return dict(
        dw=dw, region=region, bbox_step=bbox_step, module_root=module_root,
        rr_path=rr_path, z500_path=z500_path, dur_path=dur_path,
        rr_extreme_labels_path=rr_extreme_labels_path, rr_domain_pctl_q_tag=rr_domain_pctl_q_tag,
        n_bins_used=n_bins if is_logistic else None,
        bin_overrides_path_used=str(bin_overrides_path) if (is_logistic and bin_overrides_path is not None) else "",
        bin_edges_by_regime=bin_edges_by_regime if is_logistic else {},
        total_days=len(rr_labels),
        label_prefix=label_prefix,
        groups=groups, labels=labels, zvals=zvals, totals=totals,
        occ_pct=occ_pct, n_extreme=n_extreme, occ_monthly=occ_monthly,
        dur_vals=dur_vals, int_mean_vals=int_mean_vals, int_median_vals=int_median_vals,
        conn_vals=conn_vals, extent_vals=extent_vals,
        spat_mean_vals=spat_mean_vals, spat_median_vals=spat_median_vals,
        spat_max_vals=spat_max_vals,
        excess_mean_vals=excess_mean_vals, excess_median_vals=excess_median_vals,
        excess_max_vals=excess_max_vals,
        tp_alldays_mean_vals=tp_alldays_mean_vals, tp_alldays_median_vals=tp_alldays_median_vals,
        inten_metric_mean=inten_metric_mean, inten_metric_median=inten_metric_median,
        inten_mean_label=_intensity_ylabel(inten_metric_mean, "mean"),
        inten_median_label=_intensity_ylabel(inten_metric_median, "median"),
    )


def run_rr_statistics(
    module_root, region, bbox_step, smooth_window=11,
    epe_region_label=None, out_dir=None, mode=None,
    formats: tuple[str, ...] = ("png",),
    z500_cluster_method: str = "kmeans",
) -> list[Path]:
    """Build every RR statistics figure and save each in every requested format.

    ``formats`` may contain "png" (one file per figure, ``rr_statistics_<region>_<bbox>_<key>.png``)
    and/or "pdf" (single multi-page ``rr_statistics_<region>_<bbox>.pdf``, the original rr_03 behavior).

    ``z500_cluster_method``: which Z500 parent-regime source (and hence which RR sub-regime
    file) to summarize — must match how rr_02/rr_022 built the RR regimes being visualized.
    """
    inputs = load_rr_statistics_inputs(
        module_root, region, bbox_step, smooth_window=smooth_window, epe_region_label=epe_region_label,
        mode=mode, z500_cluster_method=z500_cluster_method,
    )
    labels, zvals, totals = inputs["labels"], inputs["zvals"], inputs["totals"]

    out_dir = out_dir or (inputs["dw"] / "results" / "stage2_residual" / "figures" / region)
    out_dir.mkdir(parents=True, exist_ok=True)
    ward_tag = "_zward" if z500_cluster_method == "ward" else ""

    # Summary text file (mirrors the print output with per-group stats)
    out_txt = out_dir / f"rr_statistics_summary_{region}_{bbox_step}{ward_tag}.txt"
    _write_summary_txt(
        out_txt,
        region=region, bbox_step=bbox_step,
        rr_path=inputs["rr_path"], z500_path=inputs["z500_path"], dur_path=inputs["dur_path"],
        total_days=inputs["total_days"],
        groups=inputs["groups"], labels=labels, totals=totals,
        occ_pct=inputs["occ_pct"], n_extreme=inputs["n_extreme"], inten_metric=inputs["inten_metric_mean"],
    )

    figs: dict[str, "plt.Figure"] = {
        "distribution": build_regime_distribution_figure(
            inputs["groups"], labels, zvals, totals, inputs["total_days"], region, bbox_step,
            label_prefix=inputs["label_prefix"],
        ),
        # 2×2 EPE statistics overview (intensity panel uses mean). Panel tier fonts
        # (10/9/7-8 pt) match the Z500 EPE.4 2x2 summary style.
        "overview": build_overview_figure(
            labels, zvals, region, bbox_step, inputs["dur_path"].name,
            inputs["occ_pct"], totals, inputs["dur_vals"], inputs["int_mean_vals"], inputs["conn_vals"],
            inputs["inten_mean_label"],
        ),
        # Each metric full-size (intensity shown for mean and median). Full tier fonts
        # (16/14/14 pt) match the Z500 EPE.4 singular-boxplot-page style; the region/bbox
        # context lives in the axes title (no separate suptitle), same convention as
        # ``_boxplot_to_pdf`` in ``common.epe_04_boxplot_render``.
        "epe_occurrence": build_full_bar_figure(inputs["occ_pct"], labels, zvals, region, bbox_step,
                                    "EPE occurrence by Z500-RR sub-regime",
                                    "% of group days that are extreme", counts=totals),
        "duration": build_full_box_figure(inputs["dur_vals"], labels, zvals, region, bbox_step,
                                    "Convective duration by Z500-RR sub-regime", "hours per flagged cell / day"),
        "intensity_mean": build_full_box_figure(inputs["int_mean_vals"], labels, zvals, region, bbox_step,
                                    "Precipitation intensity (mean) by Z500-RR sub-regime",
                                    inputs["inten_mean_label"]),
        "intensity_median": build_full_box_figure(inputs["int_median_vals"], labels, zvals, region, bbox_step,
                                    "Precipitation intensity (median) by Z500-RR sub-regime",
                                    inputs["inten_median_label"]),
        "connectivity": build_full_box_figure(inputs["conn_vals"], labels, zvals, region, bbox_step,
                                    "Spatial clustering (connectivity) by Z500-RR sub-regime",
                                    "# connected components / day"),
        # Spatial mean/median precip on flagged cells — RR analogue of the legacy Z500
        # pipeline's (b1) boxplots (see ``_spatial_precip_per_day`` docstring for why this
        # uses ``P_conv_total_grid_day`` rather than the full daily ``P``).
        "spatial_precip_mean": build_full_box_figure(inputs["spat_mean_vals"], labels, zvals, region, bbox_step,
                                    "Spatial mean precip on flagged cells by Z500-RR sub-regime",
                                    "mean P_conv_total_grid_day on flagged cells (mm)"),
        "spatial_precip_median": build_full_box_figure(inputs["spat_median_vals"], labels, zvals, region, bbox_step,
                                    "Spatial median precip on flagged cells by Z500-RR sub-regime",
                                    "median P_conv_total_grid_day on flagged cells (mm)"),
        "spatial_precip_max": build_full_box_figure(inputs["spat_max_vals"], labels, zvals, region, bbox_step,
                                    "Spatial max precip on flagged cells by Z500-RR sub-regime",
                                    "max P_conv_total_grid_day on flagged cells (mm)"),
        # Excess-over-threshold (POT) severity — unlike raw amount, doesn't converge across
        # regimes at high percentiles (see ``_excess_spatial_precip_per_day`` docstring).
        "excess_mean": build_full_box_figure(inputs["excess_mean_vals"], labels, zvals, region, bbox_step,
                                    "Excess over threshold (mean) by Z500-RR sub-regime",
                                    "mean excess over P_q_hourly on flagged cells (mm)"),
        "excess_median": build_full_box_figure(inputs["excess_median_vals"], labels, zvals, region, bbox_step,
                                    "Excess over threshold (median) by Z500-RR sub-regime",
                                    "median excess over P_q_hourly on flagged cells (mm)"),
        "excess_max": build_full_box_figure(inputs["excess_max_vals"], labels, zvals, region, bbox_step,
                                    "Excess over threshold (max) by Z500-RR sub-regime",
                                    "max excess over P_q_hourly on flagged cells (mm)"),
    }

    written: list[Path] = [out_txt]
    pdf_ctx = None
    if "pdf" in formats:
        out_pdf = out_dir / f"rr_statistics_{region}_{bbox_step}{ward_tag}.pdf"
        pdf_ctx = PdfPages(out_pdf)

    try:
        for key, fig in figs.items():
            if "png" in formats:
                out_png = out_dir / f"rr_statistics_{region}_{bbox_step}{ward_tag}_{key}.png"
                fig.savefig(out_png, bbox_inches="tight", dpi=200)
                written.append(out_png)
                print(f"[rr_stats] wrote {out_png}")
            if pdf_ctx is not None:
                pdf_ctx.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    finally:
        if pdf_ctx is not None:
            pdf_ctx.close()

    if pdf_ctx is not None:
        written.append(out_pdf)
        print(f"[rr_stats] Saved: {out_pdf}")

    return written


def main(argv=None) -> int:
    import sys
    p = argparse.ArgumentParser(description="RR sub-regime summary statistics (per Z500×RR).")
    p.add_argument("--region", required=True, choices=("europe", "northamerica"))
    p.add_argument("--bbox-step", required=True)
    p.add_argument("--module-root", type=Path, default=None)
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--epe-region-label", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--mode", choices=("bbox", "regime"), default=None,
                   help="Force which pipeline's RR regimes file to plot (rr_02 bbox-mode vs "
                        "rr_022 per-regime-mode). Default: auto-detect (prefers bbox-mode).")
    p.add_argument(
        "--formats", default="png",
        help="Comma-separated output formats: png (default, one file per figure) and/or "
        "pdf (single multi-page file, original rr_03 behavior).",
    )
    p.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="kmeans",
                   help="Which Z500 parent-regime source to summarize RR sub-regimes against "
                        "('ward' reads '_zward'-tagged RR/Z500 files). Must match how the "
                        "RR regimes being visualized were built.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.module_root is None:
        args.module_root = Path(__file__).resolve().parents[2]

    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())

    run_rr_statistics(
        module_root=args.module_root.expanduser().resolve(),
        region=args.region, bbox_step=args.bbox_step,
        smooth_window=args.smooth_window,
        epe_region_label=args.epe_region_label, out_dir=args.out_dir,
        mode=args.mode, formats=formats,
        z500_cluster_method=args.z500_cluster_method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
