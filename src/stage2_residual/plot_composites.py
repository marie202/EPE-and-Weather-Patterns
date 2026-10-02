#!/usr/bin/env python3
"""
Composite map figures for residual regimes — additive-colour view.

Saved as rr_composites_{region}_{bbox}.pdf, one page per Z500 parent regime.
Each page has one row per RR sub-regime occurring within that Z500 regime
(>= MIN_DAYS). Each row, left to right:

  1. Mean Z500 (all days of the Z500 regime) — signed anomaly (RdBu).
  2. One panel per impact field, in greedy-AUC importance order (rr_01 /
     write_field_selection.select_fields — field 1 = the single most important
     field for that Z500 regime, field 2 = second, field 3 = third), drawn in
     PHYSICAL units (native colorbar label, not a normalised fraction) with a
     colourmap chosen per field PER PAGE (see _field_sign_mode): a field whose
     composite value is one-signed across every bin on this page gets a plain
     sequential bar in its own RGB channel hue (Reds / Greens / Blues, or the
     complementary cyan/magenta/yellow if that field is one-signed NEGATIVE)
     with no wasted/clipped half; a field that genuinely straddles zero across
     the page's bins gets the SAME shared yellow (negative) <-> blue (positive)
     diverging bar as every other mixed-sign field, regardless of channel — so
     a field that matters for the high-risk bin but trends DOWN there still
     shows up (instead of being clipped to a blank/zero panel), and every
     diverging panel on the page reads with one consistent palette.
  3. The additive RGB composite of those impact fields (still normalised 0..1
     per channel internally, since combining physically-incommensurate units
     additively needs a common scale): a positive value adds to its own
     channel (R+G -> yellow, as before); a negative value adds to the OTHER
     two channels instead (its complementary hue), so e.g. a strong negative
     anomaly in the field mapped to R reads as cyan.

The mean Z500 of the Z500 regime (all days) is overlaid as labelled black
isolines on every panel except column 1. Coastlines, European country borders,
the German outline and the Germany AOI box are drawn on every panel.

Usage:
    python -m stage2_residual.plot_composites \
        --region europe --bbox-step B5 --module-root /path/to/detect_weather
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Must be set before netCDF4/HDF5 is loaded (i.e. before the xarray import below). Repeatedly
# opening/closing the same NetCDF file in one process (e.g. re-reading a convective_duration_*.nc
# for a second metric) segfaults inside HDF5's file-locking path when the file lives on a
# cloud-sync-managed filesystem (this project is under a Nextcloud-synced folder) — disabling
# locking is the standard fix and is safe here since every open is read-only, single-process.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from common.colorbar_discretize import (
    discretize_diverging,
    discretize_sequential_from_floor,
    style_discrete_ticks,
)
from .plot_results import _dw_root, _find_anomaly, _find_raw_z500, _find_rr_regimes, _find_z500_regimes, _load_labels
from .plot_subregimes import (
    MIN_DAYS,
    FieldData,
    _composite_over_dates,
    _crop_field_to_bbox,
    _epe_frequency_over_dates,
    _load_econv,
    _load_field,
    _load_raw_z500_field,
    _norm_dates,
    _overlay_z500_isolines,
    _z500_raw_contour_levels,
    epe_basis_label,
    find_duration_file,
    resolve_epe_label,
)
from . import rr_geo
from .feature_categories import field_label
import common.regime_alignment as regime_alignment
from common.physics import G0, z500_anomaly_units_look_physical
from common.z500_precip_panel_figures import _z500_fill_label
from common.fixed_color_scales import (
    EPE_FLAG_FREQ_HIGHRISK,
    EPE_FLAG_FREQ_RR,
    Z500_MEAN_ANOM_GPM,
    default_fixed_scales_path,
    load_fixed_color_scales,
)

_ALIGNMENT_YAML_DIR = Path(__file__).resolve().parents[2] / "src" / "common"
_ALIGNMENT_CACHE: dict[str, tuple[str, list[dict]]] = {}


def _get_regime_alignment(region: str) -> tuple[str, list[dict]]:
    """Cached ``regime_alignment.load_regime_alignment`` for ``region`` -- the same
    ``regime_alignment_<region>.yaml`` the Z500 notebook (``2_3_Results_Z500_Stats.ipynb``) and
    ``plot_rr_statistics.py`` use, so a Z500 parent regime's composite page is ordered/labelled by
    the same canonical (physical) id everywhere.
    """
    if region not in _ALIGNMENT_CACHE:
        path = _ALIGNMENT_YAML_DIR / f"regime_alignment_{region}.yaml"
        _ALIGNMENT_CACHE[region] = regime_alignment.load_regime_alignment(path)
    return _ALIGNMENT_CACHE[region]

_EPE_CMAP = "Blues"

# Per-regime composite page (_page_z500_composites, the rr_composites_*_Z<n>.png files):
# subplot titles + row labels are enlarged for readability. Colorbar tick labels use
# matplotlib's own default size; the colorbar axis label is 2pt larger than that (and the
# ternary legend's per-feature vertex labels match the colorbar label size -- see
# draw_rgb_scale's fontsize= arg in _add_mixture_legend).
_COMPOSITE_TITLE_FONTSIZE = 13
_COMPOSITE_ROWLABEL_FONTSIZE = 13
_COMPOSITE_CBAR_TICK_FONTSIZE = 10   # matplotlib's own colorbar-tick default (rcParams font.size)
_COMPOSITE_CBAR_LABEL_FONTSIZE = _COMPOSITE_CBAR_TICK_FONTSIZE + 2
_COMPOSITE_Z500_CBAR_LABEL_FONTSIZE = _COMPOSITE_CBAR_LABEL_FONTSIZE - 2

# Impact-field index -> (RGB channel name, single-hue colormap, legend colour).
_CHANNELS = ("R", "G", "B")
_CHANNEL_RGB = {"R": (0.85, 0.10, 0.10), "G": (0.10, 0.70, 0.10), "B": (0.10, 0.30, 0.90)}
_CHANNEL_CMAP = {"R": "Reds", "G": "Greens", "B": "Blues"}
# Complementary colour per channel (white minus the channel's positive hue) -- used as the
# NEGATIVE end of that channel's diverging/sequential colormap (cyan for R, magenta for G,
# yellow for B), and as what a negative composite anomaly bleeds into on the additive RGB panel.
_CHANNEL_COMPLEMENT = {c: tuple(1.0 - v for v in rgb) for c, rgb in _CHANNEL_RGB.items()}


_DIVERGING_YELLOW_BLUE = "rr_composite_diverging_yb"


def _register_diverging_yellow_blue_cmap() -> str:
    """Register (once, idempotently) ONE shared diverging colormap -- yellow (negative) -> white
    (0) -> blue (positive) -- used for every per-field panel that needs a diverging bar (see
    ``_field_sign_mode``), regardless of which RGB channel that field is mapped to. Deliberately
    NOT per-channel (an earlier version used a different hue pair per channel, e.g. red<->cyan for
    the R-mapped field): a shared, fixed diverging palette is easier to read at a glance across
    many panels than a different colour pair per field. Only the additive-RGB composite panel
    still needs each field in its own primary hue, to combine physically-incommensurate fields
    into one image -- see ``_signed_norm``."""
    name = _DIVERGING_YELLOW_BLUE
    if name not in matplotlib.colormaps:
        cmap = LinearSegmentedColormap.from_list(
            name, [(0.90, 0.75, 0.10), (1.0, 1.0, 1.0), (0.10, 0.30, 0.90)], N=256,
        )
        matplotlib.colormaps.register(cmap, name=name)
    return name


def _channel_complement_seq_cmaps() -> dict[str, str]:
    """Register (once, idempotently) a plain sequential white -> complementary-hue colormap per
    channel, for fields that are one-signed NEGATIVE across a whole page (see
    ``_field_sign_mode``) -- same colour identity as the diverging map's negative half, without
    wasting the other (unused) half of a diverging bar on a sign the field never takes."""
    names: dict[str, str] = {}
    for ch, comp in _CHANNEL_COMPLEMENT.items():
        name = f"rr_composite_{ch}_complement_seq"
        if name not in matplotlib.colormaps:
            cmap = LinearSegmentedColormap.from_list(name, [(1.0, 1.0, 1.0), comp], N=256)
            matplotlib.colormaps.register(cmap, name=name)
        names[ch] = name
    return names


_DIVERGING_YELLOW_BLUE_CMAP = _register_diverging_yellow_blue_cmap()
_CHANNEL_COMPLEMENT_CMAP = _channel_complement_seq_cmaps()


def _signed_norm(comp: np.ndarray, scale: float) -> np.ndarray:
    """Composite anomaly normalised to [-1, 1] by ``scale`` -- sign preserved. Positive draws in
    the field's own primary hue, negative in its complementary hue. Used only for the additive-
    RGB panel, which needs a common 0..1-per-channel scale to combine physically-incommensurate
    fields (J/kg, kg/m^2, ...) into one image -- the per-field panels themselves draw raw physical
    values with the shared yellow<->blue diverging bar instead (see ``_field_cmap_norm``)."""
    return np.clip(comp / scale, -1.0, 1.0)


def _field_scale(comps: dict[int, np.ndarray]) -> float:
    if not comps:
        return 1.0
    return max(float(np.nanpercentile(np.abs(np.stack(list(comps.values()))), 95)), 1e-6)


def _field_sign_mode(comps: dict[int, np.ndarray], scale: float) -> str:
    """'pos' | 'neg' | 'mixed' -- whether a field's composite values across EVERY bin on this
    page are one-signed (draw a plain sequential bar, no wasted/clipped half) or genuinely
    straddle zero (draw the signed diverging bar). Tolerance scales with the field's own
    magnitude (``scale``, the page's 95th-percentile |value|) rather than a fixed epsilon, since
    physical units span wildly different magnitudes across fields (CAPE ~1e2 vs specific
    humidity ~1e-3)."""
    if not comps:
        return "mixed"
    tol = max(scale, 1e-6) * 1e-6
    vals = [v for v in comps.values() if v is not None]
    if not vals:
        return "mixed"
    vmin = min(float(np.nanmin(v)) for v in vals)
    vmax = max(float(np.nanmax(v)) for v in vals)
    if vmin >= -tol:
        return "pos"
    if vmax <= tol:
        return "neg"
    return "mixed"


def _field_cmap_norm(channel: str, sign: str, scale: float):
    """(cmap, norm, extend) for a per-field panel's colorbar, applied directly to PHYSICAL-unit
    composite values (not a normalised fraction) -- 'pos'/'neg' draw a plain one-sided sequential
    bar in the field's own channel hue (no unused half); 'mixed' draws the SAME shared yellow
    (negative) <-> blue (positive) diverging bar, spanning [-scale, scale], for every field
    regardless of channel -- one consistent diverging palette is easier to read across many
    panels than a different hue pair per channel."""
    if sign == "mixed":
        return discretize_diverging(-scale, scale, _DIVERGING_YELLOW_BLUE_CMAP)
    if sign == "pos":
        return discretize_sequential_from_floor(scale, _CHANNEL_CMAP[channel], vmin=0.0)
    # sign == "neg": mirror image of the "pos" floor scale (floor at 0, extend downward past
    # -scale) built by negating+reversing a positive-magnitude floor scale -- discretize_
    # sequential_from_floor itself always floors at vmin and extends past vmax, which is
    # backwards for negative-only data (floor conceptually at 0, extend past -scale).
    cmap_pos, norm_pos, _ = discretize_sequential_from_floor(scale, _CHANNEL_COMPLEMENT_CMAP[channel], vmin=0.0)
    colors = cmap_pos(np.linspace(0.0, 1.0, cmap_pos.N))
    cmap_neg = ListedColormap(colors[::-1])
    cmap_neg.set_under(colors[-1])
    boundaries_neg = (-norm_pos.boundaries)[::-1]
    norm_neg = BoundaryNorm(boundaries_neg, ncolors=cmap_neg.N)
    return cmap_neg, norm_neg, "min"


def _additive_over_white(rgb: np.ndarray) -> np.ndarray:
    """Composite an additive RGB image over a white background.

    Empty pixels (all channels ~0) become white; a single channel becomes a
    pastel tint; overlaps keep their additive secondary hue (R+G -> yellow).
    Opacity is driven by the strongest channel, so colour saturation still
    tracks signal strength.
    """
    m = rgb.max(axis=-1, keepdims=True)
    return np.clip(rgb + (1.0 - m), 0.0, 1.0)


def _imshow_rgb(ax, is_geo, lats, extent, rgb, transform) -> None:
    """Render an (lat, lon, 3) additive RGB array north-up over white."""
    rgb = _additive_over_white(rgb)
    lat_ascending = lats[0] < lats[-1]
    img = rgb[::-1, :, :] if lat_ascending else rgb
    kw = dict(origin="upper", extent=[extent[0], extent[1], extent[2], extent[3]],
              interpolation="nearest", aspect="auto")
    if is_geo and transform is not None:
        kw["transform"] = transform
    ax.imshow(np.clip(img, 0, 1), **kw)


def _draw_field(ax, is_geo, lons, lats, data, *, cmap, norm=None, vmin=None, vmax=None):
    tr = rr_geo.data_transform() if is_geo else None
    kw = dict(cmap=cmap, shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if tr is not None:
        kw["transform"] = tr
    return ax.pcolormesh(lons, lats, data, **kw)


def _draw_empty_panel(ax, msg: str = "") -> None:
    """Grey placeholder panel (e.g. a Z500 regime with fewer than 3 impact fields, in the
    extreme-summary page's fixed-width field-panel columns). Mirrors plot_rgb_overview.py's
    ``_draw_empty_panel``."""
    ax.set_facecolor("#e8e8e8")
    if msg:
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=7,
                color="#999999", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# ---------------------------------------------------------------------------
# One page per Z500 regime
# ---------------------------------------------------------------------------

def _page_z500_composites(
    z500_val, z_all_dates, rr_present, group_dates_by_rr, fields_data,
    region, bbox, bbox_dict, module_root,
    econv=None, edates=None, elats=None, elons=None,
    z500_raw_fd=None, per_z500_fields=None, variant_label="all days",
    z500_vmin=None, z500_vmax=None, epe_vmax_fixed=None, row_label_prefix="SR",
) -> "plt.Figure | None":
    """Build the additive-composite page for one Z500 regime.

    ``epe_vmax_fixed``, if given, is in PERCENT (0-100) -- matching the Z500-only pipeline's
    ``precip_vmin``/``precip_vmax`` convention (``_composite_flag_frequency`` there already
    returns "% of regime days") -- NOT the 0-1 fraction ``_epe_frequency_over_dates`` returns
    internally here.

    ``row_label_prefix``: row-label prefix for the ``{prefix}{rr}`` y-axis tag. Both
    :func:`build_z500_composite_figure` (discrete k-means cluster) and
    :func:`build_z500_composite_figure_logistic` (``prob_bin``, a quantile bin of
    ``predicted_extreme_prob``) use the default "SR" here, so every composite page reads
    "SR1"/"SR2"/... regardless of which Stage-2 method produced the row. (``plot_rr_statistics.py``
    and ``plot_rgb_overview.py`` still print "Bin" for the logistic path in their own labels —
    only the composite-page row label was asked to stay "SR" uniformly.)

    Returns the ``Figure`` without saving or closing it — callers decide
    whether to hand it to a ``PdfPages`` (batch/HPC path) or display it
    inline for interactive tweaking (notebook path).
    """
    # per_z500_fields (from the RR regimes file's `per_regime_fields` attr, set by
    # per-regime-mode rr_02 runs) lets each Z500 page pick its OWN residual fields
    # instead of the one field set shared by every page. per_z500_fields is None when no
    # per-regime map exists at all -- then fields_data's insertion order is the run-wide
    # filename-encoded (importance) order, so falling back to it is safe. But if
    # per_z500_fields IS present and just lacks (or has an empty) entry for this one
    # Z500 value, fields_data's order is the *alphabetical* load-universe order (see
    # all_fields = sorted(...) above), NOT an importance order -- warn loudly so this
    # doesn't silently mislabel panels/RGB channels as importance-ranked.
    candidate_fields = per_z500_fields.get(z500_val) if per_z500_fields else None
    if per_z500_fields is not None and not candidate_fields:
        print(f"[rr_composites] WARNING: Z500={z500_val} has no per-regime field selection "
              f"(per_regime_fields attr present but empty/missing for this regime) -- falling "
              f"back to fields_data's alphabetical load order, NOT an importance order.")
    impact_fields = [f for f in (candidate_fields or fields_data) if f != "z500" and f in fields_data][:3]
    has_z500 = "z500" in fields_data
    have_epe = econv is not None
    if not impact_fields:
        return None

    # Z500 columns: regime-mean (all days) + sub-regime-days.
    ncol = (2 if has_z500 else 0) + len(impact_fields) + 1 + (1 if have_epe else 0)
    nrow = len(rr_present)
    if nrow == 0:
        return None

    # Parent Z500 anomaly mean (all days) + per-sub-regime Z500 anomaly mean, shared scale.
    parent_z500 = z_lats = z_lons = z_norm = z_cmap = z_extend = None
    z500_sub: dict[int, np.ndarray] = {}
    if has_z500:
        fdz = fields_data["z500"]
        # Magnitude-aware gpm/normalized detection (common.physics.
        # z500_anomaly_units_look_physical), same guard the Z500-only pipeline applies -- a raw
        # units-string check alone isn't enough: a Yeo-Johnson-transformed (~N(0,1)) anomaly file
        # can carry a stale "m**2 s**-2"-looking units attr forward from before the transform,
        # which would otherwise get divided by g0 a second time and mislabeled "(gpm)" at
        # ~1/500th its real scale (see preprocessing_helpers.apply_yeo_johnson's docstring).
        z500_is_gpm = z500_anomaly_units_look_physical(xr.DataArray(fdz.data), fdz.units)
        z500_unit_scale = (1.0 / G0) if z500_is_gpm else 1.0
        z_unit_label = _z500_fill_label(fdz.units if z500_is_gpm else "1")
        parent_z500 = _composite_over_dates(fdz, z_all_dates)
        if parent_z500 is not None and z500_unit_scale != 1.0:
            parent_z500 = parent_z500 * z500_unit_scale
        z_lats, z_lons = fdz.lats, fdz.lons
        for rr in rr_present:
            c = _composite_over_dates(fdz, group_dates_by_rr[rr])
            if c is not None:
                if z500_unit_scale != 1.0:
                    c = c * z500_unit_scale
                z500_sub[rr] = c
        if z500_vmin is not None and z500_vmax is not None:
            zv = max(abs(float(z500_vmin)), abs(float(z500_vmax)), 1e-6)
            z_cmap, z_norm, z_extend = discretize_diverging(-zv, zv, "RdBu_r")
        else:
            z_stack = [a for a in ([parent_z500] + list(z500_sub.values())) if a is not None]
            if z_stack:
                zv = max(float(np.nanpercentile(np.abs(np.stack(z_stack)), 98)), 1e-6)
                z_cmap, z_norm, z_extend = discretize_diverging(-zv, zv, "RdBu_r")
    z_extent = rr_geo.map_extent(bbox_dict, z_lats, z_lons) if has_z500 else None

    # Raw Z500 mean per sub-regime for isolines (one set per row, in gpm) + parent mean for col 0.
    z500_raw_sub: dict[int, np.ndarray] = {}
    z500_raw_parent: np.ndarray | None = None
    z_iso_parent: np.ndarray | int = 7
    z_lats_raw = z_lons_raw = None
    if z500_raw_fd is not None:
        z_lats_raw, z_lons_raw = z500_raw_fd.lats, z500_raw_fd.lons
        z500_raw_parent = _composite_over_dates(z500_raw_fd, z_all_dates)
        if z500_raw_parent is not None:
            z_iso_parent = _z500_raw_contour_levels(z500_raw_parent)
        for rr in rr_present:
            c = _composite_over_dates(z500_raw_fd, group_dates_by_rr[rr])
            if c is not None:
                z500_raw_sub[rr] = c

    # Per-impact-field composites (over each sub-regime group) + shared scale + sign mode (see
    # _field_sign_mode -- whether this field is one-signed across every bin on this page, so its
    # panel gets a plain sequential bar instead of a diverging one with an unused half).
    field_comps: dict[str, dict[int, np.ndarray]] = {}
    field_scale: dict[str, float] = {}
    field_sign: dict[str, str] = {}
    field_grid: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    channel_of: dict[str, str] = {}
    for i, field in enumerate(impact_fields):
        fd = fields_data[field]
        comps = {rr: c for rr in rr_present if (c := _composite_over_dates(fd, group_dates_by_rr[rr])) is not None}
        field_comps[field] = comps
        field_scale[field] = _field_scale(comps)
        field_sign[field] = _field_sign_mode(comps, field_scale[field])
        field_grid[field] = (fd.lats, fd.lons)
        channel_of[field] = _CHANNELS[i]

    ref_lats, ref_lons = field_grid[impact_fields[0]]
    ref_shape = (len(ref_lats), len(ref_lons))
    add_extent = rr_geo.map_extent(bbox_dict, ref_lats, ref_lons)

    # EPE flag-frequency per sub-regime group (shared scale across rows).
    if have_epe:
        epe_freqs = {rr: _epe_frequency_over_dates(econv, edates, group_dates_by_rr[rr]) for rr in rr_present}
        if epe_vmax_fixed is not None:
            epe_vmax_pct = max(float(epe_vmax_fixed), 0.1)
        else:
            epe_vmax = max((np.nanpercentile(f, 99) for f, _ in epe_freqs.values() if f is not None), default=1.0)
            epe_vmax_pct = 100 * max(float(epe_vmax), 1e-3)
        epe_extent = rr_geo.map_extent(None, elats, elons)

    fig = plt.figure(figsize=(3.3 * ncol, 3.3 * nrow + 0.8))

    rgb_col_ax = None   # row-0 Additive RGB panel axes, used to horizontally align the ternary
                        # legend under that column (see _add_mixture_legend)
    for row, rr in enumerate(rr_present):
        n_days = len(group_dates_by_rr[rr])
        col = 0

        # Raw Z500 isolines for this row: mean over sub-regime days, in gpm.
        z_raw_rr = z500_raw_sub.get(rr)
        z_iso_levels = _z500_raw_contour_levels(z_raw_rr) if z_raw_rr is not None else 7

        if has_z500:
            # Col 0: Z500 anomaly mean (all regime days) + raw Z500 parent regime isolines.
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=z_extent)
            if parent_z500 is not None and z_norm is not None:
                im = _draw_field(ax, is_geo, z_lons, z_lats, parent_z500, cmap=z_cmap, norm=z_norm)
                cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=z_extend)
                cb.set_label(z_unit_label, fontsize=_COMPOSITE_Z500_CBAR_LABEL_FONTSIZE)
                cb.ax.tick_params(labelsize=_COMPOSITE_CBAR_TICK_FONTSIZE)
                style_discrete_ticks(cb)
            if z500_raw_parent is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z500_raw_parent, levels=z_iso_parent)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title("Z500 anomaly\n(all days)", fontsize=_COMPOSITE_TITLE_FONTSIZE)
            ax.set_ylabel(f"{row_label_prefix}{rr}\n{n_days} d", fontsize=_COMPOSITE_ROWLABEL_FONTSIZE)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1

            # Col 1: Z500 anomaly mean (sub-regime days) + raw Z500 sub-regime isolines.
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=z_extent)
            zc = z500_sub.get(rr)
            if zc is not None and z_norm is not None:
                im = _draw_field(ax, is_geo, z_lons, z_lats, zc, cmap=z_cmap, norm=z_norm)
                cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=z_extend)
                cb.set_label(z_unit_label, fontsize=_COMPOSITE_Z500_CBAR_LABEL_FONTSIZE)
                cb.ax.tick_params(labelsize=_COMPOSITE_CBAR_TICK_FONTSIZE)
                style_discrete_ticks(cb)
            if z_raw_rr is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
            elif z500_raw_parent is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z500_raw_parent, levels=z_iso_parent)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title(f"Z500 anomaly\n(sub-regime, {variant_label})", fontsize=_COMPOSITE_TITLE_FONTSIZE)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1

        # Per-field channel-colour panels, in greedy-AUC importance rank order (rank 1 = R = the
        # most important field for this Z500 regime; see per_z500_fields / impact_fields above).
        for rank, field in enumerate(impact_fields, start=1):
            lats, lons = field_grid[field]
            extent = rr_geo.map_extent(bbox_dict, lats, lons)
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=extent)
            comp = field_comps[field].get(rr)
            if comp is not None:
                ch_cmap, ch_norm, ch_extend = _field_cmap_norm(
                    channel_of[field], field_sign[field], field_scale[field]
                )
                im = _draw_field(ax, is_geo, lons, lats, comp, cmap=ch_cmap, norm=ch_norm)
                cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=ch_extend)
                cb.set_label(fields_data[field].units or "", fontsize=_COMPOSITE_CBAR_LABEL_FONTSIZE)
                cb.ax.tick_params(labelsize=_COMPOSITE_CBAR_TICK_FONTSIZE)
                style_discrete_ticks(cb)
            if z_raw_rr is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
            elif has_z500:
                _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, parent_z500)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title(f"#{rank} {field_label(field)} ({channel_of[field]})", fontsize=_COMPOSITE_TITLE_FONTSIZE)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1

        # Additive RGB panel.
        ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=add_extent)
        if row == 0:
            rgb_col_ax = ax
        rgb = np.zeros((*ref_shape, 3), dtype=float)
        for field in impact_fields:
            comp = field_comps[field].get(rr)
            if comp is not None and comp.shape == ref_shape:
                # nan_to_num: a field missing data at a pixel (e.g. SST over land) must
                # contribute zero to its channel, not NaN -- _additive_over_white's
                # rgb.max(axis=-1) propagates a single NaN channel across all three,
                # blanking out otherwise-valid signal from the other two fields.
                val = np.nan_to_num(_signed_norm(comp, field_scale[field]), nan=0.0)
                own = _CHANNELS.index(channel_of[field])
                # Positive anomaly adds to the field's own channel (as before); negative adds
                # to the OTHER two channels instead, i.e. bleeds toward the complementary hue
                # (e.g. a strongly negative R-mapped field pushes G and B up -> reads as cyan).
                rgb[..., own] += np.clip(val, 0.0, 1.0)
                neg = np.clip(-val, 0.0, 1.0)
                for other in range(3):
                    if other != own:
                        rgb[..., other] += neg
        rgb = np.clip(rgb, 0.0, 1.0)
        tr = rr_geo.data_transform() if is_geo else None
        _imshow_rgb(ax, is_geo, ref_lats, add_extent, rgb, tr)
        if z_raw_rr is not None:
            _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
        elif has_z500:
            _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, parent_z500)
        rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
        if row == 0:
            ax.set_title("Additive RGB", fontsize=_COMPOSITE_TITLE_FONTSIZE)
        ax.set_xticks([]); ax.set_yticks([])
        col += 1

        # EPE occurrence raster (flag frequency for this sub-regime group).
        if have_epe:
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=epe_extent)
            f, n = epe_freqs[rr]
            if f is not None:
                epe_cmap, epe_norm, epe_extend = discretize_sequential_from_floor(epe_vmax_pct, _EPE_CMAP, vmin=0)
                im = _draw_field(ax, is_geo, elons, elats, 100 * f, cmap=epe_cmap, norm=epe_norm)
                cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=epe_extend)
                cb.set_label("% days", fontsize=_COMPOSITE_CBAR_LABEL_FONTSIZE)
                cb.ax.tick_params(labelsize=_COMPOSITE_CBAR_TICK_FONTSIZE)
                style_discrete_ticks(cb)
            if z_raw_rr is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
            elif has_z500:
                _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, parent_z500)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title("EPE flagging frequency", fontsize=_COMPOSITE_TITLE_FONTSIZE)
            ax.set_xticks([]); ax.set_yticks([])

    _add_mixture_legend(fig, impact_fields, channel_of, rgb_col_ax)
    return fig


def _add_mixture_legend(fig, impact_fields, channel_of, rgb_col_ax=None) -> None:
    """RGB mixing triangle (or gradient/swatches for ≠3 fields) as colour key.

    The legend is placed in a fixed-height strip below the subplot area, horizontally
    aligned under the Additive RGB column (``rgb_col_ax`` -- that column's row-0 axes)
    so the key sits directly beneath the panels it explains rather than page-centered.
    We compute the strip height in figure-fraction units from an absolute
    target in inches so it does not overlap the maps regardless of row count.
    """
    from .plot_rgb_scale import draw_rgb_scale
    legend_h_in = 1.55         # absolute height of the legend box in inches
    pad_in      = 0.15         # gap between bottom of subplots and legend top
    fig_h = fig.get_figheight()
    legend_h = legend_h_in / fig_h
    pad      = pad_in      / fig_h
    bottom_reserved = legend_h + pad + 0.02   # total fraction reserved below subplots
    # Reposition subplots above the legend strip, then place the legend -- read rgb_col_ax's
    # position AFTER tight_layout since that call itself moves/resizes every axes on the page.
    fig.tight_layout(rect=(0, bottom_reserved, 1, 0.96))
    if rgb_col_ax is not None:
        pos = rgb_col_ax.get_position()
        left, width = pos.x0, pos.x1 - pos.x0
    else:
        left, width = 0.38, 0.22
    ax = fig.add_axes([left, 0.005, width, legend_h])
    draw_rgb_scale(
        ax, channel_of, _CHANNEL_RGB, impact_fields,
        fontsize=_COMPOSITE_CBAR_LABEL_FONTSIZE, tick_fontsize=_COMPOSITE_CBAR_TICK_FONTSIZE,
        title="",
    )


# ---------------------------------------------------------------------------
# One page across ALL Z500 regimes, each row = that regime's own extreme bin
# ---------------------------------------------------------------------------

def build_z500_extreme_summary_figure(inputs: dict) -> "plt.Figure | None":
    """One page, one row per Z500 regime, each row showing ONLY that regime's own highest
    probability bin (the "extreme-like" bin — for ``--bin-overrides-path``/adaptive-bin runs
    this is that regime's own max bin id, which can differ regime to regime; for a flat
    ``--n-bins`` split it is always ``n_bins``). Same column layout as
    :func:`_page_z500_composites` (Z500 anomaly x2, per-field panels in greedy-AUC importance
    order, additive RGB, EPE), but the ternary legend is drawn PER ROW on the far right (same
    approach as ``plot_rgb_overview.py``'s per-row legends) instead of one page-wide legend at
    the bottom, since different Z500 regimes can use different impact fields.

    ``inputs`` must be the dict returned by :func:`load_composite_inputs_logistic` — "highest
    bin = highest predicted_extreme_prob" only has meaning for the continuous logistic index;
    there is no discrete-k-means-mode equivalent (an RR cluster label has no ordering). Returns
    ``None`` if no Z500 regime has a bin clearing ``MIN_DAYS``.
    """
    from .plot_rgb_scale import draw_rgb_scale

    rr_dates, z500_regime, prob_bin = inputs["rr_dates"], inputs["z500_regime"], inputs["prob_bin"]
    fields_data, bbox_dict, module_root = inputs["fields_data"], inputs["bbox_dict"], inputs["module_root"]
    region, bbox_step = inputs["region"], inputs["bbox_step"]
    per_z500_fields = inputs.get("per_z500_fields")
    z500_raw_fd = inputs.get("z500_raw_fd")
    econv, edates, elats, elons = inputs.get("econv"), inputs.get("edates"), inputs.get("elats"), inputs.get("elons")
    epe_basis = inputs.get("epe_basis", "")
    z500_canonical_map = inputs.get("z500_canonical_map", {})
    has_z500 = "z500" in fields_data
    have_epe = econv is not None

    # One row per Z500 regime: its own top (highest predicted_extreme_prob) bin clearing
    # MIN_DAYS, restricted to its own per-regime impact fields.
    rows: list[dict] = []
    for z in inputs["z500_vals"]:
        z_mask = z500_regime == z
        z_all_dates = rr_dates[z_mask]
        bins_for_z = prob_bin[z_mask]
        candidate_fields = per_z500_fields.get(z) if per_z500_fields else None
        impact_fields = [f for f in (candidate_fields or fields_data) if f != "z500" and f in fields_data][:3]
        if not impact_fields or len(z_all_dates) == 0:
            continue
        valid_bins = [b for b in sorted({int(v) for v in bins_for_z})
                      if int((bins_for_z == b).sum()) >= MIN_DAYS]
        if not valid_bins:
            print(f"[rr_composites] extreme-summary: Z500-{z} has no bin with >= {MIN_DAYS} days — skipping")
            continue
        top_bin = max(valid_bins)
        rows.append(dict(
            z=z, z_all_dates=z_all_dates, group_dates=z_all_dates[bins_for_z == top_bin],
            impact_fields=impact_fields, top_bin=top_bin, canonical_id=z500_canonical_map.get(z),
        ))
    if not rows:
        return None

    n_field_cols = 3
    ncol = (2 if has_z500 else 0) + n_field_cols + 1 + (1 if have_epe else 0)
    nrow = len(rows)

    panel_size = 3.3
    legend_pad_in = 3.0
    panels_w = panel_size * ncol
    fig_w = panels_w + legend_pad_in
    fig_h = panel_size * nrow + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h))
    panels_frac = panels_w / fig_w

    # Shared Z500 colour scale, pooled across every row's parent-mean + top-bin composite.
    z_norm = z_cmap = z_extend = z_lats = z_lons = z_unit_label = None
    if has_z500:
        fdz = fields_data["z500"]
        z500_is_gpm = z500_anomaly_units_look_physical(xr.DataArray(fdz.data), fdz.units)
        z500_unit_scale = (1.0 / G0) if z500_is_gpm else 1.0
        z_unit_label = _z500_fill_label(fdz.units if z500_is_gpm else "1")
        z_lats, z_lons = fdz.lats, fdz.lons
        z_stack = []
        for r in rows:
            pz = _composite_over_dates(fdz, r["z_all_dates"])
            tz = _composite_over_dates(fdz, r["group_dates"])
            r["_z_parent"] = pz * z500_unit_scale if pz is not None else None
            r["_z_top"] = tz * z500_unit_scale if tz is not None else None
            z_stack += [a for a in (r["_z_parent"], r["_z_top"]) if a is not None]
        if z_stack:
            zv = max(float(np.nanpercentile(np.abs(np.stack(z_stack)), 98)), 1e-6)
            z_cmap, z_norm, z_extend = discretize_diverging(-zv, zv, "RdBu_r")
    z_extent = rr_geo.map_extent(bbox_dict, z_lats, z_lons) if has_z500 else None

    z_lats_raw = z_lons_raw = None
    if z500_raw_fd is not None:
        z_lats_raw, z_lons_raw = z500_raw_fd.lats, z500_raw_fd.lons
        for r in rows:
            r["_zraw_parent"] = _composite_over_dates(z500_raw_fd, r["z_all_dates"])
            r["_zraw_top"] = _composite_over_dates(z500_raw_fd, r["group_dates"])
    else:
        for r in rows:
            r["_zraw_parent"] = r["_zraw_top"] = None

    epe_extent = epe_vmax_pct = None
    if have_epe:
        epe_extent = rr_geo.map_extent(None, elats, elons)
        # Start at a small floor (not 1.0 -- f is already a 0..1 fraction, so seeding the running
        # max at 1.0 pins the colorbar to 100% regardless of the real data and washes out the
        # actual signal to near-white; see _page_z500_composites's equivalent computation, which
        # uses max(..., default=1.0) -- a fallback for "no rows had data at all", not a floor
        # compared against every row).
        epe_vmax = 1e-3
        for r in rows:
            f, n = _epe_frequency_over_dates(econv, edates, r["group_dates"])
            r["_epe"] = (f, n)
            if f is not None:
                epe_vmax = max(epe_vmax, float(np.nanpercentile(f, 99)))
        epe_vmax_pct = 100 * epe_vmax

    row_col0_axes: dict[int, "plt.Axes"] = {}
    row_impact_fields: dict[int, list[str]] = {}
    row_channel_of: dict[int, dict[str, str]] = {}

    for row_idx, r in enumerate(rows):
        z = r["z"]
        impact_fields = r["impact_fields"]
        channel_of = {f: _CHANNELS[i] for i, f in enumerate(impact_fields)}
        row_impact_fields[z] = impact_fields
        row_channel_of[z] = channel_of
        n_days = len(r["group_dates"])
        col = 0

        z_raw_top = r["_zraw_top"]
        z_raw_parent = r["_zraw_parent"]
        z_iso_levels = _z500_raw_contour_levels(z_raw_top) if z_raw_top is not None else 7
        z_iso_parent = _z500_raw_contour_levels(z_raw_parent) if z_raw_parent is not None else 7

        if has_z500:
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row_idx * ncol + col + 1), extent=z_extent)
            row_col0_axes[z] = ax
            if r["_z_parent"] is not None and z_norm is not None:
                im = _draw_field(ax, is_geo, z_lons, z_lats, r["_z_parent"], cmap=z_cmap, norm=z_norm)
                style_discrete_ticks(
                    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=z_extend, label=z_unit_label)
                )
            if z_raw_parent is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_parent, levels=z_iso_parent)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            _canon = r["canonical_id"]
            _canon_label = f" [R{_canon}]" if _canon is not None else ""
            ax.set_ylabel(f"Z500-{z}{_canon_label}\nbin{r['top_bin']} ({n_days} d)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0:
                ax.set_title("Z500 anomaly\n(all days)", fontsize=8)
            col += 1

            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row_idx * ncol + col + 1), extent=z_extent)
            if r["_z_top"] is not None and z_norm is not None:
                im = _draw_field(ax, is_geo, z_lons, z_lats, r["_z_top"], cmap=z_cmap, norm=z_norm)
                style_discrete_ticks(
                    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=z_extend, label=z_unit_label)
                )
            if z_raw_top is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_top, levels=z_iso_levels)
            elif z_raw_parent is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_parent, levels=z_iso_parent)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row_idx == 0:
                ax.set_title("Z500 anomaly\n(extreme bin)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1
        else:
            row_col0_axes[z] = None

        field_comp: dict[str, np.ndarray | None] = {}
        field_scale: dict[str, float] = {}
        field_sign: dict[str, str] = {}
        field_grid: dict[str, tuple] = {}
        for field in impact_fields:
            fd = fields_data[field]
            comp = _composite_over_dates(fd, r["group_dates"])
            field_comp[field] = comp
            scale = _field_scale({0: comp}) if comp is not None else 1.0
            field_scale[field] = scale
            field_sign[field] = _field_sign_mode({0: comp}, scale) if comp is not None else "mixed"
            field_grid[field] = (fd.lats, fd.lons)

        ref_lats = ref_lons = ref_shape = add_extent = None
        if impact_fields:
            ref_lats, ref_lons = field_grid[impact_fields[0]]
            ref_shape = (len(ref_lats), len(ref_lons))
            add_extent = rr_geo.map_extent(bbox_dict, ref_lats, ref_lons)

        for slot in range(n_field_cols):
            field = impact_fields[slot] if slot < len(impact_fields) else None
            extent = rr_geo.map_extent(bbox_dict, *field_grid[field]) if field else z_extent
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row_idx * ncol + col + 1), extent=extent)
            if field is not None:
                lats, lons = field_grid[field]
                comp = field_comp[field]
                if comp is not None:
                    ch_cmap, ch_norm, ch_extend = _field_cmap_norm(
                        _CHANNELS[slot], field_sign[field], field_scale[field]
                    )
                    im = _draw_field(ax, is_geo, lons, lats, comp, cmap=ch_cmap, norm=ch_norm)
                    style_discrete_ticks(plt.colorbar(
                        im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=ch_extend,
                        label=fields_data[field].units or "",
                    ))
                if z_raw_top is not None:
                    _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_top, levels=z_iso_levels)
                elif r.get("_z_top") is not None:
                    _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, r["_z_top"])
                rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
                if row_idx == 0:
                    ax.set_title(f"#{slot + 1} {field_label(field)} ({_CHANNELS[slot]})", fontsize=8)
            else:
                _draw_empty_panel(ax)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1

        # Additive RGB panel.
        ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row_idx * ncol + col + 1), extent=add_extent)
        if ref_shape is not None:
            rgb = np.zeros((*ref_shape, 3), dtype=float)
            for slot, field in enumerate(impact_fields):
                comp = field_comp[field]
                if comp is not None and comp.shape == ref_shape:
                    val = np.nan_to_num(_signed_norm(comp, field_scale[field]), nan=0.0)
                    rgb[..., slot] += np.clip(val, 0.0, 1.0)
                    neg = np.clip(-val, 0.0, 1.0)
                    for other in range(3):
                        if other != slot:
                            rgb[..., other] += neg
            rgb = np.clip(rgb, 0.0, 1.0)
            tr = rr_geo.data_transform() if is_geo else None
            _imshow_rgb(ax, is_geo, ref_lats, add_extent, rgb, tr)
            if z_raw_top is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_top, levels=z_iso_levels)
            elif r.get("_z_top") is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, r["_z_top"])
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
        else:
            _draw_empty_panel(ax)
        if row_idx == 0:
            ax.set_title("Additive RGB", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        col += 1

        if have_epe:
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row_idx * ncol + col + 1), extent=epe_extent)
            f, n = r["_epe"]
            if f is not None:
                epe_cmap, epe_norm, epe_extend = discretize_sequential_from_floor(epe_vmax_pct, _EPE_CMAP, vmin=0)
                im = _draw_field(ax, is_geo, elons, elats, 100 * f, cmap=epe_cmap, norm=epe_norm)
                style_discrete_ticks(
                    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, label="% days", extend=epe_extend)
                )
            if z_raw_top is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_top, levels=z_iso_levels)
            elif r.get("_z_top") is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, r["_z_top"])
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row_idx == 0:
                epe_title = "EPE flag frequency" + (f"\n({epe_basis})" if epe_basis else "")
                ax.set_title(epe_title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=(0.0, 0.0, panels_frac, 1.0))

    # One ternary legend PER ROW, aligned with that row's own panels (via its col-0 axes' actual
    # post-tight_layout position) — different Z500 regimes use different field sets here, so a
    # single shared legend (like _add_mixture_legend) would misrepresent every row but one.
    for r in rows:
        z = r["z"]
        ax0 = row_col0_axes.get(z)
        if ax0 is None:
            continue
        pos = ax0.get_position()
        legend_left = panels_frac + 0.02
        legend_width = max(0.05, (1.0 - panels_frac) - 0.04)
        ax_legend = fig.add_axes([legend_left, pos.y0, legend_width, pos.y1 - pos.y0])
        draw_rgb_scale(ax_legend, row_channel_of[z], _CHANNEL_RGB, row_impact_fields[z], fontsize=7)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_extreme_day_mask(path: Path) -> "pd.DatetimeIndex | None":
    """Day-level extreme-day flag from a duration file, as the dates that are flagged.

    Reuses ``epe_functions._daily_extreme_mask`` (prefers ``flagged_domain_pctl`` over the
    looser ``flagged_any_gridcell`` when both are present — same precedence used by
    ``plot_rr_statistics.py``), so the extreme-only composite variant is driven by whichever
    extreme-day definition ``ext_path``/``EPE_FILE`` actually resolves to for this run, not a
    hardcoded one. Returns ``None`` if the file has neither variable.
    """
    from epe_preprocessing.epe_functions import _daily_extreme_mask

    ds = xr.open_dataset(path)
    try:
        mask = _daily_extreme_mask(extremes_ds=None, duration_ds=ds)
        if mask is None:
            return None
        time_dim = mask.dims[0]
        flagged = np.asarray(mask.values, dtype=bool)
        return _norm_dates(ds[time_dim].values)[flagged]
    finally:
        ds.close()


def _load_field_geo_epe_inputs(module_root, dw, region, bbox_step, smooth_window, all_fields,
                                area_weight_tag: str = "") -> dict:
    """Shared tail of ``load_composite_inputs``/``load_composite_inputs_logistic``:
    per-field anomaly loading, bbox geometry, raw Z500 (for isolines), and the
    optional EPE occurrence raster. Everything here is independent of which
    Stage-2 label source (discrete RR regimes vs. the continuous logistic
    index) drove ``all_fields``.
    """
    fields_data: dict[str, FieldData] = {}
    for field in all_fields:
        p = _find_anomaly(dw, region, field, bbox_step, smooth_window, area_weight_tag=area_weight_tag)
        fd = _load_field(p) if p else None
        if fd is not None:
            fields_data[field] = fd
            print(f"[rr_composites]   ✓ {field}: {p.name}")
        else:
            print(f"[rr_composites]   ✗ {field}: anomaly not found — skipping")
    if not fields_data:
        raise RuntimeError("No field anomalies loaded.")

    bbox_dict = rr_geo.load_bbox_for_step(module_root, region, bbox_step)

    # Raw Z500 geopotential (optional — isolines fall back to anomaly contours if missing).
    # Loaded from a domain-wide file (one per region, not per bbox step), so crop it to this
    # bbox's window before computing anything — otherwise contour levels get spaced across the
    # WHOLE domain's value range and only 0-1 of them ever fall inside the visible bbox.
    raw_z500_path = _find_raw_z500(dw, region)
    z500_raw_fd = None
    if raw_z500_path is not None:
        print(f"[rr_composites] Raw Z500: {raw_z500_path.name}")
        z500_raw_fd = _crop_field_to_bbox(_load_raw_z500_field(raw_z500_path), bbox_dict)
    else:
        print("[rr_composites] WARNING: raw Z500 not found; isolines will use anomaly field")

    # EPE occurrence raster (optional last column) — same source as rr_subregimes.
    epe_label = resolve_epe_label()
    ext_path = find_duration_file(dw, epe_label, module_root)
    econv = edates = elats = elons = None
    extreme_dates = None
    epe_basis = ""
    if ext_path is None:
        print(f"[rr_composites] WARNING: no EPE file (label={epe_label}); "
              "EPE column and extreme-only composites skipped")
    else:
        print(f"[rr_composites] EPE file: {ext_path}")
        epe_basis = epe_basis_label(ext_path)
        econv, edates, elats, elons = _load_econv(ext_path)
        extreme_dates = _load_extreme_day_mask(ext_path)
        if extreme_dates is None:
            print(f"[rr_composites] WARNING: no day-level extreme flag in {ext_path.name}; "
                  "extreme-only composites skipped")
        else:
            print(f"[rr_composites] extreme days: {len(extreme_dates)}")

    return dict(
        fields_data=fields_data, bbox_dict=bbox_dict, z500_raw_fd=z500_raw_fd,
        econv=econv, edates=edates, elats=elats, elons=elons,
        extreme_dates=extreme_dates, epe_basis=epe_basis,
    )


def load_composite_inputs(module_root, region, bbox_step, smooth_window=11, mode=None,
                           z500_cluster_method: str = "ward", area_weight_tag: str = "") -> dict:
    """Resolve and load every input the composite pages need.

    Separated from ``run_composite_plots`` so a notebook can load once (the
    slow part — opening the regime-label and full-resolution anomaly NetCDFs)
    and then call :func:`build_z500_composite_figure` repeatedly while tuning
    figure styling, instead of re-reading everything from disk on every tweak.

    `mode`: "bbox" or "regime" to force which pipeline's RR regimes file to use
    (rr_02 vs rr_022); None auto-detects (see `_find_rr_regimes`).

    `z500_cluster_method`: "ward" (default; pure Ward HRC, matches the Z500-only pipeline's
    default and ``regime_alignment_<region>.yaml``) or "kmeans" (also covers the Ward-seeded
    K-means hybrid) — which Z500 parent-regime source both the RR regimes file and the Z500
    regimes file must have been built from, so the Z500-regime x RR-subregime pairing shown in
    the composites is consistent.
    """
    dw = _dw_root(module_root)

    rr_path = _find_rr_regimes(dw, region, bbox_step, module_root=module_root, mode=mode,
                                z500_cluster_method=z500_cluster_method)
    if rr_path is None:
        raise FileNotFoundError(f"No RR regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})")
    z500_path = _find_z500_regimes(dw, region, bbox_step, smooth_window=smooth_window,
                                    cluster_method=z500_cluster_method)
    if z500_path is None:
        raise FileNotFoundError(f"No Z500 regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})")
    print(f"[rr_composites] RR regimes:   {rr_path.name}")
    print(f"[rr_composites] Z500 regimes: {z500_path.name}")

    rr_labels, rr_times = _load_labels(rr_path)
    rr_labels = rr_labels.astype(int)
    rr_dates = _norm_dates(rr_times)
    n_rr = int(rr_labels.max())

    z500_labels_raw, z500_times = _load_labels(z500_path)
    z500_series = pd.Series(z500_labels_raw.astype(int), index=_norm_dates(z500_times))
    z500_on_rr = z500_series.reindex(rr_dates).values
    z500_vals_raw = {int(v) for v in z500_on_rr if np.isfinite(v) and v > 0}
    if z500_cluster_method == "ward":
        _, _groups = _get_regime_alignment(region)
        z500_vals = regime_alignment.canonical_order(z500_vals_raw, _groups, bbox_step)
        z500_canonical_map = regime_alignment.relabel_ids(z500_vals_raw, _groups, bbox_step)
        print(f"[rr_composites] {n_rr} RR regimes, Z500 regimes: {z500_vals} "
              f"(canonical ids: {[z500_canonical_map[z] for z in z500_vals]}), {len(rr_dates)} days")
    else:
        z500_vals = sorted(z500_vals_raw)
        z500_canonical_map = {}
        print(f"[rr_composites] {n_rr} RR regimes, Z500 regimes: {z500_vals} "
              f"(cluster_method=kmeans -- regime_alignment_<region>.yaml is Ward-only, no canonical "
              f"ordering/labelling), {len(rr_dates)} days")

    with xr.open_dataset(rr_path) as ds_rr:
        per_regime_fields_attr = ds_rr.attrs.get("per_regime_fields")

    per_z500_fields: dict[int, list[str]] | None = None
    if per_regime_fields_attr:
        per_z500_fields = {int(k): v for k, v in json.loads(per_regime_fields_attr).items()}
        all_fields = sorted({f for fs in per_z500_fields.values() for f in fs} | {"z500"})
    else:
        m = re.search(r"z500([^_]*)_w\d+", rr_path.name)
        all_fields = ["z500"] + ([f for f in m.group(1).split("+") if f] if (m and m.group(1)) else [])

    common = _load_field_geo_epe_inputs(module_root, dw, region, bbox_step, smooth_window, all_fields,
                                         area_weight_tag=area_weight_tag)

    return dict(
        dw=dw, region=region, bbox_step=bbox_step, module_root=module_root,
        rr_dates=rr_dates, rr_labels=rr_labels, n_rr=n_rr,
        z500_on_rr=z500_on_rr, z500_vals=z500_vals, z500_canonical_map=z500_canonical_map,
        per_z500_fields=per_z500_fields,
        **common,
    )


def load_composite_inputs_logistic(
    module_root, region, bbox_step, smooth_window=11, n_bins=2, output_tag="",
    z500_cluster_method: str = "ward", area_weight_tag: str = "",
    bin_overrides_path: Path | None = None,
) -> dict:
    """Stage-2 [LOGISTIC] counterpart of ``load_composite_inputs``: resolves the
    merged continuous ``predicted_extreme_prob`` index (see stage2_logistic.py)
    instead of a discrete k-means RR regimes file, and quantile-bins each Z500
    regime's own index into ``n_bins`` groups at load time — never written to
    disk (see ``stage2_logistic.bin_predicted_prob_per_regime`` and
    README_ResidualRegimes.md TODO "Build the composite-from-probability-bins
    plotting path"). Bin ids increase with probability (bin 1 = lowest,
    bin ``n_bins`` = highest); with the current production default
    ``n_bins=2`` that is a within-regime "non-extreme-like" vs "extreme-like"
    split.

    Returns the same ``fields_data``/``bbox_dict``/EPE keys as
    ``load_composite_inputs`` (see ``_load_field_geo_epe_inputs``) plus
    ``prob_bin``/``z500_regime``/``n_bins`` in place of ``rr_labels``/``n_rr``,
    so :func:`build_z500_composite_figure_logistic` can hand
    ``_page_z500_composites`` the identical ``(rr_present,
    group_dates_by_rr)`` shape the discrete path uses — the renderer itself
    needs no changes.

    ``bin_overrides_path``, if given, points at a
    ``select_subregime_bins.py``-written YAML: regimes it covers for this
    ``bbox_step`` get their own data-driven probability edges (see
    ``stage2_logistic.select_regime_bin_edges``) instead of the uniform
    ``n_bins`` quantile split — different regimes can then end up with a
    different number of bins. The returned ``n_bins`` becomes the actual max
    bin id across regimes (for the renderer's ``range(1, n_bins + 1)`` loop),
    not the input ``n_bins``, when overrides are in effect.
    """
    from .plot_results import _find_rr_logistic_merged
    from .stage2_logistic import bin_predicted_prob_per_regime, load_bin_overrides_yaml

    dw = _dw_root(module_root)

    rr_path = _find_rr_logistic_merged(dw, region, bbox_step, output_tag=output_tag,
                                        z500_cluster_method=z500_cluster_method)
    if rr_path is None:
        raise FileNotFoundError(
            f"No merged Stage-2 [LOGISTIC] index file for {region} {bbox_step} "
            f"(z500_cluster_method={z500_cluster_method}) — run stage2_logistic.py first."
        )
    print(f"[rr_composites] RR logistic index: {rr_path.name}")

    with xr.open_dataset(rr_path) as ds:
        time_dim = ds["predicted_extreme_prob"].dims[0]
        rr_times = pd.DatetimeIndex(ds[time_dim].values)
        predicted_prob = np.asarray(ds["predicted_extreme_prob"].values, dtype=np.float64)
        z500_regime = np.asarray(ds["z500_regime"].values, dtype=int)
        per_regime_fields_attr = ds.attrs.get("per_regime_fields")

    rr_dates = _norm_dates(rr_times)
    z500_vals_raw = {int(v) for v in z500_regime}
    if z500_cluster_method == "ward":
        _, _groups = _get_regime_alignment(region)
        z500_vals = regime_alignment.canonical_order(z500_vals_raw, _groups, bbox_step)
        z500_canonical_map = regime_alignment.relabel_ids(z500_vals_raw, _groups, bbox_step)
        print(f"[rr_composites] Z500 regimes: {z500_vals} "
              f"(canonical ids: {[z500_canonical_map[z] for z in z500_vals]}), {len(rr_dates)} days, n_bins={n_bins}")
    else:
        z500_vals = sorted(z500_vals_raw)
        z500_canonical_map = {}
        print(f"[rr_composites] Z500 regimes: {z500_vals} "
              f"(cluster_method=kmeans -- regime_alignment_<region>.yaml is Ward-only, no canonical "
              f"ordering/labelling), {len(rr_dates)} days, n_bins={n_bins}")

    bin_edges_by_regime = {}
    if bin_overrides_path is not None:
        bin_edges_by_regime = load_bin_overrides_yaml(bin_overrides_path, bbox_step)
        print(f"[rr_composites] Bin overrides: {bin_overrides_path.name} -> "
              f"{ {z: len(e) + 1 for z, e in bin_edges_by_regime.items()} } bins by regime "
              f"(others kept as a single undivided group -- see stage2_logistic."
              f"bin_predicted_prob_per_regime)")

    prob_bin = bin_predicted_prob_per_regime(
        predicted_prob, z500_regime, n_bins=n_bins, bin_edges_by_regime=bin_edges_by_regime,
    )
    n_bins_effective = int(prob_bin.max()) if len(prob_bin) else int(n_bins)

    per_z500_fields: dict[int, list[str]] | None = None
    if per_regime_fields_attr:
        per_z500_fields = {int(k): v for k, v in json.loads(per_regime_fields_attr).items()}
        all_fields = sorted({f for fs in per_z500_fields.values() for f in fs} | {"z500"})
    else:
        all_fields = ["z500"]

    common = _load_field_geo_epe_inputs(module_root, dw, region, bbox_step, smooth_window, all_fields,
                                         area_weight_tag=area_weight_tag)

    return dict(
        dw=dw, region=region, bbox_step=bbox_step, module_root=module_root,
        rr_dates=rr_dates, z500_regime=z500_regime, prob_bin=prob_bin, n_bins=n_bins_effective,
        z500_vals=z500_vals, z500_canonical_map=z500_canonical_map,
        per_z500_fields=per_z500_fields,
        **common,
    )


def _restrict_to_extreme_days(
    group_dates_by_rr: dict, extreme_dates: "pd.DatetimeIndex"
) -> tuple[list, dict]:
    """Intersect each group's dates with ``extreme_dates`` and re-apply ``MIN_DAYS``.

    Shared by the discrete and logistic composite builders' ``extreme_only`` path — a group can
    only lose days here (extreme days are a subset of all days), never gain, so re-checking
    ``MIN_DAYS`` after intersecting can only drop groups the all-days page already showed.
    """
    present, restricted = [], {}
    for key, dates in group_dates_by_rr.items():
        d = dates[dates.isin(extreme_dates)]
        if len(d) >= MIN_DAYS:
            present.append(key)
            restricted[key] = d
    return present, restricted


def build_z500_composite_figure(
    inputs: dict, z: int, extreme_only: bool = False,
    z500_vmin=None, z500_vmax=None, epe_vmax_fixed=None,
) -> "plt.Figure | None":
    """Build (without saving) the composite page for one Z500 regime value.

    ``inputs`` is the dict returned by :func:`load_composite_inputs`. Returns
    ``None`` if no RR sub-regime within Z500-``z`` has >= ``MIN_DAYS`` days —
    the same skip condition ``run_composite_plots`` applies per page.

    ``extreme_only=True`` restricts each sub-regime's day group to the days
    ``inputs["extreme_dates"]`` flags as extreme (see
    :func:`_load_extreme_day_mask`) before compositing — everything else
    (column 0's parent-regime-mean reference, colour scales, EPE panel)
    behaves identically. Returns ``None`` if ``extreme_dates`` is unavailable
    for this run.
    """
    rr_dates, rr_labels, n_rr = inputs["rr_dates"], inputs["rr_labels"], inputs["n_rr"]
    z_mask = (inputs["z500_on_rr"] == z)
    z_all_dates = rr_dates[z_mask]
    rr_present, group_dates_by_rr = [], {}
    for rr in range(1, n_rr + 1):
        g = z_mask & (rr_labels == rr)
        if int(g.sum()) >= MIN_DAYS:
            rr_present.append(rr)
            group_dates_by_rr[rr] = rr_dates[g]
    if not rr_present:
        print(f"[rr_composites] Z500-{z}: no RR sub-regime with >= {MIN_DAYS} days — skipping")
        return None

    variant_label = "all days"
    if extreme_only:
        extreme_dates = inputs.get("extreme_dates")
        if extreme_dates is None:
            return None
        rr_present, group_dates_by_rr = _restrict_to_extreme_days(group_dates_by_rr, extreme_dates)
        if not rr_present:
            print(f"[rr_composites] Z500-{z}: no RR sub-regime with >= {MIN_DAYS} extreme days — skipping")
            return None
        variant_label = "extreme days only"

    print(f"[rr_composites] Z500-{z}: sub-regimes {rr_present} ({variant_label})")
    return _page_z500_composites(
        z, z_all_dates, rr_present, group_dates_by_rr,
        inputs["fields_data"], inputs["region"], inputs["bbox_step"], inputs["bbox_dict"], inputs["module_root"],
        econv=inputs["econv"], edates=inputs["edates"], elats=inputs["elats"], elons=inputs["elons"],
        z500_raw_fd=inputs["z500_raw_fd"], per_z500_fields=inputs["per_z500_fields"],
        variant_label=variant_label,
        z500_vmin=z500_vmin, z500_vmax=z500_vmax, epe_vmax_fixed=epe_vmax_fixed,
    )


def build_z500_composite_figure_logistic(
    inputs: dict, z: int, extreme_only: bool = False,
    z500_vmin=None, z500_vmax=None, epe_vmax_fixed=None,
) -> "plt.Figure | None":
    """Stage-2 [LOGISTIC] counterpart of :func:`build_z500_composite_figure`:
    groups this Z500 regime's days by ``prob_bin`` (from
    :func:`load_composite_inputs_logistic`) instead of a discrete RR cluster
    label, then hands the result to the same ``_page_z500_composites``
    renderer unchanged. Row labels are local per-page bin ids (1 = lowest
    ``predicted_extreme_prob``, ``n_bins`` = highest), consistent across every
    Z500 page rather than a globally offset regime id — there is no cross-
    regime bin identity to preserve here.

    ``inputs`` is the dict returned by :func:`load_composite_inputs_logistic`.
    Returns ``None`` if no probability bin within Z500-``z`` has >= ``MIN_DAYS``
    days — the same skip condition ``build_z500_composite_figure`` applies.

    ``extreme_only=True`` behaves like :func:`build_z500_composite_figure`'s —
    restricts each bin's days to ``inputs["extreme_dates"]`` before compositing.
    """
    rr_dates, z500_regime, prob_bin, n_bins = (
        inputs["rr_dates"], inputs["z500_regime"], inputs["prob_bin"], inputs["n_bins"],
    )
    z_mask = (z500_regime == z)
    z_all_dates = rr_dates[z_mask]
    bins_for_z = prob_bin[z_mask]
    rr_present, group_dates_by_rr = [], {}
    for b in range(1, n_bins + 1):
        g = bins_for_z == b
        if int(g.sum()) >= MIN_DAYS:
            rr_present.append(b)
            group_dates_by_rr[b] = z_all_dates[g]
    if not rr_present:
        print(f"[rr_composites] Z500-{z}: no probability bin with >= {MIN_DAYS} days — skipping")
        return None

    variant_label = "all days"
    if extreme_only:
        extreme_dates = inputs.get("extreme_dates")
        if extreme_dates is None:
            return None
        rr_present, group_dates_by_rr = _restrict_to_extreme_days(group_dates_by_rr, extreme_dates)
        if not rr_present:
            print(f"[rr_composites] Z500-{z}: no probability bin with >= {MIN_DAYS} extreme days — skipping")
            return None
        variant_label = "extreme days only"

    print(f"[rr_composites] Z500-{z}: probability bins {rr_present} (of {n_bins}) ({variant_label})")
    return _page_z500_composites(
        z, z_all_dates, rr_present, group_dates_by_rr,
        inputs["fields_data"], inputs["region"], inputs["bbox_step"], inputs["bbox_dict"], inputs["module_root"],
        econv=inputs["econv"], edates=inputs["edates"], elats=inputs["elats"], elons=inputs["elons"],
        z500_raw_fd=inputs["z500_raw_fd"], per_z500_fields=inputs["per_z500_fields"],
        variant_label=variant_label,
        z500_vmin=z500_vmin, z500_vmax=z500_vmax, epe_vmax_fixed=epe_vmax_fixed,
    )


def _collect_scale_pool_arrays(inputs: dict, is_logistic: bool) -> dict[str, np.ndarray]:
    """Flat (unfigured) Z500 mean-anomaly (gpm) + EPE flag-frequency arrays pooled across every
    (Z500 regime x RR sub-regime/bin) group for this region+bbox_step, "all days" variant only --
    same group-membership logic ``build_z500_composite_figure[_logistic]`` use, minus the figure.
    Used by ``--composite-out`` (see ``scripts/compute_fixed_composite_scales.py``), which pools
    these across every region/bbox_step call to derive one global fixed color scale.
    """
    z500_vals = list(inputs["z500_vals"] if not is_logistic else np.unique(inputs["z500_regime"]))
    fields_data = inputs["fields_data"]
    has_z500 = "z500" in fields_data
    have_epe = inputs.get("econv") is not None

    z500_unit_scale = 1.0
    if has_z500:
        fdz = fields_data["z500"]
        z500_unit_scale = (
            1.0 / G0 if z500_anomaly_units_look_physical(xr.DataArray(fdz.data), fdz.units) else 1.0
        )

    z500_vals_out: list[np.ndarray] = []
    epe_vals_out: list[np.ndarray] = []
    epe_highrisk_vals_out: list[np.ndarray] = []

    for z in z500_vals:
        if is_logistic:
            rr_dates, z500_regime, prob_bin, n_bins = (
                inputs["rr_dates"], inputs["z500_regime"], inputs["prob_bin"], inputs["n_bins"],
            )
            z_mask = z500_regime == z
            z_all_dates = rr_dates[z_mask]
            bins_for_z = prob_bin[z_mask]
            groups = {}
            for b in range(1, n_bins + 1):
                g = bins_for_z == b
                if int(g.sum()) >= MIN_DAYS:
                    groups[b] = z_all_dates[g]
        else:
            rr_dates, rr_labels, n_rr = inputs["rr_dates"], inputs["rr_labels"], inputs["n_rr"]
            z_mask = inputs["z500_on_rr"] == z
            groups = {}
            for rr in range(1, n_rr + 1):
                g = z_mask & (rr_labels == rr)
                if int(g.sum()) >= MIN_DAYS:
                    groups[rr] = rr_dates[g]
        if not groups:
            continue
        # This Z500 regime's own highest bin id -- the "high-risk bin" panel every
        # build_z500_extreme_summary_figure/_page_z500_composites row-set draws (see
        # EPE_FLAG_FREQ_HIGHRISK's docstring).
        highrisk_bin = max(groups)

        if has_z500:
            fdz = fields_data["z500"]
            for dates in groups.values():
                c = _composite_over_dates(fdz, dates)
                if c is not None:
                    z500_vals_out.append((c * z500_unit_scale).ravel())
        if have_epe:
            econv, edates = inputs["econv"], inputs["edates"]
            for b, dates in groups.items():
                f, _n = _epe_frequency_over_dates(econv, edates, dates)
                if f is None:
                    continue
                # x100: _epe_frequency_over_dates returns a 0-1 fraction, but the shared
                # fixed_color_scales.yaml (and the Z500-only pipeline's own dump) use
                # PERCENT (0-100) -- see _page_z500_composites's epe_vmax_fixed docstring.
                vals = 100.0 * np.asarray(f, dtype=float).ravel()
                epe_vals_out.append(vals)
                if b == highrisk_bin:
                    epe_highrisk_vals_out.append(vals)

    out: dict[str, np.ndarray] = {}
    if z500_vals_out:
        out["z500_anom_gpm"] = np.concatenate(z500_vals_out)
    if epe_vals_out:
        out["epe_flag_freq"] = np.concatenate(epe_vals_out)
    if epe_highrisk_vals_out:
        out["epe_flag_freq_highrisk"] = np.concatenate(epe_highrisk_vals_out)
    return out


def run_composite_plots(
    module_root, region, bbox_step, smooth_window=11, out_dir=None, mode=None,
    formats: tuple[str, ...] = ("png",),
    z500_cluster_method: str = "ward",
    n_bins: int | None = None,
    output_tag: str = "",
    area_weight_tag: str = "",
    bin_overrides_path: Path | None = None,
    z500_vmin=None, z500_vmax=None, epe_vmax_fixed=None,
    skip_extreme_only: bool = True,
) -> list[Path]:
    """Build one Figure per Z500 parent regime and save each in every requested format.

    ``formats`` may contain "png" (one file per Z500 regime, ``rr_composites_<region>_<bbox>_Z<z>.png``
    — no notebook/kernel needed, matches the per-regime PNGs the Z500-only composites script writes;
    ``<z>`` is the canonical (physical) regime id from ``regime_alignment_<region>.yaml`` when
    ``z500_cluster_method="ward"``, so the same physical regime keeps the same filename across bbox
    steps/reruns, falling back to the raw per-bbox-step cluster id for "kmeans")
    and/or "pdf" (single multi-page ``rr_composites_<region>_<bbox>.pdf``, one page per regime, the
    original rr_03 behavior).

    Writes two variants of every page/file when a day-level extreme-day mask is available (see
    :func:`_load_extreme_day_mask` — resolved via ``EPE_FILE``/``find_duration_file``, so it
    reflects whatever extreme-day definition this run is configured with): the usual "all days
    in each sub-regime/bin group" composites (unchanged filenames) plus an "``_extremeonly``"-
    tagged companion restricted to each group's extreme days only. The extreme-only variant is
    skipped (with a log message) if no duration file / day-level extreme flag was found, or if
    ``skip_extreme_only=True`` (its per-cell flag frequencies are unreliable — small per-cell
    extreme-day sample sizes, especially for northamerica's Z1/Z4 regimes, can produce
    implausible >30% flag frequencies from a handful of days).

    ``z500_cluster_method``: which Z500 parent-regime source (and hence which RR sub-regime
    file) to plot — must match how rr_02/rr_022 built the RR regimes being visualized.

    ``mode="logistic"`` plots the Stage-2 [LOGISTIC] continuous
    ``predicted_extreme_prob`` index instead of a discrete RR regimes file —
    quantile-binned into ``n_bins`` groups per Z500 regime at plot time (see
    ``load_composite_inputs_logistic``); ``n_bins`` defaults to 2
    ("non-extreme-like" vs "extreme-like") when unset. ``output_tag`` selects
    among several logistic runs that differ only by extreme-label
    definition (e.g. a domain_pctl_q sweep) and is ignored otherwise.
    ``bin_overrides_path``, if given, points at a ``select_subregime_bins.py``
    YAML of per-regime data-driven probability edges — overrides the uniform
    ``n_bins`` split for regimes it covers (see
    ``load_composite_inputs_logistic``); output filenames get an
    ``adaptive`` tag instead of a single bin count in that case, since the
    per-regime bin count can differ.
    """
    is_logistic = mode == "logistic"
    if is_logistic:
        n_bins = n_bins or 2
        inputs = load_composite_inputs_logistic(
            module_root, region, bbox_step, smooth_window=smooth_window, n_bins=n_bins,
            output_tag=output_tag, z500_cluster_method=z500_cluster_method,
            area_weight_tag=area_weight_tag, bin_overrides_path=bin_overrides_path,
        )
    else:
        inputs = load_composite_inputs(module_root, region, bbox_step, smooth_window=smooth_window, mode=mode,
                                        z500_cluster_method=z500_cluster_method, area_weight_tag=area_weight_tag)

    out_dir = out_dir or (inputs["dw"] / "results" / "stage2_residual" / "figures" / region)
    out_dir.mkdir(parents=True, exist_ok=True)
    ward_tag = "_zward" if z500_cluster_method == "ward" else ""
    if is_logistic:
        logistic_tag = "_logisticadaptivebin" if bin_overrides_path is not None else f"_logistic{n_bins}bin"
    else:
        logistic_tag = ""

    build_fig = build_z500_composite_figure_logistic if is_logistic else build_z500_composite_figure

    have_extreme_dates = inputs.get("extreme_dates") is not None
    if skip_extreme_only:
        print("[rr_composites] --skip-extreme-only set — skipping the extreme-only composite variant")
    elif not have_extreme_dates:
        print("[rr_composites] no extreme-day mask available for this run — "
              "skipping the extreme-only composite variant")

    written: list[Path] = []
    # (file-tag suffix, extreme_only flag) — always the all-days variant; the extreme-only
    # variant is skipped outright if no extreme-day mask was resolved for this run, or if
    # skip_extreme_only was requested (its per-cell flag frequencies are unreliable — see
    # --skip-extreme-only help).
    variants = [("", False)] + (
        [("_extremeonly", True)] if (have_extreme_dates and not skip_extreme_only) else []
    )

    for tag, extreme_only in variants:
        out_pdf = out_dir / f"rr_composites_{region}_{bbox_step}{ward_tag}{logistic_tag}{tag}.pdf"
        pdf_ctx = PdfPages(out_pdf) if "pdf" in formats else None
        wrote_any = False
        try:
            for z in inputs["z500_vals"]:
                fig = build_fig(
                    inputs, z, extreme_only=extreme_only,
                    z500_vmin=z500_vmin, z500_vmax=z500_vmax, epe_vmax_fixed=epe_vmax_fixed,
                )
                if fig is None:
                    continue
                wrote_any = True
                if "png" in formats:
                    # Filename carries both the canonical (physical) regime id when available
                    # (ward; see regime_alignment_<region>.yaml) -- so the same physical regime
                    # gets the same "_Z<n>" tag across bbox steps/reruns -- AND the raw
                    # per-bbox-step cluster id actually used to build this page's groups, as
                    # "_R<n>", so the two numbering schemes are both recoverable from the name.
                    z_file_id = inputs.get("z500_canonical_map", {}).get(z, z)
                    out_png = out_dir / (
                        f"rr_composites_{region}_{bbox_step}{ward_tag}{logistic_tag}{tag}_Z{z_file_id}_R{z}.png"
                    )
                    fig.savefig(out_png, bbox_inches="tight", dpi=200)
                    written.append(out_png)
                    print(f"[rr_composites] wrote {out_png}")
                if pdf_ctx is not None:
                    pdf_ctx.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        finally:
            if pdf_ctx is not None:
                pdf_ctx.close()

        if pdf_ctx is not None and wrote_any:
            written.append(out_pdf)
            print(f"[rr_composites] Saved: {out_pdf}")
        elif pdf_ctx is not None:
            out_pdf.unlink(missing_ok=True)

    # Extreme-summary page: one row per Z500 regime, its own top bin only, all on one page (see
    # build_z500_extreme_summary_figure). Only meaningful for the continuous logistic index --
    # "top bin" has no equivalent for a discrete k-means RR regime label.
    if is_logistic:
        fig = build_z500_extreme_summary_figure(inputs)
        if fig is None:
            print("[rr_composites] extreme-summary: no Z500 regime had a bin with enough days — skipping")
        else:
            stem = f"rr_composites_{region}_{bbox_step}{ward_tag}{logistic_tag}_extremesummary"
            if "png" in formats:
                out_png = out_dir / f"{stem}.png"
                fig.savefig(out_png, bbox_inches="tight", dpi=200)
                written.append(out_png)
                print(f"[rr_composites] wrote {out_png}")
            if "pdf" in formats:
                out_pdf = out_dir / f"{stem}.pdf"
                with PdfPages(out_pdf) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")
                written.append(out_pdf)
                print(f"[rr_composites] Saved: {out_pdf}")
            plt.close(fig)

    return written


def main(argv=None) -> int:
    import sys
    p = argparse.ArgumentParser(description="Per-(Z500×RR) additive composite maps for residual regimes.")
    p.add_argument("--region", required=True, choices=("europe", "northamerica"))
    p.add_argument("--bbox-step", required=True)
    p.add_argument("--module-root", type=Path, default=None)
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--mode", choices=("bbox", "regime", "logistic"), default=None,
                   help="Force which pipeline's RR regimes to plot: rr_02 bbox-mode, rr_022 "
                        "per-regime-mode discrete k-means ('regime'), or the Stage-2 [LOGISTIC] "
                        "continuous predicted_extreme_prob index quantile-binned at plot time "
                        "('logistic' — see --n-bins). Default: auto-detect between bbox/regime "
                        "(prefers bbox-mode); never auto-selects 'logistic'.")
    p.add_argument(
        "--formats", default="png",
        help="Comma-separated output formats: png (default, one file per Z500 regime) and/or "
        "pdf (single multi-page file, one page per regime).",
    )
    p.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="ward",
                   help="Which Z500 parent-regime source to plot RR sub-regimes against "
                        "(default 'ward', reads '_zward'-tagged RR/Z500 files, matches "
                        "regime_alignment_<region>.yaml). Must match how the RR regimes being "
                        "visualized were built.")
    p.add_argument("--n-bins", type=int, default=None,
                   help="--mode logistic only: quantile bins per Z500 regime for the continuous "
                        "predicted_extreme_prob index (default: 2, 'non-extreme-like' vs "
                        "'extreme-like'). Ignored otherwise.")
    p.add_argument("--output-tag", default="",
                   help="--mode logistic only: select among several logistic runs that differ "
                        "only by extreme-label definition (matches stage2_logistic.py's "
                        "--output-tag). Ignored otherwise.")
    p.add_argument("--bin-overrides-path", type=Path, default=None,
                   help="--mode logistic only: YAML written by select_subregime_bins.py giving "
                        "per-Z500-regime data-driven probability edges, overriding --n-bins for "
                        "the regimes it covers (default: src/common/"
                        "bin_overrides_<region>_custom.yaml, if it exists). Ignored otherwise.")
    p.add_argument("--area-weight-tag", default="",
                   help="Must match the rr_00 run's --area-weight-tag (e.g. 'aw2') to resolve "
                        "that rerun's per-field anomaly files exactly, not a fuzzy glob (see "
                        "plot_results._find_anomaly) — has no effect on Z500, which lives "
                        "outside the rr_00 area-weighting rerun.")
    p.add_argument("--z500-vmin", type=float, default=None, help="Override Z500 mean-anomaly fill vmin (gpm).")
    p.add_argument("--z500-vmax", type=float, default=None, help="Override Z500 mean-anomaly fill vmax (gpm).")
    p.add_argument(
        "--epe-vmax", type=float, default=None,
        help="Override EPE flag-frequency fill vmax, in PERCENT (0-100, e.g. 2.0 = 2%%) -- "
             "matches the Z500-only pipeline's --precip-vmax convention. vmin is always 0.",
    )
    p.add_argument(
        "--fixed-scales-yaml", type=Path, default=os.environ.get("FIXED_SCALES_YAML") or None,
        help=(
            "src/common/fixed_color_scales.yaml (or equivalent) supplying default "
            "z500/epe vmin/vmax when the flags above aren't passed explicitly. Auto-discovered "
            "at the default repo path if unset; missing file falls back to today's per-call "
            "auto-scaling. See scripts/compute_fixed_composite_scales.py."
        ),
    )
    p.add_argument(
        "--composite-out", type=Path, default=None,
        help=(
            "Persist pooled composites (z500_anom_gpm, epe_flag_freq) as a small NetCDF here. "
            "Enables a separate orchestrator to compute global vmin/vmax across bboxes/regions "
            "-- see scripts/compute_fixed_composite_scales.py."
        ),
    )
    p.add_argument(
        "--no-render", action="store_true",
        help="Skip PNG/PDF rendering (compute composites only). Use with --composite-out.",
    )
    p.add_argument(
        "--include-extreme-only", action="store_true",
        help="Also write the '_extremeonly' composite variant (restricted to each group's "
             "extreme days), off by default. Per-cell EPE flag-frequency in that variant is "
             "computed from very few days per (Z500 x RR) sub-regime x bbox cell, so individual "
             "grid cells can show implausibly high frequencies (e.g. >30%%) from small-sample "
             "noise -- see the all-days variant's flag-frequency panel for a more reliable "
             "per-cell estimate, or pool across a larger STEPS/bbox aggregation before trusting "
             "per-cell extreme-only percentages.",
    )
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.module_root is None:
        args.module_root = Path(__file__).resolve().parents[2]
    module_root = args.module_root.expanduser().resolve()

    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())

    bin_overrides_path = args.bin_overrides_path
    if args.mode == "logistic" and bin_overrides_path is None:
        default_overrides = module_root / "src" / "common" / f"bin_overrides_{args.region}_custom.yaml"
        if default_overrides.is_file():
            print(f"[rr_composites] Using default bin overrides: {default_overrides}")
            bin_overrides_path = default_overrides

    scales_path = args.fixed_scales_yaml or default_fixed_scales_path(module_root)
    scales = load_fixed_color_scales(scales_path, args.region)
    z500_vmin, z500_vmax = args.z500_vmin, args.z500_vmax
    if Z500_MEAN_ANOM_GPM in scales:
        _fixed_vmin, _fixed_vmax = scales[Z500_MEAN_ANOM_GPM]
        z500_vmin = _fixed_vmin if z500_vmin is None else z500_vmin
        z500_vmax = _fixed_vmax if z500_vmax is None else z500_vmax
    epe_vmax_fixed = args.epe_vmax
    if epe_vmax_fixed is None and EPE_FLAG_FREQ_RR in scales:
        # _page_z500_composites applies epe_vmax_fixed as ONE blanket vmax shared across every
        # row/bin on the page (low-risk AND high-risk alike) -- EPE_FLAG_FREQ_RR is the field
        # pooled from all bins (per-region, see compute_fixed_composite_scales.py) and is the
        # right scale for that. EPE_FLAG_FREQ_HIGHRISK below is pooled from only each regime's
        # single hottest bin, so using it here would stretch every row (not just the high-risk
        # one) to that bin's much larger range, washing out every panel's real signal -- it has
        # no consumer here today; a future per-row (not blanket-page) vmax could use it for just
        # the high-risk row specifically.
        epe_vmax_fixed = scales[EPE_FLAG_FREQ_RR][1]
    elif epe_vmax_fixed is None and EPE_FLAG_FREQ_HIGHRISK in scales:
        epe_vmax_fixed = scales[EPE_FLAG_FREQ_HIGHRISK][1]

    if args.composite_out is not None:
        is_logistic = args.mode == "logistic"
        if is_logistic:
            inputs = load_composite_inputs_logistic(
                module_root, args.region, args.bbox_step, smooth_window=args.smooth_window,
                n_bins=args.n_bins or 2, output_tag=args.output_tag,
                z500_cluster_method=args.z500_cluster_method, area_weight_tag=args.area_weight_tag,
                bin_overrides_path=bin_overrides_path,
            )
        else:
            inputs = load_composite_inputs(
                module_root, args.region, args.bbox_step, smooth_window=args.smooth_window,
                mode=args.mode, z500_cluster_method=args.z500_cluster_method,
                area_weight_tag=args.area_weight_tag,
            )
        pooled = _collect_scale_pool_arrays(inputs, is_logistic)
        comp_out = args.composite_out.expanduser().resolve()
        comp_out.parent.mkdir(parents=True, exist_ok=True)
        # Each pooled array gets its OWN sample dimension (not a shared "sample") -- z500_anom_gpm
        # is pooled off the Z500 grid while epe_flag_freq[_highrisk] are pooled off the (differently
        # shaped) EPE/precip grid, so their raveled lengths never match; a shared dim name made
        # xr.Dataset reject the whole write whenever both were present (see git history for the
        # ValueError this used to raise).
        ds_out = xr.Dataset({name: ((f"sample_{name}",), vals) for name, vals in pooled.items()})
        ds_out.attrs.update({"region": args.region, "bbox_step": args.bbox_step, "mode": args.mode or ""})
        ds_out.to_netcdf(comp_out)
        print(f"[rr_composites] wrote composites NetCDF {comp_out}")

    if args.no_render:
        print("[rr_composites] --no-render set: skipping figure output")
        return 0

    run_composite_plots(
        module_root=module_root,
        region=args.region, bbox_step=args.bbox_step,
        smooth_window=args.smooth_window, out_dir=args.out_dir,
        mode=args.mode, formats=formats,
        z500_cluster_method=args.z500_cluster_method,
        n_bins=args.n_bins, output_tag=args.output_tag,
        area_weight_tag=args.area_weight_tag,
        bin_overrides_path=bin_overrides_path,
        z500_vmin=z500_vmin, z500_vmax=z500_vmax, epe_vmax_fixed=epe_vmax_fixed,
        skip_extreme_only=not args.include_extreme_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
