#!/usr/bin/env python3
"""
Overview figure: all Z500-regime maps + their RR sub-regime RGB composites
on a single page per bbox.

  Row   = one Z500 parent regime.
  Col 0 = mean Z500 anomaly map for that regime (RdBu_r, shared scale), overlaid
          with labelled raw-Z500 isolines averaged over ALL days affected by
          that Z500 regime.
  Col r = additive-RGB composite for RR sub-regime r  (r = 1 … max_rr).
          Grey panel labelled "< MIN_DAYS" where the group has too few days.

All RGB panels share ONE global per-field normalisation scale (95th-percentile
of |composite values| pooled across every (Z500 × RR) group), so colours are
directly comparable across the entire figure. Each RGB panel is overlaid with
labelled black isolines of raw Z500 (gpm) averaged over ONLY that panel's own
(Z500 × RR) sub-regime days — so the contour differs across every RR column in
a row, not one contour shared across the whole row (falls back to the shared
parent-regime anomaly contour only if raw Z500 isn't available at all).

Saved as  rr_rgb_overview_{region}_{bbox}.png and/or .pdf (see --formats)
under     results/residual_regimes/figures/{region}/ by default, or --out-dir.

Usage:
    python -m stage2_residual.plot_rgb_overview \
        --region europe --bbox-step B5 --module-root /path/to/detect_weather
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

from common.colorbar_discretize import discretize_diverging, style_discrete_ticks
from .plot_results import _dw_root, _find_anomaly, _find_raw_z500, _find_rr_regimes, _find_z500_regimes, _load_labels
from .plot_subregimes import (
    MIN_DAYS,
    FieldData,
    _composite_over_dates,
    _crop_field_to_bbox,
    _load_field,
    _load_raw_z500_field,
    _norm_dates,
    _overlay_z500_isolines,
    _z500_raw_contour_levels,
)
from . import rr_geo
import common.regime_alignment as regime_alignment

_ALIGNMENT_YAML_DIR = Path(__file__).resolve().parents[2] / "src" / "common"
_ALIGNMENT_CACHE: dict[str, tuple[str, list[dict]]] = {}


def _get_regime_alignment(region: str) -> tuple[str, list[dict]]:
    """Cached ``regime_alignment.load_regime_alignment`` for ``region`` -- same yaml/module as
    ``plot_composites.py``/``plot_rr_statistics.py``/``2_3_Results_Z500_Stats.ipynb``, so rows here
    are ordered/labelled by the same canonical (physical) regime id everywhere.
    """
    if region not in _ALIGNMENT_CACHE:
        path = _ALIGNMENT_YAML_DIR / f"regime_alignment_{region}.yaml"
        _ALIGNMENT_CACHE[region] = regime_alignment.load_regime_alignment(path)
    return _ALIGNMENT_CACHE[region]

_CHANNELS = ("R", "G", "B")
_CHANNEL_RGB = {"R": (0.85, 0.10, 0.10), "G": (0.10, 0.70, 0.10), "B": (0.10, 0.30, 0.90)}

_PANEL_SIZE = 3.2   # inches per panel (width and height)
_LEGEND_PAD = 3.0   # extra inches reserved on the right for the ternary legend
_EMPTY_COLOR = "#e8e8e8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pos_norm(comp: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(comp / scale, 0.0, 1.0)


def _additive_over_white(rgb: np.ndarray) -> np.ndarray:
    m = rgb.max(axis=-1, keepdims=True)
    return np.clip(rgb + (1.0 - m), 0.0, 1.0)


def _global_field_scale(
    impact_fields: list[str],
    fields_data: dict[str, FieldData],
    all_group_dates: list[pd.DatetimeIndex],
) -> dict[str, float]:
    """95th percentile of |composite values| pooled across ALL (Z500×RR) groups."""
    scales: dict[str, float] = {}
    for field in impact_fields:
        fd = fields_data[field]
        all_vals: list[np.ndarray] = []
        for dates in all_group_dates:
            comp = _composite_over_dates(fd, dates)
            if comp is not None:
                all_vals.append(np.abs(comp).ravel())
        if all_vals:
            scales[field] = max(float(np.percentile(np.concatenate(all_vals), 95)), 1e-6)
        else:
            scales[field] = 1.0
    return scales


def _render_rgb(
    comp_by_field: dict[str, np.ndarray | None],
    impact_fields: list[str],
    global_scales: dict[str, float],
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Build additive-RGB array over white from per-field composites."""
    rgb = np.zeros((*ref_shape, 3), dtype=float)
    for i, field in enumerate(impact_fields):
        comp = comp_by_field.get(field)
        if comp is not None and comp.shape == ref_shape:
            # nan_to_num: a field missing data at a pixel (e.g. SST over land) must
            # contribute zero to its channel, not NaN -- _additive_over_white's
            # rgb.max(axis=-1) propagates a single NaN channel across all three,
            # blanking out otherwise-valid signal from the other two fields.
            rgb[..., i] = np.nan_to_num(_pos_norm(comp, global_scales[field]), nan=0.0)
    return _additive_over_white(rgb)


def _draw_rgb_panel(ax, is_geo, lats, lons, rgb_img, *, bbox_dict, module_root, region, bbox_step,
                    z_lons=None, z_lats=None, z500_mean=None, levels=7) -> None:
    """Render the RGB array on *ax* with geo-boundaries and optional Z500 isolines.

    ``z500_mean``/``z_lons``/``z_lats`` should be THIS panel's own (Z500 x RR) sub-regime
    composite (raw Z500 in gpm when available — see ``z500_raw_sub`` in ``run_rgb_overview``),
    not the parent Z500-regime-wide mean, so the overlay actually differs across the RR columns
    in a row instead of repeating one shared contour set.
    """
    extent = rr_geo.map_extent(bbox_dict, lats, lons)
    tr = rr_geo.data_transform() if is_geo else None
    lat_ascending = lats[0] < lats[-1]
    img = rgb_img[::-1, :, :] if lat_ascending else rgb_img
    kw = dict(origin="upper", extent=[extent[0], extent[1], extent[2], extent[3]],
              interpolation="nearest", aspect="auto")
    if is_geo and tr is not None:
        kw["transform"] = tr
    ax.imshow(np.clip(img, 0.0, 1.0), **kw)
    if z500_mean is not None and z_lons is not None and z_lats is not None:
        _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, z500_mean, levels=levels)
    rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
    ax.set_xticks([]); ax.set_yticks([])


def _draw_empty_panel(ax, msg: str = "") -> None:
    ax.set_facecolor(_EMPTY_COLOR)
    if msg:
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=7,
                color="#999999", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _add_rgb_legend(
    fig,
    impact_fields: list[str],
    channel_of: dict[str, str],
    panels_frac: float,
    physical_scales: dict[str, float] | None = None,
    units_of: dict[str, str] | None = None,
    y0: float = 0.12,
    height: float = 0.75,
    fontsize: float = 8.0,
):
    """Place a ternary legend to the right of the map panels, spanning ``[y0, y0+height]``
    in figure-fraction coordinates (defaults to nearly the full figure height — the original
    single-legend-per-figure behaviour). Pass a specific ``y0``/``height`` (e.g. from
    ``ax.get_position()`` of one row's panel) to instead place a legend aligned with just
    that row, when different rows use different field sets and therefore need their own
    R/G/B colour key (see ``run_rgb_overview``'s per-row legends).

    Pass ``physical_scales``/``units_of`` to label the ternary edges in each
    field's native physical units (its 95th-percentile composite scale) rather
    than the raw 0..1 normalisation fraction.

    Returns the created legend Axes (caller may ``.remove()`` it to redraw a
    different legend variant on the same figure).
    """
    from .plot_rgb_scale import draw_rgb_scale
    legend_left = panels_frac + 0.02
    legend_width = max(0.05, (1.0 - panels_frac) - 0.04)
    ax = fig.add_axes([legend_left, y0, legend_width, height])
    draw_rgb_scale(ax, channel_of, _CHANNEL_RGB, impact_fields, fontsize=fontsize,
                    physical_scales=physical_scales, units=units_of)
    return ax


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def run_rgb_overview(module_root, region, bbox_step, smooth_window=11, out_dir=None, mode=None,
                      z500_cluster_method="ward", formats: tuple[str, ...] = ("png",),
                      legend_units: str = "fraction", n_bins: int | None = None,
                      output_tag: str = "", area_weight_tag: str = "",
                      bin_overrides_path: Path | None = None) -> list[Path]:
    """``legend_units``: "fraction" (default; ternary ticks 0..1) | "physical"
    (ternary ticks in each field's native units) | "both" (saves two files —
    the physical-unit variant gets a "_physlegend" filename suffix). The RGB
    map panels themselves are identical across variants; only the legend
    differs, so the underlying data/composites are computed once.

    ``mode="logistic"`` plots the Stage-2 [LOGISTIC] continuous
    ``predicted_extreme_prob`` index instead of a discrete RR regimes file —
    quantile-binned into ``n_bins`` groups per Z500 regime at plot time
    (default 2), same as ``plot_composites.py``'s ``mode="logistic"``. Column
    headers read "Bin<b>" instead of "SR<rr>" in this mode. ``output_tag``
    selects among several logistic runs that differ only by extreme-label
    definition and is ignored otherwise. ``bin_overrides_path``, if given,
    points at a ``select_subregime_bins.py`` YAML of per-regime data-driven
    probability edges (see ``plot_composites.load_composite_inputs_logistic``)
    — overrides ``n_bins`` for the regimes it covers; the column count
    becomes the max bin count across regimes, and rows for a regime with
    fewer bins than that max are left as "< MIN_DAYS" grey panels.
    """
    dw = _dw_root(module_root)
    is_logistic = mode == "logistic"
    per_z500_fields: dict[int, list[str]] | None = None

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
        print(f"[rr_rgb_overview] RR logistic index: {rr_path.name}")

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
            print(f"[rr_rgb_overview] Bin overrides: {bin_overrides_path.name} -> "
                  f"{ {z: len(e) + 1 for z, e in bin_edges_by_regime.items()} } bins by regime "
                  f"(others kept as a single undivided group -- see stage2_logistic."
              f"bin_predicted_prob_per_regime)")

        rr_labels = bin_predicted_prob_per_regime(
            predicted_prob, z500_regime, n_bins=n_bins, bin_edges_by_regime=bin_edges_by_regime,
        )
        n_rr = int(rr_labels.max()) if len(rr_labels) else int(n_bins)
        z500_on_rr = z500_regime
        z500_vals_raw = {int(v) for v in z500_on_rr}
        if z500_cluster_method == "ward":
            _, _groups = _get_regime_alignment(region)
            z500_vals = regime_alignment.canonical_order(z500_vals_raw, _groups, bbox_step)
            z500_canonical_map = regime_alignment.relabel_ids(z500_vals_raw, _groups, bbox_step)
            print(f"[rr_rgb_overview] Z500 regimes: {z500_vals} "
                  f"(canonical ids: {[z500_canonical_map[z] for z in z500_vals]}), {len(rr_dates)} days, n_bins={n_bins}")
        else:
            z500_vals = sorted(z500_vals_raw)
            z500_canonical_map = {}
            print(f"[rr_rgb_overview] Z500 regimes: {z500_vals} "
                  f"(cluster_method={z500_cluster_method!r} -- regime_alignment_<region>.yaml is "
                  f"Ward-only, no canonical ordering/labelling), {len(rr_dates)} days, n_bins={n_bins}")

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
        z500_path = _find_z500_regimes(dw, region, bbox_step, smooth_window=smooth_window,
                                        cluster_method=z500_cluster_method or "ward")
        if z500_path is None:
            raise FileNotFoundError(
                f"No Z500 regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})"
            )
        print(f"[rr_rgb_overview] RR regimes:   {rr_path.name}")
        print(f"[rr_rgb_overview] Z500 regimes: {z500_path.name}")

        # --- Labels ---
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
            print(f"[rr_rgb_overview] {n_rr} RR regimes, Z500 regimes: {z500_vals} "
                  f"(canonical ids: {[z500_canonical_map[z] for z in z500_vals]}), {len(rr_dates)} days")
        else:
            z500_vals = sorted(z500_vals_raw)
            z500_canonical_map = {}
            print(f"[rr_rgb_overview] {n_rr} RR regimes, Z500 regimes: {z500_vals} "
                  f"(cluster_method={z500_cluster_method!r} -- regime_alignment_<region>.yaml is "
                  f"Ward-only, no canonical ordering/labelling), {len(rr_dates)} days")

        # --- Fields from filename ---
        m = re.search(r"z500([^_]*)_w\d+", rr_path.name)
        fields = ["z500"] + ([f for f in m.group(1).split("+") if f] if (m and m.group(1)) else [])

    fields_data: dict[str, FieldData] = {}
    for field in fields:
        p = _find_anomaly(dw, region, field, bbox_step, smooth_window, area_weight_tag=area_weight_tag)
        fd = _load_field(p) if p else None
        if fd is not None:
            fields_data[field] = fd
            print(f"[rr_rgb_overview]   ✓ {field}: {p.name}")
        else:
            print(f"[rr_rgb_overview]   ✗ {field}: not found — skipping")
    if not fields_data:
        raise RuntimeError("No field anomalies loaded.")

    # Full universe of impact fields loaded across EVERY Z500 regime (per_z500_fields, when
    # available, gives each regime/row its own subset of these, selected below per row — see
    # "Cols 1..n_rr" — since rr_02's logistic fit can pick a different 3-field combination per
    # Z500 regime; this uncapped list only sizes the pooled colour scale and is the fallback
    # when no per-regime field map exists).
    all_impact_fields = [f for f in fields_data if f != "z500"]
    has_z500 = "z500" in fields_data
    if not all_impact_fields:
        raise RuntimeError("No impact fields (non-Z500) available for RGB composites.")

    col_label = (lambda i: f"Bin{i}") if is_logistic else (lambda i: f"SR{i}")

    bbox_dict = rr_geo.load_bbox_for_step(module_root, region, bbox_step)

    # --- Per-(Z500×RR) group structures ---
    # group_dates[z][rr] = DatetimeIndex | None
    group_dates: dict[int, dict[int, pd.DatetimeIndex | None]] = {}
    z500_mean: dict[int, np.ndarray | None] = {}   # mean Z500 per parent regime
    for z in z500_vals:
        z_mask = z500_on_rr == z
        group_dates[z] = {}
        for rr in range(1, n_rr + 1):
            g = z_mask & (rr_labels == rr)
            group_dates[z][rr] = rr_dates[g] if int(g.sum()) >= MIN_DAYS else None
        if has_z500:
            z500_mean[z] = _composite_over_dates(fields_data["z500"], rr_dates[z_mask])
        else:
            z500_mean[z] = None

    # Raw Z500 (gpm) per-(Z500×RR)-group mean for isolines — overlaid on each RGB panel, so the
    # contour actually differs across the RR columns in a row instead of every column repeating
    # the parent Z500-regime-wide mean. Falls back to the anomaly field's own contours (shared
    # per row, old behavior) only when raw Z500 isn't available at all.
    # Raw Z500 is loaded from a domain-wide file (one per region, not per bbox step) — crop it to
    # this bbox's window before computing anything, otherwise contour levels get spaced across the
    # WHOLE domain's value range and only 0-1 of them ever fall inside the visible bbox.
    raw_z500_path = _find_raw_z500(dw, region)
    z500_raw_fd = None
    if raw_z500_path is not None:
        print(f"[rr_rgb_overview] Raw Z500: {raw_z500_path.name}")
        z500_raw_fd = _crop_field_to_bbox(_load_raw_z500_field(raw_z500_path), bbox_dict)
    else:
        print("[rr_rgb_overview] WARNING: raw Z500 not found; isolines will use anomaly field")

    z500_raw_sub: dict[int, dict[int, np.ndarray | None]] = {}
    # Raw Z500 (gpm) mean over ALL days of each parent Z500 regime (not split by RR sub-regime)
    # — for the col-0 isoline overlay, so it matches the "affected days" of that Z500 regime.
    z500_raw_parent: dict[int, np.ndarray | None] = {}
    z_lats_raw = z_lons_raw = None
    if z500_raw_fd is not None:
        z_lats_raw, z_lons_raw = z500_raw_fd.lats, z500_raw_fd.lons
        for z in z500_vals:
            z500_raw_sub[z] = {
                rr: (_composite_over_dates(z500_raw_fd, group_dates[z][rr])
                     if group_dates[z][rr] is not None else None)
                for rr in range(1, n_rr + 1)
            }
            z500_raw_parent[z] = _composite_over_dates(z500_raw_fd, rr_dates[z500_on_rr == z])

    # --- Global per-field scale (pooled across ALL groups) ---
    all_group_dates_flat = [
        group_dates[z][rr]
        for z in z500_vals
        for rr in range(1, n_rr + 1)
        if group_dates[z][rr] is not None
    ]
    global_scales = _global_field_scale(all_impact_fields, fields_data, all_group_dates_flat)
    print(f"[rr_rgb_overview] Global scales: { {f: f'{s:.4g}' for f, s in global_scales.items()} }")

    # Reference grid for RGB (first impact field)
    ref_fd = fields_data[all_impact_fields[0]]
    ref_lats, ref_lons = ref_fd.lats, ref_fd.lons
    ref_shape = (len(ref_lats), len(ref_lons))
    rgb_extent = rr_geo.map_extent(bbox_dict, ref_lats, ref_lons)

    # Z500 grid + shared Z500 color norm
    if has_z500:
        z_fd = fields_data["z500"]
        z_lats, z_lons = z_fd.lats, z_fd.lons
        all_z_means = [v for v in z500_mean.values() if v is not None]
        if all_z_means:
            zv = max(float(np.nanpercentile(np.abs(np.stack(all_z_means)), 98)), 1e-6)
            z_cmap, z_norm, z_extend = discretize_diverging(-zv, zv, "RdBu_r")
        else:
            z_cmap = z_norm = z_extend = None
        z_extent = rr_geo.map_extent(bbox_dict, z_lats, z_lons)
    else:
        z_lats = z_lons = z_cmap = z_norm = z_extend = z_extent = None

    # --- Figure layout ---
    n_rows = len(z500_vals)
    n_cols = 1 + n_rr   # col 0 = Z500 map, cols 1..n_rr = RGB sub-regimes
    panels_w = _PANEL_SIZE * n_cols
    fig_w = panels_w + _LEGEND_PAD
    fig_h = _PANEL_SIZE * n_rows + 1.2   # +1.2 for title + legend row
    fig = plt.figure(figsize=(fig_w, fig_h))
    panels_frac = panels_w / fig_w   # fraction of figure width used by map panels

    col_titles = ["Z500\n(regime mean)"] + [col_label(rr) for rr in range(1, n_rr + 1)]

    row_col0_axes: dict[int, "plt.Axes"] = {}
    row_impact_fields: dict[int, list[str]] = {}
    row_channel_of: dict[int, dict[str, str]] = {}

    for row_idx, z in enumerate(z500_vals):
        z_all_dates = rr_dates[z500_on_rr == z]
        n_z_days = len(z_all_dates)

        # Each Z500 regime/row gets its OWN field set — per_z500_fields (from the logistic
        # merged file's per_regime_fields attr) records exactly which fields rr_02's logistic
        # fit selected for THIS regime, which can differ regime to regime. Falls back to the
        # uncapped, run-wide field list only when no per-regime map exists (non-logistic modes,
        # or a logistic file without the attr) — same 3 fields in every row in that case.
        if per_z500_fields is not None:
            z_fields = [f for f in per_z500_fields.get(z, []) if f in fields_data]
            if not z_fields:
                # per_z500_fields is present (per-regime mode) but empty/missing for THIS
                # regime -- all_impact_fields is fields_data's alphabetical load order here,
                # not an importance order, so warn loudly rather than silently mislabeling
                # panels/RGB channels as importance-ranked.
                print(f"[rr_rgb_overview] WARNING: Z500={z} has no per-regime field selection "
                      f"(per_regime_fields attr present but empty/missing for this regime) -- "
                      f"falling back to all_impact_fields' alphabetical load order, NOT an "
                      f"importance order.")
                z_fields = all_impact_fields
        else:
            z_fields = all_impact_fields
        impact_fields = z_fields[:3]
        channel_of = {f: _CHANNELS[i] for i, f in enumerate(impact_fields)}
        row_impact_fields[z] = impact_fields
        row_channel_of[z] = channel_of

        # Col 0: Z500 mean map
        ax, is_geo = rr_geo.make_map_axes(
            fig, (n_rows, n_cols, row_idx * n_cols + 1), extent=z_extent
        )
        row_col0_axes[z] = ax
        zm = z500_mean[z]
        if zm is not None and z_norm is not None:
            tr = rr_geo.data_transform() if is_geo else None
            kw = dict(cmap=z_cmap, norm=z_norm, shading="auto")
            if tr is not None:
                kw["transform"] = tr
            im = ax.pcolormesh(z_lons, z_lats, zm, **kw)
            style_discrete_ticks(
                plt.colorbar(im, ax=ax, shrink=0.6, pad=0.03, aspect=14, orientation="vertical", extend=z_extend)
            )
        z_raw_parent = z500_raw_parent.get(z) if z500_raw_fd is not None else None
        if z_raw_parent is not None:
            _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_parent,
                                    levels=_z500_raw_contour_levels(z_raw_parent))
        elif zm is not None:
            _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, zm, levels=7)
        rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root,
                              region=region, bbox=bbox_dict)
        _canon = z500_canonical_map.get(z)
        _canon_label = f" [R{_canon}]" if _canon is not None else ""
        ax.set_ylabel(f"Z500-{z}{_canon_label}\n({n_z_days} d)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=8)

        # Cols 1..n_rr: RGB composites
        for rr in range(1, n_rr + 1):
            col_idx = rr   # column index in the subplot grid
            subplot_pos = row_idx * n_cols + col_idx + 1
            ax, is_geo = rr_geo.make_map_axes(
                fig, (n_rows, n_cols, subplot_pos), extent=rgb_extent
            )
            dates = group_dates[z][rr]
            if dates is None:
                _draw_empty_panel(ax, f"< {MIN_DAYS} d")
            else:
                comp_by_field = {
                    f: _composite_over_dates(fields_data[f], dates)
                    for f in impact_fields
                }
                rgb_img = _render_rgb(comp_by_field, impact_fields, global_scales, ref_shape)
                # Prefer this (Z500×RR) group's own raw-Z500 mean for the isoline overlay (differs
                # per RR column); fall back to the shared parent-regime anomaly mean only if raw
                # Z500 wasn't available at all.
                z_raw_rr = z500_raw_sub.get(z, {}).get(rr) if z500_raw_fd is not None else None
                if z_raw_rr is not None:
                    overlay_lons, overlay_lats, overlay_field = z_lons_raw, z_lats_raw, z_raw_rr
                    overlay_levels = _z500_raw_contour_levels(z_raw_rr)
                else:
                    overlay_lons, overlay_lats, overlay_field = z_lons, z_lats, zm
                    overlay_levels = 7
                _draw_rgb_panel(
                    ax, is_geo, ref_lats, ref_lons, rgb_img,
                    bbox_dict=bbox_dict, module_root=module_root,
                    region=region, bbox_step=bbox_step,
                    z_lons=overlay_lons, z_lats=overlay_lats, z500_mean=overlay_field,
                    levels=overlay_levels,
                )
                # Day-count label in every cell
                ax.text(0.02, 0.02, f"n={len(dates)}", transform=ax.transAxes,
                        fontsize=6, color="black",
                        bbox=dict(facecolor="white", alpha=0.6, pad=1, edgecolor="none"))
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=(0.0, 0.05, panels_frac, 1.0))

    units_of = {f: fields_data[f].units for f in all_impact_fields}

    out_dir = out_dir or (dw / "results" / "stage2_residual" / "figures" / region)
    out_dir.mkdir(parents=True, exist_ok=True)
    ward_tag = "_zward" if z500_cluster_method == "ward" else ""
    if is_logistic:
        logistic_tag = "_logisticadaptivebin" if bin_overrides_path is not None else f"_logistic{n_bins}bin"
    else:
        logistic_tag = ""
    base_stem = f"rr_rgb_overview_{region}_{bbox_step}{ward_tag}{logistic_tag}"

    # (filename suffix, physical_scales-or-None, units-or-None) — the map panels are
    # identical across variants, only the ternary legends' tick labels differ, so we
    # redraw just the legends on the SAME figure between saves rather than recomputing
    # composites twice.
    variants: list[tuple[str, dict[str, float] | None, dict[str, str] | None]] = []
    if legend_units in ("fraction", "both"):
        variants.append(("", None, None))
    if legend_units in ("physical", "both"):
        variants.append(("_physlegend", global_scales, units_of))
    if not variants:
        variants.append(("", None, None))

    written: list[Path] = []
    for suffix, phys_scales, units_map in variants:
        # One legend PER ROW, vertically aligned with that row's own panels (via its col-0
        # axes' actual post-tight_layout position) — rows can use different field sets, so a
        # single shared legend would misrepresent every row except whichever one it was drawn
        # from (see row_impact_fields/row_channel_of, set per row above).
        legend_axes = []
        for z in z500_vals:
            pos = row_col0_axes[z].get_position()
            legend_axes.append(_add_rgb_legend(
                fig, row_impact_fields[z], row_channel_of[z], panels_frac,
                physical_scales=phys_scales, units_of=units_map,
                y0=pos.y0, height=(pos.y1 - pos.y0), fontsize=7,
            ))

        stem = f"{base_stem}{suffix}"
        if "png" in formats:
            out_png = out_dir / f"{stem}.png"
            fig.savefig(out_png, bbox_inches="tight", dpi=200)
            written.append(out_png)
            print(f"[rr_rgb_overview] Saved: {out_png}")
        if "pdf" in formats:
            out_pdf = out_dir / f"{stem}.pdf"
            with PdfPages(out_pdf) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
            written.append(out_pdf)
            print(f"[rr_rgb_overview] Saved: {out_pdf}")

        for ax_legend in legend_axes:
            ax_legend.remove()

    plt.close(fig)

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import sys
    p = argparse.ArgumentParser(
        description="Single-page RGB overview: Z500 maps + sub-regime composites.")
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
    p.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="ward",
                   help="Restrict to RR/Z500 regime files built on this Z500 parent-regime "
                        "source (default 'ward', reads '_zward'-tagged files, matches "
                        "regime_alignment_<region>.yaml; 'kmeans' reads untagged files).")
    p.add_argument(
        "--formats", default="png",
        help="Comma-separated output formats: png (default) and/or pdf.",
    )
    p.add_argument(
        "--legend-units", choices=("fraction", "physical", "both"), default="fraction",
        help="Ternary-legend tick labels: 'fraction' (default, 0..1), 'physical' "
             "(each field's native units, using its 95th-pct composite scale), or "
             "'both' (saves a second file per format with a '_physlegend' suffix).",
    )
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
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.module_root is None:
        args.module_root = Path(__file__).resolve().parents[2]

    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())

    bin_overrides_path = args.bin_overrides_path
    if args.mode == "logistic" and bin_overrides_path is None:
        default_overrides = (
            args.module_root.expanduser().resolve()
            / "src" / "common" / f"bin_overrides_{args.region}_custom.yaml"
        )
        if default_overrides.is_file():
            print(f"[rr_rgb_overview] Using default bin overrides: {default_overrides}")
            bin_overrides_path = default_overrides

    run_rgb_overview(
        module_root=args.module_root.expanduser().resolve(),
        region=args.region,
        bbox_step=args.bbox_step,
        smooth_window=args.smooth_window,
        out_dir=args.out_dir,
        mode=args.mode,
        z500_cluster_method=args.z500_cluster_method,
        formats=formats,
        legend_units=args.legend_units,
        n_bins=args.n_bins, output_tag=args.output_tag,
        area_weight_tag=args.area_weight_tag,
        bin_overrides_path=bin_overrides_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
