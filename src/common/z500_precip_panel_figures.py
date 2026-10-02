"""Z500 + precipitation regime composites.

- ``build_figure_regime_z500_precip_pair``: one 1×2 figure per regime (legacy).
- ``build_figure_regime_z500_precip_mosaic``: k×2 long mosaic (one row per regime).
- ``build_figure_regime_z500_std_precip_mosaic``: k×3 long mosaic (one row per regime, columns =
  mean anomaly / std-dev / precip-or-EPE metric) -- sibling of ``build_figure_regime_z500_precip_mosaic``
  with an inserted STD-DEV column, used by ``plot_regime_z500_anomaly_std_mosaic.py``.
- ``build_figure_regime_z500_precip_mosaic_wide``: 2×k wide mosaic (Z500 top, precip bottom).
- ``build_figure_z500_precip_mosaic_by_bbox``: 2×nb wide cross-bbox mosaic.
- 3×k / 3×nb multivariate wide mosaics live in ``common.z500_precip_mv3row_wide``.
- ``build_figure_z500_precip_mosaic_by_bbox_long``: nb×2 long cross-bbox mosaic.
- ``build_figure_z500_precip_mosaic_by_bbox_with_boxplot``: same 2×nb map layout as
  ``build_figure_z500_precip_mosaic_by_bbox``, plus one extra column (spanning both rows) with
  a boxplot of a per-day metric (e.g. precip amount) across bbox steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from common.colorbar_discretize import discretize_diverging, discretize_sequential_from_floor, style_discrete_ticks
from common.boxplot_style import style_boxplot
from common.geo import (
    cartopy_lines_on_top,
    cartopy_modules,
    cartopy_underlay,
    configure_plain_composite_axis_ticks,
    configure_regime_composite_gridlines,
    draw_aoi_box,
    label_contour_heights,
    resolve_map_extent,
)
from common.physics import G0, is_geopotential_units
from common.spatial import spatial_dims, subset_da_to_bbox_or_raise, wrap_lon_sort_lat_da
from common.step1011_figures import _cmap_fill_limits, regime_panel_title
from common.page_layout import (
    PDF_FS_FIG_AXIS,
    PDF_FS_FIG_CLABEL,
    PDF_FS_FIG_SUPTITLE,
    PDF_FS_FIG_TICK,
    PDF_FS_FIG_TITLE,
)

PANEL_W_IN = 5.8
PANEL_H_IN = 4.4
MOSAIC_PANEL_W_IN = 5.4
MOSAIC_PANEL_H_IN = 4.0
# EPE.7 wide layout (2×nb): top row = Z500 per bbox; bottom = precip.
CROSSBBOX_WIDE_TOP_PANEL_W_IN = 5.4
CROSSBBOX_WIDE_TOP_PANEL_H_IN = 4.6
CROSSBBOX_WIDE_BOTTOM_PANEL_H_IN = 3.2
CROSSBBOX_WIDE_FONT_SCALE = 1.2
# Multivariate 3-row wide layout (map row + horizontal colorbar per field).
MV3_ROW_MAP_H_IN = 3.6
# Buffer between map panels and their colorbar (clears lon tick labels / gridlines).
MV3_ROW_MAP_PAD_IN = 0.24
# Grid row reserved for colorbar + ticks + label (bar itself is drawn thinner inside).
MV3_ROW_CBAR_H_IN = 0.50
# Spacer row between colorbar block and the next map row (prevents label overlap).
MV3_ROW_GAP_IN = 0.48
MOISTURE_FILL_CMAP = "YlGnBu"


def _scaled_fig_font(base: float, *, scale: float = CROSSBBOX_WIDE_FONT_SCALE) -> float:
    return float(base) * scale


def _force_equal_axes_boxes(axes: list) -> None:
    """Give every axis the same display size (fixes Cartopy extent-driven shrink)."""
    if not axes:
        return
    positions = [ax.get_position() for ax in axes]
    max_w = max(p.width for p in positions)
    max_h = max(p.height for p in positions)
    for ax, p in zip(axes, positions):
        new_x0 = p.x0 + (p.width - max_w) / 2.0
        new_y0 = p.y0 + (p.height - max_h) / 2.0
        ax.set_position([new_x0, new_y0, max_w, max_h])


def _style_colorbar_outline(cb) -> None:
    """Thin black border around the colour strip (not the full cbar axes), plus
    (for discretized colorbars) clean tick labels -- see ``style_discrete_ticks``.
    """
    cb.outline.set_visible(True)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("black")
    for spine in cb.ax.spines.values():
        spine.set_visible(False)
    style_discrete_ticks(cb)


def _mosaic_column_colorbar_layout(
    fig_h: float,
    n_rows: int,
) -> tuple[float, list[float], list[float]]:
    """Axes right edge and colorbar ``add_axes`` rects for nb×2 mosaics.

    Keeps map panels clear of shared column colorbars by shrinking ``right`` and
    placing two stacked colorbars in the margin (Z500 upper, precip lower).
    """
    suptitle_reserve = 0.55 / max(fig_h, 1.0)
    top = 1.0 - suptitle_reserve
    bottom = 0.07
    avail = max(top - bottom, 0.2)
    cb_w = 0.014
    axes_right = 0.76 if n_rows >= 5 else (0.78 if n_rows >= 3 else 0.80)
    x_cb = axes_right + 0.022
    gap = avail * 0.06
    band = (avail - gap) / 2.0
    cax_z = [x_cb, bottom + band + gap, cb_w, band]
    cax_p = [x_cb, bottom, cb_w, band]
    return axes_right, cax_z, cax_p


def _precip_cmap_for_metric(metric: str) -> str:
    """Both metrics use Blues (darker = higher): ``flag_frequency`` (EPE.6/EPE.7 -- % of
    regime days flagged) and ``mean_tp`` (mean precip amount)."""
    return "Blues"


def _z500_fill_label(units_in: str, *, std: bool = False) -> str:
    """"Z500 anomaly (gpm)" only when the source field's units say it's a real height field;
    a normalized/standardized (z-score) composite must not be captioned as physical meters."""
    suffix = " std-dev" if std else ""
    if is_geopotential_units(str(units_in or "")):
        return f"Z500 anomaly{suffix} (gpm)"
    return f"Z500 anomaly{suffix} (σ, norm.)"


def _moisture_panel_units(da: xr.DataArray) -> str:
    units = str(getattr(da, "attrs", {}).get("units", "") or "").strip()
    if units:
        return f"TCWV anomaly ({units})"
    return "TCWV anomaly (kg m$^{-2}$)"


def _mesh_2d(da: xr.DataArray, lat_dim: str, lon_dim: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = da.squeeze(drop=True)
    if {lat_dim, lon_dim}.issubset(a.dims):
        a = a.transpose(lat_dim, lon_dim)
    lon1 = np.asarray(a[lon_dim].values, dtype=float)
    lat1 = np.asarray(a[lat_dim].values, dtype=float)
    z = np.asarray(a.values, dtype=float)
    ny, nx = int(lat1.size), int(lon1.size)
    if z.shape == (nx, ny):
        z = z.T
    if z.shape != (ny, nx):
        raise ValueError(f"Expected 2D (lat,lon)=({ny},{nx}); got {z.shape}")
    lon_m, lat_m = np.meshgrid(lon1, lat1)
    return lon_m, lat_m, z


def build_figure_regime_z500_precip_pair(
    *,
    regime: int,
    z500_anom: xr.DataArray,
    z500_raw: xr.DataArray | None,
    precip: xr.DataArray,
    z500_bbox: dict[str, float],
    precip_bbox: dict[str, float] | None = None,
    title_line: str,
    subtitle_z500: str | None = None,
    subtitle_precip: str | None = None,
    n_days_by_regime: dict[int, int] | None = None,
    precip_units: str = "mm day$^{-1}$",
    precip_metric: str = "flag_frequency",
    aoi_bbox: dict[str, float] | None = None,
    align_groups: list[dict] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """1×2 Cartopy figure for a single regime."""
    p_bbox = precip_bbox if precip_bbox is not None else z500_bbox
    lat_dim, lon_dim = spatial_dims(z500_anom)
    lat_dim_p, lon_dim_p = spatial_dims(precip)

    anom = subset_da_to_bbox_or_raise(z500_anom.astype(np.float64).load(), z500_bbox)
    pr = subset_da_to_bbox_or_raise(precip.astype(np.float64).load(), p_bbox)
    anom = wrap_lon_sort_lat_da(anom, lat_dim, lon_dim)
    pr = wrap_lon_sort_lat_da(pr, lat_dim_p, lon_dim_p)
    raw_m = None
    if z500_raw is not None:
        raw = subset_da_to_bbox_or_raise(z500_raw.astype(np.float64).load(), z500_bbox)
        raw = wrap_lon_sort_lat_da(raw, lat_dim, lon_dim)
        raw_units_in = str(getattr(raw, "attrs", {}).get("units", ""))
        raw_m = (raw / G0) if is_geopotential_units(raw_units_in) else raw

    anom_units_in = str(getattr(anom, "attrs", {}).get("units", ""))
    anom_gpm = (anom / G0) if is_geopotential_units(anom_units_in) else anom

    _lat_extent = np.asarray(anom[lat_dim].values, dtype=float)
    _lon_extent = np.asarray(anom[lon_dim].values, dtype=float)
    map_extent_z = resolve_map_extent(z500_bbox, lat=_lat_extent, lon=_lon_extent)
    _lat_extent_p = np.asarray(pr[lat_dim_p].values, dtype=float)
    _lon_extent_p = np.asarray(pr[lon_dim_p].values, dtype=float)
    map_extent_p = resolve_map_extent(p_bbox, lat=_lat_extent_p, lon=_lon_extent_p)

    precip_cmap_name = _precip_cmap_for_metric(precip_metric)
    vmin_z, vmax_z = _cmap_fill_limits(np.asarray(anom_gpm.values, dtype=float), cmap_name="RdBu_r")
    vmin_p, vmax_p = _cmap_fill_limits(np.asarray(pr.values, dtype=float), cmap_name=precip_cmap_name)
    # Anchor precip at 0 (white/light = none → darker = more).
    vmin_p = 0.0
    if not (vmax_p > vmin_p):
        vmax_p = vmin_p + 1.0
    cmap_z, norm_z, extend_z = discretize_diverging(vmin_z, vmax_z, "RdBu_r")
    cmap_p, norm_p, extend_p = discretize_sequential_from_floor(vmax_p, precip_cmap_name, vmin=vmin_p)

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    fig = plt.figure(figsize=(PANEL_W_IN * 2 + 0.6, PANEL_H_IN + 0.55), facecolor="white")

    z_head = regime_panel_title(
        regime=int(regime),
        regime_case_counts=n_days_by_regime,
        align_groups=align_groups,
        canonical_id_by_regime=canonical_id_by_regime,
        literature_abbr_by_short=literature_abbr_by_short,
    )
    p_head = z_head

    for col, (field, cmap, norm, extend, head, sub, ldim, odim, mextent) in enumerate(
        (
            (anom_gpm, cmap_z, norm_z, extend_z, z_head, subtitle_z500, lat_dim, lon_dim, map_extent_z),
            (pr, cmap_p, norm_p, extend_p, p_head, subtitle_precip, lat_dim_p, lon_dim_p, map_extent_p),
        )
    ):
        ax = (
            fig.add_subplot(1, 2, col + 1, projection=ccrs.PlateCarree())
            if use_geo
            else fig.add_subplot(1, 2, col + 1)
        )
        lon_m, lat_m, za = _mesh_2d(field, ldim, odim)

        levels_raw = None
        if col == 0 and raw_m is not None:
            _, _, zr = _mesh_2d(raw_m, lat_dim, lon_dim)
            cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
            if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
            if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                mu = float(np.nanmean(zr)) if np.any(np.isfinite(zr)) else 0.0
                cmin, cmax = mu - 10.0, mu + 10.0
            levels_raw = np.linspace(cmin, cmax, 9)

        if use_geo and cfeat is not None and pc is not None and mextent is not None:
            cartopy_underlay(ax, mextent, cfeature=cfeat)
            im = ax.pcolormesh(
                lon_m,
                lat_m,
                za,
                shading="auto",
                cmap=cmap,
                norm=norm,
                alpha=0.85 if col == 0 else 1.0,
                transform=pc,
                zorder=8,
                rasterized=True,
            )
            cs = None
            if col == 0 and levels_raw is not None:
                cs = ax.contour(
                    lon_m,
                    lat_m,
                    zr,
                    levels=levels_raw,
                    colors="k",
                    linewidths=1.0,
                    alpha=0.95,
                    transform=pc,
                    zorder=12,
                )
            cartopy_lines_on_top(ax, cfeature=cfeat)
            if cs is not None:
                label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
            draw_aoi_box(ax, aoi_bbox, transform=pc)
            ax.set_extent(mextent, crs=pc)
            configure_regime_composite_gridlines(
                ax,
                panel_row=0,
                panel_col=col,
                nrow=1,
                ncol=2,
                linewidth=0.5,
                alpha=0.35,
                tick_labelsize=PDF_FS_FIG_TICK,
            )
        else:
            im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm)
            if col == 0 and levels_raw is not None:
                ax.contour(lon_m, lat_m, zr, levels=levels_raw, colors="k", linewidths=1.0)
            draw_aoi_box(ax, aoi_bbox)

        cbar_label = _z500_fill_label(anom_units_in) if col == 0 else precip_units
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label, extend=extend)
        style_discrete_ticks(cb)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def build_figure_regime_z500_precip_mosaic(
    *,
    regimes: list[int],
    z500_anom_by_regime: dict[int, xr.DataArray],
    z500_raw_by_regime: dict[int, xr.DataArray | None] | None,
    precip_by_regime: dict[int, xr.DataArray],
    z500_bbox: dict[str, float],
    precip_bbox: dict[str, float] | None = None,
    title_line: str,
    subtitle_z500: str | None = None,
    subtitle_precip: str | None = None,
    n_days_by_regime: dict[int, int] | None = None,
    precip_units: str = "% of regime days",
    z500_vmin: float | None = None,
    z500_vmax: float | None = None,
    precip_vmin: float | None = None,
    precip_vmax: float | None = None,
    precip_metric: str = "flag_frequency",
    aoi_bbox: dict[str, float] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    align_groups: list[dict] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """k×2 long mosaic: one row per regime, columns = (Z500 anomaly, precip metric).

    ``canonical_id_by_regime``, if given, maps this bbox step's raw regime id to the
    cross-bbox-aligned physical-regime id (``regime_alignment_<region>.yaml`` via
    ``common.regime_alignment``) -- shown alongside the raw id in each row's label so
    "canonical Regime N" always names the same physical pattern across bbox steps. Row order
    (the ``regimes`` list) is expected to already be canonically sorted by the caller.

    Color scales within each column are shared across regimes (single ``vmin``/``vmax``
    per column). Pass explicit ``z500_*`` / ``precip_*`` limits to also share scales
    across multiple figures (different bboxes) — otherwise they're auto-computed from
    the supplied composites.
    """
    if not regimes:
        raise ValueError("Need at least one regime to build the mosaic.")
    p_bbox = precip_bbox if precip_bbox is not None else z500_bbox
    raw_provided = z500_raw_by_regime is not None and any(
        z500_raw_by_regime.get(int(r)) is not None for r in regimes
    )

    ref_anom = next(iter(z500_anom_by_regime.values()))
    ref_precip = next(iter(precip_by_regime.values()))
    lat_dim, lon_dim = spatial_dims(ref_anom)
    lat_dim_p, lon_dim_p = spatial_dims(ref_precip)

    anom_loaded: dict[int, xr.DataArray] = {}
    pr_loaded: dict[int, xr.DataArray] = {}
    raw_loaded: dict[int, xr.DataArray | None] = {}
    z_units_in = str(getattr(ref_anom, "attrs", {}).get("units", ""))
    for r in regimes:
        r_int = int(r)
        anom = subset_da_to_bbox_or_raise(
            z500_anom_by_regime[r_int].astype(np.float64).load(), z500_bbox
        )
        anom = wrap_lon_sort_lat_da(anom, lat_dim, lon_dim)
        units_in = str(getattr(anom, "attrs", {}).get("units", ""))
        anom_loaded[r_int] = (anom / G0) if is_geopotential_units(units_in) else anom
        pr = subset_da_to_bbox_or_raise(
            precip_by_regime[r_int].astype(np.float64).load(), p_bbox
        )
        pr_loaded[r_int] = wrap_lon_sort_lat_da(pr, lat_dim_p, lon_dim_p)
        if raw_provided:
            raw_da = (z500_raw_by_regime or {}).get(r_int)
            if raw_da is None:
                raw_loaded[r_int] = None
            else:
                raw = subset_da_to_bbox_or_raise(raw_da.astype(np.float64).load(), z500_bbox)
                raw = wrap_lon_sort_lat_da(raw, lat_dim, lon_dim)
                ru = str(getattr(raw, "attrs", {}).get("units", ""))
                raw_loaded[r_int] = (raw / G0) if is_geopotential_units(ru) else raw
        else:
            raw_loaded[r_int] = None

    def _flat(da_list: list[xr.DataArray]) -> np.ndarray:
        return np.concatenate([np.asarray(da.values, dtype=float).ravel() for da in da_list])

    precip_cmap_name = _precip_cmap_for_metric(precip_metric)
    if z500_vmin is None or z500_vmax is None:
        vmin_z, vmax_z = _cmap_fill_limits(_flat(list(anom_loaded.values())), cmap_name="RdBu_r")
        z500_vmin = vmin_z if z500_vmin is None else z500_vmin
        z500_vmax = vmax_z if z500_vmax is None else z500_vmax
    _precip_vmin_explicit = precip_vmin is not None
    if precip_vmin is None or precip_vmax is None:
        _, vmax_p = _cmap_fill_limits(_flat(list(pr_loaded.values())), cmap_name=precip_cmap_name)
        if precip_vmax is None:
            precip_vmax = vmax_p
    if not _precip_vmin_explicit:
        precip_vmin = 0.0
    if precip_vmax is None or not (precip_vmax > precip_vmin):
        precip_vmax = precip_vmin + 1.0

    n_rows = len(regimes)
    fig_w = MOSAIC_PANEL_W_IN * 2 + 1.2
    fig_h = MOSAIC_PANEL_H_IN * n_rows + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    _lat_z = np.asarray(ref_anom[lat_dim].values, dtype=float)
    _lon_z = np.asarray(ref_anom[lon_dim].values, dtype=float)
    map_extent_z = resolve_map_extent(z500_bbox, lat=_lat_z, lon=_lon_z)
    _lat_p = np.asarray(ref_precip[lat_dim_p].values, dtype=float)
    _lon_p = np.asarray(ref_precip[lon_dim_p].values, dtype=float)
    map_extent_p = resolve_map_extent(p_bbox, lat=_lat_p, lon=_lon_p)

    cmap_z, norm_z, extend_z = discretize_diverging(z500_vmin, z500_vmax, "RdBu_r")
    cmap_p, norm_p, extend_p = discretize_sequential_from_floor(precip_vmax, precip_cmap_name, vmin=precip_vmin)
    im_z = im_p = None
    for row, r in enumerate(regimes):
        r_int = int(r)
        for col, (field, cmap, norm, ldim, odim, mextent) in enumerate(
            (
                (anom_loaded[r_int], cmap_z, norm_z, lat_dim, lon_dim, map_extent_z),
                (pr_loaded[r_int], cmap_p, norm_p, lat_dim_p, lon_dim_p, map_extent_p),
            )
        ):
            ax = (
                fig.add_subplot(n_rows, 2, row * 2 + col + 1, projection=ccrs.PlateCarree())
                if use_geo
                else fig.add_subplot(n_rows, 2, row * 2 + col + 1)
            )
            lon_m, lat_m, za = _mesh_2d(field, ldim, odim)
            if use_geo and cfeat is not None and pc is not None and mextent is not None:
                cartopy_underlay(ax, mextent, cfeature=cfeat)
                im = ax.pcolormesh(
                    lon_m, lat_m, za,
                    shading="auto", cmap=cmap, norm=norm,
                    alpha=0.85 if col == 0 else 1.0,
                    transform=pc, zorder=8, rasterized=True,
                )
                cs = None
                if col == 0 and raw_loaded.get(r_int) is not None:
                    _, _, zr = _mesh_2d(raw_loaded[r_int], lat_dim, lon_dim)
                    cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
                    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                        cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
                    levels_raw = np.linspace(cmin, cmax, 9)
                    cs = ax.contour(
                        lon_m, lat_m, zr,
                        levels=levels_raw, colors="k", linewidths=0.9,
                        alpha=0.95, transform=pc, zorder=12,
                    )
                cartopy_lines_on_top(ax, cfeature=cfeat)
                if cs is not None:
                    label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
                draw_aoi_box(ax, aoi_bbox, transform=pc)
                ax.set_extent(mextent, crs=pc)
                configure_regime_composite_gridlines(
                    ax,
                    panel_row=row,
                    panel_col=col,
                    nrow=n_rows,
                    ncol=2,
                    linewidth=0.4,
                    alpha=0.3,
                    tick_labelsize=PDF_FS_FIG_TICK,
                )
            else:
                im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm)
                if col == 0 and raw_loaded.get(r_int) is not None:
                    _, _, zr = _mesh_2d(raw_loaded[r_int], lat_dim, lon_dim)
                    ax.contour(lon_m, lat_m, zr, colors="k", linewidths=0.9)
                draw_aoi_box(ax, aoi_bbox)
            if col == 0:
                im_z = im
                ax.set_ylabel(
                    regime_panel_title(
                        regime=r_int,
                        regime_case_counts=n_days_by_regime,
                        align_groups=align_groups,
                        canonical_id_by_regime=canonical_id_by_regime,
                        literature_abbr_by_short=literature_abbr_by_short,
                    ),
                    fontsize=PDF_FS_FIG_TITLE,
                    labelpad=14,
                )
            else:
                im_p = im

    axes_right, cax_z_rect, cax_p_rect = _mosaic_column_colorbar_layout(fig_h, n_rows)
    fig.subplots_adjust(
        left=0.05,
        right=axes_right,
        top=1.0 - 0.6 / fig_h,
        bottom=0.06,
        hspace=0.08,
        wspace=0.10,
    )
    if im_z is not None:
        cb_z = fig.colorbar(im_z, cax=fig.add_axes(cax_z_rect), extend=extend_z)
        _style_colorbar_outline(cb_z)
        cb_z.set_label(_z500_fill_label(z_units_in), fontsize=PDF_FS_FIG_AXIS, labelpad=8)
        cb_z.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    if im_p is not None:
        cb_p = fig.colorbar(im_p, cax=fig.add_axes(cax_p_rect), extend=extend_p)
        _style_colorbar_outline(cb_p)
        cb_p.set_label(precip_units, fontsize=PDF_FS_FIG_AXIS, labelpad=8)
        cb_p.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    return fig


def _mosaic_column_colorbar_layout_n(
    fig_h: float,
    n_rows: int,
    n_cols: int,
) -> tuple[float, list[list[float]]]:
    """N-column generalization of ``_mosaic_column_colorbar_layout``: axes right edge and
    ``add_axes`` rects for ``n_cols`` stacked colorbars in the right margin, one per column,
    in column order (col 0 on top).
    """
    suptitle_reserve = 0.55 / max(fig_h, 1.0)
    top = 1.0 - suptitle_reserve
    bottom = 0.07
    avail = max(top - bottom, 0.2)
    cb_w = 0.014
    axes_right = 0.74 if n_rows >= 5 else (0.76 if n_rows >= 3 else 0.78)
    x_cb = axes_right + 0.022
    gap = avail * 0.04
    band = (avail - gap * (n_cols - 1)) / n_cols
    rects = [[x_cb, bottom + (n_cols - 1 - i) * (band + gap), cb_w, band] for i in range(n_cols)]
    return axes_right, rects


def build_figure_regime_z500_std_precip_mosaic(
    *,
    regimes: list[int],
    z500_anom_by_regime: dict[int, xr.DataArray],
    z500_std_by_regime: dict[int, xr.DataArray],
    z500_raw_by_regime: dict[int, xr.DataArray | None] | None,
    precip_by_regime: dict[int, xr.DataArray],
    z500_bbox: dict[str, float],
    precip_bbox: dict[str, float] | None = None,
    title_line: str,
    subtitle_z500: str | None = None,
    subtitle_std: str | None = None,
    subtitle_precip: str | None = None,
    n_days_by_regime: dict[int, int] | None = None,
    precip_units: str = "% of regime days",
    z500_vmin: float | None = None,
    z500_vmax: float | None = None,
    std_vmin: float | None = None,
    std_vmax: float | None = None,
    precip_vmin: float | None = None,
    precip_vmax: float | None = None,
    precip_metric: str = "flag_frequency",
    aoi_bbox: dict[str, float] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    align_groups: list[dict] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """k×3 long mosaic: one row per regime, columns = (Z500 mean anomaly, Z500 std-dev, precip/EPE metric).

    Sibling of ``build_figure_regime_z500_precip_mosaic`` with an inserted middle STD-DEV
    column. Column cmaps here are fixed regardless of ``precip_metric`` (which is now moot for
    the precip cmap itself -- see ``_precip_cmap_for_metric`` -- but still used for the vmin
    anchor / column label): mean anomaly always RdBu_r (diverging, signed), std always YlGnBu
    (sequential, >=0, yellow-to-blue), and the precip/EPE column always Blues (sequential, >=0,
    darker = higher) -- keeps the std column visually distinct
    from the precip/EPE column even when both are otherwise sequential ">=0" fields.
    """
    if not regimes:
        raise ValueError("Need at least one regime to build the mosaic.")
    p_bbox = precip_bbox if precip_bbox is not None else z500_bbox
    raw_provided = z500_raw_by_regime is not None and any(
        z500_raw_by_regime.get(int(r)) is not None for r in regimes
    )

    ref_anom = next(iter(z500_anom_by_regime.values()))
    ref_std = next(iter(z500_std_by_regime.values()))
    ref_precip = next(iter(precip_by_regime.values()))
    lat_dim, lon_dim = spatial_dims(ref_anom)
    lat_dim_p, lon_dim_p = spatial_dims(ref_precip)
    z_units_in = str(getattr(ref_anom, "attrs", {}).get("units", ""))
    z_std_units_in = str(getattr(ref_std, "attrs", {}).get("units", ""))

    anom_loaded: dict[int, xr.DataArray] = {}
    std_loaded: dict[int, xr.DataArray] = {}
    pr_loaded: dict[int, xr.DataArray] = {}
    raw_loaded: dict[int, xr.DataArray | None] = {}
    for r in regimes:
        r_int = int(r)
        anom = subset_da_to_bbox_or_raise(
            z500_anom_by_regime[r_int].astype(np.float64).load(), z500_bbox
        )
        anom = wrap_lon_sort_lat_da(anom, lat_dim, lon_dim)
        units_in = str(getattr(anom, "attrs", {}).get("units", ""))
        anom_loaded[r_int] = (anom / G0) if is_geopotential_units(units_in) else anom

        std = subset_da_to_bbox_or_raise(
            z500_std_by_regime[r_int].astype(np.float64).load(), z500_bbox
        )
        std = wrap_lon_sort_lat_da(std, lat_dim, lon_dim)
        std_units_in = str(getattr(std, "attrs", {}).get("units", ""))
        std_loaded[r_int] = (std / G0) if is_geopotential_units(std_units_in) else std

        pr = subset_da_to_bbox_or_raise(
            precip_by_regime[r_int].astype(np.float64).load(), p_bbox
        )
        pr_loaded[r_int] = wrap_lon_sort_lat_da(pr, lat_dim_p, lon_dim_p)
        if raw_provided:
            raw_da = (z500_raw_by_regime or {}).get(r_int)
            if raw_da is None:
                raw_loaded[r_int] = None
            else:
                raw = subset_da_to_bbox_or_raise(raw_da.astype(np.float64).load(), z500_bbox)
                raw = wrap_lon_sort_lat_da(raw, lat_dim, lon_dim)
                ru = str(getattr(raw, "attrs", {}).get("units", ""))
                raw_loaded[r_int] = (raw / G0) if is_geopotential_units(ru) else raw
        else:
            raw_loaded[r_int] = None

    def _flat(da_list: list[xr.DataArray]) -> np.ndarray:
        return np.concatenate([np.asarray(da.values, dtype=float).ravel() for da in da_list])

    precip_cmap_name = "Blues"
    if z500_vmin is None or z500_vmax is None:
        vmin_z, vmax_z = _cmap_fill_limits(_flat(list(anom_loaded.values())), cmap_name="RdBu_r")
        z500_vmin = vmin_z if z500_vmin is None else z500_vmin
        z500_vmax = vmax_z if z500_vmax is None else z500_vmax

    _std_vmin_explicit = std_vmin is not None
    if std_vmin is None or std_vmax is None:
        _, vmax_s = _cmap_fill_limits(_flat(list(std_loaded.values())), cmap_name="YlGnBu")
        if std_vmax is None:
            std_vmax = vmax_s
    if not _std_vmin_explicit:
        std_vmin = 0.0
    if std_vmax is None or not (std_vmax > std_vmin):
        std_vmax = std_vmin + 1.0

    _precip_vmin_explicit = precip_vmin is not None
    if precip_vmin is None or precip_vmax is None:
        _, vmax_p = _cmap_fill_limits(_flat(list(pr_loaded.values())), cmap_name=precip_cmap_name)
        if precip_vmax is None:
            precip_vmax = vmax_p
    if not _precip_vmin_explicit:
        precip_vmin = 0.0
    if precip_vmax is None or not (precip_vmax > precip_vmin):
        precip_vmax = precip_vmin + 1.0

    n_rows = len(regimes)
    n_cols = 3
    fig_w = MOSAIC_PANEL_W_IN * n_cols + 1.4
    fig_h = MOSAIC_PANEL_H_IN * n_rows + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    _lat_z = np.asarray(ref_anom[lat_dim].values, dtype=float)
    _lon_z = np.asarray(ref_anom[lon_dim].values, dtype=float)
    map_extent_z = resolve_map_extent(z500_bbox, lat=_lat_z, lon=_lon_z)
    _lat_p = np.asarray(ref_precip[lat_dim_p].values, dtype=float)
    _lon_p = np.asarray(ref_precip[lon_dim_p].values, dtype=float)
    map_extent_p = resolve_map_extent(p_bbox, lat=_lat_p, lon=_lon_p)

    cmap_z, norm_z, extend_z = discretize_diverging(z500_vmin, z500_vmax, "RdBu_r")
    cmap_s, norm_s, extend_s = discretize_sequential_from_floor(std_vmax, "YlGnBu", vmin=std_vmin)
    cmap_p, norm_p, extend_p = discretize_sequential_from_floor(precip_vmax, precip_cmap_name, vmin=precip_vmin)
    im_z = im_s = im_p = None
    for row, r in enumerate(regimes):
        r_int = int(r)
        for col, (field, cmap, norm, ldim, odim, mextent) in enumerate(
            (
                (anom_loaded[r_int], cmap_z, norm_z, lat_dim, lon_dim, map_extent_z),
                (std_loaded[r_int], cmap_s, norm_s, lat_dim, lon_dim, map_extent_z),
                (pr_loaded[r_int], cmap_p, norm_p, lat_dim_p, lon_dim_p, map_extent_p),
            )
        ):
            ax = (
                fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1, projection=ccrs.PlateCarree())
                if use_geo
                else fig.add_subplot(n_rows, n_cols, row * n_cols + col + 1)
            )
            lon_m, lat_m, za = _mesh_2d(field, ldim, odim)
            if use_geo and cfeat is not None and pc is not None and mextent is not None:
                cartopy_underlay(ax, mextent, cfeature=cfeat)
                im = ax.pcolormesh(
                    lon_m, lat_m, za,
                    shading="auto", cmap=cmap, norm=norm,
                    alpha=0.85 if col == 0 else 1.0,
                    transform=pc, zorder=8, rasterized=True,
                )
                cs = None
                if col == 0 and raw_loaded.get(r_int) is not None:
                    _, _, zr = _mesh_2d(raw_loaded[r_int], lat_dim, lon_dim)
                    cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
                    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                        cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
                    levels_raw = np.linspace(cmin, cmax, 9)
                    cs = ax.contour(
                        lon_m, lat_m, zr,
                        levels=levels_raw, colors="k", linewidths=0.9,
                        alpha=0.95, transform=pc, zorder=12,
                    )
                cartopy_lines_on_top(ax, cfeature=cfeat)
                if cs is not None:
                    label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
                draw_aoi_box(ax, aoi_bbox, transform=pc)
                ax.set_extent(mextent, crs=pc)
                configure_regime_composite_gridlines(
                    ax,
                    panel_row=row,
                    panel_col=col,
                    nrow=n_rows,
                    ncol=n_cols,
                    linewidth=0.4,
                    alpha=0.3,
                    tick_labelsize=PDF_FS_FIG_TICK,
                )
            else:
                im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm)
                if col == 0 and raw_loaded.get(r_int) is not None:
                    _, _, zr = _mesh_2d(raw_loaded[r_int], lat_dim, lon_dim)
                    ax.contour(lon_m, lat_m, zr, colors="k", linewidths=0.9)
                draw_aoi_box(ax, aoi_bbox)
            if col == 0:
                im_z = im
                ax.set_ylabel(
                    regime_panel_title(
                        regime=r_int,
                        regime_case_counts=n_days_by_regime,
                        align_groups=align_groups,
                        canonical_id_by_regime=canonical_id_by_regime,
                        literature_abbr_by_short=literature_abbr_by_short,
                    ),
                    fontsize=PDF_FS_FIG_TITLE,
                    labelpad=14,
                )
            elif col == 1:
                im_s = im
            else:
                im_p = im

    axes_right, cax_rects = _mosaic_column_colorbar_layout_n(fig_h, n_rows, n_cols)
    fig.subplots_adjust(
        left=0.05,
        right=axes_right,
        top=1.0 - 0.6 / fig_h,
        bottom=0.06,
        hspace=0.08,
        wspace=0.12,
    )
    cb_labels = (_z500_fill_label(z_units_in), _z500_fill_label(z_std_units_in, std=True), precip_units)
    cb_extends = (extend_z, extend_s, extend_p)
    for im, rect, lbl, ext in zip((im_z, im_s, im_p), cax_rects, cb_labels, cb_extends):
        if im is None:
            continue
        cb = fig.colorbar(im, cax=fig.add_axes(rect), extend=ext)
        _style_colorbar_outline(cb)
        cb.set_label(lbl, fontsize=PDF_FS_FIG_AXIS, labelpad=8)
        cb.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    return fig


def build_figure_regime_z500_precip_mosaic_wide(
    *,
    regimes: list[int],
    z500_anom_by_regime: dict[int, xr.DataArray],
    z500_raw_by_regime: dict[int, xr.DataArray | None] | None,
    precip_by_regime: dict[int, xr.DataArray],
    z500_bbox: dict[str, float],
    precip_bbox: dict[str, float] | None = None,
    title_line: str,
    subtitle_z500: str | None = None,
    subtitle_precip: str | None = None,
    n_days_by_regime: dict[int, int] | None = None,
    precip_units: str = "% of regime days",
    z500_vmin: float | None = None,
    z500_vmax: float | None = None,
    precip_vmin: float | None = None,
    precip_vmax: float | None = None,
    precip_metric: str = "flag_frequency",
    aoi_bbox: dict[str, float] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    align_groups: list[dict] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """2×k wide mosaic: top row Z500, bottom row precip (matches EPE.7 wide layout).

    ``canonical_id_by_regime``: see ``build_figure_regime_z500_precip_mosaic``'s docstring --
    same cross-bbox-aligned physical-regime id, shown alongside the raw column label.
    """
    if not regimes:
        raise ValueError("Need at least one regime to build the mosaic.")
    p_bbox = precip_bbox if precip_bbox is not None else z500_bbox
    raw_provided = z500_raw_by_regime is not None and any(
        z500_raw_by_regime.get(int(r)) is not None for r in regimes
    )

    ref_anom = next(iter(z500_anom_by_regime.values()))
    ref_precip = next(iter(precip_by_regime.values()))
    lat_dim, lon_dim = spatial_dims(ref_anom)
    lat_dim_p, lon_dim_p = spatial_dims(ref_precip)
    z_units_in = str(getattr(ref_anom, "attrs", {}).get("units", ""))

    loaded: list[dict[str, object]] = []
    for r in regimes:
        r_int = int(r)
        anom = subset_da_to_bbox_or_raise(
            z500_anom_by_regime[r_int].astype(np.float64).load(), z500_bbox
        )
        anom = wrap_lon_sort_lat_da(anom, lat_dim, lon_dim)
        units_in = str(getattr(anom, "attrs", {}).get("units", ""))
        anom_gpm = (anom / G0) if is_geopotential_units(units_in) else anom
        pr = subset_da_to_bbox_or_raise(
            precip_by_regime[r_int].astype(np.float64).load(), p_bbox
        )
        pr = wrap_lon_sort_lat_da(pr, lat_dim_p, lon_dim_p)
        raw_m: xr.DataArray | None = None
        if raw_provided:
            raw_da = (z500_raw_by_regime or {}).get(r_int)
            if raw_da is not None:
                raw = subset_da_to_bbox_or_raise(raw_da.astype(np.float64).load(), z500_bbox)
                raw = wrap_lon_sort_lat_da(raw, lat_dim, lon_dim)
                ru = str(getattr(raw, "attrs", {}).get("units", ""))
                raw_m = (raw / G0) if is_geopotential_units(ru) else raw
        loaded.append(
            {
                "regime": r_int,
                "anom_gpm": anom_gpm,
                "precip": pr,
                "raw_m": raw_m,
            }
        )

    def _flat(da_list: list[xr.DataArray]) -> np.ndarray:
        return np.concatenate([np.asarray(da.values, dtype=float).ravel() for da in da_list])

    precip_cmap_name = _precip_cmap_for_metric(precip_metric)
    if z500_vmin is None or z500_vmax is None:
        vmin_z, vmax_z = _cmap_fill_limits(
            _flat([d["anom_gpm"] for d in loaded]),  # type: ignore[list-item]
            cmap_name="RdBu_r",
        )
        z500_vmin = vmin_z if z500_vmin is None else z500_vmin
        z500_vmax = vmax_z if z500_vmax is None else z500_vmax
    _precip_vmin_explicit = precip_vmin is not None
    if precip_vmin is None or precip_vmax is None:
        _, vmax_p = _cmap_fill_limits(
            _flat([d["precip"] for d in loaded]),  # type: ignore[list-item]
            cmap_name=precip_cmap_name,
        )
        if precip_vmax is None:
            precip_vmax = vmax_p
    if not _precip_vmin_explicit:
        precip_vmin = 0.0
    if precip_vmax is None or not (precip_vmax > precip_vmin):
        precip_vmax = precip_vmin + 1.0

    _lat_z = np.asarray(ref_anom[lat_dim].values, dtype=float)
    _lon_z = np.asarray(ref_anom[lon_dim].values, dtype=float)
    map_extent_z = resolve_map_extent(z500_bbox, lat=_lat_z, lon=_lon_z)
    _lat_p = np.asarray(ref_precip[lat_dim_p].values, dtype=float)
    _lon_p = np.asarray(ref_precip[lon_dim_p].values, dtype=float)
    map_extent_p = resolve_map_extent(p_bbox, lat=_lat_p, lon=_lon_p)

    n_cols = len(loaded)
    n_rows = 2
    fs_sup = _scaled_fig_font(PDF_FS_FIG_SUPTITLE)
    fs_title = _scaled_fig_font(PDF_FS_FIG_TITLE)
    fs_axis = _scaled_fig_font(PDF_FS_FIG_AXIS)
    fs_tick = _scaled_fig_font(PDF_FS_FIG_TICK)
    fs_clabel = _scaled_fig_font(PDF_FS_FIG_CLABEL)
    fig_w = CROSSBBOX_WIDE_TOP_PANEL_W_IN * n_cols + 2.8
    fig_h = CROSSBBOX_WIDE_TOP_PANEL_H_IN + CROSSBBOX_WIDE_BOTTOM_PANEL_H_IN + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    from matplotlib.gridspec import GridSpec

    cmap_z, norm_z, extend_z = discretize_diverging(z500_vmin, z500_vmax, "RdBu_r")
    cmap_p, norm_p, extend_p = discretize_sequential_from_floor(precip_vmax, precip_cmap_name, vmin=precip_vmin)
    im_z = im_p = None
    top_axes: list = []
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        height_ratios=[CROSSBBOX_WIDE_TOP_PANEL_H_IN, CROSSBBOX_WIDE_BOTTOM_PANEL_H_IN],
        hspace=0.08,
        wspace=0.10,
    )
    for col_idx, d in enumerate(loaded):
        r_int = int(d["regime"])
        for row_idx, (field, cmap, norm, ldim, odim, mextent, raw_key) in enumerate(
            (
                (d["anom_gpm"], cmap_z, norm_z, lat_dim, lon_dim, map_extent_z, "raw_m"),
                (d["precip"], cmap_p, norm_p, lat_dim_p, lon_dim_p, map_extent_p, None),
            )
        ):
            ax = (
                fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                if use_geo
                else fig.add_subplot(gs[row_idx, col_idx])
            )
            lon_m, lat_m, za = _mesh_2d(field, ldim, odim)  # type: ignore[arg-type]
            if use_geo and cfeat is not None and pc is not None and mextent is not None:
                cartopy_underlay(ax, mextent, cfeature=cfeat)
                im = ax.pcolormesh(
                    lon_m,
                    lat_m,
                    za,
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                    alpha=0.85 if row_idx == 0 else 1.0,
                    transform=pc,
                    zorder=8,
                    rasterized=True,
                )
                cs = None
                if row_idx == 0 and raw_key and d.get(raw_key) is not None:
                    _, _, zr = _mesh_2d(d[raw_key], lat_dim, lon_dim)  # type: ignore[arg-type]
                    cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
                    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                        cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
                    levels_raw = np.linspace(cmin, cmax, 9)
                    cs = ax.contour(
                        lon_m,
                        lat_m,
                        zr,
                        levels=levels_raw,
                        colors="k",
                        linewidths=0.9,
                        alpha=0.95,
                        transform=pc,
                        zorder=12,
                    )
                cartopy_lines_on_top(ax, cfeature=cfeat)
                if cs is not None:
                    label_contour_heights(ax, cs, fontsize=fs_clabel, fmt="%.0f")
                draw_aoi_box(ax, aoi_bbox, transform=pc)
                ax.set_extent(mextent, crs=pc)
                if use_geo:
                    ax.set_aspect("auto")
                configure_regime_composite_gridlines(
                    ax,
                    panel_row=row_idx,
                    panel_col=col_idx,
                    nrow=n_rows,
                    ncol=n_cols,
                    show_right=False,
                    linewidth=0.4,
                    alpha=0.28,
                    tick_labelsize=fs_tick,
                )
            else:
                im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm)
                if row_idx == 0 and raw_key and d.get(raw_key) is not None:
                    _, _, zr = _mesh_2d(d[raw_key], lat_dim, lon_dim)  # type: ignore[arg-type]
                    ax.contour(lon_m, lat_m, zr, colors="k", linewidths=0.9)
                draw_aoi_box(ax, aoi_bbox)

            if row_idx == 0:
                im_z = im
                top_axes.append(ax)
                ax.set_title(
                    regime_panel_title(
                        regime=r_int,
                        regime_case_counts=n_days_by_regime,
                        align_groups=align_groups,
                        canonical_id_by_regime=canonical_id_by_regime,
                        literature_abbr_by_short=literature_abbr_by_short,
                    ),
                    fontsize=fs_title,
                )
            else:
                im_p = im

            if col_idx == 0:
                head = subtitle_z500 if row_idx == 0 else subtitle_precip
                if head:
                    ax.set_ylabel(head, fontsize=fs_title, labelpad=14)

    axes_right = 0.86 if n_cols <= 4 else 0.84
    fig.subplots_adjust(
        left=0.05,
        right=axes_right,
        top=1.0 - 0.62 / fig_h,
        bottom=0.07,
    )
    _force_equal_axes_boxes(top_axes)
    cb_w = 0.012
    x_cb = axes_right + 0.02
    if im_z is not None:
        cb_z = fig.colorbar(im_z, cax=fig.add_axes([x_cb, 0.56, cb_w, 0.30]), extend=extend_z)
        _style_colorbar_outline(cb_z)
        cb_z.set_label(_z500_fill_label(z_units_in), fontsize=fs_axis)
        cb_z.ax.tick_params(labelsize=fs_tick)
    if im_p is not None:
        cb_p = fig.colorbar(im_p, cax=fig.add_axes([x_cb, 0.14, cb_w, 0.30]), extend=extend_p)
        _style_colorbar_outline(cb_p)
        cb_p.set_label(precip_units, fontsize=fs_axis)
        cb_p.ax.tick_params(labelsize=fs_tick)
    return fig


@dataclass
class CrossBboxRow:
    """One row in :func:`build_figure_z500_precip_mosaic_by_bbox`.

    Each row corresponds to ONE bounding box; the regime id is the regime in
    that bbox that maps to the shared physical regime being plotted.
    """

    bbox_step: str
    regime_id: int
    z500_anom: xr.DataArray
    precip: xr.DataArray
    z500_bbox: dict[str, float]
    precip_bbox: dict[str, float]
    z500_raw: xr.DataArray | None = None
    moisture: xr.DataArray | None = None
    n_days: int | None = None
    k: int | None = None


# Month order for the EPE-occurrence stacked bar in
# build_figure_z500_precip_mosaic_by_bbox_with_boxplot -- MJJAS, matching the "relative
# occurrence" panel convention elsewhere (regime_plotting/epe_figures.py).
_OCC_MONTH_LABELS = ("May", "Jun", "Jul", "Aug", "Sep")


def build_figure_z500_precip_mosaic_by_bbox_long(
    *,
    rows: list[CrossBboxRow],
    title_line: str,
    subtitle_z500: str | None = None,
    subtitle_precip: str | None = None,
    precip_units: str = "% of regime days",
    z500_vmin: float | None = None,
    z500_vmax: float | None = None,
    precip_vmin: float | None = None,
    precip_vmax: float | None = None,
    precip_metric: str = "flag_frequency",
    aoi_bbox: dict[str, float] | None = None,
    precip_extent_bbox: dict[str, float] | None = None,
) -> Figure:
    """nb×2 mosaic (original EPE.7 layout): one row per bbox, columns (Z500, precip)."""
    if not rows:
        raise ValueError("Need at least one row to build the cross-bbox mosaic.")

    shared_precip_extent: list[float] | None = None
    if precip_extent_bbox is not None:
        shared_precip_extent = resolve_map_extent(precip_extent_bbox)

    z_units_in = str(getattr(rows[0].z500_anom, "attrs", {}).get("units", ""))
    loaded: list[dict[str, object]] = []
    for r in rows:
        lat_dim, lon_dim = spatial_dims(r.z500_anom)
        lat_dim_p, lon_dim_p = spatial_dims(r.precip)
        anom = subset_da_to_bbox_or_raise(r.z500_anom.astype(np.float64).load(), r.z500_bbox)
        anom = wrap_lon_sort_lat_da(anom, lat_dim, lon_dim)
        z_units = str(getattr(anom, "attrs", {}).get("units", ""))
        anom_gpm = (anom / G0) if is_geopotential_units(z_units) else anom
        pr = subset_da_to_bbox_or_raise(r.precip.astype(np.float64).load(), r.precip_bbox)
        pr = wrap_lon_sort_lat_da(pr, lat_dim_p, lon_dim_p)
        raw_m: xr.DataArray | None = None
        if r.z500_raw is not None:
            raw = subset_da_to_bbox_or_raise(r.z500_raw.astype(np.float64).load(), r.z500_bbox)
            raw = wrap_lon_sort_lat_da(raw, lat_dim, lon_dim)
            ru = str(getattr(raw, "attrs", {}).get("units", ""))
            raw_m = (raw / G0) if is_geopotential_units(ru) else raw
        lat_extent_z = np.asarray(anom_gpm[lat_dim].values, dtype=float)
        lon_extent_z = np.asarray(anom_gpm[lon_dim].values, dtype=float)
        lat_extent_p = np.asarray(pr[lat_dim_p].values, dtype=float)
        lon_extent_p = np.asarray(pr[lon_dim_p].values, dtype=float)
        loaded.append(
            {
                "row": r,
                "anom_gpm": anom_gpm,
                "precip": pr,
                "raw_m": raw_m,
                "lat_dim": lat_dim,
                "lon_dim": lon_dim,
                "lat_dim_p": lat_dim_p,
                "lon_dim_p": lon_dim_p,
                "extent_z": resolve_map_extent(r.z500_bbox, lat=lat_extent_z, lon=lon_extent_z),
                "extent_p": (
                    shared_precip_extent
                    if shared_precip_extent is not None
                    else resolve_map_extent(r.precip_bbox, lat=lat_extent_p, lon=lon_extent_p)
                ),
            }
        )

    def _flat(arrs: list[xr.DataArray]) -> np.ndarray:
        return np.concatenate([np.asarray(a.values, dtype=float).ravel() for a in arrs])

    precip_cmap_name = _precip_cmap_for_metric(precip_metric)
    if z500_vmin is None or z500_vmax is None:
        vmin_z, vmax_z = _cmap_fill_limits(_flat([d["anom_gpm"] for d in loaded]), cmap_name="RdBu_r")
        z500_vmin = vmin_z if z500_vmin is None else z500_vmin
        z500_vmax = vmax_z if z500_vmax is None else z500_vmax
    _precip_vmin_explicit = precip_vmin is not None
    if precip_vmin is None or precip_vmax is None:
        _, vmax_p = _cmap_fill_limits(_flat([d["precip"] for d in loaded]), cmap_name=precip_cmap_name)
        if precip_vmax is None:
            precip_vmax = vmax_p
    if not _precip_vmin_explicit:
        precip_vmin = 0.0
    if precip_vmax is None or not (precip_vmax > precip_vmin):
        precip_vmax = precip_vmin + 1.0

    n_b = len(loaded)
    fig_w = MOSAIC_PANEL_W_IN * 2 + 2.1
    fig_h = MOSAIC_PANEL_H_IN * n_b + 1.2
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(title_line, fontsize=PDF_FS_FIG_SUPTITLE, y=1.0 - 0.18 / fig_h)

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    cmap_z, norm_z, extend_z = discretize_diverging(z500_vmin, z500_vmax, "RdBu_r")
    cmap_p, norm_p, extend_p = discretize_sequential_from_floor(precip_vmax, precip_cmap_name, vmin=precip_vmin)
    im_z = im_p = None
    for row_idx, d in enumerate(loaded):
        r: CrossBboxRow = d["row"]  # type: ignore[assignment]
        for col, (field, cmap, norm, ldim, odim, mextent) in enumerate(
            (
                (d["anom_gpm"], cmap_z, norm_z, d["lat_dim"], d["lon_dim"], d["extent_z"]),
                (d["precip"], cmap_p, norm_p, d["lat_dim_p"], d["lon_dim_p"], d["extent_p"]),
            )
        ):
            ax = (
                fig.add_subplot(n_b, 2, row_idx * 2 + col + 1, projection=ccrs.PlateCarree())
                if use_geo
                else fig.add_subplot(n_b, 2, row_idx * 2 + col + 1)
            )
            lon_m, lat_m, za = _mesh_2d(field, ldim, odim)  # type: ignore[arg-type]
            if use_geo and cfeat is not None and pc is not None and mextent is not None:
                cartopy_underlay(ax, mextent, cfeature=cfeat)
                im = ax.pcolormesh(
                    lon_m,
                    lat_m,
                    za,
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                    alpha=0.85 if col == 0 else 1.0,
                    transform=pc,
                    zorder=8,
                    rasterized=True,
                )
                cs = None
                if col == 0 and d.get("raw_m") is not None:
                    _, _, zr = _mesh_2d(d["raw_m"], d["lat_dim"], d["lon_dim"])  # type: ignore[arg-type]
                    cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
                    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                        cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
                    levels_raw = np.linspace(cmin, cmax, 9)
                    cs = ax.contour(
                        lon_m,
                        lat_m,
                        zr,
                        levels=levels_raw,
                        colors="k",
                        linewidths=0.9,
                        alpha=0.95,
                        transform=pc,
                        zorder=12,
                    )
                cartopy_lines_on_top(ax, cfeature=cfeat)
                if cs is not None:
                    label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
                draw_aoi_box(ax, aoi_bbox, transform=pc)
                ax.set_extent(mextent, crs=pc)
                configure_regime_composite_gridlines(
                    ax,
                    panel_row=row_idx,
                    panel_col=col,
                    nrow=n_b,
                    ncol=2,
                    show_right=False,
                    linewidth=0.4,
                    alpha=0.28,
                    tick_labelsize=PDF_FS_FIG_TICK,
                )
            else:
                im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm)
                if col == 0 and d.get("raw_m") is not None:
                    _, _, zr = _mesh_2d(d["raw_m"], d["lat_dim"], d["lon_dim"])  # type: ignore[arg-type]
                    ax.contour(lon_m, lat_m, zr, colors="k", linewidths=0.9)
                draw_aoi_box(ax, aoi_bbox)
            if col == 0:
                im_z = im
                k_lbl = f" k={int(r.k)}" if r.k is not None else ""
                n_lbl = f"  n={int(r.n_days)}" if r.n_days is not None else ""
                ax.set_ylabel(
                    f"{r.bbox_step}{k_lbl}\nregime {int(r.regime_id)}{n_lbl}",
                    fontsize=PDF_FS_FIG_TITLE,
                    labelpad=14,
                )
            else:
                im_p = im
            if row_idx == 0:
                head = subtitle_z500 if col == 0 else subtitle_precip
                if head:
                    ax.set_title(head, fontsize=PDF_FS_FIG_TITLE)

    axes_right, cax_z_rect, cax_p_rect = _mosaic_column_colorbar_layout(fig_h, n_b)
    fig.subplots_adjust(
        left=0.07,
        right=axes_right,
        top=1.0 - 0.55 / fig_h,
        bottom=0.07,
        hspace=0.08,
        wspace=0.12,
    )
    if im_z is not None:
        cb_z = fig.colorbar(im_z, cax=fig.add_axes(cax_z_rect), extend=extend_z)
        _style_colorbar_outline(cb_z)
        cb_z.set_label(_z500_fill_label(z_units_in), fontsize=PDF_FS_FIG_AXIS, labelpad=8)
        cb_z.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    if im_p is not None:
        cb_p = fig.colorbar(im_p, cax=fig.add_axes(cax_p_rect), extend=extend_p)
        _style_colorbar_outline(cb_p)
        cb_p.set_label(precip_units, fontsize=PDF_FS_FIG_AXIS, labelpad=8)
        cb_p.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    return fig


def build_figure_z500_precip_mosaic_by_bbox(
    *,
    rows: list[CrossBboxRow],
    title_line: str,
    subtitle_z500: str | None = None,
    subtitle_precip: str | None = None,
    precip_units: str = "% of regime days",
    z500_vmin: float | None = None,
    z500_vmax: float | None = None,
    precip_vmin: float | None = None,
    precip_vmax: float | None = None,
    precip_metric: str = "flag_frequency",
    aoi_bbox: dict[str, float] | None = None,
    precip_extent_bbox: dict[str, float] | None = None,
) -> Figure:
    """2×nb wide mosaic for one physical regime across bounding boxes.

    Layout:
    - Top row: Z500 anomaly composite for each bbox step (each panel has its own map extent).
    - Bottom row: precip / flag-frequency composite (shared study-area extent when
      ``precip_extent_bbox`` is provided).

    Color scales:
    - Shared colorbar for Z500 (top row) and shared colorbar for precip (bottom row).
    - Colorbars are placed in a dedicated right margin to avoid overlap with maps.
    """
    if not rows:
        raise ValueError("Need at least one row to build the cross-bbox mosaic.")

    shared_precip_extent: list[float] | None = None
    if precip_extent_bbox is not None:
        shared_precip_extent = resolve_map_extent(precip_extent_bbox)

    z_units_in = str(getattr(rows[0].z500_anom, "attrs", {}).get("units", ""))
    loaded: list[dict[str, object]] = []
    for r in rows:
        lat_dim, lon_dim = spatial_dims(r.z500_anom)
        lat_dim_p, lon_dim_p = spatial_dims(r.precip)
        anom = subset_da_to_bbox_or_raise(r.z500_anom.astype(np.float64).load(), r.z500_bbox)
        anom = wrap_lon_sort_lat_da(anom, lat_dim, lon_dim)
        z_units = str(getattr(anom, "attrs", {}).get("units", ""))
        anom_gpm = (anom / G0) if is_geopotential_units(z_units) else anom
        pr = subset_da_to_bbox_or_raise(r.precip.astype(np.float64).load(), r.precip_bbox)
        pr = wrap_lon_sort_lat_da(pr, lat_dim_p, lon_dim_p)
        raw_m: xr.DataArray | None = None
        if r.z500_raw is not None:
            raw = subset_da_to_bbox_or_raise(r.z500_raw.astype(np.float64).load(), r.z500_bbox)
            raw = wrap_lon_sort_lat_da(raw, lat_dim, lon_dim)
            ru = str(getattr(raw, "attrs", {}).get("units", ""))
            raw_m = (raw / G0) if is_geopotential_units(ru) else raw
        lat_extent_z = np.asarray(anom_gpm[lat_dim].values, dtype=float)
        lon_extent_z = np.asarray(anom_gpm[lon_dim].values, dtype=float)
        lat_extent_p = np.asarray(pr[lat_dim_p].values, dtype=float)
        lon_extent_p = np.asarray(pr[lon_dim_p].values, dtype=float)
        loaded.append(
            {
                "row": r,
                "anom_gpm": anom_gpm,
                "precip": pr,
                "raw_m": raw_m,
                "lat_dim": lat_dim,
                "lon_dim": lon_dim,
                "lat_dim_p": lat_dim_p,
                "lon_dim_p": lon_dim_p,
                "extent_z": resolve_map_extent(r.z500_bbox, lat=lat_extent_z, lon=lon_extent_z),
                "extent_p": (
                    shared_precip_extent
                    if shared_precip_extent is not None
                    else resolve_map_extent(r.precip_bbox, lat=lat_extent_p, lon=lon_extent_p)
                ),
            }
        )

    def _flat(arrs: list[xr.DataArray]) -> np.ndarray:
        return np.concatenate([np.asarray(a.values, dtype=float).ravel() for a in arrs])

    precip_cmap_name = _precip_cmap_for_metric(precip_metric)
    if z500_vmin is None or z500_vmax is None:
        vmin_z, vmax_z = _cmap_fill_limits(_flat([d["anom_gpm"] for d in loaded]), cmap_name="RdBu_r")
        z500_vmin = vmin_z if z500_vmin is None else z500_vmin
        z500_vmax = vmax_z if z500_vmax is None else z500_vmax
    _precip_vmin_explicit = precip_vmin is not None
    if precip_vmin is None or precip_vmax is None:
        _, vmax_p = _cmap_fill_limits(_flat([d["precip"] for d in loaded]), cmap_name=precip_cmap_name)
        if precip_vmax is None:
            precip_vmax = vmax_p
    if not _precip_vmin_explicit:
        precip_vmin = 0.0
    if precip_vmax is None or not (precip_vmax > precip_vmin):
        precip_vmax = precip_vmin + 1.0

    n_cols = len(loaded)
    n_rows = 2
    fs_sup = _scaled_fig_font(PDF_FS_FIG_SUPTITLE)
    fs_title = _scaled_fig_font(PDF_FS_FIG_TITLE)
    fs_axis = _scaled_fig_font(PDF_FS_FIG_AXIS)
    fs_tick = _scaled_fig_font(PDF_FS_FIG_TICK)
    fs_clabel = _scaled_fig_font(PDF_FS_FIG_CLABEL)
    # Wide layout: larger top row (Z500); reserve a right margin for colorbars.
    fig_w = CROSSBBOX_WIDE_TOP_PANEL_W_IN * n_cols + 2.8
    fig_h = CROSSBBOX_WIDE_TOP_PANEL_H_IN + CROSSBBOX_WIDE_BOTTOM_PANEL_H_IN + 1.4
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    fig.suptitle(title_line, fontsize=fs_sup, y=1.0 - 0.18 / fig_h)

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    from matplotlib.gridspec import GridSpec

    cmap_z, norm_z, extend_z = discretize_diverging(z500_vmin, z500_vmax, "RdBu_r")
    cmap_p, norm_p, extend_p = discretize_sequential_from_floor(precip_vmax, precip_cmap_name, vmin=precip_vmin)
    im_z = im_p = None
    top_axes: list = []
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        height_ratios=[CROSSBBOX_WIDE_TOP_PANEL_H_IN, CROSSBBOX_WIDE_BOTTOM_PANEL_H_IN],
        hspace=0.08,
        wspace=0.10,
    )
    for col_idx, d in enumerate(loaded):
        r: CrossBboxRow = d["row"]  # type: ignore[assignment]
        # Top row: Z500 anomaly (+ optional raw contours)
        for row_idx, (field, cmap, norm, ldim, odim, mextent, raw_key) in enumerate(
            (
                (d["anom_gpm"], cmap_z, norm_z, d["lat_dim"], d["lon_dim"], d["extent_z"], "raw_m"),
                (d["precip"], cmap_p, norm_p, d["lat_dim_p"], d["lon_dim_p"], d["extent_p"], None),
            )
        ):
            ax = (
                fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                if use_geo
                else fig.add_subplot(gs[row_idx, col_idx])
            )
            lon_m, lat_m, za = _mesh_2d(field, ldim, odim)  # type: ignore[arg-type]
            if use_geo and cfeat is not None and pc is not None and mextent is not None:
                cartopy_underlay(ax, mextent, cfeature=cfeat)
                im = ax.pcolormesh(
                    lon_m,
                    lat_m,
                    za,
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                    alpha=0.85 if row_idx == 0 else 1.0,
                    transform=pc,
                    zorder=8,
                    rasterized=True,
                )
                cs = None
                if row_idx == 0 and raw_key and d.get(raw_key) is not None:
                    _, _, zr = _mesh_2d(d[raw_key], d["lat_dim"], d["lon_dim"])  # type: ignore[arg-type]
                    cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
                    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                        cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
                    levels_raw = np.linspace(cmin, cmax, 9)
                    cs = ax.contour(
                        lon_m,
                        lat_m,
                        zr,
                        levels=levels_raw,
                        colors="k",
                        linewidths=0.9,
                        alpha=0.95,
                        transform=pc,
                        zorder=12,
                    )
                cartopy_lines_on_top(ax, cfeature=cfeat)
                if cs is not None:
                    label_contour_heights(ax, cs, fontsize=fs_clabel, fmt="%.0f")
                draw_aoi_box(ax, aoi_bbox, transform=pc)
                ax.set_extent(mextent, crs=pc)
                if use_geo:
                    ax.set_aspect("auto")
                configure_regime_composite_gridlines(
                    ax,
                    panel_row=row_idx,
                    panel_col=col_idx,
                    nrow=n_rows,
                    ncol=n_cols,
                    show_right=False,
                    linewidth=0.4,
                    alpha=0.28,
                    tick_labelsize=fs_tick,
                )
            else:
                im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm)
                if row_idx == 0 and raw_key and d.get(raw_key) is not None:
                    _, _, zr = _mesh_2d(d[raw_key], d["lat_dim"], d["lon_dim"])  # type: ignore[arg-type]
                    ax.contour(lon_m, lat_m, zr, colors="k", linewidths=0.9)
                draw_aoi_box(ax, aoi_bbox)

            if row_idx == 0:
                im_z = im
                top_axes.append(ax)
                # Column title: bbox/regime info per panel
                k_lbl = f"k={int(r.k)}" if r.k is not None else ""
                n_lbl = f"n={int(r.n_days)}" if r.n_days is not None else ""
                bits = " | ".join(b for b in (r.bbox_step, k_lbl, f"regime {int(r.regime_id)}", n_lbl) if b)
                ax.set_title(bits, fontsize=fs_title, pad=8)
            else:
                im_p = im

            if col_idx == 0:
                head = subtitle_z500 if row_idx == 0 else subtitle_precip
                if head:
                    ax.set_ylabel(head, fontsize=fs_title, labelpad=14)

    # Reserve a generous right margin for colorbars (wide layout).
    axes_right = 0.86 if n_cols <= 4 else 0.84
    fig.subplots_adjust(
        left=0.05,
        right=axes_right,
        top=1.0 - 0.62 / fig_h,
        bottom=0.07,
    )
    _force_equal_axes_boxes(top_axes)
    cb_w = 0.012
    x_cb = axes_right + 0.02
    if im_z is not None:
        cb_z = fig.colorbar(im_z, cax=fig.add_axes([x_cb, 0.56, cb_w, 0.30]), extend=extend_z)
        _style_colorbar_outline(cb_z)
        cb_z.set_label(_z500_fill_label(z_units_in), fontsize=fs_axis)
        cb_z.ax.tick_params(labelsize=fs_tick)
    if im_p is not None:
        cb_p = fig.colorbar(im_p, cax=fig.add_axes([x_cb, 0.14, cb_w, 0.30]), extend=extend_p)
        _style_colorbar_outline(cb_p)
        cb_p.set_label(precip_units, fontsize=fs_axis)
        cb_p.ax.tick_params(labelsize=fs_tick)
    return fig


def build_figure_z500_precip_mosaic_by_bbox_with_boxplot(
    *,
    rows: list[CrossBboxRow],
    boxplot_values: dict[str, np.ndarray],
    boxplot_title: str,
    boxplot_ylabel: str,
    occurrence_month_fracs: dict[str, dict[str, float]],
    occurrence_n: dict[str, int],
    occurrence_title: str = "EPE occurrence (month composition, extreme days)",
    occurrence_ylabel: str = "fraction of extreme days",
    subtitle_z500: str | None = None,
    subtitle_precip: str | None = None,
    precip_units: str = "% of regime days",
    z500_vmin: float | None = None,
    z500_vmax: float | None = None,
    precip_vmin: float | None = None,
    precip_vmax: float | None = None,
    precip_metric: str = "flag_frequency",
    aoi_bbox: dict[str, float] | None = None,
    precip_extent_bbox: dict[str, float] | None = None,
    box_color=None,
) -> Figure:
    """2×(nb+1) wide mosaic: same top (Z500)/bottom (precip) map rows as
    :func:`build_figure_z500_precip_mosaic_by_bbox`, one column per bbox step, plus one extra
    column with two stacked stats panels instead of a single boxplot:

      - top (aligned with the Z500 row): stacked bar of ``occurrence_month_fracs`` -- the
        MJJAS month composition of this regime's extreme days, one bar per bbox step.
      - bottom (aligned with the precip row): boxplot of ``boxplot_values`` (e.g. daily max
        precip on flagged cells) per bbox step.

    The two panels share the bbox-step x-axis (only the bottom one draws tick labels). Lets a
    regime that "stands out" more strongly in one bbox size than another show up right next to
    its own maps, in the same figure, instead of needing a separate cross-reference.

    ``boxplot_values`` / ``occurrence_month_fracs`` / ``occurrence_n`` keys must match ``rows``'
    ``bbox_step`` values; a missing/empty key renders as empty for that step rather than raising.
    """
    if not rows:
        raise ValueError("Need at least one row to build the cross-bbox mosaic.")

    shared_precip_extent: list[float] | None = None
    if precip_extent_bbox is not None:
        shared_precip_extent = resolve_map_extent(precip_extent_bbox)

    z_units_in = str(getattr(rows[0].z500_anom, "attrs", {}).get("units", ""))
    loaded: list[dict[str, object]] = []
    for r in rows:
        lat_dim, lon_dim = spatial_dims(r.z500_anom)
        lat_dim_p, lon_dim_p = spatial_dims(r.precip)
        anom = subset_da_to_bbox_or_raise(r.z500_anom.astype(np.float64).load(), r.z500_bbox)
        anom = wrap_lon_sort_lat_da(anom, lat_dim, lon_dim)
        z_units = str(getattr(anom, "attrs", {}).get("units", ""))
        anom_gpm = (anom / G0) if is_geopotential_units(z_units) else anom
        pr = subset_da_to_bbox_or_raise(r.precip.astype(np.float64).load(), r.precip_bbox)
        pr = wrap_lon_sort_lat_da(pr, lat_dim_p, lon_dim_p)
        raw_m: xr.DataArray | None = None
        if r.z500_raw is not None:
            raw = subset_da_to_bbox_or_raise(r.z500_raw.astype(np.float64).load(), r.z500_bbox)
            raw = wrap_lon_sort_lat_da(raw, lat_dim, lon_dim)
            ru = str(getattr(raw, "attrs", {}).get("units", ""))
            raw_m = (raw / G0) if is_geopotential_units(ru) else raw
        lat_extent_z = np.asarray(anom_gpm[lat_dim].values, dtype=float)
        lon_extent_z = np.asarray(anom_gpm[lon_dim].values, dtype=float)
        lat_extent_p = np.asarray(pr[lat_dim_p].values, dtype=float)
        lon_extent_p = np.asarray(pr[lon_dim_p].values, dtype=float)
        loaded.append(
            {
                "row": r,
                "anom_gpm": anom_gpm,
                "precip": pr,
                "raw_m": raw_m,
                "lat_dim": lat_dim,
                "lon_dim": lon_dim,
                "lat_dim_p": lat_dim_p,
                "lon_dim_p": lon_dim_p,
                "extent_z": resolve_map_extent(r.z500_bbox, lat=lat_extent_z, lon=lon_extent_z),
                "extent_p": (
                    shared_precip_extent
                    if shared_precip_extent is not None
                    else resolve_map_extent(r.precip_bbox, lat=lat_extent_p, lon=lon_extent_p)
                ),
            }
        )

    def _flat(arrs: list[xr.DataArray]) -> np.ndarray:
        return np.concatenate([np.asarray(a.values, dtype=float).ravel() for a in arrs])

    precip_cmap_name = _precip_cmap_for_metric(precip_metric)
    if z500_vmin is None or z500_vmax is None:
        vmin_z, vmax_z = _cmap_fill_limits(_flat([d["anom_gpm"] for d in loaded]), cmap_name="RdBu_r")
        z500_vmin = vmin_z if z500_vmin is None else z500_vmin
        z500_vmax = vmax_z if z500_vmax is None else z500_vmax
    _precip_vmin_explicit = precip_vmin is not None
    if precip_vmin is None or precip_vmax is None:
        _, vmax_p = _cmap_fill_limits(_flat([d["precip"] for d in loaded]), cmap_name=precip_cmap_name)
        if precip_vmax is None:
            precip_vmax = vmax_p
    if not _precip_vmin_explicit:
        precip_vmin = 0.0
    if precip_vmax is None or not (precip_vmax > precip_vmin):
        precip_vmax = precip_vmin + 1.0

    n_maps = len(loaded)
    n_rows = 2
    # Single font size for every text element in this figure (titles, axis labels, ticks,
    # colorbar labels, boxplot) -- no per-element size tiers here.
    fs_all = _scaled_fig_font(PDF_FS_FIG_TITLE)
    fs_title = fs_axis = fs_tick = fs_clabel = fs_all
    box_col_w_in = CROSSBBOX_WIDE_TOP_PANEL_W_IN * 0.7
    cbar_col_w_in = 0.55
    spacer_col_w_in = 1.15
    fig_w = CROSSBBOX_WIDE_TOP_PANEL_W_IN * n_maps + cbar_col_w_in + spacer_col_w_in + box_col_w_in + 2.0
    fig_h = CROSSBBOX_WIDE_TOP_PANEL_H_IN + CROSSBBOX_WIDE_BOTTOM_PANEL_H_IN + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    from matplotlib.gridspec import GridSpec

    cmap_z, norm_z, extend_z = discretize_diverging(z500_vmin, z500_vmax, "RdBu_r")
    cmap_p, norm_p, extend_p = discretize_sequential_from_floor(precip_vmax, precip_cmap_name, vmin=precip_vmin)
    im_z = im_p = None
    top_axes: list = []
    bottom_axes: list = []
    cbar_col = n_maps
    spacer_col = n_maps + 1
    box_col = n_maps + 2
    n_cols = n_maps + 3
    width_ratios = (
        [1.0] * n_maps
        + [cbar_col_w_in / CROSSBBOX_WIDE_TOP_PANEL_W_IN]
        + [spacer_col_w_in / CROSSBBOX_WIDE_TOP_PANEL_W_IN]
        + [box_col_w_in / CROSSBBOX_WIDE_TOP_PANEL_W_IN]
    )
    gs = GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        width_ratios=width_ratios,
        height_ratios=[CROSSBBOX_WIDE_TOP_PANEL_H_IN, CROSSBBOX_WIDE_BOTTOM_PANEL_H_IN],
        hspace=0.08,
        wspace=0.16,
    )
    for col_idx, d in enumerate(loaded):
        r: CrossBboxRow = d["row"]  # type: ignore[assignment]
        # Top row: Z500 anomaly (+ optional raw contours)
        for row_idx, (field, cmap, norm, ldim, odim, mextent, raw_key) in enumerate(
            (
                (d["anom_gpm"], cmap_z, norm_z, d["lat_dim"], d["lon_dim"], d["extent_z"], "raw_m"),
                (d["precip"], cmap_p, norm_p, d["lat_dim_p"], d["lon_dim_p"], d["extent_p"], None),
            )
        ):
            ax = (
                fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                if use_geo
                else fig.add_subplot(gs[row_idx, col_idx])
            )
            lon_m, lat_m, za = _mesh_2d(field, ldim, odim)  # type: ignore[arg-type]
            if use_geo and cfeat is not None and pc is not None and mextent is not None:
                cartopy_underlay(ax, mextent, cfeature=cfeat)
                im = ax.pcolormesh(
                    lon_m,
                    lat_m,
                    za,
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                    alpha=0.85 if row_idx == 0 else 1.0,
                    transform=pc,
                    zorder=8,
                    rasterized=True,
                )
                cs = None
                if row_idx == 0 and raw_key and d.get(raw_key) is not None:
                    _, _, zr = _mesh_2d(d[raw_key], d["lat_dim"], d["lon_dim"])  # type: ignore[arg-type]
                    cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
                    if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                        cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
                    levels_raw = np.linspace(cmin, cmax, 9)
                    cs = ax.contour(
                        lon_m,
                        lat_m,
                        zr,
                        levels=levels_raw,
                        colors="k",
                        linewidths=0.9,
                        alpha=0.95,
                        transform=pc,
                        zorder=12,
                    )
                cartopy_lines_on_top(ax, cfeature=cfeat)
                if cs is not None:
                    label_contour_heights(ax, cs, fontsize=fs_clabel, fmt="%.0f")
                draw_aoi_box(ax, aoi_bbox, transform=pc)
                ax.set_extent(mextent, crs=pc)
                if use_geo:
                    ax.set_aspect("auto")
                configure_regime_composite_gridlines(
                    ax,
                    panel_row=row_idx,
                    panel_col=col_idx,
                    nrow=n_rows,
                    ncol=n_maps,
                    show_right=False,
                    linewidth=0.4,
                    alpha=0.28,
                    tick_labelsize=fs_tick,
                )
            else:
                im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm)
                if row_idx == 0 and raw_key and d.get(raw_key) is not None:
                    _, _, zr = _mesh_2d(d[raw_key], d["lat_dim"], d["lon_dim"])  # type: ignore[arg-type]
                    ax.contour(lon_m, lat_m, zr, colors="k", linewidths=0.9)
                draw_aoi_box(ax, aoi_bbox)

            if row_idx == 0:
                im_z = im
                top_axes.append(ax)
            else:
                im_p = im
                bottom_axes.append(ax)

            if col_idx == 0:
                head = subtitle_z500 if row_idx == 0 else subtitle_precip
                if head:
                    ax.set_ylabel(head, fontsize=fs_title, labelpad=14)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.08)
    _force_equal_axes_boxes(top_axes)
    _force_equal_axes_boxes(bottom_axes)

    # Colorbars sit directly right of the composite maps, before the boxplot column. Extra
    # labelpad keeps the rotated label clear of the boxplot's y-label across the spacer column.
    if im_z is not None:
        cb_z = fig.colorbar(im_z, cax=fig.add_subplot(gs[0, cbar_col]), extend=extend_z)
        _style_colorbar_outline(cb_z)
        cb_z.set_label(_z500_fill_label(z_units_in), fontsize=fs_axis, labelpad=10)
        cb_z.ax.tick_params(labelsize=fs_tick)
    if im_p is not None:
        cb_p = fig.colorbar(im_p, cax=fig.add_subplot(gs[1, cbar_col]), extend=extend_p)
        _style_colorbar_outline(cb_p)
        cb_p.set_label(precip_units, fontsize=fs_axis, labelpad=10)
        cb_p.ax.tick_params(labelsize=fs_tick)

    # Right column: two stats panels instead of one boxplot spanning both rows -- (top) EPE
    # occurrence stacked bar aligned with the Z500 map row, (bottom) precip boxplot aligned
    # with the precip map row. Both share the same bbox-step categories on the x-axis; only
    # the bottom panel draws tick labels.
    box_steps = [str(r.bbox_step) for r in rows]
    x_pos = np.arange(1, len(box_steps) + 1)  # matches ax.boxplot()'s default positions (1..N)

    ax_box_bottom = fig.add_subplot(gs[1, box_col])
    ax_box_top = fig.add_subplot(gs[0, box_col], sharex=ax_box_bottom)

    blues = plt.cm.Blues(np.linspace(0.38, 0.92, len(_OCC_MONTH_LABELS)))
    occ_bottom = np.zeros(len(box_steps))
    for mi, month_lbl in enumerate(_OCC_MONTH_LABELS):
        vals = np.array(
            [float(occurrence_month_fracs.get(step, {}).get(month_lbl, 0.0)) for step in box_steps],
            dtype=float,
        )
        ax_box_top.bar(x_pos, vals, bottom=occ_bottom, color=blues[mi], label=month_lbl, width=0.65)
        occ_bottom = occ_bottom + vals
    # Headroom above the (fraction-of-1) stacked bars for the n= labels, so they sit inside
    # the axes below the title instead of floating over it.
    ax_box_top.set_ylim(0, 1.14)
    for xi, step in zip(x_pos, box_steps):
        n = occurrence_n.get(step)
        if n is not None:
            ax_box_top.text(
                xi, 1.03, f"n={n}", ha="center", va="bottom", fontsize=max(fs_all - 3, 6),
            )
    ax_box_top.set_ylabel(occurrence_ylabel, fontsize=fs_all, labelpad=8)
    ax_box_top.grid(True, axis="y", alpha=0.25)
    ax_box_top.tick_params(axis="both", labelsize=fs_all)
    ax_box_top.legend(
        title="month", fontsize=max(fs_all - 2, 6), title_fontsize=max(fs_all - 2, 6),
        loc="upper right", framealpha=0.92, ncol=1,
    )
    plt.setp(ax_box_top.get_xticklabels(), visible=False)

    box_data = [np.asarray(boxplot_values.get(step, []), dtype=float) for step in box_steps]
    box_colors = [box_color] * len(box_steps) if box_color is not None else None
    style_boxplot(
        ax_box_bottom,
        box_data,
        box_steps,
        title=boxplot_title,
        ylabel=boxplot_ylabel,
        xlabel="",
        colors=box_colors,
        tier="panel",
    )
    ax_box_bottom.title.set_fontsize(fs_all)
    ax_box_bottom.xaxis.label.set_fontsize(fs_all)
    ax_box_bottom.yaxis.label.set_fontsize(fs_all)
    ax_box_bottom.tick_params(axis="both", labelsize=fs_all)

    # Pin each stats panel's vertical span to the ACTUAL (post-equalization) map row extent
    # rather than the nominal gridspec cell -- Cartopy can still shrink a GeoAxes within its
    # cell even with aspect="auto", which would otherwise leave a gap between a stats panel
    # and its corresponding map row.
    if top_axes:
        top_y0 = min(a.get_position().y0 for a in top_axes)
        top_y1 = max(a.get_position().y1 for a in top_axes)
        pos = ax_box_top.get_position()
        ax_box_top.set_position([pos.x0, top_y0, pos.width, top_y1 - top_y0])
    if bottom_axes:
        bottom_y0 = min(a.get_position().y0 for a in bottom_axes)
        bottom_y1 = max(a.get_position().y1 for a in bottom_axes)
        pos = ax_box_bottom.get_position()
        ax_box_bottom.set_position([pos.x0, bottom_y0, pos.width, bottom_y1 - bottom_y0])

    return fig


def _mv3row_field_rows(field_idx: int) -> tuple[int, int, int]:
    """Outer GridSpec row indices for map / map–cbar pad / colorbar (field_idx 0..2)."""
    base = int(field_idx) * 4
    return base, base + 1, base + 2


def mv3row_figure_height_in(*, pad_top: float = 1.0) -> float:
    """Total figure height (inches) for the 3-field mv3row layout."""
    block = MV3_ROW_MAP_H_IN + MV3_ROW_MAP_PAD_IN + MV3_ROW_CBAR_H_IN + MV3_ROW_GAP_IN
    last = MV3_ROW_MAP_H_IN + MV3_ROW_MAP_PAD_IN + MV3_ROW_CBAR_H_IN
    return block * 2 + last + pad_top


def _wide_three_row_gridspec(
    fig: Figure,
    n_cols: int,
    *,
    font_scale: float = CROSSBBOX_WIDE_FONT_SCALE,
) -> tuple[object, float, float, float, float]:
    """GridSpec: per field (map, pad, cbar, gap) × 3; last gap omitted."""
    from matplotlib.gridspec import GridSpec

    block = [MV3_ROW_MAP_H_IN, MV3_ROW_MAP_PAD_IN, MV3_ROW_CBAR_H_IN, MV3_ROW_GAP_IN]
    height_ratios = block + block + block[:3]
    gs = GridSpec(
        len(height_ratios),
        n_cols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.08,
        wspace=0.10,
        left=0.05,
        right=0.98,
        top=0.92,
        bottom=0.06,
    )
    fs_title = _scaled_fig_font(PDF_FS_FIG_TITLE, scale=font_scale)
    fs_axis = _scaled_fig_font(PDF_FS_FIG_AXIS, scale=font_scale)
    fs_tick = _scaled_fig_font(PDF_FS_FIG_TICK, scale=font_scale)
    fs_clabel = _scaled_fig_font(PDF_FS_FIG_CLABEL, scale=font_scale)
    return gs, fs_title, fs_axis, fs_tick, fs_clabel


def _draw_wide_map_panel(
    fig: Figure,
    gs: object,
    row_map: int,
    col_idx: int,
    field: xr.DataArray,
    *,
    cmap,
    norm,
    lat_dim: str,
    lon_dim: str,
    map_extent,
    use_geo: bool,
    ccrs,
    pc,
    cfeat,
    aoi_bbox: dict[str, float] | None,
    raw_m: xr.DataArray | None,
    raw_lat_dim: str,
    raw_lon_dim: str,
    panel_row: int,
    panel_col: int,
    nrow_maps: int,
    ncol: int,
    fs_tick: float,
    fs_clabel: float,
    alpha: float = 1.0,
    show_bottom_labels: bool = False,
) -> tuple[object, object]:
    ax = (
        fig.add_subplot(gs[row_map, col_idx], projection=ccrs.PlateCarree())
        if use_geo
        else fig.add_subplot(gs[row_map, col_idx])
    )
    lon_m, lat_m, za = _mesh_2d(field, lat_dim, lon_dim)
    if use_geo and cfeat is not None and pc is not None and map_extent is not None:
        cartopy_underlay(ax, map_extent, cfeature=cfeat)
        im = ax.pcolormesh(
            lon_m,
            lat_m,
            za,
            shading="auto",
            cmap=cmap,
            norm=norm,
            alpha=alpha,
            transform=pc,
            zorder=8,
            rasterized=True,
        )
        cs = None
        if raw_m is not None:
            _, _, zr = _mesh_2d(raw_m, raw_lat_dim, raw_lon_dim)
            cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
            if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
            levels_raw = np.linspace(cmin, cmax, 9)
            cs = ax.contour(
                lon_m,
                lat_m,
                zr,
                levels=levels_raw,
                colors="k",
                linewidths=0.9,
                alpha=0.95,
                transform=pc,
                zorder=12,
            )
        cartopy_lines_on_top(ax, cfeature=cfeat)
        if cs is not None:
            label_contour_heights(ax, cs, fontsize=fs_clabel, fmt="%.0f")
        draw_aoi_box(ax, aoi_bbox, transform=pc)
        ax.set_extent(map_extent, crs=pc)
        ax.set_aspect("auto")
        configure_regime_composite_gridlines(
            ax,
            panel_row=panel_row,
            panel_col=panel_col,
            nrow=nrow_maps,
            ncol=ncol,
            show_bottom=show_bottom_labels,
            show_right=False,
            linewidth=0.4,
            alpha=0.28,
            tick_labelsize=fs_tick,
        )
    else:
        im = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=norm, alpha=alpha)
        if raw_m is not None:
            _, _, zr = _mesh_2d(raw_m, raw_lat_dim, raw_lon_dim)
            ax.contour(lon_m, lat_m, zr, colors="k", linewidths=0.9)
        draw_aoi_box(ax, aoi_bbox)
        configure_plain_composite_axis_ticks(
            ax,
            panel_row=panel_row,
            panel_col=panel_col,
            nrow=nrow_maps,
            ncol=ncol,
            tick_labelsize=fs_tick,
        )
        if not show_bottom_labels:
            ax.tick_params(axis="x", labelbottom=False)
    return ax, im


def _style_mv3row_colorbar_outline(cb) -> None:
    """Alias for :func:`_style_colorbar_outline` (mv3row / EPE.6)."""
    _style_colorbar_outline(cb)


def _add_wide_row_horizontal_colorbar(
    fig: Figure,
    gs: object,
    row_cbar: int,
    mappable,
    label: str,
    *,
    fs_axis: float,
    fs_tick: float,
    extend: str = "neither",
) -> None:
    """Centred ¼-width horizontal colorbar; tick labels above strip, title below."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    inner = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[row_cbar, :],
        height_ratios=[1.15, 0.40],
        hspace=0.32,
    )
    bar_row = GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=inner[0, 0],
        width_ratios=[3, 2, 3],
        wspace=0.0,
    )
    cax_bar = fig.add_subplot(bar_row[0, 1])
    cb = fig.colorbar(mappable, cax=cax_bar, orientation="horizontal", extend=extend)
    _style_colorbar_outline(cb)
    # Put tick labels above the colour strip so the axis title can sit below them.
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("bottom")
    cb.ax.tick_params(
        axis="x",
        which="major",
        direction="out",
        length=2.5,
        width=0.6,
        labelsize=fs_tick,
        pad=2.0,
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )
    cb.set_label(label, fontsize=fs_axis, labelpad=12.0)
    fig.add_subplot(inner[1, 0]).set_axis_off()
