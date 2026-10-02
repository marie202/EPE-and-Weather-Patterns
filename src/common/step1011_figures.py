"""Matplotlib figure builders for Step 10/11 regime diagnostics (notebooks + PDF)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from common.colorbar_discretize import discretize_diverging, style_discrete_ticks
from common.figure_counter import bump_figure_number
from common.geo import (
    apply_plain_axes_bbox_limits,
    cartopy_lines_on_top,
    cartopy_modules,
    cartopy_underlay,
    configure_plain_composite_axis_ticks,
    configure_regime_composite_gridlines,
    draw_aoi_box,
    is_bottom_panel_row,
    label_contour_heights,
    resolve_map_extent,
)
from common.physics import G0, is_geopotential_units, z500_anomaly_units_look_physical
from common.spatial import spatial_dims, subset_da_to_bbox_or_raise, wrap_lon_sort_lat_da
from common.page_layout import (
    PDF_FS_FIG_ANNOT,
    PDF_FS_FIG_AXIS,
    PDF_FS_FIG_CLABEL,
    PDF_FS_FIG_FOOTER,
    PDF_FS_FIG_LEGEND,
    PDF_FS_FIG_STRIP_PANEL,
    PDF_FS_FIG_SUPTITLE,
    PDF_FS_FIG_TICK,
    PDF_FS_FIG_TITLE,
)

# Regime composite map grids (legacy Z500, Step 8/9/10–11, multivariate summary PDFs).
# Panel size + right margin are tuned so PDF maps stay large and the vertical
# colorbar sits in the margin (not on top of the right-hand column).
COMPOSITE_PANEL_W_IN = 6.20
COMPOSITE_PANEL_H_IN = 4.00
COMPOSITE_FIG_PAD_W_IN = 0.72
COMPOSITE_FIG_PAD_H_IN = 0.95
COMPOSITE_WSPACE = 0.08
# Matches the row spacing used throughout z500_precip_panel_figures.py's mosaics (0.08) --
# was 0.10 here, giving this module's composite grids a visibly wider row gap than every
# other composite figure.
COMPOSITE_HSPACE = 0.08
# ``subplots_adjust(right=…)`` — keep maps left of this; colorbar is placed just right.
COMPOSITE_CBAR_AXIS_W = 0.80
COMPOSITE_CBAR_WIDTH = 0.030
# Per-field column in multivariate composite rows (vo850 | tcwv | …): map grid | colorbar.
MV_FIELD_CBAR_WIDTH_RATIO = 0.07
MV_FIELD_CBAR_WSPACE = 0.10
MV_FIELD_OUTER_WSPACE = 0.14


def _composite_regime_grid_shape(
    n_reg: int, *, grid_shape: tuple[int, int] | None = None, panel_aspect: float | None = None,
) -> tuple[int, int, tuple[float, float]]:
    """``panel_aspect`` (plotted bbox's lon-extent / lat-extent), when given, sizes each panel's
    width so a PlateCarree map fills its cell edge-to-edge regardless of bbox shape. Cartopy's
    equal-aspect GeoAxes otherwise pads the map inside whatever fixed-ratio cell it's given,
    leaving asymmetric blank space around it -- most visible as inconsistent ROW spacing across
    bbox steps: confirmed B0 (wide, ~2:1 lon:lat) rendered noticeably wider row gaps than
    B4/B5 (narrower bboxes closer to the fixed COMPOSITE_PANEL_W_IN:H_IN ratio) even though
    COMPOSITE_HSPACE was identical for all of them.
    """
    if grid_shape is not None:
        nrow, ncol = grid_shape
    else:
        ncol = min(2, n_reg)
        nrow = int(np.ceil(n_reg / ncol))
    panel_w_in = COMPOSITE_PANEL_W_IN
    if panel_aspect is not None and np.isfinite(panel_aspect) and panel_aspect > 0:
        # Clamp so a pathologically thin/wide bbox can't blow up the figure -- this is meant to
        # fill the cell at typical Euro-Atlantic/N.America bbox shapes, not resize unbounded.
        panel_w_in = COMPOSITE_PANEL_H_IN * min(max(panel_aspect, 0.6), 3.2)
    fig_w = panel_w_in * ncol + COMPOSITE_FIG_PAD_W_IN
    fig_h = COMPOSITE_PANEL_H_IN * nrow + COMPOSITE_FIG_PAD_H_IN
    return nrow, ncol, (fig_w, fig_h)


def _composite_layout_top_bottom(*, has_footer: bool, numbered: bool) -> tuple[float, float]:
    top = 0.88 if numbered else 0.90
    bot = 0.14 if has_footer else 0.10
    return top, bot


def _composite_colorbar_cax_rect(top: float, bot: float) -> list[float]:
    """Figure coords for a vertical colorbar in the right margin."""
    y0 = bot + 0.03
    height = max(0.22, top - bot - 0.06)
    left = min(COMPOSITE_CBAR_AXIS_W + 0.028, 0.905)
    return [left, y0, COMPOSITE_CBAR_WIDTH, height]


def _apply_composite_subplots_adjust(
    fig: Figure,
    *,
    top: float,
    bot: float,
    left: float = 0.04,
) -> None:
    fig.subplots_adjust(
        left=left,
        right=COMPOSITE_CBAR_AXIS_W,
        top=top,
        bottom=bot,
        wspace=COMPOSITE_WSPACE,
        hspace=COMPOSITE_HSPACE,
    )


def _attach_composite_colorbar(
    fig: Figure,
    mappable,
    *,
    label: str,
    top: float,
    bot: float,
    extend: str = "both",
) -> None:
    cax = fig.add_axes(_composite_colorbar_cax_rect(top, bot))
    cbar = fig.colorbar(mappable, cax=cax, extend=extend)
    cbar.set_label(label, fontsize=PDF_FS_FIG_AXIS)
    cbar.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    style_discrete_ticks(cbar)


def _multivariate_field_cbar_label(col_title: str, da: xr.DataArray, *, units_in: str) -> str:
    """Colorbar label for one field column in a multivariate composite row."""
    units = str(da.attrs.get("units", "") or "").strip()
    if is_geopotential_units(units_in):
        return "Z500 anomaly (gpm)"
    long_name = str(da.attrs.get("long_name", "") or "").strip()
    head = long_name or col_title
    if units:
        return f"{head} ({units})"
    return head


_DIVERGING_FILL_CMAPS = frozenset(
    {"rdbu_r", "rdbu", "piyg_r", "piyg", "brbg_r", "coolwarm", "bwr", "seismic"}
)


def _cmap_fill_limits(values: np.ndarray, *, cmap_name: str) -> tuple[float, float]:
    """Diverging maps: symmetric limits; sequential maps (e.g. YlGnBu): 2–98% range."""
    flat = np.asarray(values, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    key = str(cmap_name).strip().lower().replace(" ", "")
    if key not in _DIVERGING_FILL_CMAPS:
        vmin = float(np.nanpercentile(flat, 2))
        vmax = float(np.nanpercentile(flat, 98))
        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
            vmin, vmax = float(np.nanmin(flat)), float(np.nanmax(flat))
        if not (np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin):
            return 0.0, 1.0
        return vmin, vmax
    vmax = float(np.nanpercentile(np.abs(flat), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(np.abs(flat))) if np.any(np.abs(flat) > 0) else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def regime_panel_title(
    *,
    regime: int,
    regime_case_counts: dict[int, int] | None = None,
    align_groups: list[dict] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> str:
    """``"<short> [<lit_abbr>] (<pct>%)"`` -- ``short`` is the regime-alignment yaml's short
    label (e.g. ``"Z3"``, see ``common.regime_alignment``) and ``pct`` this regime's
    share of all classified days in the run (``regime_case_counts`` summed across every regime,
    not just the ones plotted in this figure). Falls back to ``"Regime N"`` when alignment info
    isn't available (e.g. kmeans runs -- the alignment yaml is Ward-only) and drops the
    percentage when day counts aren't available.

    ``literature_abbr_by_short``, when given (B0 only -- see
    ``common.literature_regime_names.literature_abbr_by_short``), inserts the
    literature-matched abbreviation (e.g. ``"Z1 ScBl (22%)"``) for regimes it covers.
    """
    from common.regime_alignment import short_name_for_canonical

    canon = None if canonical_id_by_regime is None else canonical_id_by_regime.get(int(regime))
    short = short_name_for_canonical(align_groups, canon)
    label = short or f"Regime {int(regime)}"
    if short and literature_abbr_by_short and short in literature_abbr_by_short:
        label = f"{short} {literature_abbr_by_short[short]}"
    if regime_case_counts:
        total = sum(regime_case_counts.values())
        count = regime_case_counts.get(int(regime))
        if count is not None and total > 0:
            pct = round(100.0 * count / total)
            return f"{label} ({pct}%)"
    return label


def build_figure_regime_anomaly_with_raw_contours(
    anom: xr.DataArray,
    raw: xr.DataArray,
    *,
    bbox_step: str,
    k: int,
    fig_num: list[int] | None,
    bbox: dict[str, float] | None = None,
    footer_line: str | None = None,
    suptitle_line1: str | None = None,
    suptitle_line2: str | None = None,
    regime_case_counts: dict[int, int] | None = None,
    fill_cmap: str = "RdBu_r",
    aoi_bbox: dict[str, float] | None = None,
    grid_shape: tuple[int, int] | None = None,
    vmin_fill: float | None = None,
    vmax_fill: float | None = None,
    align_groups: list[dict] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """Notebook 2d: field anomaly fill + black contours of regime-mean raw Z500 (m).

    ``vmin_fill``/``vmax_fill``, if given, override the auto-computed (``_cmap_fill_limits``)
    fill color scale one bound at a time -- pass both to pin the scale across multiple calls
    (different bboxes/regions).
    """
    lat_dim, lon_dim = spatial_dims(anom)
    anom2 = subset_da_to_bbox_or_raise(anom.astype(np.float64).load(), bbox)
    raw2 = subset_da_to_bbox_or_raise(raw.astype(np.float64).load(), bbox)
    anom2 = wrap_lon_sort_lat_da(anom2, lat_dim, lon_dim)
    raw2 = wrap_lon_sort_lat_da(raw2, lat_dim, lon_dim)

    reg_anom = anom2["regime"].values.astype(int)
    reg_raw = raw2["regime"].values.astype(int)
    regimes_plot = np.intersect1d(reg_anom, reg_raw)
    if regimes_plot.size == 0:
        raise ValueError(f"No overlapping regimes between anom={reg_anom.tolist()} and raw={reg_raw.tolist()}")
    if canonical_id_by_regime is not None:
        regimes_plot = np.array(
            sorted(regimes_plot, key=lambda r: canonical_id_by_regime.get(int(r), int(r))),
            dtype=regimes_plot.dtype,
        )

    anom2 = anom2.sel(regime=regimes_plot)
    raw2 = raw2.sel(regime=regimes_plot)

    anom_units_in = str(getattr(anom2, "attrs", {}).get("units", ""))
    raw_units_in = str(getattr(raw2, "attrs", {}).get("units", ""))
    anom_is_gpm = z500_anomaly_units_look_physical(anom2, anom_units_in)
    raw_is_gpm = is_geopotential_units(raw_units_in)
    anom_gpm = (anom2 / G0) if anom_is_gpm else anom2
    anom_gpm.attrs.update({"units": "gpm" if anom_is_gpm else (anom_units_in or "1")})
    raw_m = (raw2 / G0) if raw_is_gpm else raw2
    raw_m.attrs.update({"units": "m" if raw_is_gpm else (raw_units_in or "1")})

    _lat_extent = np.asarray(anom2[lat_dim].values, dtype=float)
    _lon_extent = np.asarray(anom2[lon_dim].values, dtype=float)
    map_extent = resolve_map_extent(bbox, lat=_lat_extent, lon=_lon_extent)

    if vmin_fill is None or vmax_fill is None:
        _auto_vmin, _auto_vmax = _cmap_fill_limits(np.asarray(anom_gpm.values, dtype=float), cmap_name=fill_cmap)
        vmin_fill = _auto_vmin if vmin_fill is None else vmin_fill
        vmax_fill = _auto_vmax if vmax_fill is None else vmax_fill
    cmap, fill_norm, fill_extend = discretize_diverging(vmin_fill, vmax_fill, str(fill_cmap))
    fill_units = str(getattr(anom_gpm, "attrs", {}).get("units", "") or "")
    if anom_is_gpm:
        cbar_fill_label = "Z500 anomaly (gpm)"
    elif fill_units in ("1", "sigma", ""):
        cbar_fill_label = "Z500 anomaly (σ, normalized)"
    elif fill_units:
        cbar_fill_label = f"anomaly ({fill_units})"
    else:
        cbar_fill_label = "anomaly"
    n_reg = int(regimes_plot.size)
    panel_aspect = None
    if map_extent is not None:
        lon_span = map_extent[1] - map_extent[0]
        lat_span = map_extent[3] - map_extent[2]
        if lat_span > 0:
            panel_aspect = lon_span / lat_span
    nrow, ncol, figsize = _composite_regime_grid_shape(
        n_reg, grid_shape=grid_shape, panel_aspect=panel_aspect
    )

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    fig = plt.figure(figsize=figsize, facecolor="white")
    n = bump_figure_number(fig_num)

    axes_list = []
    im_last = None
    for i, r in enumerate(regimes_plot):
        ax = (
            fig.add_subplot(nrow, ncol, i + 1, projection=ccrs.PlateCarree())
            if use_geo
            else fig.add_subplot(nrow, ncol, i + 1)
        )
        axes_list.append(ax)
        r_int = int(r)
        ax.set_title(
            regime_panel_title(
                regime=r_int,
                regime_case_counts=regime_case_counts,
                align_groups=align_groups,
                canonical_id_by_regime=canonical_id_by_regime,
                literature_abbr_by_short=literature_abbr_by_short,
            ),
            fontsize=PDF_FS_FIG_TITLE,
        )
        a = anom_gpm.sel(regime=r_int).squeeze(drop=True)
        rr = raw_m.sel(regime=r_int).squeeze(drop=True)
        if {lat_dim, lon_dim}.issubset(a.dims):
            a = a.transpose(lat_dim, lon_dim)
        if {lat_dim, lon_dim}.issubset(rr.dims):
            rr = rr.transpose(lat_dim, lon_dim)

        lon1 = np.asarray(a[lon_dim].values, dtype=float)
        lat1 = np.asarray(a[lat_dim].values, dtype=float)
        za = np.asarray(a.values, dtype=float)
        zr = np.asarray(rr.values, dtype=float)
        ny, nx = int(lat1.size), int(lon1.size)
        if za.shape == (nx, ny):
            za = za.T
        if zr.shape == (nx, ny):
            zr = zr.T
        if za.shape != (ny, nx) or zr.shape != (ny, nx):
            raise ValueError(
                f"Expected 2D fields shaped (lat,lon)=({ny},{nx}); got za={za.shape}, zr={zr.shape}"
            )
        lon_m, lat_m = np.meshgrid(lon1, lat1)
        cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
        if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
            cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
        if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
            mu = float(np.nanmean(zr)) if np.any(np.isfinite(zr)) else 0.0
            cmin, cmax = mu - 10.0, mu + 10.0
        levels_raw_r = np.linspace(cmin, cmax, 9)

        if use_geo and cfeat is not None and pc is not None and map_extent is not None:
            cartopy_underlay(ax, map_extent, cfeature=cfeat)
            im_last = ax.pcolormesh(
                lon_m,
                lat_m,
                za,
                shading="auto",
                cmap=cmap,
                norm=fill_norm,
                alpha=0.75,
                transform=pc,
                zorder=8,
                rasterized=True,
            )
            cs = ax.contour(
                lon_m,
                lat_m,
                zr,
                levels=levels_raw_r,
                colors="k",
                linewidths=1.05,
                alpha=0.95,
                transform=pc,
                zorder=12,
            )
            cartopy_lines_on_top(ax, cfeature=cfeat)
            label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
            draw_aoi_box(ax, aoi_bbox, transform=pc)
            ax.set_extent(map_extent, crs=pc)
            configure_regime_composite_gridlines(
                ax,
                panel_row=i // ncol,
                panel_col=i % ncol,
                nrow=nrow,
                ncol=ncol,
                n_panels=n_reg,
                tick_labelsize=PDF_FS_FIG_TICK,
            )
        elif use_geo and pc is not None:
            im_last = ax.pcolormesh(
                lon_m,
                lat_m,
                za,
                shading="auto",
                cmap=cmap,
                norm=fill_norm,
                alpha=0.75,
                transform=pc,
                zorder=8,
                rasterized=True,
            )
            cs = ax.contour(
                lon_m,
                lat_m,
                zr,
                levels=levels_raw_r,
                colors="k",
                linewidths=1.05,
                alpha=0.95,
                transform=pc,
                zorder=12,
            )
            if cfeat is not None:
                cartopy_lines_on_top(ax, cfeature=cfeat)
            label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
            draw_aoi_box(ax, aoi_bbox, transform=pc)
        else:
            im_last = ax.pcolormesh(
                lon_m,
                lat_m,
                za,
                shading="auto",
                cmap=cmap,
                norm=fill_norm,
                alpha=0.75,
            )
            cs = ax.contour(
                lon_m, lat_m, zr, levels=levels_raw_r, colors="k", linewidths=1.05, alpha=0.95
            )
            label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
            apply_plain_axes_bbox_limits(ax, bbox)
            configure_plain_composite_axis_ticks(
                ax,
                panel_row=i // ncol,
                panel_col=i % ncol,
                nrow=nrow,
                ncol=ncol,
                tick_labelsize=PDF_FS_FIG_TICK,
            )
            if i // ncol == nrow - 1:
                ax.set_xlabel(lon_dim, fontsize=PDF_FS_FIG_AXIS)
            if i % ncol == 0:
                ax.set_ylabel(lat_dim, fontsize=PDF_FS_FIG_AXIS)
            ax.set_aspect("auto")
            draw_aoi_box(ax, aoi_bbox)

    for j in range(n_reg, nrow * ncol):
        if use_geo:
            ax = fig.add_subplot(nrow, ncol, j + 1, projection=ccrs.PlateCarree())
            if map_extent is not None:
                ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(nrow, ncol, j + 1)
        ax.axis("off")

    top, bot = _composite_layout_top_bottom(has_footer=bool(footer_line), numbered=bool(n))
    _apply_composite_subplots_adjust(fig, top=top, bot=bot)
    if im_last is not None:
        _attach_composite_colorbar(fig, im_last, label=cbar_fill_label, top=top, bot=bot, extend=fill_extend)
    if footer_line:
        fig.text(0.5, 0.012, footer_line, ha="center", va="bottom", fontsize=PDF_FS_FIG_FOOTER, color="0.32")
    return fig


def build_figure_composite_grid_notebook_style(
    da: xr.DataArray,
    *,
    suptitle: str,
    zlabel: str,
    with_contours: bool,
    pcolormesh_alpha: float,
    fig_num: list[int] | None,
    bbox: dict[str, float] | None = None,
    footer_line: str | None = None,
    regime_case_counts: dict[int, int] | None = None,
    aoi_bbox: dict[str, float] | None = None,
    align_groups: list[dict] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """RdBu_r symmetric panels; optional black contours; Cartopy basemap if available."""
    lat_dim, lon_dim = spatial_dims(da)
    da = subset_da_to_bbox_or_raise(da.astype(np.float64), bbox)
    regs = np.asarray(da.coords["regime"].values).astype(int)
    if canonical_id_by_regime is not None:
        regs = np.array(
            sorted(regs, key=lambda r: canonical_id_by_regime.get(int(r), int(r))),
            dtype=regs.dtype,
        )
    n_reg = int(regs.size)
    flat = np.asarray(da.values, dtype=float)
    vmax = float(np.nanpercentile(np.abs(flat), 98))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    contour_levels = np.linspace(-vmax, vmax, 21)
    fill_cmap, fill_norm, fill_extend = discretize_diverging(-vmax, vmax, "RdBu_r")

    nrow, ncol, figsize = _composite_regime_grid_shape(n_reg)

    fig = plt.figure(figsize=figsize, facecolor="white")
    n = bump_figure_number(fig_num)
    title_head = str(suptitle).split("\n")[0].strip()
    if n:
        fig.suptitle(f"Figure {n} — {title_head}", fontsize=PDF_FS_FIG_SUPTITLE, y=0.98)
    else:
        fig.suptitle(title_head, fontsize=PDF_FS_FIG_SUPTITLE, y=0.98)

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    m_last = None
    axes_list: list = []
    map_extent: list[float] | None = None
    map_extent = resolve_map_extent(bbox) if use_geo else None

    for i, r in enumerate(regs):
        ax = (
            fig.add_subplot(nrow, ncol, i + 1, projection=ccrs.PlateCarree())
            if use_geo
            else fig.add_subplot(nrow, ncol, i + 1)
        )
        axes_list.append(ax)
        sl = da.sel(regime=int(r))
        lat = np.asarray(sl[lat_dim].values, dtype=float)
        lon = np.asarray(sl[lon_dim].values, dtype=float)
        z = np.asarray(sl.values, dtype=float)
        if lat.size >= 2 and lat[0] > lat[-1]:
            z = np.flipud(z)
            lat = lat[::-1]
        lon_m, lat_m = np.meshgrid(lon, lat)
        if use_geo and cfeat is not None and pc is not None and map_extent is not None:
            cartopy_underlay(ax, map_extent, cfeature=cfeat)
        pkw: dict = dict(
            shading="auto",
            cmap=fill_cmap,
            norm=fill_norm,
            alpha=pcolormesh_alpha if with_contours else 1.0,
        )
        if use_geo and pc is not None:
            pkw["transform"] = pc
            pkw["zorder"] = 8
            pkw["rasterized"] = True
        m_last = ax.pcolormesh(lon_m, lat_m, z, **pkw)
        if with_contours:
            ckw = dict(levels=contour_levels, colors="k", linewidths=0.75)
            if use_geo and pc is not None:
                ckw["transform"] = pc
                ckw["zorder"] = 12
            ax.contour(lon_m, lat_m, z, **ckw)
        if use_geo and cfeat is not None:
            cartopy_lines_on_top(ax, cfeature=cfeat)
        if use_geo and pc is not None:
            draw_aoi_box(ax, aoi_bbox, transform=pc)
        else:
            draw_aoi_box(ax, aoi_bbox)
        ax.set_title(
            regime_panel_title(
                regime=int(r),
                regime_case_counts=regime_case_counts,
                align_groups=align_groups,
                canonical_id_by_regime=canonical_id_by_regime,
                literature_abbr_by_short=literature_abbr_by_short,
            ),
            fontsize=PDF_FS_FIG_TITLE,
        )
        if use_geo and pc is not None and map_extent is not None:
            ax.set_extent(map_extent, crs=pc)
            configure_regime_composite_gridlines(
                ax,
                panel_row=i // ncol,
                panel_col=i % ncol,
                nrow=nrow,
                ncol=ncol,
                n_panels=n_reg,
                tick_labelsize=PDF_FS_FIG_TICK,
            )
        else:
            apply_plain_axes_bbox_limits(ax, bbox)
            configure_plain_composite_axis_ticks(
                ax,
                panel_row=i // ncol,
                panel_col=i % ncol,
                nrow=nrow,
                ncol=ncol,
                n_panels=n_reg,
                tick_labelsize=PDF_FS_FIG_TICK,
            )
            if is_bottom_panel_row(i // ncol, i % ncol, nrow, ncol, n_reg):
                ax.set_xlabel(lon_dim, fontsize=PDF_FS_FIG_AXIS)
            if i % ncol == 0:
                ax.set_ylabel(lat_dim, fontsize=PDF_FS_FIG_AXIS)
            ax.set_aspect("auto")

    for j in range(n_reg, nrow * ncol):
        if use_geo:
            ax = fig.add_subplot(nrow, ncol, j + 1, projection=ccrs.PlateCarree())
            if map_extent is not None:
                import cartopy.crs as ccrs

                ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(nrow, ncol, j + 1)
        ax.axis("off")

    top, bot = _composite_layout_top_bottom(has_footer=bool(footer_line), numbered=bool(n))
    _apply_composite_subplots_adjust(fig, top=top, bot=bot, left=0.05)
    if m_last is not None:
        _attach_composite_colorbar(fig, m_last, label=zlabel or "anomaly", top=top, bot=bot, extend=fill_extend)
    if footer_line:
        fig.text(0.5, 0.01, footer_line, ha="center", va="bottom", fontsize=PDF_FS_FIG_FOOTER, color="0.32")
    return fig


def build_figure_regime_field_panels_cell12(
    da: xr.DataArray,
    *,
    suptitle: str,
    fig_num: list[int] | None,
    bbox: dict[str, float] | None = None,
    regime_case_counts: dict[int, int] | None = None,
    fill_cmap: str = "RdBu_r",
    aoi_bbox: dict[str, float] | None = None,
    align_groups: list[dict] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """Notebook cell 12 style: mean composite per regime; optional field-specific fill colormap."""
    lat_dim, lon_dim = spatial_dims(da)
    da = subset_da_to_bbox_or_raise(da.astype(np.float64), bbox)
    regs = np.asarray(da.coords["regime"].values).astype(int)
    if canonical_id_by_regime is not None:
        regs = np.array(
            sorted(regs, key=lambda r: canonical_id_by_regime.get(int(r), int(r))),
            dtype=regs.dtype,
        )
    n_reg = int(regs.size)
    vals = np.asarray(da.values, dtype=float)
    vmin, vmax = _cmap_fill_limits(vals, cmap_name=fill_cmap)

    units = str(da.attrs.get("units", "") or "")
    long_name = str(da.attrs.get("long_name", "") or "").strip()
    name = da.name or "value"
    cbar_label = f"{long_name or name} ({units})" if units else (long_name or name or "value")
    levels_c = np.linspace(vmin, vmax, 8)

    nrow, ncol, figsize = _composite_regime_grid_shape(n_reg)
    fig = plt.figure(figsize=figsize, facecolor="white")
    n = bump_figure_number(fig_num)
    title_head = str(suptitle).split("\n")[0].strip()
    if n:
        fig.suptitle(f"Figure {n} — {title_head}", fontsize=PDF_FS_FIG_SUPTITLE, y=0.98)
    else:
        fig.suptitle(title_head, fontsize=PDF_FS_FIG_SUPTITLE, y=0.98)

    cmap, fill_norm, fill_extend = discretize_diverging(vmin, vmax, str(fill_cmap))
    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        pc = None

    axes_list = []
    m_last = None
    map_extent_r = resolve_map_extent(bbox) if use_geo else None

    for i, r in enumerate(regs):
        ax = (
            fig.add_subplot(nrow, ncol, i + 1, projection=ccrs.PlateCarree())
            if use_geo
            else fig.add_subplot(nrow, ncol, i + 1)
        )
        axes_list.append(ax)
        m = da.sel(regime=int(r))
        lat = np.asarray(m[lat_dim].values, dtype=float)
        lon = np.asarray(m[lon_dim].values, dtype=float)
        z = np.asarray(m.values, dtype=float)
        if lat.size >= 2 and lat[0] > lat[-1]:
            z = np.flipud(z)
            lat = lat[::-1]
        lon_m, lat_m = np.meshgrid(lon, lat)
        if use_geo and cfeat is not None and map_extent_r is not None:
            cartopy_underlay(ax, map_extent_r, cfeature=cfeat)
        pkw = dict(shading="auto", cmap=cmap, norm=fill_norm, alpha=0.6, rasterized=True)
        if use_geo and pc is not None:
            pkw["transform"] = pc
            pkw["zorder"] = 8
        m_last = ax.pcolormesh(lon_m, lat_m, z, **pkw)
        ckw = dict(levels=levels_c, colors="k", linewidths=0.85, alpha=0.82)
        if use_geo and pc is not None:
            ckw["transform"] = pc
            ckw["zorder"] = 12
        ax.contour(lon_m, lat_m, z, **ckw)
        if use_geo and cfeat is not None:
            cartopy_lines_on_top(ax, cfeature=cfeat)
        if use_geo and pc is not None:
            draw_aoi_box(ax, aoi_bbox, transform=pc)
        else:
            draw_aoi_box(ax, aoi_bbox)
        ax.set_title(
            regime_panel_title(
                regime=int(r),
                regime_case_counts=regime_case_counts,
                align_groups=align_groups,
                canonical_id_by_regime=canonical_id_by_regime,
                literature_abbr_by_short=literature_abbr_by_short,
            ),
            fontsize=PDF_FS_FIG_TITLE,
        )
        if use_geo and pc is not None and map_extent_r is not None:
            ax.set_extent(map_extent_r, crs=pc)
            configure_regime_composite_gridlines(
                ax,
                panel_row=i // ncol,
                panel_col=i % ncol,
                nrow=nrow,
                ncol=ncol,
                n_panels=n_reg,
                tick_labelsize=PDF_FS_FIG_TICK,
            )
        else:
            apply_plain_axes_bbox_limits(ax, bbox)
            configure_plain_composite_axis_ticks(
                ax,
                panel_row=i // ncol,
                panel_col=i % ncol,
                nrow=nrow,
                ncol=ncol,
                n_panels=n_reg,
                tick_labelsize=PDF_FS_FIG_TICK,
            )
            if is_bottom_panel_row(i // ncol, i % ncol, nrow, ncol, n_reg):
                ax.set_xlabel(lon_dim, fontsize=PDF_FS_FIG_AXIS)
            if i % ncol == 0:
                ax.set_ylabel(lat_dim, fontsize=PDF_FS_FIG_AXIS)
            ax.set_aspect("auto")
    for j in range(n_reg, nrow * ncol):
        if use_geo:
            ax = fig.add_subplot(nrow, ncol, j + 1, projection=ccrs.PlateCarree())
            if map_extent_r is not None:
                ax.set_extent(map_extent_r, crs=ccrs.PlateCarree())
        else:
            ax = fig.add_subplot(nrow, ncol, j + 1)
        ax.axis("off")
    top, bot = _composite_layout_top_bottom(has_footer=False, numbered=bool(n))
    _apply_composite_subplots_adjust(fig, top=top, bot=bot)
    if m_last is not None:
        _attach_composite_colorbar(fig, m_last, label=cbar_label, top=top, bot=bot, extend=fill_extend)
    return fig


def build_figure_transitions_pair(ds: xr.Dataset, *, fig_num: list[int] | None) -> Figure | None:
    """Transition counts + probabilities on one row (one ``Figure N`` for the PDF)."""
    has_c = "transition_counts" in ds.data_vars
    has_p = "transition_prob" in ds.data_vars
    if not has_c and not has_p:
        return None

    def _one_matrix(
        ax,
        arr: np.ndarray,
        xlabs: list[str],
        ylabs: list[str],
        *,
        title: str,
        cmap: str,
        vmin: float,
        vmax: float,
        annotate: bool,
    ) -> None:
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=PDF_FS_FIG_TITLE)
        ax.set_xlabel("to regime", fontsize=PDF_FS_FIG_AXIS)
        ax.set_ylabel("from regime", fontsize=PDF_FS_FIG_AXIS)
        ax.set_xticks(np.arange(arr.shape[1]))
        ax.set_yticks(np.arange(arr.shape[0]))
        ax.set_xticklabels(xlabs)
        ax.set_yticklabels(ylabs)
        ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
        if annotate and arr.size:
            mx = float(np.nanmax(arr)) if arr.size else 1.0
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    ax.text(
                        j,
                        i,
                        str(int(v)),
                        ha="center",
                        va="center",
                        color="white" if v > mx / 2 else "black",
                        fontsize=PDF_FS_FIG_ANNOT,
                    )

    n = bump_figure_number(fig_num)

    if has_c and has_p:
        Tc = ds["transition_counts"]
        Tp = ds["transition_prob"]
        ac = np.asarray(Tc.values, dtype=float)
        ap = np.asarray(Tp.values, dtype=float)
        vmax_c = float(np.nanmax(ac)) if ac.size else 1.0
        vmax_p = float(np.nanmax(ap))
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.05), facecolor="white")
        _one_matrix(
            axes[0],
            ac,
            [str(int(x)) for x in Tc["regime_to"].values],
            [str(int(x)) for x in Tc["regime_from"].values],
            title="Counts",
            cmap="viridis",
            vmin=0.0,
            vmax=max(vmax_c, 1e-6) if np.isfinite(vmax_c) else 1e-6,
            annotate=True,
        )
        _one_matrix(
            axes[1],
            ap,
            [str(int(x)) for x in Tp.coords["regime_to"].values],
            [str(int(x)) for x in Tp.coords["regime_from"].values],
            title="Probabilities (row-normalised)",
            cmap="magma",
            vmin=0.0,
            vmax=vmax_p if np.isfinite(vmax_p) and vmax_p > 0 else 1e-6,
            annotate=False,
        )
        st2 = f"Figure {n} — Transitions (counts | probabilities)" if n else "Transitions (counts | probabilities)"
        fig.suptitle(st2, fontsize=PDF_FS_FIG_SUPTITLE, y=1.02)
        fig.tight_layout()
        return fig

    if has_c:
        Tc = ds["transition_counts"]
        ac = np.asarray(Tc.values, dtype=float)
        vmax_c = float(np.nanmax(ac)) if ac.size else 1.0
        fig, ax = plt.subplots(figsize=(4.9, 4.25), facecolor="white")
        _one_matrix(
            ax,
            ac,
            [str(int(x)) for x in Tc["regime_to"].values],
            [str(int(x)) for x in Tc["regime_from"].values],
            title="Transition counts",
            cmap="viridis",
            vmin=0.0,
            vmax=max(vmax_c, 1e-6) if np.isfinite(vmax_c) else 1e-6,
            annotate=True,
        )
        st = f"Figure {n} — Transition counts" if n else "Transition counts"
    else:
        Tp = ds["transition_prob"]
        ap = np.asarray(Tp.values, dtype=float)
        vmax_p = float(np.nanmax(ap))
        fig, ax = plt.subplots(figsize=(4.9, 4.25), facecolor="white")
        _one_matrix(
            ax,
            ap,
            [str(int(x)) for x in Tp.coords["regime_to"].values],
            [str(int(x)) for x in Tp.coords["regime_from"].values],
            title="Transition probabilities (row-normalised)",
            cmap="magma",
            vmin=0.0,
            vmax=vmax_p if np.isfinite(vmax_p) and vmax_p > 0 else 1e-6,
            annotate=False,
        )
        st = f"Figure {n} — Transition probabilities (row-normalised)" if n else "Transition probabilities (row-normalised)"
    fig.suptitle(st, fontsize=PDF_FS_FIG_SUPTITLE, y=1.02)
    fig.tight_layout()
    return fig


def build_figure_persistence_histograms(ds: xr.Dataset, *, fig_num: list[int] | None) -> Figure | None:
    if "run_regime" not in ds.data_vars or "run_length" not in ds.data_vars:
        return None
    run_regime = np.asarray(ds["run_regime"].values, dtype=int)
    run_len = np.asarray(ds["run_length"].values, dtype=int)
    if run_len.size == 0:
        return None
    regimes = np.unique(run_regime)
    n_reg = int(regimes.size)
    max_len = int(np.max(run_len))
    ncol = min(2, n_reg)
    nrow = int(np.ceil(n_reg / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.8 * nrow), facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    n = bump_figure_number(fig_num)
    title = "Persistence histograms (episode length by regime)"
    if n:
        fig.suptitle(f"Figure {n} — {title}", fontsize=PDF_FS_FIG_SUPTITLE, y=1.02)
    else:
        fig.suptitle(title, fontsize=PDF_FS_FIG_SUPTITLE, y=1.02)

    for i, r in enumerate(regimes):
        ax = axes[i]
        r_int = int(r)
        L = run_len[run_regime == r_int]
        ax.hist(L, bins=np.arange(1, max_len + 2) - 0.5, alpha=0.85, color="C0")
        ax.set_title(f"Regime {r_int} persistence", fontsize=PDF_FS_FIG_TITLE)
        ax.set_xlabel("episode length (days)", fontsize=PDF_FS_FIG_AXIS)
        ax.set_ylabel("count", fontsize=PDF_FS_FIG_AXIS)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    for j in range(n_reg, axes.size):
        axes[j].axis("off")
    fig.tight_layout()
    return fig


def build_figure_persistence_boxplots(ds: xr.Dataset, *, fig_num: list[int] | None) -> Figure | None:
    if "run_regime" not in ds.data_vars or "run_length" not in ds.data_vars:
        return None
    run_regime = np.asarray(ds["run_regime"].values, dtype=int)
    run_len = np.asarray(ds["run_length"].values, dtype=int)
    if run_len.size == 0:
        return None
    regimes = np.unique(run_regime)
    groups = [run_len[run_regime == int(r)] for r in regimes]
    if not groups or not any(g.size for g in groups):
        return None

    # Match the report boxplot footprint used elsewhere (wide + low) so the persistence
    # page reads like the other regime-wise boxplots in the PDF.
    fig_w = max(7.5, 0.95 * len(groups) + 4.2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5), facecolor="white")
    n = bump_figure_number(fig_num)
    title = "Persistence boxplots (episode length by regime)"
    if n:
        ax.set_title(f"Figure {n} — {title}", fontsize=PDF_FS_FIG_TITLE)
    else:
        ax.set_title(title, fontsize=PDF_FS_FIG_TITLE)

    ax.boxplot(
        groups,
        labels=[str(int(r)) for r in regimes],
        showfliers=False,
        notch=True,
    )

    ax.set_xlabel("regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("episode length (days)", fontsize=PDF_FS_FIG_AXIS)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    fig.tight_layout()
    return fig


def build_figure_monthly_fraction(ds: xr.Dataset, *, fig_num: list[int] | None) -> Figure | None:
    if "monthly_fraction" not in ds.data_vars:
        return None
    mf = ds["monthly_fraction"]
    if "month" not in mf.dims or "regime" not in mf.dims:
        return None
    vals = np.asarray(mf.values, dtype=float)
    months = np.asarray(mf["month"].values)
    regimes = np.asarray(mf["regime"].values, dtype=int)

    import pandas as pd

    df = pd.DataFrame(vals, index=months, columns=[str(int(r)) for r in regimes])

    fig, ax = plt.subplots(figsize=(7.0, 3.8), facecolor="white")
    n = bump_figure_number(fig_num)
    inner = "Regime occurrence by month (fractions)"
    if n:
        ax.set_title(f"Figure {n} — {inner}", fontsize=PDF_FS_FIG_TITLE)
    else:
        ax.set_title(inner, fontsize=PDF_FS_FIG_TITLE)

    df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", legend=True, width=0.85)
    ax.set_xlabel("month", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("fraction of days", fontsize=PDF_FS_FIG_AXIS)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    leg = ax.legend(title="regime", bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=PDF_FS_FIG_LEGEND)
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(PDF_FS_FIG_LEGEND)
    fig.tight_layout()
    return fig


def build_figure_transition_counts(ds: xr.Dataset, *, fig_num: list[int] | None) -> Figure | None:
    if "transition_counts" not in ds.data_vars:
        return None
    T = ds["transition_counts"]
    arr = np.asarray(T.values, dtype=float)
    n = bump_figure_number(fig_num)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor="white")
    im = ax.imshow(arr, cmap="viridis")
    ax.set_title(
        f"Figure {n} — Transition counts" if n else "Transition counts",
        fontsize=PDF_FS_FIG_TITLE,
    )
    ax.set_xlabel("to regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("from regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_xticks(np.arange(arr.shape[1]))
    ax.set_yticks(np.arange(arr.shape[0]))
    ax.set_xticklabels([str(int(x)) for x in T["regime_to"].values])
    ax.set_yticklabels([str(int(x)) for x in T["regime_from"].values])
    cb3 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb3.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    mx = float(np.nanmax(arr)) if arr.size else 1.0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(
                j,
                i,
                str(int(v)),
                ha="center",
                va="center",
                color="white" if v > mx / 2 else "black",
                fontsize=PDF_FS_FIG_ANNOT,
            )
    fig.tight_layout()
    return fig


def build_figure_multivariate_composite_row(
    field_blocks: list[tuple[str, xr.DataArray, str]],
    raw: xr.DataArray,
    *,
    bbox_step: str,
    k: int,
    fig_num: list[int] | None,
    bbox: dict[str, float] | None = None,
    suptitle_line1: str | None = None,
    footer_line: str | None = None,
    regime_case_counts: dict[int, int] | None = None,
    aoi_bbox: dict[str, float] | None = None,
    align_groups: list[dict] | None = None,
    canonical_id_by_regime: dict[int, int] | None = None,
    literature_abbr_by_short: dict[str, str] | None = None,
) -> Figure:
    """One or more fields side by side: anomaly fill + shared Z500 contours per regime."""
    from matplotlib import gridspec

    n_fields = len(field_blocks)
    if n_fields < 1:
        raise ValueError("Expected at least 1 field block")

    lat_dim, lon_dim = spatial_dims(field_blocks[0][1])
    raw2 = subset_da_to_bbox_or_raise(raw.astype(np.float64).load(), bbox)
    raw2 = wrap_lon_sort_lat_da(raw2, lat_dim, lon_dim)
    raw_units_in = str(getattr(raw2, "attrs", {}).get("units", ""))
    raw_is_gpm = is_geopotential_units(raw_units_in)
    raw_m = (raw2 / G0) if raw_is_gpm else raw2
    raw_m.attrs.update({"units": "m" if raw_is_gpm else (raw_units_in or "1")})

    _lat_extent = np.asarray(field_blocks[0][1][lat_dim].values, dtype=float)
    _lon_extent = np.asarray(field_blocks[0][1][lon_dim].values, dtype=float)
    map_extent = resolve_map_extent(bbox, lat=_lat_extent, lon=_lon_extent)

    reg_common = np.asarray(raw_m["regime"].values).astype(int)
    for _title, comp, _cmap in field_blocks:
        reg_common = np.intersect1d(reg_common, np.asarray(comp["regime"].values).astype(int))
    if reg_common.size == 0:
        raise ValueError("No overlapping regimes across field composites and raw Z500.")
    if canonical_id_by_regime is not None:
        reg_common = np.array(
            sorted(reg_common, key=lambda r: canonical_id_by_regime.get(int(r), int(r))),
            dtype=reg_common.dtype,
        )

    raw_m = raw_m.sel(regime=reg_common)
    n_reg = int(reg_common.size)
    nrow, ncol, inner_size = _composite_regime_grid_shape(n_reg)
    block_w, block_h = inner_size
    # Extra width per field column for a dedicated colorbar beside each variable block.
    cbar_w_in = max(0.32, block_w * MV_FIELD_CBAR_WIDTH_RATIO)
    fig = plt.figure(
        figsize=(block_w * n_fields + cbar_w_in * n_fields + COMPOSITE_FIG_PAD_W_IN, block_h + COMPOSITE_FIG_PAD_H_IN),
        facecolor="white",
    )
    n = bump_figure_number(fig_num)
    st = suptitle_line1 or f"Multivariate composites — {bbox_step} k={k}"
    if n:
        fig.suptitle(f"Figure {n} — {st}", fontsize=PDF_FS_FIG_SUPTITLE, y=0.99)
    else:
        fig.suptitle(st, fontsize=PDF_FS_FIG_SUPTITLE, y=0.99)

    _ccrs_m, cfeat = cartopy_modules()
    use_geo = _ccrs_m is not None and cfeat is not None
    if use_geo:
        import cartopy.crs as ccrs

        pc = ccrs.PlateCarree()
    else:
        ccrs = None  # type: ignore[assignment]
        pc = None

    gs_outer = gridspec.GridSpec(
        1,
        n_fields,
        figure=fig,
        wspace=MV_FIELD_OUTER_WSPACE,
        width_ratios=[1] * n_fields,
    )
    gs_fields: list = []

    for col, (col_title, anom_in, fill_cmap) in enumerate(field_blocks):
        anom2 = subset_da_to_bbox_or_raise(anom_in.astype(np.float64).load(), bbox)
        anom2 = wrap_lon_sort_lat_da(anom2, lat_dim, lon_dim).sel(regime=reg_common)
        anom_units_in = str(getattr(anom2, "attrs", {}).get("units", ""))
        anom_gpm = (anom2 / G0) if is_geopotential_units(anom_units_in) else anom2
        vmin_fill, vmax_fill = _cmap_fill_limits(np.asarray(anom_gpm.values, dtype=float), cmap_name=fill_cmap)
        cmap, fill_norm, fill_extend = discretize_diverging(vmin_fill, vmax_fill, str(fill_cmap))

        gs_field = gs_outer[col].subgridspec(
            1,
            2,
            width_ratios=[1.0, MV_FIELD_CBAR_WIDTH_RATIO],
            wspace=MV_FIELD_CBAR_WSPACE,
        )
        gs_fields.append((col_title, gs_field))
        gs_inner = gs_field[0, 0].subgridspec(nrow, ncol, wspace=COMPOSITE_WSPACE, hspace=COMPOSITE_HSPACE)
        im_col = None

        for i, r in enumerate(reg_common):
            r_int = int(r)
            ax = (
                fig.add_subplot(gs_inner[i // ncol, i % ncol], projection=ccrs.PlateCarree())
                if use_geo
                else fig.add_subplot(gs_inner[i // ncol, i % ncol])
            )
            a = anom_gpm.sel(regime=r_int).squeeze(drop=True)
            rr = raw_m.sel(regime=r_int).squeeze(drop=True)
            if {lat_dim, lon_dim}.issubset(a.dims):
                a = a.transpose(lat_dim, lon_dim)
            if {lat_dim, lon_dim}.issubset(rr.dims):
                rr = rr.transpose(lat_dim, lon_dim)
            lon1 = np.asarray(a[lon_dim].values, dtype=float)
            lat1 = np.asarray(a[lat_dim].values, dtype=float)
            za = np.asarray(a.values, dtype=float)
            zr = np.asarray(rr.values, dtype=float)
            ny, nx = int(lat1.size), int(lon1.size)
            if za.shape == (nx, ny):
                za = za.T
            if zr.shape == (nx, ny):
                zr = zr.T
            lon_m, lat_m = np.meshgrid(lon1, lat1)
            cmin, cmax = (float(x) for x in np.nanpercentile(zr, [2, 98]))
            if not (np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin):
                cmin, cmax = float(np.nanmin(zr)), float(np.nanmax(zr))
            levels_raw_r = np.linspace(cmin, cmax, 9)

            if use_geo and cfeat is not None and pc is not None and map_extent is not None:
                cartopy_underlay(ax, map_extent, cfeature=cfeat)
                im_col = ax.pcolormesh(
                    lon_m,
                    lat_m,
                    za,
                    shading="auto",
                    cmap=cmap,
                    norm=fill_norm,
                    alpha=0.75,
                    transform=pc,
                    zorder=8,
                    rasterized=True,
                )
                cs = ax.contour(
                    lon_m,
                    lat_m,
                    zr,
                    levels=levels_raw_r,
                    colors="k",
                    linewidths=0.9,
                    alpha=0.95,
                    transform=pc,
                    zorder=12,
                )
                cartopy_lines_on_top(ax, cfeature=cfeat)
                label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
                draw_aoi_box(ax, aoi_bbox, transform=pc)
                ax.set_extent(map_extent, crs=pc)
                # Multivariate row: tick labels only on the first field column (vo850) --
                # other field columns share the same geographic extent and would otherwise
                # repeat identical lat/lon labels -- and, within that column, only on the
                # regime sub-grid's own outer-bottom/outer-left panels.
                configure_regime_composite_gridlines(
                    ax,
                    panel_row=i // ncol,
                    panel_col=i % ncol,
                    nrow=nrow,
                    ncol=ncol,
                    n_panels=n_reg,
                    show_left=(col == 0 and i % ncol == 0),
                    show_bottom=(col == 0 and is_bottom_panel_row(i // ncol, i % ncol, nrow, ncol, n_reg)),
                    tick_labelsize=PDF_FS_FIG_TICK,
                )
            else:
                im_col = ax.pcolormesh(lon_m, lat_m, za, shading="auto", cmap=cmap, norm=fill_norm, alpha=0.75)
                cs = ax.contour(lon_m, lat_m, zr, levels=levels_raw_r, colors="k", linewidths=0.9)
                label_contour_heights(ax, cs, fontsize=PDF_FS_FIG_CLABEL, fmt="%.0f")
                apply_plain_axes_bbox_limits(ax, bbox)
                draw_aoi_box(ax, aoi_bbox)
                configure_plain_composite_axis_ticks(
                    ax,
                    panel_row=i // ncol,
                    panel_col=i % ncol,
                    nrow=nrow,
                    ncol=ncol,
                    n_panels=n_reg,
                    tick_labelsize=PDF_FS_FIG_TICK,
                )
                if not (col == 0 and i % ncol == 0):
                    ax.tick_params(axis="y", labelleft=False)
                if not (col == 0 and is_bottom_panel_row(i // ncol, i % ncol, nrow, ncol, n_reg)):
                    ax.tick_params(axis="x", labelbottom=False)
            ax.set_title(
                regime_panel_title(
                    regime=r_int,
                    regime_case_counts=regime_case_counts,
                    align_groups=align_groups,
                    canonical_id_by_regime=canonical_id_by_regime,
                    literature_abbr_by_short=literature_abbr_by_short,
                ),
                fontsize=PDF_FS_FIG_STRIP_PANEL,
            )

        if im_col is not None:
            cax = fig.add_subplot(gs_field[0, 1])
            cb = fig.colorbar(im_col, cax=cax, extend=fill_extend)
            cb.set_label(
                _multivariate_field_cbar_label(col_title, anom_gpm, units_in=anom_units_in),
                fontsize=PDF_FS_FIG_AXIS,
            )
            cb.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
            style_discrete_ticks(cb)

    top, bot = _composite_layout_top_bottom(has_footer=bool(footer_line), numbered=bool(n))
    gs_outer.update(left=0.03, right=0.98, top=top, bottom=bot)

    for col_title, gs_field in gs_fields:
        field_box = gs_field[0, 0].get_position(fig)
        cbar_box = gs_field[0, 1].get_position(fig)
        col_x = 0.5 * (field_box.x0 + cbar_box.x1)
        fig.text(
            col_x,
            top + 0.01,
            col_title,
            ha="center",
            va="bottom",
            fontsize=PDF_FS_FIG_TITLE,
            transform=fig.transFigure,
        )

    if footer_line:
        fig.text(0.5, 0.01, footer_line, ha="center", va="bottom", fontsize=PDF_FS_FIG_FOOTER, color="0.32")
    return fig


# Backward-compatible alias (Pipeline C used to require exactly three fields).
build_figure_multivariate_composite_triplet = build_figure_multivariate_composite_row


def build_figure_transition_probabilities(ds: xr.Dataset, *, fig_num: list[int] | None) -> Figure | None:
    if "transition_prob" not in ds.data_vars:
        return None
    tp = ds["transition_prob"]
    arr = np.asarray(tp.values, dtype=float)
    vmax = float(np.nanmax(arr))
    n = bump_figure_number(fig_num)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor="white")
    im = ax.imshow(arr, cmap="magma", vmin=0, vmax=vmax if np.isfinite(vmax) and vmax > 0 else 1e-6)
    t_main = "Transition probabilities (row-normalised)"
    ax.set_title(f"Figure {n} — {t_main}" if n else t_main, fontsize=PDF_FS_FIG_TITLE)
    ax.set_xlabel("to regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_ylabel("from regime", fontsize=PDF_FS_FIG_AXIS)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_yticks(range(arr.shape[0]))
    ax.set_xticklabels([str(int(x)) for x in tp.coords["regime_to"].values])
    ax.set_yticklabels([str(int(x)) for x in tp.coords["regime_from"].values])
    cb4 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb4.ax.tick_params(labelsize=PDF_FS_FIG_TICK)
    ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
    fig.tight_layout()
    return fig

