from __future__ import annotations

from pathlib import Path

import numpy as np


# Region -> regions.yaml study-area key used for the AOI indicator overlay on
# composite maps. ``europe`` runs target Germany; ``northamerica`` targets the
# southern-Ontario window (regions.yaml key ``ontario``; on-disk study_area
# remains ``ontario_south``).
_REGION_TO_AOI_KEY: dict[str, str] = {
    "europe": "germany",
    "northamerica": "ontario",
}


def cartopy_modules():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except Exception:
        return None, None


def extent_platecarree(lat: np.ndarray, lon: np.ndarray) -> list[float]:
    lon_min, lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    return [lon_min, lon_max, lat_min, lat_max]


def extent_platecarree_from_bbox(bbox: dict[str, float]) -> list[float]:
    """Cartopy ``set_extent`` order: [lon_min, lon_max, lat_min, lat_max] from N,W,S,E."""
    return [
        float(bbox["W"]),
        float(bbox["E"]),
        float(bbox["S"]),
        float(bbox["N"]),
    ]


def resolve_map_extent(
    bbox: dict[str, float] | None,
    *,
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    pad: bool = False,
) -> list[float] | None:
    """Prefer YAML/analysis bbox for map limits (not the full B0 grid when a smaller step is plotted).

    Default ``pad=False`` keeps the map extent equal to the bounding box. Padding past
    ``[-180, 180]`` would otherwise wrap across the antimeridian under Cartopy PlateCarree
    (e.g. North America B0 bbox W=-180 padded to -186 becomes +174 and the map shows the
    opposite hemisphere).
    """
    if bbox is not None:
        ext = extent_platecarree_from_bbox(bbox)
        return pad_extent_for_map_margin(ext) if pad else ext
    if lat is not None and lon is not None:
        ext = extent_platecarree(lat, lon)
        return pad_extent_for_map_margin(ext) if pad else ext
    return None


def pad_extent_for_map_margin(
    extent: list[float],
    *,
    frac: float = 0.04,
    min_pad_deg: float = 0.35,
) -> list[float]:
    """Add a small margin to a PlateCarree extent, clamped to ``[-180, 180]`` / ``[-90, 90]``.

    Clamping prevents Cartopy from wrapping the map across the antimeridian when the bbox
    already touches a global edge (e.g. North America B0 with W=-180).
    """
    lon0, lon1, lat0, lat1 = extent
    lon_s = lon1 - lon0
    lat_s = lat1 - lat0
    if lon_s <= 0 or lat_s <= 0:
        return extent
    plon = max(lon_s * frac, min_pad_deg)
    plat = max(lat_s * frac, min_pad_deg)
    return [
        max(lon0 - plon, -180.0),
        min(lon1 + plon, 180.0),
        max(lat0 - plat, -90.0),
        min(lat1 + plat, 90.0),
    ]


def apply_plain_axes_bbox_limits(ax, bbox: dict[str, float] | None, *, pad_deg: float = 0.0) -> None:
    """Set x/y limits on non-Cartopy axes from YAML bbox (W,E,S,N).

    Default ``pad_deg=0`` keeps the axes limits exactly at the bounding box; pass a positive
    value to add a small visual margin (clamped to global PlateCarree ranges).
    """
    if bbox is None:
        return
    ext = extent_platecarree_from_bbox(bbox)
    if pad_deg > 0:
        ext = pad_extent_for_map_margin(ext, min_pad_deg=pad_deg)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])


def cartopy_underlay(ax, extent_map: list[float], *, cfeature) -> None:
    import cartopy.crs as ccrs

    ax.set_extent(extent_map, crs=ccrs.PlateCarree())
    ax.set_facecolor("#e5ecf6")
    try:
        ax.patch.set_facecolor("#e5ecf6")
    except Exception:
        pass
    for scale in ("50m", "110m"):
        try:
            ax.add_feature(
                cfeature.OCEAN.with_scale(scale),
                facecolor="#e5ecf6",
                edgecolor="none",
                zorder=1,
            )
            ax.add_feature(
                cfeature.LAND.with_scale(scale),
                facecolor="#f2f2f2",
                edgecolor="none",
                zorder=2,
            )
            ax.add_feature(
                cfeature.LAKES.with_scale(scale),
                facecolor="#e5ecf6",
                edgecolor="none",
                zorder=3,
            )
            return
        except Exception:
            continue
    try:
        ax.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor="#f0f0f0",
            edgecolor="none",
            zorder=2,
        )
    except Exception:
        pass
    try:
        ax.coastlines(resolution="110m", linewidth=0.75, color="0.35", zorder=3)
    except Exception:
        pass


def cartopy_lines_on_top(ax, *, cfeature, zoutline: float = 18.0) -> None:
    for scale in ("50m", "110m"):
        try:
            ax.coastlines(resolution=scale, linewidth=1.05, color="0.18", zorder=zoutline)
            ax.add_feature(
                cfeature.BORDERS.with_scale(scale),
                linewidth=0.55,
                edgecolor="0.25",
                facecolor="none",
                zorder=zoutline + 0.5,
            )
            return
        except Exception:
            continue


def label_contour_heights(
    ax,
    cs,
    *,
    fontsize: float,
    fmt: str = "%.0f",
    zorder: float = 25.0,
    rightside_up: bool = True,
) -> None:
    """Annotate contour levels (e.g. mean raw Z500 in m).

    Call after ``cartopy_lines_on_top`` so labels sit above coastlines/borders.
    """
    try:
        from matplotlib import patheffects
    except Exception:
        patheffects = None  # type: ignore[assignment]

    def _style_label_texts(labels) -> int:
        if not labels:
            return 0
        n = 0
        for text in labels:
            text.set_zorder(zorder)
            text.set_color("k")
            if patheffects is not None:
                text.set_path_effects(
                    [patheffects.withStroke(linewidth=2.2, foreground="white")]
                )
            n += 1
        return n

    clabel_attempts = (
        dict(inline=True, fontsize=fontsize, fmt=fmt, rightside_up=rightside_up, colors="k"),
        dict(inline=True, fontsize=fontsize, fmt=fmt, rightside_up=rightside_up, color="k"),
        dict(inline=False, fontsize=fontsize, fmt=fmt, rightside_up=rightside_up, color="k"),
    )
    for kwargs in clabel_attempts:
        try:
            labels = ax.clabel(cs, **kwargs)
            if _style_label_texts(labels) > 0:
                return
        except (TypeError, ValueError):
            continue
        except Exception:
            continue

    # Cartopy / tight layouts sometimes yield zero inline labels; place one per level.
    try:
        levels = np.asarray(getattr(cs, "levels", []), dtype=float)
    except Exception:
        return
    if levels.size == 0:
        return

    transform = getattr(ax, "projection", None)
    for i, level in enumerate(levels):
        segs = getattr(cs, "allsegs", None)
        if not segs or i >= len(segs) or not segs[i]:
            continue
        seg = max(segs[i], key=len)
        if len(seg) < 2:
            continue
        mid = len(seg) // 2
        x, y = float(seg[mid, 0]), float(seg[mid, 1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        tkw: dict = dict(
            fontsize=fontsize,
            color="k",
            ha="center",
            va="center",
            zorder=zorder,
            clip_on=True,
        )
        if transform is not None:
            tkw["transform"] = transform
        try:
            text = ax.text(x, y, fmt % level, **tkw)
            if patheffects is not None:
                text.set_path_effects(
                    [patheffects.withStroke(linewidth=2.2, foreground="white")]
                )
        except Exception:
            continue


def is_bottom_panel_row(
    panel_row: int, panel_col: int, nrow: int, ncol: int, n_panels: int | None,
) -> bool:
    """True if this panel is the bottom-most occupied one in its column.

    Panels fill row-major (row 0 left-to-right, then row 1, ...) up to ``n_panels`` total,
    same order ``fig.add_subplot(nrow, ncol, i + 1)`` is called in. When ``n_panels`` doesn't
    fill the grid evenly (e.g. 7 panels in a 2x4 grid), the last row is short, so the trailing
    columns' bottom-most panel actually sits in the row *above* -- that panel needs the x-axis
    (it has no panel below it), not row ``nrow - 1`` unconditionally.
    """
    if n_panels is None or n_panels >= nrow * ncol:
        return panel_row == nrow - 1
    remainder = n_panels % ncol
    last_row_for_col = nrow - 1 if (remainder == 0 or panel_col < remainder) else nrow - 2
    return panel_row == last_row_for_col


def configure_regime_composite_gridlines(
    ax,
    *,
    panel_row: int,
    panel_col: int,
    nrow: int,
    ncol: int,
    n_panels: int | None = None,
    show_left: bool | None = None,
    show_right: bool | None = None,
    show_bottom: bool | None = None,
    linewidth: float = 0.5,
    alpha: float = 0.35,
    tick_labelsize: float | int = 8,
) -> object:
    """Cartopy gridlines with tick labels only on the outer-bottom/outer-left edges of a
    regime mosaic -- never on the right or top, regardless of panel position (the interior
    gridlines themselves still draw on every panel, giving the visual effect of "ticks" on
    panels without labels).

    ``n_panels``, if given, is the actual number of data panels (may be less than
    ``nrow * ncol`` for a ragged last row) -- see ``is_bottom_panel_row``.
    """
    gl = ax.gridlines(draw_labels=True, linewidth=linewidth, alpha=alpha)
    gl.top_labels = False
    gl.bottom_labels = (
        show_bottom
        if show_bottom is not None
        else is_bottom_panel_row(panel_row, panel_col, nrow, ncol, n_panels)
    )
    gl.left_labels = show_left if show_left is not None else panel_col == 0
    gl.right_labels = show_right if show_right is not None else False
    gl.xlabel_style = {"size": tick_labelsize}
    gl.ylabel_style = {"size": tick_labelsize}
    return gl


def configure_plain_composite_axis_ticks(
    ax,
    *,
    panel_row: int,
    panel_col: int,
    nrow: int,
    ncol: int,
    n_panels: int | None = None,
    tick_labelsize: float | int = 8,
) -> None:
    """Matplotlib axis ticks: labels only on the outer-bottom/outer-left edges of a regime
    mosaic; every other panel still gets tick marks (no labels); right/top never get ticks or
    labels, on any panel.

    ``n_panels``, if given, is the actual number of data panels (may be less than
    ``nrow * ncol`` for a ragged last row) -- see ``is_bottom_panel_row``.
    """
    is_bottom = is_bottom_panel_row(panel_row, panel_col, nrow, ncol, n_panels)
    is_left = panel_col == 0
    ax.tick_params(
        axis="x",
        which="both",
        bottom=True,
        top=False,
        labelbottom=is_bottom,
        labeltop=False,
        labelsize=tick_labelsize,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=True,
        right=False,
        labelleft=is_left,
        labelright=False,
        labelsize=tick_labelsize,
    )


def draw_aoi_box(
    ax,
    bbox: dict[str, float] | None,
    *,
    edgecolor: str = "k",
    linewidth: float = 1.1,
    linestyle: str = "-",
    zorder: float = 20.0,
    transform=None,
) -> None:
    """Draw a thin rectangle at the AOI bbox edges on top of an existing axes.

    No-op when ``bbox`` is ``None``. Works for both Cartopy ``GeoAxes`` (when
    ``transform`` is a PlateCarree projection) and plain Matplotlib axes. The
    rectangle is unfilled and sits above coastline/border features by default
    (``zorder=20``).
    """
    if bbox is None:
        return
    try:
        n = float(bbox["N"])
        s = float(bbox["S"])
        w = float(bbox["W"])
        e = float(bbox["E"])
    except (KeyError, TypeError, ValueError):
        return
    if not (np.isfinite(n) and np.isfinite(s) and np.isfinite(w) and np.isfinite(e)):
        return
    xs = [w, e, e, w, w]
    ys = [s, s, n, n, s]
    plot_kwargs: dict = dict(
        color=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
        solid_capstyle="butt",
        solid_joinstyle="miter",
    )
    if transform is not None:
        plot_kwargs["transform"] = transform
    ax.plot(xs, ys, **plot_kwargs)


def resolve_region_aoi_bbox(
    module_root: Path | str | None,
    region: str | None,
) -> dict[str, float] | None:
    """Return the AOI bbox (``regions.yaml``) for the given region key.

    ``region`` is the top-level run region (``europe`` / ``northamerica``); the
    function maps it to the corresponding ``regions.yaml`` study-area key
    (``germany`` / ``ontario``) and returns its bbox dict (``N,W,S,E``). Returns
    ``None`` if the region is unknown or the YAML cannot be loaded.
    """
    if not region:
        return None
    key = _REGION_TO_AOI_KEY.get(str(region).strip().lower())
    if key is None:
        return None
    try:
        from epe_preprocessing.region_config import load_region_config, resolve_step_root  # noqa: WPS433
    except Exception:
        return None
    root: Path | None = None
    if module_root is not None:
        root = Path(module_root).resolve()
    else:
        try:
            root = resolve_step_root(Path.cwd())
        except Exception:
            root = None
    if root is None:
        return None
    candidates: list[Path] = [root]
    if (root / "1_Detect_Weather_Patterns" / "src" / "epe_preprocessing" / "regions.yaml").is_file():
        candidates.append(root / "1_Detect_Weather_Patterns")
    if root.name != "1_Detect_Weather_Patterns" and (root.parent / "1_Detect_Weather_Patterns").is_dir():
        candidates.append(root.parent / "1_Detect_Weather_Patterns")
    for step_root in candidates:
        try:
            cfg = load_region_config(step_root=step_root, region_id=key)
        except Exception:
            continue
        bbox = getattr(cfg, "bbox", None)
        if not bbox:
            continue
        try:
            return {k: float(bbox[k]) for k in ("N", "W", "S", "E")}
        except (KeyError, TypeError, ValueError):
            continue
    return None
