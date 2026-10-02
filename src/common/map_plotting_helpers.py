"""Shared lon/lat map styling: optional Contextily basemap under raster data."""

from __future__ import annotations

from typing import Any


def add_natural_earth_coastlines_and_borders(
    ax,
    *,
    scale: str = "50m",
    zorder: float = 1.0,
    coastline_color: str = "0.25",
    border_color: str = "0.45",
    coastline_linewidth: float = 0.6,
    border_linewidth: float = 0.45,
    border_linestyle: str | tuple[float, tuple[float, ...]] = ":",
) -> bool:
    """Add Natural Earth coastlines and country borders on a Cartopy ``GeoAxes``.

    Expects ``ax`` to use a geographic CRS (e.g. ``cartopy.crs.PlateCarree()``).
    Draws vector features from Cartopy's bundled Natural Earth interface.

    Returns False if Cartopy is missing or ``ax`` has no ``projection`` (plain
    Matplotlib axes). Never raises.
    """
    try:
        import cartopy.feature as cfeature
    except Exception:
        return False
    if not hasattr(ax, "projection"):
        return False
    try:
        ax.add_feature(
            cfeature.COASTLINE.with_scale(scale),
            zorder=zorder,
            edgecolor=coastline_color,
            facecolor="none",
            linewidth=coastline_linewidth,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale(scale),
            zorder=zorder,
            edgecolor=border_color,
            facecolor="none",
            linewidth=border_linewidth,
            linestyle=border_linestyle,
        )
    except Exception:
        return False
    return True


def add_cartopy_land_ocean_basemap(
    ax,
    *,
    extent: list[float] | tuple[float, float, float, float] | None = None,
    zorder: float = 0,
    linewidth: float = 0.6,
    land_color: str = "#f2f2f2",
    ocean_color: str = "#ffffff",
    coastline_color: str = "0.2",
    border_color: str = "0.2",
) -> None:
    """Cartopy-only basemap: filled land/ocean/lakes + coastlines/borders, no network.

    Expects ``ax`` to be a Cartopy ``GeoAxes``. Use ``extent=[lon_min, lon_max,
    lat_min, lat_max]`` to match your data domain (PlateCarree).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=ocean_color, edgecolor="none", zorder=zorder)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=land_color, edgecolor="none", zorder=zorder + 0.1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=ocean_color, edgecolor="none", zorder=zorder + 0.2)

    ax.coastlines(resolution="50m", linewidth=linewidth, color=coastline_color, zorder=zorder + 1.0)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=linewidth * 0.8,
        edgecolor=border_color,
        facecolor="none",
        zorder=zorder + 1.1,
    )

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())


def _contextily_crs(ax: Any, crs: str | Any) -> Any:
    """Use the Cartopy axis projection when present; otherwise keep ``crs``."""
    proj = getattr(ax, "projection", None)
    if proj is not None:
        return proj
    return crs


def add_contextily_basemap_under_spatial_raster(
    ax,
    *,
    crs: str | Any = "EPSG:4326",
    source: Any | None = None,
    basemap_zorder: int = 0,
    data_alpha: float = 0.78,
    data_zorder: int = 2,
    attribution_size: int = 6,
) -> bool:
    """Place a Contextily tile basemap under gridded map data on ``ax``.

    Expects axis limits in degrees (WGS84-style longitude / latitude). After
    plotting with ``xarray`` or ``pcolormesh``, call this so raster map layers
    are slightly transparent and sit above the basemap.

    If ``ax`` is a Cartopy ``GeoAxes``, ``crs`` is taken from ``ax.projection``
    for ``contextily.add_basemap`` so tiles align with ``transform=`` on your
    plot calls (typically ``PlateCarree()``).

    Returns True if a basemap was added, False if Contextily is unavailable or
    tile fetch/plot fails (offline runs, bad extent, missing .sif deps, etc.).
    Never raises: safe for batch / Singularity images without Contextily.
    """
    try:
        import contextily as cx  # type: ignore[import-not-found]
        from matplotlib.collections import QuadMesh
    except Exception:
        # Not installed, incomplete image, or broken native deps (common on HPC .sif)
        return False

    try:
        if source is None:
            source = cx.providers.CartoDB.Positron

        for art in list(ax.collections):
            if isinstance(art, QuadMesh):
                art.set_alpha(data_alpha)
                art.set_zorder(data_zorder)
        for im in ax.images:
            im.set_alpha(data_alpha)
            im.set_zorder(data_zorder)

        cx.add_basemap(
            ax,
            crs=_contextily_crs(ax, crs),
            source=source,
            zorder=basemap_zorder,
            attribution_size=attribution_size,
        )
    except Exception:
        return False
    return True
