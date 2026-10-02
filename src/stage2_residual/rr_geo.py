from __future__ import annotations

"""Shared geographic-overlay helpers for residual regime figures.

Thin wrapper around ``common.geo`` that:
  - resolves the analysis bbox (B0..B5) for a region/step,
  - creates map axes with Cartopy when available (PlateCarree), falling back
    to plain Matplotlib axes,
  - draws coastlines + European country borders, emphasises the German
    national outline, and marks the Germany AOI box.

All overlay functions are no-ops (or graceful fallbacks) when Cartopy is not
installed, so figures still render in minimal environments.
"""

from pathlib import Path

import numpy as np

try:
    from common.geo import (
        cartopy_lines_on_top,
        cartopy_modules,
        cartopy_underlay,
        draw_aoi_box,
        resolve_map_extent,
        resolve_region_aoi_bbox,
    )
except Exception:  # pragma: no cover - regime_plotting always present on HPC
    cartopy_modules = lambda: (None, None)  # noqa: E731
    cartopy_underlay = None
    cartopy_lines_on_top = None
    draw_aoi_box = None
    resolve_map_extent = None
    resolve_region_aoi_bbox = lambda *a, **k: None  # noqa: E731


def load_bbox_for_step(module_root: Path, region: str, bbox_step: str) -> dict[str, float] | None:
    """Return the {N,W,S,E} bbox dict for a region/step from bbox_steps_<region>.yaml."""
    try:
        from common.preprocessing_helpers import _load_bbox_steps_yaml
    except Exception:
        return None
    cfg_name = (
        "bbox_steps_northamerica.yaml"
        if str(region).strip().lower() == "northamerica"
        else "bbox_steps_europe.yaml"
    )
    path = Path(module_root) / "src" / cfg_name
    steps = _load_bbox_steps_yaml(path)
    entry = steps.get(bbox_step)
    if not entry:
        return None
    bbox = entry.get("bbox") if isinstance(entry, dict) else None
    if not bbox:
        return None
    try:
        return {k: float(bbox[k]) for k in ("N", "W", "S", "E")}
    except (KeyError, TypeError, ValueError):
        return None


def make_map_axes(fig, subplot_spec, *, extent: list[float] | None):
    """Create a Cartopy GeoAxes (PlateCarree) for ``subplot_spec`` if possible.

    ``subplot_spec`` may be a GridSpec cell or a 3-int (nrow, ncol, idx).
    Returns (ax, is_geo).  Falls back to a plain Matplotlib axes when Cartopy
    is unavailable.
    """
    ccrs, _ = cartopy_modules()
    if ccrs is not None:
        try:
            if isinstance(subplot_spec, tuple):
                ax = fig.add_subplot(*subplot_spec, projection=ccrs.PlateCarree())
            else:
                ax = fig.add_subplot(subplot_spec, projection=ccrs.PlateCarree())
            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            return ax, True
        except Exception:
            pass
    if isinstance(subplot_spec, tuple):
        ax = fig.add_subplot(*subplot_spec)
    else:
        ax = fig.add_subplot(subplot_spec)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return ax, False


def data_transform():
    """Transform to pass to pcolormesh/contour when drawing on a GeoAxes."""
    ccrs, _ = cartopy_modules()
    return ccrs.PlateCarree() if ccrs is not None else None


def add_boundaries(
    ax,
    *,
    is_geo: bool,
    module_root: Path | None,
    region: str,
    bbox: dict[str, float] | None,
    underlay: bool = False,
) -> None:
    """Draw coastlines + European country borders, German outline, and AOI box.

    On a plain (non-geo) axes only the AOI box is drawn (best effort).
    """
    _, cfeature = cartopy_modules()
    if is_geo and cfeature is not None:
        if underlay and cartopy_underlay is not None and bbox is not None:
            try:
                from common.geo import extent_platecarree_from_bbox
                cartopy_underlay(ax, extent_platecarree_from_bbox(bbox), cfeature=cfeature)
            except Exception:
                pass
        if cartopy_lines_on_top is not None:
            try:
                cartopy_lines_on_top(ax, cfeature=cfeature)
            except Exception:
                pass
        _emphasise_germany(ax)

    # AOI (Germany) bounding box — works on geo and plain axes
    aoi = resolve_region_aoi_bbox(module_root, region) if module_root is not None else None
    if aoi is None:
        aoi = bbox if (region or "").lower() == "europe" else None
    if aoi is not None and draw_aoi_box is not None:
        try:
            ax.plot  # noqa: B018 - ensure axes is usable
            draw_aoi_box(
                ax, aoi,
                edgecolor="k", linewidth=1.3, linestyle="--", zorder=22,
                transform=data_transform() if is_geo else None,
            )
        except Exception:
            pass


def _emphasise_germany(ax) -> None:
    """Overlay a thicker outline on the German national border when available."""
    ccrs, _ = cartopy_modules()
    if ccrs is None:
        return
    try:
        from cartopy.io import shapereader
    except Exception:
        return
    try:
        shp = shapereader.natural_earth(resolution="50m", category="cultural", name="admin_0_countries")
        reader = shapereader.Reader(shp)
        for rec in reader.records():
            name = (rec.attributes.get("NAME") or rec.attributes.get("ADMIN") or "")
            if str(name).lower() in {"germany", "deutschland"}:
                ax.add_geometries(
                    [rec.geometry], crs=ccrs.PlateCarree(),
                    facecolor="none", edgecolor="black", linewidth=1.8, zorder=21,
                )
                break
    except Exception:
        return


def map_extent(bbox: dict[str, float] | None, lat: np.ndarray, lon: np.ndarray) -> list[float] | None:
    """Cartopy-style extent [lon_min, lon_max, lat_min, lat_max] from bbox or data."""
    if resolve_map_extent is not None:
        try:
            return resolve_map_extent(bbox, lat=lat, lon=lon)
        except Exception:
            pass
    return [float(np.nanmin(lon)), float(np.nanmax(lon)),
            float(np.nanmin(lat)), float(np.nanmax(lat))]
