from __future__ import annotations

import numpy as np
import xarray as xr


def harmonize_raw_regime_coord_to_anomaly(raw: xr.DataArray, anom: xr.DataArray) -> xr.DataArray:
    """Align ``regime`` coordinate labels when raw uses 0..K-1 and anomaly uses 1..K (notebook convention)."""
    if "regime" not in raw.dims or "regime" not in anom.dims:
        return raw
    r_raw = np.asarray(raw["regime"].values).astype(int)
    r_an = np.asarray(anom["regime"].values).astype(int)
    an_set = set(r_an.tolist())
    if an_set.issubset(set(r_raw.tolist())):
        return raw
    shifted = r_raw + 1
    if an_set.issubset(set(shifted.tolist())):
        return raw.assign_coords(regime=shifted)
    down = r_raw - 1
    if an_set.issubset(set(down.tolist())) and int(np.nanmin(down)) >= 0:
        return raw.assign_coords(regime=down)
    return raw


def spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    for lat in ("latitude", "lat"):
        for lon in ("longitude", "lon"):
            if lat in da.dims and lon in da.dims:
                return lat, lon
    raise ValueError(f"Could not infer lat/lon from dims={da.dims}")


def _lon_selection_mask(lon: np.ndarray, W: float, E: float) -> np.ndarray:
    """Boolean mask along ``lon`` for bbox west/east (handles 0–360° grids vs -180…180° YAML)."""
    lon = np.asarray(lon, dtype=float)
    W = float(W)
    E = float(E)

    lon_180 = ((lon + 180.0) % 360.0) - 180.0
    if W <= E:
        mask_180 = (lon_180 >= W) & (lon_180 <= E)
    else:
        mask_180 = (lon_180 >= W) | (lon_180 <= E)
    if np.any(mask_180):
        return mask_180

    lon_360 = (lon + 360.0) % 360.0
    w360 = (W + 360.0) % 360.0
    e360 = (E + 360.0) % 360.0
    if w360 <= e360:
        return (lon_360 >= w360) & (lon_360 <= e360)
    return (lon_360 >= w360) | (lon_360 <= e360)


def raw_z500_is_invariant_across_regimes(raw: xr.DataArray, *, atol: float = 1.0) -> bool:
    """True when every regime carries the same 2D Z500 field (within ``atol`` m), e.g. global time mean broadcast."""
    if "regime" not in raw.dims:
        return False
    n = int(raw.sizes["regime"])
    if n <= 1:
        return False
    v0 = np.asarray(raw.isel(regime=0).values, dtype=np.float64)
    for i in range(1, n):
        vi = np.asarray(raw.isel(regime=i).values, dtype=np.float64)
        if v0.shape != vi.shape:
            return False
        if not np.allclose(v0, vi, rtol=0.0, atol=atol, equal_nan=True):
            return False
    return True


def regime_composite_spatial_compatible(
    anom: xr.DataArray, raw: xr.DataArray, *, atol_deg: float = 0.15
) -> bool:
    """True if anomaly and raw composites share the same lat/lon coordinates (same analysis grid)."""
    try:
        la1, lo1 = spatial_dims(anom)
        la2, lo2 = spatial_dims(raw)
        if la1 != la2 or lo1 != lo2:
            return False
        lat1 = np.asarray(anom[la1].values, dtype=float)
        lon1 = np.asarray(anom[lo1].values, dtype=float)
        lat2 = np.asarray(raw[la2].values, dtype=float)
        lon2 = np.asarray(raw[lo2].values, dtype=float)
        if lat1.shape != lat2.shape or lon1.shape != lon2.shape:
            return False
        return bool(
            np.allclose(lat1, lat2, rtol=0.0, atol=atol_deg)
            and np.allclose(lon1, lon2, rtol=0.0, atol=atol_deg)
        )
    except Exception:
        return False


def wrap_lon_sort_lat_da(da: xr.DataArray, lat_dim: str, lon_dim: str) -> xr.DataArray:
    out = da
    lon = out[lon_dim]
    try:
        lon_max = float(lon.max())
    except Exception:
        lon_max = None
    if lon_max is not None and lon_max > 180:
        lon_wrapped = ((lon + 180) % 360) - 180
        out = out.assign_coords({lon_dim: lon_wrapped}).sortby(lon_dim)
    lat_1d = out[lat_dim]
    if lat_1d.size > 1 and float(lat_1d[0]) > float(lat_1d[-1]):
        out = out.sortby(lat_dim)
    return out


def subset_da_to_bbox(
    da: xr.DataArray,
    bbox: dict[str, float],
    *,
    pad_deg: float = 0.5,
) -> xr.DataArray:
    """Crop a lat/lon field to the analysis bbox (with padding) for maps and color scales."""
    lat_dim, lon_dim = spatial_dims(da)
    N = float(bbox["N"]) + pad_deg
    S = float(bbox["S"]) - pad_deg
    W = float(bbox["W"]) - pad_deg
    E = float(bbox["E"]) + pad_deg

    lat = np.asarray(da[lat_dim].values, dtype=float)
    lon = np.asarray(da[lon_dim].values, dtype=float)
    lat_asc = lat.size < 2 or float(lat[0]) < float(lat[-1])
    lat_slice = slice(S, N) if lat_asc else slice(N, S)

    lon_mask = _lon_selection_mask(lon, W, E)
    if not np.any(lon_mask):
        raise ValueError(
            f"bbox lon mask empty (W={W}, E={E}); lon range "
            f"[{float(np.nanmin(lon)):.2f}, {float(np.nanmax(lon)):.2f}]"
        )

    lon_sel = lon[np.where(lon_mask)[0]]
    out = da.sel({lat_dim: lat_slice, lon_dim: lon_sel})
    if out.sizes.get(lat_dim, 0) < 1 or out.sizes.get(lon_dim, 0) < 1:
        raise ValueError(
            f"bbox crop produced empty grid (lat={out.sizes.get(lat_dim, 0)}, "
            f"lon={out.sizes.get(lon_dim, 0)}; bbox N,W,S,E={N},{W},{S},{E})"
        )
    return out


def subset_da_to_bbox_or_raise(da: xr.DataArray, bbox: dict[str, float] | None, *, pad_deg: float = 0.5) -> xr.DataArray:
    """Wrap ``subset_da_to_bbox``; return ``da`` unchanged only when ``bbox`` is None."""
    if bbox is None:
        return da
    lat_dim, lon_dim = spatial_dims(da)
    work = wrap_lon_sort_lat_da(da.astype(np.float64), lat_dim, lon_dim)
    return subset_da_to_bbox(work, bbox, pad_deg=pad_deg)


def align_raw_composite_to_anomaly(
    da_primary: xr.DataArray, raw: xr.DataArray | None
) -> tuple[xr.DataArray | None, bool]:
    """Return ``(raw_on_anom_grid, used_subset_interp)`` for map plotting."""
    if raw is None:
        return None, False
    r = harmonize_raw_regime_coord_to_anomaly(raw.load(), da_primary)
    if regime_composite_spatial_compatible(da_primary, r):
        return r, False
    sub = subset_raw_composite_to_anomaly_grid(da_primary, r)
    if sub is not None:
        return sub, True
    return None, False


def subset_raw_composite_to_anomaly_grid(anom: xr.DataArray, raw: xr.DataArray) -> xr.DataArray | None:
    """Use a full-domain (e.g. B0) raw regime composite on a cropped anomaly grid via nearest-neighbour interp."""
    try:
        la_a, lo_a = spatial_dims(anom)
        la_r, lo_r = spatial_dims(raw)
        lat_a = np.asarray(anom[la_a].values, dtype=float)
        lon_a = np.asarray(anom[lo_a].values, dtype=float)
        lat_r = np.asarray(raw[la_r].values, dtype=float)
        lon_r = np.asarray(raw[lo_r].values, dtype=float)
        lat_rm = float(np.nanmin(lat_r)), float(np.nanmax(lat_r))
        lon_rv = lon_r.copy()
        if float(np.nanmax(lon_rv)) > 180.0 or float(np.nanmin(lon_rv)) > 180.0:
            lon_rv = ((lon_rv + 180.0) % 360.0) - 180.0
        lon_rm = float(np.nanmin(lon_rv)), float(np.nanmax(lon_rv))
        lat_am = float(np.nanmin(lat_a)), float(np.nanmax(lat_a))
        lon_av = lon_a.copy()
        if float(np.nanmax(lon_av)) > 180.0 or float(np.nanmin(lon_av)) > 180.0:
            lon_av = ((lon_av + 180.0) % 360.0) - 180.0
        lon_am = float(np.nanmin(lon_av)), float(np.nanmax(lon_av))
        tol = 2.5
        if lat_am[0] < lat_rm[0] - tol or lat_am[1] > lat_rm[1] + tol:
            return None
        if lon_am[0] < lon_rm[0] - tol or lon_am[1] > lon_rm[1] + tol:
            return None
        raw_r = raw.astype(np.float64)
        if (la_r, lo_r) != (la_a, lo_a):
            raw_r = raw_r.rename({la_r: la_a, lo_r: lo_a})
        raw_r = harmonize_raw_regime_coord_to_anomaly(
            wrap_lon_sort_lat_da(raw_r.load(), la_a, lo_a),
            anom.astype(np.float64),
        )
        anom_s = wrap_lon_sort_lat_da(anom.astype(np.float64).load(), la_a, lo_a)
        out = raw_r.interp({la_a: anom_s[la_a], lo_a: anom_s[lo_a]}, method="nearest")
        out = out.assign_coords({la_a: anom_s[la_a], lo_a: anom_s[lo_a]})
        if "regime" in out.dims and "regime" in anom_s.dims:
            r_a = np.asarray(anom_s["regime"].values).astype(int)
            r_o = set(np.asarray(out["regime"].values).astype(int))
            if not set(r_a).issubset(r_o):
                return None
            out = out.sel(regime=r_a)
        if not regime_composite_spatial_compatible(anom_s, out):
            return None
        return out
    except Exception:
        return None
