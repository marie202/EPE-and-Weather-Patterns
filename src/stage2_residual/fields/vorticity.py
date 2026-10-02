from __future__ import annotations

import numpy as np
import xarray as xr

_U_CANDIDATES = ("u", "u_component_of_wind", "U")
_V_CANDIDATES = ("v", "v_component_of_wind", "V")


def _pick_var(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise KeyError(f"Expected one of {candidates}, found {list(ds.data_vars)}")


def relative_vorticity_from_uv(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    """Compute relative vorticity (s⁻¹) from zonal/meridional wind on a regular lat–lon grid."""

    lat = np.deg2rad(np.asarray(u[lat_dim].values, dtype=np.float64))
    lon = np.deg2rad(np.asarray(u[lon_dim].values, dtype=np.float64))
    a = 6_371_000.0

    u_vals = np.asarray(u.values, dtype=np.float64)
    v_vals = np.asarray(v.values, dtype=np.float64)

    if u_vals.ndim != 3:
        raise ValueError(f"Expected 3D u/v fields (time, lat, lon); got u shape {u_vals.shape}")

    dlon = np.gradient(lon)
    dlat = np.gradient(lat)
    cos_lat = np.cos(lat)
    cos_lat = np.where(cos_lat < 1.0e-8, 1.0e-8, cos_lat)

    dvdx = np.empty_like(v_vals)
    dudy = np.empty_like(u_vals)
    for t in range(u_vals.shape[0]):
        dvdx[t] = np.gradient(v_vals[t], axis=-1) / dlon[None, :] / a / cos_lat[:, None]
        dudy[t] = np.gradient(u_vals[t], axis=-2) / dlat[:, None] / a

    zeta = dvdx - dudy
    out = xr.DataArray(
        zeta.astype(np.float32),
        dims=u.dims,
        coords=u.coords,
        name="vo850",
        attrs={
            "long_name": "relative vorticity",
            "units": "s-1",
            "derived_from": "u/v wind components",
        },
    )
    return out


def open_vo850_or_compute_from_uv(
    ds: xr.Dataset,
    *,
    lat_dim: str,
    lon_dim: str,
    field_spec_candidates: tuple[str, ...],
) -> xr.DataArray:
    """Return vorticity at 850 hPa from ``vo`` if present, else derive from ``u``/``v``."""

    for candidate in field_spec_candidates:
        if candidate in ds.data_vars:
            return ds[candidate]
    u_name = _pick_var(ds, _U_CANDIDATES)
    v_name = _pick_var(ds, _V_CANDIDATES)
    return relative_vorticity_from_uv(ds[u_name], ds[v_name], lat_dim=lat_dim, lon_dim=lon_dim)
