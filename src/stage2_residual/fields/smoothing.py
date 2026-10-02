from __future__ import annotations

import numpy as np
import xarray as xr

from common.preprocessing_helpers import infer_core_dims, pick_coord_name, PreprocessConfig


def _gaussian_filter_nd(arr: np.ndarray, sigma_lat: float, sigma_lon: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Gaussian smoothing requires scipy.ndimage.gaussian_filter.") from exc
    return gaussian_filter(arr, sigma=(sigma_lat, sigma_lon), mode="nearest")


def apply_spatial_smoothing(
    da: xr.DataArray,
    *,
    method: str,
    gaussian_sigma: float = 1.0,
    coarsen_factor: int = 0,
) -> xr.DataArray:
    """Apply optional spatial filtering before PCA.

    The filter acts on lat/lon only and preserves the time axis.
    """

    method = str(method).strip().lower()
    if method in {"", "none"}:
        return da

    cfg = PreprocessConfig()
    dims = infer_core_dims(da, cfg=cfg)
    lat_dim = dims["lat"]
    lon_dim = dims["lon"]

    if method == "gaussian":
        sigma = float(gaussian_sigma)
        if sigma <= 0:
            return da
        out = xr.apply_ufunc(
            _gaussian_filter_nd,
            da,
            input_core_dims=[[lat_dim, lon_dim]],
            output_core_dims=[[lat_dim, lon_dim]],
            vectorize=True,
            dask="parallelized",
            kwargs={"sigma_lat": sigma, "sigma_lon": sigma},
            output_dtypes=[da.dtype],
        )
        out.attrs = dict(da.attrs)
        out.attrs["smoothing_method"] = "gaussian"
        out.attrs["gaussian_sigma_gridcells"] = float(sigma)
        return out.rename(da.name or "smoothed_field")

    if method == "coarsen":
        factor = int(coarsen_factor)
        if factor <= 1:
            return da
        coarsened = da.coarsen({lat_dim: factor, lon_dim: factor}, boundary="trim").mean()
        restored = coarsened.interp({lat_dim: da[lat_dim], lon_dim: da[lon_dim]}, method="linear")
        restored.attrs = dict(da.attrs)
        restored.attrs["smoothing_method"] = "coarsen"
        restored.attrs["coarsen_factor"] = int(factor)
        return restored.rename(da.name or "smoothed_field")

    raise ValueError(f"Unknown smoothing method {method!r}. Expected one of: none, gaussian, coarsen.")
