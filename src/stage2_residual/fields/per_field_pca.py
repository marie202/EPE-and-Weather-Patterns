from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from common.preprocessing_helpers import PreprocessConfig, infer_core_dims


@dataclass(frozen=True)
class PerFieldPcaResult:
    field_key: str
    pca_path: Path
    retained_modes: int
    source_preprocessed: Path
    reused: bool = False


def _infer_lat_lon_from_preprocessed(ds: xr.Dataset) -> tuple[str, str, np.ndarray, np.ndarray]:
    for var_name in ("weighted_field", "field_norm"):
        if var_name in ds:
            dims = infer_core_dims(ds[var_name], cfg=PreprocessConfig())
            lat_dim = dims["lat"]
            lon_dim = dims["lon"]
            return (
                lat_dim,
                lon_dim,
                np.asarray(ds[var_name][lat_dim].values),
                np.asarray(ds[var_name][lon_dim].values),
            )
    lat = ds.get("lat", ds.get("latitude"))
    lon = ds.get("lon", ds.get("longitude"))
    if lat is None or lon is None:
        raise KeyError("Could not infer lat/lon from preprocessed dataset.")
    return "lat", "lon", np.asarray(lat.values), np.asarray(lon.values)


def run_per_field_pca(
    *,
    field_key: str,
    in_preprocessed_path: Path,
    out_pca_path: Path,
    var_threshold: float,
    max_modes: int | None = None,
    overwrite: bool = False,
) -> PerFieldPcaResult:
    if out_pca_path.is_file() and not overwrite:
        with xr.open_dataset(out_pca_path, chunks=None) as ds:
            retained = int(ds.attrs.get("k", ds["PCs_reduced"].sizes.get("mode", 0)))
        return PerFieldPcaResult(
            field_key=field_key,
            pca_path=out_pca_path,
            retained_modes=retained,
            source_preprocessed=in_preprocessed_path,
            reused=True,
        )

    with xr.open_dataset(in_preprocessed_path, chunks=None) as ds:
        if "X_matrix" not in ds:
            raise KeyError(f"Expected X_matrix in {in_preprocessed_path}, found vars={list(ds.data_vars)}")
        X_da = ds["X_matrix"].load()
        time_dim = X_da.dims[0]
        time_coord = X_da[time_dim]
        lat_dim, lon_dim, lat, lon = _infer_lat_lon_from_preprocessed(ds)
        latitude_space = ds.get("latitude_space")
        longitude_space = ds.get("longitude_space")
        X = np.asarray(X_da.values, dtype=np.float32)

    X_mean = X.mean(axis=0, keepdims=True)
    X_centered = X - X_mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    PCs = U * S
    EOFs_space = Vt
    if X.shape[0] > 1:
        eigenvalues = (S**2) / float(X.shape[0] - 1)
        explained = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else np.zeros_like(eigenvalues)
    else:
        explained = np.ones(S.shape, dtype=np.float32)
    cumulative = np.cumsum(explained)
    retained = int(np.searchsorted(cumulative, float(var_threshold)) + 1)
    retained = max(1, min(retained, PCs.shape[1]))
    if max_modes is not None and retained > max_modes:
        print(f"[per_field_pca] {field_key}: capping {retained} modes → {max_modes} (max_modes limit)")
        retained = max_modes

    pcs_reduced = PCs[:, :retained].astype(np.float32)
    eofs_reduced = EOFs_space[:retained, :].astype(np.float32)
    eof_maps = eofs_reduced.reshape((retained, lat.shape[0], lon.shape[0])).astype(np.float32)
    mode = np.arange(retained, dtype=int)
    space_coord = X_da["space"].values if "space" in X_da.coords else np.arange(X.shape[1], dtype=int)

    ds_out = xr.Dataset(
        data_vars={
            "PCs_reduced": xr.DataArray(
                pcs_reduced,
                dims=[time_dim, "mode"],
                coords={time_dim: time_coord, "mode": mode},
                name="PCs_reduced",
            ),
            "EOFs_space": xr.DataArray(
                eofs_reduced,
                dims=["mode", "space"],
                coords={"mode": mode, "space": space_coord},
                name="EOFs_space",
            ),
            "EOF_maps": xr.DataArray(
                eof_maps,
                dims=["mode", lat_dim, lon_dim],
                coords={"mode": mode, lat_dim: lat, lon_dim: lon},
                name="EOF_maps",
            ),
            "explained_variance_ratio": xr.DataArray(
                explained[:retained].astype(np.float32),
                dims=["mode"],
                coords={"mode": mode},
                name="explained_variance_ratio",
            ),
            "cumulative_explained_variance": xr.DataArray(
                cumulative[:retained].astype(np.float32),
                dims=["mode"],
                coords={"mode": mode},
                name="cumulative_explained_variance",
            ),
        },
        attrs={
            "workflow_family": "multifield",
            "field_key": field_key,
            "step": "07_per_field_pca",
            "source_preprocessed": str(in_preprocessed_path),
            "var_threshold": float(var_threshold),
            "k": int(retained),
        },
    )
    if latitude_space is not None:
        ds_out["latitude_space"] = latitude_space.astype("float32")
    if longitude_space is not None:
        ds_out["longitude_space"] = longitude_space.astype("float32")
    out_pca_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_pca_path)
    return PerFieldPcaResult(
        field_key=field_key,
        pca_path=out_pca_path,
        retained_modes=int(retained),
        source_preprocessed=in_preprocessed_path,
        reused=False,
    )
