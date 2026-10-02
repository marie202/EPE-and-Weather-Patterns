from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _standardize_pcs(pc_da: xr.DataArray) -> xr.DataArray:
    mean = pc_da.mean(dim=pc_da.dims[0], skipna=True)
    std = pc_da.std(dim=pc_da.dims[0], skipna=True)
    std = xr.where(std == 0, 1.0, std)
    out = (pc_da - mean) / std
    out.attrs = dict(pc_da.attrs)
    out.attrs["standardized_to_unit_variance"] = "true"
    return out


def build_joint_feature_dataset(
    *,
    pca_paths_by_field: dict[str, Path],
    out_path: Path,
    overwrite: bool = False,
    regime_set_label: str | None = None,
    regime_set_id: str | None = None,
) -> Path:
    if out_path.is_file() and not overwrite:
        return out_path

    loaded: dict[str, xr.DataArray] = {}
    for field_key, pca_path in pca_paths_by_field.items():
        with xr.open_dataset(pca_path, chunks=None) as ds:
            if "PCs_reduced" not in ds:
                raise KeyError(f"Expected PCs_reduced in {pca_path}, found vars={list(ds.data_vars)}")
            loaded[field_key] = ds["PCs_reduced"].load()

    time_indexes = [pd.Index(np.asarray(da[da.dims[0]].values)) for da in loaded.values()]
    common_time = time_indexes[0]
    for idx in time_indexes[1:]:
        common_time = common_time.intersection(idx)
    if len(common_time) == 0:
        raise ValueError("No common timestamps across per-field PCA outputs.")

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    feature_fields: list[str] = []
    feature_modes: list[int] = []
    for field_key, da in loaded.items():
        time_dim = da.dims[0]
        std_da = _standardize_pcs(da.sel({time_dim: common_time}))
        arr = np.asarray(std_da.values, dtype=np.float32)
        blocks.append(arr)
        n_mode = arr.shape[1]
        feature_names.extend(f"{field_key}_pc{idx + 1:02d}" for idx in range(n_mode))
        feature_fields.extend([field_key] * n_mode)
        feature_modes.extend(list(np.arange(1, n_mode + 1, dtype=int)))

    X_joint = np.concatenate(blocks, axis=1).astype(np.float32)
    time_name = loaded[next(iter(loaded))].dims[0]
    feature = np.arange(X_joint.shape[1], dtype=int)
    ds_out = xr.Dataset(
        data_vars={
            "joint_features": xr.DataArray(
                X_joint,
                dims=[time_name, "feature"],
                coords={time_name: common_time.values, "feature": feature},
                name="joint_features",
            ),
            "feature_mode": xr.DataArray(
                np.asarray(feature_modes, dtype=int),
                dims=["feature"],
                coords={"feature": feature},
                name="feature_mode",
            ),
        },
        coords={
            "feature_name": ("feature", np.asarray(feature_names, dtype=object)),
            "feature_field": ("feature", np.asarray(feature_fields, dtype=object)),
        },
        attrs={
            "workflow_family": "multifield",
            "step": "08_joint_features",
            "source_pca_paths": "|".join(f"{k}:{v}" for k, v in pca_paths_by_field.items()),
            "source_fields": ",".join(pca_paths_by_field.keys()),
            **(
                {"regime_set_label": regime_set_label, "regime_set_id": regime_set_id}
                if regime_set_label is not None
                else {}
            ),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    return out_path
