from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _load_labels_and_metrics(path: Path) -> dict[str, object]:
    with xr.open_dataset(path, chunks=None) as ds:
        if "regime_labels" not in ds:
            raise KeyError(f"Expected regime_labels in {path}, found vars={list(ds.data_vars)}")
        labels = np.asarray(ds["regime_labels"].values, dtype=int)
        out: dict[str, object] = {
            "labels": labels,
            "final_k": int(ds.attrs.get("final_k", int(np.max(labels)) if labels.size else 0)),
        }
        for name in ("frequency", "persistence_mean", "transition_counts", "transition_prob"):
            if name in ds:
                out[name] = ds[name].load()
        return out


def _silhouette_from_dataset(path: Path, feature_var: str, *, labels_path: Path | None = None) -> float:
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except Exception:
        return float("nan")
    with xr.open_dataset(path, chunks=None) as ds:
        if feature_var not in ds:
            return float("nan")
        X = np.asarray(ds[feature_var].values, dtype=float)
    if labels_path is None:
        labels_path = path
    with xr.open_dataset(labels_path, chunks=None) as ds_lab:
        if "regime_labels" not in ds_lab:
            return float("nan")
        labels = np.asarray(ds_lab["regime_labels"].values, dtype=int)
    if X.shape[0] != labels.shape[0] or len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels))


def write_baseline_comparison(
    *,
    multifield_regimes_path: Path,
    baseline_regimes_path: Path | None,
    multifield_joint_features_path: Path | None,
    baseline_pca_path: Path | None,
    out_path: Path,
    overwrite: bool = False,
) -> Path | None:
    if baseline_regimes_path is None or not baseline_regimes_path.is_file():
        return None
    if out_path.is_file() and not overwrite:
        return out_path

    mf = _load_labels_and_metrics(multifield_regimes_path)
    base = _load_labels_and_metrics(baseline_regimes_path)
    ds_out = xr.Dataset(
        attrs={
            "workflow_family": "multifield",
            "step": "12_baseline_comparison",
            "source_multifield_regimes": str(multifield_regimes_path),
            "source_baseline_regimes": str(baseline_regimes_path),
            "source_baseline_pca": str(baseline_pca_path) if baseline_pca_path is not None else "",
            "source_multifield_joint_features": (
                str(multifield_joint_features_path) if multifield_joint_features_path is not None else ""
            ),
        }
    )
    if "frequency" in mf:
        ds_out["multifield_frequency"] = mf["frequency"]  # type: ignore[assignment]
    if "frequency" in base:
        ds_out["baseline_frequency"] = base["frequency"]  # type: ignore[assignment]
    if "persistence_mean" in mf:
        ds_out["multifield_persistence_mean"] = mf["persistence_mean"]  # type: ignore[assignment]
    if "persistence_mean" in base:
        ds_out["baseline_persistence_mean"] = base["persistence_mean"]  # type: ignore[assignment]
    if "transition_counts" in mf:
        ds_out["multifield_transition_counts"] = mf["transition_counts"]  # type: ignore[assignment]
    if "transition_counts" in base:
        ds_out["baseline_transition_counts"] = base["transition_counts"]  # type: ignore[assignment]
    if "transition_prob" in mf:
        ds_out["multifield_transition_prob"] = mf["transition_prob"]  # type: ignore[assignment]
    if "transition_prob" in base:
        ds_out["baseline_transition_prob"] = base["transition_prob"]  # type: ignore[assignment]

    mf_labels = np.asarray(mf["labels"], dtype=np.int16)
    base_labels = np.asarray(base["labels"], dtype=np.int16)
    ds_out["multifield_regime_labels"] = xr.DataArray(mf_labels, dims=["time_index"], coords={"time_index": np.arange(mf_labels.size)})
    ds_out["baseline_regime_labels"] = xr.DataArray(base_labels, dims=["time_index"], coords={"time_index": np.arange(base_labels.size)})

    if multifield_joint_features_path is not None and multifield_joint_features_path.is_file():
        ds_out.attrs["multifield_silhouette_from_joint_features"] = str(
            _silhouette_from_dataset(
                multifield_joint_features_path,
                feature_var="joint_features",
                labels_path=multifield_regimes_path,
            )
        )
    if baseline_pca_path is not None and baseline_pca_path.is_file():
        with xr.open_dataset(baseline_pca_path, chunks=None) as ds_pca:
            if "PCs_reduced" in ds_pca and base_labels.size == ds_pca["PCs_reduced"].shape[0]:
                try:
                    from sklearn.metrics import silhouette_score  # type: ignore

                    ds_out.attrs["baseline_silhouette_from_pca"] = str(
                        float(silhouette_score(np.asarray(ds_pca["PCs_reduced"].values), base_labels))
                    )
                except Exception:
                    pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    return out_path
