from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def _frequency(labels: np.ndarray, k: int) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=k + 1)[1 : k + 1]
    total = counts.sum()
    if total == 0:
        return np.zeros(k, dtype=np.float32)
    return (counts / total).astype(np.float32)


def _persistence(labels: np.ndarray, k: int) -> np.ndarray:
    out = np.full(k, np.nan, dtype=np.float32)
    for regime in range(1, k + 1):
        runs: list[int] = []
        count = 0
        for value in labels:
            if int(value) == regime:
                count += 1
            elif count:
                runs.append(count)
                count = 0
        if count:
            runs.append(count)
        if runs:
            out[regime - 1] = float(np.mean(runs))
    return out


def _transition_counts(labels: np.ndarray, k: int) -> np.ndarray:
    T = np.zeros((k, k), dtype=np.int32)
    if labels.size < 2:
        return T
    for a, b in zip(labels[:-1], labels[1:]):
        if a > 0 and b > 0:
            T[int(a) - 1, int(b) - 1] += 1
    return T


def _run_numpy_kmeans(
    X: np.ndarray,
    *,
    k: int,
    random_state: int,
    n_init: int = 20,
    max_iter: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(random_state))
    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = float("inf")
    n_samples = X.shape[0]
    if n_samples < k:
        raise ValueError(f"k={k} exceeds sample count n={n_samples}")

    for _ in range(int(n_init)):
        centers = X[rng.choice(n_samples, size=k, replace=False)].copy()
        labels = np.zeros(n_samples, dtype=int)
        for _iter in range(int(max_iter)):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for idx in range(k):
                members = X[labels == idx]
                if members.size == 0:
                    centers[idx] = X[rng.integers(0, n_samples)]
                else:
                    centers[idx] = members.mean(axis=0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    if best_labels is None or best_centers is None:  # pragma: no cover
        raise RuntimeError("NumPy k-means failed to converge.")
    return best_labels, best_centers


def _regime_metadata_attrs(
    regime_set_label: str | None,
    regime_set_id: str | None,
) -> dict[str, str]:
    attrs: dict[str, str] = {}
    if regime_set_label is not None:
        attrs["regime_set_label"] = regime_set_label
    if regime_set_id is not None:
        attrs["regime_set_id"] = regime_set_id
    return attrs


def run_ward_hierarchical_clustering(
    *,
    joint_features_path: Path,
    out_hc_path: Path,
    overwrite: bool = False,
    regime_set_label: str | None = None,
    regime_set_id: str | None = None,
) -> Path:
    if out_hc_path.is_file() and not overwrite:
        return out_hc_path

    try:
        from scipy.cluster.hierarchy import linkage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ward clustering requires scipy.cluster.hierarchy.linkage.") from exc

    with xr.open_dataset(joint_features_path, chunks=None) as ds:
        if "joint_features" not in ds:
            raise KeyError(f"Expected joint_features in {joint_features_path}, found vars={list(ds.data_vars)}")
        X = np.asarray(ds["joint_features"].values, dtype=np.float64)

    # Use fastcluster if available — significantly faster than scipy for large n
    try:
        import fastcluster  # type: ignore
        Z = fastcluster.linkage_vector(X, method="ward")
        print(f"[ward_hc] Using fastcluster (n={X.shape[0]})")
    except ImportError:
        print(f"[ward_hc] fastcluster not available, using scipy (n={X.shape[0]}, may be slow)")
        Z = linkage(X, method="ward")
    ds_out = xr.Dataset(
        data_vars={
            "ward_linkage": xr.DataArray(
                Z.astype(np.float64),
                dims=["merge", "col"],
                coords={"merge": np.arange(Z.shape[0], dtype=int), "col": np.arange(4, dtype=int)},
                name="ward_linkage",
            )
        },
        attrs={
            "workflow_family": "multifield",
            "step": "08_joint_hierarchical_clustering",
            "source_joint_features": str(joint_features_path),
            **_regime_metadata_attrs(regime_set_label, regime_set_id),
        },
    )
    out_hc_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_hc_path)
    return out_hc_path


def run_kmeans_regimes(
    *,
    joint_features_path: Path,
    out_regimes_path: Path,
    final_k: int,
    random_state: int,
    overwrite: bool = False,
    regime_set_label: str | None = None,
    regime_set_id: str | None = None,
    ward_hc_path: Path | None = None,
) -> Path:
    if out_regimes_path.is_file() and not overwrite:
        return out_regimes_path

    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:
        KMeans = None  # type: ignore[assignment]
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
    except Exception:
        silhouette_score = None  # type: ignore[assignment]

    with xr.open_dataset(joint_features_path, chunks=None) as ds:
        if "joint_features" not in ds:
            raise KeyError(f"Expected joint_features in {joint_features_path}, found vars={list(ds.data_vars)}")
        X_da = ds["joint_features"].load()
        time_dim = X_da.dims[0]
        time_coord = X_da[time_dim]
        feature_names = np.asarray(ds["feature_name"].values, dtype=object)
        feature_fields = np.asarray(ds["feature_field"].values, dtype=object)
        feature_modes = np.asarray(ds["feature_mode"].values, dtype=int)
        X = np.asarray(X_da.values, dtype=np.float64)

    # Compute Ward-derived centroids for initialisation when a linkage file is available.
    ward_init: np.ndarray | None = None
    if ward_hc_path is not None:
        try:
            from scipy.cluster.hierarchy import fcluster  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Ward init requires scipy.cluster.hierarchy.fcluster.") from exc
        with xr.open_dataset(ward_hc_path, chunks=None) as ds_hc:
            Z_ward = np.asarray(ds_hc["ward_linkage"].values, dtype=np.float64)
        ward_assignments = fcluster(Z_ward, int(final_k), criterion="maxclust")  # 1-indexed
        ward_init = np.stack(
            [X[ward_assignments == c].mean(axis=0) for c in range(1, int(final_k) + 1)],
            axis=0,
        )

    if KMeans is not None:
        if ward_init is not None:
            km = KMeans(n_clusters=int(final_k), init=ward_init, n_init=1, random_state=int(random_state))
        else:
            km = KMeans(n_clusters=int(final_k), n_init=50, random_state=int(random_state))
        labels0 = km.fit_predict(X)
        centers = np.asarray(km.cluster_centers_, dtype=np.float32)
    else:
        labels0, centers = _run_numpy_kmeans(X, k=int(final_k), random_state=int(random_state))
    labels = labels0.astype(np.int16) + 1
    # Computed uniformly from labels/centers (rather than reading km.inertia_) so the
    # value is consistent regardless of whether the sklearn or numpy-fallback path ran.
    inertia = float(((X - centers[labels0]) ** 2).sum())
    silhouette = (
        float(silhouette_score(X, labels0))  # type: ignore[misc]
        if silhouette_score is not None and len(np.unique(labels0)) >= 2
        else float("nan")
    )
    T = _transition_counts(labels, int(final_k))
    row_sums = T.sum(axis=1, keepdims=True)
    T_prob = np.zeros_like(T, dtype=np.float32)
    np.divide(T, row_sums, out=T_prob, where=row_sums != 0)

    ds_out = xr.Dataset(
        data_vars={
            "regime_labels": xr.DataArray(
                labels,
                dims=[time_dim],
                coords={time_dim: time_coord},
                name="regime_labels",
            ),
            "cluster_centers": xr.DataArray(
                centers.astype(np.float32),
                dims=["regime", "feature"],
                coords={"regime": np.arange(1, int(final_k) + 1, dtype=int), "feature": np.arange(X.shape[1], dtype=int)},
                name="cluster_centers",
            ),
            "frequency": xr.DataArray(
                _frequency(labels, int(final_k)),
                dims=["regime"],
                coords={"regime": np.arange(1, int(final_k) + 1, dtype=int)},
                name="frequency",
            ),
            "persistence_mean": xr.DataArray(
                _persistence(labels, int(final_k)),
                dims=["regime"],
                coords={"regime": np.arange(1, int(final_k) + 1, dtype=int)},
                name="persistence_mean",
            ),
            "transition_counts": xr.DataArray(
                T,
                dims=["regime_from", "regime_to"],
                coords={
                    "regime_from": np.arange(1, int(final_k) + 1, dtype=int),
                    "regime_to": np.arange(1, int(final_k) + 1, dtype=int),
                },
                name="transition_counts",
            ),
            "transition_prob": xr.DataArray(
                T_prob.astype(np.float32),
                dims=["regime_from", "regime_to"],
                coords={
                    "regime_from": np.arange(1, int(final_k) + 1, dtype=int),
                    "regime_to": np.arange(1, int(final_k) + 1, dtype=int),
                },
                name="transition_prob",
            ),
            "feature_mode": xr.DataArray(
                feature_modes.astype(np.int16),
                dims=["feature"],
                coords={"feature": np.arange(X.shape[1], dtype=int)},
                name="feature_mode",
            ),
        },
        coords={
            "feature_name": ("feature", feature_names),
            "feature_field": ("feature", feature_fields),
        },
        attrs={
            "workflow_family": "multifield",
            "step": "09_joint_kmeans_regimes",
            "source_joint_features": str(joint_features_path),
            "final_k": int(final_k),
            "random_state": int(random_state),
            "silhouette_score": silhouette,
            "inertia": inertia,
            **_regime_metadata_attrs(regime_set_label, regime_set_id),
        },
    )
    out_regimes_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_regimes_path)
    return out_regimes_path


def write_regime_characterisation(
    *,
    regimes_path: Path,
    anomaly_paths_by_field: dict[str, Path],
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    if out_path.is_file() and not overwrite:
        return out_path

    with xr.open_dataset(regimes_path, chunks=None) as ds_reg:
        if "regime_labels" not in ds_reg:
            raise KeyError(f"Expected regime_labels in {regimes_path}, found vars={list(ds_reg.data_vars)}")
        labels_da = ds_reg["regime_labels"].load()
        regime_labels = np.asarray(labels_da.values, dtype=int)
        time_dim = labels_da.dims[0]
        time_coord = labels_da[time_dim]
        final_k = int(ds_reg.attrs.get("final_k", int(np.max(regime_labels))))
        out = xr.Dataset(attrs=dict(ds_reg.attrs))
        out["regime_labels"] = labels_da.astype(np.int16)
        for name in ("frequency", "persistence_mean", "transition_counts", "transition_prob"):
            if name in ds_reg:
                out[name] = ds_reg[name].load()

    regime_coord = np.arange(1, final_k + 1, dtype=int)
    for field_key, anomaly_path in anomaly_paths_by_field.items():
        with xr.open_dataset(anomaly_path, chunks=None) as ds_field:
            if "field_anom" in ds_field:
                anom = ds_field["field_anom"].load()
            else:
                candidates = [name for name in ds_field.data_vars if "anom" in name.lower()]
                if not candidates:
                    raise KeyError(f"No anomaly-like field found in {anomaly_path}")
                anom = ds_field[candidates[0]].load()
        field_time_dim = anom.dims[0]
        anom = anom.sel({field_time_dim: time_coord.values})
        comps: list[xr.DataArray] = []
        for regime in regime_coord:
            mask = xr.DataArray(
                regime_labels == int(regime),
                dims=[time_dim],
                coords={time_dim: time_coord},
            )
            comp = anom.where(mask, drop=True).mean(dim=field_time_dim, skipna=True)
            comps.append(comp.expand_dims(regime=[int(regime)]))
        out[f"regime_composite_{field_key}_anom"] = xr.concat(comps, dim="regime").astype("float32")

    out.attrs["workflow_family"] = "multifield"
    out.attrs["step"] = "10_11_multifield_regime_characterisation"
    out.attrs["source_regimes"] = str(regimes_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return out_path
