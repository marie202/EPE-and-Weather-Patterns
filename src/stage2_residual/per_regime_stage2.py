from __future__ import annotations

"""Per-Z500-regime Stage 2 residual sub-clustering.

The global Stage 2 (see auxiliary.py) builds ONE joint-feature space over ALL days
with ONE field set and ONE k. This module lets Stage 2 run independently PER Z500
circulation regime instead — its own residual field subset (as selected by rr_01's
regime-mode field importance) and its own subregime count k (explicit, or auto via
a silhouette sweep over that regime's own day-subset).

It wraps the existing, unmodified building blocks (run_per_field_pca is called by the
caller beforehand; build_joint_feature_dataset, run_ward_hierarchical_clustering,
run_kmeans_regimes are all subset-agnostic — they operate on whatever rows/fields are
in the files handed to them) rather than modifying them, then flattens the per-regime
results back into one contiguous 1..K_total ``regime_labels`` so downstream consumers
(write_regime_characterisation, remap_rare_regimes_per_z500, notebooks, plotting) keep
working unmodified.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from stage2_residual.fields.cluster import (
    _frequency,
    _persistence,
    _transition_counts,
    run_kmeans_regimes,
    run_ward_hierarchical_clustering,
)
from stage2_residual.fields.joint_features import build_joint_feature_dataset

from .residuals import demean_joint_features

__all__ = [
    "PerRegimeStage2Result",
    "silhouette_sweep_k",
    "resolve_z500_day_subset",
    "prepare_stage2_inputs_for_z500_regime",
    "run_stage2_for_z500_regime",
    "merge_per_regime_outputs",
]


@dataclass(frozen=True)
class PerRegimeStage2Result:
    z_regime_id: int
    fields: tuple[str, ...]
    k_used: int
    k_selection_method: str  # "explicit" | "auto"
    silhouette_scores: dict[int, float]
    time_dim: str
    day_index: pd.DatetimeIndex
    labels_local: np.ndarray
    cluster_centers_local: np.ndarray


def silhouette_sweep_k(
    joint_features_path: Path,
    ward_hc_path: Path,
    k_range: Sequence[int],
    *,
    sample_size: int = 10000,
    random_state: int = 42,
) -> tuple[int, dict[int, float]]:
    """Pick k by max silhouette over Ward-linkage cuts of a joint-features file.

    Python extraction of the FINAL_K=auto bash+heredoc sweep also present in
    the removed rr_05 clustering-test job (around line 250):
    fcluster the Ward linkage at each k, score with sampled silhouette, keep the best.
    """
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import silhouette_score

    with xr.open_dataset(joint_features_path, chunks=None) as ds:
        X = np.asarray(ds["joint_features"].values, dtype=np.float32)
    with xr.open_dataset(ward_hc_path, chunks=None) as ds:
        Z = np.asarray(ds["ward_linkage"].values, dtype=np.float64)

    n_samples = X.shape[0]
    scores: dict[int, float] = {}
    best_k: int | None = None
    best_sil = -1.0

    for k in k_range:
        k = int(k)
        if k < 2 or k >= n_samples:
            continue
        labels = fcluster(Z, k, criterion="maxclust") - 1
        if len(np.unique(labels)) < 2:
            continue
        n_samp = min(sample_size, n_samples)
        sil = float(
            silhouette_score(
                X,
                labels,
                sample_size=n_samp if n_samp < n_samples else None,
                random_state=random_state,
            )
        )
        scores[k] = sil
        if sil > best_sil:
            best_sil, best_k = sil, k

    if best_k is None:
        # Degenerate day-subset (too few days, or every candidate k collapses) —
        # fall back to the smallest requested k that still fits the sample count.
        fallback = min((int(k) for k in k_range if int(k) < n_samples), default=2)
        return int(fallback), scores
    return int(best_k), scores


def resolve_z500_day_subset(z500_regimes_path: Path, z_regime_id: int) -> pd.DatetimeIndex:
    """Normalized calendar dates where the Z500 regime label equals ``z_regime_id``."""
    with xr.open_dataset(z500_regimes_path, chunks=None) as ds:
        if "regime_labels" in ds:
            var_name = "regime_labels"
        else:
            var_name = next(
                v for v in ds.data_vars if any(d in ds[v].dims for d in ("time", "valid_time"))
            )
        da = ds[var_name].load()

    time_dim = da.dims[0]
    times = pd.DatetimeIndex(da[time_dim].values).normalize()
    labels = np.asarray(da.values, dtype=int)
    mask = labels == int(z_regime_id)
    return times[mask]


def prepare_stage2_inputs_for_z500_regime(
    *,
    z_regime_id: int,
    z500_regimes_path: Path,
    pca_paths_by_field: dict[str, Path],
    joint_features_alldays_path: Path,
    joint_features_subset_path: Path,
    ward_hc_path: Path,
    overwrite: bool = False,
    residualize: bool = False,
) -> None:
    """Build (or reuse cached) joint features + Ward linkage for one Z500 regime's
    own field subset and own day subset — the part of Stage 2 that does NOT depend
    on how many clusters we later cut it into.

    Shared by `run_stage2_for_z500_regime` (production, single explicit-or-auto k)
    and the rr_021 subregime-k sweep (many candidate k's over the same inputs) —
    both need identical joint-features/Ward-HC files, built once. See
    `paths.per_regime_fields_signature` for the matching path-naming side of this.

    ``pca_paths_by_field`` must already be the subset of fields selected for THIS
    Z500 regime (excluding "z500" — z500 is the Stage-1 conditioning variable, not
    a Stage-2 clustering feature, matching the existing global-mode convention in
    auxiliary.py).

    ``residualize=True`` subtracts this regime's own day-subset column mean from
    ``joint_features`` before it's written/Ward-clustered — i.e.
    ``residual[t] = anomaly[t] - mean(anomaly | regime[t])`` (Option 2 of the
    three Stage-2 conditioning modes in README_ResidualRegimes.md; see
    `residuals.demean_joint_features`). Since ``joint_features_subset_path``
    already contains only this one Z500 regime's days, its own column mean IS
    the regime-conditional mean — no explicit day-to-regime labels are needed
    here. Default False preserves the pre-Jul-2026 (non-residualized) behavior.
    """
    build_joint_feature_dataset(
        pca_paths_by_field=pca_paths_by_field,
        out_path=joint_features_alldays_path,
        overwrite=overwrite,
        regime_set_id=f"rr_perregime_Z{z_regime_id}",
        regime_set_label=f"Residual Stage 2 per-Z500 (Z500-{z_regime_id})",
    )

    if not joint_features_subset_path.is_file() or overwrite:
        day_index = resolve_z500_day_subset(z500_regimes_path, z_regime_id)
        with xr.open_dataset(joint_features_alldays_path, chunks=None) as ds:
            time_name = ds["joint_features"].dims[0]
            file_times = pd.DatetimeIndex(ds[time_name].values).normalize()
            common = file_times.intersection(day_index)
            if len(common) < len(day_index):
                print(
                    f"[per_regime_stage2] Z500-{z_regime_id}: {len(day_index)} regime days "
                    f"∩ {len(file_times)} joint-feature days = {len(common)} common days"
                )
            idx = pd.Index(file_times).get_indexer(common)
            ds_subset = ds.isel({time_name: idx}).load()
        if residualize:
            ds_subset = demean_joint_features(ds_subset)
        joint_features_subset_path.parent.mkdir(parents=True, exist_ok=True)
        ds_subset.to_netcdf(joint_features_subset_path)

    run_ward_hierarchical_clustering(
        joint_features_path=joint_features_subset_path,
        out_hc_path=ward_hc_path,
        overwrite=overwrite,
        regime_set_id=f"rr_perregime_Z{z_regime_id}",
        regime_set_label=f"Residual Stage 2 per-Z500 (Z500-{z_regime_id}) Ward HC",
    )


def run_stage2_for_z500_regime(
    *,
    z_regime_id: int,
    z500_regimes_path: Path,
    pca_paths_by_field: dict[str, Path],
    k_spec: int | str,
    final_k_range: Sequence[int],
    joint_features_alldays_path: Path,
    joint_features_subset_path: Path,
    ward_hc_path: Path,
    local_regimes_path: Path,
    random_state: int = 42,
    overwrite: bool = False,
    residualize: bool = False,
) -> PerRegimeStage2Result:
    """Run Stage 2 (joint features -> Ward HC -> k-means) for one Z500 regime's
    own field subset and own day subset.

    ``pca_paths_by_field`` must already be the subset of fields selected for THIS
    Z500 regime (excluding "z500" — z500 is the Stage-1 conditioning variable, not
    a Stage-2 clustering feature, matching the existing global-mode convention in
    auxiliary.py).

    ``residualize`` — see `prepare_stage2_inputs_for_z500_regime`.
    """
    prepare_stage2_inputs_for_z500_regime(
        z_regime_id=z_regime_id,
        z500_regimes_path=z500_regimes_path,
        pca_paths_by_field=pca_paths_by_field,
        joint_features_alldays_path=joint_features_alldays_path,
        joint_features_subset_path=joint_features_subset_path,
        ward_hc_path=ward_hc_path,
        overwrite=overwrite,
        residualize=residualize,
    )

    if isinstance(k_spec, str) and k_spec == "auto":
        k_used, sil_scores = silhouette_sweep_k(
            joint_features_subset_path, ward_hc_path, final_k_range, random_state=random_state,
        )
        k_method = "auto"
    else:
        k_used, sil_scores, k_method = int(k_spec), {}, "explicit"

    run_kmeans_regimes(
        joint_features_path=joint_features_subset_path,
        out_regimes_path=local_regimes_path,
        final_k=k_used,
        random_state=random_state,
        overwrite=overwrite,
        regime_set_id=f"rr_perregime_Z{z_regime_id}_k{k_used}",
        regime_set_label=f"Residual Stage 2 per-Z500 (Z500-{z_regime_id}, K={k_used})",
        ward_hc_path=ward_hc_path,
    )

    with xr.open_dataset(local_regimes_path, chunks=None) as ds:
        labels_da = ds["regime_labels"].load()
        centers = np.asarray(ds["cluster_centers"].values, dtype=np.float64)
        time_dim = labels_da.dims[0]
        day_index_out = pd.DatetimeIndex(labels_da[time_dim].values)
        labels_local = np.asarray(labels_da.values, dtype=int)

    return PerRegimeStage2Result(
        z_regime_id=int(z_regime_id),
        # Preserve `pca_paths_by_field`'s own key order rather than re-sorting alphabetically --
        # callers (e.g. auxiliary.py, stage2_logistic.py) build it from config.per_regime_field_map,
        # which is itself rr_01's greedy-AUC field order (see write_field_selection.select_fields).
        # Downstream, plot_composites.py relies on this order to pick each Z500 regime's 3 impact
        # fields and assign them R/G/B by importance rank (most important = red).
        fields=tuple(pca_paths_by_field.keys()),
        k_used=int(k_used),
        k_selection_method=k_method,
        silhouette_scores=sil_scores,
        time_dim=time_dim,
        day_index=day_index_out,
        labels_local=labels_local,
        cluster_centers_local=centers,
    )


def merge_per_regime_outputs(
    results: dict[int, PerRegimeStage2Result],
    *,
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    """Flatten independent per-Z500-regime local clusterings into one contiguous
    1..K_total ``regime_labels`` file, drop-in compatible with the global-mode
    Stage-2 output (same variable names/shapes) so remap_rare_regimes_per_z500,
    write_regime_characterisation, and downstream plotting/notebooks keep working
    unmodified. Per-regime provenance (fields/k/silhouette used per Z500 regime) is
    stored as new JSON attrs rather than replacing the flat label scheme.
    """
    if out_path.is_file() and not overwrite:
        return out_path

    z_ids = sorted(results.keys())
    offsets: dict[int, int] = {}
    running = 0
    for z in z_ids:
        offsets[z] = running
        running += results[z].k_used
    k_total = running

    max_feat = max(results[z].cluster_centers_local.shape[1] for z in z_ids)
    time_dim = results[z_ids[0]].time_dim

    all_times = [np.asarray(results[z].day_index.values) for z in z_ids]
    all_labels = [
        results[z].labels_local.astype(int) + offsets[z] for z in z_ids
    ]
    times_concat = np.concatenate(all_times)
    labels_concat = np.concatenate(all_labels)
    order = np.argsort(times_concat)
    times_sorted = times_concat[order]
    labels_sorted = labels_concat[order].astype(np.int16)

    centers_blocks = []
    for z in z_ids:
        c = results[z].cluster_centers_local.astype(np.float32)
        if c.shape[1] < max_feat:
            pad = np.zeros((c.shape[0], max_feat - c.shape[1]), dtype=np.float32)
            c = np.concatenate([c, pad], axis=1)
        centers_blocks.append(c)
    centers = np.concatenate(centers_blocks, axis=0)

    freq = _frequency(labels_sorted, k_total)
    pers = _persistence(labels_sorted, k_total)
    T = _transition_counts(labels_sorted, k_total)
    row_sums = T.sum(axis=1, keepdims=True)
    T_prob = np.zeros_like(T, dtype=np.float32)
    np.divide(T, row_sums, out=T_prob, where=row_sums != 0)

    regime_coord = np.arange(1, k_total + 1, dtype=int)
    feature_coord = np.arange(max_feat, dtype=int)

    ds_out = xr.Dataset(
        data_vars={
            "regime_labels": xr.DataArray(
                labels_sorted, dims=[time_dim], coords={time_dim: times_sorted}, name="regime_labels",
            ),
            "cluster_centers": xr.DataArray(
                centers, dims=["regime", "feature"],
                coords={"regime": regime_coord, "feature": feature_coord}, name="cluster_centers",
            ),
            "frequency": xr.DataArray(freq, dims=["regime"], coords={"regime": regime_coord}, name="frequency"),
            "persistence_mean": xr.DataArray(
                pers, dims=["regime"], coords={"regime": regime_coord}, name="persistence_mean",
            ),
            "transition_counts": xr.DataArray(
                T, dims=["regime_from", "regime_to"],
                coords={"regime_from": regime_coord, "regime_to": regime_coord}, name="transition_counts",
            ),
            "transition_prob": xr.DataArray(
                T_prob, dims=["regime_from", "regime_to"],
                coords={"regime_from": regime_coord, "regime_to": regime_coord}, name="transition_prob",
            ),
        },
        attrs={
            "workflow_family": "stage2_residual",
            "step": "09_per_regime_merge",
            "final_k": int(k_total),
            "per_regime_mode": "true",
            "per_regime_fields": json.dumps({str(z): list(results[z].fields) for z in z_ids}),
            "per_regime_k": json.dumps({str(z): results[z].k_used for z in z_ids}),
            "per_regime_k_method": json.dumps({str(z): results[z].k_selection_method for z in z_ids}),
            "per_regime_silhouette": json.dumps({str(z): results[z].silhouette_scores for z in z_ids}),
            "per_regime_offsets": json.dumps({str(z): offsets[z] for z in z_ids}),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    return out_path
