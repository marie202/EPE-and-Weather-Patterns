from __future__ import annotations

"""Post-processing for residual regime labels.

Two filtering functions:

remap_rare_regimes (legacy)
    Global filter: if a RR has fewer than min_days total it is merged
    everywhere. Treats RR subregimes as overarching labels.

remap_rare_regimes_per_z500 (preferred)
    Per-Z500-regime filter: for each Z500 circulation regime j, any RR
    subregime k with fewer than min_days occurrences *within j* is merged
    into the nearest valid RR within j.  Different Z500 regimes can therefore
    end up with different active subsets of RR labels — there is no requirement
    that a subregime must exist in every Z500 regime.

Both produce a 09b_RR_regimes_filtered_*.nc file that preserves the raw
k-means output (09_RR_regimes_kmeans_*.nc).
"""

from pathlib import Path

import numpy as np
import xarray as xr


def remap_rare_regimes(
    regimes_path: Path,
    out_path: Path,
    min_days: int = 10,
    min_frac: float = 0.005,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Reassign days from rare regimes to the nearest valid regime centroid.

    A regime is considered rare if its day count is below:
        max(min_days, total_days * min_frac)

    The fraction-based floor (default 0.5 % of all days) prevents degenerate
    regimes that technically exceed the absolute min_days but represent a
    negligible fraction of the dataset.

    Returns (out_path, effective_k).  If out_path exists and not overwrite,
    reads effective_k from attrs and returns without recomputing.
    """
    if out_path.is_file() and not overwrite:
        with xr.open_dataset(out_path) as ds:
            effective_k = int(ds.attrs.get("effective_k", int(ds["regime_labels"].max())))
        return out_path, effective_k

    with xr.open_dataset(regimes_path, chunks=None) as ds:
        labels_da = ds["regime_labels"].load()
        centers = np.asarray(ds["cluster_centers"].values, dtype=np.float64)  # (k, n_feat)
        time_dim = labels_da.dims[0]
        time_coord = labels_da[time_dim]
        src_attrs = dict(ds.attrs)
        original_k = int(src_attrs.get("final_k", int(labels_da.max())))

    labels = np.asarray(labels_da.values, dtype=int)
    total_days = len(labels)
    threshold = max(min_days, int(np.ceil(total_days * min_frac)))
    print(f"[rr_postprocess] Rare-regime threshold: {threshold} days "
          f"(max({min_days}, {total_days}×{min_frac:.1%} = {int(np.ceil(total_days * min_frac))}))")

    unique, counts = np.unique(labels, return_counts=True)
    rare = sorted(int(r) for r, c in zip(unique, counts) if c < threshold)
    valid = sorted(int(r) for r, c in zip(unique, counts) if c >= threshold)

    if not rare:
        import shutil
        print(f"[rr_postprocess] No rare regimes (all >= {min_days} days) — copying as-is.")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(regimes_path, out_path)
        effective_k = len(valid)
        # Patch effective_k into attrs
        with xr.open_dataset(out_path) as ds_tmp:
            ds_patch = ds_tmp.load()
        ds_patch.attrs["effective_k"] = effective_k
        ds_patch.attrs["min_days_threshold"] = min_days
        ds_patch.to_netcdf(out_path)
        return out_path, effective_k

    print(f"[rr_postprocess] Rare regimes (< {min_days} days): {rare}")
    print(f"[rr_postprocess] Valid regimes retained: {valid}")

    valid_arr = np.array(valid, dtype=int)
    valid_centers = centers[valid_arr - 1]  # (n_valid, n_feat)

    # Reassign each rare regime to its nearest valid centroid (centroid→centroid).
    # With < min_days days this approximation is good enough and avoids loading
    # the large joint-features file.
    new_labels = labels.copy()
    remap_log: dict[int, int] = {}
    for r in rare:
        dists = np.linalg.norm(valid_centers - centers[r - 1], axis=1)
        nearest = int(valid_arr[np.argmin(dists)])
        new_labels[labels == r] = nearest
        remap_log[r] = nearest
        n_moved = int((labels == r).sum())
        print(f"[rr_postprocess]   SR{r} ({n_moved} days) → SR{nearest}")

    # Renumber valid regimes 1, 2, 3, ...
    old_to_new = {old: new for new, old in enumerate(valid, start=1)}
    final_labels = np.vectorize(old_to_new.__getitem__)(new_labels).astype(np.int16)
    effective_k = len(valid)

    # Recompute per-regime statistics
    from stage2_residual.fields.cluster import (
        _frequency,
        _persistence,
        _transition_counts,
    )
    freq = _frequency(final_labels, effective_k)
    pers = _persistence(final_labels, effective_k)
    T = _transition_counts(final_labels, effective_k)
    row_sums = T.sum(axis=1, keepdims=True)
    T_prob = np.zeros_like(T, dtype=np.float32)
    np.divide(T, row_sums, out=T_prob, where=row_sums != 0)

    new_centers = centers[valid_arr - 1].astype(np.float32)
    regime_coord = np.arange(1, effective_k + 1, dtype=int)
    feature_coord = np.arange(centers.shape[1], dtype=int)

    new_attrs = dict(src_attrs)
    new_attrs["final_k"] = effective_k
    new_attrs["effective_k"] = effective_k
    new_attrs["original_k"] = original_k
    new_attrs["min_days_threshold"] = min_days
    new_attrs["rare_regimes_remapped"] = str(remap_log)

    ds_out = xr.Dataset(
        data_vars={
            "regime_labels": xr.DataArray(
                final_labels, dims=[time_dim], coords={time_dim: time_coord},
            ),
            "cluster_centers": xr.DataArray(
                new_centers, dims=["regime", "feature"],
                coords={"regime": regime_coord, "feature": feature_coord},
            ),
            "frequency": xr.DataArray(freq, dims=["regime"], coords={"regime": regime_coord}),
            "persistence_mean": xr.DataArray(pers, dims=["regime"], coords={"regime": regime_coord}),
            "transition_counts": xr.DataArray(
                T, dims=["regime_from", "regime_to"],
                coords={"regime_from": regime_coord, "regime_to": regime_coord},
            ),
            "transition_prob": xr.DataArray(
                T_prob, dims=["regime_from", "regime_to"],
                coords={"regime_from": regime_coord, "regime_to": regime_coord},
            ),
        },
        attrs=new_attrs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"[rr_postprocess] Filtered regimes saved: {out_path.name}  (K={original_k} → K={effective_k})")
    return out_path, effective_k


def remap_rare_regimes_per_z500(
    regimes_path: Path,
    z500_regimes_path: Path,
    out_path: Path,
    min_days: int = 10,
    overwrite: bool = False,
    restrict_fallback_to_same_regime: bool = False,
) -> tuple[Path, dict[int, list[int]]]:
    """Filter rare RR subregimes independently within each Z500 circulation regime.

    For each Z500 regime j:
      - Count how many days fall in each RR subregime k.
      - If count(RR_k | Z500_j) < min_days:
            reassign those days to the nearest valid RR centroid within Z500_j.
      - "Valid" within Z500_j = RRs that have >= min_days in that Z500 regime.
      - If no RR is valid within Z500_j (very small Z500 regime), fall back to
        the globally most frequent RR — unless ``restrict_fallback_to_same_regime``
        is set, in which case fall back to the most frequent RR *within Z500_j
        only*. The global fallback assumes every RR label lives in one shared
        feature space (true for the single joint-clustering global mode); when
        RR labels come from independent per-Z500-regime clusterings with
        different fields/dimensionality (per-regime Stage 2 mode), a centroid
        from another Z500 regime is not comparable, so that mode must pass
        ``restrict_fallback_to_same_regime=True``.

    Result: the same RR label can be active in some Z500 regimes but absent in
    others.  The output stores which RRs are active per Z500 regime in the
    dataset attributes so that plotting code can respect regime-specific subsets.

    Returns
    -------
    (out_path, active_rrs_per_z500)
        active_rrs_per_z500 : dict mapping z500_label (int) → sorted list of
                              active RR labels (int) for that Z500 regime.
    """
    import pandas as pd

    if out_path.is_file() and not overwrite:
        with xr.open_dataset(out_path) as ds:
            import json
            active = json.loads(ds.attrs.get("active_rrs_per_z500", "{}"))
            active_rrs = {int(k): [int(v) for v in vs] for k, vs in active.items()}
        return out_path, active_rrs

    # ------------------------------------------------------------------
    # Load RR labels
    # ------------------------------------------------------------------
    with xr.open_dataset(regimes_path, chunks=None) as ds:
        rr_da = ds["regime_labels"].load()
        centers = np.asarray(ds["cluster_centers"].values, dtype=np.float64)
        time_dim = rr_da.dims[0]
        src_attrs = dict(ds.attrs)
        original_k = int(src_attrs.get("final_k", int(rr_da.max())))

    rr_times = pd.DatetimeIndex(rr_da[time_dim].values).normalize()
    rr_labels = np.asarray(rr_da.values, dtype=int)

    # ------------------------------------------------------------------
    # Load Z500 labels and align by date
    # ------------------------------------------------------------------
    with xr.open_dataset(z500_regimes_path, chunks=None) as ds_z:
        z500_var = "regime_labels" if "regime_labels" in ds_z else list(ds_z.data_vars)[0]
        z500_da = ds_z[z500_var].load()
        z500_time_dim = z500_da.dims[0]

    z500_times = pd.DatetimeIndex(z500_da[z500_time_dim].values).normalize()
    z500_labels_full = np.asarray(z500_da.values, dtype=int)

    # Align: only keep days present in both
    common_dates = rr_times.intersection(z500_times)
    if len(common_dates) < len(rr_times):
        print(f"[rr_postprocess] Alignment: {len(rr_times)} RR days ∩ {len(z500_times)} Z500 days "
              f"= {len(common_dates)} common days")

    rr_idx   = rr_times.get_indexer(common_dates)
    z500_idx = z500_times.get_indexer(common_dates)

    rr_aligned   = rr_labels[rr_idx]
    z500_aligned = z500_labels_full[z500_idx]

    # Work on a copy
    new_rr = rr_aligned.copy()
    all_rr_labels = np.unique(rr_aligned)
    active_rrs_per_z500: dict[int, list[int]] = {}
    remap_summary: list[str] = []

    for z_regime in np.unique(z500_aligned):
        z_mask = z500_aligned == z_regime
        z_days = z_mask.sum()

        # Count RR occurrences within this Z500 regime
        rr_in_z, counts_in_z = np.unique(rr_aligned[z_mask], return_counts=True)
        valid_in_z = [int(r) for r, c in zip(rr_in_z, counts_in_z) if c >= min_days]
        rare_in_z  = [int(r) for r, c in zip(rr_in_z, counts_in_z) if c < min_days]

        if not rare_in_z:
            active_rrs_per_z500[int(z_regime)] = sorted(valid_in_z)
            continue

        # Fallback: if nothing is valid within this Z500 regime, use all global RRs
        # (or, when the RR feature space is per-Z500-regime-specific, the most
        # frequent RR within this same Z500 regime only).
        if not valid_in_z:
            if restrict_fallback_to_same_regime:
                valid_in_z = [int(rr_in_z[np.argmax(counts_in_z)])]
                print(f"[rr_postprocess]   Z500-{z_regime}: no RR has >= {min_days} days — "
                      f"falling back to most frequent RR within Z500-{z_regime} only "
                      f"(restrict_fallback_to_same_regime=True)")
            else:
                valid_in_z = sorted(all_rr_labels.tolist())
                print(f"[rr_postprocess]   Z500-{z_regime}: no RR has >= {min_days} days — "
                      f"falling back to all global RRs for reassignment")

        valid_arr    = np.array(valid_in_z, dtype=int)
        valid_centers = centers[valid_arr - 1]  # (n_valid, n_feat)

        for r in rare_in_z:
            r_count = int((rr_aligned[z_mask] == r).sum())
            dists   = np.linalg.norm(valid_centers - centers[r - 1], axis=1)
            nearest = int(valid_arr[np.argmin(dists)])
            # Only reassign days that belong to this Z500 regime
            target = z_mask & (rr_aligned == r)
            new_rr[target] = nearest
            msg = (f"Z500-{z_regime}: SR{r} ({r_count} days) → SR{nearest} "
                   f"(centroid distance {dists.min():.3f})")
            remap_summary.append(msg)
            print(f"[rr_postprocess]   {msg}")

        active_rrs_per_z500[int(z_regime)] = sorted(valid_in_z)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"[rr_postprocess] Active RRs per Z500 regime:")
    for z_regime, active in sorted(active_rrs_per_z500.items()):
        print(f"  Z500-{z_regime}: SR{{{', '.join(str(r) for r in active)}}}")

    # ------------------------------------------------------------------
    # Renumber surviving labels consecutively: e.g. {1,2,4} → {1,2,3}
    # ------------------------------------------------------------------
    surviving = sorted(int(r) for r in np.unique(new_rr))
    old_to_new = {old: new for new, old in enumerate(surviving, start=1)}

    if surviving != list(range(1, len(surviving) + 1)):
        print(f"[rr_postprocess] Renumbering: {surviving} → {list(range(1, len(surviving)+1))}")
        new_rr = np.vectorize(old_to_new.__getitem__)(new_rr).astype(np.int16)
        active_rrs_per_z500 = {
            z: sorted(old_to_new[r] for r in rrs if r in old_to_new)
            for z, rrs in active_rrs_per_z500.items()
        }

    # Always slice centers to surviving regimes (regardless of renumbering)
    centers = centers[[r - 1 for r in surviving]]

    if len(surviving) == 1:
        print("[rr_postprocess] WARNING: all days collapsed to a single regime after per-Z500 filtering.")
        print("[rr_postprocess]   → Consider re-running rr_02 with a smaller FINAL_K or lower min_days.")

    # ------------------------------------------------------------------
    # Write output — same structure as remap_rare_regimes
    # ------------------------------------------------------------------
    import json
    from stage2_residual.fields.cluster import _frequency, _persistence, _transition_counts

    effective_k = len(surviving)
    freq = _frequency(new_rr, effective_k)
    pers = _persistence(new_rr, effective_k)
    T    = _transition_counts(new_rr, effective_k)
    row_sums = T.sum(axis=1, keepdims=True)
    T_prob   = np.zeros_like(T, dtype=np.float32)
    np.divide(T, row_sums, out=T_prob, where=row_sums != 0)

    regime_coord = np.arange(1, effective_k + 1, dtype=int)
    feature_coord = np.arange(centers.shape[1], dtype=int)

    # Reconstruct output time coordinate from the aligned subset
    time_coord_out = rr_da[time_dim].values[rr_idx]

    new_attrs = dict(src_attrs)
    new_attrs["final_k"]              = original_k
    new_attrs["effective_k"]          = effective_k
    new_attrs["original_k"]           = original_k
    new_attrs["min_days_threshold"]   = min_days
    new_attrs["filter_mode"]          = "per_z500"
    new_attrs["effective_k"]          = effective_k
    new_attrs["active_rrs_per_z500"]  = json.dumps(
        {str(k): v for k, v in active_rrs_per_z500.items()}
    )
    new_attrs["label_renumber_map"]   = json.dumps(
        {str(k): v for k, v in old_to_new.items()}
    )
    new_attrs["remap_log"]            = " | ".join(remap_summary)

    ds_out = xr.Dataset(
        data_vars={
            "regime_labels": xr.DataArray(
                new_rr.astype(np.int16), dims=[time_dim],
                coords={time_dim: time_coord_out},
            ),
            "cluster_centers": xr.DataArray(
                centers.astype(np.float32), dims=["regime", "feature"],
                coords={"regime": regime_coord, "feature": feature_coord},
            ),
            "frequency": xr.DataArray(freq, dims=["regime"], coords={"regime": regime_coord}),
            "persistence_mean": xr.DataArray(pers, dims=["regime"], coords={"regime": regime_coord}),
            "transition_counts": xr.DataArray(
                T, dims=["regime_from", "regime_to"],
                coords={"regime_from": regime_coord, "regime_to": regime_coord},
            ),
            "transition_prob": xr.DataArray(
                T_prob, dims=["regime_from", "regime_to"],
                coords={"regime_from": regime_coord, "regime_to": regime_coord},
            ),
        },
        attrs=new_attrs,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print(f"[rr_postprocess] Saved per-Z500 filtered regimes: {out_path.name}")
    return out_path, active_rrs_per_z500


def main(argv=None) -> int:
    """Standalone CLI: apply rare-regime filter to an existing k-means regimes file."""
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Filter rare regimes from a k-means output file.")
    p.add_argument("--regimes-path", type=Path, required=True,
                   help="Path to 09_RR_regimes_kmeans_*.nc")
    p.add_argument("--out-path", type=Path, default=None,
                   help="Output path (default: 09b_RR_regimes_filtered_* next to input)")
    p.add_argument("--min-days", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    regimes_path = args.regimes_path.expanduser().resolve()
    if args.out_path is None:
        stem = regimes_path.stem.replace("09_RR_regimes_kmeans_", "09b_RR_regimes_filtered_")
        out_path = regimes_path.parent / (stem + ".nc")
    else:
        out_path = args.out_path.expanduser().resolve()

    filtered_path, effective_k = remap_rare_regimes(
        regimes_path=regimes_path,
        out_path=out_path,
        min_days=args.min_days,
        overwrite=args.overwrite,
    )
    print(f"Done. Effective K={effective_k}  →  {filtered_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
