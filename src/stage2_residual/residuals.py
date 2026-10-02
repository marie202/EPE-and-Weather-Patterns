from __future__ import annotations

"""Regime-conditioned mean subtraction for Stage 2 residual clustering.

    residual[t] = joint_features[t] - mean(joint_features | regime[t])

This is Option 2 of the three Stage-2 conditioning modes documented in
README_ResidualRegimes.md ("Bridge — Residual computation" / "Three Stage-2
conditioning modes"): discrete, per-Z500-regime mean subtraction, as opposed
to no conditioning at all (Option 1, global mode) or continuous regression-
based orthogonalization against Z500's full PC state (Option 3, not
implemented).

`build_joint_feature_dataset` (multivariate_regimes/joint_features.py) already
projects each field's anomaly onto a fixed, pooled PCA basis and z-scores it
with the GLOBAL (all-days) mean/std. Because that projection is linear and the
same basis is shared across every Z500 regime, subtracting a regime-conditional
mean from these joint-feature PCs is mathematically equivalent to residualizing
the raw anomaly field itself:

    PC[t] = W @ anomaly[t]
    mean(PC | regime) = W @ mean(anomaly | regime)
    PC[t] - mean(PC | regime) = W @ (anomaly[t] - mean(anomaly | regime))

So this module only needs to operate on the already-PCA-projected joint
features — no upstream PCA/anomaly recomputation is needed. It intentionally
does NOT also rescale by a per-regime std after demeaning: the existing global
std (from build_joint_feature_dataset's z-scoring) is left untouched, so a
regime that is genuinely less variable in moisture/instability stays that way
rather than being artificially equalized with the others.
"""

import numpy as np
import xarray as xr

__all__ = ["demean_joint_features"]


def demean_joint_features(
    ds: xr.Dataset,
    *,
    group_labels: np.ndarray | None = None,
) -> xr.Dataset:
    """Subtract the per-feature mean from ``ds["joint_features"]``.

    ``group_labels=None`` (the per-regime-mode case): demean over ALL rows in
    ``ds`` as one group. Use this when ``ds`` already contains only one Z500
    regime's days (e.g. the day-subset built by
    ``per_regime_stage2.prepare_stage2_inputs_for_z500_regime``) — the
    subset's own column mean over time already equals ``mean(anomaly | regime)``
    for that regime, so no explicit regime labels are needed.

    ``group_labels`` (one Z500 regime id per row): demean per group instead —
    for residualizing a SINGLE shared joint-feature space spanning multiple
    regimes (e.g. a future global-mode residualized variant), where each row
    must be demeaned by its OWN regime's mean, not the mean of the whole array.

    Returns a copy of ``ds`` with ``joint_features`` replaced by the residual
    and a new ``regime_mean`` data var recording the mean(s) removed, so a
    residualized cluster center can be mapped back to physical PC units later
    (e.g. by regime characterisation/plotting code) by adding it back.
    """
    X = np.asarray(ds["joint_features"].values, dtype=np.float64)

    if group_labels is None:
        mean_vec = X.mean(axis=0)
        X_resid = X - mean_vec
        mean_da = xr.DataArray(
            mean_vec.astype(np.float32),
            dims=["feature"],
            coords={"feature": ds["feature"]},
            name="regime_mean",
        )
    else:
        group_labels = np.asarray(group_labels)
        if group_labels.shape[0] != X.shape[0]:
            raise ValueError(
                f"group_labels length {group_labels.shape[0]} != "
                f"joint_features rows {X.shape[0]}"
            )
        X_resid = np.empty_like(X)
        means_by_group: dict[int, np.ndarray] = {}
        for g in np.unique(group_labels):
            mask = group_labels == g
            group_mean = X[mask].mean(axis=0)
            means_by_group[int(g)] = group_mean
            X_resid[mask] = X[mask] - group_mean
        groups_sorted = sorted(means_by_group)
        mean_da = xr.DataArray(
            np.stack([means_by_group[g] for g in groups_sorted]).astype(np.float32),
            dims=["z500_regime", "feature"],
            coords={"z500_regime": groups_sorted, "feature": ds["feature"]},
            name="regime_mean",
        )

    ds_out = ds.copy()
    ds_out["joint_features"] = xr.DataArray(
        X_resid.astype(np.float32),
        dims=ds["joint_features"].dims,
        coords=ds["joint_features"].coords,
    )
    ds_out["regime_mean"] = mean_da
    ds_out.attrs["residualized"] = "true"
    return ds_out
