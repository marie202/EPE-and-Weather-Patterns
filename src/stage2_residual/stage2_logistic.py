from __future__ import annotations

"""Production Stage 2 [LOGISTIC]: continuous per-day, per-Z500-regime
predicted-extreme-probability index, replacing discrete subregime cluster
labels.

See README_ResidualRegimes.md "Main Finding (2026-07-29): Stage 2 is a
gradient, not groups" — k-means/Ward/HDBSCAN all agree the residual
fields do not organize into discrete subregimes, while a per-regime logistic
regression against the extreme-day label shows strong, permutation-significant
signal (see residual_extreme_signal_diag_LOGISTIC.py, the diagnostic this
module promotes into a production artifact). Decision: Stage 2 moves from
cluster labels to this continuous index; composites are built from probability
quantile bins instead of cluster ids (see the README TODO list for the
remaining follow-up items — bin-count selection, the composite plotting path,
grid-cell risk maps).

Pinned EPE definition for this production path (still under evaluation, may
change — see README_DOMAIN_PCTL_Q.md): domain_pctl_q=0.95, --domain-agg
lexmax, i.e. --extreme-var flagged_domain_pctl against a convective_duration
NetCDF produced by epe_01b_detect_convective_hourly_duration_test.py at that
threshold. ExtremesSpec's filename-tag resolution (summary_reports/
extremes_spec.py) does not encode domain_pctl_q/domain_agg tokens, so unlike
rr_01/the diagnostic's --extremes-q tag matching, this module requires an
explicit --extreme-labels-path (no tag-based auto-resolution) to avoid
silently picking the wrong domain-percentile TEST file among several with the
same point-wise --q tag.

FIT MODE — `--fit-mode {in_sample, out_of_fold_blocked_cv}` (default:
`in_sample`, unchanged legacy behaviour): `fit_regime_logistic` always fits
ONE logistic regression per Z500 regime on ALL of that regime's own days
(same recipe as residual_extreme_signal_diag_LOGISTIC.py's
`fit_residual_index`: StandardScaler + balanced LogisticRegression) to
produce `logistic_coef`/`logistic_intercept` — a SINGLE coefficient vector
per regime is kept for the attribution-map TODO (projecting coefficients
back through the residual-field PCA/EOF loadings to get a physical spatial
pattern) regardless of `--fit-mode`, since an out-of-fold variant would give
one coefficient vector per CV fold instead, complicating that reuse.

What DOES change with `--fit-mode` is `predicted_extreme_prob`, the array
that feeds composites/quantile bins:
- `in_sample` (default): the same in-sample fitted probabilities used for
  `logistic_coef`. Optimistic relative to a genuinely held-out estimate,
  since the same days are used to fit and to evaluate — see
  README_ResidualRegimes.md "Open Decisions (Revisit Later)".
- `out_of_fold_blocked_cv`: `fit_regime_logistic_out_of_fold` — per-day
  held-out probabilities from the SAME blocked-CV fold scheme
  rr_021_stage2_LogRegr_Robustness.py::blocked_cv_auc uses (contiguous
  `--cv-block-size`-day blocks assigned round-robin to `--cv-n-splits`
  folds, guarding against a day's autocorrelated neighbour leaking across
  train/test), refit `--cv-n-splits` times and reassembled into one per-day
  array. `logistic_coef` is still the separate in-sample fit above (tagged
  coefficients-only) — decoupled from what gets binned/plotted.

Switching `--fit-mode` does NOT change the output filename (same
`per_regime_fields_signature`/`experiment_id`) — pass a distinct
`--output-tag` (e.g. `oofcv`) when running `out_of_fold_blocked_cv` so it
doesn't collide with an existing `in_sample` cache. `run_stage2_logistic_for_z500_regime`
refuses to reuse a cached file whose `fit_mode` attr doesn't match the
requested one (rather than silently mixing the two).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .feature_relevance import _load_extreme_labels
from .per_regime_stage2 import prepare_stage2_inputs_for_z500_regime

__all__ = [
    "RegimeLogisticResult",
    "fit_regime_logistic",
    "fit_regime_logistic_out_of_fold",
    "run_stage2_logistic_for_z500_regime",
    "merge_per_regime_logistic_outputs",
    "bin_predicted_prob_per_regime",
    "select_regime_bin_edges",
    "load_bin_overrides_yaml",
    "run_logistic_stage2",
    "main",
]

DEFAULT_N_BINS = 2

FIT_MODE_IN_SAMPLE = "in_sample"
FIT_MODE_OOF_BLOCKED_CV = "out_of_fold_blocked_cv"
VALID_FIT_MODES = (FIT_MODE_IN_SAMPLE, FIT_MODE_OOF_BLOCKED_CV)

# Same blocked-CV defaults as rr_021_stage2_LogRegr_Robustness.py's
# DEFAULT_N_SPLITS_BLOCKED/DEFAULT_BLOCK_SIZE, kept as separate constants here
# (rather than imported) since this module must not depend on the throwaway
# rr_021 diagnostic -- rr_021 imports FROM this module, not the other way.
DEFAULT_CV_N_SPLITS = 5
DEFAULT_CV_BLOCK_SIZE = 10  # days

FIT_MODE_CAVEAT_IN_SAMPLE = (
    "In-sample fit (not out-of-fold/cross-validated): predicted_extreme_prob "
    "and any quantile-bin frequency table derived from it are optimistic "
    "relative to a held-out estimate, since the same days were used to fit "
    "and to evaluate. logistic_coef is always this in-sample fit, kept "
    "deliberately so a single coefficient vector per regime exists for the "
    "attribution-map projection (see README_ResidualRegimes.md TODO) -- pass "
    "--fit-mode out_of_fold_blocked_cv if the composite/quantile-bin "
    "frequencies need to be a fair held-out estimate rather than an "
    "in-sample one."
)

FIT_MODE_CAVEAT_OOF_BLOCKED_CV = (
    "predicted_extreme_prob is out-of-fold (blocked-CV) held-out probability "
    "-- see cv_n_splits/cv_block_size attrs for the fold scheme. logistic_coef "
    "is a SEPARATE in-sample fit on all days (coefficients-only; not used to "
    "produce predicted_extreme_prob), kept so a single coefficient vector per "
    "regime exists for the attribution-map projection."
)


@dataclass(frozen=True)
class RegimeLogisticResult:
    z_regime_id: int
    fields: tuple[str, ...]
    n_days: int
    n_extreme: int
    time_dim: str
    day_index: pd.DatetimeIndex
    predicted_prob: np.ndarray  # in-sample fitted probability per day, shape (n_days,)
    coef: np.ndarray  # logistic coefficients in the StandardScaler-ed feature space
    intercept: float
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    degenerate: bool


def fit_regime_logistic(
    X: np.ndarray, y: np.ndarray, *, random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit ONE logistic regression on ALL of a regime's days (in-sample — see
    module docstring) and return (predicted_prob, coef, intercept, scaler_mean,
    scaler_scale). Same StandardScaler + balanced LogisticRegression recipe as
    residual_extreme_signal_diag_LOGISTIC.py's `fit_residual_index`, but
    returns actual class-1 probabilities (`predict_proba`) and the fitted
    parameters instead of only the raw decision-function score, since
    production needs both the probability index and, later, the coefficient
    vector for the attribution-map TODO.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state, solver="lbfgs")
    clf.fit(X_sc, y)
    prob = np.asarray(clf.predict_proba(X_sc)[:, 1], dtype=np.float64)
    coef = np.asarray(clf.coef_[0], dtype=np.float64)
    intercept = float(clf.intercept_[0])
    return prob, coef, intercept, np.asarray(scaler.mean_, dtype=np.float64), np.asarray(scaler.scale_, dtype=np.float64)


def _block_fold_assignment(n: int, n_splits: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Assign each of n days to one of n_splits folds via contiguous block_size-day blocks,
    with BLOCKS (not days) shuffled round-robin across folds -- keeps class balance roughly
    even across folds while making within-block train/test leakage impossible. Canonical copy;
    rr_021_stage2_LogRegr_Robustness.py imports this rather than keeping its own."""
    block_ids = np.arange(n) // block_size
    n_blocks = int(block_ids.max()) + 1
    block_order = rng.permutation(n_blocks)
    fold_of_block = np.empty(n_blocks, dtype=int)
    fold_of_block[block_order] = np.arange(n_blocks) % n_splits
    return fold_of_block[block_ids]


def fit_regime_logistic_out_of_fold(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = DEFAULT_CV_N_SPLITS,
    block_size: int = DEFAULT_CV_BLOCK_SIZE,
    random_state: int = 42,
) -> np.ndarray:
    """Per-day held-out predicted probability for `--fit-mode
    out_of_fold_blocked_cv`: same blocked-CV fold assignment as
    rr_021_stage2_LogRegr_Robustness.py::blocked_cv_auc (contiguous
    block_size-day blocks, round-robin-shuffled at the block level across
    n_splits folds -- guards against a day's autocorrelated neighbour landing
    in the same fold's train split), but returns each held-out day's
    predicted probability instead of only a scalar AUC. Same StandardScaler +
    balanced LogisticRegression recipe as `fit_regime_logistic`, refit once
    per fold on that fold's training days only.

    Returns an array of shape (len(y),). A day's value is NaN if its fold's
    training split had too few (<2) or too many (>len(y_tr)-2) positive
    labels to fit a balanced logistic regression -- callers should fill NaN
    with the regime base rate and warn, same as the degenerate-regime guard
    in `run_stage2_logistic_for_z500_regime`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    rng = np.random.default_rng(random_state)
    fold_of_day = _block_fold_assignment(n, n_splits, block_size, rng)

    oof_prob = np.full(n, np.nan, dtype=np.float64)
    for f in range(n_splits):
        va = fold_of_day == f
        tr = ~va
        if not va.any():
            continue
        y_tr = y[tr]
        if y_tr.sum() < 2 or y_tr.sum() > len(y_tr) - 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        oof_prob[va] = clf.predict_proba(X_va)[:, 1]
    return oof_prob


def run_stage2_logistic_for_z500_regime(
    *,
    z_regime_id: int,
    z500_regimes_path: Path,
    pca_paths_by_field: dict[str, Path],
    joint_features_alldays_path: Path,
    joint_features_subset_path: Path,
    ward_hc_path: Path,
    extreme_labels_path: Path,
    extreme_var: str,
    out_path: Path,
    random_state: int = 42,
    overwrite: bool = False,
    residualize: bool = False,
    fit_mode: str = FIT_MODE_IN_SAMPLE,
    cv_n_splits: int = DEFAULT_CV_N_SPLITS,
    cv_block_size: int = DEFAULT_CV_BLOCK_SIZE,
) -> RegimeLogisticResult:
    """Stage 2 [LOGISTIC, production] for one Z500 regime: reuse the SAME
    cached joint-features/Ward-HC building blocks the discrete k-means
    production path uses (`per_regime_stage2.prepare_stage2_inputs_for_z500_regime`
    — unmodified, so the two Stage-2 modes never recompute or diverge on their
    shared inputs), then fit a logistic regression against the extreme-day
    label and write the per-day predicted probability in place of a discrete
    subregime cluster label.

    `fit_mode` controls what `predicted_extreme_prob` is (see module
    docstring): `"in_sample"` (default) or `"out_of_fold_blocked_cv"`.
    `logistic_coef`/`logistic_intercept` are always the in-sample fit either
    way. The cached `out_path` (if present) is only reused when its `fit_mode`
    attr matches the requested one -- switching `--fit-mode` without also
    changing `--output-tag` raises rather than silently mixing the two.
    """
    if fit_mode not in VALID_FIT_MODES:
        raise ValueError(f"fit_mode must be one of {VALID_FIT_MODES}, got {fit_mode!r}")

    if out_path.is_file() and not overwrite:
        with xr.open_dataset(out_path, chunks=None) as ds:
            cached_fit_mode = str(ds.attrs.get("fit_mode", FIT_MODE_IN_SAMPLE))
            if cached_fit_mode != fit_mode:
                raise ValueError(
                    f"{out_path} was written with fit_mode={cached_fit_mode!r} but "
                    f"fit_mode={fit_mode!r} was requested -- pass a distinct --output-tag "
                    "for this fit mode, or --overwrite to replace the cached file."
                )
            time_dim = ds["predicted_extreme_prob"].dims[0]
            day_index = pd.DatetimeIndex(ds[time_dim].values)
            predicted_prob = np.asarray(ds["predicted_extreme_prob"].values, dtype=np.float64)
            coef = np.asarray(ds["logistic_coef"].values, dtype=np.float64)
            scaler_mean = np.asarray(json.loads(ds.attrs["logistic_scaler_mean"]), dtype=np.float64)
            scaler_scale = np.asarray(json.loads(ds.attrs["logistic_scaler_scale"]), dtype=np.float64)
            return RegimeLogisticResult(
                z_regime_id=int(z_regime_id),
                # See per_regime_stage2.run_stage2_for_z500_regime: preserve caller-supplied
                # (greedy-AUC) order rather than re-sorting alphabetically.
                fields=tuple(pca_paths_by_field.keys()),
                n_days=int(ds.attrs["n_days"]),
                n_extreme=int(ds.attrs["n_extreme"]),
                time_dim=time_dim,
                day_index=day_index,
                predicted_prob=predicted_prob,
                coef=coef,
                intercept=float(ds.attrs["logistic_intercept"]),
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                degenerate=bool(ds.attrs.get("degenerate", False)),
            )

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

    with xr.open_dataset(joint_features_subset_path, chunks=None) as ds:
        time_dim = ds["joint_features"].dims[0]
        feat_times = pd.DatetimeIndex(ds[time_dim].values).normalize()
        X_all = np.asarray(ds["joint_features"].values, dtype=np.float64)

    label_times, y_all = _load_extreme_labels(extreme_labels_path, extreme_var)
    label_times = pd.DatetimeIndex(label_times).normalize()

    common = pd.Index(feat_times).intersection(pd.Index(label_times))
    if len(common) < len(feat_times):
        print(f"[stage2_logistic] Z500-{z_regime_id}: {len(feat_times)} regime days ∩ "
              f"{len(label_times)} extreme-label days = {len(common)} common days")
    idx_X = pd.Index(feat_times).get_indexer(common)
    idx_y = pd.Index(label_times).get_indexer(common)
    X = X_all[idx_X]
    y = np.asarray(y_all[idx_y], dtype=np.int64)
    day_index = pd.DatetimeIndex(common)

    n_days = len(y)
    n_extreme = int(y.sum())
    n_feat = X.shape[1]

    degenerate = n_extreme < 2 or n_extreme > n_days - 2
    if degenerate:
        base_rate = float(n_extreme / n_days) if n_days else float("nan")
        print(f"[stage2_logistic] WARNING Z500-{z_regime_id}: degenerate label "
              f"(n_extreme={n_extreme} of {n_days} days) -- writing a constant "
              f"base-rate probability instead of a fitted logistic index.")
        predicted_prob = np.full(n_days, base_rate, dtype=np.float64)
        coef = np.zeros(n_feat, dtype=np.float64)
        intercept = float(np.log(base_rate / (1.0 - base_rate))) if 0.0 < base_rate < 1.0 else float("nan")
        scaler_mean = np.zeros(n_feat, dtype=np.float64)
        scaler_scale = np.ones(n_feat, dtype=np.float64)
    elif fit_mode == FIT_MODE_OOF_BLOCKED_CV:
        # logistic_coef/intercept: still the in-sample fit (coefficients-only -- see
        # module docstring), decoupled from predicted_extreme_prob below.
        _, coef, intercept, scaler_mean, scaler_scale = fit_regime_logistic(X, y, random_state=random_state)
        predicted_prob = fit_regime_logistic_out_of_fold(
            X, y, n_splits=cv_n_splits, block_size=cv_block_size, random_state=random_state,
        )
        n_nan = int(np.isnan(predicted_prob).sum())
        if n_nan:
            base_rate = float(y.mean())
            print(f"[stage2_logistic] WARNING Z500-{z_regime_id}: {n_nan}/{n_days} days had no "
                  f"out-of-fold prediction (their fold's training split had too few/many positive "
                  f"labels) -- filled with the regime base rate ({base_rate:.3f}).")
            predicted_prob = np.where(np.isnan(predicted_prob), base_rate, predicted_prob)
    else:
        predicted_prob, coef, intercept, scaler_mean, scaler_scale = fit_regime_logistic(
            X, y, random_state=random_state,
        )

    fit_mode_caveat = FIT_MODE_CAVEAT_OOF_BLOCKED_CV if fit_mode == FIT_MODE_OOF_BLOCKED_CV else FIT_MODE_CAVEAT_IN_SAMPLE
    ds_out = xr.Dataset(
        data_vars={
            "predicted_extreme_prob": xr.DataArray(
                predicted_prob, dims=[time_dim], coords={time_dim: day_index}, name="predicted_extreme_prob",
            ),
            "logistic_coef": xr.DataArray(
                coef, dims=["feature"], coords={"feature": np.arange(n_feat)}, name="logistic_coef",
            ),
        },
        attrs={
            "workflow_family": "stage2_residual",
            "step": "09_per_regime_logistic",
            "z500_regime": int(z_regime_id),
            "fields": "+".join(pca_paths_by_field.keys()),
            "n_days": int(n_days),
            "n_extreme": int(n_extreme),
            "degenerate": int(degenerate),
            "logistic_intercept": float(intercept),
            "logistic_scaler_mean": json.dumps(scaler_mean.tolist()),
            "logistic_scaler_scale": json.dumps(scaler_scale.tolist()),
            "fit_mode": fit_mode,
            "fit_mode_caveat": fit_mode_caveat,
            "cv_n_splits": int(cv_n_splits) if fit_mode == FIT_MODE_OOF_BLOCKED_CV else -1,
            "cv_block_size": int(cv_block_size) if fit_mode == FIT_MODE_OOF_BLOCKED_CV else -1,
            "extreme_labels_path": str(extreme_labels_path),
            "extreme_var": str(extreme_var),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)

    return RegimeLogisticResult(
        z_regime_id=int(z_regime_id),
        # See per_regime_stage2.run_stage2_for_z500_regime: preserve caller-supplied
        # (greedy-AUC) order rather than re-sorting alphabetically.
        fields=tuple(pca_paths_by_field.keys()),
        n_days=n_days,
        n_extreme=n_extreme,
        time_dim=time_dim,
        day_index=day_index,
        predicted_prob=predicted_prob,
        coef=coef,
        intercept=intercept,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        degenerate=degenerate,
    )


def _domain_q_tag_from_filename(path: Path | str | None) -> str:
    """Extract the `domainQ<NNN>` token (e.g. "domainQ900") from a convective_duration
    filename, or "" if absent/not given — same convention rr_022/rr_023's TAG_SUFFIX
    auto-derivation uses (see README_DOMAIN_PCTL_Q.md)."""
    import re

    if path is None:
        return ""
    m = re.search(r"domainQ\d+", Path(path).name)
    return m.group(0) if m else ""


def merge_per_regime_logistic_outputs(
    results: dict[int, RegimeLogisticResult],
    *,
    out_path: Path,
    overwrite: bool = False,
    fit_mode: str = FIT_MODE_IN_SAMPLE,
    extreme_labels_path: Path | str | None = None,
) -> Path:
    """Flatten independent per-Z500-regime logistic fits into one file: a
    continuous `predicted_extreme_prob` per day plus which Z500 regime each
    day belongs to — the continuous-index analogue of
    `per_regime_stage2.merge_per_regime_outputs`'s discrete `regime_labels`
    merge. `logistic_coef` is kept as one row per Z500 regime (padded to the
    widest regime's feature count) rather than flattened, since a regime's
    coefficient vector is only meaningful against that regime's own field
    subset/scaler.

    `extreme_labels_path`, if given, is recorded in the output's attrs (and the
    `domainQ<NNN>` token parsed out of its filename, if any) so downstream
    provenance checks (e.g. `export_regime_rr_statistics_csv.py`'s PROVENANCE.json)
    can confirm which EPE domain-percentile threshold actually produced this file —
    unlike the per-regime pre-merge files (which already carry it, see
    `run_stage2_logistic_for_z500_regime`), the merged file had no such record before.
    """
    if out_path.is_file() and not overwrite:
        return out_path

    z_ids = sorted(results.keys())
    time_dim = results[z_ids[0]].time_dim

    all_times = [np.asarray(results[z].day_index.values) for z in z_ids]
    all_probs = [results[z].predicted_prob for z in z_ids]
    all_zregime = [np.full(len(results[z].day_index), z, dtype=np.int16) for z in z_ids]

    times_concat = np.concatenate(all_times)
    probs_concat = np.concatenate(all_probs)
    zregime_concat = np.concatenate(all_zregime)
    order = np.argsort(times_concat)

    max_feat = max(len(results[z].coef) for z in z_ids)
    coef_blocks = []
    for z in z_ids:
        c = results[z].coef.astype(np.float32)
        if len(c) < max_feat:
            c = np.concatenate([c, np.zeros(max_feat - len(c), dtype=np.float32)])
        coef_blocks.append(c)
    coef_arr = np.stack(coef_blocks, axis=0)

    regime_coord = np.array(z_ids, dtype=int)
    ds_out = xr.Dataset(
        data_vars={
            "predicted_extreme_prob": xr.DataArray(
                probs_concat[order].astype(np.float32), dims=[time_dim],
                coords={time_dim: times_concat[order]}, name="predicted_extreme_prob",
            ),
            "z500_regime": xr.DataArray(
                zregime_concat[order], dims=[time_dim],
                coords={time_dim: times_concat[order]}, name="z500_regime",
            ),
            "logistic_coef": xr.DataArray(
                coef_arr, dims=["regime", "feature"],
                coords={"regime": regime_coord, "feature": np.arange(max_feat)}, name="logistic_coef",
            ),
        },
        attrs={
            "workflow_family": "stage2_residual",
            "step": "09_per_regime_logistic_merge",
            "per_regime_mode": "true",
            "fit_mode": fit_mode,
            "fit_mode_caveat": FIT_MODE_CAVEAT_OOF_BLOCKED_CV if fit_mode == FIT_MODE_OOF_BLOCKED_CV else FIT_MODE_CAVEAT_IN_SAMPLE,
            "per_regime_fields": json.dumps({str(z): list(results[z].fields) for z in z_ids}),
            "per_regime_n_days": json.dumps({str(z): results[z].n_days for z in z_ids}),
            "per_regime_n_extreme": json.dumps({str(z): results[z].n_extreme for z in z_ids}),
            "per_regime_intercept": json.dumps({str(z): results[z].intercept for z in z_ids}),
            "per_regime_degenerate": json.dumps({str(z): results[z].degenerate for z in z_ids}),
            "extreme_labels_path": str(extreme_labels_path) if extreme_labels_path is not None else "",
            "domain_pctl_q_tag": _domain_q_tag_from_filename(extreme_labels_path),
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    return out_path


def bin_predicted_prob_per_regime(
    predicted_prob: np.ndarray,
    z500_regime: np.ndarray,
    *,
    n_bins: int = DEFAULT_N_BINS,
    bin_edges_by_regime: dict[int, Sequence[float]] | None = None,
) -> np.ndarray:
    """Quantile-bin each Z500 regime's own `predicted_extreme_prob` into
    `n_bins` groups (bin ids 1..n_bins, increasing with probability) — turns
    the continuous "gradient, not groups" index into composite-able day
    groups without ever refitting or storing a discrete label on disk (see
    README_ResidualRegimes.md TODO "Build the composite-from-probability-bins
    plotting path"). Bin count is a plotting-time choice, deliberately not
    baked into the merged NetCDF the way k-means `regime_labels` are, so it
    can be swept without rerunning the logistic fit.

    Fixed n_bins=2 ("extreme-like" vs "non-extreme-like" half of each regime)
    is the LEGACY interim default, only used when `bin_edges_by_regime` is
    entirely omitted (e.g. an ad-hoc exploratory call). Production callers
    (`plot_composites.py`/`plot_rgb_overview.py`/`plot_rr_statistics.py`/
    `plot_subregimes_variance.py`) always pass the data-driven
    `bin_edges_by_regime` from `select_subregime_bins.py`.

    `bin_edges_by_regime`, if given (non-empty), overrides the uniform
    `n_bins` quantile split for any regime present as a key: probability
    edges (ascending, e.g. from `select_regime_bin_edges` /
    `select_subregime_bins.py`) are applied with `np.digitize` instead, so
    different regimes can end up with a different — and non-uniform-width —
    bin count, chosen from where each regime's own extreme-day frequency
    actually changes rather than a flat split everywhere. A regime NOT
    present in the dict is kept as a single undivided group (bin=1 for all
    its days) rather than falling back to the plain `n_bins` quantile split
    — `select_subregime_bins.py` only omits a regime when it found no
    data-driven evidence of a meaningful sub-split (too few extreme days to
    clear `--min-extreme-per-bin`), so forcing an arbitrary near-equal 2-way
    cut there would be exactly the legacy behavior this override mechanism
    replaced.

    Same qcut/duplicates="drop" recipe as
    `residual_extreme_signal_diag_LOGISTIC.quantile_frequency_table`. A
    regime whose probabilities are too tied to support `n_bins` distinct
    quantile edges (e.g. many days sharing the exact same fitted value) is
    reported with a WARNING and silently falls back to as many bins as the
    data actually support, down to a single bin in the fully degenerate
    case — this quantile-tie fallback only applies in the no-overrides
    (legacy) mode described above.
    """
    bin_edges_by_regime = bin_edges_by_regime or {}
    using_overrides = bool(bin_edges_by_regime)
    prob_bin = np.zeros(len(predicted_prob), dtype=np.int16)
    for z in sorted({int(v) for v in z500_regime}):
        mask = z500_regime == z
        p = predicted_prob[mask]

        edges = bin_edges_by_regime.get(z)
        if edges is not None and len(edges):
            b = np.digitize(p, np.asarray(edges, dtype=float)) + 1
            prob_bin[mask] = b
            print(f"[stage2_logistic] Z500-{z}: data-driven bins ({len(edges) + 1} groups, "
                  f"overriding n_bins={n_bins})")
            continue

        if using_overrides:
            # select_subregime_bins.py deliberately omits a regime from bin_edges_by_regime
            # when it couldn't clear --min-extreme-per-bin even once (see
            # select_bins_for_region's docstring) -- i.e. this regime has no data-driven
            # evidence of a meaningful sub-split. Keeping it as a single undivided group is
            # the current (non-legacy) behavior: falling back to the flat n_bins quantile
            # split here would silently force an arbitrary near-equal-sized cut with no
            # data-driven meaning, which is exactly the "legacy 50/50" behavior this
            # override mechanism replaced.
            prob_bin[mask] = 1
            print(f"[stage2_logistic] Z500-{z}: no data-driven bin split available -- "
                  f"keeping as a single undivided group ({int(mask.sum())} days)")
            continue

        n_bins_z = max(1, min(int(n_bins), len(np.unique(p))))
        try:
            raw = pd.qcut(p, q=n_bins_z, labels=False, duplicates="drop")
            if pd.isna(raw).any():
                raise ValueError("qcut collapsed to NaN bins")
            b = np.asarray(raw, dtype=int) + 1
        except ValueError:
            b = np.ones(len(p), dtype=int)
        n_actual = int(b.max()) if len(b) else 0
        if n_actual < n_bins:
            print(f"[stage2_logistic] WARNING Z500-{z}: requested n_bins={n_bins} but only "
                  f"{n_actual} distinct bin(s) resulted (predicted_extreme_prob too tied/degenerate)")
        prob_bin[mask] = b
    return prob_bin


def select_regime_bin_edges(
    predicted_prob: np.ndarray,
    y: np.ndarray,
    *,
    max_bins: int = 2,
    min_extreme_per_bin: int = 5,
    n_fine_bins: int | None = None,
    min_recall_frac: float | None = None,
    target_day_frac: float | None = None,
) -> np.ndarray:
    """Data-driven probability edges for ONE Z500 regime, for
    `bin_predicted_prob_per_regime`'s `bin_edges_by_regime` — the
    finest-split-per-regime rule from README_ResidualRegimes.md's "Decide
    per-regime bin count" TODO, in place of a flat `n_bins` everywhere.

    If `target_day_frac` is given, it TAKES PRECEDENCE over everything below
    (`max_bins`/`min_extreme_per_bin`/`n_fine_bins`/`min_recall_frac` are all
    ignored): the edge is simply `quantile(predicted_prob, 1 - target_day_frac)`,
    i.e. a fixed FRACTION OF DAYS in the high bin (e.g. 0.3 -> top 30% of this
    regime's days by predicted probability), always producing exactly 2
    groups. Unlike `min_recall_frac` (which fixes the fraction of EXTREME DAYS
    captured and lets the day-count fall out wherever it lands) this fixes the
    day-count and lets recall/precision fall out wherever THEY land -- useful
    once you already know from a `min_recall_frac` run how concentrated a
    regime's signal is and want to ask the reverse question ("what recall do I
    get if I'm willing to flag 30% of days as high-risk?", which can be far
    higher than the recall you'd get by targeting a small, pure bin). Read the
    resulting `recall_of_regime_extreme`/`extreme_frequency` columns in
    `select_subregime_bins.py`'s summary CSV afterward to see the trade-off
    this landed on -- it isn't targeted directly by this mode.

    Recipe (when `target_day_frac` is NOT given): quantile-split `predicted_prob`
    into `n_fine_bins` fine bins (same
    qcut/duplicates="drop" as `bin_predicted_prob_per_regime`) — this is the
    RESOLUTION of candidate cut points, independent of the output bin count —
    then greedily merge fine bins from the HIGH-probability end downward,
    closing a group as soon as its accumulated `y` count clears the CLOSING
    THRESHOLD, and STOP once `max_bins - 1` boundaries have been closed (the
    OUTPUT cap): any remaining fine bins are folded into a single floor group
    rather than becoming further groups.

    The closing threshold is `min_extreme_per_bin` (an absolute extreme-day
    count) by default. If `min_recall_frac` is given instead, the threshold
    becomes `max(min_extreme_per_bin, ceil(min_recall_frac * y.sum()))` — a
    FRACTION of this regime's own total extreme days, with `min_extreme_per_bin`
    kept as a floor so a low-signal regime doesn't collapse to a 1-2-day bin.
    This matters because `min_extreme_per_bin` alone is an absolute count
    applied uniformly across regimes with very different total extreme-day
    counts, so the same value can close the high bin at wildly different
    RECALL levels per regime (e.g. 17% vs 50% of that regime's own extreme
    days observed on real europe B5 data with min_extreme_per_bin=40) --
    `min_recall_frac` targets a consistent recall level instead, letting the
    resulting day-count/purity trade-off vary per regime rather than the
    count threshold.

    `n_fine_bins` defaults to `max_bins` (old behavior, unchanged) — but note
    that with the default `max_bins=2`, `n_fine_bins=2` gives exactly ONE
    candidate boundary (the median), so the result can only ever be a 50/50
    split or, if that fails to clear `min_extreme_per_bin`, a fallback to the
    plain median split. To get a non-50/50 "high" bin (e.g. a purer top 30%)
    while still capping the OUTPUT at 2 bins, pass a larger `n_fine_bins`
    (e.g. 10, for decile-resolution candidates) — the greedy merge then has
    room to place the single boundary somewhere other than the median, based
    on where `min_extreme_per_bin` extreme days actually accumulate from the
    top down.

    Any leftover low-probability fine bins that never individually or
    cumulatively clear the threshold, or that fall below the `max_bins` cap,
    become their own single floor group — a "no elevated risk" catch-all
    sitting below every group that did clear the threshold — rather than
    being folded into the lowest DATA-CLEARED group (which would silently
    erase that group's own boundary) or left unassigned. This is what lets a
    regime with a genuine gradual extreme-frequency increase (e.g. europe R4,
    ontario R1 in the 2026-07-30 freq tables) keep several bins (when
    `max_bins` allows it), while a step-function regime (e.g. europe R1/R3,
    ontario R3, where only the top ~20% of days carry any signal) collapses
    to few.

    Returns ascending probability edges (length = n_groups - 1, possibly 0 if
    even the coarsest 1-bin merge can't clear the threshold — e.g. the regime
    has fewer than `min_extreme_per_bin` extreme days in total). Callers
    should treat an empty return as "fall back to the plain n_bins split" for
    this regime, not as "0 bins".
    """
    predicted_prob = np.asarray(predicted_prob, dtype=np.float64)
    y = np.asarray(y)
    if predicted_prob.size == 0:
        return np.array([])

    if target_day_frac is not None:
        edge = float(np.quantile(predicted_prob, 1.0 - target_day_frac))
        return np.array([edge])

    n_fine_bins_eff = max_bins if n_fine_bins is None else n_fine_bins
    n_fine_bins_eff = max(1, min(int(n_fine_bins_eff), len(np.unique(predicted_prob))))
    if n_fine_bins_eff <= 1:
        return np.array([])
    try:
        raw = pd.qcut(predicted_prob, q=n_fine_bins_eff, labels=False, duplicates="drop")
    except ValueError:
        return np.array([])
    raw = np.asarray(raw, dtype=int)
    k_fine = int(raw.max()) + 1 if raw.size else 0
    if k_fine <= 1:
        return np.array([])

    extreme_per_fine = np.array([int(y[raw == b].sum()) for b in range(k_fine)])
    # edges_fine[i] is the probability value pd.qcut actually cut at between
    # fine-bin i and fine-bin i+1 (i.e. the min of fine-bin i+1).
    edges_fine = np.array([predicted_prob[raw == b].min() for b in range(1, k_fine)])

    closing_threshold = min_extreme_per_bin
    if min_recall_frac is not None:
        total_extreme = int(y.sum())
        closing_threshold = max(min_extreme_per_bin, int(np.ceil(min_recall_frac * total_extreme)))

    group_starts: list[int] = []
    acc = 0
    for b in range(k_fine - 1, -1, -1):
        if len(group_starts) >= max_bins - 1:
            # Output cap reached -- everything from here down folds into the
            # single floor group added below, regardless of fine resolution.
            break
        acc += extreme_per_fine[b]
        if acc >= closing_threshold:
            group_starts.append(b)
            acc = 0
    if not group_starts:
        # Not even a single (all-fine-bins) group clears the threshold.
        return np.array([])

    group_starts = sorted(group_starts)
    if group_starts[0] != 0:
        # Fine bins below the lowest closed group never individually or
        # cumulatively cleared the threshold -- fold them into their own
        # "no elevated risk" floor group (start at fine-bin 0) rather than
        # dropping them or merging them INTO the lowest closed group (which
        # would silently erase that group's own boundary).
        group_starts = [0] + group_starts
    edges = [edges_fine[s - 1] for s in group_starts if s > 0]
    return np.asarray(sorted(set(edges)), dtype=float)


def load_bin_overrides_yaml(path: Path, bbox_step: str) -> dict[int, list[float]]:
    """Read a `select_subregime_bins.py`-written bin-overrides YAML (mirrors
    `k_overrides_<region>_custom.yaml`'s one-file-per-region, keyed-by-step
    layout) and return `{z500_regime: edges}` for this `bbox_step`, ready to
    pass as `bin_predicted_prob_per_regime`'s `bin_edges_by_regime`. Returns
    `{}` if the file has no entry for this step -- since an empty dict reads
    as "no overrides requested" to `bin_predicted_prob_per_regime`, that
    silently drops ALL regimes in this step back to the legacy flat `n_bins`
    split (not just the single-undivided-group treatment a regime individually
    omitted from a non-empty dict gets) -- printed loudly here rather than
    left for a downstream identical-bin-count symptom to surface silently."""
    import yaml

    doc = yaml.safe_load(Path(path).read_text()) or {}
    step_entry = (doc.get("bins_by_step") or {}).get(bbox_step) or {}
    if not step_entry:
        print(f"[stage2_logistic] WARNING: {path} has no bin-overrides entry for bbox_step="
              f"{bbox_step!r} -- every Z500 regime in this step will fall back to the LEGACY "
              f"flat n_bins quantile split (run select_subregime_bins.py for this bbox_step "
              f"to get data-driven bins instead).")
    return {int(z): [float(e) for e in edges] for z, edges in step_entry.items()}


def run_logistic_stage2(
    *,
    config,
    z500_regimes_path: Path,
    pca_paths_by_field: dict[str, Path],
    extreme_labels_path: Path,
    extreme_var: str,
    regimes: Sequence[int] | None = None,
    output_tag: str = "",
    fit_mode: str = FIT_MODE_IN_SAMPLE,
    cv_n_splits: int = DEFAULT_CV_N_SPLITS,
    cv_block_size: int = DEFAULT_CV_BLOCK_SIZE,
) -> Path:
    """Orchestrate the per-regime logistic fits + merge + summary CSV — the
    LOGISTIC counterpart of per_regime_stage2/auxiliary.run_pipeline's discrete
    per-regime k-means loop, without the rare-regime remap or regime
    characterisation steps (both cluster-label-specific; not meaningful for a
    continuous index)."""
    from .paths import ResidualWorkflowPaths

    paths = ResidualWorkflowPaths(config)
    all_z = sorted(config.per_regime_field_map.keys())
    target_z = sorted(regimes) if regimes else all_z
    unknown = set(target_z) - set(all_z)
    if unknown:
        raise ValueError(f"--regimes contains ids not in the field map: {sorted(unknown)} (have {all_z})")

    results: dict[int, RegimeLogisticResult] = {}
    summary_rows = []
    for z in target_z:
        fields_for_z = {f: pca_paths_by_field[f] for f in config.per_regime_field_map[z] if f != "z500"}
        print(f"[stage2_logistic] Z500-{z}: fields={'+'.join(sorted(fields_for_z))} — "
              f"reusing/building cached joint features...")
        res = run_stage2_logistic_for_z500_regime(
            z_regime_id=z,
            z500_regimes_path=z500_regimes_path,
            pca_paths_by_field=fields_for_z,
            joint_features_alldays_path=paths.per_region_joint_features_alldays_path(z),
            joint_features_subset_path=paths.per_region_joint_features_path(z),
            ward_hc_path=paths.per_region_ward_hc_path(z),
            extreme_labels_path=extreme_labels_path,
            extreme_var=extreme_var,
            out_path=paths.per_region_logistic_index_path(z, output_tag=output_tag),
            random_state=config.random_state,
            overwrite=config.overwrite,
            residualize=config.residualize,
            fit_mode=fit_mode,
            cv_n_splits=cv_n_splits,
            cv_block_size=cv_block_size,
        )
        results[z] = res
        base_rate = res.n_extreme / res.n_days if res.n_days else float("nan")
        print(f"[stage2_logistic] Z500-{z}: n_days={res.n_days} n_extreme={res.n_extreme} "
              f"base_rate={base_rate:.3f} degenerate={res.degenerate}")
        summary_rows.append({
            "z500_regime": z,
            "residualize": bool(config.residualize),
            "fields": "+".join(sorted(fields_for_z)),
            "n_days": res.n_days,
            "n_extreme": res.n_extreme,
            "base_rate": round(base_rate, 4) if not np.isnan(base_rate) else float("nan"),
            "degenerate": res.degenerate,
            "fit_mode": fit_mode,
        })

    merged_path = merge_per_regime_logistic_outputs(
        results, out_path=paths.logistic_index_path_per_regime(output_tag=output_tag), overwrite=config.overwrite,
        fit_mode=fit_mode, extreme_labels_path=extreme_labels_path,
    )
    print(f"[stage2_logistic] Wrote {merged_path}")

    summary_csv = paths.logistic_index_summary_csv_path(output_tag=output_tag)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[stage2_logistic] Wrote {summary_csv}")

    return merged_path


def main() -> int:
    import sys

    from .auxiliary import _find_legacy_z500_regimes
    from .config import build_arg_parser, config_from_args
    from .paths import resolve_mf_pca_path
    from common.extremes_spec import resolve_convective_duration_path

    parser = build_arg_parser()
    parser.add_argument("--extreme-labels-path", type=Path, required=True,
                         help="Explicit convective_duration NetCDF for the pinned EPE definition "
                              "(e.g. the domain_pctl_q=0.90, --domain-agg lexmax TEST file — see "
                              "README_DOMAIN_PCTL_Q.md). Required: ExtremesSpec tag matching does not "
                              "encode domain_pctl_q/domain_agg, so auto-resolution could silently pick "
                              "the wrong domain-percentile threshold among several sharing the same "
                              "point-wise --q tag.")
    parser.add_argument("--extreme-var", default="flagged_domain_pctl",
                         help="Binary extreme-day label variable (default: the pinned domain_pctl_q "
                              "definition's flag, not rr_01's flagged_any_gridcell).")
    parser.add_argument("--regimes", default=None,
                         help="Comma-separated Z500 regime ids to restrict to (default: all regimes "
                              "in --regime-fields-map).")
    parser.add_argument("--output-tag", default="",
                         help="Appended to output filenames so runs that differ only by "
                              "--extreme-labels-path/--extreme-var/--fit-mode (e.g. a domain_pctl_q "
                              "comparison sweep, or an in_sample vs. out_of_fold_blocked_cv comparison) "
                              "don't overwrite each other's output. Default: unchanged filenames -- "
                              "REQUIRED to differ from any existing in_sample run's tag when using "
                              "--fit-mode out_of_fold_blocked_cv, since the cache-read guard refuses to "
                              "reuse a file whose fit_mode attr doesn't match the requested one.")
    parser.add_argument("--fit-mode", default=FIT_MODE_IN_SAMPLE, choices=list(VALID_FIT_MODES),
                         help=f"What predicted_extreme_prob is (default: {FIT_MODE_IN_SAMPLE}, unchanged "
                              f"legacy behaviour). {FIT_MODE_OOF_BLOCKED_CV}: per-day held-out probability "
                              "from the same blocked-CV fold scheme as "
                              "rr_021_stage2_LogRegr_Robustness.py::blocked_cv_auc -- see module "
                              "docstring. logistic_coef is always the in-sample fit either way.")
    parser.add_argument("--cv-n-splits", type=int, default=DEFAULT_CV_N_SPLITS,
                         help=f"Fold count for --fit-mode {FIT_MODE_OOF_BLOCKED_CV} (default {DEFAULT_CV_N_SPLITS}, "
                              "matching rr_021_stage2_LogRegr_Robustness.py). Ignored for in_sample.")
    parser.add_argument("--cv-block-size", type=int, default=DEFAULT_CV_BLOCK_SIZE,
                         help=f"Contiguous-day block length for --fit-mode {FIT_MODE_OOF_BLOCKED_CV}'s fold "
                              f"assignment (default {DEFAULT_CV_BLOCK_SIZE}, matching "
                              "rr_021_stage2_LogRegr_Robustness.py). Ignored for in_sample.")
    args = parser.parse_args()

    if not args.per_regime_mode:
        print("ERROR: stage2_logistic requires --per-regime-mode (with --regime-fields-map) — "
              "it fits one logistic regression per Z500 regime.", file=sys.stderr)
        return 1

    config = config_from_args(args)

    regimes = None
    if args.regimes:
        regimes = [int(z.strip()) for z in args.regimes.split(",") if z.strip()]

    try:
        extreme_labels_path = resolve_convective_duration_path(
            config.module_root, config.region, legacy_path=args.extreme_labels_path,
        )
    except FileNotFoundError as exc:
        print(f"[stage2_logistic] ERROR: {exc}", file=sys.stderr)
        return 1

    z500_regimes_path = config.z500_regimes_override or _find_legacy_z500_regimes(config)
    if z500_regimes_path is None:
        print(f"ERROR: No Stage 1 Z500 regimes file found for z500_k={config.z500_k}, "
              f"bbox={config.bbox_step}, region={config.region}.", file=sys.stderr)
        return 1

    all_fields = sorted({f for fs in config.per_regime_field_map.values() for f in fs if f != "z500"})
    pca_paths_by_field: dict[str, Path] = {}
    missing = []
    for f in all_fields:
        p = resolve_mf_pca_path(config, f)
        if p is not None:
            pca_paths_by_field[f] = p
        else:
            missing.append(f)
    if missing:
        print(f"ERROR: missing rr_00 07_MF_PCA files: {missing}", file=sys.stderr)
        return 1

    if args.fit_mode == FIT_MODE_OOF_BLOCKED_CV and not args.output_tag:
        print(f"[stage2_logistic] WARNING: --fit-mode {FIT_MODE_OOF_BLOCKED_CV} with no --output-tag -- "
              "this will collide with (and refuse to silently overwrite) any existing in_sample cache "
              "at the same path unless --overwrite is also set. Recommended: --output-tag oofcv.")

    run_logistic_stage2(
        config=config, z500_regimes_path=z500_regimes_path, pca_paths_by_field=pca_paths_by_field,
        extreme_labels_path=extreme_labels_path, extreme_var=args.extreme_var, regimes=regimes,
        output_tag=args.output_tag, fit_mode=args.fit_mode,
        cv_n_splits=args.cv_n_splits, cv_block_size=args.cv_block_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
