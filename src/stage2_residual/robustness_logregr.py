from __future__ import annotations

"""Standalone diagnostic -- robustness/stability checks for the production Stage 2 LOGISTIC
per-regime fit (`stage2_logistic.fit_regime_logistic`). NOT part of the production pipeline
and NOT imported by any other module. Safe to delete this file (and its sbatch launcher,
scripts/rr_021_stage2_LogRegr_Robustness_tu_hpc.sh) at any time without touching
stage2_logistic.py, residual_extreme_signal_diag_LOGISTIC.py, or anything else in
residual_regimes/.

`residual_extreme_signal_diag_LOGISTIC.py` already answers "is there real signal at all"
(cv_auc vs. a permutation-shuffled null -- see README_ResidualRegimes.md's Main Finding
"Stage 2 is a gradient, not groups"). This diagnostic answers the follow-up question a
"the logistic regression is robust" argument actually needs -- not just "does the model beat
chance on average," but "would the SAME coefficient pattern and skill hold up on a different
sample / a stricter CV scheme / a different regularization strength":

  1. **Block-bootstrap coefficient stability** (`coefficient_stability`): refit
     `stage2_logistic.fit_regime_logistic` -- the exact production per-regime recipe -- on
     `n_bootstrap` block-bootstrap resamples (contiguous `block_size`-day blocks, matching
     `feature_relevance._bootstrap_greedy_stability`'s block-bootstrap rationale: block
     resampling instead of iid day resampling preserves synoptic-scale autocorrelation).
     Reports, per residual-field PC, the fraction of resamples that reproduce the
     full-sample coefficient's SIGN -- the direct logistic-regression analogue of the ARI
     instability that disqualified k-means/Ward/HDBSCAN clustering on this same feature
     space. Same interpretation convention as `feature_relevance`'s `selection_frequency`:
     >=0.8 sign-consistency -> robust, 0.4-0.8 -> context-dependent, <0.4 -> unstable (don't
     over-interpret that feature's sign/attribution).
  2. **Autocorrelation-aware CV AUC** (`blocked_cv_auc`, `leave_one_year_out_auc`): the
     existing `_cv_auc` (used for the permutation-null test) is a per-day
     `StratifiedKFold(shuffle=True)` -- with daily weather's serial correlation, a day's
     immediate neighbour (often the same synoptic event) can land in the opposite fold,
     optimistically inflating AUC. `blocked_cv_auc` reruns the identical fit recipe with
     `block_size`-day contiguous blocks assigned round-robin to folds (BLOCKS, not days, are
     shuffled), so within-block leakage across train/test is structurally impossible.
     `leave_one_year_out_auc` is the stricter, no-shuffle-at-all version: train on all-but-one
     calendar year, test on the held-out year, repeated for every year -- true temporal
     generalization, immune to both short-range (block) and any residual long-range/seasonal
     leakage. If `cv_auc` (random-fold) collapses under either stricter scheme, that is itself
     an important, reportable caveat -- not a reason to suppress the check.
  3. **Regularization sensitivity** (`c_sweep_auc`): `blocked_cv_auc` at a small grid of the
     logistic regression's inverse-regularization strength `C` -- shows the reported skill
     isn't an artifact of sklearn's default `C=1.0` (used everywhere else in this pipeline).

Run (mirrors residual_extreme_signal_diag_LOGISTIC.py's CLI so it resolves the IDENTICAL
cached joint-features files):

  python -m stage2_residual.robustness_logregr \\
    --region europe --bbox-step B5 --z500-k 4 --per-regime-mode \\
    --regime-fields-map '{"1":"z500+cape+w700","2":"z500+cape"}' \\
    --study-area germany --extremes-q 0.999 \\
    --extreme-labels-path /path/to/convective_duration_TEST_..._domainQ950....nc

Or via the sbatch launcher (rebuilds the same field map rr_021/the LOGISTIC diagnostic used):
  sbatch --export=ALL,REGION=europe,BBOX_STEP=B5,EXTREME_LABELS_PATH=... \\
    rr_021_stage2_LogRegr_Robustness_tu_hpc.sh

Output (never collides with rr021_k_sweep_*/rr_HDBSCAN_diag_*/rr_LOGISTIC_diag_* files):
  results/residual_regimes/diagnostics/{region}/rr021_stage2_LogRegr_Robustness_Z{z}_coefstability_*.csv
  results/residual_regimes/diagnostics/{region}/rr021_stage2_LogRegr_Robustness_summary_*.csv
  figures/residual_regimes/{region}/rr021_stage2_LogRegr_Robustness_*.png
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .paths import ResidualWorkflowPaths, experiment_id, per_regime_fields_signature, resolve_mf_pca_path
from .per_regime_stage2 import prepare_stage2_inputs_for_z500_regime
from .feature_relevance import _cv_auc, _load_extreme_labels
from .stage2_logistic import _block_fold_assignment, fit_regime_logistic

__all__ = [
    "RegimeRobustnessResult",
    "block_bootstrap_indices",
    "coefficient_stability",
    "blocked_cv_auc",
    "leave_one_year_out_auc",
    "permutation_null_blocked_cv_auc",
    "permutation_null_loyo_auc",
    "c_sweep_auc",
    "analyze_regime_robustness",
    "run_robustness_diagnostic",
    "main",
]

DEFAULT_BLOCK_SIZE = 10  # days; matches feature_relevance._bootstrap_greedy_stability's
# synoptic-autocorrelation rationale -- kept identical so results read as one methodology.
DEFAULT_N_BOOTSTRAP = 200
DEFAULT_N_SPLITS_BLOCKED = 5
DEFAULT_C_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
SIGN_CONSISTENCY_ROBUST = 0.8
SIGN_CONSISTENCY_CONTEXT = 0.4
# 2026-08-12 (by request): dedicated permutation-null significance tests FOR cv_auc_blocked and
# loyo_auc specifically (residual_extreme_signal_diag_LOGISTIC.py's permutation test only
# covers cv_auc_randomfold -- using that p-value to color/report these stricter numbers by
# proxy was judged not precise enough). Costly: leave_one_year_out_auc alone already refits one
# model per usable calendar year (~20-45 for this pipeline's 1979-2025 span); the null below
# reruns that whole loop n_permutations times per regime. Lower this via
# --n-permutations-robustness / N_PERMUTATIONS_ROBUSTNESS if a full region/bbox sweep's runtime
# becomes a problem -- see the sbatch launcher's SBATCH --time comment.
DEFAULT_N_PERMUTATIONS_ROBUSTNESS = 100


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """One block-bootstrap resample of ``range(n)``: contiguous ``block_size``-day blocks
    sampled with replacement, concatenated and truncated back to length n. Same recipe as
    feature_relevance._bootstrap_greedy_stability."""
    starts = np.arange(0, n, block_size)
    chosen = rng.choice(starts, size=len(starts), replace=True)
    idx = np.concatenate([np.arange(s, min(s + block_size, n)) for s in chosen])[:n]
    return idx


def coefficient_stability(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Block-bootstrap sign/CI stability of stage2_logistic.fit_regime_logistic's
    coefficients -- the direct logistic-regression analogue of
    feature_relevance._bootstrap_greedy_stability's selection_frequency, but for this
    regime's single in-sample production fit rather than greedy field selection.

    Returns (per-feature stability table, array of bootstrap-resample in-sample AUCs [same
    fit recipe as production -- NOT a held-out estimate, only used here to check the AUC
    itself doesn't swing wildly across resamples], n_valid resamples actually used)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(random_state)
    n = len(y)
    _, full_coef, _, _, _ = fit_regime_logistic(X, y, random_state=random_state)
    full_sign = np.sign(full_coef)
    n_feat = X.shape[1]

    sign_match = np.zeros(n_feat, dtype=np.int64)
    coef_samples = np.full((n_bootstrap, n_feat), np.nan)
    auc_samples = np.full(n_bootstrap, np.nan)
    n_valid = 0

    for b in range(n_bootstrap):
        idx = block_bootstrap_indices(n, block_size, rng)
        X_b, y_b = X[idx], y[idx]
        n_ex = int(y_b.sum())
        if n_ex < 2 or n_ex > len(y_b) - 2:
            continue  # degenerate resample -- same guard stage2_logistic's production fit uses
        try:
            prob_b, coef_b, _, _, _ = fit_regime_logistic(X_b, y_b, random_state=random_state)
        except Exception:
            continue
        coef_samples[b] = coef_b
        sign_match += (np.sign(coef_b) == full_sign).astype(np.int64)
        try:
            auc_samples[b] = roc_auc_score(y_b, prob_b)
        except ValueError:
            pass
        n_valid += 1

    denom = max(1, n_valid)
    sign_consistency = sign_match / denom
    verdict = np.select(
        [sign_consistency >= SIGN_CONSISTENCY_ROBUST, sign_consistency >= SIGN_CONSISTENCY_CONTEXT],
        ["robust", "context-dependent"],
        default="unstable",
    )
    table = pd.DataFrame({
        "feature": list(feature_names),
        "full_sample_coef": full_coef,
        "coef_ci_lo": np.nanpercentile(coef_samples, 2.5, axis=0),
        "coef_ci_hi": np.nanpercentile(coef_samples, 97.5, axis=0),
        "sign_consistency": np.round(sign_consistency, 3),
        "verdict": verdict,
    }).sort_values("sign_consistency", ascending=False).reset_index(drop=True)
    return table, auc_samples[~np.isnan(auc_samples)], n_valid


def blocked_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = DEFAULT_N_SPLITS_BLOCKED,
    block_size: int = DEFAULT_BLOCK_SIZE,
    random_state: int = 42,
    C: float = 1.0,
) -> tuple[float, float, float, int]:
    """Same CV AUC recipe as feature_relevance._cv_auc (StandardScaler + balanced
    LogisticRegression), but folds are assigned by contiguous block_size-day BLOCK
    (round-robin-shuffled at the block level) instead of _cv_auc's per-day
    StratifiedKFold(shuffle=True) -- guards against a day's highly-autocorrelated neighbour
    landing in the opposite fold.

    Returns (mean, std, median, n_folds_valid) across the n_splits per-fold AUCs -- std/median
    let a caller judge how much a single fold is driving the mean (only n_splits=5 points by
    default, so read the spread as indicative, not a formal CI -- leave_one_year_out_auc's
    per-year spread, with many more points, is the more statistically meaningful one)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    rng = np.random.default_rng(random_state)
    fold_of_day = _block_fold_assignment(n, n_splits, block_size, rng)

    aucs = []
    for f in range(n_splits):
        va = fold_of_day == f
        tr = ~va
        y_tr, y_va = y[tr], y[va]
        if y_va.sum() < 1 or y_va.sum() == len(y_va) or y_tr.sum() < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state,
                                   solver="lbfgs", C=C)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y_va, proba))
        except ValueError:
            pass
    if not aucs:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.asarray(aucs, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std, float(np.median(arr)), len(aucs)


def leave_one_year_out_auc(
    X: np.ndarray, y: np.ndarray, years: np.ndarray, *, random_state: int = 42,
) -> tuple[float, float, float, int]:
    """Strict temporal-generalization AUC: train on all-but-one calendar year, test on the
    held-out year, repeated for every year with enough extreme days in both the held-out year
    and the remaining training years. No shuffling anywhere -- immune to both short-range
    (block) and long-range/seasonal leakage, unlike blocked_cv_auc/_cv_auc. Returns (mean AUC
    across usable years, std, median, number of years actually used) -- with O(10-40) usable
    years typical for this pipeline's 1979-2025 span, the per-year spread is a much more
    meaningful sample than blocked_cv_auc's 5-fold spread, and a year or two with very few
    extreme days can pull the mean around more than the median."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    aucs = []
    for yr in sorted(np.unique(years)):
        va = years == yr
        tr = ~va
        y_tr, y_va = y[tr], y[va]
        if y_va.sum() < 2 or y_va.sum() == len(y_va) or y_tr.sum() < 2:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y_va, proba))
        except ValueError:
            pass
    if not aucs:
        return float("nan"), float("nan"), float("nan"), 0
    arr = np.asarray(aucs, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return float(np.mean(arr)), std, float(np.median(arr)), len(aucs)


def permutation_null_blocked_cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = DEFAULT_N_SPLITS_BLOCKED,
    block_size: int = DEFAULT_BLOCK_SIZE,
    n_permutations: int = DEFAULT_N_PERMUTATIONS_ROBUSTNESS,
    random_state: int = 42,
) -> np.ndarray:
    """Label-shuffled null distribution of blocked_cv_auc's own mean AUC -- same recipe
    residual_extreme_signal_diag_LOGISTIC.py's permutation_null_auc uses for cv_auc_randomfold,
    applied here to the stricter blocked-fold scheme instead so its own p-value doesn't rely on
    a different (easier) CV scheme's significance test by proxy."""
    rng = np.random.RandomState(random_state)
    null_aucs = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        y_shuffled = rng.permutation(y)
        null_aucs[i] = blocked_cv_auc(X, y_shuffled, n_splits=n_splits, block_size=block_size,
                                       random_state=random_state)[0]
    return null_aucs


def permutation_null_loyo_auc(
    X: np.ndarray,
    y: np.ndarray,
    years: np.ndarray,
    *,
    n_permutations: int = DEFAULT_N_PERMUTATIONS_ROBUSTNESS,
    random_state: int = 42,
) -> np.ndarray:
    """Label-shuffled null distribution of leave_one_year_out_auc's own mean AUC. Expensive:
    each of the n_permutations reps reruns the FULL leave-one-year-out loop (one model fit per
    usable calendar year), so this is roughly n_permutations x as many model fits as a single
    leave_one_year_out_auc call -- see the DEFAULT_N_PERMUTATIONS_ROBUSTNESS module comment for
    how to dial this down if runtime becomes a problem."""
    rng = np.random.RandomState(random_state)
    null_aucs = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        y_shuffled = rng.permutation(y)
        null_aucs[i] = leave_one_year_out_auc(X, y_shuffled, years, random_state=random_state)[0]
    return null_aucs


def _perm_p_value(observed: float, null_aucs: np.ndarray) -> tuple[float, int]:
    """Shared permutation p-value recipe (matches residual_extreme_signal_diag_LOGISTIC.py's
    inline version exactly): fraction of null-distribution AUCs at or above the observed AUC,
    with the +1/+1 continuity correction so p is never reported as exactly 0. Returns
    (p_value, n_valid_permutations)."""
    valid = null_aucs[~np.isnan(null_aucs)]
    if not len(valid) or np.isnan(observed):
        return float("nan"), 0
    p_value = float((1 + np.sum(valid >= observed)) / (1 + len(valid)))
    return p_value, len(valid)


def c_sweep_auc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_splits: int = DEFAULT_N_SPLITS_BLOCKED,
    block_size: int = DEFAULT_BLOCK_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """blocked_cv_auc at each C in c_grid (inverse regularization strength), all on the SAME
    block/fold assignment (one rng draw, reused across C) so the sweep isolates the C effect
    rather than mixing it with fold-assignment noise. Shows the reported skill survives away
    from sklearn's LogisticRegression default (C=1.0, used everywhere else in this pipeline)."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    fold_of_day = _block_fold_assignment(n, n_splits, block_size, rng)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rows = []
    for c in c_grid:
        aucs = []
        for f in range(n_splits):
            va = fold_of_day == f
            tr = ~va
            y_tr, y_va = y[tr], y[va]
            if y_va.sum() < 1 or y_va.sum() == len(y_va) or y_tr.sum() < 2:
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state,
                                       solver="lbfgs", C=c)
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_va)[:, 1]
            try:
                aucs.append(roc_auc_score(y_va, proba))
            except ValueError:
                pass
        rows.append({
            "C": c,
            "blocked_cv_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "n_folds_valid": len(aucs),
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class RegimeRobustnessResult:
    z_regime_id: int
    n_days: int
    n_extreme: int
    n_years: int
    cv_auc_randomfold: float  # feature_relevance._cv_auc -- same number reported by
    # residual_extreme_signal_diag_LOGISTIC.py, kept here purely as the reference point
    # the blocked/LOYO numbers below are compared against.
    cv_auc_blocked: float
    cv_auc_blocked_std: float
    cv_auc_blocked_median: float
    cv_auc_blocked_perm_p_value: float
    cv_auc_blocked_perm_n_valid: int
    loyo_auc: float
    loyo_auc_std: float
    loyo_auc_median: float
    loyo_auc_perm_p_value: float
    loyo_auc_perm_n_valid: int
    loyo_n_years_used: int
    auc_gap_blocked_vs_random: float
    auc_gap_loyo_vs_random: float
    coef_table: pd.DataFrame
    n_bootstrap_valid: int
    frac_features_robust_sign: float
    frac_features_unstable_sign: float
    c_sweep: pd.DataFrame
    verdict: str


def analyze_regime_robustness(
    *,
    z_regime_id: int,
    z500_regimes_path: Path,
    pca_paths_by_field: dict[str, Path],
    paths: ResidualWorkflowPaths,
    extreme_labels_path: Path,
    extreme_var: str,
    n_splits: int = 5,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_permutations: int = DEFAULT_N_PERMUTATIONS_ROBUSTNESS,
    random_state: int = 42,
    overwrite: bool = False,
    residualize: bool = False,
) -> RegimeRobustnessResult:
    joint_features_alldays_path = paths.per_region_joint_features_alldays_path(z_regime_id)
    joint_features_subset_path = paths.per_region_joint_features_path(z_regime_id)
    ward_hc_path = paths.per_region_ward_hc_path(z_regime_id)

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
        if "feature_name" in ds.coords:
            feature_names_all = [str(v) for v in ds["feature_name"].values]
        else:
            feature_names_all = [f"feature_{i}" for i in range(X_all.shape[1])]

    label_times, y_all = _load_extreme_labels(extreme_labels_path, extreme_var)
    label_times = pd.DatetimeIndex(label_times).normalize()

    common = pd.Index(feat_times).intersection(pd.Index(label_times))
    idx_X = pd.Index(feat_times).get_indexer(common)
    idx_y = pd.Index(label_times).get_indexer(common)
    X = X_all[idx_X]
    y = np.asarray(y_all[idx_y], dtype=np.int64)
    day_index = pd.DatetimeIndex(common)
    years = day_index.year.values

    n_days = len(y)
    n_extreme = int(y.sum())
    n_years = int(len(np.unique(years)))

    if n_extreme < 2 or n_extreme > n_days - 2:
        empty_coef = pd.DataFrame(columns=["feature", "full_sample_coef", "coef_ci_lo",
                                             "coef_ci_hi", "sign_consistency", "verdict"])
        empty_c_sweep = pd.DataFrame(columns=["C", "blocked_cv_auc", "n_folds_valid"])
        return RegimeRobustnessResult(
            z_regime_id=z_regime_id, n_days=n_days, n_extreme=n_extreme, n_years=n_years,
            cv_auc_randomfold=float("nan"), cv_auc_blocked=float("nan"), cv_auc_blocked_std=float("nan"),
            cv_auc_blocked_median=float("nan"), cv_auc_blocked_perm_p_value=float("nan"),
            cv_auc_blocked_perm_n_valid=0,
            loyo_auc=float("nan"), loyo_auc_std=float("nan"),
            loyo_auc_median=float("nan"), loyo_auc_perm_p_value=float("nan"), loyo_auc_perm_n_valid=0,
            loyo_n_years_used=0, auc_gap_blocked_vs_random=float("nan"),
            auc_gap_loyo_vs_random=float("nan"), coef_table=empty_coef, n_bootstrap_valid=0,
            frac_features_robust_sign=float("nan"), frac_features_unstable_sign=float("nan"),
            c_sweep=empty_c_sweep,
            verdict=f"DEGENERATE (n_extreme={n_extreme} of {n_days} days -- too few/many for CV/bootstrap)",
        )

    cv_auc_randomfold = _cv_auc(X, y, n_splits=n_splits)
    cv_auc_blocked, cv_auc_blocked_std, cv_auc_blocked_median, _n_folds_valid = blocked_cv_auc(
        X, y, n_splits=n_splits, block_size=block_size, random_state=random_state
    )
    loyo_auc, loyo_auc_std, loyo_auc_median, loyo_n_years_used = leave_one_year_out_auc(
        X, y, years, random_state=random_state
    )

    cv_auc_blocked_null = permutation_null_blocked_cv_auc(
        X, y, n_splits=n_splits, block_size=block_size, n_permutations=n_permutations, random_state=random_state
    )
    cv_auc_blocked_perm_p_value, cv_auc_blocked_perm_n_valid = _perm_p_value(cv_auc_blocked, cv_auc_blocked_null)
    loyo_auc_null = permutation_null_loyo_auc(
        X, y, years, n_permutations=n_permutations, random_state=random_state
    )
    loyo_auc_perm_p_value, loyo_auc_perm_n_valid = _perm_p_value(loyo_auc, loyo_auc_null)

    gap_blocked = cv_auc_blocked - cv_auc_randomfold if not np.isnan(cv_auc_blocked) else float("nan")
    gap_loyo = loyo_auc - cv_auc_randomfold if not np.isnan(loyo_auc) else float("nan")

    coef_table, boot_aucs, n_valid = coefficient_stability(
        X, y, feature_names_all, n_bootstrap=n_bootstrap, block_size=block_size, random_state=random_state,
    )
    frac_robust = float((coef_table["verdict"] == "robust").mean()) if len(coef_table) else float("nan")
    frac_unstable = float((coef_table["verdict"] == "unstable").mean()) if len(coef_table) else float("nan")

    c_sweep = c_sweep_auc(X, y, c_grid=c_grid, n_splits=n_splits, block_size=block_size, random_state=random_state)

    verdict_bits = []
    if not np.isnan(gap_blocked) and abs(gap_blocked) > 0.05:
        verdict_bits.append(f"BLOCKED-CV GAP ({gap_blocked:+.3f})")
    if not np.isnan(gap_loyo) and abs(gap_loyo) > 0.05:
        verdict_bits.append(f"LOYO GAP ({gap_loyo:+.3f})")
    if not np.isnan(frac_robust) and frac_robust < 0.5:
        verdict_bits.append(f"LOW SIGN-STABILITY ({frac_robust:.0%} robust)")
    verdict = "ROBUST" if not verdict_bits else "CAVEATS: " + "; ".join(verdict_bits)

    return RegimeRobustnessResult(
        z_regime_id=z_regime_id, n_days=n_days, n_extreme=n_extreme, n_years=n_years,
        cv_auc_randomfold=cv_auc_randomfold, cv_auc_blocked=cv_auc_blocked,
        cv_auc_blocked_std=cv_auc_blocked_std, cv_auc_blocked_median=cv_auc_blocked_median,
        cv_auc_blocked_perm_p_value=cv_auc_blocked_perm_p_value,
        cv_auc_blocked_perm_n_valid=cv_auc_blocked_perm_n_valid,
        loyo_auc=loyo_auc, loyo_auc_std=loyo_auc_std, loyo_auc_median=loyo_auc_median,
        loyo_auc_perm_p_value=loyo_auc_perm_p_value, loyo_auc_perm_n_valid=loyo_auc_perm_n_valid,
        loyo_n_years_used=loyo_n_years_used, auc_gap_blocked_vs_random=gap_blocked,
        auc_gap_loyo_vs_random=gap_loyo, coef_table=coef_table, n_bootstrap_valid=n_valid,
        frac_features_robust_sign=frac_robust, frac_features_unstable_sign=frac_unstable,
        c_sweep=c_sweep, verdict=verdict,
    )


def plot_robustness_diagnostic(out_path: Path, results: dict[int, RegimeRobustnessResult]) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(results.items())
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.5), squeeze=False)
    for col, (z, res) in enumerate(items):
        ax_top = axes[0][col]
        aucs = {"random-fold": res.cv_auc_randomfold, "blocked-fold": res.cv_auc_blocked,
                "leave-1-year-out": res.loyo_auc}
        labels = list(aucs.keys())
        vals = [aucs[k] for k in labels]
        ax_top.bar(labels, vals, color=["tab:blue", "tab:orange", "tab:green"])
        ax_top.set_ylim(0.4, 1.0)
        ax_top.set_ylabel("AUC")
        ax_top.tick_params(axis="x", rotation=20)
        ax_top.set_title(f"Z500-{z} (n={res.n_days}d, {res.n_years}yr)\n{res.verdict}", fontsize=8)

        ax_bot = axes[1][col]
        ct = res.coef_table
        if len(ct):
            ax_bot.barh(ct["feature"], ct["sign_consistency"], color="tab:purple")
            ax_bot.axvline(SIGN_CONSISTENCY_ROBUST, color="black", linestyle=":", linewidth=1)
            ax_bot.axvline(SIGN_CONSISTENCY_CONTEXT, color="gray", linestyle=":", linewidth=1)
        ax_bot.set_xlim(0, 1)
        ax_bot.set_xlabel("coefficient sign-consistency\n(block bootstrap)")
        ax_bot.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[rr021_stage2_LogRegr_Robustness] Wrote {out_path}")
    return out_path


def run_robustness_diagnostic(
    *,
    config,
    z500_regimes_path: Path,
    pca_paths_by_field: dict[str, Path],
    extreme_labels_path: Path,
    extreme_var: str,
    regimes: list[int] | None = None,
    n_splits: int = 5,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_permutations: int = DEFAULT_N_PERMUTATIONS_ROBUSTNESS,
    output_tag: str = "",
) -> Path:
    paths = ResidualWorkflowPaths(config)
    all_z = sorted(config.per_regime_field_map.keys())
    target_z = sorted(regimes) if regimes else all_z
    unknown = set(target_z) - set(all_z)
    if unknown:
        raise ValueError(f"--regimes contains ids not in the field map: {sorted(unknown)} (have {all_z})")

    sig = per_regime_fields_signature(config)
    eid = experiment_id(config)
    tag_suffix = f"_{output_tag}" if output_tag else ""

    results: dict[int, RegimeRobustnessResult] = {}
    summary_rows = []
    for z in target_z:
        fields_for_z = {f: pca_paths_by_field[f] for f in config.per_regime_field_map[z] if f != "z500"}
        print(f"[rr021_stage2_LogRegr_Robustness] Z500-{z}: fields={'+'.join(sorted(fields_for_z))} -- "
              f"reusing/building cached joint features...")
        res = analyze_regime_robustness(
            z_regime_id=z, z500_regimes_path=z500_regimes_path, pca_paths_by_field=fields_for_z,
            paths=paths, extreme_labels_path=extreme_labels_path, extreme_var=extreme_var,
            n_splits=n_splits, n_bootstrap=n_bootstrap, block_size=block_size, c_grid=c_grid,
            n_permutations=n_permutations,
            random_state=config.random_state, overwrite=config.overwrite, residualize=config.residualize,
        )
        results[z] = res
        print(f"[rr021_stage2_LogRegr_Robustness] Z500-{z}: {res.verdict} "
              f"(cv_auc random={res.cv_auc_randomfold:.3f} "
              f"blocked={res.cv_auc_blocked:.3f}+/-{res.cv_auc_blocked_std:.3f} (p={res.cv_auc_blocked_perm_p_value:.3f}) "
              f"loyo={res.loyo_auc:.3f}+/-{res.loyo_auc_std:.3f} (p={res.loyo_auc_perm_p_value:.3f}) "
              f"[{res.loyo_n_years_used} yrs])")

        coef_csv = paths.diagnostics_root / (
            f"rr021_stage2_LogRegr_Robustness_Z{z}_coefstability_{sig}_{eid}{tag_suffix}.csv"
        )
        coef_csv.parent.mkdir(parents=True, exist_ok=True)
        res.coef_table.to_csv(coef_csv, index=False)
        print(f"[rr021_stage2_LogRegr_Robustness] Wrote {coef_csv}")

        c_sweep_csv = paths.diagnostics_root / (
            f"rr021_stage2_LogRegr_Robustness_Z{z}_Csweep_{sig}_{eid}{tag_suffix}.csv"
        )
        res.c_sweep.to_csv(c_sweep_csv, index=False)
        print(f"[rr021_stage2_LogRegr_Robustness] Wrote {c_sweep_csv}")

        summary_rows.append({
            "z500_regime": z,
            "residualize": bool(config.residualize),
            "fields": "+".join(sorted(fields_for_z)),
            "n_days": res.n_days,
            "n_extreme": res.n_extreme,
            "n_years": res.n_years,
            "cv_auc_randomfold": round(res.cv_auc_randomfold, 4) if not np.isnan(res.cv_auc_randomfold) else float("nan"),
            "cv_auc_blocked": round(res.cv_auc_blocked, 4) if not np.isnan(res.cv_auc_blocked) else float("nan"),
            "cv_auc_blocked_std": round(res.cv_auc_blocked_std, 4) if not np.isnan(res.cv_auc_blocked_std) else float("nan"),
            "cv_auc_blocked_median": round(res.cv_auc_blocked_median, 4) if not np.isnan(res.cv_auc_blocked_median) else float("nan"),
            "cv_auc_blocked_perm_p_value": round(res.cv_auc_blocked_perm_p_value, 4) if not np.isnan(res.cv_auc_blocked_perm_p_value) else float("nan"),
            "cv_auc_blocked_perm_n_valid": res.cv_auc_blocked_perm_n_valid,
            "auc_gap_blocked_vs_random": round(res.auc_gap_blocked_vs_random, 4) if not np.isnan(res.auc_gap_blocked_vs_random) else float("nan"),
            "loyo_auc": round(res.loyo_auc, 4) if not np.isnan(res.loyo_auc) else float("nan"),
            "loyo_auc_std": round(res.loyo_auc_std, 4) if not np.isnan(res.loyo_auc_std) else float("nan"),
            "loyo_auc_median": round(res.loyo_auc_median, 4) if not np.isnan(res.loyo_auc_median) else float("nan"),
            "loyo_auc_perm_p_value": round(res.loyo_auc_perm_p_value, 4) if not np.isnan(res.loyo_auc_perm_p_value) else float("nan"),
            "loyo_auc_perm_n_valid": res.loyo_auc_perm_n_valid,
            "loyo_n_years_used": res.loyo_n_years_used,
            "auc_gap_loyo_vs_random": round(res.auc_gap_loyo_vs_random, 4) if not np.isnan(res.auc_gap_loyo_vs_random) else float("nan"),
            "n_bootstrap_valid": res.n_bootstrap_valid,
            "frac_features_robust_sign": round(res.frac_features_robust_sign, 3) if not np.isnan(res.frac_features_robust_sign) else float("nan"),
            "frac_features_unstable_sign": round(res.frac_features_unstable_sign, 3) if not np.isnan(res.frac_features_unstable_sign) else float("nan"),
            "verdict": res.verdict,
        })

    summary_df = pd.DataFrame(summary_rows)

    logistic_diag_summary_path = paths.diagnostics_root / f"rr_LOGISTIC_diag_summary_{sig}_{eid}{tag_suffix}.csv"
    if logistic_diag_summary_path.is_file():
        try:
            ldf = pd.read_csv(logistic_diag_summary_path)[["z500_regime", "cv_auc", "perm_p_value", "verdict"]]
            ldf = ldf.rename(columns={"cv_auc": "logistic_diag_cv_auc", "perm_p_value": "logistic_diag_perm_p_value",
                                        "verdict": "logistic_diag_verdict"})
            summary_df = summary_df.merge(ldf, on="z500_regime", how="left")
        except Exception as exc:  # noqa: BLE001 - best-effort context, never fatal
            print(f"[rr021_stage2_LogRegr_Robustness] Note: could not merge LOGISTIC-diag context ({exc})")
    else:
        print(f"[rr021_stage2_LogRegr_Robustness] Note: no LOGISTIC signal-diagnostic summary found at "
              f"{logistic_diag_summary_path} (run residual_extreme_signal_diag_LOGISTIC.py first for "
              "side-by-side cv_auc/permutation-p context) -- robustness-only output below.")

    summary_csv = paths.diagnostics_root / f"rr021_stage2_LogRegr_Robustness_summary_{sig}_{eid}{tag_suffix}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[rr021_stage2_LogRegr_Robustness] Wrote {summary_csv}")
    print("[rr021_stage2_LogRegr_Robustness] === SUMMARY ===")
    print(summary_df.to_string(index=False))

    fig_path = paths.figures_root / f"rr021_stage2_LogRegr_Robustness_{sig}_{eid}{tag_suffix}.png"
    plot_robustness_diagnostic(fig_path, results)

    return summary_csv


def main() -> int:
    from .auxiliary import _find_legacy_z500_regimes
    from .config import build_arg_parser, config_from_args
    from common.extremes_spec import ExtremesSpec, add_extremes_spec_arguments, resolve_convective_duration_path

    parser = build_arg_parser()
    add_extremes_spec_arguments(parser)
    parser.set_defaults(
        extremes_q=0.999, quantile_sample="all_hours", extremes_wet_threshold_mm=0.1,
        extremes_cape_threshold=0.0, extremes_cp_frac_min=0.01, extremes_year_span="1979-2025",
    )
    parser.add_argument("--extreme-labels-path", type=Path, default=None,
                         help="LEGACY: explicit convective_duration NetCDF, bypassing the "
                              "extremes-spec tag resolution (matches rr_01/the LOGISTIC diagnostic's "
                              "--extreme-labels-path).")
    parser.add_argument("--extreme-var", default="flagged_any_gridcell",
                         help="Variable name for the binary extreme label (default matches rr_01).")
    parser.add_argument("--study-area", default=None,
                         help="Processed-data folder (germany, ontario_south). Auto-derived from --region if omitted.")
    parser.add_argument("--regimes", default=None,
                         help="Comma-separated Z500 regime ids to restrict to (default: all regimes "
                              "in --regime-fields-map).")
    parser.add_argument("--n-splits", type=int, default=5,
                         help="Fold count for both the random-fold reference AUC (_cv_auc) and blocked_cv_auc/c_sweep_auc.")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP,
                         help="Block-bootstrap resamples for coefficient sign/CI stability.")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                         help="Contiguous-day block length for the bootstrap and blocked-CV fold assignment "
                              "(default 10, matching feature_relevance._bootstrap_greedy_stability).")
    parser.add_argument("--c-grid", default=",".join(str(c) for c in DEFAULT_C_GRID),
                         help="Comma-separated LogisticRegression C values for the regularization sweep.")
    parser.add_argument("--n-permutations-robustness", type=int, default=DEFAULT_N_PERMUTATIONS_ROBUSTNESS,
                         help="Label-shuffle repetitions for the dedicated permutation-null significance "
                              "test of cv_auc_blocked and loyo_auc (2026-08-12 addition -- separate from, "
                              "and more precise than, residual_extreme_signal_diag_LOGISTIC.py's "
                              "permutation test which only covers cv_auc_randomfold). Costly for LOYO in "
                              "particular: each rep reruns the full leave-one-year-out loop. Lower this if "
                              "a full region/bbox sweep's runtime becomes a problem.")
    parser.add_argument("--output-tag", default="",
                         help="Appended to output filenames so runs that differ only by "
                              "--extreme-labels-path/--extreme-var (e.g. a domain_pctl_q comparison "
                              "sweep) don't overwrite each other's output. Also used to look up the "
                              "matching residual_extreme_signal_diag_LOGISTIC.py summary CSV for "
                              "side-by-side cv_auc/permutation-p context.")
    args = parser.parse_args()

    if not args.per_regime_mode:
        print("ERROR: this diagnostic requires --per-regime-mode (with --regime-fields-map) -- "
              "it reuses rr_021's per-Z500-regime joint-features files.", file=sys.stderr)
        return 1

    config = config_from_args(args)

    regimes = None
    if args.regimes:
        regimes = [int(z.strip()) for z in args.regimes.split(",") if z.strip()]
    c_grid = tuple(float(c.strip()) for c in args.c_grid.split(",") if c.strip())

    extremes_spec = ExtremesSpec(
        q=float(args.extremes_q), quantile_sample=str(args.quantile_sample),
        wet_threshold_mm=float(args.extremes_wet_threshold_mm),
        cape_threshold=float(args.extremes_cape_threshold),
        cp_frac_min=float(args.extremes_cp_frac_min),
        year_span=str(args.extremes_year_span).strip(),
    )
    try:
        extreme_labels_path = resolve_convective_duration_path(
            config.module_root, config.region, study_area=args.study_area,
            spec=extremes_spec, legacy_path=args.extreme_labels_path,
        )
    except FileNotFoundError as exc:
        print(f"[rr021_stage2_LogRegr_Robustness] ERROR: {exc}", file=sys.stderr)
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

    run_robustness_diagnostic(
        config=config, z500_regimes_path=z500_regimes_path, pca_paths_by_field=pca_paths_by_field,
        extreme_labels_path=extreme_labels_path, extreme_var=args.extreme_var, regimes=regimes,
        n_splits=args.n_splits, n_bootstrap=args.n_bootstrap, block_size=args.block_size,
        c_grid=c_grid, n_permutations=args.n_permutations_robustness, output_tag=args.output_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
