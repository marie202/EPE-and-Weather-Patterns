#!/usr/bin/env python3
"""
Feature relevance diagnostic for the residual regime pipeline.

For each candidate ERA5 field, scores its FULL multi-mode PCA block —
``resolve_mf_pca_path``'s ``PCs_reduced``, the SAME per-field representation
``stage2_logistic.py`` actually fits on in production — then trains logistic
regression models against the binary extreme event label using 5-fold
cross-validated ROC AUC. Three complementary evaluation approaches:

Representation (2026-08-14): every candidate field except Z500 is scored on its
full PCA block (truncated to its leading ``--max-modes-per-field`` modes,
default 100 — matching ``MultivariateRunConfig.max_modes_per_field``, the same
cap rr_00 itself applies when building these PCA files, so this stays
apples-to-apples with production rather than an independently chosen number).
This REPLACES the previous single-domain-mean-column-per-field scoring:
domain-mean throws away everything but the spatial average, so a field with a
spatially heterogeneous but predictive pattern looked weak here while still
being exactly what the production model actually uses. See git history and
README_ResidualRegimes.md's "PCA mode-count disparity" note for the
investigation (formerly a separate standalone diagnostic,
``rr_023_greedy_pca_relevance.py``, now merged in here — there is deliberately
only one feature-selection script). Z500 itself has no per-field PCA block (it
lives outside the rr_00 multivariate step and is the regime-conditioning
variable, not a candidate feature) and keeps its previous domain-mean scoring —
this only affects the OTHER candidate fields.

Per-fold PCA refit (2026-08-18): every non-Z500 field's PCA block used to be
``resolve_mf_pca_path``'s ``PCs_reduced`` — a single SVD fit ONCE on the full
1979-2025 record, before any CV split existed. Every fold's "training" PCA
components were therefore computed using rows that also sat in that fold's
own validation set: a leak that let CV AUC climb even under blocked CV
(day-adjacency-safe fold assignment) and MAX_MODES_PER_FIELD capping (dimension-
imbalance control) alone, confirmed via HPC runs where B4 regimes cleared
0.80-0.90 with both mitigations active — a symptom pattern originally only
expected at B5. Fields now load their pre-PCA raw matrix instead (via
``resolve_mf_preprocessed_path``'s ``X_matrix``, the same matrix
``run_per_field_pca`` fits its SVD on), and PCA is refit separately inside each
CV fold using ONLY that fold's training rows (``PerFoldFeatureCache``). Because
PCA is unsupervised (depends only on which rows are train vs. val for a given
fold, never on which other fields are combined with it or which label is being
predicted), each field's per-fold blocks are computed ONCE per bbox/regime
subset and reused across every greedy step, leave-one-out drop, random-field
draw, and permutation-null shuffle for that subset — the fit-per-candidate-
combination cost this would otherwise imply never actually happens.

Per-fold PCA refit result (2026-08-19): re-running the B4/europe regime job that
originally showed 0.85-0.90 AUCs, with the fix above AND with CV_BLOCK_SIZE raised
10->30 (testing whether block length vs. field autocorrelation timescale explained the
rest), moved every regime's AUC by at most ~0.02-0.03 -- noise-level, no systematic
drop either way. Both leakage hypotheses are therefore ruled out for this symptom;
the elevated AUC is more likely genuine signal (near-surface/thermodynamic fields ARE
the proximate drivers of convective precip, more so than Z500's large-scale pattern
alone) than a CV artifact. ``--field-representation domain_mean`` (below) exists to
confirm this directly: same blocked CV, same per-fold-refit machinery (moot for
domain-mean fields, which were never PCA'd), only the representation itself changes.

Field representation (``--field-representation``, 2026-08-19; default flipped to
``domain_mean`` 2026-09-02): controlled A/B between ``domain_mean`` (default/production)
and ``pca_block`` (the full multi-mode PCA block described above). ``domain_mean`` scores
every non-Z500 field on the SAME single area-weighted spatial-average column Z500 always
used pre-2026-08-14 (``_domain_mean_field``, reusing ``_domain_mean_anomaly`` on each
field's own ``03_MF_..._anomaly`` file rather than ``X_matrix``, whose values are
already area-weighted for PCA and would double up under a second plain mean). Both
representations are now ALWAYS explicitly tagged in every output filename
(``rr01_paths.rr01_tag``'s ``_domainmean``/``_pcablock`` suffix) — previously neither was,
so a ``pca_block`` run and a ``domain_mean`` run for the same region/bbox/mode silently
overwrote the exact same CSV, and every downstream consumer (``write_field_selection.py``,
rr_02/rr_022's ``--build-regime-map``) picked up whichever one happened to run last with
no way to tell which representation it actually reflected.
Single-column fields are exempt from PCA fitting entirely in ``PerFoldFeatureCache``
(fitting PCA with n_components=1 is a no-op), so this isolates representation as the
only variable — everything else (blocked CV, mode capping, per-fold refit machinery)
is unchanged.

Field streaming (2026-08-19): the per-fold PCA refit above requires each field's
PRE-PCA raw grid (``X_matrix``, up to tens of thousands of columns at large bboxes —
vs. the old ``PCs_reduced``'s <=100 columns). Loading and holding all ~27 candidate
fields' full raw grids simultaneously (the naive approach: build one ``field_raw``
dict up front, reuse it across bbox-mode and every Z500 regime) OOM-killed HPC jobs at
B0-B3 for both regions (exit=137, ``oom_kill``) — at north-america B2 alone (50935 grid
points), 27 fields x ~7200 days x float64 is on the order of 80GB; B0/B1 are larger
still. Fixed by streaming: ``run_feature_relevance`` now (1) peeks every field's time
coordinate ONLY (cheap — reading a DataArray's dims/coords doesn't trigger xarray's
lazy backend to materialize the bulk variable) to compute the common date range and,
for MODE=regime, every valid regime's row mask/fold count, all BEFORE any field's full
data is touched; (2) streams each field's full raw grid through ONE AT A TIME,
immediately reducing it to its small per-scope per-fold PCA blocks
(``PerFoldFeatureCache.from_prebuilt``) and discarding the full array before the next
field loads. Peak memory is bounded by ONE field's full grid, not all of them at once.
``X_matrix`` is also loaded as float32 (matching its on-disk dtype and
``run_per_field_pca``'s own SVD input — the previous float64 upcast doubled memory for
no precision benefit). Two callers still need genuine raw per-field access at a scope
smaller than "stream once, keep only blocks" — ``_bootstrap_greedy_stability`` (block-
bootstrap RESAMPLES rows, so it must refit PCA on resampled rows, which no precomputed
per-fold block covers) and ``_in_sample_trace``'s no-CV ceiling (fits on ALL of a
scope's rows at once, not per-fold) — both opt-in and off by default, so
``run_feature_relevance``/``_run_ab_subset`` rebuild a scope-local ``field_raw`` dict
lazily via ``_stream_field_raw_for_rows``, ONLY when explicitly requested, trading
extra I/O for correctness rather than holding everything in memory just in case.

Yeo-Johnson: resolves the pipeline's _yj_ files by default, matching the
pipeline-wide YEO_JOHNSON default (Steps 02-07 and PCA assume roughly Gaussian
input, so every field including Z500 is transformed consistently before
detrending/anomaly/PCA). rr_01 ranks fields on that same representation rather
than a differently-preprocessed one — pass --no-yeo-johnson to resolve the raw
(non-yj) files instead, e.g. for an explicit raw-vs-yj comparison.

OPTION A: Independent Residual-Based Importance (Diagnostic)
    - Trains Z500-only model to predict extreme events
    - Tests each field independently: can it predict Z500's prediction errors?
    - Reveals which fields individually capture variability Z500 misses
    - Identifies candidate fields (but may have redundancy)

OPTION B: Sequential Field Selection (PRIMARY - For rr_02)
    - Greedily adds fields based on marginal contribution to AUC
    - Z500 → Z500+field1 → Z500+field1+field2 → ...
    - Shows actual non-redundant contribution of each field
    - Defends field selection by demonstrating each adds new information
    - RECOMMENDED for residual clustering field selection

OPTION C: Global Importance (Diagnostic, bbox mode only)
    - Trains on all fields together (baseline AUC)
    - Tests leave-one-out: how much does removing each field hurt prediction?
    - Reveals which fields are globally predictive of extreme events
    - Secondary diagnostic for understanding overall field importance
    - Skipped in regime mode: identical computation to the bbox-mode run for
      the same bbox, so it would just duplicate that output under a different tag

Run as:
    python -m stage2_residual.feature_relevance --help

Outputs land under --out-dir/<mode>/ (mode = bbox | regime), keeping the two
modes in physically separate directories so nothing from one leaks into the
other. Within a mode directory, tagged by <region>_<bbox>_<mode>:
    bbox/feature_relevance_residual_<region>_<bbox>_bbox_optionA.csv
    bbox/feature_relevance_sequential_<region>_<bbox>_bbox_optionB.csv
    bbox/feature_relevance_hierarchical_<region>_<bbox>_bbox.csv
    bbox/feature_relevance_hierarchical_<region>_<bbox>_bbox_optionC.png
    regime/per_regime/feature_relevance_sequential_<region>_<bbox>_regime_R<N>_optionB.csv
    regime/per_regime/feature_relevance_residual_<region>_<bbox>_regime_R<N>_optionA.csv
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from stage2_residual.rr01_paths import rr01_tag
from stage2_residual.paths import resolve_mf_anomaly_path, resolve_mf_preprocessed_path
from common.preprocessing_helpers import filter_paths_by_yeo_johnson

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_MAX_MODES_PER_FIELD = 100  # matches MultivariateRunConfig.max_modes_per_field —
# the same cap rr_00 itself applies when building these PCA files, so scoring here stays
# apples-to-apples with production rather than an independently chosen number.

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _filter_by_area_weight_tag(paths: list[Path], area_weight_tag: str) -> list[Path]:
    """Pin ``MultivariateRunConfig.area_weight_tag`` exactly, the same way
    ``stage2_residual.paths.resolve_mf_pca_path`` does for Stage 2's per-field PCA
    lookup — rather than leaving it to glob + alphabetical-last tie-breaking, which
    only happened to pick a tagged rerun (e.g. ``_aw2.nc``) over the untagged file
    because it sorts after ``.nc`` by accident of string ordering. With
    ``area_weight_tag=""`` this rejects any tagged file just as much as it accepts
    untagged ones, so an unset tag can't silently match either.

    Not used by this module's own PCA-block scoring anymore (that goes through
    ``resolve_mf_pca_path`` directly) -- kept here only because ``plot_results.py``
    (and the RR feature-selection notebook) still import it for their own anomaly-file
    globbing.
    """
    if area_weight_tag:
        suffix = f"_{area_weight_tag}.nc"
        return [p for p in paths if p.name.endswith(suffix)]
    return [p for p in paths if re.search(r"_k\d+\.nc$", p.name)]


def _find_z500_anomaly_file(
    module_root: Path,
    region: str,
    bbox_step: str | None = None,
    smooth_window: int = 11,
    use_yeo_johnson: bool | None = None,
) -> Path | None:
    """Glob for Z500's domain-mean anomaly NetCDF for a given bbox step.

    Z500 lives in ``data/preprocessed/``, not ``data/multivariate/preprocessed/`` — it
    sits outside the rr_00 multivariate step entirely and has no per-field PCA block
    (see module docstring), so it keeps its original domain-mean-column scoring while
    every other candidate field now scores on its full PCA block, refit per CV fold via
    ``PerFoldFeatureCache`` (see module docstring's "Per-fold PCA refit" entry).

    ``use_yeo_johnson`` explicitly picks between ``_yj_`` and plain files rather than
    leaving it to alphabetical sort order. Defaults to ``None``, which defers to the
    pipeline-wide ``YEO_JOHNSON`` env var (default enabled) via
    ``preprocessing_helpers.filter_paths_by_yeo_johnson``.
    """
    w_tag = f"_w{int(smooth_window)}_"
    search_dir = module_root / "data" / "preprocessed" / region
    if not search_dir.is_dir():
        return None
    pattern = (
        f"03_Z500_anomaly_*step{bbox_step}_*.nc"
        if bbox_step
        else "03_Z500_anomaly_*.nc"
    )
    pool = sorted(search_dir.glob(pattern))
    if not pool:
        return None
    pool = filter_paths_by_yeo_johnson(pool, enabled=use_yeo_johnson)
    preferred = [m for m in pool if w_tag in m.name]
    chosen_pool = preferred if preferred else pool
    if len(chosen_pool) > 1:
        print(
            f"  [feature_relevance] WARNING: multiple Z500 files for step{bbox_step}, "
            f"using: {chosen_pool[-1].name}"
        )
    return chosen_pool[-1]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _area_weight(lat: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(lat)).clip(min=0)


def _get_time_dim(da: xr.DataArray) -> tuple[str, pd.DatetimeIndex]:
    """Extract time dimension from a DataArray, handling both 'time' and 'valid_time'."""
    dims = da.dims
    time_dim = None
    if "time" in dims:
        time_dim = "time"
    elif "valid_time" in dims:
        time_dim = "valid_time"
    else:
        time_dim = dims[0]
    times = pd.DatetimeIndex(da[time_dim].values)
    return time_dim, times


def _domain_mean_anomaly(path: Path, field_key: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Load field_anom and return (time_index, domain_mean_1d)."""
    with xr.open_dataset(path, chunks=None) as ds:
        candidates = [v for v in ds.data_vars if "anom" in v.lower()]
        if not candidates:
            candidates = list(ds.data_vars)
        var = candidates[0]
        da = ds[var].load()

    # identify dims and get time
    dims = da.dims
    time_dim, times = _get_time_dim(da)

    # spatial mean — area-weighted if lat present
    spatial_dims = [d for d in dims[1:]]
    lat_dim = next((d for d in spatial_dims if "lat" in d.lower()), None)

    if lat_dim is not None:
        lat_vals = da[lat_dim].values
        w = xr.DataArray(_area_weight(lat_vals), dims=[lat_dim], coords={lat_dim: da[lat_dim]})
        mean_1d = da.weighted(w).mean(dim=spatial_dims).values.astype(np.float32)
    else:
        mean_1d = da.mean(dim=spatial_dims).values.astype(np.float32)

    return times, mean_1d


@dataclass(frozen=True)
class _MFPCAPathConfig:
    """Minimal stand-in for ``stage2_residual.config.ResidualRunConfig``, carrying only
    the attributes ``resolve_mf_pca_path``/``resolve_mf_preprocessed_path`` actually read.
    Avoids pulling in the full Stage-2 config parser (per-regime-map, residualize, z500-k,
    ...) just to resolve a PCA/preprocessed file path from rr_01's own
    region/bbox_step/smooth_window/area_weight_tag args.
    """
    module_root: Path
    region: str
    bbox_step: str
    smooth_window: int
    area_weight_tag: str
    sigma_mode: str
    yeo_johnson: bool


def _raw_field_matrix(
    module_root: Path,
    region: str,
    field_key: str,
    bbox_step: str,
    smooth_window: int = 11,
    use_yeo_johnson: bool = True,
    area_weight_tag: str = "",
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """Load a candidate field's pre-PCA raw matrix — ``resolve_mf_preprocessed_path``'s
    ``X_matrix``, the same (n_days, n_space) flattened field ``run_per_field_pca`` fits
    its record-wide SVD on to produce ``resolve_mf_pca_path``'s ``PCs_reduced``.

    Replaces the old ``_pca_block`` (loaded ``PCs_reduced`` directly, already PCA'd on
    the FULL record) as of 2026-08-18 — see module docstring's "Per-fold PCA refit"
    entry. Every candidate field now scores on PCA fit separately per CV fold
    (``PerFoldFeatureCache``), which needs the pre-PCA matrix, not the leaked
    already-projected one. Returns (time_index, raw_matrix), or None if no
    preprocessed file is found (run rr_00 for this field/region/bbox first).
    """
    path_config = _MFPCAPathConfig(
        module_root=module_root, region=region, bbox_step=bbox_step,
        smooth_window=smooth_window, area_weight_tag=area_weight_tag,
        sigma_mode="field", yeo_johnson=use_yeo_johnson,
    )
    path = resolve_mf_preprocessed_path(path_config, field_key)
    if path is None:
        return None
    with xr.open_dataset(path, chunks=None) as ds:
        if "X_matrix" not in ds:
            return None
        da = ds["X_matrix"].load()
    time_dim, times = _get_time_dim(da)
    # float32, not float64: X_matrix is already stored float32 on disk (matching
    # run_per_field_pca's own SVD input dtype) and these grids run into the tens of
    # thousands of columns at large bboxes — an unnecessary float64 upcast here doubles
    # peak memory for no precision benefit (see module docstring's "Field streaming"
    # entry). The small per-fold PCA-transformed OUTPUTS stay float64 (_pca_fold_blocks).
    matrix = np.asarray(da.values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return times, matrix


def _domain_mean_field(
    module_root: Path,
    region: str,
    field_key: str,
    bbox_step: str,
    smooth_window: int = 11,
    use_yeo_johnson: bool = True,
    area_weight_tag: str = "",
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """Non-Z500 counterpart to ``_domain_mean_anomaly`` for the
    ``--field-representation domain_mean`` A/B test (2026-08-19): resolves the field's
    still-gridded ``03_MF_..._anomaly`` file (via ``resolve_mf_anomaly_path``) and applies
    the SAME area-weighted spatial mean Z500 already uses — NOT a mean over ``X_matrix``,
    whose values are already area-weighted for PCA and would double up under a plain mean.
    Returns (time_index, single-column domain-mean), or None if no anomaly file is found.
    """
    path_config = _MFPCAPathConfig(
        module_root=module_root, region=region, bbox_step=bbox_step,
        smooth_window=smooth_window, area_weight_tag=area_weight_tag,
        sigma_mode="field", yeo_johnson=use_yeo_johnson,
    )
    path = resolve_mf_anomaly_path(path_config, field_key)
    if path is None:
        return None
    times, vals = _domain_mean_anomaly(path, field_key)
    return times, vals.reshape(-1, 1).astype(np.float64)


def _peek_field_dates(
    module_root: Path,
    region: str,
    field_key: str,
    bbox_step: str,
    smooth_window: int = 11,
    use_yeo_johnson: bool = True,
    area_weight_tag: str = "",
    field_representation: str = "domain_mean",
) -> pd.DatetimeIndex | None:
    """Cheap first pass: resolve the field's file and read ONLY its time coordinate, not
    the full (n_days, n_space) data array — accessing a DataArray's dims/coords doesn't
    trigger xarray's lazy backend to materialize the bulk variable, only ``.load()``/
    ``.values`` on the DATA variable itself does. Lets run_feature_relevance compute the
    common date intersection across all candidate fields BEFORE committing to loading
    any field's full raw grid into memory — see module docstring's "Field streaming"
    entry (2026-08-19), added after large-bbox runs (tens of thousands of grid points x
    27 fields held in memory at once) OOM-killed HPC jobs.
    """
    path_config = _MFPCAPathConfig(
        module_root=module_root, region=region, bbox_step=bbox_step,
        smooth_window=smooth_window, area_weight_tag=area_weight_tag,
        sigma_mode="field", yeo_johnson=use_yeo_johnson,
    )
    if field_representation == "domain_mean":
        path = resolve_mf_anomaly_path(path_config, field_key)
        if path is None:
            return None
        with xr.open_dataset(path, chunks=None) as ds:
            var = next((v for v in ds.data_vars if "anom" in v.lower()), None) or next(iter(ds.data_vars), None)
            if var is None:
                return None
            _, times = _get_time_dim(ds[var])
        return times

    path = resolve_mf_preprocessed_path(path_config, field_key)
    if path is None:
        return None
    with xr.open_dataset(path, chunks=None) as ds:
        if "X_matrix" not in ds:
            return None
        _, times = _get_time_dim(ds["X_matrix"])
    return times


def _stream_field_raw_for_rows(
    module_root: Path,
    region: str,
    fields: list[str],
    bbox_step: str,
    *,
    smooth_window: int,
    use_yeo_johnson: bool,
    area_weight_tag: str,
    field_representation: str,
    common: pd.DatetimeIndex,
    row_idx: np.ndarray,
    z500_aligned: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Rebuilds a raw field_raw dict (field -> row-subset raw matrix) for ONE scope by
    re-reading each field from disk, for the two callers that genuinely need raw
    per-field access at a scope smaller than "stream once, keep only per-fold PCA
    blocks": _bootstrap_greedy_stability (block-bootstrap RESAMPLES rows, so it must
    refit PCA on resampled rows — no precomputed per-fold block covers that) and
    _in_sample_trace's no-CV ceiling (deliberately fits on ALL of a scope's rows at
    once, not per-fold — see _build_global_fit_blocks's docstring).

    Both are opt-in and off by default (n_bootstrap=0, report_in_sample_auc=False in
    every run so far) — this trades extra I/O for correctness only when explicitly
    requested, rather than holding every field's full raw grid in memory for the whole
    function just in case one of these rarely-used options gets turned on. See module
    docstring's "Field streaming" entry (2026-08-19).
    """
    out: dict[str, np.ndarray] = {}
    if z500_aligned is not None:
        out["z500"] = z500_aligned[row_idx]
    for field_key in fields:
        if field_key.lower() == "z500":
            continue
        if field_representation == "domain_mean":
            result = _domain_mean_field(
                module_root, region, field_key, bbox_step, smooth_window=smooth_window,
                use_yeo_johnson=use_yeo_johnson, area_weight_tag=area_weight_tag,
            )
        else:
            result = _raw_field_matrix(
                module_root, region, field_key, bbox_step, smooth_window=smooth_window,
                use_yeo_johnson=use_yeo_johnson, area_weight_tag=area_weight_tag,
            )
        if result is None:
            continue
        times, matrix = result
        idx = pd.Index(pd.DatetimeIndex(times.normalize())).get_indexer(common)
        out[field_key] = matrix[idx][row_idx].astype(np.float64)
    return out


# ---------------------------------------------------------------------------
# Extreme label loading
# ---------------------------------------------------------------------------

def _robust_open_dataset(path: Path) -> xr.Dataset:
    """xr.open_dataset with a fallback for an xarray/cftime decode bug observed
    in this environment (xarray + numpy 2.4.x + cftime 1.6.5 on Python 3.14):
    a perfectly valid CF "hours since <ref>" time coordinate with
    calendar='proleptic_gregorian' raises a spurious "unable to decode time
    units" ValueError during dtype probing, even though decode_times=False
    loads the identical raw values fine -- confirmed by direct inspection
    (no NaN/sentinel/out-of-range values) on one of the files that triggers
    this. Falls back to a manual pandas-based decode, exact for
    proleptic_gregorian (the ordinary Gregorian calendar Python/pandas
    already use) over the 1979-2025 ERA5/observation record this pipeline
    covers -- not a hack that changes the resulting timestamps, just bypasses
    the buggy decode path.
    """
    try:
        return xr.open_dataset(path, chunks=None)
    except ValueError as exc:
        if "unable to decode time units" not in str(exc):
            raise
        ds = xr.open_dataset(path, chunks=None, decode_times=False)
        unit_aliases = {
            "hours": "h", "hour": "h", "days": "D", "day": "D",
            "minutes": "min", "minute": "min", "seconds": "s", "second": "s",
        }
        for name in list(ds.coords) + list(ds.data_vars):
            da = ds[name]
            units = da.attrs.get("units", "")
            if not isinstance(units, str):
                continue
            m = re.match(r"(\w+)\s+since\s+(.+)", units)
            if m is None or m.group(1).lower() not in unit_aliases:
                continue
            ref = pd.Timestamp(m.group(2).strip())
            decoded = (ref + pd.to_timedelta(np.asarray(da.values), unit=unit_aliases[m.group(1).lower()])).values
            if name in ds.coords:
                ds = ds.assign_coords({name: (da.dims, decoded)})
            else:
                ds[name] = (da.dims, decoded)
        return ds


def _load_extreme_labels(path: Path, var: str) -> tuple[pd.DatetimeIndex, np.ndarray]:
    with _robust_open_dataset(path) as ds:
        if var not in ds:
            available = list(ds.data_vars)
            raise KeyError(f"Variable {var!r} not in {path.name}. Available: {available}")
        da = ds[var].load()
    time_dim, times = _get_time_dim(da)
    labels = (da.values > 0).astype(np.int8)
    return times, labels


# ---------------------------------------------------------------------------
# Logistic regression + cross-validated AUC
# ---------------------------------------------------------------------------

def _make_fold_iter(
    y: np.ndarray,
    n_splits: int,
    cv_mode: str,
    block_size: int,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Builds the (train_idx, val_idx) pairs for either cv_mode, with degenerate folds
    (all-one-class validation, or too few training positives) already dropped. Extracted
    from _cv_auc so both it and PerFoldFeatureCache below use IDENTICAL fold membership
    for a given (y, cv_mode, n_splits, block_size, random_state) -- fold assignment
    itself doesn't depend on which field(s) are being scored, only on y and the CV recipe.

    cv_mode="blocked": contiguous block_size-day blocks assigned to folds instead of
    per-day shuffling (via stage2_logistic._block_fold_assignment, the same recipe
    rr_021_stage2_LogRegr_Robustness.py's blocked_cv_auc already uses) -- guards against a
    day's autocorrelated neighbour landing in the opposite fold.

    cv_mode="random": StratifiedKFold(shuffle=True). Split on a dummy (n, 1) array rather
    than an actual feature matrix -- StratifiedKFold.split only reads len(X) and y, never
    X's values, so this is behaviourally identical to splitting on any real X but doesn't
    tie fold assignment to one specific field's block (needed once PCA blocks vary per
    field per fold, see PerFoldFeatureCache).
    """
    n = len(y)
    if cv_mode == "blocked":
        from .stage2_logistic import _block_fold_assignment

        rng = np.random.default_rng(random_state)
        fold_of_day = _block_fold_assignment(n, n_splits, block_size, rng)
        raw_iter = [
            (np.flatnonzero(fold_of_day != f), np.flatnonzero(fold_of_day == f))
            for f in range(n_splits)
        ]
    elif cv_mode == "random":
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        raw_iter = list(skf.split(np.zeros((n, 1)), y))
    else:
        raise ValueError(f"cv_mode must be 'random' or 'blocked', got {cv_mode!r}")

    fold_iter = []
    for train_idx, val_idx in raw_iter:
        if (len(val_idx) == 0 or y[val_idx].sum() < 1
                or y[val_idx].sum() == len(val_idx) or y[train_idx].sum() < 2):
            continue
        fold_iter.append((train_idx, val_idx))
    return fold_iter


def _cv_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    *,
    cv_mode: str = "random",
    block_size: int = 10,
    C: float = 1.0,
    random_state: int = 42,
) -> float:
    """cv_mode="random" (default): StratifiedKFold(shuffle=True) — unchanged from before
    2026-08-17, kept as the default so existing callers/output are untouched.

    cv_mode="blocked": contiguous block_size-day blocks assigned to folds instead of
    per-day shuffling — see _make_fold_iter's docstring.

    C: LogisticRegression's inverse regularization strength (default 1.0, sklearn's own
    default — unchanged unless explicitly passed). Lower C = stronger L2 shrinkage,
    directly counteracting the extra capacity from wide PCA blocks.

    Still used as-is (single pre-assembled X, no per-fold PCA) by
    rr_021_stage2_LogRegr_Robustness.py and residual_extreme_signal_diag_LOGISTIC.py,
    which score their own already-fixed feature matrices — this signature/behaviour is
    unchanged by the 2026-08-18 per-fold-PCA fix in this module (see PerFoldFeatureCache),
    which only affects feature_relevance.py's OWN field-combination scoring.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    fold_iter = _make_fold_iter(y, n_splits, cv_mode, block_size, random_state)

    aucs: list[float] = []
    for train_idx, val_idx in fold_iter:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state,
                                  solver="lbfgs", C=C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y[train_idx])
        proba = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y[val_idx], proba))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


def _pca_fold_blocks(
    X_raw: np.ndarray,
    fold_iter: list[tuple[np.ndarray, np.ndarray]],
    max_modes_per_field: int | None,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fits PCA separately for each CV fold, using ONLY that fold's training rows — the
    per-fold analogue of the old _pca_block's single record-wide fit (see module
    docstring's "Per-fold PCA refit" entry). Returns one (train_block, val_block) pair per
    fold in fold_iter, so a field's PCA basis (and the mean it's centered on) never sees a
    row that ends up in that fold's own validation set.

    n_components is capped at max_modes_per_field AND at len(train_idx)-1 (PCA can't
    return more components than training samples minus one), so small folds/regimes don't
    error out -- this means a small regime's per-fold block can end up narrower than its
    bbox-mode counterpart, which is expected/correct, not a bug.
    """
    from sklearn.decomposition import PCA

    blocks = []
    n_native = X_raw.shape[1]
    for train_idx, val_idx in fold_iter:
        k = n_native if max_modes_per_field is None else min(max_modes_per_field, n_native)
        k = max(1, min(k, len(train_idx) - 1))
        pca = PCA(n_components=k, svd_solver="randomized", random_state=random_state)
        train_block = pca.fit_transform(X_raw[train_idx]).astype(np.float64)
        val_block = pca.transform(X_raw[val_idx]).astype(np.float64)
        blocks.append((train_block, val_block))
    return blocks


def _field_fold_blocks(
    X_scope: np.ndarray,
    fold_iter: list[tuple[np.ndarray, np.ndarray]],
    field: str,
    max_modes_per_field: int | None,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shared by PerFoldFeatureCache.__init__ and run_feature_relevance's field-streaming
    loader: single-column fields (z500 always; any field under --field-representation
    domain_mean) skip PCA entirely — fitting PCA with n_components=1 is a pointless,
    potentially sign-flipping no-op, not a genuine dimensionality reduction — and are just
    sliced per fold directly. Every other field goes through _pca_fold_blocks."""
    if field == "z500" or X_scope.shape[1] == 1:
        return [(X_scope[train_idx], X_scope[val_idx]) for train_idx, val_idx in fold_iter]
    return _pca_fold_blocks(X_scope, fold_iter, max_modes_per_field, random_state=random_state)


class PerFoldFeatureCache:
    """Precomputes, once per (region/bbox/regime) subset, every field's per-fold PCA
    train/val blocks — fit strictly on that fold's training rows, closing the global-fit
    leak (see module docstring). Every downstream scoring call for this subset (greedy
    selection, leave-one-out, random field draws, permutation null, bootstrap) reuses
    these same precomputed blocks via cv_auc() rather than refitting PCA per candidate
    combination: PCA is unsupervised, so a field's per-fold blocks depend only on which
    rows are train vs. val for that fold, never on which OTHER fields are being combined
    with it or which label (real or shuffled) is being predicted. That's what makes
    reusing one cache across an entire greedy search / random-draw batch / permutation-null
    run correct, not just convenient.

    z500 is exempt (matches the old _pca_block docstring: it has no per-field PCA block,
    kept as its single domain-mean column) — stored directly, sliced per fold, no PCA fit.
    """

    def __init__(
        self,
        field_raw: dict[str, np.ndarray],
        y: np.ndarray,
        *,
        n_splits: int = 5,
        cv_mode: str = "random",
        block_size: int = 10,
        max_modes_per_field: int | None = DEFAULT_MAX_MODES_PER_FIELD,
        random_state: int = 42,
    ):
        self.field_names = list(field_raw)
        self.fold_iter = _make_fold_iter(y, n_splits, cv_mode, block_size, random_state)
        self._blocks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            field: _field_fold_blocks(X_raw, self.fold_iter, field, max_modes_per_field, random_state)
            for field, X_raw in field_raw.items()
        }

    @classmethod
    def from_prebuilt(
        cls,
        fold_iter: list[tuple[np.ndarray, np.ndarray]],
        blocks: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    ) -> "PerFoldFeatureCache":
        """Build directly from externally-computed per-field per-fold blocks, bypassing
        __init__'s raw-matrix loop entirely. Used by run_feature_relevance's field-
        streaming loader (2026-08-19 memory fix — see module docstring's "Field
        streaming" entry): fields are loaded and PCA-refit ONE AT A TIME, immediately
        discarding each field's full raw grid once its small per-fold blocks are
        computed, rather than holding every field's full raw matrix in memory
        simultaneously (the OOM __init__'s all-at-once field_raw dict caused at large
        bboxes — tens of thousands of grid points x ~7000 days x 27 fields).
        """
        self = cls.__new__(cls)
        self.fold_iter = fold_iter
        self.field_names = list(blocks)
        self._blocks = blocks
        return self

    def cv_auc(self, field_names: list[str], y: np.ndarray, C: float = 1.0) -> float:
        """CV AUC for the given field subset. ``y`` is looked up by the SAME train_idx/
        val_idx this cache's fold_iter was built from — pass a label-shuffled y (same
        length/order as the y the cache was built with) to reuse this cache's fold
        partition and PCA blocks for a permutation-null draw at zero extra PCA cost."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        aucs: list[float] = []
        for i, (train_idx, val_idx) in enumerate(self.fold_iter):
            X_tr = np.hstack([self._blocks[f][i][0] for f in field_names])
            X_va = np.hstack([self._blocks[f][i][1] for f in field_names])
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)
            clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                      random_state=42, solver="lbfgs", C=C)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr, y[train_idx])
            proba = clf.predict_proba(X_va)[:, 1]
            try:
                aucs.append(roc_auc_score(y[val_idx], proba))
            except ValueError:
                pass
        return float(np.mean(aucs)) if aucs else float("nan")


def _build_global_fit_blocks(
    field_raw: dict[str, np.ndarray],
    max_modes_per_field: int | None = DEFAULT_MAX_MODES_PER_FIELD,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """No-CV companion structure for _in_sample_auc/_in_sample_trace ONLY: fits each
    field's PCA on ALL of field_raw's rows at once (the ceiling reference _cv_auc's
    honest, held-out number is bounded by — not a generalization claim, see
    _in_sample_auc's docstring). Deliberately NOT fold-aware — reusing a per-fold-fit
    block here would defeat the point of an in-sample ceiling. z500 has no PCA
    (domain-mean, unchanged)."""
    from sklearn.decomposition import PCA

    blocks: dict[str, np.ndarray] = {}
    for field, X in field_raw.items():
        if field == "z500" or X.shape[1] == 1:
            blocks[field] = X
            continue
        k = X.shape[1] if max_modes_per_field is None else min(max_modes_per_field, X.shape[1])
        k = max(1, min(k, X.shape[0] - 1))
        pca = PCA(n_components=k, svd_solver="randomized", random_state=random_state)
        blocks[field] = pca.fit_transform(X).astype(np.float64)
    return blocks


def _in_sample_auc(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> float:
    """No-CV companion to _cv_auc: fits on ALL of X/y (no held-out fold) and scores on
    those SAME rows. Not a generalization estimate — it's the ceiling _cv_auc is bounded
    by. The gap between this and _cv_auc's honest, held-out number at the same feature set
    is a direct, cheap overfitting readout: small gap means the CV number reflects real
    generalization; a large gap even after blocked CV + regularization means the model is
    still fitting something that doesn't hold up out of sample."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs", C=C)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_scaled, y)
    proba = clf.predict_proba(X_scaled)[:, 1]
    try:
        return float(roc_auc_score(y, proba))
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Greedy selection (factored for reuse in bootstrap)
# ---------------------------------------------------------------------------

def _greedy_once(
    cache: PerFoldFeatureCache,
    y: np.ndarray,
    auc_threshold: float = 0.51,
    max_auc: float = 0.95,
    *,
    C: float = 1.0,
    max_fields: int | None = None,
) -> list[tuple[str, float]]:
    """Single greedy sequential feature selection pass (silent).

    Starts from z500 and at each step adds the field whose FULL block (all its PCA
    modes, added as one indivisible unit) gives the highest marginal AUC, as long as it
    exceeds `auc_threshold` and cumulative AUC < `max_auc`.

    ``cache``: PerFoldFeatureCache built for this y/subset — see its docstring and the
    module docstring's "Per-fold PCA refit" entry. ``y`` here is what's scored against
    (the cache's OWN fold partition/PCA blocks don't change even if this y is a
    label-shuffled variant for a permutation-null draw — see cache.cv_auc's docstring).

    ``max_fields``: stop after this many non-Z500 fields have been added, regardless of
    `auc_threshold`/`max_auc` (default None = unbounded) — for callers that want a
    fixed-N comparison, e.g. a plotting convention that reads "AUC after 3 fields" the
    same way across every bbox, or the permutation-null test below, which needs a fixed
    budget to keep runtime bounded.

    Returns a list of (label, cumulative_auc) tuples where the first entry is
    always ("Z500 baseline", z500_auc) and subsequent entries are (field_name, auc).
    """
    if "z500" not in cache.field_names:
        return []

    def cv_auc(names: list[str]) -> float:
        return cache.cv_auc(names, y, C=C)

    z500_auc = cv_auc(["z500"])

    remaining = [f for f in cache.field_names if f != "z500"]
    selected = ["z500"]
    cumulative_auc = z500_auc
    results: list[tuple[str, float]] = [("Z500 baseline", z500_auc)]

    while remaining and cumulative_auc < max_auc:
        if max_fields is not None and len(selected) - 1 >= max_fields:
            break
        best_name, best_auc = None, auc_threshold
        for name in remaining:
            auc = cv_auc(selected + [name])
            if auc > best_auc:
                best_auc, best_name = auc, name
        if best_name is None:
            break
        selected.append(best_name)
        cumulative_auc = best_auc
        remaining.remove(best_name)
        results.append((best_name, best_auc))

    return results


def _summarize_sequential(
    sequential_results: list[tuple[str, float]],
    *,
    n_fields_report: int = 3,
) -> dict:
    """Derives the two numbers actually used for plotting/discussion from a
    _greedy_once trace, without changing the greedy algorithm or its stopping
    behavior: the PEAK cumulative AUC reached along the trace (regardless of how
    many more fields get appended afterward with flat/negative marginal gain),
    and the cumulative AUC after exactly `n_fields_report` fields have been added
    (index n_fields_report in the trace, since index 0 is always "Z500 baseline").

    Returns a dict with z500_baseline_auc, max_auc, field_at_max, n_fields_at_max,
    fields_at_max (the field order up to and including the peak), auc_at_n_fields,
    fields_at_n_fields -- or NaN/None values if sequential_results is too short.
    """
    if not sequential_results:
        return {
            "z500_baseline_auc": float("nan"), "max_auc": float("nan"),
            "field_at_max": None, "n_fields_at_max": 0, "fields_at_max": [],
            "auc_at_n_fields": float("nan"), "fields_at_n_fields": [],
            "n_fields_report": n_fields_report,
        }

    z500_baseline_auc = sequential_results[0][1]
    peak_idx = max(range(len(sequential_results)), key=lambda i: sequential_results[i][1])
    max_auc = sequential_results[peak_idx][1]
    field_at_max = sequential_results[peak_idx][0]
    n_fields_at_max = peak_idx  # peak_idx=0 means baseline alone is the peak
    fields_at_max = [r[0] for r in sequential_results[1:peak_idx + 1]]

    n_at_n = min(n_fields_report, len(sequential_results) - 1)
    auc_at_n_fields = sequential_results[n_at_n][1]
    fields_at_n_fields = [r[0] for r in sequential_results[1:n_at_n + 1]]

    return {
        "z500_baseline_auc": z500_baseline_auc,
        "max_auc": max_auc,
        "field_at_max": field_at_max,
        "n_fields_at_max": n_fields_at_max,
        "fields_at_max": fields_at_max,
        "auc_at_n_fields": auc_at_n_fields,
        "fields_at_n_fields": fields_at_n_fields,
        "n_fields_report": n_fields_report,
    }


def _save_sequential_summary(
    sequential_results: list[tuple[str, float]],
    out_dir: Path,
    tag: str,
    *,
    n_fields_report: int = 3,
    log_prefix: str = "[feature_relevance]",
) -> dict:
    """_summarize_sequential + print + write the sidecar JSON every caller of
    _greedy_once wants (bbox-mode Option B and per-regime _run_ab_subset alike).
    Kept as a separate sidecar file (feature_relevance_sequential_{tag}_optionB_summary.json)
    rather than new columns on the existing per-row CSV -- write_field_selection.py and
    other CSV consumers read that file's row-per-stage schema, so adding columns there
    risks silently changing what they parse."""
    summary = _summarize_sequential(sequential_results, n_fields_report=n_fields_report)
    print(f"{log_prefix} {tag}: max AUC={summary['max_auc']:.4f} "
          f"after {summary['n_fields_at_max']} field(s) "
          f"({' + '.join(['z500'] + summary['fields_at_max'])})")
    print(f"{log_prefix} {tag}: AUC after {summary['n_fields_report']} field(s)="
          f"{summary['auc_at_n_fields']:.4f} "
          f"({' + '.join(['z500'] + summary['fields_at_n_fields'])})")
    with open(out_dir / f"feature_relevance_sequential_{tag}_optionB_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _in_sample_trace(
    field_blocks: dict[str, np.ndarray],
    y: np.ndarray,
    sequential_results: list[tuple[str, float]],
    C: float = 1.0,
) -> list[float]:
    """No-CV companion trace to a _greedy_once result: re-fits the SAME field order
    _greedy_once already chose (via honest, held-out CV) on ALL days with no held-out
    fold, then scores on those same days -- one row per sequential_results entry, in the
    same order, so it drops straight into the sequential_optionB CSV as a same-length
    ``in_sample_auc`` column next to ``cumulative_auc``. See _in_sample_auc's docstring
    for why the gap between the two columns is the useful number, not this column alone.
    """
    if "z500" not in field_blocks or not sequential_results:
        return []
    selected_blocks = [field_blocks["z500"]]
    trace = [_in_sample_auc(selected_blocks[0], y, C=C)]
    for field_name, _ in sequential_results[1:]:
        selected_blocks.append(field_blocks[field_name])
        trace.append(_in_sample_auc(np.hstack(selected_blocks), y, C=C))
    return trace


# ---------------------------------------------------------------------------
# Permutation-null test for the greedy search itself
# ---------------------------------------------------------------------------

def _perm_p_value(observed: float, null_aucs: np.ndarray) -> tuple[float, int]:
    """Fraction of null-distribution AUCs at or above the observed AUC, with the +1/+1
    continuity correction so p is never reported as exactly 0. Same recipe
    rr_021_stage2_LogRegr_Robustness.py's _perm_p_value / residual_extreme_signal_diag_LOGISTIC.py
    already use elsewhere in this pipeline. Returns (p_value, n_valid_permutations)."""
    valid = null_aucs[~np.isnan(null_aucs)]
    if not len(valid) or np.isnan(observed):
        return float("nan"), 0
    p_value = float((1 + np.sum(valid >= observed)) / (1 + len(valid)))
    return p_value, len(valid)


def permutation_null_greedy_auc(
    cache: PerFoldFeatureCache,
    y: np.ndarray,
    *,
    max_fields: int = 3,
    n_permutations: int = 100,
    C: float = 1.0,
    auc_threshold: float = 0.51,
    random_state: int = 42,
) -> tuple[float, np.ndarray, float, int]:
    """Tests whether the greedy search's OWN wrapper-search selection bias — picking the
    single best of up to len(cache.field_names)-1 candidate PCA blocks at each stage,
    scored on the same data the reported AUC comes from — can by itself explain a high
    cumulative_auc, independent of any real X-y relationship.

    Label-shuffles ``y`` (i.i.d. permutation, same convention as
    rr_021_stage2_LogRegr_Robustness.py's permutation_null_blocked_cv_auc /
    residual_extreme_signal_diag_LOGISTIC.py's permutation_null_auc — NOT a block
    permutation; the null is about the search process finding spurious combinations
    among decorrelated candidates, not about preserving y's own autocorrelation), then
    reruns the FULL max_fields-deep greedy search on the shuffled labels and records the
    resulting cumulative_auc. Repeated n_permutations times.

    Cheap despite the repeated greedy search: ``cache`` already holds every field's
    per-fold PCA blocks (PCA is unsupervised — a shuffled y changes what the
    LogisticRegression fits and how AUC scores, never what PCA basis a field's block
    was fit on), so no permutation here triggers a single extra PCA fit.

    Returns (observed_auc, null_aucs, p_value, n_valid_permutations). observed_auc is the
    REAL (non-shuffled) cumulative_auc after max_fields fields — recomputed here rather
    than reusing a caller's already-computed value, so this function is self-contained.
    """
    observed_results = _greedy_once(
        cache, y, auc_threshold=auc_threshold, C=C, max_fields=max_fields,
    )
    observed_auc = observed_results[-1][1] if observed_results else float("nan")

    rng = np.random.RandomState(random_state)
    null_aucs = np.empty(int(n_permutations), dtype=np.float64)
    for i in range(int(n_permutations)):
        y_shuffled = rng.permutation(y)
        seq = _greedy_once(
            cache, y_shuffled, auc_threshold=auc_threshold, C=C, max_fields=max_fields,
        )
        null_aucs[i] = seq[-1][1] if seq else float("nan")

    p_value, n_valid = _perm_p_value(observed_auc, null_aucs)
    return observed_auc, null_aucs, p_value, n_valid


def random_field_selection_auc(
    cache: PerFoldFeatureCache,
    y: np.ndarray,
    *,
    n_fields: int = 3,
    n_draws: int = 100,
    C: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Tests whether the GREEDY SELECTION step itself matters, as opposed to the
    permutation-null test above (which only checks whether the search can invent
    structure from noise). Here y is the REAL, unshuffled label; what varies is WHICH
    n_fields non-Z500 fields get combined with Z500 -- picked uniformly at random
    instead of greedily chosen for their marginal AUC.

    If random field triples reliably score close to the greedily-selected cumulative_auc,
    that means almost ANY combination of fields reaches a high score -- the greedy search
    isn't finding something field-specific, and the jump from Z500-only (~0.5-0.6) to
    Z500+3-fields (~0.8+) is coming from something common to nearly every candidate field
    (e.g. shared seasonal/synoptic-scale autocorrelation across all atmospheric fields
    sampled on the same days, or a subtler leakage upstream of this script — a climatology
    window or the extreme-label definition itself overlapping with how anomalies were
    built) rather than genuine, differentiated physical information. If random triples
    instead cluster well below the greedy number (closer to the Z500-only baseline), that
    supports the greedy search actually finding meaningfully informative combinations.

    Returns an array of length n_draws (or fewer valid rows if len(cache.field_names)-1 <
    n_fields, in which case an empty array is returned).
    """
    non_z500 = [k for k in cache.field_names if k != "z500"]
    if "z500" not in cache.field_names or len(non_z500) < n_fields:
        return np.array([], dtype=np.float64)

    rng = np.random.default_rng(random_state)
    aucs = np.empty(int(n_draws), dtype=np.float64)
    for i in range(int(n_draws)):
        chosen = rng.choice(non_z500, size=n_fields, replace=False)
        aucs[i] = cache.cv_auc(["z500"] + list(chosen), y, C=C)
    return aucs


# ---------------------------------------------------------------------------
# Bootstrap stability selection
# ---------------------------------------------------------------------------

def _bootstrap_greedy_stability(
    field_raw: dict[str, np.ndarray],
    y: np.ndarray,
    n_bootstrap: int = 50,
    block_size: int = 10,
    n_splits: int = 3,
    auc_threshold: float = 0.51,
    *,
    cv_mode: str = "random",
    cv_block_size: int = 10,
    C: float = 1.0,
    max_modes_per_field: int | None = DEFAULT_MAX_MODES_PER_FIELD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap stability of greedy feature selection.

    Runs `n_bootstrap` block-bootstrap resamples of the time series (block
    bootstrap preserves temporal autocorrelation of daily weather data), applying
    the greedy sequential selection on each resample.  For each non-z500 feature
    records how often it was selected, at which greedy stage, and (whenever
    selected) the marginal AUC gain it contributed in that resample.

    Interpretation of selection_frequency:
        >= 0.8  → robustly selected (feature matters across many realisations)
        0.4–0.8 → context-dependent  (sometimes selected, sensitive to sample)
        < 0.4   → unstable (rarely selected; do not rely on for regional comparison)

    Args:
        block_size: length of contiguous blocks for block bootstrap.  A value of
            10 days matches typical synoptic autocorrelation in daily weather data.
        n_splits:   CV folds inside each bootstrap run.  3 is sufficient since
            we only need a stable ranking signal, not a precise AUC estimate.
        cv_mode/cv_block_size/C: forwarded to each resample's PerFoldFeatureCache/
            _greedy_once (kept as a separate cv_block_size from block_size above — one
            is the outer block-bootstrap resample length, the other is the inner CV fold
            block length; they don't have to match).
        max_modes_per_field: forwarded to each resample's PerFoldFeatureCache.

    Each bootstrap draw rebuilds its own PerFoldFeatureCache from field_raw (2026-08-18)
    rather than reusing one across draws — row identity changes under block-bootstrap
    resampling-with-replacement, so a field's per-fold PCA basis has to be refit on the
    resampled rows too, not reused from a different row set.

    Returns (df_stability, df_deltas):
        df_stability: field, selection_frequency, mean_stage, std_stage,
            selection_count, n_bootstrap — sorted descending by selection_frequency.
        df_deltas: long-format, one row per (field, bootstrap_run) the field was
            selected in — field, bootstrap_run, stage, marginal_auc_gain. Feeds a
            per-field boxplot of the AUC effect size, which selection_frequency
            alone can't show.
    """
    rng = np.random.default_rng(42)
    n = len(y)
    non_z500 = [f for f in field_raw if f != "z500"]
    counts: dict[str, int] = {f: 0 for f in non_z500}
    stage_lists: dict[str, list[int]] = {f: [] for f in non_z500}
    delta_records: list[dict] = []
    n_valid = 0

    for b in range(n_bootstrap):
        # Block bootstrap: sample contiguous blocks with replacement
        starts = np.arange(0, n, block_size)
        chosen_starts = rng.choice(starts, size=len(starts), replace=True)
        idx = np.concatenate([
            np.arange(s, min(s + block_size, n)) for s in chosen_starts
        ])[:n]
        field_raw_b = {k: v[idx] for k, v in field_raw.items()}
        y_b = y[idx]

        if int(y_b.sum()) < 5:
            continue  # skip degenerate resamples
        n_valid += 1

        try:
            cache_b = PerFoldFeatureCache(
                field_raw_b, y_b, n_splits=n_splits, cv_mode=cv_mode,
                block_size=cv_block_size, max_modes_per_field=max_modes_per_field,
            )
            seq = _greedy_once(cache_b, y_b, auc_threshold=auc_threshold, C=C)
        except Exception:
            n_valid -= 1
            continue

        prev_auc = seq[0][1] if seq else float("nan")
        for stage, (field, cum_auc) in enumerate(seq):
            if field == "Z500 baseline":
                continue
            if field in counts:
                counts[field] += 1
                stage_lists[field].append(stage)  # stage 1 = first field added
                delta_records.append({
                    "field": field,
                    "bootstrap_run": b,
                    "stage": stage,
                    "marginal_auc_gain": cum_auc - prev_auc,
                })
            prev_auc = cum_auc

    n_denom = max(1, n_valid)
    rows = []
    for field in non_z500:
        cnt = counts[field]
        freq = cnt / n_denom
        stages = stage_lists[field]
        mean_stage = float(np.mean(stages)) if stages else float("nan")
        std_stage = float(np.std(stages, ddof=1)) if len(stages) > 1 else float("nan")
        rows.append({
            "field": field,
            "selection_frequency": round(freq, 3),
            "mean_stage": round(mean_stage, 2) if not np.isnan(mean_stage) else float("nan"),
            "std_stage": round(std_stage, 2) if not np.isnan(std_stage) else float("nan"),
            "selection_count": cnt,
            "n_bootstrap": n_valid,
        })

    df_stability = (pd.DataFrame(rows)
                     .sort_values("selection_frequency", ascending=False)
                     .reset_index(drop=True))
    df_deltas = pd.DataFrame(delta_records)
    return df_stability, df_deltas


def _plot_bootstrap_deltas(
    df_deltas: pd.DataFrame,
    out_path: Path,
    *,
    region: str,
    bbox_step: str,
) -> None:
    """Boxplot of per-run marginal AUC gain (only rows where the field was selected).

    A field selected often but with a tight, near-zero-gain box is a weak/marginal
    contributor despite a high selection_frequency; a field with a wide box shifted
    well above zero is a robust, high-effect contributor.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[feature_relevance] WARNING: matplotlib not available, skipping bootstrap delta plot")
        return

    df_plot = df_deltas
    if df_plot.empty:
        return

    order = (df_plot.groupby("field")["marginal_auc_gain"].median()
             .sort_values(ascending=True).index.tolist())
    data = [df_plot.loc[df_plot["field"] == f, "marginal_auc_gain"].values for f in order]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(order))))
    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=order,
                     medianprops=dict(color="black"))
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
        patch.set_alpha(0.8)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="-")
    ax.set_xlabel("Marginal AUC gain when selected (per bootstrap run)", fontsize=10)
    ax.set_title(
        f"Bootstrap effect size — {region} {bbox_step}\n"
        "All fields  |  box spans IQR across runs where selected",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_feature_relevance(
    module_root: Path,
    region: str,
    bbox_step: str,
    extreme_labels_path: Path,
    extreme_var: str,
    out_dir: Path,
    fields: list[str],
    n_splits: int = 5,
    z500_regimes_path: Path | None = None,
    mode: str = "bbox",
    smooth_window: int = 11,
    n_bootstrap: int = 0,
    bootstrap_block_size: int = 10,
    tag_suffix: str = "",
    use_yeo_johnson: bool = True,
    z500_cluster_method: str = "kmeans",
    area_weight_tag: str = "",
    max_modes_per_field: int | None = DEFAULT_MAX_MODES_PER_FIELD,
    cv_mode: str = "random",
    cv_block_size: int = 10,
    C: float = 1.0,
    n_permutations_null: int = 0,
    permutation_max_fields: int = 3,
    report_in_sample_auc: bool = False,
    n_random_field_draws: int = 0,
    field_representation: str = "domain_mean",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # mode="bbox" never touches the Z500 regime file at all (global selection across all
    # days, no per-regime split) — its output is identical regardless of z500_cluster_method,
    # so tagging it "_zward" would be misleading (implies a Z500-regime dependency that
    # doesn't exist) and would needlessly orphan the existing untagged bbox-mode CSVs.
    _tag_cluster_method = z500_cluster_method if mode == "regime" else "kmeans"
    tag = rr01_tag(region, bbox_step, mode, tag_suffix=tag_suffix, z500_cluster_method=_tag_cluster_method,
                   field_representation=field_representation)

    # --- load extreme labels ---
    print(f"[feature_relevance] Loading extreme labels: {extreme_labels_path.name}  var={extreme_var!r}")
    label_times, y = _load_extreme_labels(extreme_labels_path, extreme_var)
    n_pos = int(y.sum())
    print(f"[feature_relevance] Extreme days: {n_pos} / {len(y)} ({100*n_pos/len(y):.1f}%)")
    if n_pos < 100:
        print(f"[feature_relevance] WARNING: only {n_pos} positive cases — AUC estimates will be noisy")

    # Record which extreme-day definition this run used, as one small sidecar file next to
    # this run's CSVs -- the tag alone doesn't encode it, and a CSV's AUC numbers are only
    # comparable to another run's if this matches (see the flagged_any_gridcell vs
    # flagged_domain_pctl mismatch this was added to make visible/traceable).
    extreme_def = {
        "extreme_labels_path": str(extreme_labels_path),
        "extreme_var": extreme_var,
        "n_days": int(len(y)),
        "n_extreme": n_pos,
        "base_rate": round(n_pos / len(y), 4) if len(y) else float("nan"),
        "cv_mode": cv_mode,
        "cv_block_size": cv_block_size if cv_mode == "blocked" else None,
        "C": C,
        "max_modes_per_field": max_modes_per_field,
    }
    with open(out_dir / f"feature_relevance_{tag}_extreme_def.json", "w") as f:
        json.dump(extreme_def, f, indent=2)
    print(f"[feature_relevance] Extreme-day definition: {extreme_def}")

    # --- Field streaming (2026-08-19 memory fix — see module docstring's "Field
    # streaming" entry): loading every field's FULL raw grid (tens of thousands of
    # columns x ~7000 days at large bboxes) for all candidate fields at once OOM-killed
    # HPC jobs at B0-B3. Instead: (1) peek every field's time coordinate ONLY (cheap,
    # doesn't touch the bulk data) to compute the common date range and, for
    # mode=="regime", each Z500 regime's row mask/fold count; (2) stream each field's
    # full data through ONE AT A TIME, immediately reducing it to its small per-scope
    # per-fold PCA blocks (PerFoldFeatureCache) and discarding the full array before the
    # next field loads. Peak memory is bounded by ONE field's full grid, not all of them
    # at once. z500 itself is small (single domain-mean column) regardless of bbox size,
    # so it's loaded fully up front as before -- no streaming needed for it.
    z500_times: pd.DatetimeIndex | None = None
    z500_col: np.ndarray | None = None
    path = _find_z500_anomaly_file(
        module_root, region, bbox_step=bbox_step, smooth_window=smooth_window,
        use_yeo_johnson=use_yeo_johnson,
    )
    if path is None:
        print(f"[feature_relevance] SKIP z500: no anomaly file found in data/preprocessed/{region}/ for {bbox_step}")
    else:
        print(f"[feature_relevance] Loading z500 (domain-mean — no per-field PCA block; see module docstring): {path.name}")
        try:
            z500_times, vals = _domain_mean_anomaly(path, "z500")
            z500_col = vals.reshape(-1, 1).astype(np.float64)
        except Exception as exc:
            print(f"[feature_relevance] SKIP z500: failed to load — {exc}")

    non_z500_fields = [f for f in fields if f.lower() != "z500"]

    # --- peek pass: dates only, no full arrays ---
    field_dates: dict[str, pd.DatetimeIndex] = {}
    if z500_col is not None:
        field_dates["z500"] = z500_times
    for field_key in non_z500_fields:
        times = _peek_field_dates(
            module_root, region, field_key, bbox_step, smooth_window=smooth_window,
            use_yeo_johnson=use_yeo_johnson, area_weight_tag=area_weight_tag,
            field_representation=field_representation,
        )
        if times is None:
            print(f"[feature_relevance] SKIP {field_key}: no per-field file found under "
                  f"data/multivariate/preprocessed/{region}/{field_key}/ (run rr_00 for this field/bbox first)")
            continue
        field_dates[field_key] = times

    if len(field_dates) < 2:
        raise RuntimeError(f"Only {len(field_dates)} field(s) available — need at least 2 to run diagnostic.")

    # --- align to common time (intersection of all fields + labels) ---
    # normalize times to dates (midnight) for alignment, since fields and labels may have different hours
    label_dates = pd.DatetimeIndex(label_times.normalize())
    field_dates_norm = {k: pd.DatetimeIndex(v.normalize()) for k, v in field_dates.items()}

    common = label_dates
    for dates in field_dates_norm.values():
        common = common.intersection(dates)
    common = common.sort_values()

    if len(common) == 0:
        raise RuntimeError(
            f"No overlapping dates found between labels (dates {label_dates[0].date()} - {label_dates[-1].date()}) "
            f"and fields (e.g., {list(field_dates_norm.keys())[0]} dates {field_dates_norm[list(field_dates_norm.keys())[0]][0].date()} - {field_dates_norm[list(field_dates_norm.keys())[0]][-1].date()})"
        )

    print(f"[feature_relevance] Common time steps: {len(common)}  ({common[0].date()} – {common[-1].date()})")

    # Get indices using normalized dates
    label_idx = pd.Index(label_dates).get_indexer(common)
    y_aligned = y[label_idx]

    z500_aligned: np.ndarray | None = None
    if z500_col is not None:
        z500_idx = pd.Index(field_dates_norm["z500"]).get_indexer(common)
        z500_aligned = z500_col[z500_idx]

    # --- determine scopes: "bbox" (all rows, mode=="bbox") and/or one scope per valid
    # Z500 regime (z500_regimes_path given) -- mirrors the skip/n_splits_r logic the
    # per-regime loop used to apply AFTER loading everything; now applied BEFORE any
    # non-z500 field's full data is touched, so every scope's fold assignment is fixed
    # before streaming begins. A run in practice only ever populates one of the two
    # (main() only resolves z500_regimes_path when mode=="regime"), preserved here as
    # two independent conditions to match the original code's structure.
    scopes: list[tuple[str, np.ndarray, np.ndarray, int, str]] = []  # (scope_id, row_idx_into_common, y_scope, n_splits_scope, bbox_label)
    if mode == "bbox":
        scopes.append(("bbox", np.arange(len(common)), y_aligned, n_splits, bbox_step))

    all_regime_seq: dict[int, list] = {}
    per_regime_out = out_dir / "per_regime"
    if z500_regimes_path is not None:
        print(f"\n[feature_relevance] === PER-REGIME ANALYSIS ===")
        print(f"[feature_relevance] Loading Z500 regime labels: {z500_regimes_path.name}")
        regime_labels_aligned = _load_z500_regime_labels(z500_regimes_path, common)
        if regime_labels_aligned is not None:
            unique_regimes = sorted(int(r) for r in np.unique(regime_labels_aligned) if r >= 0)
            print(f"[feature_relevance] Z500 regimes found: {unique_regimes}")
            per_regime_out.mkdir(parents=True, exist_ok=True)
            for regime_id in unique_regimes:
                mask = regime_labels_aligned == regime_id
                n_r = int(mask.sum())
                n_pos_r = int(y_aligned[mask].sum())
                print(f"\n[feature_relevance] --- Z500 Regime {regime_id}: {n_r} days, "
                      f"{n_pos_r} extreme ({100*n_pos_r/max(1, n_r):.1f}%) ---")
                if n_pos_r < 10:
                    print(f"[feature_relevance] SKIP: fewer than 10 positive cases")
                    continue
                n_splits_r = min(n_splits, max(2, n_pos_r // 5))
                if n_splits_r < n_splits:
                    print(f"[feature_relevance] Reducing n_splits {n_splits} → {n_splits_r} (small regime)")
                scopes.append((
                    f"R{regime_id}", np.flatnonzero(mask), y_aligned[mask], n_splits_r,
                    f"{bbox_step}_R{regime_id}",
                ))

    # --- build each scope's fold assignment + seed z500's per-fold blocks (cheap: a
    # single column) -- everything below streams non-z500 fields through these fixed
    # scopes, computing every scope's blocks for a field in one pass before discarding it.
    fold_iter_by_scope: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    blocks_by_scope: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for scope_id, row_idx, y_scope, n_splits_scope, _ in scopes:
        fold_iter = _make_fold_iter(y_scope, n_splits_scope, cv_mode, cv_block_size)
        fold_iter_by_scope[scope_id] = fold_iter
        blocks_by_scope[scope_id] = {}
        if z500_aligned is not None:
            blocks_by_scope[scope_id]["z500"] = _field_fold_blocks(
                z500_aligned[row_idx], fold_iter, "z500", max_modes_per_field,
            )

    # --- stream non-z500 fields one at a time ---
    available_fields = ["z500"] if z500_aligned is not None else []
    for field_key in non_z500_fields:
        if field_key not in field_dates:
            continue  # failed the peek pass above

        if field_representation == "domain_mean":
            result = _domain_mean_field(
                module_root, region, field_key, bbox_step, smooth_window=smooth_window,
                use_yeo_johnson=use_yeo_johnson, area_weight_tag=area_weight_tag,
            )
            load_note = "domain-mean (--field-representation domain_mean, no PCA — see module docstring)"
        else:
            result = _raw_field_matrix(
                module_root, region, field_key, bbox_step, smooth_window=smooth_window,
                use_yeo_johnson=use_yeo_johnson, area_weight_tag=area_weight_tag,
            )
            load_note = None  # printed below with grid-point count, once matrix is known

        if result is None:
            print(f"[feature_relevance] SKIP {field_key}: file disappeared between peek and load pass")
            continue
        times, matrix = result
        if load_note is None:
            modes_note = f"capped at {max_modes_per_field} modes" if max_modes_per_field is not None else "uncapped"
            print(f"[feature_relevance] Loading {field_key}: raw field, {matrix.shape[1]} grid points "
                  f"(PCA refit per CV fold, {modes_note})")
        else:
            print(f"[feature_relevance] Loading {field_key}: {load_note}")

        idx = pd.Index(pd.DatetimeIndex(times.normalize())).get_indexer(common)
        X_common = matrix[idx]
        del matrix

        for scope_id, row_idx, _, _, _ in scopes:
            blocks_by_scope[scope_id][field_key] = _field_fold_blocks(
                X_common[row_idx], fold_iter_by_scope[scope_id], field_key, max_modes_per_field,
            )
        del X_common
        available_fields.append(field_key)

    if len(available_fields) < 2:
        raise RuntimeError(f"Only {len(available_fields)} field(s) available — need at least 2 to run diagnostic.")

    def _get_field_raw_for_scope(row_idx: np.ndarray):
        return _stream_field_raw_for_rows(
            module_root, region, fields, bbox_step,
            smooth_window=smooth_window, use_yeo_johnson=use_yeo_johnson,
            area_weight_tag=area_weight_tag, field_representation=field_representation,
            common=common, row_idx=row_idx, z500_aligned=z500_aligned,
        )

    # --- OPTION A+B: global selection — only in bbox mode ---
    # Skipped in regime mode to avoid overwriting the bbox-mode output files
    # (feature_relevance_sequential_*_optionB.csv etc. are read by write_field_selection.py)
    rows_a: list[dict] = []

    if mode == "bbox":
        # Built once for the whole bbox (Options A, B, and C below all reuse it — see
        # PerFoldFeatureCache's docstring for why that's correct, not just convenient).
        cache = PerFoldFeatureCache.from_prebuilt(fold_iter_by_scope["bbox"], blocks_by_scope["bbox"])

        print(f"\n[feature_relevance] === OPTION A: Independent Residual Signal (Diagnostic) ===")
        print(f"[feature_relevance] Tests each field independently against Z500 prediction errors")
        print(f"[feature_relevance] Computing z500-only AUC...")

        if "z500" in cache.field_names:
            X_z500 = z500_aligned
            z500_auc = cache.cv_auc(["z500"], y_aligned, C=C)
            print(f"[feature_relevance] Z500-only AUC: {z500_auc:.4f}")
            print(f"[feature_relevance] Z500 explains {100*z500_auc:.1f}% of extreme event ranking")

            # For each other field, evaluate its ability to explain residuals from z500
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            residual_results = []

            for field_key in cache.field_names:
                if field_key == "z500":
                    rows_a.append({
                        "field": "z500",
                        "z500_auc": round(z500_auc, 4),
                        "residual_auc": float("nan"),
                        "delta_auc_residual": float("nan"),
                        "evaluation_type": "residual",
                    })
                    continue

                # Train z500 model and get residuals
                scaler = StandardScaler()
                X_z500_scaled = scaler.fit_transform(X_z500)
                clf_z500 = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs")
                clf_z500.fit(X_z500_scaled, y_aligned)
                residuals = (y_aligned - clf_z500.predict_proba(X_z500_scaled)[:, 1]).astype(np.float32)

                # Test if other field's block can predict these residuals
                residual_auc = cache.cv_auc([field_key], (residuals > residuals.mean()).astype(int), C=C)
                delta_residual = residual_auc - 0.5  # 0.5 is random guessing

                rows_a.append({
                    "field": field_key,
                    "z500_auc": round(z500_auc, 4),
                    "residual_auc": round(residual_auc, 4),
                    "delta_auc_residual": round(delta_residual, 4),
                    "evaluation_type": "residual",
                })
                residual_results.append((field_key, delta_residual, residual_auc))

            # Print sorted by importance
            print(f"[feature_relevance] Fields ranked by ability to explain Z500's prediction errors:")
            residual_results_sorted = sorted(residual_results, key=lambda x: x[1], reverse=True)
            for field_key, delta_residual, residual_auc in residual_results_sorted:
                print(f"[feature_relevance]   {field_key:10s}: AUC={residual_auc:.4f}  ΔAUC={delta_residual:+.4f}")

            # Sequential/hierarchical selection (greedy approach)
            print(f"\n[feature_relevance] === OPTION B: Sequential Field Selection (PRIMARY - For rr_02) ===")
            print(f"[feature_relevance] Z500 baseline: AUC={z500_auc:.4f} (explains {100*z500_auc:.1f}% of extremes)")

            sequential_results = _greedy_once(cache, y_aligned, C=C)

            # Print verbose stage-by-stage results
            selected_fields = ["z500"]
            prev_auc = sequential_results[0][1]
            for stage_i, (field_name, cum_auc) in enumerate(sequential_results):
                if field_name == "Z500 baseline":
                    print(f"\nStage 1: Z500 baseline")
                    print(f"  → Explains {100*cum_auc:.1f}% of extreme event ranking")
                    continue
                selected_fields.append(field_name)
                marginal_gain = cum_auc - prev_auc
                print(f"\nStage {stage_i + 1}: Add {field_name}")
                print(f"  → Z500 + {' + '.join(selected_fields[1:])} AUC = {cum_auc:.4f}")
                print(f"  → Marginal gain: {marginal_gain:+.4f}")
                print(f"  → Cumulative: {100*cum_auc:.1f}% of extreme event ranking")
                prev_auc = cum_auc

            print(f"\n[feature_relevance] SEQUENTIAL SELECTION RESULTS:")
            print(f"[feature_relevance] Fields at 80%+ AUC: {' + '.join(selected_fields)}")

            # Save Option B data to CSV
            df_b_seq = pd.DataFrame([
                {"field_combination": r[0], "cumulative_auc": r[1]}
                for r in sequential_results
            ])
            if report_in_sample_auc:
                in_sample_blocks = _build_global_fit_blocks(_get_field_raw_for_scope(np.arange(len(common))), max_modes_per_field)
                df_b_seq["in_sample_auc"] = _in_sample_trace(in_sample_blocks, y_aligned, sequential_results, C=C)
                for r, in_auc in zip(sequential_results, df_b_seq["in_sample_auc"]):
                    print(f"[feature_relevance]   {r[0]:12s}: cv_auc={r[1]:.4f}  in_sample_auc={in_auc:.4f}  "
                          f"gap={in_auc - r[1]:+.4f}")
            csv_path_b_seq = out_dir / f"feature_relevance_sequential_{tag}_optionB.csv"
            df_b_seq.to_csv(csv_path_b_seq, index=False)
            print(f"[feature_relevance] Saved Option B data: {csv_path_b_seq}")

            _save_sequential_summary(sequential_results, out_dir, tag,
                                      n_fields_report=permutation_max_fields)

            # Plot Option B results
            png_path_b = out_dir / f"feature_relevance_sequential_{tag}_optionB.png"
            _plot_sequential_elbow(sequential_results, png_path_b, region=region, bbox_step=bbox_step)
            print(f"[feature_relevance] Saved Option B plot: {png_path_b}")

            # --- Permutation-null test on the greedy search itself (optional) ---
            if n_permutations_null > 0:
                print(f"\n[feature_relevance] === PERMUTATION NULL (greedy search, "
                      f"max_fields={permutation_max_fields}, n={n_permutations_null}) ===")
                obs_auc, null_aucs, p_value, n_valid = permutation_null_greedy_auc(
                    cache, y_aligned,
                    max_fields=permutation_max_fields, n_permutations=n_permutations_null, C=C,
                )
                print(f"[feature_relevance] Observed AUC @ {permutation_max_fields} fields: {obs_auc:.4f}")
                print(f"[feature_relevance] Null (label-shuffled) AUC: mean={np.nanmean(null_aucs):.4f}  "
                      f"std={np.nanstd(null_aucs):.4f}  max={np.nanmax(null_aucs):.4f}  "
                      f"(n_valid={n_valid}/{n_permutations_null})")
                print(f"[feature_relevance] p-value: {p_value:.4f}")
                df_null = pd.DataFrame({"permutation": range(len(null_aucs)), "null_auc": null_aucs})
                df_null.attrs["observed_auc"] = obs_auc
                csv_path_null = out_dir / f"feature_relevance_permnull_{tag}.csv"
                df_null.to_csv(csv_path_null, index=False)
                with open(out_dir / f"feature_relevance_permnull_{tag}_summary.json", "w") as f:
                    json.dump({
                        "max_fields": permutation_max_fields, "n_permutations": n_permutations_null,
                        "observed_auc": obs_auc, "null_mean": float(np.nanmean(null_aucs)),
                        "null_std": float(np.nanstd(null_aucs)), "null_max": float(np.nanmax(null_aucs)),
                        "p_value": p_value, "n_valid": n_valid,
                    }, f, indent=2)
                print(f"[feature_relevance] Saved permutation-null: {csv_path_null}")

            # --- Random field-selection test (optional): does WHICH fields matter? ---
            if n_random_field_draws > 0:
                print(f"\n[feature_relevance] === RANDOM FIELD SELECTION (n_fields={permutation_max_fields}, "
                      f"n_draws={n_random_field_draws}) ===")
                greedy_at_n = _greedy_once(
                    cache, y_aligned, C=C, max_fields=permutation_max_fields,
                )
                greedy_auc_at_n = greedy_at_n[-1][1] if greedy_at_n else float("nan")
                random_aucs = random_field_selection_auc(
                    cache, y_aligned,
                    n_fields=permutation_max_fields, n_draws=n_random_field_draws, C=C,
                )
                print(f"[feature_relevance] Greedy-selected AUC @ {permutation_max_fields} fields: {greedy_auc_at_n:.4f}")
                if len(random_aucs):
                    print(f"[feature_relevance] Random field-triples AUC: mean={np.nanmean(random_aucs):.4f}  "
                          f"std={np.nanstd(random_aucs):.4f}  min={np.nanmin(random_aucs):.4f}  "
                          f"max={np.nanmax(random_aucs):.4f}  (n={len(random_aucs)})")
                    frac_over_80 = float(np.nanmean(random_aucs >= 0.80))
                    print(f"[feature_relevance] Fraction of RANDOM triples clearing 0.80: {frac_over_80:.2f}")
                    # random_max alone is a weak/biased comparison -- with n_draws=100 the max is an
                    # order statistic that drifts toward the greedy number by chance alone, especially
                    # with noisy small-n AUC estimates. The empirical p-value (same recipe as the
                    # label-shuffle permutation null above) is the honest test: fraction of random
                    # draws that reach or exceed the greedy AUC.
                    p_value_rand, n_valid_rand = _perm_p_value(greedy_auc_at_n, random_aucs)
                    print(f"[feature_relevance] p-value (fraction of random draws >= greedy AUC): "
                          f"{p_value_rand:.4f}  (n_valid={n_valid_rand}/{n_random_field_draws})")
                else:
                    p_value_rand, n_valid_rand = float("nan"), 0
                df_rand = pd.DataFrame({"draw": range(len(random_aucs)), "random_field_auc": random_aucs})
                csv_path_rand = out_dir / f"feature_relevance_randomfields_{tag}.csv"
                df_rand.to_csv(csv_path_rand, index=False)
                with open(out_dir / f"feature_relevance_randomfields_{tag}_summary.json", "w") as f:
                    json.dump({
                        "n_fields": permutation_max_fields, "n_draws": n_random_field_draws,
                        "greedy_auc": greedy_auc_at_n,
                        "random_mean": float(np.nanmean(random_aucs)) if len(random_aucs) else float("nan"),
                        "random_std": float(np.nanstd(random_aucs)) if len(random_aucs) else float("nan"),
                        "random_min": float(np.nanmin(random_aucs)) if len(random_aucs) else float("nan"),
                        "random_max": float(np.nanmax(random_aucs)) if len(random_aucs) else float("nan"),
                        "frac_random_over_080": float(np.nanmean(random_aucs >= 0.80)) if len(random_aucs) else float("nan"),
                        "p_value": p_value_rand, "n_valid": n_valid_rand,
                    }, f, indent=2)
                print(f"[feature_relevance] Saved random field-selection test: {csv_path_rand}")

            # --- Bootstrap stability (optional) ---
            if n_bootstrap > 0:
                print(f"\n[feature_relevance] === BOOTSTRAP STABILITY (n={n_bootstrap} runs, "
                      f"block={bootstrap_block_size} days) ===")
                df_stab, df_deltas = _bootstrap_greedy_stability(
                    _get_field_raw_for_scope(np.arange(len(common))), y_aligned,
                    n_bootstrap=n_bootstrap,
                    block_size=bootstrap_block_size,
                    n_splits=min(n_splits, 3),
                    cv_mode=cv_mode, cv_block_size=cv_block_size, C=C,
                    max_modes_per_field=max_modes_per_field,
                )
                robust = df_stab[df_stab["selection_frequency"] >= 0.8]["field"].tolist()
                unstable = df_stab[df_stab["selection_frequency"] < 0.4]["field"].tolist()
                print(f"[feature_relevance] Robust (≥80%): {robust}")
                print(f"[feature_relevance] Unstable (<40%): {unstable}")
                for _, row in df_stab.iterrows():
                    print(f"[feature_relevance]   {row['field']:15s}: "
                          f"freq={row['selection_frequency']:.2f}  "
                          f"mean_stage={row['mean_stage']:.1f}  "
                          f"std_stage={row['std_stage']:.1f}")

                csv_stab = out_dir / f"feature_relevance_bootstrap_{tag}_stability.csv"
                df_stab.to_csv(csv_stab, index=False)
                print(f"[feature_relevance] Saved bootstrap stability: {csv_stab}")

                if not df_deltas.empty:
                    csv_deltas = out_dir / f"feature_relevance_bootstrap_{tag}_deltas.csv"
                    df_deltas.to_csv(csv_deltas, index=False)
                    print(f"[feature_relevance] Saved bootstrap deltas: {csv_deltas}")

                    png_deltas = out_dir / f"feature_relevance_bootstrap_{tag}_deltas.png"
                    _plot_bootstrap_deltas(df_deltas, png_deltas, region=region, bbox_step=bbox_step)
                    print(f"[feature_relevance] Saved bootstrap deltas plot: {png_deltas}")
        else:
            print(f"[feature_relevance] WARNING: z500 not in cache.field_names, skipping residual-based evaluation")

    else:
        print(f"[feature_relevance] mode=regime: skipping global Options A+B "
              f"(bbox results in {out_dir} are preserved)")

    # --- OPTION C: Hierarchical/additive importance (global baseline) ---
    # bbox mode only: this is a whole-bbox computation identical regardless of
    # regime labels, so running it under mode=regime would just duplicate the
    # bbox-mode output under a different tag — exactly the leakage we removed
    # the subfolder split to prevent.
    csv_path_b: Path | None = None
    if mode == "bbox":
        print(f"\n[feature_relevance] === OPTION C: Global Importance (All Fields Together - Diagnostic) ===")
        all_field_names = cache.field_names
        print(f"[feature_relevance] Computing baseline AUC ({n_splits}-fold CV, {len(all_field_names)} fields, "
              f"PCA refit per fold)...")
        baseline_auc = cache.cv_auc(all_field_names, y_aligned, C=C)
        print(f"[feature_relevance] Baseline AUC (all fields): {baseline_auc:.4f}")

        rows_b: list[dict] = []
        for field_key in all_field_names:
            subset = [f for f in all_field_names if f != field_key]
            auc_without = cache.cv_auc(subset, y_aligned, C=C)
            delta = baseline_auc - auc_without
            print(f"[feature_relevance]   drop {field_key:10s}: AUC={auc_without:.4f}  ΔAUC={delta:+.4f}")
            rows_b.append({
                "field": field_key,
                "baseline_auc": round(baseline_auc, 4),
                "auc_without": round(auc_without, 4),
                "delta_auc": round(delta, 4),
                "evaluation_type": "hierarchical",
            })

        # mark skipped fields
        for field_key in fields:
            if field_key not in all_field_names:
                rows_b.append({
                    "field": field_key,
                    "baseline_auc": float("nan"),
                    "auc_without": float("nan"),
                    "delta_auc": float("nan"),
                    "evaluation_type": "hierarchical",
                })

        # Save both evaluation types
        df_b = pd.DataFrame(rows_b).sort_values("delta_auc", ascending=False).reset_index(drop=True)
        csv_path_b = out_dir / f"feature_relevance_hierarchical_{tag}.csv"
        df_b.to_csv(csv_path_b, index=False)
        print(f"\n[feature_relevance] Saved hierarchical evaluation: {csv_path_b}")

        if rows_a:
            df_a = pd.DataFrame(rows_a).sort_values("delta_auc_residual", ascending=False, na_position="last").reset_index(drop=True)
            csv_path_a = out_dir / f"feature_relevance_residual_{tag}_optionA.csv"
            df_a.to_csv(csv_path_a, index=False)
            print(f"[feature_relevance] Saved residual evaluation: {csv_path_a}")

            # Plot for Option A
            png_path_a = out_dir / f"feature_relevance_residual_{tag}_optionA.png"
            _plot_bar_residual(df_a, z500_auc, png_path_a, region=region, bbox_step=bbox_step)
            print(f"[feature_relevance] Saved Option A plot: {png_path_a}")

        # --- bar chart (hierarchical) ---
        png_path = out_dir / f"feature_relevance_hierarchical_{tag}_optionC.png"
        _plot_bar(df_b, baseline_auc, png_path, region=region, bbox_step=bbox_step, method="Hierarchical")
        print(f"[feature_relevance] Saved Option C plot: {png_path}")

        # summary
        kept_c = df_b[(df_b["delta_auc"] > 0.01) & (df_b["delta_auc"].notna())]["field"].tolist()
        print(f"\n[feature_relevance] === SUMMARY ===")
        print(f"[feature_relevance] Option A (Independent): Fields explaining Z500 residuals (each tested alone)")

        if rows_a:
            kept_a = df_a[(df_a["delta_auc_residual"] > 0.01) & (df_a["delta_auc_residual"].notna())]["field"].tolist()
            print(f"[feature_relevance]   KEEP (ΔAUC > 0.01): {kept_a}")

        print(f"\n[feature_relevance] Option B (Sequential - RECOMMENDED for rr_02):")
        print(f"[feature_relevance]   See recommendations above (marginal contributions)")

        print(f"\n[feature_relevance] Option C (Global): Fields with ΔAUC > 0.01 when all combined: {kept_c}")

        print(f"\n[feature_relevance] Baseline AUC={baseline_auc:.4f}  n_fields={len(all_field_names)}  n_days={len(common)}  n_extreme={int(y_aligned.sum())}")
    else:
        print(f"[feature_relevance] mode=regime: skipping global Option C "
              f"(identical to bbox-mode result for this bbox — see bbox/ subfolder)")

    # --- Per-regime feature selection (optional): dispatch over the regime scopes
    # already determined + streamed above (see "Field streaming" section) -- regime
    # discovery/masking/skip-logic already happened and printed before any non-z500
    # field's data was touched; this just builds each regime's cache from its
    # precomputed blocks and hands off to _run_ab_subset. ---
    regime_scopes = [s for s in scopes if s[0] != "bbox"]
    if regime_scopes:
        for scope_id, row_idx, y_scope, n_splits_scope, bbox_label in regime_scopes:
            regime_id = int(scope_id[1:])
            regime_tag = f"{tag}_R{regime_id}"
            cache_r = PerFoldFeatureCache.from_prebuilt(fold_iter_by_scope[scope_id], blocks_by_scope[scope_id])
            X_z500_r = z500_aligned[row_idx] if z500_aligned is not None else None
            seq_results = _run_ab_subset(
                cache_r, X_z500_r, y_scope,
                per_regime_out, regime_tag,
                region, bbox_label, n_splits_scope,
                n_bootstrap=n_bootstrap,
                bootstrap_block_size=bootstrap_block_size,
                cv_mode=cv_mode, cv_block_size=cv_block_size, C=C,
                max_modes_per_field=max_modes_per_field,
                n_permutations_null=n_permutations_null, permutation_max_fields=permutation_max_fields,
                report_in_sample_auc=report_in_sample_auc,
                n_random_field_draws=n_random_field_draws,
                get_field_raw=(lambda ri=row_idx: _get_field_raw_for_scope(ri)),
            )
            if seq_results:
                all_regime_seq[regime_id] = seq_results

        if len(all_regime_seq) > 1:
            _plot_regime_comparison(
                all_regime_seq,
                out_dir / f"feature_relevance_sequential_{tag}_all_regimes.png",
                region=region,
                bbox_step=bbox_step,
            )

        print(f"\n[feature_relevance] === PER-REGIME SUMMARY ===")
        for rid, seq_res in sorted(all_regime_seq.items()):
            summary = _summarize_sequential(seq_res, n_fields_report=permutation_max_fields)
            fields_at_max = ["z500"] + summary["fields_at_max"]
            print(f"[feature_relevance]   Z500-Regime {rid}: {' + '.join(fields_at_max)}  "
                  f"(max AUC={summary['max_auc']:.4f} @ {summary['n_fields_at_max']} field(s); "
                  f"AUC@{summary['n_fields_report']} fields={summary['auc_at_n_fields']:.4f})")

    return csv_path_b


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_bar(df: pd.DataFrame, baseline_auc: float, out_path: Path, *, region: str, bbox_step: str, method: str = "Leave-One-Out") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[feature_relevance] WARNING: matplotlib not available, skipping plot")
        return

    available = df[df["delta_auc"].notna()].sort_values("delta_auc", ascending=True)
    colors = ["#2196F3" if v > 0.01 else "#90CAF9" if v > 0 else "#FFCDD2" for v in available["delta_auc"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(available))))
    ax.barh(available["field"], available["delta_auc"], color=colors, edgecolor="white", height=0.6)
    ax.axvline(0.01, color="#E53935", linewidth=1.2, linestyle="--", label="keep threshold (0.01)")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="-")
    ax.set_xlabel("ΔAUC (baseline − AUC without field)", fontsize=10)
    ax.set_title(
        f"Feature relevance — {region} {bbox_step} ({method})\n"
        f"Baseline AUC = {baseline_auc:.3f}  (logistic regression, 5-fold CV)",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    # annotate skipped fields
    skipped = df[df["delta_auc"].isna()]["field"].tolist()
    if skipped:
        ax.text(
            0.98, 0.02, f"Not available: {', '.join(skipped)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="gray",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_bar_residual(df: pd.DataFrame, z500_auc: float, out_path: Path, *, region: str, bbox_step: str) -> None:
    """Plot residual-based feature importance (Option A)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[feature_relevance] WARNING: matplotlib not available, skipping residual plot")
        return

    # Exclude z500 itself (has NaN residual AUC)
    available = df[df["delta_auc_residual"].notna()].sort_values("delta_auc_residual", ascending=True)
    if len(available) == 0:
        print("[feature_relevance] WARNING: no residual AUC values to plot")
        return

    colors = ["#2196F3" if v > 0.01 else "#90CAF9" if v > 0 else "#FFCDD2" for v in available["delta_auc_residual"]]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(available))))
    ax.barh(available["field"], available["delta_auc_residual"], color=colors, edgecolor="white", height=0.6)
    ax.axvline(0.01, color="#E53935", linewidth=1.2, linestyle="--", label="keep threshold (0.01)")
    ax.axvline(0, color="black", linewidth=0.6, linestyle="-")
    ax.set_xlabel("ΔAUC = Improvement over random guessing (AUC − 0.5)\nHow well field explains Z500's prediction errors", fontsize=9)
    ax.set_title(
        f"Residual-Based Feature Importance — {region} {bbox_step}\n"
        f"Z500-only AUC = {z500_auc:.3f}  |  Which fields explain what Z500 misses?",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sequential_elbow(sequential_results: list, out_path: Path, *, region: str, bbox_step: str) -> None:
    """Plot sequential field selection (Option B) with 2 subplots: incremental contribution (left) and cumulative AUC (right)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[feature_relevance] WARNING: matplotlib not available, skipping sequential plot")
        return

    if not sequential_results:
        return

    # Extract field names and AUCs
    field_names = [f[0] for f in sequential_results]
    aucs = [f[1] for f in sequential_results]

    # Build simplified x-axis labels
    simplified_labels = ["z500 baseline"] + [f"+{field_names[i]}" for i in range(1, len(field_names))]
    x_pos = range(len(field_names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # ===== LEFT PLOT: Marginal Contributions Beyond Z500 (Line Plot) =====
    # Calculate marginal gains (contribution of each field added)
    marginal_gains = [aucs[0]]  # Z500 baseline
    for i in range(1, len(aucs)):
        marginal_gains.append(aucs[i] - aucs[i-1])

    ax1.plot(x_pos, marginal_gains, marker='o', linewidth=2.5, markersize=9,
             color="black", markeredgecolor="black", markerfacecolor="white", markeredgewidth=1.5)

    ax1.set_xlabel("Fields Added Sequentially", fontsize=10, fontweight="bold")
    ax1.set_ylabel("AUC Contribution", fontsize=10, fontweight="bold")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim([0, 0.7])
    ax1.grid(True, alpha=0.3, linestyle=":", axis='y')
    ax1.set_title("Contribution of Each Field", fontsize=11, fontweight="bold")

    # ===== RIGHT PLOT: Cumulative AUC (Line Plot, No Area) =====
    ax2.plot(x_pos, aucs, marker='o', linewidth=2.5, markersize=9,
             color="black", markeredgecolor="black", markerfacecolor="white", markeredgewidth=1.5, label="Cumulative AUC")

    # Threshold lines
    ax2.axhline(0.80, color="#F57C00", linestyle="--", linewidth=1.8, label="80% threshold", zorder=2)
    ax2.axhline(0.90, color="#C62828", linestyle="--", linewidth=1.8, label="90% threshold", zorder=2)

    ax2.set_xlabel("Fields Added Sequentially", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Cumulative AUC", fontsize=10, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(simplified_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_title("Cumulative AUC", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Regime-specific helpers
# ---------------------------------------------------------------------------

def _load_z500_regime_labels(
    z500_regimes_path: Path,
    common_dates: pd.DatetimeIndex,
) -> np.ndarray | None:
    """Load Z500 Stage 1 regime labels and align to common_dates.

    Returns a 1D int array (same length as common_dates) or None on failure.
    Days not found in the regime file are set to -1.
    """
    try:
        with xr.open_dataset(z500_regimes_path, chunks=None) as ds:
            if "regime_labels" in ds:
                da = ds["regime_labels"].load()
            else:
                var = next(
                    (v for v in ds.data_vars
                     if any(d in ds[v].dims for d in ("time", "valid_time"))),
                    None,
                )
                if var is None:
                    raise ValueError("No time-dimensioned variable found in regime file")
                da = ds[var].load()

        _, times = _get_time_dim(da)
        regime_dates = pd.DatetimeIndex(times).normalize()
        labels_full = da.values.astype(int)

        common_norm = pd.DatetimeIndex(common_dates).normalize()
        idx = pd.Index(regime_dates).get_indexer(common_norm)
        n_missing = int((idx == -1).sum())
        if n_missing > 0:
            pct = 100 * n_missing / len(common_norm)
            print(f"[feature_relevance] WARNING: {n_missing}/{len(common_norm)} ({pct:.1f}%) common dates "
                  f"not found in regime file — those days excluded from per-regime analysis")
            if pct > 20:
                print(f"[feature_relevance] ERROR: >20% missing dates — regime file likely covers wrong region/period")
                return None

        safe_idx = np.where(idx >= 0, idx, 0)
        aligned = np.where(idx >= 0, labels_full[safe_idx], -1)
        return aligned.astype(int)

    except Exception as exc:
        print(f"[feature_relevance] ERROR loading z500 regime labels: {exc}")
        return None


def _run_ab_subset(
    cache: PerFoldFeatureCache,
    X_z500: np.ndarray | None,
    y: np.ndarray,
    out_dir: Path,
    tag: str,
    region: str,
    bbox_step: str,
    n_splits: int = 5,
    n_bootstrap: int = 0,
    bootstrap_block_size: int = 10,
    *,
    cv_mode: str = "random",
    cv_block_size: int = 10,
    C: float = 1.0,
    max_modes_per_field: int | None = DEFAULT_MAX_MODES_PER_FIELD,
    n_permutations_null: int = 0,
    permutation_max_fields: int = 3,
    report_in_sample_auc: bool = False,
    n_random_field_draws: int = 0,
    get_field_raw=None,
) -> list:
    """Run Options A (residual) and B (sequential) on an already-built PerFoldFeatureCache
    for one scope's rows (``cache``, ``X_z500`` — the raw z500 column for the SAME rows,
    needed for Option A's non-CV residual-model fit — and ``y`` all correspond to the same
    scope; see run_feature_relevance's "Field streaming" section, module docstring's
    "Field streaming" entry, 2026-08-19).

    ``get_field_raw``: zero-arg callable returning field_key -> raw matrix for this SAME
    scope's rows, ONLY invoked (lazily) if report_in_sample_auc or n_bootstrap>0 — both
    need genuine raw per-field access at this scope (in-sample fits on ALL rows at once,
    bootstrap refits PCA on RESAMPLED rows), which cache's precomputed per-fold blocks
    don't cover. Left None when neither option is in use, since building it means
    re-reading every field from disk for this one scope.

    Every other diagnostic below (greedy, permutation null, random field selection) reuses
    ``cache`` directly — PCA fit per (field, fold) exactly once, not once per candidate
    combination.

    Saves CSVs and PNGs to out_dir using tag in filenames.
    Returns sequential_results: list of (field_name, cumulative_auc) tuples.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if "z500" not in cache.field_names:
        print(f"[feature_relevance] WARNING: z500 not in cache.field_names for {tag} — skipping")
        return []

    n_pos = int(y.sum())
    if n_pos < 10:
        print(f"[feature_relevance] WARNING: only {n_pos} positive cases for {tag} — skipping")
        return []

    z500_auc = cache.cv_auc(["z500"], y, C=C)
    print(f"[feature_relevance] {tag}: Z500-only AUC={z500_auc:.4f}  n_days={len(y)}  n_extreme={n_pos}")

    # --- Option A: residual importance ---
    rows_a: list[dict] = []
    for field_key in cache.field_names:
        if field_key == "z500":
            rows_a.append({
                "field": "z500",
                "z500_auc": round(z500_auc, 4),
                "residual_auc": float("nan"),
                "delta_auc_residual": float("nan"),
                "evaluation_type": "residual",
            })
            continue
        scaler = StandardScaler()
        X_z500_scaled = scaler.fit_transform(X_z500)
        clf_z500 = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf_z500.fit(X_z500_scaled, y)
        residuals = (y - clf_z500.predict_proba(X_z500_scaled)[:, 1]).astype(np.float32)
        residual_auc = cache.cv_auc([field_key], (residuals > residuals.mean()).astype(int), C=C)
        rows_a.append({
            "field": field_key,
            "z500_auc": round(z500_auc, 4),
            "residual_auc": round(residual_auc, 4),
            "delta_auc_residual": round(residual_auc - 0.5, 4),
            "evaluation_type": "residual",
        })

    df_a = (pd.DataFrame(rows_a)
            .sort_values("delta_auc_residual", ascending=False, na_position="last")
            .reset_index(drop=True))
    df_a.to_csv(out_dir / f"feature_relevance_residual_{tag}_optionA.csv", index=False)
    _plot_bar_residual(df_a, z500_auc,
                       out_dir / f"feature_relevance_residual_{tag}_optionA.png",
                       region=region, bbox_step=bbox_step)

    # --- Option B: sequential greedy selection ---
    sequential_results = _greedy_once(cache, y, C=C)

    selected_fields = ["z500"]
    prev_auc = sequential_results[0][1]
    for field_name, cum_auc in sequential_results[1:]:
        marginal_gain = cum_auc - prev_auc
        selected_fields.append(field_name)
        print(f"[feature_relevance]   {tag}: +{field_name}  "
              f"AUC={cum_auc:.4f}  gain={marginal_gain:+.4f}")
        prev_auc = cum_auc

    df_b = pd.DataFrame([{"field_combination": r[0], "cumulative_auc": r[1]} for r in sequential_results])
    if report_in_sample_auc:
        in_sample_blocks = _build_global_fit_blocks(get_field_raw(), max_modes_per_field)
        df_b["in_sample_auc"] = _in_sample_trace(in_sample_blocks, y, sequential_results, C=C)
        for r, in_auc in zip(sequential_results, df_b["in_sample_auc"]):
            print(f"[feature_relevance]   {tag}: {r[0]:12s} cv_auc={r[1]:.4f}  in_sample_auc={in_auc:.4f}  "
                  f"gap={in_auc - r[1]:+.4f}")
    df_b.to_csv(out_dir / f"feature_relevance_sequential_{tag}_optionB.csv", index=False)
    _plot_sequential_elbow(sequential_results,
                           out_dir / f"feature_relevance_sequential_{tag}_optionB.png",
                           region=region, bbox_step=bbox_step)

    print(f"[feature_relevance] {tag}: final selection = {' + '.join(selected_fields)}")
    _save_sequential_summary(sequential_results, out_dir, tag,
                              n_fields_report=permutation_max_fields)

    # --- Permutation-null test on the greedy search itself (optional) ---
    if n_permutations_null > 0:
        print(f"[feature_relevance] {tag}: permutation null (max_fields={permutation_max_fields}, "
              f"n={n_permutations_null}) ...")
        obs_auc, null_aucs, p_value, n_valid = permutation_null_greedy_auc(
            cache, y,
            max_fields=permutation_max_fields, n_permutations=n_permutations_null, C=C,
        )
        print(f"[feature_relevance]   {tag}: observed AUC @ {permutation_max_fields} fields = {obs_auc:.4f}  "
              f"null mean={np.nanmean(null_aucs):.4f}  std={np.nanstd(null_aucs):.4f}  "
              f"max={np.nanmax(null_aucs):.4f}  p={p_value:.4f}  (n_valid={n_valid}/{n_permutations_null})")
        df_null = pd.DataFrame({"permutation": range(len(null_aucs)), "null_auc": null_aucs})
        df_null.to_csv(out_dir / f"feature_relevance_permnull_{tag}.csv", index=False)
        with open(out_dir / f"feature_relevance_permnull_{tag}_summary.json", "w") as f:
            json.dump({
                "max_fields": permutation_max_fields, "n_permutations": n_permutations_null,
                "observed_auc": obs_auc, "null_mean": float(np.nanmean(null_aucs)),
                "null_std": float(np.nanstd(null_aucs)), "null_max": float(np.nanmax(null_aucs)),
                "p_value": p_value, "n_valid": n_valid,
            }, f, indent=2)

    # --- Random field-selection test (optional): does WHICH fields matter? ---
    if n_random_field_draws > 0:
        print(f"[feature_relevance] {tag}: random field selection (n_fields={permutation_max_fields}, "
              f"n_draws={n_random_field_draws}) ...")
        greedy_at_n = _greedy_once(cache, y, C=C, max_fields=permutation_max_fields)
        greedy_auc_at_n = greedy_at_n[-1][1] if greedy_at_n else float("nan")
        random_aucs = random_field_selection_auc(
            cache, y,
            n_fields=permutation_max_fields, n_draws=n_random_field_draws, C=C,
        )
        if len(random_aucs):
            p_value_rand, n_valid_rand = _perm_p_value(greedy_auc_at_n, random_aucs)
            print(f"[feature_relevance]   {tag}: greedy AUC={greedy_auc_at_n:.4f}  "
                  f"random mean={np.nanmean(random_aucs):.4f}  std={np.nanstd(random_aucs):.4f}  "
                  f"max={np.nanmax(random_aucs):.4f}  frac>=0.80={float(np.nanmean(random_aucs >= 0.80)):.2f}  "
                  f"p={p_value_rand:.4f}")
        else:
            p_value_rand, n_valid_rand = float("nan"), 0
        df_rand = pd.DataFrame({"draw": range(len(random_aucs)), "random_field_auc": random_aucs})
        df_rand.to_csv(out_dir / f"feature_relevance_randomfields_{tag}.csv", index=False)
        with open(out_dir / f"feature_relevance_randomfields_{tag}_summary.json", "w") as f:
            json.dump({
                "n_fields": permutation_max_fields, "n_draws": n_random_field_draws,
                "greedy_auc": greedy_auc_at_n,
                "random_mean": float(np.nanmean(random_aucs)) if len(random_aucs) else float("nan"),
                "random_std": float(np.nanstd(random_aucs)) if len(random_aucs) else float("nan"),
                "random_min": float(np.nanmin(random_aucs)) if len(random_aucs) else float("nan"),
                "random_max": float(np.nanmax(random_aucs)) if len(random_aucs) else float("nan"),
                "frac_random_over_080": float(np.nanmean(random_aucs >= 0.80)) if len(random_aucs) else float("nan"),
                "p_value": p_value_rand, "n_valid": n_valid_rand,
            }, f, indent=2)

    # --- Bootstrap stability (optional) ---
    if n_bootstrap > 0:
        print(f"[feature_relevance] {tag}: bootstrap stability (n={n_bootstrap}, block={bootstrap_block_size}d) ...")
        df_stab, df_deltas = _bootstrap_greedy_stability(
            get_field_raw(), y,
            n_bootstrap=n_bootstrap,
            block_size=bootstrap_block_size,
            n_splits=min(n_splits, 3),
            cv_mode=cv_mode, cv_block_size=cv_block_size, C=C,
            max_modes_per_field=max_modes_per_field,
        )
        robust = df_stab[df_stab["selection_frequency"] >= 0.8]["field"].tolist()
        print(f"[feature_relevance] {tag}: robust features (≥80%): {robust}")
        for _, row in df_stab.iterrows():
            print(f"[feature_relevance]   {row['field']:15s}: "
                  f"freq={row['selection_frequency']:.2f}  "
                  f"mean_stage={row['mean_stage']:.1f}")
        df_stab.to_csv(out_dir / f"feature_relevance_bootstrap_{tag}_stability.csv", index=False)

        if not df_deltas.empty:
            df_deltas.to_csv(out_dir / f"feature_relevance_bootstrap_{tag}_deltas.csv", index=False)
            _plot_bootstrap_deltas(
                df_deltas,
                out_dir / f"feature_relevance_bootstrap_{tag}_deltas.png",
                region=region, bbox_step=bbox_step,
            )

    return sequential_results


def _plot_regime_comparison(
    all_regime_results: dict[int, list],
    out_path: Path,
    *,
    region: str,
    bbox_step: str,
) -> None:
    """One figure comparing sequential AUC curves across all Z500 regimes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors
    max_stages = max((len(v) for v in all_regime_results.values() if v), default=1)

    for i, (regime_id, seq_results) in enumerate(sorted(all_regime_results.items())):
        if not seq_results:
            continue
        aucs = [r[1] for r in seq_results]
        ax.plot(range(len(aucs)), aucs,
                marker="o", linewidth=2, markersize=7,
                color=colors[i % len(colors)],
                label=f"Z500-Regime {regime_id}")

    ax.axhline(0.80, color="#F57C00", linestyle="--", linewidth=1.5, label="80% threshold", zorder=2)
    ax.axhline(0.90, color="#C62828", linestyle="--", linewidth=1.5, label="90% threshold", zorder=2)
    ax.set_xlabel("Fields added sequentially (0 = z500 baseline)", fontsize=10)
    ax.set_ylabel("Cumulative AUC", fontsize=10)
    ax.set_ylim([0.5, 1.0])
    ax.set_xticks(range(max_stages))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"Regime-specific sequential feature selection — {region} {bbox_step}\n"
        "Each line = one Z500 major regime; diverging curves = regime-dependent importances",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[feature_relevance] Saved regime comparison plot: {out_path}")


def _auto_find_z500_regimes(
    module_root: Path,
    region: str,
    bbox_step: str,
    smooth_window: int = 11,
    *,
    use_yeo_johnson: bool = True,
    cluster_method: str = "kmeans",
) -> Path | None:
    """Resolve the Stage 1 Z500 regime labels file for a given region/bbox_step.

    Reads k from this region's ``k_overrides_<region>_custom.yaml`` — the same
    reviewed, single-source-of-truth file z_03_run_regime_clustering_tu_hpc.sh /
    z_04_run_regime_composites_tu_hpc.sh / z_05_run_regime_stats_csv_tu_hpc.sh use to drive
    KMeans(n_clusters=...) and to resolve its output — rather than picking
    whatever k happens to have the most cluster files on disk. A stray
    higher-k exploratory clustering run must never silently outrank the
    reviewed production k for a bbox. No fallbacks or proxies — returns None
    if the k-overrides YAML or a matching regime file isn't found.
    """
    from common.resolver import (
        PathSpec,
        filter_candidates_by_yeo_johnson,
        glob_regimes_candidates,
        k_for_step,
        load_k_overrides,
        narrow_paths_by_step_token,
    )

    k_config_defaults = {
        "europe": module_root / "src" / "common" / "k_overrides_europe_custom.yaml",
        "northamerica": module_root / "src" / "common" / "k_overrides_northamerica_custom.yaml",
    }
    k_config_path = k_config_defaults.get(
        region, module_root / "src" / "common" / "k_overrides.yaml"
    )
    if not k_config_path.is_file():
        print(f"[feature_relevance] ERROR: k-overrides YAML not found: {k_config_path}\n"
              f"[feature_relevance] This must be the same file z_03/z_04 use to drive "
              f"KMeans for this region. Pass --z500-regimes-path explicitly instead.")
        return None

    k_cfg = load_k_overrides(k_config_path)
    k = k_for_step(k_cfg, bbox_step)

    spec = PathSpec(window_days=int(smooth_window), epsilon_m=20.0, var_threshold=0.90, year_span="1979-2025")
    candidates = glob_regimes_candidates(module_root, region, spec, k, bbox_step=bbox_step, method=cluster_method)
    # glob_regimes_candidates() includes a loose "any k" fallback pattern (for files
    # that don't carry the usual w/eps/var tokens) — re-assert the exact k here so a
    # stray exploratory run at a different k can never leak into the pool.
    k_tag = f"_k{k:02d}_"
    candidates = [p for p in candidates if k_tag in p.name]
    candidates = narrow_paths_by_step_token(candidates, bbox_step)
    candidates = filter_candidates_by_yeo_johnson(candidates, use_yeo_johnson)

    if not candidates:
        print(f"[feature_relevance] ERROR: no Z500 regime file found for {region}/{bbox_step} "
              f"at k={k} method={cluster_method} (from {k_config_path.name}) under "
              f"{module_root / 'results' / 'clustering' / region}\n"
              f"[feature_relevance] Run the clustering pipeline for this bbox/k first, "
              f"or pass --z500-regimes-path explicitly.")
        return None
    if len(candidates) > 1:
        print(f"[feature_relevance] WARNING: {len(candidates)} Z500 regime candidates for "
              f"{region}/{bbox_step} at k={k} (expected exactly one) — using newest by mtime: "
              + ", ".join(p.name for p in candidates))
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)

    chosen = candidates[-1]
    print(f"[feature_relevance] Resolved Z500 regimes ({region}/{bbox_step}, k={k} from "
          f"{k_config_path.name}): {chosen.name}")
    return chosen


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    from common.extremes_spec import add_extremes_spec_arguments

    p = argparse.ArgumentParser(description="Feature relevance diagnostic for residual regime pipeline.")
    p.add_argument("--region", required=True, choices=("europe", "northamerica"))
    p.add_argument("--bbox-step", default="B0")
    p.add_argument(
        "--extreme-labels-path",
        type=Path,
        default=None,
        help="LEGACY: explicit path to epe_01b convective_duration NetCDF. "
             "Prefer --extremes-q / --extremes-cape-threshold / --extremes-cp-frac-min.",
    )
    p.add_argument("--extreme-var", default="flagged_any_gridcell",
                   help="Variable name for the binary extreme label (default: flagged_any_gridcell).")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--fields", default="z500,vo850,tcwv,deg0l,cape,w700,rh700",
                   help="Comma-separated field keys to test.")
    p.add_argument("--module-root", type=Path, default=None)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=0,
        metavar="N",
        help="Number of block-bootstrap runs for stability selection (0 = disabled). "
             "Recommended: 50–100. Each run re-runs the greedy selection on a "
             "block-resampled time series. Outputs selection_frequency per feature, "
             "which is the key statistic for robust regional comparison. "
             "Uses 3-fold CV inside each bootstrap run for speed.",
    )
    p.add_argument(
        "--bootstrap-block-size",
        type=int,
        default=10,
        metavar="DAYS",
        help="Block length (days) for block bootstrap (default 10). "
             "Should be ≥ autocorrelation length of the daily time series (~5–14 days "
             "for synoptic weather patterns).",
    )
    p.add_argument("--study-area", default=None,
                   help="Processed-data folder (germany, ontario_south). Auto-derived from --region if omitted.")
    p.add_argument("--mode", choices=("bbox", "regime"), default="bbox",
                   help="Feature selection mode. "
                        "'bbox' (default): global selection across all days for this bbox — "
                        "the original approach, one field set per bbox. "
                        "'regime': run selection separately for each Z500 major regime within the bbox — "
                        "requires --z500-regimes-path or auto-resolves from results/clustering/.")
    p.add_argument("--z500-regimes-path", type=Path, default=None,
                   help="Explicit path to Stage 1 Z500 k-means regime labels NetCDF "
                        "(only used when --mode=regime). "
                        "If omitted with --mode=regime, auto-resolved from results/clustering/<region>/.")
    p.add_argument("--z500-smooth-window", type=int, default=11,
                   help="Smoothing window to prefer when auto-resolving the Z500 regime file "
                        "(default 11 = pipeline standard w11). Only used when --mode=regime "
                        "and --z500-regimes-path is not set.")
    p.add_argument("--cluster-method", choices=("kmeans", "ward"), default="kmeans",
                   help="On-disk Z500 regimes filename family to auto-resolve: 'kmeans' (default; "
                        "also covers the Ward-seeded K-means hybrid) or 'ward' (pure Ward HRC, "
                        "z_03's CLUSTER_MODE=ward). Only used when --mode=regime and "
                        "--z500-regimes-path is not set.")
    p.add_argument("--tag-suffix", default="",
                   help="Extra suffix appended to output filenames after the mode tag "
                        "(e.g. 'B0k7' when evaluating this bbox against another bbox's "
                        "regime labels via --z500-regimes-path). Leave unset for canonical "
                        "bbox/regime runs — resolve_rr01_csv/write_field_selection.py never "
                        "look for suffixed files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and print the extremes NetCDF (EPE definition) and, for "
                        "--mode=regime, the Z500 regime labels NetCDF (with its k, sourced "
                        "from k_overrides_<region>_custom.yaml) — then exit without running "
                        "the (expensive) feature relevance computation.")
    p.add_argument(
        "--area-weight-tag",
        default="",
        help="Must match the rr_00 run's --area-weight-tag (e.g. 'aw2') to resolve that "
             "rerun's per-field PCA files instead of the default untagged ones (pinned "
             "exactly by resolve_mf_pca_path, not a fuzzy glob) — has no effect on Z500, "
             "which lives outside the rr_00 area-weighting rerun.",
    )
    p.add_argument(
        "--max-modes-per-field",
        type=int,
        default=DEFAULT_MAX_MODES_PER_FIELD,
        help="Truncate every candidate field's PCA block to its leading N modes (default "
             f"{DEFAULT_MAX_MODES_PER_FIELD}, matching MultivariateRunConfig.max_modes_per_field "
             "— the same cap rr_00 itself applies when building these PCA files). Pass 0 or "
             "a negative value to disable truncation entirely. Has no effect on Z500 (domain-mean, "
             "no PCA block). See module docstring's 2026-08 note on why domain-mean scoring "
             "was replaced with this.",
    )
    p.add_argument(
        "--field-representation",
        choices=["pca_block", "domain_mean"],
        default="domain_mean",
        help="How every non-Z500 candidate field is scored: 'domain_mean' (default, "
             "production) uses the same single area-weighted spatial-average column Z500 "
             "always uses, no PCA; 'pca_block' uses its full per-fold-refit PCA block. "
             "Always explicitly tagged in the output filename ('_domainmean'/'_pcablock', "
             "see rr01_paths.rr01_tag) so the two representations can never silently "
             "overwrite each other's CSVs — a prior version left this untagged, so whichever "
             "representation last ran for a given region/bbox/mode silently became what "
             "every downstream consumer (write_field_selection.py, rr_02/rr_022's "
             "--build-regime-map) read, regardless of which one was actually intended as "
             "production. A controlled A/B (2026-08-19): everything else (blocked CV, "
             "per-fold PCA refit, mode capping) stays identical, isolating whether AUC "
             "differences trace to the representation switch itself rather than a leakage "
             "artifact — see module docstring's 'Per-fold PCA refit' entry for the "
             "investigation this was added to settle.",
    )
    p.add_argument("--no-yeo-johnson", dest="use_yeo_johnson", action="store_false", default=True,
                   help="Resolve the RAW (non-yj) anomaly files instead of the pipeline's "
                        "default _yj_ ones (default: use _yj_, matching the pipeline-wide "
                        "YEO_JOHNSON default — Steps 02-07/PCA assume roughly Gaussian input, "
                        "so every field including Z500 goes through the same YJ-transformed "
                        "representation for consistency, and rr_01 should rank fields on that "
                        "same representation rather than a differently-preprocessed one). "
                        "Use only for an explicit raw-vs-yj comparison diagnostic.")
    p.add_argument(
        "--cv-mode",
        choices=("random", "blocked"),
        default="random",
        help="'random' (default, unchanged pre-2026-08-17 behaviour): StratifiedKFold(shuffle=True) "
             "— per-day shuffled folds. 'blocked': contiguous --cv-block-size-day blocks assigned "
             "to folds instead (same recipe as rr_021_stage2_LogRegr_Robustness.py's blocked_cv_auc) "
             "— guards against a day's autocorrelated neighbour landing in the opposite fold, which "
             "matters much more now that each field contributes its full multi-mode PCA block "
             "instead of a single domain-mean column. Try 'blocked' and compare against 'random' "
             "on the same bbox before trusting a cumulative_auc that only appeared after the "
             "2026-08-14 PCA-block change.",
    )
    p.add_argument(
        "--cv-block-size",
        type=int,
        default=10,
        metavar="DAYS",
        help="Block length (days) for --cv-mode=blocked (default 10, matching this pipeline's "
             "synoptic-autocorrelation convention elsewhere — see --bootstrap-block-size). "
             "Ignored when --cv-mode=random.",
    )
    p.add_argument(
        "--c",
        dest="C",
        type=float,
        default=1.0,
        metavar="C",
        help="LogisticRegression's inverse regularization strength (default 1.0, sklearn's own "
             "default — unchanged unless passed explicitly). Lower C = stronger L2 shrinkage. "
             "Meaningful use: sweep a few values (e.g. 0.1, 0.3, 1.0, 3.0) on the SAME --cv-mode "
             "and compare cumulative_auc — don't just pick one value by guesswork; look for where "
             "the curve stops changing (rr_021_stage2_LogRegr_Robustness.py's c_sweep_auc does "
             "exactly this sweep for the fixed production model).",
    )
    p.add_argument(
        "--n-permutations-null",
        type=int,
        default=0,
        metavar="N",
        help="If > 0, after Option B (bbox mode: once globally; regime mode: once per Z500 "
             "regime) runs a label-shuffled permutation-null "
             "test on the greedy search ITSELF (not just the fixed production model) capped "
             "at --permutation-max-fields, to check whether the greedy wrapper search's own "
             "selection bias (picking the best of up to ~27 candidate PCA blocks each stage, "
             "scored on the same data reported) can by itself produce a high cumulative_auc "
             "with no real X-y relationship. Expensive: each permutation reruns the whole "
             "capped greedy search (~permutation_max_fields * n_fields _cv_auc calls). "
             "Default 0 = disabled.",
    )
    p.add_argument(
        "--permutation-max-fields",
        type=int,
        default=3,
        metavar="N",
        help="Cap the permutation-null test's greedy search (both observed and each null "
             "draw) at this many non-Z500 fields (default 3, matching the paper's plotting "
             "convention of reading cumulative_auc after 3 fields for cross-bbox "
             "comparability). Ignored unless --n-permutations-null > 0.",
    )
    p.add_argument(
        "--report-in-sample-auc",
        action="store_true",
        default=False,
        help="Add an in_sample_auc column to the sequential_optionB CSV: the SAME greedy "
             "field order re-fit on ALL days with no held-out CV fold, scored on those same "
             "days. Not a generalization estimate on its own -- always >= cumulative_auc at "
             "the same row. The gap between the two columns at a given field count is a "
             "cheap overfitting readout: small gap means cumulative_auc's held-out number "
             "reflects real generalization; a large gap even under --cv-mode blocked and a "
             "lowered --c means the model is still fitting something that doesn't hold up "
             "out of sample. Default off (adds one extra LogisticRegression fit per stage).",
    )
    p.add_argument(
        "--n-random-field-draws",
        type=int,
        default=0,
        metavar="N",
        help="If > 0, tests whether the GREEDY SELECTION itself matters: draws N random "
             "combinations of --permutation-max-fields non-Z500 fields (uniform, no "
             "replacement) on the REAL (unshuffled) label, and scores each with the same "
             "cv_mode/block_size/C recipe as the greedy search. If random field triples "
             "reliably score close to the greedily-selected cumulative_auc, that means "
             "almost ANY combination of fields reaches a high score -- the jump from "
             "Z500-only to Z500+N-fields is coming from something common to nearly every "
             "candidate field, not genuine field-specific information. Reuses "
             "--permutation-max-fields for N (default 3). Default 0 = disabled.",
    )
    add_extremes_spec_arguments(p)
    # rr_01 production defaults (override via CLI or run_configs.yaml on HPC)
    p.set_defaults(
        extremes_q=0.999,
        quantile_sample="all_hours",
        extremes_wet_threshold_mm=0.1,
        extremes_cape_threshold=0.0,
        extremes_cp_frac_min=0.01,
        extremes_year_span="1979-2025",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    import sys
    from common.extremes_spec import ExtremesSpec, resolve_convective_duration_path

    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    # resolve module root
    if args.module_root is not None:
        module_root = args.module_root.expanduser().resolve()
    else:
        here = Path(__file__).resolve()
        # src/stage2_residual/feature_relevance.py → go up to repo root
        module_root = here.parents[2]

    out_dir = args.out_dir or (module_root / "results" / "stage2_residual" / "feature_relevance" / args.region)
    fields = [f.strip().lower() for f in args.fields.split(",") if f.strip()]

    extremes_spec = ExtremesSpec(
        q=float(args.extremes_q),
        quantile_sample=str(args.quantile_sample),
        wet_threshold_mm=float(args.extremes_wet_threshold_mm),
        cape_threshold=float(args.extremes_cape_threshold),
        cp_frac_min=float(args.extremes_cp_frac_min),
        year_span=str(args.extremes_year_span).strip(),
    )
    try:
        extreme_labels_path = resolve_convective_duration_path(
            module_root,
            args.region,
            study_area=args.study_area,
            spec=extremes_spec,
            legacy_path=args.extreme_labels_path,
        )
    except FileNotFoundError as exc:
        print(f"[feature_relevance] ERROR: {exc}")
        return 1

    if args.mode == "regime":
        if args.z500_regimes_path is not None:
            z500_regimes_path = args.z500_regimes_path.expanduser().resolve()
        else:
            z500_regimes_path = _auto_find_z500_regimes(
                module_root, args.region, args.bbox_step,
                smooth_window=args.z500_smooth_window,
                use_yeo_johnson=args.use_yeo_johnson,
                cluster_method=args.cluster_method,
            )
            if z500_regimes_path is None:
                print(f"[feature_relevance] ERROR: --mode=regime requested but no regime file found. "
                      f"Pass --z500-regimes-path explicitly or run the clustering pipeline first.")
                return 1
    else:
        z500_regimes_path = None

    if args.dry_run:
        print("[feature_relevance] --- DRY RUN (no computation performed) ---")
        print(f"[feature_relevance] region={args.region}  bbox_step={args.bbox_step}  mode={args.mode}")
        print(f"[feature_relevance] EPE definition: {extremes_spec.summary_label()}")
        print(f"[feature_relevance] Extremes NetCDF: {extreme_labels_path}")
        if args.mode == "regime":
            print(f"[feature_relevance] Z500 regimes NetCDF (yeo_johnson={args.use_yeo_johnson}): "
                  f"{z500_regimes_path}")
        print("[feature_relevance] --- DRY RUN OK ---")
        return 0

    max_modes_per_field = args.max_modes_per_field if args.max_modes_per_field and args.max_modes_per_field > 0 else None

    run_feature_relevance(
        module_root=module_root,
        region=args.region,
        bbox_step=args.bbox_step,
        extreme_labels_path=extreme_labels_path,
        extreme_var=args.extreme_var,
        out_dir=out_dir,
        fields=fields,
        n_splits=args.n_splits,
        z500_regimes_path=z500_regimes_path,
        mode=args.mode,
        smooth_window=args.z500_smooth_window,
        n_bootstrap=args.n_bootstrap,
        bootstrap_block_size=args.bootstrap_block_size,
        tag_suffix=args.tag_suffix,
        use_yeo_johnson=args.use_yeo_johnson,
        z500_cluster_method=args.cluster_method,
        area_weight_tag=args.area_weight_tag,
        max_modes_per_field=max_modes_per_field,
        cv_mode=args.cv_mode,
        cv_block_size=args.cv_block_size,
        C=args.C,
        n_permutations_null=args.n_permutations_null,
        permutation_max_fields=args.permutation_max_fields,
        report_in_sample_auc=args.report_in_sample_auc,
        n_random_field_draws=args.n_random_field_draws,
        field_representation=args.field_representation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
