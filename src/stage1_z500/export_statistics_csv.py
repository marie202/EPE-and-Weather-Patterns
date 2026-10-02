#!/usr/bin/env python3
"""
Export Z500 regime + EPE.4 statistics to CSV — no notebook, no plotting.

CSV counterpart to ``the removed Z500 statistics plotting script``: same load calls
(``epe_04_boxplot_render.load_epe_04_statistics_inputs`` +
``compute_epe_04_core_metrics``), but writes the underlying numbers instead of
rendering figures. Meant to run wherever the source NetCDF lives (HPC or a
local mount with ``results/clustering/<region>/`` and
``data/processed/<study_area>/extreme_precip_events/`` populated) — the
one-time NetCDF read is the part that can exhaust memory in a notebook
kernel. ``2_3_Results_Z500_Stats.ipynb`` then only reads the small CSVs this
script writes and rebuilds the figures locally with pandas + matplotlib.

Per (region, bbox_step, k, cluster_method) this writes rows into six CSVs (one file per
statistic, covering every bbox_step/k found for the region):

  - regime_frequency_<region>.csv          Z500 regime frequency, all aligned
                                            days (% and day count).
  - regime_frequency_monthly_<region>.csv  same, split by MJJAS month.
  - epe_occurrence_<region>.csv            EPE occurrence: regime frequency
                                            among extreme/event days only.
  - epe_occurrence_monthly_<region>.csv    same, split by MJJAS month.
  - epe_boxplot_values_<region>.csv        long format (metric, regime,
                                            value) — one row per day, for
                                            persistence/intensity/connectivity
                                            boxplots.
  - epe_boxplot_meta_<region>.csv          (metric, title, ylabel) per
                                            bbox_step/k, for axis labels.

Also writes/updates a single shared ``PROVENANCE.json`` (one record per (region, bbox_step,
cluster_method), same merge-by-key behavior as the CSVs above) recording exactly which
extreme-day definition (domain_pctl_q, domain_agg, q, ...) and which source NetCDF filename
produced each combo currently on disk -- 2_3_Results_Z500_Stats.ipynb reads this to confirm
what it's plotting instead of trusting an assumed default.

--cluster-method (default kmeans) selects which Step-9-equivalent regime-labels file drives a
given call: kmeans = 09_Z500_regimes_kmeans_*.nc, ward = the Step-8 Ward HRC cut at k, saved as
09_Z500_regimes_ward_*.nc (identical regime_labels/frequency schema to the kmeans file, so no
other code path changes). Every row carries a cluster_method column, and re-running for one
method only replaces that method's rows in each CSV — the other method's rows already on disk
are preserved. Run once per method to get both kmeans and ward rows into the same CSVs, e.g.:
    python3 steps/stage1/04_export_statistics_csv.py --region europe northamerica
    python3 steps/stage1/04_export_statistics_csv.py --region europe northamerica --cluster-method ward

Usage:
    python3 steps/stage1/04_export_statistics_csv.py --region europe
    python3 steps/stage1/04_export_statistics_csv.py --region europe --bbox-step B5 --k 7
    python3 steps/stage1/04_export_statistics_csv.py --region europe northamerica
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Must be set before netCDF4/HDF5 is loaded (i.e. before matplotlib/xarray below pull it in).
# compute_epe_04_core_metrics reopens the convective_duration_*.nc file more than once (once per
# EPE.4 metric) — on this Nextcloud-synced project folder that segfaults inside HDF5's
# file-locking path (confirmed: "Fatal Python error: Segmentation fault" in
# xarray/backends/file_manager.py, reproduced and fixed 2026-07-27 in the sibling
# plot_regime_z500_statistics.py). Disabling locking is the standard fix and is safe here since
# every open is read-only, single-process.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib

matplotlib.use("Agg")  # epe_04_boxplot_render imports pyplot; no display in a batch/background run
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from epe_preprocessing.epe_functions import _daily_extreme_mask
from common.preprocessing_helpers import filter_paths_by_yeo_johnson, find_module_root
from common.extremes_spec import ExtremesSpec, add_extremes_spec_arguments, extremes_spec_from_args

_MJJAS_MONTHS = (5, 6, 7, 8, 9)
_MONTH_NAMES = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
_STUDY_AREA_BY_REGION = {"europe": "germany", "northamerica": "ontario_south"}

# Precip/extreme-day filter used by every CSV this script writes, unless overridden by a
# --domain-pctl-q/--extremes-* CLI flag (see add_extremes_spec_arguments). Kept as an explicit,
# visible module constant here -- not just extremes_spec.py's internal
# default_domain_lexmax_extremes_spec() default -- so the choice governing every number in these
# CSVs is obvious at a glance and doesn't require chasing an import to find.  Production pinned
# at domain_pctl_q=0.90 (lexmax domain aggregation) on 2026-08-03; see README_DOMAIN_PCTL_Q.md.
DEFAULT_DOMAIN_PCTL_Q = 0.90
DEFAULT_DOMAIN_AGG = "lexmax"


def _discover_bbox_k(module_root: Path, region: str, cluster_method: str) -> dict[str, int]:
    """Bbox_step -> k pairs found among local ``09_Z500_regimes_<cluster_method>_*_step*.nc`` files.

    Groups candidates per bbox step first, then prefers YJ-tagged files (current preprocessing
    convention) and the newest file among ties — otherwise a stale pre-YJ rerun at a higher k
    (e.g. an old k07 file sitting alongside the current k04 YJ rerun) could win just by being
    the last one the directory glob happened to yield.
    """
    clustering_dir = module_root / "results" / "clustering" / region
    pattern = re.compile(rf"09_Z500_regimes_{re.escape(cluster_method)}_k(\d+)_.*_step(B\d+)_")
    by_step: dict[str, list[Path]] = {}
    for f in clustering_dir.glob(f"09_Z500_regimes_{cluster_method}_k*_step*.nc"):
        m = pattern.search(f.name)
        if m:
            by_step.setdefault(m.group(2), []).append(f)
    found: dict[str, int] = {}
    for step, files in by_step.items():
        files = filter_paths_by_yeo_johnson(sorted(files, key=lambda p: p.stat().st_mtime))
        m = pattern.search(files[-1].name)
        if m:
            found[step] = int(m.group(1))
    return dict(sorted(found.items(), key=lambda kv: int(kv[0][1:])))


def _resolve_regimes_nc(module_root: Path, region: str, bbox_step: str, k: int, cluster_method: str) -> Path:
    """Same Step-9 file ``plot_regime_z500_statistics.py`` resolves (ward: cut from Step-8 HC at k;
    kmeans: the final Step-9 K-means labels) — both carry identical regime_labels/frequency schema."""
    clustering_dir = module_root / "results" / "clustering" / region
    pattern = f"09_Z500_regimes_{cluster_method}_k{k:02d}_*_step{bbox_step}_*.nc"
    hits = sorted(clustering_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No regimes NetCDF matching {pattern!r} in {clustering_dir}")
    # Disambiguate YJ vs pre-YJ Step-9 files at the same k/w/eps/var/bbox — same fix as
    # plot_regime_z500_precip_composites.py's label resolver; without it, an old pre-YJ
    # rerun at the same k can sort last and silently override the current YJ convention.
    hits = filter_paths_by_yeo_johnson(hits)
    return hits[-1]


def _day_counts_by_regime(regime_labels: np.ndarray) -> pd.DataFrame:
    """One row per regime: n_days, pct_days (of every non-zero-labelled row passed in)."""
    regime_labels = np.asarray(regime_labels, dtype=int)
    regime_labels = regime_labels[regime_labels > 0]
    total = int(regime_labels.size)
    regimes, counts = np.unique(regime_labels, return_counts=True)
    df = pd.DataFrame({"regime": regimes.astype(int), "n_days": counts.astype(int)})
    df["n_days_total"] = total
    df["pct_days"] = 100.0 * df["n_days"] / max(total, 1)
    return df.sort_values("regime").reset_index(drop=True)


def _monthly_counts_by_regime(days: np.ndarray, regime_labels: np.ndarray) -> pd.DataFrame:
    """One row per (regime, MJJAS month): n_days, pct_within_regime, pct_of_all_days."""
    months = pd.DatetimeIndex(np.asarray(days).astype("datetime64[ns]")).month.to_numpy()
    regime_labels = np.asarray(regime_labels, dtype=int)
    keep = (regime_labels > 0) & np.isin(months, _MJJAS_MONTHS)
    months = months[keep]
    regimes_arr = regime_labels[keep]
    total_all = int(keep.sum())

    regimes = np.unique(regimes_arr)
    full_index = pd.MultiIndex.from_product([regimes, _MJJAS_MONTHS], names=["regime", "month"])
    counts = (
        pd.DataFrame({"regime": regimes_arr, "month": months})
        .groupby(["regime", "month"])
        .size()
        .rename("n_days")
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    regime_totals = counts.groupby("regime")["n_days"].transform("sum")
    counts["pct_within_regime"] = (100.0 * counts["n_days"] / regime_totals.replace(0, np.nan)).fillna(0.0)
    counts["pct_of_all_days"] = 100.0 * counts["n_days"] / max(total_all, 1)
    counts["month_name"] = counts["month"].map(_MONTH_NAMES)
    return counts.sort_values(["regime", "month"]).reset_index(drop=True)


def _flatten_core_metrics(core_metrics: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_rows: list[dict] = []
    meta_rows: list[dict] = []
    for metric, m in core_metrics.items():
        meta_rows.append({"metric": metric, "title": m["title"], "ylabel": m["ylabel"]})
        for reg_id, vals in zip(m["regime_ids"], m["data"], strict=False):
            for v in np.asarray(vals, dtype=float).ravel():
                value_rows.append({"metric": metric, "regime": int(reg_id), "value": float(v)})
    values_df = pd.DataFrame(value_rows, columns=["metric", "regime", "value"])
    meta_df = pd.DataFrame(meta_rows, columns=["metric", "title", "ylabel"])
    return values_df, meta_df


def export_one_bbox_step(
    module_root: Path, region: str, bbox_step: str, k: int, study_area: str, cluster_method: str,
    extremes_spec: ExtremesSpec | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    import common.epe_04_boxplot_render as epe_04_boxplot_render

    p_reg = _resolve_regimes_nc(module_root, region, bbox_step, k, cluster_method)
    print(f"[export] {region} {bbox_step} k={k} cluster_method={cluster_method} regimes from: {p_reg.name}")

    stats_inputs = epe_04_boxplot_render.load_epe_04_statistics_inputs(
        module_root, region, regimes_nc_path=p_reg, study_area=study_area, extremes_spec=extremes_spec,
    )
    reg_day_ds = stats_inputs["reg_day_ds"]
    ext_c = stats_inputs["ext_c"]
    labels_ext = stats_inputs["labels_ext"]
    print(f"[export] {region} {bbox_step} source: {stats_inputs['source_msg']}")

    core_metrics = epe_04_boxplot_render.compute_epe_04_core_metrics(
        ext_c, labels_ext, reg_day_ds,
        module_root=module_root, region=region, study_area=study_area,
        extremes_spec=stats_inputs["extremes_spec"], regimes=stats_inputs["regimes"],
    )

    # Excess-over-threshold (POT) severity metric -- unlike raw amount, doesn't converge
    # across regimes at high percentiles. Lives in stage2_residual.explore_occurrence_excess
    # (not epe_04_boxplot_render) since it's a newer, separately-developed metric; merged in
    # here so it flows through the same CSV/notebook path as every other boxplot metric.
    from stage2_residual.explore_occurrence_excess import compute_z500_excess_core_metrics

    core_metrics.update(compute_z500_excess_core_metrics(
        ext_c, labels_ext, stats_inputs["regimes"],
        reg_day_ds=reg_day_ds, duration_audit_path=stats_inputs.get("duration_audit_path"),
    ))

    # All-days precip amount (mean/median) -- the non-circular baseline: EVERY MJJAS day in
    # reg_day_ds (not just extreme/flagged-cell days), joined to the fixed-AOI daily total precip
    # field written by epe_01_aggregate_daily.py *before* any extreme-day selection happens. See
    # common.alldays_precip's module docstring for why this exists alongside (not
    # instead of) the domain-pctl-0.90 EPE occurrence table above and the cell-flagged
    # duration/intensity/spatial-precip/excess metrics -- those stay on their existing samples
    # since they're inherently threshold-defined and have no non-arbitrary all-days analogue.
    from common.alldays_precip import load_daily_tp_alldays

    tp_alldays = load_daily_tp_alldays(module_root, region, study_area=study_area)
    reg_days = pd.DatetimeIndex(reg_day_ds["day"].values).normalize()
    alldays_df = pd.DataFrame(
        {"regime": reg_day_ds["regime_labels"].values.astype(int)}, index=reg_days,
    ).join(tp_alldays, how="inner")
    alldays_df = alldays_df[alldays_df["regime"] > 0]
    alldays_regs = sorted(alldays_df["regime"].unique())
    for metric_key, col, agg_name in (
        ("precip_alldays_mean", "tp_mean_mm", "Mean"),
        ("precip_alldays_median", "tp_median_mm", "Median"),
    ):
        core_metrics[metric_key] = {
            "data": [alldays_df.loc[alldays_df["regime"] == r, col].to_numpy() for r in alldays_regs],
            "ticks": [str(r) for r in alldays_regs],
            "regime_ids": alldays_regs,
            "title": f"{agg_name} daily precip by regime (ALL MJJAS days, no extreme filter)",
            "ylabel": f"domain-{'mean' if agg_name == 'Mean' else 'median'} tp_daily_sum, all days (mm)",
        }

    # `ext_c`/`labels_ext` are aligned to EVERY day in the extremes/duration source (not just
    # flagged extreme days) -- `_align_extremes_to_regimes` only intersects calendar days, it
    # never subsets by the domain-extreme flag. Without this mask, "EPE occurrence" silently
    # degenerated into a second copy of the all-days "regime frequency" table (confirmed: byte-
    # identical output for every bbox_step/k/region), regardless of which domain_pctl_q was
    # configured. Build the same day mask compute_epe_04_core_metrics uses internally for the
    # boxplot metrics, and apply it here too before counting occurrence.
    extreme_mask_da = _daily_extreme_mask(extremes_ds=ext_c, duration_ds=None)
    if extreme_mask_da is None:
        raise RuntimeError(
            f"{region} {bbox_step} k={k}: no domain-extreme flag (e.g. 'domain_extreme') found "
            "in the aligned extremes dataset -- cannot compute EPE occurrence (extreme-days-only) stats."
        )
    extreme_mask = np.asarray(extreme_mask_da.values, dtype=bool)
    n_flagged = int(extreme_mask.sum())
    print(f"[export] {region} {bbox_step} k={k} EPE occurrence: {n_flagged}/{extreme_mask.size} days flagged extreme")

    freq = _day_counts_by_regime(reg_day_ds["regime_labels"].values)
    freq_month = _monthly_counts_by_regime(reg_day_ds["day"].values, reg_day_ds["regime_labels"].values)
    occ = _day_counts_by_regime(labels_ext.values[extreme_mask])
    occ_month = _monthly_counts_by_regime(ext_c["day"].values[extreme_mask], labels_ext.values[extreme_mask])
    values_df, meta_df = _flatten_core_metrics(core_metrics)

    tables = {
        "regime_frequency": freq,
        "regime_frequency_monthly": freq_month,
        "epe_occurrence": occ,
        "epe_occurrence_monthly": occ_month,
        "epe_boxplot_values": values_df,
        "epe_boxplot_meta": meta_df,
    }
    for df in tables.values():
        df.insert(0, "k", k)
        df.insert(0, "cluster_method", cluster_method)
        df.insert(0, "bbox_step", bbox_step)
        df.insert(0, "region", region)

    # Resolved spec (as echoed back by load_epe_04_statistics_inputs) plus the actual filename it
    # matched against on disk -- this is the one piece of evidence that proves which extreme-day
    # definition really drove these numbers, independent of what CLI flag/default was intended.
    resolved_spec = stats_inputs.get("extremes_spec") or extremes_spec
    provenance = {
        "region": region, "bbox_step": bbox_step, "k": k, "cluster_method": cluster_method,
        **(asdict(resolved_spec) if resolved_spec is not None else {}),
        "regimes_source": p_reg.name,
        "extremes_source": stats_inputs.get("source_msg", ""),
        "n_extreme_days": n_flagged,
        "n_total_days": int(extreme_mask.size),
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    provenance["duration_nc"] = str(provenance.get("duration_nc")) if provenance.get("duration_nc") else None
    return tables, provenance


def export_region(
    module_root: Path, region: str, *,
    bbox_step: str | None, k: int | None, study_area: str | None, out_dir: Path, cluster_method: str,
    extremes_spec: ExtremesSpec | None = None,
) -> None:
    study_area = study_area or _STUDY_AREA_BY_REGION.get(region)
    if not study_area:
        raise ValueError(f"No default study_area for region={region!r}; pass --study-area.")

    if bbox_step:
        if k is None:
            discovered = _discover_bbox_k(module_root, region, cluster_method)
            if bbox_step not in discovered:
                raise KeyError(
                    f"bbox_step {bbox_step!r} not found under results/clustering/{region}/ for "
                    f"cluster_method={cluster_method!r}; pass --k explicitly."
                )
            k = discovered[bbox_step]
        bbox_k = {bbox_step: k}
    else:
        bbox_k = _discover_bbox_k(module_root, region, cluster_method)
        if not bbox_k:
            print(
                f"[export] {region}: no bbox-tagged 09_Z500_regimes_{cluster_method}_*.nc found under "
                f"results/clustering/{region}/, skipping."
            )
            return

    accum: dict[str, list[pd.DataFrame]] = {
        "regime_frequency": [], "regime_frequency_monthly": [],
        "epe_occurrence": [], "epe_occurrence_monthly": [],
        "epe_boxplot_values": [], "epe_boxplot_meta": [],
    }
    provenance_records: list[dict] = []
    for step, kk in bbox_k.items():
        tables, provenance = export_one_bbox_step(
            module_root, region, step, kk, study_area, cluster_method, extremes_spec=extremes_spec,
        )
        for key, df in tables.items():
            accum[key].append(df)
        provenance_records.append(provenance)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Keyed on (region, bbox_step, cluster_method) -- NOT k -- so that re-running a bbox_step at
    # a new k (e.g. B0 going from k=6 to k=7 after a clustering rerun) evicts the old k's rows
    # instead of leaving them alongside the new ones. Keying on the exact (bbox_step, k,
    # cluster_method) tuple used to let stale k rows accumulate forever for any bbox_step whose
    # k changed between export runs, silently duplicating every downstream figure.
    written_step_keys = {(region, step, cluster_method) for step in bbox_k}
    for key, frames in accum.items():
        path = out_dir / f"{key}_{region}.csv"
        if not frames:
            print(f"[export] (nothing to write) {path.name}")
            continue
        new_df = pd.concat(frames, ignore_index=True)
        # Merge with whatever's already on disk for this region (e.g. a prior kmeans export),
        # replacing only the (bbox_step, cluster_method) combos this run just recomputed — so
        # running kmeans then ward (or re-running one bbox) doesn't clobber the other's rows.
        if path.is_file():
            old_df = pd.read_csv(path)
            old_step_keys = old_df[["region", "bbox_step", "cluster_method"]].apply(tuple, axis=1)
            old_df = old_df[~old_step_keys.isin(written_step_keys)]
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)
        print(f"[export] wrote {path} ({len(new_df)} rows)")

    # Sidecar recording exactly which extreme-day definition (domain_pctl_q/domain_agg/...) and
    # which source NetCDF produced each (region, bbox_step, cluster_method) combo currently on
    # disk -- the CSVs themselves carry none of this, so without it there's no way, short of
    # re-grepping an HPC job log, to confirm what a given stats_csv/ export actually used. Merged
    # with the same (region, bbox_step, cluster_method) eviction key as the CSVs above, so this
    # stays in sync with them.
    provenance_path = out_dir / "PROVENANCE.json"
    all_records: list[dict] = []
    if provenance_path.is_file():
        try:
            existing = json.loads(provenance_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []
        for rec in existing:
            key = (rec.get("region"), rec.get("bbox_step"), rec.get("cluster_method"))
            if key not in written_step_keys:
                all_records.append(rec)
    all_records.extend(provenance_records)
    all_records.sort(key=lambda r: (r.get("region", ""), r.get("cluster_method", ""), r.get("bbox_step", "")))
    provenance_path.write_text(json.dumps(all_records, indent=2))
    print(f"[export] wrote {provenance_path} ({len(all_records)} records)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", nargs="+", choices=("europe", "northamerica"), required=True)
    ap.add_argument("--bbox-step", default=None, help="Single bbox step (e.g. B5); default: every step found on disk.")
    ap.add_argument("--k", type=int, default=None, help="Only used together with --bbox-step to skip auto-discovery.")
    ap.add_argument(
        "--cluster-method", default="ward", choices=("kmeans", "ward"),
        help="ward (default) = Step-8 Ward HRC cut at k; kmeans = Step-9 K-means. Adds a "
        "cluster_method column so both can coexist in the same per-region CSV.",
    )
    ap.add_argument("--study-area", default=None, help="Step-12 study area (germany / ontario_south)")
    ap.add_argument("--module-root", type=Path, default=None)
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Default: results/01_paper_plots/2_results/01_z500_weather_regimes/stats_csv",
    )
    add_extremes_spec_arguments(ap)
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])
    # Explicit default extreme-day/precip filter -- see DEFAULT_DOMAIN_PCTL_Q/DEFAULT_DOMAIN_AGG
    # at the top of this file. Any --extremes-*/--domain-* flag on the CLI overrides this.
    extremes_spec = extremes_spec_from_args(args) or ExtremesSpec(
        domain_pctl_q=DEFAULT_DOMAIN_PCTL_Q, domain_agg=DEFAULT_DOMAIN_AGG,
    )
    print(f"[export] extremes filter: domain_pctl_q={extremes_spec.domain_pctl_q} domain_agg={extremes_spec.domain_agg!r}")

    module_root = (args.module_root or find_module_root()).resolve()
    out_dir = args.out_dir or (
        module_root / "results" / "01_paper_plots" / "2_results" / "01_z500_weather_regimes" / "stats_csv"
    )
    out_dir = out_dir.expanduser().resolve()

    if args.k is not None and args.bbox_step is None:
        ap.error("--k requires --bbox-step")
    if len(args.region) > 1 and args.bbox_step is not None:
        ap.error("--bbox-step only makes sense with a single --region")

    for region in args.region:
        export_region(
            module_root, region,
            bbox_step=args.bbox_step, k=args.k, study_area=args.study_area, out_dir=out_dir,
            cluster_method=args.cluster_method, extremes_spec=extremes_spec,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
