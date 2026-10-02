#!/usr/bin/env python3
"""
Export RR (Z500 x residual) sub-regime statistics to CSV — no notebook, no plotting.

CSV counterpart to ``stage2_residual.plot_rr_statistics``, modeled on the sibling
``steps/stage1/04_export_statistics_csv.py``: same load call
(``stage2_residual.plot_rr_statistics.load_rr_statistics_inputs``), but writes the underlying
per-group/per-day numbers instead of rendering figures. Meant to run wherever the source data
lives (HPC or a local mount with ``results/residual_regimes/clustering/<region>/``,
``results/clustering/<region>/`` and the ``convective_duration_*.nc`` EPE file populated) — the
one-time NetCDF read is the part that's slow/can exhaust memory in a notebook kernel.
``3_3_Results_RR_Stats.ipynb`` then only reads the small CSVs this script writes and rebuilds the
overview figure locally with pandas + matplotlib.

Per (region, bbox_step, z500_cluster_method, mode) this writes rows into four CSVs (one file per
statistic, covering every bbox_step/method/mode combo ever exported for the region):

  - rr_group_summary_<region>.csv           One row per (Z500 x RR) sub-regime group: n_days, EPE
                                             occurrence (% of group days), n_extreme, total_days
                                             (RR-labeled days for this bbox_step/method/mode, same
                                             value on every row of that combo).
  - rr_epe_occurrence_monthly_<region>.csv  Extreme days per (Z500×RR) group × MJJAS month
                                             (n_days, pct_of_all_days among extreme days in kept
                                             groups) — drives the stacked-bar left panel of the
                                             comprehensive overview figure.
  - rr_boxplot_values_<region>.csv          Long format (metric, z, rr, label, value) — one row per
                                             day assigned to the group with >=1 flagged cell (NOT
                                             restricted to domain-level extreme days — see the "Note
                                             on the boxplot day sample" in plot_rr_statistics.py's
                                             module docstring), for duration/intensity/connectivity/
                                             spatial boxplots.
  - rr_boxplot_meta_<region>.csv            (metric, ylabel) per bbox_step/method/mode, for axis labels
                                             (the intensity metric's ylabel varies by run).

Also writes/updates a single shared ``PROVENANCE.json`` (one record per (region, bbox_step,
z500_cluster_method, mode)) recording which EPE domain-percentile threshold (``domain_pctl_q_tag``,
``rr_extreme_labels_path``) actually produced the currently-on-disk CSVs for that combo -- for
mode="logistic" this comes from ``merge_per_regime_logistic_outputs``' NetCDF attrs (added
2026-08-06; blank for merged files written before then -- rerun stage2_logistic.py with
OVERWRITE=1 to backfill). ``3_3_Results_RR_Stats.ipynb``'s "Confirm EPE definition" cell reads
this the same way ``2_3_Results_Z500_Stats.ipynb`` reads the Z500 sibling's PROVENANCE.json.

For mode="logistic" the same record also carries ``n_bins``/``bin_overrides_path``/
``bin_edges_by_regime`` -- the actual per-Z500-regime probability edges (from
``select_subregime_bins.py``'s bin-overrides YAML, e.g. after a ``--max-bins`` change) that
produced the RR sub-regime groups in this export, so a notebook can confirm the loaded CSVs
reflect the CURRENT bin-overrides YAML rather than one exported before it was last regenerated.

--mode (bbox default, or regime) forces which pipeline's RR regimes file to read — rr_02
(bbox-wide field selection) vs rr_022 (per-Z500-regime field selection); same meaning as
plot_rr_statistics.py's --mode. --z500-cluster-method (kmeans default, or ward) selects which
Z500 parent-regime source the RR regimes were built from. Every row carries both as columns, and
re-running for one combo only replaces that combo's rows in each CSV.

Usage:
    python3 steps/stage2/09_export_statistics_csv.py --region europe --bbox-step B5
    python3 steps/stage2/09_export_statistics_csv.py --region europe --bbox-step B5 \
        --mode regime --z500-cluster-method ward
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Must be set before netCDF4/HDF5 is loaded — load_rr_statistics_inputs reopens the
# convective_duration_*.nc file for more than one metric, which segfaults inside HDF5's
# file-locking path on this Nextcloud-synced project folder (same fix as the sibling
# export_regime_z500_statistics_csv.py / plot_rr_statistics.py / plot_composites.py).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib

matplotlib.use("Agg")  # plot_rr_statistics imports pyplot; no display in a batch/background run
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.preprocessing_helpers import find_module_root  # noqa: E402

_KEY_COLS = ("region", "bbox_step", "z500_cluster_method", "mode")
_BOXPLOT_METRICS = (
    "duration", "intensity_mean", "intensity_median", "connectivity", "extent",
    "spatial_precip_mean", "spatial_precip_median", "spatial_precip_max",
    "excess_mean", "excess_median", "excess_max",
    "precip_alldays_mean", "precip_alldays_median",
)


def export_one_bbox_step(
    module_root: Path, region: str, bbox_step: str, *,
    smooth_window: int, epe_region_label: str | None, mode: str | None, z500_cluster_method: str,
    n_bins: int | None = None, output_tag: str = "", bin_overrides_path: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    import stage2_residual.plot_rr_statistics as plot_rr_statistics

    inputs = plot_rr_statistics.load_rr_statistics_inputs(
        module_root, region, bbox_step, smooth_window=smooth_window,
        epe_region_label=epe_region_label, mode=mode, z500_cluster_method=z500_cluster_method,
        n_bins=n_bins, output_tag=output_tag, bin_overrides_path=bin_overrides_path,
    )
    labels, zvals, groups, totals = inputs["labels"], inputs["zvals"], inputs["groups"], inputs["totals"]
    rr_vals = [r for _z, r in groups]

    group_summary = pd.DataFrame({
        "z": zvals, "rr": rr_vals, "label": labels,
        "n_days": totals, "occ_pct": inputs["occ_pct"], "n_extreme": inputs["n_extreme"],
    })
    # Total RR-labeled days for this bbox_step/method/mode (same value on every row) — needed to
    # reconstruct the "% of all days" distribution figure from the CSV alone.
    group_summary["total_days"] = inputs["total_days"]

    occ_monthly = inputs["occ_monthly"].copy()

    value_rows: list[dict] = []
    meta_rows: list[dict] = []
    metric_arrays = {
        "duration": (inputs["dur_vals"], "hours per flagged cell / day"),
        "intensity_mean": (inputs["int_mean_vals"], inputs["inten_mean_label"]),
        "intensity_median": (inputs["int_median_vals"], inputs["inten_median_label"]),
        "connectivity": (inputs["conn_vals"], "# connected components / day"),
        "extent": (inputs["extent_vals"], "mean patch size (# grid cells, 8-connectivity)"),
        "spatial_precip_mean": (
            inputs["spat_mean_vals"], "mean P_conv_total_grid_day on flagged cells (mm)",
        ),
        "spatial_precip_median": (
            inputs["spat_median_vals"], "median P_conv_total_grid_day on flagged cells (mm)",
        ),
        "spatial_precip_max": (
            inputs["spat_max_vals"], "max P_conv_total_grid_day on flagged cells (mm)",
        ),
        "excess_mean": (
            inputs["excess_mean_vals"], "mean excess over P_q_hourly on flagged cells (mm)",
        ),
        "excess_median": (
            inputs["excess_median_vals"], "median excess over P_q_hourly on flagged cells (mm)",
        ),
        "excess_max": (
            inputs["excess_max_vals"], "max excess over P_q_hourly on flagged cells (mm)",
        ),
        # Non-circular baseline: EVERY day in the group (no flagged-cell gating either) --
        # see common.alldays_precip's module docstring.
        "precip_alldays_mean": (
            inputs["tp_alldays_mean_vals"], "domain-mean tp_daily_sum, all days (mm)",
        ),
        "precip_alldays_median": (
            inputs["tp_alldays_median_vals"], "domain-median tp_daily_sum, all days (mm)",
        ),
    }
    for metric in _BOXPLOT_METRICS:
        per_group_values, ylabel = metric_arrays[metric]
        meta_rows.append({"metric": metric, "ylabel": ylabel})
        for z, rr, label, vals in zip(zvals, rr_vals, labels, per_group_values, strict=False):
            for v in np.asarray(vals, dtype=float).ravel():
                value_rows.append({"metric": metric, "z": int(z), "rr": int(rr), "label": label, "value": float(v)})

    boxplot_values = pd.DataFrame(value_rows, columns=["metric", "z", "rr", "label", "value"])
    boxplot_meta = pd.DataFrame(meta_rows, columns=["metric", "ylabel"])

    # Significance test: within each Z500 parent regime, are its RR sub-regime bins actually
    # different on the all-days precip amount (not the domain-pctl-0.90 label the bins were
    # trained on -- see common.alldays_precip's module docstring for why that
    # comparison would be close to circular). Mann-Whitney U for the production 2-bin case;
    # Kruskal-Wallis omnibus if --n-bins was set above 2 for a given regime.
    from common.alldays_precip import omnibus_significance, two_group_significance

    by_z: dict[int, list[tuple[int, np.ndarray]]] = {}
    for z, rr, vals in zip(zvals, rr_vals, inputs["tp_alldays_mean_vals"], strict=False):
        by_z.setdefault(int(z), []).append((int(rr), np.asarray(vals, dtype=float)))
    sig_rows: list[dict] = []
    for z, bins in sorted(by_z.items()):
        bins = sorted(bins, key=lambda t: t[0])
        if len(bins) == 2:
            (rr_a, va), (rr_b, vb) = bins
            test, stat, p = two_group_significance(va, vb)
            sig_rows.append({
                "z500_regime": z, "n_bins": 2, "bin_a": rr_a, "bin_b": rr_b, "test": test,
                "statistic": stat, "p_value": p,
                "n_a": int(np.isfinite(va).sum()), "n_b": int(np.isfinite(vb).sum()),
                "median_a_mm": float(np.nanmedian(va)) if va.size else float("nan"),
                "median_b_mm": float(np.nanmedian(vb)) if vb.size else float("nan"),
            })
        elif len(bins) > 2:
            test, stat, p = omnibus_significance([v for _, v in bins])
            sig_rows.append({
                "z500_regime": z, "n_bins": len(bins), "bin_a": None, "bin_b": None, "test": test,
                "statistic": stat, "p_value": p, "n_a": None, "n_b": None,
                "median_a_mm": None, "median_b_mm": None,
            })
    significance = pd.DataFrame(
        sig_rows,
        columns=["z500_regime", "n_bins", "bin_a", "bin_b", "test", "statistic", "p_value",
                 "n_a", "n_b", "median_a_mm", "median_b_mm"],
    )

    tables = {
        "rr_group_summary": group_summary,
        "rr_epe_occurrence_monthly": occ_monthly,
        "rr_boxplot_values": boxplot_values,
        "rr_boxplot_meta": boxplot_meta,
        "rr_significance": significance,
    }
    for df in tables.values():
        # NOTE: the (region, bbox_step, z500_cluster_method, mode) replace-key in
        # export_region_bbox() does NOT include n_bins/output_tag/bin_overrides_path —
        # re-running mode="logistic" with a different --n-bins, --output-tag, or
        # --bin-overrides-path silently overwrites the previous logistic rows for this
        # bbox_step/method rather than keeping both. Fine while production is pinned to
        # a single choice of each (see README_ResidualRegimes.md TODO); revisit if/when
        # that's actually swept.
        df.insert(0, "mode", mode or "bbox")
        df.insert(0, "z500_cluster_method", z500_cluster_method)
        df.insert(0, "bbox_step", bbox_step)
        df.insert(0, "region", region)

    # Which EPE domain-percentile threshold actually produced the RR sub-regime groups above --
    # only recorded for mode="logistic" (see load_rr_statistics_inputs' rr_extreme_labels_path /
    # rr_domain_pctl_q_tag, sourced from merge_per_regime_logistic_outputs' NetCDF attrs). Blank
    # for bbox/regime mode (no such record exists there) or for a merged file written before
    # 2026-08-06 (predates the attr) -- both print as "" / "(none)" below rather than guessing.
    provenance = {
        "region": region, "bbox_step": bbox_step, "z500_cluster_method": z500_cluster_method,
        "mode": mode or "bbox", "output_tag": output_tag,
        "rr_extreme_labels_path": inputs.get("rr_extreme_labels_path", ""),
        "domain_pctl_q_tag": inputs.get("rr_domain_pctl_q_tag", ""),
        # mode="logistic" only -- which bin edges actually produced the rows above, so a
        # notebook can confirm these CSVs reflect the CURRENT bin_overrides_<region>_custom.yaml
        # (e.g. after re-running select_subregime_bins.py with a new --max-bins) rather than a
        # stale export. "" / {} for bbox/regime mode, where no per-regime binning happens.
        "n_bins": inputs.get("n_bins_used"),
        "bin_overrides_path": inputs.get("bin_overrides_path_used", ""),
        "bin_edges_by_regime": inputs.get("bin_edges_by_regime", {}),
        "rr_source": Path(inputs["rr_path"]).name if inputs.get("rr_path") else "",
        "duration_source": Path(inputs["dur_path"]).name if inputs.get("dur_path") else "",
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return tables, provenance


def export_region_bbox(
    module_root: Path, region: str, bbox_step: str, *, out_dir: Path,
    smooth_window: int, epe_region_label: str | None, mode: str | None, z500_cluster_method: str,
    n_bins: int | None = None, output_tag: str = "", bin_overrides_path: Path | None = None,
) -> None:
    tables, provenance = export_one_bbox_step(
        module_root, region, bbox_step,
        smooth_window=smooth_window, epe_region_label=epe_region_label,
        mode=mode, z500_cluster_method=z500_cluster_method,
        n_bins=n_bins, output_tag=output_tag, bin_overrides_path=bin_overrides_path,
    )
    written_key = (region, bbox_step, z500_cluster_method, mode or "bbox")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, new_df in tables.items():
        path = out_dir / f"{name}_{region}.csv"
        # Merge with whatever's already on disk for this region, replacing only the
        # (region, bbox_step, z500_cluster_method, mode) combo this run just recomputed — so
        # re-running one bbox/method/mode doesn't clobber the others' rows already on disk.
        if path.is_file():
            old_df = pd.read_csv(path)
            old_keys = old_df[list(_KEY_COLS)].apply(tuple, axis=1)
            old_df = old_df[old_keys != written_key]
            new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)
        print(f"[export] wrote {path} ({len(new_df)} rows)")

    # Sidecar recording which EPE domain-percentile threshold produced this (region, bbox_step,
    # z500_cluster_method, mode) combo -- mirrors export_regime_z500_statistics_csv.py's
    # PROVENANCE.json. Keyed the same way as _KEY_COLS so it stays in sync with the CSVs above.
    provenance_path = out_dir / "PROVENANCE.json"
    all_records: list[dict] = []
    if provenance_path.is_file():
        try:
            existing = json.loads(provenance_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []
        for rec in existing:
            key = (rec.get("region"), rec.get("bbox_step"), rec.get("z500_cluster_method"), rec.get("mode"))
            if key != written_key:
                all_records.append(rec)
    all_records.append(provenance)
    all_records.sort(key=lambda r: (r.get("region", ""), r.get("mode", ""), r.get("bbox_step", "")))
    provenance_path.write_text(json.dumps(all_records, indent=2))
    print(f"[export] wrote {provenance_path} ({len(all_records)} records)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=("europe", "northamerica"))
    ap.add_argument("--bbox-step", required=True)
    ap.add_argument("--module-root", type=Path, default=None)
    ap.add_argument("--smooth-window", type=int, default=11)
    ap.add_argument("--epe-region-label", default=None)
    ap.add_argument("--mode", choices=("bbox", "regime", "logistic"), default=None,
                    help="Force which pipeline's RR regimes to read: rr_02 bbox-mode, rr_022 "
                         "per-regime-mode discrete k-means ('regime'), or the Stage-2 [LOGISTIC] "
                         "continuous predicted_extreme_prob index quantile-binned at load time "
                         "('logistic' — see --n-bins). Default: auto-detect between bbox/regime "
                         "(prefers bbox-mode); never auto-selects 'logistic'.")
    ap.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="ward",
                    help="Which Z500 parent-regime source to summarize RR sub-regimes against "
                         "(default 'ward', reads '_zward'-tagged RR/Z500 files, matches "
                         "regime_alignment_<region>.yaml). Must match how the RR regimes being "
                         "exported were built.")
    ap.add_argument("--n-bins", type=int, default=None,
                    help="--mode logistic only: quantile bins per Z500 regime for the continuous "
                         "predicted_extreme_prob index (default: 2, 'non-extreme-like' vs "
                         "'extreme-like'). Ignored otherwise.")
    ap.add_argument("--output-tag", default="",
                    help="--mode logistic only: select among several logistic runs that differ "
                         "only by extreme-label definition (matches stage2_logistic.py's "
                         "--output-tag). Ignored otherwise.")
    ap.add_argument("--bin-overrides-path", type=Path, default=None,
                    help="--mode logistic only: YAML written by select_subregime_bins.py giving "
                         "per-Z500-regime data-driven probability edges, overriding --n-bins for "
                         "the regimes it covers — same file plot_composites.py/plot_rgb_overview.py "
                         "consume (default: src/common/bin_overrides_<region>_custom.yaml, "
                         "if it exists). Ignored otherwise.")
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Default: results/01_paper_plots/2_results/03_residual_regimes/stats_csv",
    )
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    module_root = (args.module_root or find_module_root()).resolve()
    out_dir = args.out_dir or (
        module_root / "results" / "01_paper_plots" / "2_results" / "03_residual_regimes" / "stats_csv"
    )
    out_dir = out_dir.expanduser().resolve()

    bin_overrides_path = args.bin_overrides_path
    if args.mode == "logistic" and bin_overrides_path is None:
        default_overrides = (
            module_root / "src" / "common" / f"bin_overrides_{args.region}_custom.yaml"
        )
        if default_overrides.is_file():
            print(f"[export] Using default bin overrides: {default_overrides}")
            bin_overrides_path = default_overrides

    export_region_bbox(
        module_root, args.region, args.bbox_step, out_dir=out_dir,
        smooth_window=args.smooth_window, epe_region_label=args.epe_region_label,
        mode=args.mode, z500_cluster_method=args.z500_cluster_method,
        n_bins=args.n_bins, output_tag=args.output_tag, bin_overrides_path=bin_overrides_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
