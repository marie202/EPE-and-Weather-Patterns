from __future__ import annotations

"""Data-driven, per-Z500-regime bin edges for the production LOGISTIC Stage 2
continuous `predicted_extreme_prob` index — implements the
"Decide per-regime bin count" TODO in README_ResidualRegimes.md, in place of
the flat `n_bins=2` interim default `stage2_logistic.bin_predicted_prob_per_regime`
otherwise uses everywhere.

Reads the already-fitted merged NetCDF `stage2_logistic.py` wrote
(`09_RR_logistic_perregime_merged_*.nc`) — no refit, no PCA/joint-features
rebuild needed, just the per-day `predicted_extreme_prob`/`z500_regime`
already on disk — reloads the same extreme-day labels, and calls
`stage2_logistic.select_regime_bin_edges` per Z500 regime: greedily merges a
fine quantile split of `predicted_extreme_prob` from the high-probability end
down until each merged group clears `--min-extreme-per-bin` observed extreme
days, folding any leftover low-probability days into the bottom
"no elevated risk" group.

Writes the resulting per-regime probability edges to a YAML bin-overrides
file (mirrors `src/common/k_overrides_<region>_custom.yaml`) that
`plot_composites.py`/`plot_rgb_overview.py`'s `--bin-overrides-path` picks
up, plus a per-regime/per-bin summary CSV (n_days, n_extreme,
extreme_frequency, prob_range) for human review.

Usage — same field-map/z500-k/z500-cluster-method/output-tag/
extreme-labels-path the rr_02 run you're selecting bins for used, so the
merged NetCDF path resolves to the SAME file (no cache-key mismatch):

  python -m stage2_residual.select_subregime_bins \\
    --region europe --bbox-step B5 --z500-k 4 --per-regime-mode \\
    --regime-fields-map '{"1":"z500+cape+w700","2":"z500+cape"}' \\
    --z500-cluster-method ward \\
    --extreme-labels-path /path/to/convective_duration_..._domainQ950....nc \\
    --max-bins 2 --min-extreme-per-bin 5 --n-fine-bins 10

Or via the sbatch launcher (rebuilds the same field map rr_02 used):
  sbatch --export=ALL,REGION=europe,BBOX_STEP=B5 \\
    rr_022_select_subregime_bins_tu_hpc.sh
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from .feature_relevance import _load_extreme_labels
from .paths import ResidualWorkflowPaths, experiment_id, per_regime_fields_signature
from .stage2_logistic import select_regime_bin_edges

__all__ = ["select_bins_for_region", "main"]


def select_bins_for_region(
    *,
    config,
    extreme_labels_path: Path,
    extreme_var: str,
    max_bins: int = 2,
    min_extreme_per_bin: int = 5,
    n_fine_bins: int | None = None,
    min_recall_frac: float | None = None,
    target_day_frac: float | None = None,
    output_tag: str = "",
) -> tuple[dict[int, list[float]], pd.DataFrame]:
    """Load the merged logistic index this config resolves to, and compute
    data-driven bin edges per Z500 regime. Returns (edges_by_regime, a
    per-regime/per-bin summary DataFrame) — edges_by_regime omits any regime
    for which `select_regime_bin_edges` couldn't clear the threshold even
    once (caller/plot-time code falls back to the plain n_bins split there).
    """
    paths = ResidualWorkflowPaths(config)
    merged_path = paths.logistic_index_path_per_regime(output_tag=output_tag)
    if not merged_path.is_file():
        raise FileNotFoundError(
            f"No merged logistic index at {merged_path} -- run rr_02_stage2_logistic_tu_hpc.sh "
            "for this region/bbox/field-map/output-tag first (see README_ResidualRegimes.md)."
        )

    with xr.open_dataset(merged_path, chunks=None) as ds:
        time_dim = ds["predicted_extreme_prob"].dims[0]
        times = pd.DatetimeIndex(ds[time_dim].values).normalize()
        predicted_prob_full = np.asarray(ds["predicted_extreme_prob"].values, dtype=np.float64)
        z500_regime_full = np.asarray(ds["z500_regime"].values, dtype=int)

    label_times, y_all = _load_extreme_labels(extreme_labels_path, extreme_var)
    label_times = pd.DatetimeIndex(label_times).normalize()
    common = pd.Index(times).intersection(pd.Index(label_times))
    if len(common) < len(times):
        print(f"[select_subregime_bins] {len(times)} merged-index days ∩ "
              f"{len(label_times)} extreme-label days = {len(common)} common days")
    idx_p = pd.Index(times).get_indexer(common)
    idx_y = pd.Index(label_times).get_indexer(common)
    predicted_prob = predicted_prob_full[idx_p]
    z500_regime = z500_regime_full[idx_p]
    y = np.asarray(y_all[idx_y], dtype=np.int64)

    edges_by_regime: dict[int, list[float]] = {}
    summary_rows = []
    for z in sorted(set(z500_regime.tolist())):
        mask = z500_regime == z
        p_z, y_z = predicted_prob[mask], y[mask]
        # target_day_frac ignores y entirely (see select_regime_bin_edges) -- its quantile must
        # be taken over the SAME predicted_prob population plot_composites.py/plot_rgb_overview.py
        # later digitize against (the full merged-index series, every day with a z500_regime
        # label), not the narrower subset that happens to also have an extreme-day label (e.g.
        # EPE labels covering only MJJAS while the Z500/RR clustering runs over more months).
        # Calibrating the "top 30%" cut on a warm-season-only subset and then applying it to the
        # full year-round series silently drifts the day-share far off target (observed: ~50-60%
        # of days landing in the "high" bin instead of the intended 30% for regimes with a
        # moderate, non-near-zero edge) -- the OTHER modes below still use the extreme-label-
        # intersected p_z/y_z, since they need y for their closing-threshold/recall logic.
        if target_day_frac is not None:
            p_for_edges = predicted_prob_full[z500_regime_full == z]
        else:
            p_for_edges = p_z
        edges = select_regime_bin_edges(
            p_for_edges, y_z, max_bins=max_bins, min_extreme_per_bin=min_extreme_per_bin,
            n_fine_bins=n_fine_bins, min_recall_frac=min_recall_frac,
            target_day_frac=target_day_frac,
        )
        data_driven = len(edges) > 0
        if data_driven:
            edges_by_regime[int(z)] = [float(e) for e in edges]
            b = np.digitize(p_z, edges) + 1
        else:
            n2 = max(1, min(2, len(np.unique(p_z))))
            try:
                b = np.asarray(pd.qcut(p_z, q=n2, labels=False, duplicates="drop"), dtype=int) + 1
            except ValueError:
                b = np.ones(len(p_z), dtype=int)

        n_groups = int(b.max()) if len(b) else 0
        total_extreme_z = int(y_z.sum())
        total_days_z = len(y_z)
        for grp in sorted(set(b.tolist())):
            gmask = b == grp
            n_days = int(gmask.sum())
            n_extreme = int(y_z[gmask].sum())
            summary_rows.append({
                "z500_regime": int(z), "bin": int(grp), "n_groups": n_groups,
                "n_days": n_days, "n_extreme": n_extreme,
                # extreme_frequency = precision (P(extreme | in this bin));
                # recall_of_regime_extreme = P(in this bin | extreme) -- the
                # fraction of THIS REGIME's own extreme days this bin
                # captures. A narrow, high-precision bin can still have very
                # low recall (see README_ResidualRegimes.md bin-selection
                # notes) -- both numbers are needed to judge a split.
                "extreme_frequency": round(n_extreme / n_days, 4) if n_days else float("nan"),
                "recall_of_regime_extreme": round(n_extreme / total_extreme_z, 4) if total_extreme_z else float("nan"),
                "day_share_of_regime": round(n_days / total_days_z, 4) if total_days_z else float("nan"),
                "prob_min": round(float(p_z[gmask].min()), 4) if n_days else float("nan"),
                "prob_max": round(float(p_z[gmask].max()), 4) if n_days else float("nan"),
                "data_driven": data_driven,
            })

        tag = "data-driven" if data_driven else (
            f"FALLBACK plain 2-way split (regime has < {min_extreme_per_bin} extreme days in "
            f"total -- {int(y_z.sum())} of {len(y_z)} days)"
        )
        print(f"[select_subregime_bins] Z500-{z}: {n_groups} bins ({tag})")

    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        summary_df = summary_df.sort_values(["z500_regime", "bin"]).reset_index(drop=True)
    return edges_by_regime, summary_df


def _write_bin_overrides_yaml(
    out_path: Path, *, region: str, bbox_step: str, edges_by_regime: dict[int, list[float]],
) -> None:
    """Merge this bbox_step's edges into the region's bin-overrides YAML,
    preserving any other bbox_step entries already written (mirrors
    k_overrides_<region>_custom.yaml's one-file-per-region, keyed-by-step
    layout)."""
    doc: dict = {}
    if out_path.is_file():
        doc = yaml.safe_load(out_path.read_text()) or {}
    doc.setdefault("region", region)
    doc.setdefault("bins_by_step", {})
    doc["bins_by_step"][bbox_step] = {str(z): edges for z, edges in sorted(edges_by_regime.items())}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (temp file + os.replace): this is a read-modify-write over the WHOLE file
    # (all bbox_steps, not just this one), so two rr_022 runs for different bbox_steps racing on
    # the same region's YAML could otherwise clobber each other's step under a plain write_text
    # (classic lost-update: both read the old doc, both write their own version back, last one
    # wins and silently drops the other's step). os.replace is atomic on POSIX, so a concurrent
    # reader (plot_composites.py/plot_rgb_overview.py) never sees a half-written file either.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(yaml.safe_dump(doc, sort_keys=False))
    os.replace(tmp_path, out_path)
    print(f"[select_subregime_bins] Wrote {out_path}")


def main() -> int:
    from .config import build_arg_parser, config_from_args

    parser = build_arg_parser()
    parser.add_argument("--extreme-labels-path", type=Path, required=True,
                         help="Same convective_duration NetCDF the rr_02 run being selected-for used "
                              "(see README_DOMAIN_PCTL_Q.md) -- only used to reload extreme-day labels, "
                              "the merged predicted_extreme_prob is read as-is, not refit.")
    parser.add_argument("--extreme-var", default="flagged_domain_pctl",
                         help="Binary extreme-day label variable (default matches rr_02's production default).")
    parser.add_argument("--output-tag", default="",
                         help="Must match the rr_02 run's --output-tag, so the merged NetCDF path this "
                              "reads resolves to the same file.")
    parser.add_argument("--max-bins", type=int, default=2,
                         help="OUTPUT cap on groups per regime -- capped at 2 by default so every regime "
                              "resolves to at most two clean risk classes (low/high); the greedy top-down "
                              "merge in select_regime_bin_edges stops once max_bins-1 boundaries are closed, "
                              "regardless of --n-fine-bins resolution. Does NOT control candidate-boundary "
                              "resolution -- see --n-fine-bins for that.")
    parser.add_argument("--min-extreme-per-bin", type=int, default=5,
                         help="Minimum observed extreme days a merged group must clear to stay its own bin; "
                              "otherwise it's folded into the next-lower group (see select_regime_bin_edges). "
                              "Used as-is unless --min-recall-frac is also given, in which case it becomes a "
                              "floor under the recall-derived threshold.")
    parser.add_argument("--min-recall-frac", type=float, default=None,
                         help="Target a FRACTION of each regime's own total extreme days for the high bin to "
                              "capture (e.g. 0.4 = high bin closes once it holds >=40%% of that regime's "
                              "extreme days), instead of the absolute --min-extreme-per-bin count. Recommended "
                              "over --min-extreme-per-bin alone when regimes have very different total extreme "
                              "counts -- an absolute count can close the high bin at wildly different recall "
                              "levels per regime (e.g. 17%% vs 50%% observed on real europe B5 data with a "
                              "shared --min-extreme-per-bin=40). --min-extreme-per-bin still applies as a floor "
                              "so a low-signal regime doesn't collapse to a 1-2-day bin. See the new "
                              "recall_of_regime_extreme/day_share_of_regime columns in the summary CSV to check "
                              "the resulting trade-off per regime.")
    parser.add_argument("--target-day-frac", type=float, default=None,
                         help="Fix the high bin to a FRACTION OF DAYS instead of a fraction of extreme days -- "
                              "e.g. 0.3 = top 30%% of this regime's days by predicted probability, always "
                              "exactly 2 groups. Takes precedence over --max-bins/--min-extreme-per-bin/"
                              "--n-fine-bins/--min-recall-frac when set (they're ignored). The resulting "
                              "recall/precision fall out wherever the data lands rather than being targeted -- "
                              "check the summary CSV's recall_of_regime_extreme/extreme_frequency columns "
                              "afterward. Useful once a --min-recall-frac run has shown how concentrated a "
                              "regime's signal is, to ask the reverse question (e.g. 'what recall do I get for "
                              "30%% of days flagged', which can be much higher than a small pure bin's recall).")
    parser.add_argument("--n-fine-bins", type=int, default=None,
                         help="Resolution of candidate cut points the greedy merge chooses boundaries from, "
                              "independent of --max-bins (the OUTPUT bin cap). Defaults to --max-bins, which "
                              "reproduces the old behavior -- with the default --max-bins=2 that means only ONE "
                              "candidate boundary (the median), so the result can only ever be a 50/50 split. "
                              "Pass a larger value (e.g. 10, for decile resolution) to let the merge place that "
                              "single boundary somewhere other than the median, based on where "
                              "--min-extreme-per-bin extreme days actually accumulate from the top down, while "
                              "still capping the output at --max-bins groups (see select_regime_bin_edges).")
    parser.add_argument("--bin-overrides-path", type=Path, default=None,
                         help="YAML to write/update (default: "
                              "src/common/bin_overrides_<region>_custom.yaml).")
    args = parser.parse_args()

    if not args.per_regime_mode:
        print("ERROR: select_subregime_bins requires --per-regime-mode (with --regime-fields-map) -- "
              "it must resolve the SAME merged NetCDF path rr_02_stage2_logistic wrote.", file=sys.stderr)
        return 1

    config = config_from_args(args)

    try:
        edges_by_regime, summary_df = select_bins_for_region(
            config=config, extreme_labels_path=args.extreme_labels_path, extreme_var=args.extreme_var,
            max_bins=args.max_bins, min_extreme_per_bin=args.min_extreme_per_bin,
            n_fine_bins=args.n_fine_bins, min_recall_frac=args.min_recall_frac,
            target_day_frac=args.target_day_frac, output_tag=args.output_tag,
        )
    except FileNotFoundError as exc:
        print(f"[select_subregime_bins] ERROR: {exc}", file=sys.stderr)
        return 1

    paths = ResidualWorkflowPaths(config)
    sig = per_regime_fields_signature(config)
    eid = experiment_id(config)
    tag_suffix = f"_{args.output_tag}" if args.output_tag else ""
    summary_csv = paths.diagnostics_root / f"rr_bin_select_summary_{sig}_{eid}{tag_suffix}.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    print(f"[select_subregime_bins] Wrote {summary_csv}")
    print("[select_subregime_bins] === SUMMARY ===")
    print(summary_df.to_string(index=False))

    bin_overrides_path = args.bin_overrides_path or (
        config.module_root / "src" / "common" / f"bin_overrides_{config.region}_custom.yaml"
    )
    _write_bin_overrides_yaml(
        bin_overrides_path, region=config.region, bbox_step=config.bbox_step,
        edges_by_regime=edges_by_regime,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
