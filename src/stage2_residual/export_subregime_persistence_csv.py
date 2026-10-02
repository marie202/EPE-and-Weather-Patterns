#!/usr/bin/env python3
"""
Export Stage-2 (RR) sub-regime persistence / episode statistics to CSV — no notebook, no plotting.

Fifth sibling of ``steps/stage2/09_export_statistics_csv.py``, for the one quantity that
script does not write: how long a day stays in the same sub-regime. Runs wherever the merged
Stage-2 NetCDF lives (HPC, or a local mount with
``results/residual_regimes/clustering/<region>/`` populated), because the per-day
``predicted_extreme_prob`` / ``z500_regime`` arrays are the part a notebook kernel should not
have to open. ``3_3_Results_RR_Stats.ipynb`` then reads the small CSV this writes into
``RR_PERSISTENCE`` and fills the "Median persist." column of
``tab:rr_subregime_characterisation``.

Bins are resolved exactly as every other production consumer resolves them -- the same
``_find_rr_logistic_merged`` file, the same ``load_bin_overrides_yaml`` edges, the same
``bin_predicted_prob_per_regime`` call -- so the SR1/SR2 groups counted here are by construction
the same groups the statistics CSVs and the composites use. Pass the SAME
``--output-tag``/``--bin-overrides-path`` you passed to ``export_regime_rr_statistics_csv.py``.

Definitions
-----------
A *run* is a maximal stretch of consecutive calendar days in the same (Z500 regime, sub-regime
bin) state. Gaps of more than one day break a run, so the MJJAS season boundary (30 Sep -> 1 May)
ends a run automatically and no run ever spans two summers.

A *Z500 episode* is the same construction on the parent Z500 regime alone, ignoring the bin --
directly comparable to the "median persistence" column of ``tab:regime_names``. The
``pct_episodes_with_high_risk`` column then says how often such an episode contains at least one
high-risk day, which is the Stage-2 analogue of a regime transition probability.

Usage
-----
  python3 steps/stage2/10_export_subregime_persistence.py \\
      --region europe --bbox-step B5 --z500-cluster-method ward \\
      --output-tag domainQ900_blockedcv_lexmax \\
      --bin-overrides-path src/common/bin_overrides_europe_domainQ900_blockedcv_lexmax.yaml

Re-running for one (region, bbox_step, z500_cluster_method) combo replaces only that combo's
rows, same as the sibling export scripts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_KEY_COLS = ["region", "bbox_step", "z500_cluster_method", "mode"]


def _run_lengths(day_numbers: np.ndarray, state: np.ndarray) -> list[tuple[int, int]]:
    """[(state, run_length), ...] for maximal stretches of consecutive days in the same state.

    ``day_numbers`` must be sorted integer day ordinals; a jump of more than 1 breaks the run
    even when the state is unchanged (season boundary, or a gap in the input)."""
    if day_numbers.size == 0:
        return []
    new_run = np.empty(day_numbers.size, dtype=bool)
    new_run[0] = True
    new_run[1:] = (np.diff(day_numbers) != 1) | (state[1:] != state[:-1])
    starts = np.flatnonzero(new_run)
    lengths = np.diff(np.append(starts, day_numbers.size))
    return list(zip(state[starts].tolist(), lengths.tolist(), strict=True))


def subregime_run_stats(dates, z500_regime, bins) -> pd.DataFrame:
    """Per (Z500 regime, sub-regime bin) run-length statistics, plus the parent Z500 episode
    statistics repeated on each of that regime's rows. Pure function -- unit-testable without
    touching NetCDF."""
    df = pd.DataFrame({
        "date": pd.DatetimeIndex(dates).normalize(),
        "z": np.asarray(z500_regime, dtype=int),
        "bin": np.asarray(bins, dtype=int),
    }).sort_values("date").reset_index(drop=True)
    day_num = (df["date"] - pd.Timestamp("1900-01-01")).dt.days.to_numpy()

    # Parent Z500 episodes: runs of the regime label alone, and whether each contains a
    # high-risk day (the top bin of that regime).
    top_bin = df.groupby("z")["bin"].max().to_dict()
    df["is_high"] = df.apply(lambda r: r["bin"] == top_bin[r["z"]], axis=1)
    z_runs = _run_lengths(day_num, df["z"].to_numpy())
    episode_id = np.repeat(np.arange(len(z_runs)), [n for _, n in z_runs])
    df["episode"] = episode_id
    ep = df.groupby(["z", "episode"]).agg(length=("bin", "size"), any_high=("is_high", "any"))

    rows = []
    for z, sub in df.groupby("z"):
        runs = _run_lengths(
            (sub["date"] - pd.Timestamp("1900-01-01")).dt.days.to_numpy(),
            sub["bin"].to_numpy(),
        )
        ep_z = ep.loc[z]
        for b in sorted(sub["bin"].unique()):
            lens = np.array([n for s, n in runs if s == b], dtype=float)
            n_days = int(lens.sum())
            rows.append({
                "z": int(z), "rr": int(b), "n_days": n_days, "n_runs": int(lens.size),
                "median_run_days": float(np.median(lens)) if lens.size else np.nan,
                "mean_run_days": float(lens.mean()) if lens.size else np.nan,
                "p90_run_days": float(np.percentile(lens, 90)) if lens.size else np.nan,
                "max_run_days": float(lens.max()) if lens.size else np.nan,
                # share of this bin's DAYS that sit in a run of length 1
                "pct_isolated_days": 100.0 * float((lens == 1).sum()) / n_days if n_days else np.nan,
                # parent-regime context, identical on every row of the same z
                "z500_median_episode_days": float(ep_z["length"].median()),
                "z500_n_episodes": int(len(ep_z)),
                "pct_episodes_with_high_risk": 100.0 * float(ep_z["any_high"].mean()),
            })
    return pd.DataFrame(rows)


def _upsert(path: Path, new_rows: pd.DataFrame) -> None:
    """Replace this combo's rows in an existing CSV, keeping every other combo."""
    if path.is_file():
        old = pd.read_csv(path)
        keys = {c: new_rows[c].iloc[0] for c in _KEY_COLS}
        mask = np.ones(len(old), dtype=bool)
        for c, v in keys.items():
            mask &= (old[c] == v)
        new_rows = pd.concat([old[~mask], new_rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_rows.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=("europe", "northamerica"))
    ap.add_argument("--bbox-step", default="B5")
    ap.add_argument("--module-root", type=Path, default=None)
    ap.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="ward")
    ap.add_argument("--output-tag", default="",
                    help="Same --output-tag the rr_02 run / statistics export used.")
    ap.add_argument("--bin-overrides-path", type=Path, default=None,
                    help="select_subregime_bins.py YAML. Pass the SAME file the statistics "
                         "export used, or the bins here will not match the other CSVs.")
    ap.add_argument("--n-bins", type=int, default=2,
                    help="Only used for regimes absent from the bin-overrides YAML.")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    from stage2_residual.plot_results import _find_rr_logistic_merged
    from stage2_residual.stage2_logistic import (
        bin_predicted_prob_per_regime, load_bin_overrides_yaml,
    )

    module_root = args.module_root or Path(__file__).resolve().parents[1]
    dw = module_root
    rr_path = _find_rr_logistic_merged(
        dw, args.region, args.bbox_step, output_tag=args.output_tag,
        z500_cluster_method=args.z500_cluster_method,
    )
    if rr_path is None:
        raise SystemExit(
            f"No merged Stage-2 LOGISTIC file for {args.region} {args.bbox_step} "
            f"(z500_cluster_method={args.z500_cluster_method}, output_tag={args.output_tag!r}). "
            "Run stage2_logistic.py first, or check --output-tag."
        )
    print(f"[rr_persist] merged index: {rr_path.name}")

    with xr.open_dataset(rr_path) as ds:
        time_dim = ds["predicted_extreme_prob"].dims[0]
        dates = pd.DatetimeIndex(ds[time_dim].values)
        prob = np.asarray(ds["predicted_extreme_prob"].values, dtype=float)
        z500 = np.asarray(ds["z500_regime"].values, dtype=int)
        extreme_labels_path = str(ds.attrs.get("extreme_labels_path", ""))

    edges = {}
    if args.bin_overrides_path is not None:
        edges = load_bin_overrides_yaml(args.bin_overrides_path, args.bbox_step)
        print(f"[rr_persist] bin edges from {args.bin_overrides_path.name}: {edges}")
    else:
        print(f"[rr_persist] WARNING: no --bin-overrides-path; falling back to a flat "
              f"n_bins={args.n_bins} quantile split, which will NOT match the production bins.")

    bins = bin_predicted_prob_per_regime(
        prob, z500, n_bins=args.n_bins, bin_edges_by_regime=edges or None,
    )

    finite = np.isfinite(prob)
    if not finite.all():
        print(f"[rr_persist] dropping {int((~finite).sum())} days with no predicted probability")
    stats = subregime_run_stats(dates[finite], z500[finite], np.asarray(bins)[finite])

    stats.insert(0, "region", args.region)
    stats.insert(1, "bbox_step", args.bbox_step)
    stats.insert(2, "z500_cluster_method", args.z500_cluster_method)
    stats.insert(3, "mode", "logistic")
    stats["output_tag"] = args.output_tag
    stats["rr_source"] = rr_path.name
    stats["extreme_labels_path"] = extreme_labels_path

    out_dir = args.out_dir or (
        module_root / "results" / "01_paper_plots" / "2_results" / "03_residual_regimes" / "stats_csv"
    )
    out_path = out_dir / f"rr_subregime_persistence_{args.region}.csv"
    _upsert(out_path, stats)
    print(f"[rr_persist] wrote {out_path}")
    print(stats.drop(columns=["rr_source", "extreme_labels_path"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
