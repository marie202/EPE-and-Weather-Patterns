#!/usr/bin/env python3
"""
Derive additional ``domain_pctl_q`` threshold labels from an EXISTING
``convective_duration_TEST_*.nc`` file, without rerunning the expensive
hourly-detection pipeline (``epe_01b_detect_convective_hourly_duration_test.py``,
8-16h/job).

Why this is safe / cheap
-------------------------
In that pipeline, ``domain_extreme_pctl`` (the continuous, recalibrated
domain-day severity index) is computed once from the hourly data and does NOT
depend on ``--domain-pctl-q`` at all -- only the final binary label,
``flagged_domain_pctl = (domain_extreme_pctl >= domain_pctl_q)``, does (see
that script's Step 7 vs Step 8). Since every TEST file already stores
``domain_extreme_pctl`` for the full record, any other threshold's label can
be re-derived from ONE existing file in seconds, for any region/domain-agg
combination already run once.

This script does exactly that: point it at one on-disk TEST file (any
domain_pctl_q), give it either explicit thresholds (``--domain-pctl-q``) or
target flagged-day counts (``--target-days``, converted via
``q = 1 - target_days / n_days``, rounded to 3 decimals -- matching the
``domainQ<NNN>`` filename-tag precision the rest of the pipeline uses), and
it writes one minimal output NetCDF per threshold: just ``time``,
``flagged_domain_pctl``, and a passthrough of ``domain_extreme_pctl`` (kept
for provenance/debugging, e.g. to confirm two thresholds were derived from
the same underlying severity index). It intentionally does NOT copy the
large hourly (``time_hour`` x lat x lon) variables -- downstream consumers
that need those (composites, ``EPE_FILE``-based plotting) should rerun the
one winning threshold through the real pipeline; this script is for
``rr_01`` feature-relevance sensitivity sweeps only, which read a single
daily binary variable (see ``stage2_residual.feature_relevance._load_extreme_labels``).

Usage
-----
    python epe_derive_domain_pctl_labels.py \\
        --source-file /path/to/convective_duration_TEST_..._domainQ900_....nc \\
        --target-days 1000,1300,1500,2000 \\
        --out-dir /path/to/same/extreme_precip_events/dir

    # or specify exact thresholds directly:
    python epe_derive_domain_pctl_labels.py \\
        --source-file ... --domain-pctl-q 0.862,0.820,0.793,0.724
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import xarray as xr


def _domain_q_tag(q: float) -> str:
    return f"domainQ{int(round(q * 1000)):03d}"


def derive_one(
    source_ds: xr.Dataset,
    source_path: Path,
    domain_pctl_q: float,
    out_dir: Path,
) -> Path:
    n_days = source_ds.sizes["time"]
    pctl = source_ds["domain_extreme_pctl"]
    flagged = (pctl >= float(domain_pctl_q)).astype("i1")
    flagged.name = "flagged_domain_pctl"
    n_flagged = int(flagged.sum())

    out_ds = xr.Dataset(
        {
            "flagged_domain_pctl": flagged,
            "domain_extreme_pctl": pctl,
        },
        attrs={
            "derived_from": str(source_path),
            "domain_pctl_q": float(domain_pctl_q),
            "note": (
                "Derived by re-thresholding domain_extreme_pctl from an existing "
                "TEST file -- hourly detection was NOT rerun. Minimal file: time + "
                "flagged_domain_pctl + domain_extreme_pctl only."
            ),
        },
    )

    new_tag = _domain_q_tag(domain_pctl_q)
    old_name = source_path.name
    if not re.search(r"domainQ\d+", old_name):
        raise SystemExit(f"Source filename has no domainQ<NNN> token to replace: {old_name}")
    new_name = re.sub(r"domainQ\d+", new_tag, old_name)
    out_path = out_dir / new_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(out_path)

    print(
        f"[derive] q={domain_pctl_q:.3f} ({new_tag}) -> {n_flagged} / {n_days} days "
        f"({100 * n_flagged / n_days:.2f}%) -> {out_path}"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-file", type=Path, required=True, help="Existing convective_duration_TEST_*.nc with domain_extreme_pctl.")
    p.add_argument("--out-dir", type=Path, default=None, help="Defaults to source file's own directory.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--domain-pctl-q", type=str, default=None, help="Comma-separated list of thresholds, e.g. 0.862,0.820,0.793,0.724")
    g.add_argument("--target-days", type=str, default=None, help="Comma-separated list of target flagged-day counts, e.g. 1000,1300,1500,2000")
    args = p.parse_args()

    if not args.source_file.exists():
        raise SystemExit(f"Source file not found: {args.source_file}")
    out_dir = args.out_dir if args.out_dir is not None else args.source_file.parent

    with xr.open_dataset(args.source_file, chunks=None) as ds:
        if "domain_extreme_pctl" not in ds:
            raise SystemExit(
                f"{args.source_file.name} has no domain_extreme_pctl variable -- "
                "must be a lexmax TEST file from detect_convective_duration.py"
            )
        n_days = ds.sizes["time"]
        ds = ds[["domain_extreme_pctl"]].load()

    if args.domain_pctl_q is not None:
        thresholds = [float(x) for x in args.domain_pctl_q.split(",") if x.strip()]
    else:
        targets = [int(x) for x in args.target_days.split(",") if x.strip()]
        thresholds = [round(1.0 - t / n_days, 3) for t in targets]
        print(f"[derive] n_days={n_days}; target-days -> rounded q: "
              + ", ".join(f"{t}d->{q:.3f}" for t, q in zip(targets, thresholds)))

    for q in thresholds:
        if not (0.0 < q < 1.0):
            raise SystemExit(f"Derived/given domain_pctl_q out of (0,1): {q}")
        derive_one(ds, args.source_file, q, out_dir)


if __name__ == "__main__":
    main()
