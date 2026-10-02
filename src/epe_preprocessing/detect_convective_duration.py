#!/usr/bin/env python3
"""
EPE (epe_01b) -- **TEST / EXPERIMENTAL VARIANT**: domain-day label via per-cell,
climatology-normalized daily-precip percentiles, instead of an OR over cell-hours.

This is a *copy* of ``epe_01b_detect_convective_hourly_duration.py`` with the
same hourly detection machinery (grid-point quantile threshold, CAPE filter,
day grouping) left untouched. Only the DOMAIN-DAY LABEL is different. It is
kept as a separate ``_test`` script (not merged into the production script) so
production ``flagged_any_gridcell`` outputs are never touched -- see the
"do not add without explicit project decision" policy note in
``epe_01b_detect_convective_hourly_duration.py``.

Motivation
----------
The production label ``flagged_any_gridcell`` is
``(n_conv_hours_anywhere > 0)``: a day is "extreme" if ANY grid cell, at ANY
hour, anywhere in the domain, crosses its local q999 threshold. That is an OR
across (grid cells x hours), which is a massive multiple-comparisons
inflation -- empirically 32-65% of warm-season days get flagged per Z500
regime, nowhere near the ~0.1% a point-wise q999 threshold implies. A fixed
minimum-duration or minimum-cell-count gate would fix the base rate but
penalizes small, fast-moving convective cells that clear an ERA5 gridcell in
a single hourly step and never accumulate 2h at one point.

New method (see also README_EPE.md, section marked "TEST")
------------------------------------------------------------
Still fully hourly for detection (steps 1-4 identical to epe_01b); only the
hourly -> daily -> domain-day rollup (step 5+) changes:

1-4) Identical to epe_01b: P_q (grid-point local climatology quantile),
     E_raw_hour, E_conv_hour (CAPE filter), P_conv_total_grid_day (mm of
     precip during convective-extreme hours, per cell per day; duration-
     agnostic -- a 1h supercell and a 6h moderate event score the same if
     they drop the same mm).

5) Per-cell climatological percentile (NEW): for each grid cell that is ever
   extreme (``extreme_grid_any``), rank that cell's own
   ``P_conv_total_grid_day`` time series (its own record, zero-days
   included) via an empirical CDF (average-rank ties, so the very common
   "0 mm today" case is handled correctly under zero-inflation) ->
   ``cell_pctl_grid_day`` in (0, 1]. This re-applies local-climatology
   normalization AT THE AGGREGATION STEP (an Alpine cell's routine 15mm
   convective day and a north-German lowland cell's routine 4mm convective
   day both land near the same percentile of their own record), not just at
   the hourly-threshold step.

6) Domain-day statistic (NEW): default ``--domain-agg lexmax`` ranks days by
   their single most extreme cell (``max_cell cell_pctl_grid_day``), breaking
   ties with the 2nd-most-extreme cell, then 3rd, etc. -- never averaging, so
   a genuinely small/localized event is never diluted by cells that were not
   actually part of it. Still effectively a max-type operation across the
   whole domain, so a moving cell is fully caught wherever it happens to be
   that day -- this directly answers "won't a moving storm get missed by a
   duration/count cutoff?": it isn't gated on persistence at a fixed cell at
   all. (``--domain-agg max``: plain max, kept for comparison -- confirmed on
   real data to produce a large tied block, see step 7; ``--domain-agg
   topk_mean``: averages the top-k cells, smooths single-cell noise but can
   dilute a small-footprint event.)

7) Day-level recalibration (NEW): for ``max``/``topk_mean``, ``domain_stat``
   is the max (or top-k mean) of many per-cell percentiles, so its raw value
   drifts upward with domain size even after per-cell normalization
   (max-of-many is not itself uniform) -- worse, on real data a large
   fraction of days independently contain some cell at ITS OWN all-time high
   (percentile exactly 1.0), so ``max`` produces a large block of days tied
   at an identical value; averaging their tied ranks clamps ALL of them
   (including the genuinely most extreme ones) to the same middling
   percentile, which can make ``--domain-pctl-q`` mathematically unreachable
   (confirmed: 2.75% of an ontario_south 1979-2025 record tied at max,
   capping the achievable percentile at 0.986 -- flagged_domain_pctl was 0%
   at the default 0.999 cutoff). ``lexmax`` sidesteps this: the full
   descending-sorted per-cell vector for a day is its own tie-break key, so
   two days are only equal if EVERY cell matches in order -- practically
   never. ``domain_extreme_pctl`` in (0, 1] is a genuinely continuous
   domain-day severity index either way (usable directly as a regression
   target, e.g. in ``residual_severity_signal_diag_REGRESSION.py``).

8) Binary label (NEW): ``flagged_domain_pctl = (domain_extreme_pctl >=
   --domain-pctl-q)`` (default 0.999) -- a single calibrated test per day
   against the domain-day statistic's own distribution, instead of an OR
   over thousands of correlated per-cell-hour tests. This is what should
   land close to the intended ~0.1-1% base rate.

Outputs
-------
ONE NetCDF, with BOTH an hourly axis (``time_hour``) and a daily axis
(``time``) -- same layout as epe_01b, not two separate files:
  - Hourly (``time_hour``): ``E_raw_hour`` / ``E_conv_hour`` (via
    ``--hourly-mask raw|conv|both``, default ``conv``, matching the
    production ``HOURLY_MASK_OUTPUT`` default), plus optional ``P_hour`` /
    ``P_conv_hour`` (``--hourly-precip``) and per-day peak/timing summaries
    (``--hourly-peak``, ``--hourly-timing``) -- all identical in definition
    to epe_01b, since hourly detection is untouched.
  - Daily (``time``): all production epe_01b daily variables (incl.
    unchanged ``flagged_any_gridcell``), PLUS the TEST additions:
      - cell_pctl_grid_day       (time x lat x lon, float, NaN outside
                                   extreme_grid_any): per-cell day-total percentile
      - domain_stat_raw          (time): pre-recalibration max/topk-mean of
                                   cell_pctl_grid_day (diagnostic only)
      - domain_extreme_pctl      (time): recalibrated domain-day severity index,
                                   continuous in (0, 1]
      - flagged_domain_pctl      (time): thresholded 0/1 label (the TEST
                                   replacement for flagged_any_gridcell)
``flagged_any_gridcell`` and all production variables are still written
unchanged, so both labels can be compared side by side in the same file.

Filenames are tagged ``convective_duration_TEST_...`` (never
``convective_duration_...``) so this can never collide with or be picked up
in place of a production epe_01b file by ``newest_duration_file`` /
``newest_extremes_file`` filename globs.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import rankdata

from epe_preprocessing.epe_functions import cape_convective_ok, cape_filter_description, netcdf_attr_bool


def parse_years(spec: str) -> list[str]:
    spec = spec.strip()
    if not spec:
        raise ValueError("empty years spec")
    m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", spec)
    if m:
        y0, y1 = int(m.group(1)), int(m.group(2))
        if y1 < y0:
            raise ValueError(f"years range end before start: {spec!r}")
        return [str(y) for y in range(y0, y1 + 1)]
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"no years in spec: {spec!r}")
    for p in parts:
        if not re.fullmatch(r"\d{4}", p):
            raise ValueError(f"invalid year token {p!r} in {spec!r}")
    return parts


def years_filename_slug(years: list[str]) -> str:
    yints = sorted(int(y) for y in years)
    if len(yints) == 1:
        return f"years_{yints[0]}"
    contiguous = all(b == a + 1 for a, b in zip(yints, yints[1:]))
    if contiguous:
        return f"years_{yints[0]}-{yints[-1]}"
    return "years_" + "_".join(str(y) for y in yints)


def resolve_paper_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / "1_Detect_Weather_Patterns").is_dir():
            return p / "1_Detect_Weather_Patterns"
    return None


def _stripped_or_none(s: str | None) -> str | None:
    if s is None:
        return None
    t = str(s).strip()
    return t if t else None


def resolve_in_dir(raw_subdir: str, paper_root: Path | None, module_root: str | None) -> Path:
    return (
        (paper_root / "data" / "raw" / "era5" / "extreme_precip_events" / raw_subdir)
        if paper_root is not None
        else (Path(module_root) / "data" / "raw" / "era5" / "extreme_precip_events" / raw_subdir)
        if module_root
        else (Path.cwd() / "data" / "raw" / "era5" / "extreme_precip_events" / raw_subdir)
    )


def _open_hourly_inputs(
    *,
    raw_subdir: str,
    filename_token: str,
    paper_root: Path | None,
    module_root: str | None,
    years: list[str],
    months: list[str],
    years_slug: str,
    months_tag: str,
    in_file: Path | None,
    chunks: dict,
) -> xr.Dataset:
    if in_file is not None:
        if not in_file.exists():
            raise SystemExit(f"Input file not found: {in_file}")
        return xr.open_dataset(in_file, chunks=chunks)

    in_dir = resolve_in_dir(raw_subdir, paper_root, module_root)

    # Prefer "all-in-one" legacy file if present
    legacy = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_{years_slug}_m{months_tag}_hourly.nc"
    if legacy.exists():
        return xr.open_dataset(legacy, chunks=chunks)

    # One file per year
    per_year: list[Path] = []
    for y in years:
        p = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m{months_tag}_hourly.nc"
        if not p.exists():
            per_year = []
            break
        per_year.append(p)
    if per_year:
        return xr.open_mfdataset(per_year, chunks=chunks, combine="by_coords")

    # Fallback: exactly one …_yYYYY_m*_hourly.nc per year (non-standard month tag).
    per_year_any_monthstag: list[Path] = []
    for y in years:
        matches = sorted(in_dir.glob(f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m*_hourly.nc"))
        pref = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m{months_tag}_hourly.nc"
        if pref.exists() and pref in matches:
            per_year_any_monthstag.append(pref)
            continue
        if len(matches) == 1:
            per_year_any_monthstag.append(matches[0])
            continue
        per_year_any_monthstag = []
        break
    if per_year_any_monthstag:
        return xr.open_mfdataset(per_year_any_monthstag, chunks=chunks, combine="by_coords")

    # Otherwise year-month chunks
    paths: list[Path] = []
    for y in years:
        for m in months:
            p = in_dir / f"era5_single_levels_tp-cp-cape_{filename_token}_y{y}_m{m}_hourly.nc"
            if not p.exists():
                hint = (
                    f"Expected hourly files under:\n  {in_dir}\n"
                    f"with basename token `{filename_token}` (e.g. …_tp-cp-cape_{filename_token}_y{y}_m{months_tag}_hourly.nc).\n"
                )
                if raw_subdir != filename_token:
                    hint += (
                        "If you moved data into a folder named after `--region-label` but kept download filenames, "
                        "set `--hourly-filename-token` (or env `HOURLY_FILENAME_TOKEN`) to the token used in epe_00 "
                        "(e.g. ontario_south_tight).\n"
                    )
                hint += "Run epe_00_download_era5.py (default --chunk-by year writes …_yYYYY_m0506070809_hourly.nc)."
                raise SystemExit("Missing expected chunked hourly input file:\n" f"  {p}\n" + hint)
            paths.append(p)
    return xr.open_mfdataset(paths, chunks=chunks, combine="by_coords")


def guess_time_dim(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"Could not find a time coordinate in dataset; coords={list(ds.coords)} dims={list(ds.dims)}")


def guess_lat_lon(da: xr.DataArray) -> tuple[str, str]:
    lat = "latitude" if "latitude" in da.coords else "lat" if "lat" in da.coords else None
    lon = "longitude" if "longitude" in da.coords else "lon" if "lon" in da.coords else None
    if lat is None or lon is None:
        raise KeyError(f"Could not find lat/lon coords in {list(da.coords)}")
    return lat, lon


def resolve_out_dir(region_label: str, paper_root: Path | None, module_root: str | None) -> Path:
    if paper_root is not None:
        return paper_root / "data" / "processed" / region_label / "extreme_precip_events"
    if module_root:
        return Path(module_root) / "data" / "processed" / region_label / "extreme_precip_events"
    return Path.cwd() / "data" / "processed" / region_label / "extreme_precip_events"


def q_filename_tag(q: float) -> str:
    """Encode quantile in output stem: 0.99 -> q99, 0.999 -> q999."""
    pct1000 = int(round(float(q) * 1000.0))
    if pct1000 % 10 == 0:
        return f"q{pct1000 // 10:02d}"
    return f"q{pct1000:03d}"


def quantile_sample_filename_tag(sample: str, *, wet_threshold_mm: float) -> str:
    if sample == "wet_hours":
        return f"sampleWetmm{int(round(wet_threshold_mm * 10)):02d}"
    return "sampleAll"


def _tz_slug(tz_name: str) -> str:
    return (tz_name or "UTC").replace("/", "-")


def local_day_start_time_utc(time_utc: xr.DataArray, *, tz_name: str, day_start_hour_local: int) -> xr.DataArray:
    """
    Return per-timestep labels equal to the UTC instant of the local-day start.
    Handles DST via timezone conversion (IANA tz database).
    """
    if not (0 <= int(day_start_hour_local) <= 23):
        raise ValueError("day_start_hour_local must be 0..23")

    t = pd.DatetimeIndex(pd.to_datetime(time_utc.values))
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    local = t.tz_convert(tz_name)
    shifted = local - pd.Timedelta(hours=int(day_start_hour_local))
    start_local = shifted.normalize() + pd.Timedelta(hours=int(day_start_hour_local))
    start_utc = start_local.tz_convert("UTC").tz_localize(None)
    return xr.DataArray(start_utc.to_numpy(dtype="datetime64[ns]"), coords=time_utc.coords, dims=time_utc.dims, name="day")


def _pctl_rank_1d(x: np.ndarray) -> np.ndarray:
    """Empirical-CDF percentile rank in (0, 1], average-rank ties.

    Average-rank ties matter here because both use sites are heavily
    zero-inflated (per-cell daily totals: most days are 0; the recalibration
    step: most days are far from the tail) -- average-rank ties spread all
    tied values to the same, correctly-central percentile instead of an
    arbitrary ordering among equal values.
    """
    return (rankdata(x, method="average") / x.size).astype("float32")


def _topk_mean_1d(x: np.ndarray, *, k: int) -> np.float32:
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.float32(np.nan)
    kk = min(k, x.size)
    return np.float32(np.mean(np.sort(x)[-kk:]))


def _lexicographic_domain_pctl(mat: np.ndarray) -> np.ndarray:
    """Rank days (rows) by descending-sorted lexicographic order, never averaging.

    ``mat`` is (n_days x n_cells) of per-cell day percentiles (already filled
    with a sentinel below any real percentile, e.g. -1, for inactive cells).
    Day i outranks day j if i's single largest cell value is bigger; ties are
    broken by the 2nd-largest, then 3rd-largest, and so on, over every cell in
    the domain. This fixes the ``--domain-agg max`` pathology (many days tied
    at an identical max value, e.g. because independent cells each hit their
    own all-time high on different days, collapsing them to the same score)
    without averaging in cells that were not actually part of a (possibly
    small/localized) event, which is the dilution risk of ``topk_mean``.

    Returns a percentile in (0, 1] per day (rank / n_days), on the same scale
    as the other --domain-agg options.
    """
    n_days, n_cells = mat.shape
    sorted_desc = -np.sort(-mat, axis=1)  # each row: largest cell first, smallest last
    # np.lexsort's LAST key is the primary sort key; reverse column order so
    # column 0 (each day's max) is compared first, column 1 breaks ties, etc.
    keys = [sorted_desc[:, c] for c in range(n_cells - 1, -1, -1)]
    order = np.lexsort(keys)  # ascending: order[0] = least extreme day
    ranks = np.empty(n_days, dtype=np.float64)
    ranks[order] = np.arange(1, n_days + 1)
    return (ranks / n_days).astype("float32")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EPE (TEST/EXPERIMENTAL): hourly convective detection identical to epe_01b; "
            "domain-day label replaced by a per-cell climatology-percentile + domain-max "
            "+ day-recalibration statistic instead of an OR over cell-hours."
        )
    )
    parser.add_argument("--region-label", default="germany")
    parser.add_argument("--years", default="2023-2025")
    parser.add_argument(
        "--timezone",
        default="UTC",
        help='IANA timezone name used to define the analysis day in local time (e.g. "Europe/Berlin", "America/Toronto").',
    )
    parser.add_argument(
        "--day-start-hour-local",
        type=int,
        default=6,
        help="Define analysis day start hour in *local* time (default 6 -> 06:00–06:00 local).",
    )
    # Backwards compatible alias (historically interpreted as UTC)
    parser.add_argument(
        "--day-start-hour",
        type=int,
        default=None,
        help="Alias for --day-start-hour-local (kept for backward compatibility).",
    )
    parser.add_argument("--hourly-precip-metric", default="tp_hourly", choices=["tp_hourly", "cp_hourly"])
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument(
        "--quantile-sample",
        default="all_hours",
        choices=["all_hours", "wet_hours"],
        help="all_hours: q over every hour (zeros included); wet_hours: q over tp>wet-threshold only",
    )
    parser.add_argument(
        "--wet-threshold-mm",
        type=float,
        default=0.1,
        help=(
            "Minimum hourly tp (mm/h) to count as extreme: always required for E_raw_hour. "
            "When --quantile-sample=wet_hours, P_q is also computed only on hours above this."
        ),
    )
    parser.add_argument(
        "--cape-threshold",
        type=float,
        default=0.0,
        help=(
            "Convective CAPE filter: 0 means strictly CAPE>0 (CAPE=0 excluded); "
            "values >0 require CAPE above that value."
        ),
    )
    parser.add_argument(
        "--skip-cape-filter",
        action="store_true",
        help="Disable CAPE filter entirely — all hours pass regardless of CAPE.",
    )
    parser.add_argument("--cp-frac-min", type=float, default=0.0)
    parser.add_argument(
        "--hourly-mask",
        default="conv",
        choices=["none", "raw", "conv", "both"],
        help=(
            "Include hourly 2D masks (time_hour x lat x lon) in the output NetCDF. "
            "'raw' writes E_raw_hour; 'conv' writes E_conv_hour (default -- this is the hourly "
            "mask most downstream steps want); 'both' writes both."
        ),
    )
    parser.add_argument(
        "--hourly-precip",
        default="none",
        choices=["none", "all", "conv", "both"],
        help=(
            "Optionally include hourly precipitation fields (time_hour x lat x lon). "
            "'all' writes P_hour; 'conv' writes P_conv_hour (=P_hour masked by E_conv_hour); "
            "'both' writes both."
        ),
    )
    parser.add_argument(
        "--hourly-peak",
        default="none",
        choices=["none", "raw", "conv", "both"],
        help="Per-day, per-grid-cell peak hour/amount for extreme hours (see epe_01b for definition).",
    )
    parser.add_argument(
        "--hourly-timing",
        default="none",
        choices=["none", "raw", "conv", "both"],
        help="Per-day, per-grid-cell first/last hour and longest consecutive run (see epe_01b for definition).",
    )
    parser.add_argument(
        "--domain-agg",
        default="lexmax",
        choices=["max", "topk_mean", "lexmax"],
        help=(
            "How to combine per-cell day percentiles (cell_pctl_grid_day) into the domain-day "
            "statistic. 'lexmax' (default): rank days by their single most extreme cell, "
            "breaking ties with the 2nd/3rd/... most extreme cell as needed (never averages, "
            "so a small/localized event is never diluted by irrelevant calm cells) -- fixes a "
            "confirmed pathology in 'max' where many days tie at an identical max value and "
            "get clamped to the same middling recalibrated percentile (see README_EPE.md "
            "12.3b). 'max': plain max, kept for comparison -- prone to that tie pathology on "
            "real domains. 'topk_mean': average of the top --domain-topk cells -- smooths "
            "single-cell noise but can dilute a genuinely small-footprint event."
        ),
    )
    parser.add_argument("--domain-topk", type=int, default=5, help="k for --domain-agg topk_mean (unused by lexmax/max).")
    parser.add_argument(
        "--domain-pctl-q",
        type=float,
        default=0.999,
        help=(
            "Quantile of the domain-day statistic's OWN distribution across the record used to "
            "set flagged_domain_pctl (the TEST replacement for flagged_any_gridcell)."
        ),
    )
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--paper-root", type=Path, default=None)
    parser.add_argument(
        "--hourly-raw-subdir",
        default=None,
        help=(
            "Subdirectory under data/raw/era5/extreme_precip_events/ for hourly inputs "
            "(default: same as --region-label). Use when downloads live in a different folder name."
        ),
    )
    parser.add_argument(
        "--hourly-filename-token",
        default=None,
        help=(
            "Middle token in era5_single_levels_tp-cp-cape_<TOKEN>_yYYYY_m…_hourly.nc (default: "
            "same as --hourly-raw-subdir if that is set, else --region-label). "
            "Use when the folder was renamed to match REGION_LABEL but filenames still use the old download token."
        ),
    )
    parser.add_argument("--in-file", type=Path, default=None)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--chunks-time", type=int, default=24 * 30)
    parser.add_argument("--chunks-lat", type=int, default=100)
    parser.add_argument("--chunks-lon", type=int, default=100)
    args = parser.parse_args()
    if not args.skip_cape_filter and args.cape_threshold < 0.0:
        raise SystemExit("--cape-threshold must be >= 0 (use 0 for strictly CAPE>0; CAPE=0 never counts)")

    day_start_hour_local = int(args.day_start_hour_local)
    if args.day_start_hour is not None:
        day_start_hour_local = int(args.day_start_hour)
    if not (0 <= day_start_hour_local <= 23):
        raise SystemExit("--day-start-hour-local/--day-start-hour must be 0..23")
    if not (0.0 < args.q < 1.0):
        raise SystemExit("--q must be between 0 and 1 (exclusive)")
    if args.cp_frac_min < 0.0 or args.cp_frac_min > 1.0:
        raise SystemExit("--cp-frac-min must be in [0,1]")
    if args.wet_threshold_mm < 0.0:
        raise SystemExit("--wet-threshold-mm must be >= 0")
    if not (0.0 < args.domain_pctl_q < 1.0):
        raise SystemExit("--domain-pctl-q must be between 0 and 1 (exclusive)")
    if args.domain_topk < 1:
        raise SystemExit("--domain-topk must be >= 1")

    years = parse_years(args.years)
    years_slug = years_filename_slug(years)
    # Always use MJJAS (May–Sep). Keep this fixed to avoid needing to pass --months in pipeline scripts.
    months = ["05", "06", "07", "08", "09"]
    months_tag = "".join(months)

    paper_root: Path | None = args.paper_root
    if paper_root is None:
        env_paper = os.environ.get("PAPER_ROOT")
        if env_paper:
            paper_root = Path(env_paper)
        else:
            paper_root = resolve_paper_root(Path.cwd().resolve())
    module_root = os.environ.get("MODULE_ROOT")

    def _env_nonempty(name: str) -> str | None:
        v = os.environ.get(name)
        if v is None:
            return None
        t = str(v).strip()
        return t if t else None

    raw_arg = (
        _stripped_or_none(args.hourly_raw_subdir)
        or _env_nonempty("HOURLY_RAW_SUBDIR")
        or _env_nonempty("HOURLY_INPUT_REGION_LABEL")
    )
    fn_arg = _stripped_or_none(args.hourly_filename_token) or _env_nonempty("HOURLY_FILENAME_TOKEN")
    raw_subdir = raw_arg if raw_arg is not None else args.region_label
    if fn_arg is not None:
        filename_token = fn_arg
    elif raw_arg is not None:
        filename_token = raw_arg
    else:
        filename_token = args.region_label

    # Typical epe_00 download stem for Southern Ontario is `ontario_south_tight` while REGION_LABEL is `ontario_south`.
    if args.region_label == "ontario_south" and fn_arg is None and filename_token == "ontario_south":
        filename_token = "ontario_south_tight"

    out_dir = resolve_out_dir(args.region_label, paper_root, module_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cp_frac_tag = f"cpFracMin{int(round(args.cp_frac_min * 100)):02d}"
    tz_slug = _tz_slug(args.timezone)
    q_tag = q_filename_tag(args.q)
    sample_tag = quantile_sample_filename_tag(
        args.quantile_sample, wet_threshold_mm=float(args.wet_threshold_mm)
    )
    domain_agg_tag = (
        args.domain_agg if args.domain_agg in ("max", "lexmax") else f"{args.domain_agg}{args.domain_topk}"
    )
    domain_q_tag = f"domainQ{int(round(args.domain_pctl_q * 1000)):03d}"
    hourly_tag = f"hprecip{args.hourly_precip}_hmask{args.hourly_mask}_hpeak{args.hourly_peak}_htiming{args.hourly_timing}"
    out_path = args.out_file if args.out_file is not None else (
        out_dir
        / (
            f"convective_duration_TEST_{args.region_label}_{years_slug}_m{months_tag}"
            f"_dayStartLocal{day_start_hour_local:02d}_tz{tz_slug}"
            f"_{args.hourly_precip_metric}_{q_tag}_{sample_tag}"
            f"_{'capeOff' if args.skip_cape_filter else f'cape{int(round(args.cape_threshold))}'}_{cp_frac_tag}"
            f"_{domain_agg_tag}_{domain_q_tag}_{hourly_tag}.nc"
        )
    )

    chunks = {
        "time": args.chunks_time,
        "valid_time": args.chunks_time,
        "latitude": args.chunks_lat,
        "longitude": args.chunks_lon,
    }
    ds = _open_hourly_inputs(
        raw_subdir=raw_subdir,
        filename_token=filename_token,
        paper_root=paper_root,
        module_root=module_root,
        years=years,
        months=months,
        years_slug=years_slug,
        months_tag=months_tag,
        in_file=args.in_file,
        chunks=chunks,
    )

    tdim = guess_time_dim(ds)
    if tdim != "time":
        ds = ds.rename({tdim: "time"})

    required = {"tp", "cp", "cape"}
    missing = sorted(required - set(ds.data_vars))
    if missing:
        raise SystemExit(f"Missing required vars in hourly dataset: {missing}. Present: {list(ds.data_vars)}")

    # Restrict to MJJAS (in case inputs are larger)
    ds = ds.sel(time=ds["time.month"].isin([5, 6, 7, 8, 9]))

    # Choose hourly precipitation metric
    if args.hourly_precip_metric == "tp_hourly":
        P = ds["tp"].astype("float32")
    else:
        P = ds["cp"].astype("float32")
    CAPE = ds["cape"].astype("float32")
    lat_name, lon_name = guess_lat_lon(P)

    wet_threshold_m = float(args.wet_threshold_mm) / 1000.0
    wet = P > wet_threshold_m
    if args.quantile_sample == "wet_hours":
        P_q = P.where(wet).chunk({"time": -1}).quantile(args.q, dim="time", skipna=True, keep_attrs=False)
    else:
        P_q = P.chunk({"time": -1}).quantile(args.q, dim="time", keep_attrs=False)
    if "quantile" in P_q.dims:
        P_q = P_q.squeeze("quantile", drop=True)
    if args.persist and hasattr(P_q.data, "persist"):
        P_q = P_q.persist()

    # Always require measurable rain at the cell-hour; keeps dry-hour noise out of flags.
    # --- Steps 1-4: identical to production epe_01b (hourly detection unchanged) ---
    E_raw_hour = (P > P_q) & wet
    if args.skip_cape_filter:
        E_conv_hour = E_raw_hour
    else:
        E_conv_hour = E_raw_hour & cape_convective_ok(CAPE, args.cape_threshold)

    if args.cp_frac_min > 0.0:
        tp = ds["tp"].astype("float32")
        cp = ds["cp"].astype("float32")
        cp_frac = cp / tp.where(tp != 0)
        E_conv_hour = E_conv_hour & (cp_frac >= args.cp_frac_min)

    # Keep only grid cells that are "extreme in the first place":
    # at least one hour exceeds the grid-point precip threshold (E_raw_hour==1).
    extreme_grid_any = E_raw_hour.any(dim="time").rename("extreme_grid_any")  # lat x lon (bool)

    # Define analysis day as local 06–06 (or configured hour) in the given timezone, handling DST.
    day = local_day_start_time_utc(ds["time"], tz_name=args.timezone, day_start_hour_local=day_start_hour_local)

    # Aggregate: per-gridcell convective-extreme hours per day
    n_conv_hours_grid = E_conv_hour.groupby(day).sum(dim="time").astype("i2")

    # Number of hours available in each day (handles missing hours)
    n_hours_in_day = xr.ones_like(E_conv_hour.astype("i1")).groupby(day).sum(dim="time").astype("i2")

    frac_conv_hours_grid = (n_conv_hours_grid / n_hours_in_day.where(n_hours_in_day != 0)).astype("f4")

    # Daily precipitation totals restricted to convective-extreme hours.
    P_conv = P.where(E_conv_hour)
    P_conv_day_total = P_conv.groupby(day).sum(dim=("time", lat_name, lon_name), skipna=True).astype("f4")
    # Unmasked per-cell/day precip total (0 where no convective hours) -- used below for the
    # per-cell percentile ranking; must NOT be masked to NaN by extreme_grid_any yet, since the
    # rank needs every day (including zero-days) in each ever-extreme cell's own record.
    P_conv_day_grid_raw = P_conv.groupby(day).sum(dim="time", skipna=True).astype("f4")

    # Apply the "ever extreme" grid mask for the production-style masked copy (NaN outside it,
    # for map plotting parity with epe_01b's P_conv_total_grid_day).
    n_conv_hours_grid_masked = n_conv_hours_grid.where(extreme_grid_any)
    frac_conv_hours_grid_masked = frac_conv_hours_grid.where(extreme_grid_any)
    P_conv_day_grid = P_conv_day_grid_raw.where(extreme_grid_any)

    # Daily count of grid cells that were convective-extreme at least once during that day.
    n_conv_gridcells_any_hour_day = (
        E_conv_hour.groupby(day).any(dim="time").sum(dim=(lat_name, lon_name)).astype("i4")
    )

    # Domain-level (hourly): how many grid cells are convective-extreme at each hour?
    n_conv_gridcells_hour = E_conv_hour.sum(dim=(lat_name, lon_name)).astype("i4")
    any_gridcell_hour = (n_conv_gridcells_hour >= 1).astype("i1")
    time_hour = ds["time"].values

    # Domain-level (daily): number of hours in the day where any grid cell is convective-extreme
    n_conv_hours_anywhere = any_gridcell_hour.groupby(day).sum(dim="time").astype("i2")
    # Total cell-hours per day: sum of flagged cells over all hours (two cells at same hour = 2)
    n_conv_cellhours_day = n_conv_gridcells_hour.groupby(day).sum(dim="time").astype("i4")

    # Production label, kept unchanged for direct side-by-side comparison.
    flagged_any_gridcell = (n_conv_hours_anywhere > 0).astype("i1")

    # --- Step 5 (NEW/TEST): per-cell climatological percentile of P_conv_day_grid_raw ---
    # Rank each cell's own day-total-precip series against its own record (zero-inflation-safe
    # average-rank ties). Cheap enough to compute for every cell; masked to extreme_grid_any
    # afterward so never-extreme cells (degenerate all-zero series) don't contaminate the
    # domain-max in step 6.
    cell_pctl_grid_day = xr.apply_ufunc(
        _pctl_rank_1d,
        P_conv_day_grid_raw.chunk({"day": -1}),
        input_core_dims=[["day"]],
        output_core_dims=[["day"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
    ).where(extreme_grid_any)

    # --- Step 6 (NEW/TEST): domain-day statistic from per-cell percentiles ---
    if args.domain_agg in ("lexmax", "max"):
        # For lexmax this is diagnostic-only (same quantity --domain-agg max reports, for
        # direct comparison in the output file) -- it is NOT what the ranking below uses.
        domain_stat_raw = cell_pctl_grid_day.max(dim=(lat_name, lon_name), skipna=True)
    else:
        k = int(args.domain_topk)
        stacked = cell_pctl_grid_day.stack(cell=(lat_name, lon_name)).chunk({"cell": -1})
        domain_stat_raw = xr.apply_ufunc(
            lambda x: _topk_mean_1d(x, k=k),
            stacked,
            input_core_dims=[["cell"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.float32],
        )

    # --- Step 7 (NEW/TEST): recalibrate against the domain-day statistic's own distribution ---
    if args.domain_agg == "lexmax":
        # Full lexicographic ranking IS the recalibration here -- it directly produces a
        # properly tie-broken percentile, so the generic rankdata(domain_stat_raw) path
        # below (which is what produced the confirmed tie pathology for 'max') is skipped.
        filled = cell_pctl_grid_day.fillna(-1.0)
        stacked_all = filled.stack(cell=(lat_name, lon_name)).transpose("day", "cell")
        mat = np.asarray(stacked_all.compute().values, dtype="float64")
        domain_extreme_pctl_vals = _lexicographic_domain_pctl(mat)
        domain_extreme_pctl = xr.DataArray(
            domain_extreme_pctl_vals, coords={"day": cell_pctl_grid_day["day"]}, dims=("day",)
        )
    else:
        # domain_stat_raw is small (one value per day); compute eagerly.
        domain_stat_vals = np.asarray(domain_stat_raw.compute().values, dtype="float64")
        domain_stat_filled = np.nan_to_num(domain_stat_vals, nan=-1.0)  # NaN days rank at the bottom
        domain_extreme_pctl_vals = (
            rankdata(domain_stat_filled, method="average") / domain_stat_filled.size
        ).astype("float32")
        domain_extreme_pctl = xr.DataArray(
            domain_extreme_pctl_vals, coords={"day": domain_stat_raw["day"]}, dims=("day",)
        )

    # --- Step 8 (NEW/TEST): binary label from the recalibrated domain-day statistic ---
    flagged_domain_pctl = (domain_extreme_pctl >= float(args.domain_pctl_q)).astype("i1")

    # Use day coordinate as 'time' for daily products (place at day_start_hour)
    day_time = n_conv_hours_anywhere["day"].rename("time")

    # --- Hourly outputs (unchanged from epe_01b) ---
    # This is the "hourly version" of the mask: E_raw_hour / E_conv_hour on their own
    # time_hour axis, identical in definition to production epe_01b. Default --hourly-mask=conv
    # matches the production shell wrapper's HOURLY_MASK_OUTPUT default, so E_conv_hour is
    # written unless explicitly disabled with --hourly-mask none.
    peak_vars: dict[str, xr.DataArray] = {}
    if args.hourly_peak != "none":
        def _peak_from_mask(mask: xr.DataArray, *, prefix: str) -> None:
            Pm = P.where(mask)
            has_any = (Pm.groupby(day).count(dim="time") > 0)

            p_peak = Pm.groupby(day).max(dim="time", skipna=True).where(has_any)

            def _peak_hour_utc(x: xr.DataArray) -> xr.DataArray:
                x_max = x.max(dim="time", skipna=True)
                hour = x["time"].dt.hour.astype("float32")
                return hour.where(x == x_max).max(dim="time", skipna=True)

            hour = Pm.groupby(day).map(_peak_hour_utc).where(has_any)
            hour_i2 = hour.fillna(-1).astype("i2")

            peak_vars[f"{prefix}_peak_hour_utc"] = (
                hour_i2.rename({"day": "time"}).assign_coords(time=day_time.values)
            )
            peak_vars[f"{prefix}_peak_P_hourly"] = (
                p_peak.astype("float32").rename({"day": "time"}).assign_coords(time=day_time.values)
            )

        if args.hourly_peak in {"raw", "both"}:
            _peak_from_mask(E_raw_hour, prefix="raw")
        if args.hourly_peak in {"conv", "both"}:
            _peak_from_mask(E_conv_hour, prefix="conv")

    timing_vars: dict[str, xr.DataArray] = {}
    if args.hourly_timing != "none":
        def _timing_from_mask(mask: xr.DataArray, *, prefix: str) -> None:
            hour = ds["time"].dt.hour.astype("float32")
            hour3 = hour.broadcast_like(mask)

            has_any = mask.groupby(day).any(dim="time")
            first_hour = hour3.where(mask).groupby(day).min(dim="time", skipna=True).where(has_any)
            last_hour = hour3.where(mask).groupby(day).max(dim="time", skipna=True).where(has_any)

            def _longest_run_1d(x: np.ndarray) -> np.int16:
                best = 0
                cur = 0
                for v in x:
                    if bool(v):
                        cur += 1
                        if cur > best:
                            best = cur
                    else:
                        cur = 0
                return np.int16(best)

            runlen = xr.apply_ufunc(
                _longest_run_1d,
                mask.groupby(day),
                input_core_dims=[["time"]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.int16],
            ).where(has_any, other=0)

            timing_vars[f"{prefix}_first_hour_utc"] = (
                first_hour.fillna(-1).astype("i2").rename({"day": "time"}).assign_coords(time=day_time.values)
            )
            timing_vars[f"{prefix}_last_hour_utc"] = (
                last_hour.fillna(-1).astype("i2").rename({"day": "time"}).assign_coords(time=day_time.values)
            )
            timing_vars[f"{prefix}_longest_run_hours"] = (
                runlen.astype("i2").rename({"day": "time"}).assign_coords(time=day_time.values)
            )

        if args.hourly_timing in {"raw", "both"}:
            _timing_from_mask(E_raw_hour, prefix="raw")
        if args.hourly_timing in {"conv", "both"}:
            _timing_from_mask(E_conv_hour, prefix="conv")

    hourly_mask_vars: dict[str, xr.DataArray] = {}
    if args.hourly_mask != "none":
        if args.hourly_mask in {"raw", "both"}:
            hourly_mask_vars["E_raw_hour"] = (
                E_raw_hour.astype("i1").rename({"time": "time_hour"}).assign_coords(time_hour=time_hour)
            )
        if args.hourly_mask in {"conv", "both"}:
            hourly_mask_vars["E_conv_hour"] = (
                E_conv_hour.astype("i1").rename({"time": "time_hour"}).assign_coords(time_hour=time_hour)
            )

    hourly_precip_vars: dict[str, xr.DataArray] = {}
    if args.hourly_precip != "none":
        P_hour = P.rename({"time": "time_hour"}).assign_coords(time_hour=time_hour)
        if args.hourly_precip in {"all", "both"}:
            hourly_precip_vars["P_hour"] = P_hour.where(extreme_grid_any).astype("float32")
        if args.hourly_precip in {"conv", "both"}:
            P_conv_hour = P.where(E_conv_hour).rename({"time": "time_hour"}).assign_coords(time_hour=time_hour)
            hourly_precip_vars["P_conv_hour"] = P_conv_hour.where(extreme_grid_any).astype("float32")

    # Ensure all daily 1D/2D products share the same explicit daily time coordinate (day_time).
    n_hours_in_day_1d = (
        n_hours_in_day.mean(dim=(lat_name, lon_name)).rename({"day": "time"}).assign_coords(time=day_time.values).astype("i2")
    )
    n_conv_hours_anywhere_1d = (
        n_conv_hours_anywhere.rename({"day": "time"}).assign_coords(time=day_time.values).astype("i2")
    )
    flagged_any_gridcell_1d = (
        flagged_any_gridcell.rename({"day": "time"}).assign_coords(time=day_time.values).astype("i1")
    )
    P_conv_day_total_1d = (
        P_conv_day_total.rename({"day": "time"}).assign_coords(time=day_time.values).astype("f4")
    )
    P_conv_day_grid_3d = (
        P_conv_day_grid.rename({"day": "time"}).assign_coords(time=day_time.values).astype("f4")
    )
    n_conv_gridcells_any_hour_day_1d = (
        n_conv_gridcells_any_hour_day.rename({"day": "time"}).assign_coords(time=day_time.values).astype("i4")
    )
    n_conv_cellhours_day_1d = (
        n_conv_cellhours_day.rename({"day": "time"}).assign_coords(time=day_time.values).astype("i4")
    )
    cell_pctl_grid_day_3d = (
        cell_pctl_grid_day.rename({"day": "time"}).assign_coords(time=day_time.values).astype("f4")
    )
    domain_stat_raw_1d = (
        domain_stat_raw.rename({"day": "time"}).assign_coords(time=day_time.values).astype("f4")
    )
    domain_extreme_pctl_1d = (
        domain_extreme_pctl.rename({"day": "time"}).assign_coords(time=day_time.values).astype("f4")
    )
    flagged_domain_pctl_1d = (
        flagged_domain_pctl.rename({"day": "time"}).assign_coords(time=day_time.values).astype("i1")
    )

    out = xr.Dataset(
        {
            "P_q_hourly": P_q.astype("f4"),
            "extreme_grid_any": extreme_grid_any.astype("i1"),
            "n_hours_in_day": n_hours_in_day_1d,
            "n_conv_hours_anywhere": n_conv_hours_anywhere_1d,
            "flagged_any_gridcell": flagged_any_gridcell_1d,
            "P_conv_total_day": P_conv_day_total_1d,
            "P_conv_total_grid_day": P_conv_day_grid_3d,  # time x lat x lon
            "n_conv_gridcells_any_hour_day": n_conv_gridcells_any_hour_day_1d,
            "n_conv_cellhours_day": n_conv_cellhours_day_1d,
            "n_conv_gridcells_hour": n_conv_gridcells_hour.rename({"time": "time_hour"}).assign_coords(time_hour=time_hour),
            "any_gridcell_hour": any_gridcell_hour.rename({"time": "time_hour"}).assign_coords(time_hour=time_hour),
            "n_conv_hours_grid": n_conv_hours_grid_masked.rename({"day": "time"}).assign_coords(time=day_time.values).astype("f4"),
            "frac_conv_hours_grid": frac_conv_hours_grid_masked.rename({"day": "time"}).assign_coords(time=day_time.values),
            **hourly_mask_vars,
            **hourly_precip_vars,
            **peak_vars,
            **timing_vars,
            # --- TEST-only additions ---
            "cell_pctl_grid_day": cell_pctl_grid_day_3d,
            "domain_stat_raw": domain_stat_raw_1d,
            "domain_extreme_pctl": domain_extreme_pctl_1d,
            "flagged_domain_pctl": flagged_domain_pctl_1d,
        }
    )

    out = out.assign_coords(time=day_time.values)
    src_hourly = str(args.in_file) if args.in_file is not None else (
        f"auto: raw_subdir={raw_subdir}; filename_token={filename_token}"
        if raw_subdir != args.region_label or filename_token != args.region_label
        else "auto (legacy or chunked inputs)"
    )
    out.attrs.update(
        {
            "methodology": "TEST_domain_percentile_v1",
            "source_hourly_inputs": src_hourly,
            "hourly_raw_subdir": str(raw_subdir),
            "hourly_filename_token": str(filename_token),
            "region_label": str(args.region_label),
            "years_spec": str(args.years),
            "years_slug": str(years_slug),
            "months_tag": str(months_tag),
            "hourly_precip_metric": args.hourly_precip_metric,
            "q": float(args.q),
            "quantile_sample": str(args.quantile_sample),
            "wet_threshold_mm": float(args.wet_threshold_mm),
            "min_tp_mm_for_extreme_hour": float(args.wet_threshold_mm),
            "skip_cape_filter": netcdf_attr_bool(args.skip_cape_filter),
            "cape_threshold": float("nan") if args.skip_cape_filter else float(args.cape_threshold),
            "cape_filter": "none" if args.skip_cape_filter else cape_filter_description(args.cape_threshold),
            "cape_strictly_positive": netcdf_attr_bool(not args.skip_cape_filter and float(args.cape_threshold) <= 0.0),
            "cp_frac_min": float(args.cp_frac_min),
            "timezone": str(args.timezone),
            "day_start_hour_local": int(day_start_hour_local),
            "months": months,
            "hourly_mask": str(args.hourly_mask),
            "hourly_precip": str(args.hourly_precip),
            "hourly_peak": str(args.hourly_peak),
            "hourly_timing": str(args.hourly_timing),
            "domain_agg": str(args.domain_agg),
            "domain_topk": int(args.domain_topk),
            "domain_pctl_q": float(args.domain_pctl_q),
        }
    )

    n_days = int(flagged_any_gridcell_1d.sizes["time"])
    n_flagged = int(flagged_any_gridcell_1d.sum().values)
    pct = 100.0 * n_flagged / n_days if n_days else float("nan")
    n_flagged_pctl = int(flagged_domain_pctl_1d.sum().values)
    pct_pctl = 100.0 * n_flagged_pctl / n_days if n_days else float("nan")
    print(
        f"[epe_01b_test] {args.quantile_sample} q={args.q:g} wet>{args.wet_threshold_mm} mm/h "
        f"domain-agg={args.domain_agg} domain-q={args.domain_pctl_q:g}: "
        f"flagged_any_gridcell {n_flagged}/{n_days} ({pct:.1f}%) | "
        f"flagged_domain_pctl {n_flagged_pctl}/{n_days} ({pct_pctl:.1f}%)"
    )

    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=encoding)
    print(f"[epe_01b_test] Wrote: {out_path}")


if __name__ == "__main__":
    main()
