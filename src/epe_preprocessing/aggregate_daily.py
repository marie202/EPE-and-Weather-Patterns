#!/usr/bin/env python3
"""
EPE (epe_01): Aggregate hourly ERA5 tp/cp/CAPE to daily sums/maxima.

Inputs (hourly, single-levels):
  - tp   (total precipitation): daily sum and daily max (hourly intensity proxy)
  - cp   (convective precipitation): daily sum
  - cape (convective available potential energy): daily max

Outputs (daily, MJJAS only):
  - tp_daily_sum
  - tp_daily_max
  - cp_daily_sum
  - cape_daily_max
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


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


def resolve_in_path(region_label: str, paper_root: Path | None, module_root: str | None, *, years_slug: str, months_tag: str) -> Path:
    base = (
        (paper_root / "data" / "raw" / "era5" / "extreme_precip_events" / region_label)
        if paper_root is not None
        else (Path(module_root) / "data" / "raw" / "era5" / "extreme_precip_events" / region_label)
        if module_root
        else (Path.cwd() / "data" / "raw" / "era5" / "extreme_precip_events" / region_label)
    )
    return base / f"era5_single_levels_tp-cp-cape_{region_label}_{years_slug}_m{months_tag}_hourly.nc"


def resolve_in_dir(region_label: str, paper_root: Path | None, module_root: str | None) -> Path:
    return (
        (paper_root / "data" / "raw" / "era5" / "extreme_precip_events" / region_label)
        if paper_root is not None
        else (Path(module_root) / "data" / "raw" / "era5" / "extreme_precip_events" / region_label)
        if module_root
        else (Path.cwd() / "data" / "raw" / "era5" / "extreme_precip_events" / region_label)
    )


def _open_hourly_inputs(
    *,
    region_label: str,
    paper_root: Path | None,
    module_root: str | None,
    years: list[str],
    months: list[str],
    years_slug: str,
    months_tag: str,
    in_file: Path | None,
    chunks: dict,
) -> xr.Dataset:
    # Explicit override takes precedence
    if in_file is not None:
        if not in_file.exists():
            raise SystemExit(f"Input file not found: {in_file}")
        return xr.open_dataset(in_file, chunks=chunks)

    # Prefer legacy "all-in-one" file if present
    legacy = resolve_in_path(region_label, paper_root, module_root, years_slug=years_slug, months_tag=months_tag)
    if legacy.exists():
        return xr.open_dataset(legacy, chunks=chunks)

    in_dir = resolve_in_dir(region_label, paper_root, module_root)

    # Next, try "one file per year" layout (chunk-by year), e.g.
    #   era5_single_levels_tp-cp-cape_<region>_y2023_m0506070809_hourly.nc
    per_year: list[Path] = []
    for y in years:
        p = in_dir / f"era5_single_levels_tp-cp-cape_{region_label}_y{y}_m{months_tag}_hourly.nc"
        if not p.exists():
            per_year = []
            break
        per_year.append(p)
    if per_year:
        return xr.open_mfdataset(per_year, chunks=chunks, combine="by_coords")

    # Otherwise, load chunked (year-month) files
    paths: list[Path] = []
    for y in years:
        for m in months:
            p = in_dir / f"era5_single_levels_tp-cp-cape_{region_label}_y{y}_m{m}_hourly.nc"
            if not p.exists():
                raise SystemExit(
                    "Missing expected chunked hourly input file:\n"
                    f"  {p}\n"
                    "Run epe_00_download_era5.py first (chunk-by year-month or year)."
                )
            paths.append(p)

    # combine="by_coords" works for these monotonic monthly time segments
    return xr.open_mfdataset(paths, chunks=chunks, combine="by_coords")


def resolve_out_dir(region_label: str, paper_root: Path | None, module_root: str | None) -> Path:
    if paper_root is not None:
        return paper_root / "data" / "processed" / region_label / "extreme_precip_events"
    if module_root:
        return Path(module_root) / "data" / "processed" / region_label / "extreme_precip_events"
    return Path.cwd() / "data" / "processed" / region_label / "extreme_precip_events"


def guess_time_dim(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.dims or cand in ds.coords:
            return cand
    raise KeyError(f"Could not find a time coordinate in dataset; coords={list(ds.coords)} dims={list(ds.dims)}")


def _tz_slug(tz_name: str) -> str:
    return (tz_name or "UTC").replace("/", "-")


def local_day_start_time_utc(time_utc: xr.DataArray, *, tz_name: str, day_start_hour_local: int) -> xr.DataArray:
    """
    Return a per-timestep label equal to the UTC instant of the *local* day start.

    Example: with tz_name="Europe/Berlin" and day_start_hour_local=6,
    all timestamps in the same local 06:00–06:00 window will share the same label.
    Handles DST via timezone conversion.
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
    return xr.DataArray(start_utc.to_numpy(dtype="datetime64[ns]"), coords=time_utc.coords, dims=time_utc.dims, name="time_day_start")


def main() -> None:
    parser = argparse.ArgumentParser(description="EPE: aggregate hourly ERA5 tp/cp/CAPE to daily sums/maxima.")
    parser.add_argument("--region-label", default="germany")
    parser.add_argument("--years", default="2023-2025")
    parser.add_argument("--months", default="05,06,07,08,09")
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
    parser.add_argument("--paper-root", type=Path, default=None)
    parser.add_argument("--in-file", type=Path, default=None)
    parser.add_argument("--out-file", type=Path, default=None)
    parser.add_argument("--chunks-time", type=int, default=24 * 30)
    parser.add_argument("--chunks-lat", type=int, default=100)
    parser.add_argument("--chunks-lon", type=int, default=100)
    args = parser.parse_args()

    day_start_hour_local = int(args.day_start_hour_local)
    if args.day_start_hour is not None:
        day_start_hour_local = int(args.day_start_hour)
    if not (0 <= day_start_hour_local <= 23):
        raise SystemExit("--day-start-hour-local/--day-start-hour must be 0..23")

    years = parse_years(args.years)
    years_slug = years_filename_slug(years)
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    months_tag = "".join(months)

    paper_root: Path | None = args.paper_root
    if paper_root is None:
        env_paper = os.environ.get("PAPER_ROOT")
        if env_paper:
            paper_root = Path(env_paper)
        else:
            paper_root = resolve_paper_root(Path.cwd().resolve())

    module_root = os.environ.get("MODULE_ROOT")

    out_dir = resolve_out_dir(args.region_label, paper_root, module_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    tz_slug = _tz_slug(args.timezone)
    out_path = args.out_file if args.out_file is not None else (
        out_dir
        / (
            f"era5_daily_tpSum_tpMax_cpSum_capeMax_{args.region_label}_{years_slug}_m{months_tag}"
            f"_dayStartLocal{day_start_hour_local:02d}_tz{tz_slug}.nc"
        )
    )

    chunks = {"time": args.chunks_time, "valid_time": args.chunks_time, "latitude": args.chunks_lat, "longitude": args.chunks_lon}
    ds = _open_hourly_inputs(
        region_label=args.region_label,
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

    missing = [v for v in ("tp", "cp", "cape") if v not in ds.data_vars]
    if missing:
        raise SystemExit(
            f"Expected variables tp/cp/cape in input dataset, missing: {missing}. Present: {list(ds.data_vars)}"
        )

    ds = ds[["tp", "cp", "cape"]].astype("float32")

    # Define "day" as local 06–06 (or configured hour) in the given timezone, handling DST.
    day_start_utc = local_day_start_time_utc(ds["time"], tz_name=args.timezone, day_start_hour_local=day_start_hour_local)
    tp_daily_sum = ds["tp"].groupby(day_start_utc).sum(dim="time").rename({"time_day_start": "time"})
    tp_daily_max = ds["tp"].groupby(day_start_utc).max(dim="time").rename({"time_day_start": "time"})
    cp_daily_sum = ds["cp"].groupby(day_start_utc).sum(dim="time").rename({"time_day_start": "time"})
    cape_daily_max = ds["cape"].groupby(day_start_utc).max(dim="time").rename({"time_day_start": "time"})

    mjjas = [int(m) for m in months]
    tp_daily_sum = tp_daily_sum.sel(time=tp_daily_sum["time.month"].isin(mjjas))
    tp_daily_max = tp_daily_max.sel(time=tp_daily_max["time.month"].isin(mjjas))
    cp_daily_sum = cp_daily_sum.sel(time=cp_daily_sum["time.month"].isin(mjjas))
    cape_daily_max = cape_daily_max.sel(time=cape_daily_max["time.month"].isin(mjjas))

    out = xr.Dataset(
        {
            "tp_daily_sum": tp_daily_sum,
            "tp_daily_max": tp_daily_max,
            "cp_daily_sum": cp_daily_sum,
            "cape_daily_max": cape_daily_max,
        }
    )
    out.attrs.update(
        {
            "source": "ERA5 hourly single-levels (tp/cp/cape) aggregated to daily",
            "region_label": str(args.region_label),
            "years_spec": str(args.years),
            "years_slug": str(years_slug),
            "months_tag": str(months_tag),
            "timezone": str(args.timezone),
            "day_start_hour_local": int(day_start_hour_local),
            "input_file": str(args.in_file) if args.in_file is not None else "auto (legacy single file or chunked year-month inputs)",
            "months": months,
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=encoding)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

