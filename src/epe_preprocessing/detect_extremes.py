#!/usr/bin/env python3
"""
EPE (epe_02): Detect raw and convective extremes from daily aggregates.

Input: output from `src/epe_preprocessing/epe_01_aggregate_daily.py` (daily NetCDF) with:
  - tp_daily_sum, tp_daily_max, cp_daily_sum, cape_daily_max

Output: cached extremes products including:
  - P_q (grid-point percentile threshold)
  - E_raw, E_conv (grid-point masks)
  - extreme_frac (area-weighted fraction of extreme grid points)
  - domain_extreme (optional)
  - tp_max_domain, event_score (ranking helpers)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr

from epe_preprocessing.epe_functions import cape_convective_ok, cape_filter_description, netcdf_attr_bool
from common.extremes_spec import q_filename_tag


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


def resolve_in_path(
    region_label: str,
    paper_root: Path | None,
    module_root: str | None,
    *,
    years_slug: str,
    months_tag: str,
    day_start_hour_local: int,
    timezone: str,
) -> Path:
    base = (
        (paper_root / "data" / "processed" / region_label / "extreme_precip_events")
        if paper_root is not None
        else (Path(module_root) / "data" / "processed" / region_label / "extreme_precip_events")
        if module_root
        else (Path.cwd() / "data" / "processed" / region_label / "extreme_precip_events")
    )
    tz_slug = (timezone or "UTC").replace("/", "-")
    return base / (
        f"era5_daily_tpSum_tpMax_cpSum_capeMax_{region_label}_{years_slug}_m{months_tag}"
        f"_dayStartLocal{day_start_hour_local:02d}_tz{tz_slug}.nc"
    )


def resolve_in_path_compat(
    region_label: str,
    paper_root: Path | None,
    module_root: str | None,
    *,
    years_slug: str,
    months_tag: str,
    day_start_hour_local: int,
    timezone: str,
) -> Path:
    """
    Prefer the new tz-aware daily filename; fall back to the legacy dayStartXX (UTC) name if needed.
    """
    p_new = resolve_in_path(
        region_label,
        paper_root,
        module_root,
        years_slug=years_slug,
        months_tag=months_tag,
        day_start_hour_local=day_start_hour_local,
        timezone=timezone,
    )
    if p_new.exists():
        return p_new
    # Legacy name (older pipeline versions): no timezone in name; dayStart was treated as UTC shift.
    base = p_new.parent
    p_old = base / f"era5_daily_tpSum_tpMax_cpSum_capeMax_{region_label}_{years_slug}_m{months_tag}_dayStart{day_start_hour_local:02d}.nc"
    return p_old


def resolve_out_dir(region_label: str, paper_root: Path | None, module_root: str | None) -> Path:
    if paper_root is not None:
        return paper_root / "data" / "processed" / region_label / "extreme_precip_events"
    if module_root:
        return Path(module_root) / "data" / "processed" / region_label / "extreme_precip_events"
    return Path.cwd() / "data" / "processed" / region_label / "extreme_precip_events"


def guess_lat_lon(da: xr.DataArray) -> tuple[str, str]:
    lat = "latitude" if "latitude" in da.coords else "lat" if "lat" in da.coords else None
    lon = "longitude" if "longitude" in da.coords else "lon" if "lon" in da.coords else None
    if lat is None or lon is None:
        raise KeyError(f"Could not find lat/lon coords in {list(da.coords)}")
    return lat, lon


def area_weights(lat: xr.DataArray) -> xr.DataArray:
    w = np.cos(np.deg2rad(lat))
    w = xr.DataArray(w, coords={lat.name: lat}, dims=(lat.name,))
    return w.clip(min=0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="EPE: detect raw + convective extremes from daily aggregates.")
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
    parser.add_argument("--precip-metric", default="tp_daily_sum", choices=["tp_daily_sum", "tp_daily_max", "cp_daily_sum"])
    parser.add_argument("--q", type=float, default=0.95)
    parser.add_argument("--cape-threshold", type=float, default=200.0)
    parser.add_argument("--cp-frac-min", type=float, default=0.0)
    parser.add_argument(
        "--cp-threshold",
        type=float,
        default=0.0,
        help="Require cp_daily_sum > cp_threshold. Default 0.0 → cp > 0 (strictly positive). Set < 0 to disable.",
    )
    parser.add_argument("--event-area-frac", type=float, default=0.01)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--paper-root", type=Path, default=None)
    parser.add_argument("--in-file", type=Path, default=None)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    day_start_hour_local = int(args.day_start_hour_local)
    if args.day_start_hour is not None:
        day_start_hour_local = int(args.day_start_hour)
    if not (0 <= day_start_hour_local <= 23):
        raise SystemExit("--day-start-hour-local/--day-start-hour must be 0..23")
    if not (0.0 < args.q < 1.0):
        raise SystemExit("--q must be between 0 and 1 (exclusive)")
    if args.cp_frac_min < 0.0 or args.cp_frac_min > 1.0:
        raise SystemExit("--cp-frac-min must be in [0,1]")
    if args.cp_threshold >= 0.0 and args.cp_frac_min > 0.0:
        raise SystemExit("--cp-threshold >= 0 and --cp-frac-min > 0 are mutually exclusive; use one or the other.")
    if args.event_area_frac < 0.0 or args.event_area_frac > 1.0:
        raise SystemExit("--event-area-frac must be in [0,1]")

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

    in_path = args.in_file if args.in_file is not None else resolve_in_path_compat(
        args.region_label,
        paper_root,
        module_root,
        years_slug=years_slug,
        months_tag=months_tag,
        day_start_hour_local=day_start_hour_local,
        timezone=args.timezone,
    )
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    out_dir = resolve_out_dir(args.region_label, paper_root, module_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cp_threshold >= 0.0:
        cp_tag = f"cp{int(round(args.cp_threshold))}"
    else:
        cp_tag = f"cpFracMin{int(round(args.cp_frac_min * 100)):02d}"
    area_tag = f"areaFrac{int(round(args.event_area_frac * 100)):03d}pct"
    tz_slug = (args.timezone or "UTC").replace("/", "-")
    q_tag = q_filename_tag(args.q)
    out_path = args.out_file if args.out_file is not None else (
        out_dir
        / f"extremes_{args.region_label}_{years_slug}_m{months_tag}_dayStartLocal{day_start_hour_local:02d}_tz{tz_slug}"
        f"_{args.precip_metric}_{q_tag}_cape{int(round(args.cape_threshold))}_{cp_tag}_{area_tag}.nc"
    )

    ds = xr.open_dataset(in_path, chunks={"time": 31, "latitude": 100, "longitude": 100})
    required = {"tp_daily_sum", "tp_daily_max", "cp_daily_sum", "cape_daily_max"}
    missing = sorted(required - set(ds.data_vars))
    if missing:
        raise SystemExit(f"Missing required variables in {in_path.name}: {missing}. Present: {list(ds.data_vars)}")

    P = ds[args.precip_metric]
    CAPEmax = ds["cape_daily_max"]
    lat_name, lon_name = guess_lat_lon(P)

    P_q = P.chunk({"time": -1}).quantile(args.q, dim="time", keep_attrs=False)
    if args.persist and hasattr(P_q.data, "persist"):
        P_q = P_q.persist()

    E_raw = (P > P_q)
    E_conv = E_raw & cape_convective_ok(CAPEmax, args.cape_threshold)
    if args.cp_frac_min > 0.0:
        tp_sum = ds["tp_daily_sum"]
        cp_sum = ds["cp_daily_sum"]
        cp_frac = cp_sum / tp_sum.where(tp_sum != 0)
        E_conv = E_conv & (cp_frac >= args.cp_frac_min)
    if args.cp_threshold >= 0.0:
        E_conv = E_conv & (ds["cp_daily_sum"] > args.cp_threshold)

    w = area_weights(ds[lat_name])
    extreme_frac = E_conv.weighted(w).mean(dim=(lat_name, lon_name))
    tp_max_domain = ds["tp_daily_max"].max(dim=(lat_name, lon_name))

    out = xr.Dataset(
        {
            "P_q": P_q,
            "E_raw": E_raw.astype("i1"),
            "E_conv": E_conv.astype("i1"),
            "P": P,
            "CAPE_max": CAPEmax,
            "extreme_frac": extreme_frac.astype("f4"),
            "domain_extreme": (extreme_frac >= args.event_area_frac).astype("i1"),
            "tp_max_domain": tp_max_domain.astype("f4"),
            "event_score": (extreme_frac * tp_max_domain).astype("f4"),
        }
    )
    out.attrs.update(
        {
            "source_daily_inputs": str(in_path),
            "region_label": str(args.region_label),
            "years_spec": str(args.years),
            "years_slug": str(years_slug),
            "months_tag": str(months_tag),
            "precip_metric": args.precip_metric,
            "q": float(args.q),
            "cape_threshold": float(args.cape_threshold),
            "cape_filter": cape_filter_description(args.cape_threshold),
            "cape_strictly_positive": netcdf_attr_bool(float(args.cape_threshold) <= 0.0),
            "cp_frac_min": float(args.cp_frac_min),
            "cp_threshold": float(args.cp_threshold),
            "timezone": str(args.timezone),
            "day_start_hour_local": int(day_start_hour_local),
            "event_area_frac_threshold": float(args.event_area_frac),
        }
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=encoding)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

