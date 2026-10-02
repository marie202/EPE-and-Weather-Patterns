#!/usr/bin/env python3
"""
EPE (epe_00): Download ERA5 variables needed for convective extreme precipitation events.

Writes raw downloads under:
  data/raw/era5/extreme_precip_events/<region_label>/

Downloads hourly ERA5 single-level fields:
  - total precipitation (tp)
  - convective precipitation (cp)
  - CAPE (cape)

Notes
-----
- CDS "area" order is [North, West, South, East].
- For robust daily extremes (e.g. 06–06 UTC) you typically want *hourly* data.
  This script downloads hourly values (00–23 UTC). Aggregate in epe_01.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import xarray as xr
import pandas as pd

from region_config import bbox_cds_area_list, find_region_yaml_key, load_region_config, resolve_step_root


def _ensure_cds_credentials() -> None:
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return
    dotrc = os.environ.get("CDSAPI_RC", os.path.expanduser("~/.cdsapirc"))
    if not os.path.isfile(dotrc):
        raise SystemExit(
            "CDS API credentials not found.\n"
            f"  Tried config file: {dotrc}\n"
            "  Create ~/.cdsapirc with 'url:' and 'key:' or set CDSAPI_URL/CDSAPI_KEY.\n"
            "  See: https://cds.climate.copernicus.eu/api-how-to\n"
        )
    text = Path(dotrc).read_text(encoding="utf-8", errors="replace")
    if "url:" not in text or "key:" not in text:
        raise SystemExit(f"CDS config at {dotrc} must contain lines starting with 'url:' and 'key:'.")


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


def parse_area(spec: str) -> list[float]:
    """
    Parse CDS bbox "N,W,S,E". Accepts comma or semicolon separators (semicolons are
    Slurm-safe in `sbatch --export=AREA_STR=55.1;5.9;47.2;15.0`).
    """
    sep = ";" if ";" in spec else ","
    tokens = [t.strip() for t in spec.split(sep) if t.strip()]
    if len(tokens) != 4:
        raise ValueError(
            f'--area must have four values "N,W,S,E" (comma- or semicolon-separated); got {len(tokens)} from {spec!r}'
        )
    return [float(x) for x in tokens]


def resolve_area_from_region_label(region_label: str, *, paper_root: Path | None, module_root: str | None) -> list[float]:
    cwd = Path.cwd().resolve()
    if paper_root is not None:
        step_root = paper_root
    elif module_root:
        step_root = Path(module_root)
    else:
        step_root = resolve_step_root(cwd)
    yaml_key = find_region_yaml_key(step_root, region_label)
    cfg = load_region_config(step_root=step_root, region_id=yaml_key)
    return bbox_cds_area_list(cfg.bbox)


def resolve_out_dir(region_label: str, paper_root: Path | None, module_root: str | None) -> Path:
    if paper_root is not None:
        return paper_root / "data" / "raw" / "era5" / "extreme_precip_events" / region_label
    if module_root:
        return Path(module_root) / "data" / "raw" / "era5" / "extreme_precip_events" / region_label
    return Path.cwd() / "data" / "raw" / "era5" / "extreme_precip_events" / region_label


def extract_single_nc(zip_path: Path, *, out_nc_path: Path) -> None:
    with tempfile.TemporaryDirectory(dir=zip_path.parent) as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)
        nc_files = sorted(tmp_path.rglob("*.nc"))
        if len(nc_files) != 1:
            raise RuntimeError(f"Expected exactly 1 .nc inside the CDS zip, found {len(nc_files)}: {nc_files}")
        out_nc_path.unlink(missing_ok=True)
        shutil.move(str(nc_files[0]), str(out_nc_path))


def extract_merge_ncs(zip_path: Path, *, out_nc_path: Path) -> None:
    """
    CDS may return multiple NetCDFs per ZIP (e.g. separate streams for accumulated vs instantaneous fields).
    Merge them into a single NetCDF so downstream scripts see tp/cp/cape in one file.
    """
    with tempfile.TemporaryDirectory(dir=zip_path.parent) as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)

        nc_files = sorted(tmp_path.rglob("*.nc"))
        if not nc_files:
            raise RuntimeError(f"No .nc found inside the CDS zip: {zip_path}")

        if len(nc_files) == 1:
            out_nc_path.unlink(missing_ok=True)
            shutil.move(str(nc_files[0]), str(out_nc_path))
            return

        dsets: list[xr.Dataset] = []
        for p in nc_files:
            dsets.append(xr.open_dataset(p, chunks={"time": 24 * 31, "valid_time": 24 * 31}))

        merged = xr.merge(dsets, compat="override", join="outer")
        out_nc_path.unlink(missing_ok=True)
        encoding = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
        merged.to_netcdf(out_nc_path, encoding=encoding)


def build_single_levels_request(*, years: list[str], months: list[str], times: list[str], area: list[float]) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": [
            "total_precipitation",
            "convective_precipitation",
            "convective_available_potential_energy",
        ],
        "year": years,
        "month": months,
        "day": [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
            "31",
        ],
        "time": times,
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }


def hourly_times() -> list[str]:
    return [f"{h:02d}:00" for h in range(24)]


def _guess_time_coord(ds: xr.Dataset) -> str:
    for cand in ("valid_time", "time"):
        if cand in ds.coords:
            return cand
    for cand in ("valid_time", "time"):
        if cand in ds.variables:
            return cand
    raise KeyError(f"Could not find time coordinate; coords={list(ds.coords)} vars={list(ds.variables)}")


def validate_yearly_mjjas_file(path: Path, *, year: str, months: list[str]) -> tuple[bool, str]:
    """
    Best-effort sanity check to decide if an existing yearly file is "complete enough" to skip.
    We verify it opens, has required variables, and covers all requested months for that year.
    """
    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        return False, f"could not open: {e!r}"

    try:
        want_vars = {"tp", "cp", "cape"}
        have_vars = set(ds.data_vars)
        if not want_vars.issubset(have_vars):
            return False, f"missing vars: {sorted(want_vars - have_vars)}"

        tname = _guess_time_coord(ds)
        dt = pd.to_datetime(ds[tname].values)
        idx = pd.DatetimeIndex(dt)

        years_present = set(int(y) for y in idx.year.unique())
        if int(year) not in years_present:
            return False, f"year {year} not present (years seen: {sorted(years_present)[:10]}...)"

        months_present = set(int(m) for m in idx.month.unique())
        want_months = set(int(m) for m in months)
        if not want_months.issubset(months_present):
            missing = sorted(want_months - months_present)
            return False, f"missing months: {missing} (present: {sorted(months_present)})"

        return True, "ok"
    finally:
        ds.close()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _download_one(client: cdsapi.Client, *, request: dict, zip_path: Path, out_nc_path: Path, overwrite: bool) -> bool:
    """
    Returns True if a download happened, False if skipped (already exists and not overwriting).
    """
    if not overwrite and out_nc_path.exists():
        print(f"Skip (exists): {out_nc_path.name}")
        return False

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _safe_unlink(zip_path)
        _safe_unlink(out_nc_path)

    client.retrieve("reanalysis-era5-single-levels", request).download(str(zip_path))
    extract_merge_ncs(zip_path, out_nc_path=out_nc_path)
    _safe_unlink(zip_path)
    print(f"Wrote: {out_nc_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="EPE: download hourly ERA5 tp/cp/cape (single-levels).")
    parser.add_argument("--region-label", default="germany", help="Output subfolder under extreme_precip_events/.")
    parser.add_argument("--years", default="2023-2025", help='Years as "2023-2025" or comma list.')
    parser.add_argument("--months", default="05,06,07,08,09", help='Comma-separated months (zero-padded), default MJJAS.')
    parser.add_argument(
        "--area",
        default=None,
        help='Bounding box "N,W,S,E" (CDS order). Use commas or semicolons. '
        "Omit when using --use-region-bbox.",
    )
    parser.add_argument(
        "--use-region-bbox",
        action="store_true",
        help="Load N,W,S,E from src/epe_preprocessing/regions.yaml for --region-label (recommended on Slurm).",
    )
    parser.add_argument("--paper-root", type=Path, default=None, help="Path to `1_Detect_Weather_Patterns/`.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NetCDF outputs if present.")
    parser.add_argument(
        "--chunk-by",
        choices=["year-month", "year", "all"],
        default="year",
        help="Split CDS requests: 'year' = one NetCDF per year (…_yYYYY_m<months>_hourly.nc; default for s12). "
        "'year-month' = one file per year×month (smaller CDS jobs). 'all' = single file for full year span.",
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    for m in months:
        if not re.fullmatch(r"\d{2}", m) or not (1 <= int(m) <= 12):
            raise SystemExit(f"invalid month {m!r} in --months")

    paper_root: Path | None = args.paper_root
    if paper_root is None:
        env_paper = os.environ.get("PAPER_ROOT")
        if env_paper:
            paper_root = Path(env_paper)
        else:
            paper_root = resolve_paper_root(Path.cwd().resolve())

    module_root = os.environ.get("MODULE_ROOT")

    if args.use_region_bbox:
        if args.area is not None:
            raise SystemExit("Use only one of --use-region-bbox or --area.")
        area = resolve_area_from_region_label(args.region_label, paper_root=paper_root, module_root=module_root)
        print(f"Using bbox from regions.yaml for {args.region_label!r}: {area}")
    elif args.area is None:
        area = resolve_area_from_region_label(args.region_label, paper_root=paper_root, module_root=module_root)
        print(f"Using bbox from regions.yaml for {args.region_label!r}: {area}")
    else:
        try:
            area = parse_area(args.area)
        except ValueError as e:
            raise SystemExit(
                f"{e}\n"
                "  If submitting via Slurm, commas in `sbatch --export=AREA_STR=...` are split into separate "
                "variables. Prefer `export USE_REGION_BBOX=1` + `sbatch --export=ALL`, or "
                "`export BBOX_N=... BBOX_W=... BBOX_S=... BBOX_E=...`, or semicolons in AREA_STR.\n"
            ) from e

    out_dir = args.out_dir if args.out_dir is not None else resolve_out_dir(args.region_label, paper_root, module_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    years_slug = years_filename_slug(years)
    months_tag = "".join(months)
    times = hourly_times()

    _ensure_cds_credentials()
    client = cdsapi.Client()

    if args.chunk_by == "all":
        zip_path = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_{years_slug}_m{months_tag}_hourly.zip"
        out_nc_path = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_{years_slug}_m{months_tag}_hourly.nc"
        request = build_single_levels_request(years=years, months=months, times=times, area=area)
        _download_one(client, request=request, zip_path=zip_path, out_nc_path=out_nc_path, overwrite=args.overwrite)
        return

    if args.chunk_by == "year":
        wrote = 0
        lines = []
        for y in years:
            zip_path = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_y{y}_m{months_tag}_hourly.zip"
            out_nc_path = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_y{y}_m{months_tag}_hourly.nc"
            request = build_single_levels_request(years=[y], months=months, times=times, area=area)
            if out_nc_path.exists() and not args.overwrite:
                ok, why = validate_yearly_mjjas_file(out_nc_path, year=y, months=months)
                if ok:
                    print(f"Skip (exists, validated): {out_nc_path.name}")
                else:
                    print(f"Re-download (exists but invalid: {why}): {out_nc_path.name}")
                    if _download_one(client, request=request, zip_path=zip_path, out_nc_path=out_nc_path, overwrite=True):
                        wrote += 1
            else:
                if _download_one(client, request=request, zip_path=zip_path, out_nc_path=out_nc_path, overwrite=args.overwrite):
                    wrote += 1
            lines.append(out_nc_path.name)

        manifest = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_{years_slug}_m{months_tag}_hourly_MANIFEST.txt"
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote manifest: {manifest} ({len(lines)} files listed, {wrote} downloaded this run)")
        return

    # year-month: submit many smaller CDS jobs (year × month) to stay under cost limits.
    wrote = 0
    for y in years:
        for m in months:
            zip_path = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_y{y}_m{m}_hourly.zip"
            out_nc_path = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_y{y}_m{m}_hourly.nc"
            request = build_single_levels_request(years=[y], months=[m], times=times, area=area)
            if _download_one(client, request=request, zip_path=zip_path, out_nc_path=out_nc_path, overwrite=args.overwrite):
                wrote += 1

    # Manifest for downstream scripts / debugging
    manifest = out_dir / f"era5_single_levels_tp-cp-cape_{args.region_label}_{years_slug}_m{months_tag}_hourly_MANIFEST.txt"
    lines = []
    for y in years:
        for m in months:
            lines.append(f"era5_single_levels_tp-cp-cape_{args.region_label}_y{y}_m{m}_hourly.nc")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest} ({len(lines)} files listed, {wrote} downloaded this run)")


if __name__ == "__main__":
    main()

