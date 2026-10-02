#!/usr/bin/env python3
"""
Download ERA5 Z500 (geopotential at 500 hPa) from the CDS API, unzip, and rename to a stable .nc path.

Mirrors the logic in `1_CDS_API_Request.ipynb`. CDS ``area`` order is [North, West, South, East].

HPC / Levante: run from the ``1_Detect_Weather_Patterns`` directory via
``the removed download job script`` (Singularity job; this file under ``src/``).
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


def _ensure_cds_credentials() -> None:
    """Fail fast with a clear message if cdsapi has nothing to authenticate with."""
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return
    dotrc = os.environ.get("CDSAPI_RC", os.path.expanduser("~/.cdsapirc"))
    if not os.path.isfile(dotrc):
        root_hint = os.environ.get("DETECT_WEATHER_ROOT") or "/work/bb1546/detect_weather"
        raise SystemExit(
            "CDS API credentials not found.\n"
            f"  Tried config file: {dotrc}\n"
            "  Fix one of:\n"
            "    • Create ~/.cdsapirc with 'url:' and 'key:' (see CDS web account).\n"
            f"    • Or on HPC: copy to your project tree, e.g.\n"
            f"        cp ~/.cdsapirc {root_hint}/.cdsapirc && chmod 600 {root_hint}/.cdsapirc\n"
            f"      and export CDSAPI_RC={root_hint}/.cdsapirc\n"
            "      (01_sbatch_cds_era5_z500_download.sh sets CDSAPI_RC when that file exists.)\n"
            "    • Or set environment variables CDSAPI_URL and CDSAPI_KEY.\n"
        )
    text = Path(dotrc).read_text(encoding="utf-8", errors="replace")
    if "url:" not in text or "key:" not in text:
        raise SystemExit(
            f"CDS config at {dotrc} must contain lines starting with 'url:' and 'key:' "
            "(see https://cds.climate.copernicus.eu/api-how-to)."
        )


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
    """Unambiguous ``years_…`` segment for zip/NetCDF names (actual calendar years requested)."""
    yints = sorted(int(y) for y in years)
    if len(yints) == 1:
        return f"years_{yints[0]}"
    contiguous = all(b == a + 1 for a, b in zip(yints, yints[1:]))
    if contiguous:
        return f"years_{yints[0]}-{yints[-1]}"
    return "years_" + "_".join(str(y) for y in yints)


def apply_coord_extension(area: list[float], extend_deg: float) -> list[float]:
    """Expand [N, W, S, E] by ``extend_deg`` on each side (lat/lon)."""
    if extend_deg == 0:
        return area
    n, w, s, e = area
    return [n + extend_deg, w - extend_deg, s - extend_deg, e + extend_deg]


def resolve_paper_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / "1_Detect_Weather_Patterns").is_dir():
            return p / "1_Detect_Weather_Patterns"
    return None


def resolve_out_dir(region_label: str, paper_root: Path | None, module_root: str | None) -> Path:
    if paper_root is not None:
        return paper_root / "data" / "raw" / "era5" / "z500" / region_label
    if module_root:
        return Path(module_root) / "data" / "raw" / "era5" / "z500" / region_label
    return Path.cwd() / "data" / "raw" / "era5" / "z500" / region_label


def build_request(
    *,
    years: list[str],
    months: list[str],
    time_hhmm: str,
    area: list[float],
) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "pressure_level": ["500"],
        "year": years,
        "month": months,
        "day": [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
            "31",
        ],
        "time": [time_hhmm],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CDS ERA5 Z500 download (MJJAS-style months, single daily time).")
    parser.add_argument(
        "--region-label",
        required=True,
        help="Subfolder name under .../era5/z500/, e.g. europe, northamerica.",
    )
    parser.add_argument(
        "--years",
        required=True,
        help='Years as "2020-2023" or comma list "2020,2021,2022".',
    )
    parser.add_argument(
        "--area",
        required=True,
        help='Bounding box "N,W,S,E" in degrees (CDS order: North, West, South, East), e.g. "90,-80,30,40".',
    )
    parser.add_argument(
        "--coord-extend-deg",
        type=float,
        default=0.0,
        help="Expand the box by this many degrees on each side (applied to N/W/S/E).",
    )
    parser.add_argument(
        "--months",
        default="05,06,07,08,09",
        help='Comma-separated zero-padded months, default MJJAS: "05,06,07,08,09".',
    )
    parser.add_argument(
        "--time",
        dest="time_hhmm",
        default="12:00",
        help='CDS time of day, default "12:00".',
    )
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=None,
        help="Path to the `1_Detect_Weather_Patterns` directory (sets data/... under it).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: auto from paper-root / MODULE_ROOT / cwd).",
    )
    parser.add_argument(
        "--dataset",
        default="reanalysis-era5-pressure-levels",
        help="CDS dataset id.",
    )
    args = parser.parse_args()

    area_tokens = [t.strip() for t in args.area.split(",")]
    if len(area_tokens) != 4:
        raise SystemExit(f'--area must have four comma-separated numbers "N,W,S,E"; got {args.area!r}')
    base_area = [float(x) for x in area_tokens]
    area = apply_coord_extension(base_area, args.coord_extend_deg)

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
            found = resolve_paper_root(Path.cwd().resolve())
            paper_root = found

    module_root = os.environ.get("MODULE_ROOT")

    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = resolve_out_dir(args.region_label, paper_root, module_root)

    out_dir.mkdir(parents=True, exist_ok=True)

    years_slug = years_filename_slug(years)
    months_tag = "".join(months)
    time_slug = args.time_hhmm.replace(":", "")
    zip_path = out_dir / f"era5_z500_{args.region_label}_{years_slug}_m{months_tag}_{time_slug}UTC.zip"
    out_nc_path = out_dir / f"era5_z500_{args.region_label}_{years_slug}_m{months_tag}_{time_slug}UTC.nc"

    request = build_request(
        years=years,
        months=months,
        time_hhmm=args.time_hhmm,
        area=area,
    )

    print(f"region_label={args.region_label}")
    print(f"years={years}  filename_years={years_slug}")
    print(f"base_area={base_area}  coord_extend_deg={args.coord_extend_deg}  effective_area={area}")
    print(f"out_dir={out_dir}")
    print(f"zip_path={zip_path}")

    _ensure_cds_credentials()
    client = cdsapi.Client()
    client.retrieve(args.dataset, request).download(str(zip_path))

    # Leftovers from older script versions (extracted *.nc next to the zip) — safe to remove.
    for orphan in sorted(out_dir.glob("data_stream-oper*.nc")):
        print(f"Removing stale CDS extract file: {orphan.name}")
        orphan.unlink(missing_ok=True)

    # Extract into a temp dir so older *.nc in out_dir (e.g. prior runs) do not break discovery.
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)
        nc_files = sorted(tmp_path.rglob("*.nc"))
        if len(nc_files) != 1:
            raise RuntimeError(
                f"Expected exactly 1 .nc inside the CDS zip, found {len(nc_files)}: {nc_files}"
            )
        out_nc_path.unlink(missing_ok=True)
        shutil.move(str(nc_files[0]), str(out_nc_path))

    print(f"Wrote: {out_nc_path}")


if __name__ == "__main__":
    main()
