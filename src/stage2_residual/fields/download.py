from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from .field_specs import FieldSpec, get_field_spec


def _ensure_cds_credentials() -> None:
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return
    dotrc = os.environ.get("CDSAPI_RC", os.path.expanduser("~/.cdsapirc"))
    if not os.path.isfile(dotrc):
        root_hint = os.environ.get("DETECT_WEATHER_ROOT") or "/beegfs/scratch/marieeckert/detect_weather"
        raise SystemExit(
            "CDS API credentials not found.\n"
            f"  Tried config file: {dotrc}\n"
            "  Create ~/.cdsapirc with 'url:' and 'key:' or copy to project tree, e.g.\n"
            f"    cp ~/.cdsapirc {root_hint}/.cdsapirc && chmod 600 {root_hint}/.cdsapirc\n"
            f"  and export CDSAPI_RC={root_hint}/.cdsapirc before sbatch.\n"
        )
    text = Path(dotrc).read_text(encoding="utf-8", errors="replace")
    if "url:" not in text or "key:" not in text:
        raise SystemExit(f"CDS config at {dotrc} must contain 'url:' and 'key:' lines.")


def apply_coord_extension(area: list[float], extend_deg: float) -> list[float]:
    if extend_deg == 0:
        return area
    n, w, s, e = area
    return [n + extend_deg, w - extend_deg, s - extend_deg, e + extend_deg]


def parse_years(spec: str) -> list[str]:
    raw = str(spec).strip()
    if not raw:
        raise ValueError("years spec must not be empty")
    if "-" in raw:
        a, b = (int(tok.strip()) for tok in raw.split("-", 1))
        if b < a:
            raise ValueError(f"Invalid years range {spec!r}")
        return [str(year) for year in range(a, b + 1)]
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def years_filename_slug(years: list[str]) -> str:
    if len(years) == 1:
        return f"years_{years[0]}"
    return f"years_{years[0]}-{years[-1]}"


def build_request(*, field_spec: FieldSpec, years: list[str], months: list[str], time_hhmm: str, area: list[float]) -> dict:
    request = {
        "product_type": ["reanalysis"],
        "variable": [field_spec.cds_variable],
        "year": years,
        "month": months,
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [time_hhmm],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }
    if field_spec.pressure_level_hpa is not None:
        request["pressure_level"] = [str(int(field_spec.pressure_level_hpa))]
    return request


def resolve_out_dir(module_root: Path, *, field_spec: FieldSpec, region: str) -> Path:
    return module_root / "data" / "multivariate" / "raw" / "era5" / field_spec.raw_dir_name / region


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic CDS ERA5 downloader for multivariate workflow fields.")
    parser.add_argument("--field-key", required=True, help="Field key such as z500, vo850, vo500.")
    parser.add_argument("--region-label", required=True)
    parser.add_argument("--years", required=True)
    parser.add_argument("--area", required=True, help='Bounding box "N,W,S,E".')
    parser.add_argument(
        "--coord-extend-deg",
        type=float,
        default=0.0,
        help="Expand the bbox by this many degrees on each side (N/W/S/E).",
    )
    parser.add_argument("--months", default="05,06,07,08,09")
    parser.add_argument("--time", dest="time_hhmm", default="12:00")
    parser.add_argument("--module-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        default=None,
        help="CDS dataset id. Defaults to the field spec (pressure-levels or single-levels).",
    )
    args = parser.parse_args()

    _ensure_cds_credentials()
    field_spec = get_field_spec(args.field_key)
    if field_spec.key == "z500":
        raise SystemExit(
            "Z500 is provided by the legacy pipeline (data/raw/era5/z500/). "
            "Use the removed download job script or omit z500 from multivariate CDS jobs."
        )
    dataset = str(args.dataset or field_spec.cds_dataset)
    years = parse_years(args.years)
    months = [tok.strip() for tok in str(args.months).split(",") if tok.strip()]
    area_tokens = [tok.strip() for tok in str(args.area).split(",")]
    if len(area_tokens) != 4:
        raise SystemExit(f'--area must contain "N,W,S,E", got {args.area!r}')
    area = apply_coord_extension([float(x) for x in area_tokens], float(args.coord_extend_deg))

    out_dir = resolve_out_dir(args.module_root.resolve(), field_spec=field_spec, region=str(args.region_label))
    out_dir.mkdir(parents=True, exist_ok=True)
    years_slug = years_filename_slug(years)
    months_tag = "".join(months)
    time_slug = str(args.time_hhmm).replace(":", "")
    base = f"era5_{field_spec.key}_{args.region_label}_{years_slug}_m{months_tag}_{time_slug}UTC"
    zip_path = out_dir / f"{base}.zip"
    out_nc_path = out_dir / f"{base}.nc"

    # Skip download if output file already exists
    if out_nc_path.exists() and out_nc_path.stat().st_size > 0:
        print(f"✓ File already exists (skipping download): {out_nc_path}")
        return

    try:
        import cdsapi  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Downloading ERA5 data requires cdsapi.") from exc

    client = cdsapi.Client()
    request = build_request(
        field_spec=field_spec,
        years=years,
        months=months,
        time_hhmm=str(args.time_hhmm),
        area=area,
    )
    print(f"Downloading {field_spec.key} for {args.region_label}...")
    client.retrieve(dataset, request).download(str(zip_path))
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)
        nc_files = sorted(tmp_path.rglob("*.nc"))
        if len(nc_files) != 1:
            raise RuntimeError(f"Expected exactly 1 .nc file in CDS zip, found {len(nc_files)}")
        out_nc_path.unlink(missing_ok=True)
        shutil.move(str(nc_files[0]), str(out_nc_path))
    # Clean up zip file to save space
    zip_path.unlink(missing_ok=True)
    print(f"✓ Downloaded and extracted: {out_nc_path}")


if __name__ == "__main__":
    main()
