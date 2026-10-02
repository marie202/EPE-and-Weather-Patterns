"""Resolve ``10_regime_composite_z500_raw*.nc`` paths (shared by PDF and notebooks).

Self-contained path filtering (mirrors ``common.resolver`` logic) using only
``preprocessing_helpers.bbox_slug`` — no import from ``summary_reports``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

from common.preprocessing_helpers import bbox_slug


def _norm_step_token(step: str | None) -> str | None:
    if step is None:
        return None
    s = str(step).strip().upper()
    return s if s else None


def _bbox_from_attrs(ds: xr.Dataset) -> dict[str, float] | None:
    keys = ("bbox_N", "bbox_W", "bbox_S", "bbox_E")
    if not all(k in ds.attrs for k in keys):
        return None
    return {k[5]: float(ds.attrs[k]) for k in keys}


def _bbox_close(a: dict[str, float], b: dict[str, float], *, tol: float) -> bool:
    for k in ("N", "W", "S", "E"):
        if abs(float(a[k]) - float(b[k])) > tol:
            return False
    return True


def attrs_match_step_bbox(
    ds: xr.Dataset,
    *,
    step: str,
    bbox_expected: dict[str, float],
    tol_deg: float = 0.02,
) -> bool:
    step_u = _norm_step_token(step)
    assert step_u is not None
    bs = ds.attrs.get("bbox_step")
    if bs is not None and str(bs).strip().upper() != step_u:
        return False
    got = _bbox_from_attrs(ds)
    if got is None:
        return bs is not None and str(bs).strip().upper() == step_u
    return _bbox_close(got, bbox_expected, tol=tol_deg)


def filter_candidates_by_attrs(
    paths: list[Path],
    *,
    step: str,
    bbox_expected: dict[str, float],
    tol_deg: float = 0.02,
) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        try:
            with xr.open_dataset(p, chunks=None) as ds:
                if attrs_match_step_bbox(ds, step=step, bbox_expected=bbox_expected, tol_deg=tol_deg):
                    out.append(p)
        except Exception:
            continue
    return out


def filter_candidates_by_filename_step(paths: list[Path], step: str) -> list[Path]:
    tok = _norm_step_token(step)
    assert tok is not None
    needle = f"_step{tok}_"
    return [p for p in paths if needle in p.name]


def filter_candidates_by_bbox_slug(paths: list[Path], bbox_expected: dict[str, float]) -> list[Path]:
    slug = bbox_slug(bbox_expected)
    return [p for p in paths if slug in p.name]


def _sample_spatial_regime_da(ds: xr.Dataset) -> xr.DataArray | None:
    preferred = (
        "regime_composite_anom",
        "composite_weighted",
        "regime_composite_z500_raw",
        "regime_composite_weighted",
        "regime_composite_norm",
    )
    for v in preferred:
        if v in ds.data_vars:
            da = ds[v]
            if "regime" in da.dims:
                return da
    for v in sorted(ds.data_vars, key=str):
        da = ds[v]
        if "regime" not in da.dims:
            continue
        for lat in ("latitude", "lat"):
            for lon in ("longitude", "lon"):
                if lat in da.dims and lon in da.dims:
                    return da
    return None


def dataset_spatial_corners_match_bbox(
    ds: xr.Dataset,
    bbox_expected: dict[str, float],
    *,
    tol_deg: float = 1.75,
) -> bool:
    da = _sample_spatial_regime_da(ds)
    if da is None:
        return False
    lat_dim = lon_dim = None
    for lat in ("latitude", "lat"):
        for lon in ("longitude", "lon"):
            if lat in da.dims and lon in da.dims:
                lat_dim, lon_dim = lat, lon
                break
        if lat_dim is not None:
            break
    if lat_dim is None or lon_dim is None:
        return False
    lat = np.asarray(da[lat_dim].values, dtype=float)
    lon = np.asarray(da[lon_dim].values, dtype=float)
    lat_s, lat_n = float(np.nanmin(lat)), float(np.nanmax(lat))
    lonv = lon.copy()
    if float(np.nanmax(lonv)) > 180.0 or float(np.nanmin(lonv)) > 180.0:
        lonv = ((lonv + 180.0) % 360.0) - 180.0
    lon_w, lon_e = float(np.nanmin(lonv)), float(np.nanmax(lonv))
    if lon_w > lon_e:
        return False
    S = float(bbox_expected["S"])
    N = float(bbox_expected["N"])
    W = float(bbox_expected["W"])
    E = float(bbox_expected["E"])
    return (
        abs(lat_s - S) <= tol_deg
        and abs(lat_n - N) <= tol_deg
        and abs(lon_w - W) <= tol_deg
        and abs(lon_e - E) <= tol_deg
    )


def filter_candidates_by_spatial_grid_bbox(
    paths: list[Path],
    bbox_expected: dict[str, float],
    *,
    tol_deg: float = 1.75,
) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        try:
            with xr.open_dataset(p, chunks=None) as ds:
                if dataset_spatial_corners_match_bbox(ds, bbox_expected, tol_deg=tol_deg):
                    out.append(p)
        except Exception:
            continue
    return out


def dataset_grid_covers_bbox(
    ds: xr.Dataset,
    bbox_inner: dict[str, float],
    *,
    pad_deg: float = 2.5,
) -> bool:
    """True if the outer lat/lon grid of ``ds`` fully covers ``bbox_inner`` (N,W,S,E) with padding.

    Used to reuse a large-domain (e.g. B0) raw composite NetCDF for smaller bbox steps when corner
    matching to YAML is slightly off but the grid still spans the analysis window.
    """
    da = _sample_spatial_regime_da(ds)
    if da is None:
        return False
    lat_dim = lon_dim = None
    for lat in ("latitude", "lat"):
        for lon in ("longitude", "lon"):
            if lat in da.dims and lon in da.dims:
                lat_dim, lon_dim = lat, lon
                break
        if lat_dim is not None:
            break
    if lat_dim is None or lon_dim is None:
        return False
    lat = np.asarray(da[lat_dim].values, dtype=float)
    lon = np.asarray(da[lon_dim].values, dtype=float)
    lat_lo, lat_hi = float(np.nanmin(lat)), float(np.nanmax(lat))
    lonv = lon.copy()
    if float(np.nanmax(lonv)) > 180.0 or float(np.nanmin(lonv)) > 180.0:
        lonv = ((lonv + 180.0) % 360.0) - 180.0
    lon_lo, lon_hi = float(np.nanmin(lonv)), float(np.nanmax(lonv))
    if lon_lo > lon_hi:
        return False
    N, S = float(bbox_inner["N"]), float(bbox_inner["S"])
    W, E = float(bbox_inner["W"]), float(bbox_inner["E"])
    if S > N:
        S, N = N, S
    if W > E:
        return False
    if lat_lo > S + pad_deg + 1e-6:
        return False
    if lat_hi < N - pad_deg - 1e-6:
        return False
    if lon_lo > W - pad_deg + 1e-6:
        return False
    if lon_hi < E + pad_deg - 1e-6:
        return False
    return True


def filter_candidates_covering_bbox(
    paths: list[Path],
    bbox_inner: dict[str, float],
    *,
    pad_deg: float = 2.5,
) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        try:
            with xr.open_dataset(p, chunks=None) as ds:
                if dataset_grid_covers_bbox(ds, bbox_inner, pad_deg=pad_deg):
                    out.append(p)
        except Exception:
            continue
    return out


def collect_raw_composite_nc_candidates(module_root: Path, region: str) -> list[Path]:
    d = module_root / "results" / "regimes" / region
    if not d.is_dir():
        return []
    patterns = (
        "10_regime_composite_z500_raw*.nc",
        "10_regime_composite*z500*raw*.nc",
        "*regime_composite*z500_raw*.nc",
    )
    seen: dict[Path, Path] = {}
    for pat in patterns:
        for p in d.glob(pat):
            if "raw" not in p.name.lower():
                continue
            seen[p.resolve()] = p
    return list(seen.values())


def _filename_matches_k(name: str, k: int) -> bool:
    """Match ``_k07_`` / ``_k07.`` and single-digit ``_k7_`` / ``...k7_`` when *k* < 10."""
    s = Path(name).name
    if re.search(rf"_k{k:02d}(?:_|\.|$)", s):
        return True
    if k < 10:
        if re.search(rf"_k{k}(?!\d)(?:_|\.|$)", s):
            return True
        return bool(re.search(rf"(?<![0-9])k{k}(?![0-9])(?:_|\.|$)", s, re.I))
    return False


def pool_prefer_k_year_mtime(pool: list[Path], k: int, year_span_hint: str) -> Path | None:
    out = pool
    khit = [p for p in out if _filename_matches_k(p.name, k)]
    if khit:
        out = khit
    if year_span_hint:
        yhit = [p for p in out if year_span_hint in p.name]
        if yhit:
            out = yhit
    return max(out, key=lambda p: p.stat().st_mtime) if out else None


def pick_raw_z500_nc_path_impl(
    work: list[Path],
    *,
    bbox_step: str,
    bbox_expected: dict[str, float],
    k: int,
    year_span_hint: str,
    skip_step_bbox_filters: bool = False,
) -> Path | None:

    def mtime_pick(paths: list[Path]) -> Path | None:
        if not paths:
            return None
        return pool_prefer_k_year_mtime(paths, k, year_span_hint)

    if skip_step_bbox_filters:
        return mtime_pick(work)

    by_attr = filter_candidates_by_attrs(work, step=bbox_step, bbox_expected=bbox_expected, tol_deg=0.02)
    if len(by_attr) == 1:
        return by_attr[0]
    if len(by_attr) > 1:
        fn = filter_candidates_by_filename_step(by_attr, bbox_step)
        if len(fn) == 1:
            return fn[0]
        slug_hit = filter_candidates_by_bbox_slug(by_attr, bbox_expected)
        if len(slug_hit) == 1:
            return slug_hit[0]
        pool = slug_hit if slug_hit else (fn if fn else by_attr)
        return mtime_pick(pool)

    fn_only = filter_candidates_by_filename_step(work, bbox_step)
    if len(fn_only) == 1:
        return fn_only[0]

    slug_all = filter_candidates_by_bbox_slug(work, bbox_expected)
    if len(slug_all) == 1:
        return slug_all[0]

    if len(fn_only) > 1:
        slug_fn = filter_candidates_by_bbox_slug(fn_only, bbox_expected)
        if len(slug_fn) == 1:
            return slug_fn[0]
        pool = slug_fn if slug_fn else fn_only
        return mtime_pick(pool)

    if len(slug_all) > 1:
        return mtime_pick(slug_all)

    return mtime_pick(work)


def resolve_regime_composite_z500_raw_path(
    module_root: Path,
    region: str,
    k: int,
    ten_eleven_nc: Path,
    *,
    bbox_step: str,
    bbox_expected: dict[str, float] | None,
    explicit_nc: Path | None = None,
    bbox_b0_expected: dict[str, float] | None = None,
) -> Path | None:
    if explicit_nc is not None:
        ep = explicit_nc.expanduser().resolve()
        if ep.is_file():
            return ep

    if bbox_expected is None:
        return None

    cands = collect_raw_composite_nc_candidates(module_root, region)
    ym = re.search(r"(19|20)\d{2}-(19|20)\d{2}", ten_eleven_nc.name)
    ys = ym.group(0) if ym else ""

    if not cands:
        return None

    grid_ok = filter_candidates_by_spatial_grid_bbox(cands, bbox_expected, tol_deg=1.75)
    skip_filters = False
    if grid_ok:
        work = grid_ok
    else:
        work = []
        if bbox_b0_expected is not None and str(bbox_step).upper() != "B0":
            b0_hits = filter_candidates_by_spatial_grid_bbox(cands, bbox_b0_expected, tol_deg=10.0)
            if b0_hits:
                work = b0_hits
                skip_filters = True
        if not work:
            cover = filter_candidates_covering_bbox(cands, bbox_expected, pad_deg=2.5)
            if cover:
                work = cover
                skip_filters = True
        if not work and bbox_b0_expected is not None and str(bbox_step).upper() != "B0":
            cover_b0 = filter_candidates_covering_bbox(cands, bbox_b0_expected, pad_deg=2.5)
            if cover_b0:
                work = cover_b0
                skip_filters = True
        if not work and len(cands) == 1:
            work = list(cands)
            skip_filters = True
        if not work:
            return None

    return pick_raw_z500_nc_path_impl(
        work,
        bbox_step=bbox_step,
        bbox_expected=bbox_expected,
        k=k,
        year_span_hint=ys,
        skip_step_bbox_filters=skip_filters,
    )


def pick_raw_composite_var(ds: xr.Dataset) -> xr.DataArray | None:
    if "regime_composite_z500_raw" in ds.data_vars:
        return ds["regime_composite_z500_raw"]

    def _has_lat_lon(da: xr.DataArray) -> bool:
        return any(lat in da.dims and lon in da.dims for lat in ("latitude", "lat") for lon in ("longitude", "lon"))

    def _is_non_raw_composite(name: str) -> bool:
        n = str(name).lower()
        return any(x in n for x in ("anom", "norm", "weighted", "composite_weighted"))

    scored: list[tuple[int, str]] = []
    for v in ds.data_vars:
        da = ds[v]
        if "regime" not in da.dims or not _has_lat_lon(da):
            continue
        sv = str(v).lower()
        if "raw" in sv and "z500" in sv:
            scored.append((0, str(v)))
        elif "raw" in sv:
            scored.append((1, str(v)))
        elif not _is_non_raw_composite(str(v)):
            scored.append((5, str(v)))
    if scored:
        scored.sort(key=lambda t: (t[0], t[1]))
        return ds[scored[0][1]]

    names = sorted(str(v) for v in ds.data_vars if str(v).startswith("regime_composite_"))
    if not names:
        return None
    return ds[names[0]]
