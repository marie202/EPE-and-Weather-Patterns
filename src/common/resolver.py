from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from common.preprocessing_helpers import bbox_slug, filter_paths_by_yeo_johnson, yeo_johnson_enabled

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for summary_reports (pip install pyyaml).")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return data


def load_bbox_steps_yaml(path: Path) -> dict[str, dict[str, Any]]:
    """Return `steps` mapping B0.. from bbox_steps_*.yaml."""
    data = _load_yaml(path)
    steps = data.get("steps", {})
    if not isinstance(steps, dict) or not steps:
        raise ValueError(f"No 'steps' in {path}")
    return steps  # type: ignore[return-value]


def load_k_overrides(path: Path) -> dict[str, Any]:
    return _load_yaml(path)


def k_for_step(cfg: dict[str, Any], step: str) -> int:
    """Resolve cluster count for bbox step Bn."""
    step = str(step).strip().upper()
    default_k = int(cfg.get("default_k", 7))
    m = cfg.get("k_by_step") or {}
    if not isinstance(m, dict):
        return default_k
    if step in m:
        return int(m[step])
    return default_k


def _norm_step_token(step: str | None) -> str | None:
    if step is None:
        return None
    s = str(step).strip().upper()
    return s if s else None


def _bbox_from_attrs(ds: xr.Dataset) -> dict[str, float] | None:
    keys = ("bbox_N", "bbox_W", "bbox_S", "bbox_E")
    if not all(k in ds.attrs for k in keys):
        return None
    return {k[5]: float(ds.attrs[k]) for k in keys}  # N,W,S,E


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
    """Match NetCDF global attrs to a bbox step (bbox_step + corners)."""
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
    """Keep paths whose datasets match bbox_step/bbox_* attrs; open each briefly."""
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
    """Match pipeline-style ``..._stepBn_N..W..S..E.._...nc`` tokens in filenames."""
    slug = bbox_slug(bbox_expected)
    return [p for p in paths if slug in p.name]


def _normalize_sigma_mode(value: str | None) -> str | None:
    v = str(value or "").strip().lower()
    if not v:
        return None
    if v in {"field", "grid", "gridpoint", "sigf"}:
        return "field"
    if v in {"scalar", "sigsc", "scalar_area_mean_cosphi", "area_mean", "domain_mean"}:
        return "scalar"
    return None


def filter_candidates_by_sigma_mode(paths: list[Path], sigma_mode: str | None) -> list[Path]:
    """Prefer candidates whose attr or filename matches the requested sigma mode."""
    target = _normalize_sigma_mode(sigma_mode)
    if target is None:
        return paths

    out_attr: list[Path] = []
    for p in paths:
        try:
            with xr.open_dataset(p, chunks=None) as ds:
                got = _normalize_sigma_mode(str(ds.attrs.get("sigma_mode", "")))
                if got == target:
                    out_attr.append(p)
        except Exception:
            continue
    if out_attr:
        return out_attr

    out_name: list[Path] = []
    needle = "sigsc" if target == "scalar" else "sigf"
    for p in paths:
        if needle in p.name.lower():
            out_name.append(p)
    return out_name if out_name else paths


def filter_candidates_by_yeo_johnson(
    paths: list[Path],
    yeo_johnson: bool | str | None = None,
) -> list[Path]:
    """Prefer candidates with/without ``_yj_`` in the filename (default: Yeo-Johnson on)."""
    if yeo_johnson is None:
        enabled = yeo_johnson_enabled()
    elif isinstance(yeo_johnson, bool):
        enabled = yeo_johnson
    else:
        enabled = yeo_johnson_enabled(yeo_johnson)
    return filter_paths_by_yeo_johnson(paths, enabled=enabled)


def _sample_spatial_regime_da(ds: xr.Dataset) -> xr.DataArray | None:
    """First regime-labelled field with latitude/longitude dims (composite or raw Z500)."""
    preferred = (
        "regime_composite_anom",
        "composite_weighted",  # stage1_z500/notebooks/09_final_clustering_regimes.ipynb
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
    """True if lat/lon outer grid corners match ``bbox_expected`` N,W,S,E (lon wrapped to -180..180)."""
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
    """Keep NetCDF paths whose gridded regime composite corners match the YAML bbox (attrs may be missing)."""
    out: list[Path] = []
    for p in paths:
        try:
            with xr.open_dataset(p, chunks=None) as ds:
                if dataset_spatial_corners_match_bbox(ds, bbox_expected, tol_deg=tol_deg):
                    out.append(p)
        except Exception:
            continue
    return out


def _kind_is_step1011(kind: str) -> bool:
    k = str(kind).replace(" ", "")
    return "10_11" in k


def _kind_is_step9_regimes(kind: str) -> bool:
    k = str(kind).lower().replace(" ", "")
    return "step9" in k or ("9" in k and "regime" in k)


def _pick_newest(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


@dataclass(frozen=True)
class PathSpec:
    window_days: int
    epsilon_m: float
    var_threshold: float  # 0.90
    year_span: str


def narrow_paths_by_step_token(paths: list[Path], step: str) -> list[Path]:
    """Prefer files whose names contain ``_stepBn_`` (pipeline) or ``stepBn_``."""
    tok = _norm_step_token(step)
    assert tok is not None
    hits = [
        p
        for p in paths
        if f"_step{tok}_" in p.name or f"step{tok}_" in p.name or f"_step{tok}." in p.name
    ]
    return hits if hits else paths


def glob_hc_candidates(
    module_root: Path,
    region: str,
    spec: PathSpec,
    *,
    bbox_step: str | None = None,
) -> list[Path]:
    processed = module_root / "data" / "processed" / region
    eps = int(round(spec.epsilon_m))
    var = int(round(spec.var_threshold * 100))
    pats = [
        f"08_Z500_HC_Ward_w{spec.window_days:02d}_eps{eps:02d}_var{var:02d}_*{spec.year_span}*.nc",
        f"08_Z500_HC_Ward_w{spec.window_days:02d}_eps{eps:02d}_var{var:02d}_{spec.year_span}.nc",
    ]
    tok = _norm_step_token(bbox_step)
    if tok is not None:
        pats.extend(
            [
                f"08_Z500_HC_Ward_w{spec.window_days:02d}_eps{eps:02d}_var{var:02d}_*step{tok}_*.nc",
                f"08_Z500_HC_Ward_*_step{tok}_*.nc",
            ]
        )
    cands: list[Path] = []
    for pat in pats:
        cands.extend(sorted(processed.glob(pat)))
    # unique preserve order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def glob_regimes_candidates(
    module_root: Path,
    region: str,
    spec: PathSpec,
    k: int,
    *,
    bbox_step: str | None = None,
    method: str = "kmeans",
) -> list[Path]:
    """``method``: on-disk Step-9 filename family — ``"kmeans"`` (default; also covers the
    Ward-seeded K-means hybrid, which keeps the "kmeans" tag) or ``"ward"`` (pure Ward HRC,
    cut directly from Step-8 linkage — see notebook 09's CLUSTER_MODE)."""
    clustering = module_root / "results" / "clustering" / region
    eps = int(round(spec.epsilon_m))
    var = int(round(spec.var_threshold * 100))
    pats = [
        f"09_Z500_regimes_{method}_k{k:02d}_w{spec.window_days:02d}_eps{eps:02d}_var{var:02d}_*{spec.year_span}*.nc",
        f"09_Z500_regimes_{method}_k{k:02d}_w{spec.window_days:02d}_eps{eps:02d}_var{var:02d}_{spec.year_span}.nc",
    ]
    tok = _norm_step_token(bbox_step)
    if tok is not None:
        pats.extend(
            [
                f"09_Z500_regimes_{method}_k{k:02d}_*step{tok}_*.nc",
                f"09_Z500_regimes_{method}_*_step{tok}_*.nc",
            ]
        )
    cands: list[Path] = []
    for pat in pats:
        cands.extend(sorted(clustering.glob(pat)))
    if not cands and clustering.is_dir():
        cands.extend(sorted(clustering.glob(f"09_Z500_regimes_{method}_k{k:02d}_*.nc")))
        cands.extend(sorted(clustering.glob(f"09_Z500_regimes_{method}_*.nc")))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def glob_step1011_candidates(
    module_root: Path,
    region: str,
    spec: PathSpec,
    k: int,
    *,
    bbox_step: str | None = None,
    method: str = "kmeans",
) -> list[Path]:
    """``method``: see ``glob_regimes_candidates`` — must match the Step-9 file's own tag."""
    regimes_dir = module_root / "results" / "regimes" / region
    eps = int(round(spec.epsilon_m))
    var = int(round(spec.var_threshold * 100))
    pats = [
        f"10_11_Z500_regime_reconstruction_characterisation_{method}_k{k:02d}_w{spec.window_days:02d}_eps{eps:02d}_var{var:02d}_*{spec.year_span}*.nc",
        f"10_11_Z500_regime_reconstruction_characterisation_{method}_k{k:02d}_w{spec.window_days:02d}_eps{eps:02d}_var{var:02d}_{spec.year_span}.nc",
    ]
    tok = _norm_step_token(bbox_step)
    if tok is not None:
        pats.extend(
            [
                f"10_11_Z500_*_{method}_k{k:02d}_*step{tok}_*.nc",
                f"10_11_*_step{tok}_*.nc",
            ]
        )
    cands: list[Path] = []
    for pat in pats:
        cands.extend(sorted(regimes_dir.glob(pat)))
    if not cands and regimes_dir.is_dir():
        cands.extend(
            sorted(regimes_dir.glob(f"10_11_Z500_*_{method}_k{k:02d}_*.nc"))
        )
        cands.extend(sorted(regimes_dir.glob(f"10_11_Z500_regime_reconstruction_characterisation_{method}_*.nc")))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def format_missing_netcdf_hint(
    module_root: Path,
    region: str,
    *,
    step_label: str,
    search_dir: Path,
    glob_examples: list[str],
) -> str:
    """Human-readable hint when strict path resolution finds zero files."""
    lines = [
        f"No {step_label} files matched the preprocessing tag under:",
        f"  {search_dir.resolve()}",
        "",
        "Expected name fragments (examples):",
    ]
    lines.extend(f"  - {g}" for g in glob_examples)
    if search_dir.is_dir():
        found = sorted(search_dir.glob("*.nc"))[:12]
        if found:
            lines.append("")
            lines.append(f"Found {len(list(search_dir.glob('*.nc')))} .nc file(s) in that folder; first examples:")
            lines.extend(f"  {p.name}" for p in found)
        else:
            lines.append("")
            lines.append("(That directory exists but contains no .nc files — run notebook 09 / 10_11 or run_regime_clustering.py first.)")
    else:
        lines.append("")
        lines.append(f"(Directory does not exist: {search_dir})")
    lines.append("")
    lines.append(
        "On TU HPC, after Step 8 preprocessing, submit clustering (notebooks 09/10_11) via:\n"
        "  sbatch z_03_run_regime_clustering_tu_hpc.sh\n"
        "then build composites / stats via:\n"
        "  sbatch z_04_run_regime_composites_tu_hpc.sh\n"
        "  sbatch z_05_run_regime_stats_csv_tu_hpc.sh"
    )
    return "\n".join(lines)


def pick_single(paths: list[Path], *, kind: str) -> Path:
    if len(paths) == 1:
        return paths[0]
    if len(paths) == 0:
        raise FileNotFoundError(f"No candidates for {kind}.")
    names = "\n  - ".join(p.name for p in paths)
    raise FileNotFoundError(f"Ambiguous candidates for {kind} ({len(paths)}):\n  - {names}")


def _narrow_by_spatial_grid_or_raise(
    paths: list[Path],
    *,
    bbox_expected: dict[str, float],
    kind: str,
    tol_deg: float = 1.75,
    bbox_step: str | None = None,
) -> list[Path]:
    """Restrict to NetCDFs whose lat/lon outer grid matches ``bbox_expected`` (Step 9 / 10_11 fields)."""
    if not paths:
        return paths
    grid_hit = filter_candidates_by_spatial_grid_bbox(paths, bbox_expected, tol_deg=tol_deg)
    if grid_hit:
        return grid_hit
    step_u = _norm_step_token(bbox_step)
    hint_nb = ""
    if step_u:
        if _kind_is_step1011(kind):
            hint_nb = (
                f"\n\nRe-run stage1_z500/notebooks/10_11_regime_reconstruction_characterisation.ipynb with BBOX_STEP='{step_u}' "
                f"so `paths['out_10_11']` includes `_step{step_u}_` in the basename and the composite grid "
                f"matches this bbox."
                f"\nOr pass an explicit file (use with a single --steps value): "
                f"--step1011-nc /absolute/path/to/10_11_….nc"
                f"\nOr for drafts only (domain-wide file): --allow-step9-spatial-mismatch"
            )
        elif _kind_is_step9_regimes(kind):
            hint_nb = (
                f"\n\nRe-run stage1_z500/notebooks/09_final_clustering_regimes.ipynb with BBOX_STEP='{step_u}' "
                f"so `paths['out_regimes']` ends with `_step{step_u}_…N…W…S…E….nc` (see `bbox_step_filename_suffix`) "
                f"and the `composite_weighted` grid matches the bbox above."
                f"\nOr pass an explicit file (use with a single --steps value): "
                f"--regimes-nc /absolute/path/to/09_….nc"
                f"\nOr for drafts only (domain-wide file): --allow-step9-spatial-mismatch"
            )
    if len(paths) == 1:
        try:
            with xr.open_dataset(paths[0], chunks=None) as ds:
                ok = dataset_spatial_corners_match_bbox(ds, bbox_expected, tol_deg=tol_deg)
        except Exception:
            ok = False
        if not ok:
            slug_hint = ""
            if step_u and f"_step{step_u}_" not in paths[0].name:
                slug_hint = (
                    f"No filename contains `_step{step_u}_`; discovery fell back to this untagged file. "
                    f"For `--steps {step_u}` you need a Step 9 (and 10_11) NetCDF saved for that bbox step. "
                )
            raise FileNotFoundError(
                f"The only {kind} candidate ({paths[0].name}) does not match the YAML bbox grid corners "
                f"(expected outer domain N,W,S,E = {bbox_expected['N']}, {bbox_expected['W']}, "
                f"{bbox_expected['S']}, {bbox_expected['E']}). "
                f"{slug_hint}"
                f"This file is almost certainly from another domain (e.g. B0 / full window). "
                f"Generate Step 9/10_11 outputs for bbox step {step_u or '?'} and matching k/window settings, "
                f"or remove stale `09_*.nc` / `10_11_*.nc` so discovery can find the right file."
                f"{hint_nb}"
            )
        return paths
    names = "\n  - ".join(p.name for p in paths[:25])
    more = f"\n  ... ({len(paths)} total)" if len(paths) > 25 else ""
    raise FileNotFoundError(
        f"No {kind} candidate matches the YAML bbox corners (lat/lon outer grid). "
        f"N,W,S,E = {bbox_expected['N']}, {bbox_expected['W']}, {bbox_expected['S']}, {bbox_expected['E']}. "
        f"Candidates:\n  - {names}{more}"
        f"{hint_nb}"
    )


def resolve_step_file(
    candidates: list[Path],
    *,
    step: str,
    bbox_expected: dict[str, float],
    kind: str,
    relax_step9_spatial: bool = False,
    relax_step1011_spatial: bool = False,
) -> Path:
    """Disambiguate pipeline/clustering outputs that may omit bbox in the basename.

    If ``relax_step9_spatial`` / ``relax_step1011_spatial`` are True, that step's NetCDF is
    **not** filtered by outer lat/lon grid vs YAML bbox (draft when only B0-wide files exist).
    """
    if not candidates:
        raise FileNotFoundError(f"No candidates for {kind}.")
    work = list(candidates)
    if _kind_is_step1011(kind) and not relax_step1011_spatial:
        work = _narrow_by_spatial_grid_or_raise(
            work,
            bbox_expected=bbox_expected,
            kind=kind,
            tol_deg=1.75,
            bbox_step=step,
        )
    elif _kind_is_step9_regimes(kind) and not relax_step9_spatial:
        work = _narrow_by_spatial_grid_or_raise(
            work,
            bbox_expected=bbox_expected,
            kind=kind,
            tol_deg=1.75,
            bbox_step=step,
        )

    by_attr = filter_candidates_by_attrs(work, step=step, bbox_expected=bbox_expected)
    if len(by_attr) == 1:
        return by_attr[0]
    if len(by_attr) > 1:
        fn = filter_candidates_by_filename_step(by_attr, step)
        if len(fn) == 1:
            return fn[0]
        slug_hit = filter_candidates_by_bbox_slug(by_attr, bbox_expected)
        if len(slug_hit) == 1:
            return slug_hit[0]
        pool = slug_hit if slug_hit else (fn if fn else by_attr)
        return _pick_newest(pool) if len(pool) > 1 else pool[0]

    fn_only = filter_candidates_by_filename_step(work, step)
    if len(fn_only) == 1:
        return fn_only[0]
    slug_all = filter_candidates_by_bbox_slug(work, bbox_expected)
    if len(slug_all) == 1:
        return slug_all[0]
    if len(fn_only) > 1:
        slug_fn = filter_candidates_by_bbox_slug(fn_only, bbox_expected)
        if len(slug_fn) == 1:
            return slug_fn[0]
        return _pick_newest(fn_only) if len(fn_only) > 1 else fn_only[0]

    if len(slug_all) > 1:
        return _pick_newest(slug_all)

    return _pick_newest(work) if len(work) > 1 else work[0]


def figure_prefix_for_region(region: str) -> str:
    if region == "europe":
        return "EU_step10_11"
    if region == "northamerica":
        return "NA_step10_11"
    raise ValueError(f"Unknown region for figure prefix: {region!r}")


def glob_step1011_figures(
    module_root: Path,
    region: str,
    step: str,
) -> list[Path]:
    """PNG figures saved by 10_11 notebooks with BBOX_FIGTAG=_bboxBn."""
    prefix = figure_prefix_for_region(region)
    tok = _norm_step_token(step)
    assert tok is not None
    tag = f"_bbox{tok}"
    globs_dirs = [
        module_root / "figures",
        module_root / "src" / "figures",
    ]
    out: list[Path] = []
    for d in globs_dirs:
        if not d.is_dir():
            continue
        out.extend(sorted(d.glob(f"{prefix}_*{tag}.png")))
    # de-dupe by resolve
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return sorted(uniq, key=lambda p: p.name)


def package_dir(module_root: Path) -> Path:
    return module_root / "src" / "common"
