from __future__ import annotations

import argparse
import gc
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

try:  # optional: pipeline can run headless without plotting
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    from common.map_plotting_helpers import add_contextily_basemap_under_spatial_raster
except Exception:  # pragma: no cover — optional module; missing on some HPC .sif images
    add_contextily_basemap_under_spatial_raster = None  # type: ignore[misc, assignment]


def apply_yeo_johnson(da: xr.DataArray) -> xr.DataArray:
    """Apply Yeo-Johnson power transform field-wide: one λ fitted over all finite values.

    Handles signed (vorticity, omega), zero-inflated (CAPE), and bounded (RH) fields
    without requiring per-field configuration. λ is estimated via MLE from the data.

    To prevent float32 overflow for large-magnitude fields (e.g. Z500 ~5500 m), the
    data is centred and scaled before fitting and applying the transform, then rescaled
    back to the original magnitude. This keeps downstream epsilon thresholds valid.
    """
    try:
        from sklearn.preprocessing import PowerTransformer  # type: ignore
    except ImportError as e:
        raise RuntimeError("apply_yeo_johnson requires scikit-learn (sklearn).") from e

    vals = da.values
    shape = vals.shape
    flat = vals.ravel().astype("float64")
    mask = np.isfinite(flat)
    data = flat[mask]

    loc = float(data.mean())
    scale = float(data.std())
    if scale == 0.0:
        scale = 1.0

    data_norm = (data - loc) / scale

    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    pt.fit(data_norm.reshape(-1, 1))

    flat_norm = (flat - loc) / scale
    transformed = flat_norm.copy()
    transformed[mask] = pt.transform(flat_norm[mask].reshape(-1, 1)).ravel()
    # No back-transform: output stays in YJ-transformed normalised space (~N(0,1)).
    # All downstream steps (detrend, anomaly, sigma normalisation, PCA) operate on
    # these transformed values, which is the intended behaviour.

    out = da.copy(data=transformed.reshape(shape).astype("float32"))
    out.attrs = dict(da.attrs)
    # The transformed values are no longer in the source field's physical units (e.g. Z500's
    # "m**2 s**-2") -- clear/override units and the geopotential-specific name attrs so nothing
    # downstream (geopotential_to_height_if_needed, common.physics.is_geopotential_units)
    # mistakes this YJ-transformed, ~N(0,1) array for still-physical data and re-applies a g0
    # conversion or a "(gpm)" label to it. Confirmed in production: an already-YJ-transformed
    # Z500 anomaly composite (~0.6 in transformed space) carried the original "m**2 s**-2" units
    # forward unchanged, got divided by g0 a second time downstream, and was mislabeled "(gpm)"
    # at ~0.06 instead of the correct "(σ, normalized)".
    out.attrs["units"] = "1"
    out.attrs.pop("standard_name", None)
    out.attrs["long_name"] = f"{da.attrs.get('long_name') or da.name or 'field'} (Yeo-Johnson transformed)"
    out.attrs["yeo_johnson_lambda"] = float(pt.lambdas_[0])
    out.attrs["yeo_johnson_applied"] = "true"
    return out


@dataclass(frozen=True)
class PreprocessConfig:
    """Shared configuration for preprocessing Steps 1–6."""

    # Season definition
    season_months: tuple[int, ...] = (5, 6, 7, 8, 9)  # MJJAS

    # Coordinate conventions (ERA5 exports differ by API / post-processing)
    time_dim_candidates: tuple[str, ...] = ("time", "valid_time")
    lat_dim_candidates: tuple[str, ...] = ("latitude", "lat")
    lon_dim_candidates: tuple[str, ...] = ("longitude", "lon")
    pressure_dim_candidates: tuple[str, ...] = ("pressure_level", "level", "plev")

    # Step 2: climatology + detrending + cyclic smoothing
    smooth_windows: tuple[int, ...] = (21,)
    target_pressure_hpa: int = 500
    detrend_method: str = "gridpoint"  # "gridpoint" | "domain_mean"
    climatology_subtract: str = "raw"  # "raw" | "smooth"

    # Step 4: standardisation σ smoothing + epsilon floor
    epsilon_m: float = 20.0
    sigma_mode: str = "field"  # "field" | "scalar"

    # If the dataset contains ERA5 geopotential (m^2/s^2), convert to height (m)
    g0: float = 9.80665

    # Optional Yeo-Johnson transform applied to the raw daily field before detrending
    yeo_johnson: bool = False


def _default_bbox_steps() -> dict[str, dict]:
    # Fallback when PyYAML is not available (keeps pipeline runnable in minimal envs).
    # Order: N,W,S,E (CDS/EPE convention).
    return {
        "B0": {"label": "Euro-Atlantic regimes (baseline)", "bbox": {"N": 90.0, "W": -80.0, "S": 30.0, "E": 40.0}},
        "B1": {"label": "Shrink step B1", "bbox": {"N": 83.8, "W": -63.4, "S": 33.0, "E": 36.0}},
        "B2": {"label": "Shrink step B2", "bbox": {"N": 77.6, "W": -46.8, "S": 36.0, "E": 32.0}},
        "B3": {"label": "Shrink step B3", "bbox": {"N": 71.4, "W": -30.2, "S": 39.0, "E": 28.0}},
        "B4": {"label": "Shrink step B4", "bbox": {"N": 65.2, "W": -13.6, "S": 42.0, "E": 24.0}},
        "B5": {"label": "Germany-centered (minimum)", "bbox": {"N": 59.0, "W": 3.0, "S": 45.0, "E": 20.0}},
    }


def _load_bbox_steps_yaml(path: Path) -> dict[str, dict]:
    if yaml is None:  # pragma: no cover
        return _default_bbox_steps()
    if not path.exists():
        return _default_bbox_steps()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    steps = data.get("steps", {}) if isinstance(data, dict) else {}
    if not isinstance(steps, dict) or not steps:
        return _default_bbox_steps()
    return steps


def _parse_bbox_string(bbox: str) -> dict[str, float]:
    # Input format: "N,W,S,E" (CDS order, same as EPE).
    toks = [t.strip() for t in str(bbox).split(",")]
    if len(toks) != 4:
        raise ValueError(f'--bbox must be four comma-separated numbers "N,W,S,E"; got {bbox!r}')
    n, w, s, e = (float(t) for t in toks)
    return {"N": n, "W": w, "S": s, "E": e}


def _validate_bbox(bbox: dict[str, float]) -> dict[str, float]:
    for k in ("N", "W", "S", "E"):
        if k not in bbox:
            raise KeyError(f"bbox missing key {k!r} (expected N,W,S,E)")
    out = {k: float(bbox[k]) for k in ("N", "W", "S", "E")}
    if not (out["N"] > out["S"]):
        raise ValueError(f"Invalid bbox: expected N>S, got N={out['N']} S={out['S']}")
    if not (out["E"] > out["W"]):
        raise ValueError(f"Invalid bbox: expected E>W, got W={out['W']} E={out['E']}")
    if out["N"] > 90.0 or out["S"] < -90.0:
        raise ValueError(f"Invalid bbox latitude bounds: {out}")
    if out["W"] < -180.0 or out["E"] > 180.0:
        raise ValueError(f"Invalid bbox longitude bounds (expected -180..180): {out}")
    return out


def _slug_num(x: float) -> str:
    # Filesystem-safe numeric token: -30.2 -> m30p2, 65.0 -> 65p0
    s = f"{float(x):.1f}"
    return s.replace("-", "m").replace(".", "p")


def bbox_slug(bbox: dict[str, float]) -> str:
    b = _validate_bbox(bbox)
    return f"N{_slug_num(b['N'])}W{_slug_num(b['W'])}S{_slug_num(b['S'])}E{_slug_num(b['E'])}"


def bbox_step_filename_suffix(module_root: Path, region: str, bbox_step: str | None) -> str:
    """Suffix for Step 8/9/10_11 NetCDF filenames: ``_stepBn_N..W..S..E..`` when ``bbox_step`` is set.

    Uses ``src/bbox_steps_europe.yaml`` or ``src/bbox_steps_northamerica.yaml`` depending on ``region``.
    Returns ``\"\"`` when ``bbox_step`` is None/empty (full-domain / legacy naming).
    """
    step = (bbox_step or "").strip()
    if not step:
        return ""
    cfg_name = (
        "bbox_steps_northamerica.yaml"
        if str(region).strip().lower() == "northamerica"
        else "bbox_steps_europe.yaml"
    )
    cfg_path = (module_root / "src" / cfg_name).resolve()
    steps = _load_bbox_steps_yaml(cfg_path)
    if step not in steps:
        raise KeyError(f"bbox_step={step!r} not found in {cfg_path} (available: {sorted(steps)})")
    entry = steps[step]
    if not isinstance(entry, dict) or "bbox" not in entry:
        raise KeyError(f"bbox_steps[{step!r}] missing 'bbox' in {cfg_path}")
    bbox = _validate_bbox(entry["bbox"])  # type: ignore[arg-type]
    return f"_step{step}_{bbox_slug(bbox)}"


def yeo_johnson_enabled(value: str | bool | int | None = None) -> bool:
    """True unless ``value`` / ``YEO_JOHNSON`` explicitly disables the ``_yj_`` filename tag."""
    if value is None:
        value = os.environ.get("YEO_JOHNSON", "1")
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"0", "no", "false", "off", "none", "disable", "disabled"}:
        return False
    if s in {"no_yj", "no-yj", "noyeojohnson", "no_yeo_johnson", "no-yeo-johnson"}:
        return False
    return True


def _path_has_yj_tag(path: Path | str) -> bool:
    name = path.name if isinstance(path, Path) else str(path)
    return "_yj_" in name or name.endswith("_yj.nc")


def filter_paths_by_yeo_johnson(
    paths: list[Path],
    *,
    enabled: bool | None = None,
) -> list[Path]:
    """Disambiguate preprocessing NetCDFs with vs without ``_yj_`` in the filename."""
    use_yj = yeo_johnson_enabled() if enabled is None else bool(enabled)
    if use_yj:
        hits = [p for p in paths if _path_has_yj_tag(p)]
    else:
        hits = [p for p in paths if not _path_has_yj_tag(p)]
    return hits if hits else paths


def pick_single(paths: list[Path], *, kind: str) -> Path:
    """Return the sole candidate in ``paths``, or raise with a helpful listing."""
    if len(paths) == 1:
        return paths[0]
    if len(paths) == 0:
        raise FileNotFoundError(f"No candidates found for {kind}.")
    names = "\n  - " + "\n  - ".join(p.name for p in paths)
    raise FileNotFoundError(f"Ambiguous candidates for {kind} ({len(paths)}). Please refine config. Candidates:{names}")


def filter_by_sigma_mode(paths: list[Path], sigma_mode: str, *, kind: str) -> list[Path]:
    """Disambiguate preprocessing NetCDFs by ``sigsc``/``sigf`` filename token."""
    token = "sigsc" if sigma_mode == "scalar" else "sigf" if sigma_mode == "field" else ""
    if not token:
        return paths
    hits = [p for p in paths if token in p.name.lower()]
    if hits:
        return hits
    return paths


def filter_by_bbox_step(paths: list[Path], bbox_step: str | None, *, kind: str) -> list[Path]:
    """Disambiguate preprocessing NetCDFs by ``_step<BBOX_STEP>_`` filename token."""
    if not bbox_step:
        return paths
    step = str(bbox_step).strip().upper()
    step_hits = [p for p in paths if f"_step{step}_" in p.name or f"step{step}_" in p.name]
    if step_hits:
        return step_hits
    available = sorted({part.split("_", 1)[0] for p in paths for part in p.stem.split("_") if part.startswith("stepB")})
    hint = f" Available bbox steps in matching files: {', '.join(available)}." if available else ""
    raise FileNotFoundError(
        f"No {kind} candidates matched bbox_step={step!r}.{hint} "
        f"Run the previous steps with BBOX_STEP={step!r}, or set BBOX_STEP to an available step."
    )


def find_module_root(start: Path | None = None) -> Path:
    """Directory containing ``src/`` (e.g. git clone ``1_Detect_Weather_Patterns/``, or flat WORK root).

    Tries subdirs ``1_Detect_Weather_Patterns`` and ``detect-weather`` under each ancestor, then
    treats ``cwd`` as module root if ``src/preprocessing_helpers.py`` exists (flat Levante tree).

    Override with env ``MODULE_ROOT`` (absolute path set by batch wrappers) or
    ``WEATHER_PATTERNS_MODULE_SUBDIR`` (single folder name under the parent).
    """
    env_root = os.environ.get("MODULE_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "src" / "preprocessing_helpers.py").is_file():
            return root
    env = os.environ.get("WEATHER_PATTERNS_MODULE_SUBDIR", "").strip()
    names: tuple[str, ...] = (env,) if env else ("1_Detect_Weather_Patterns", "detect-weather")
    start = (start or Path.cwd()).resolve()
    for p in (start, *start.parents):
        for name in names:
            cand = p / name
            if cand.is_dir() and (cand / "src").is_dir():
                return cand.resolve()
        # cwd may already be the module root (e.g. Singularity --pwd .../detect-weather)
        if (p / "src" / "preprocessing_helpers.py").is_file():
            return p.resolve()
    raise RuntimeError(
        "Could not locate module root (tried "
        + ", ".join(repr(n) for n in names)
        + f" with src/) starting from {start}"
    )


def find_repo_root(start: Path | None = None) -> Path:
    """Parent directory that contains the weather-patterns module folder."""
    return find_module_root(start=start).parent


def pick_dim_name(ds: xr.Dataset, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in ds.dims:
            return c
    raise KeyError(f"None of the candidate dims {list(candidates)} found in ds.dims={list(ds.dims)}")


def pick_coord_name(da: xr.DataArray, candidates: Iterable[str]) -> str:
    for c in candidates:
        if c in da.dims or c in da.coords:
            return c
    raise KeyError(
        f"None of the candidate dims/coords {list(candidates)} found in da.dims={list(da.dims)} / da.coords={list(da.coords)}"
    )


def pick_time_dim_from_dataset(ds: xr.Dataset, *, time_dim_candidates: Iterable[str]) -> str:
    for c in time_dim_candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(
        f"Could not infer time coordinate from candidates {list(time_dim_candidates)}. "
        f"coords={list(ds.coords)}, dims={list(ds.dims)}"
    )


def sort_raw_era5_paths_by_time_lo(
    paths: list[Path],
    *,
    time_dim_candidates: tuple[str, ...],
) -> list[Path]:
    """Lexicographic ``glob`` / filesystem order often disagrees with ``valid_time`` order across split downloads."""

    def sort_key(p: Path) -> tuple:
        with xr.open_dataset(p, decode_times=True) as ds:
            for c in time_dim_candidates:
                if c not in ds.coords or ds[c].size == 0:
                    continue
                t0 = ds[c].min().item()
                try:
                    return (0, pd.Timestamp(t0), p.name)
                except Exception:
                    return (0, str(t0), p.name)
        return (1, p.name)

    return sorted([Path(x).resolve() for x in paths], key=sort_key)


def _concat_era5_datasets_along_time(
    paths: list[Path],
    *,
    time_dim: str,
    chunks: dict | str | None = None,
) -> xr.Dataset:
    """Explicit time concat + sort + drop duplicate timestamps (overlapping split files)."""
    parts = [xr.open_dataset(p, chunks=chunks, decode_times=True) for p in paths]
    merged = xr.concat(parts, dim=time_dim, data_vars="minimal", coords="minimal", compat="override")
    merged = merged.sortby(time_dim)
    tvals = merged[time_dim].values
    seen: set[int] = set()
    keep: list[int] = []
    for i in range(int(np.asarray(tvals).shape[0])):
        key = pd.Timestamp(tvals[i]).value
        if key not in seen:
            seen.add(key)
            keep.append(i)
    if len(keep) < merged.sizes[time_dim]:
        merged = merged.isel({time_dim: keep})
    return merged


def open_era5_z500_raw_mfdataset(
    in_files: list[Path],
    *,
    cfg: PreprocessConfig | None = None,
    chunks: dict | str | None = None,
    data_vars: str = "minimal",
    coords: str = "minimal",
    compat: str = "override",
    parallel: bool = False,
) -> xr.Dataset:
    """Open one or many raw ERA5 NetCDF paths along the time axis.

    ``xr.open_mfdataset(..., combine='by_coords')`` requires globally monotonic time. CDS-era
    split files are often named so that lexicographic order ≠ chronological order, or spans
    overlap, which triggers ``ValueError: ... monotonic ... valid_time``. We sort paths by each
    file's minimum time, then combine; on failure we concat explicitly, ``sortby`` time, and drop
    duplicate timestamps.
    """
    cfg = cfg or PreprocessConfig()
    cand = cfg.time_dim_candidates
    paths = [Path(p).resolve() for p in in_files]
    if len(paths) == 0:
        raise FileNotFoundError("No input NetCDF paths given.")
    if len(paths) == 1:
        return xr.open_dataset(paths[0], chunks=chunks, decode_times=True)

    ordered = sort_raw_era5_paths_by_time_lo(paths, time_dim_candidates=cand)
    mfd_kw: dict = {
        "combine": "by_coords",
        "data_vars": data_vars,
        "coords": coords,
        "compat": compat,
        "parallel": parallel,
    }
    if chunks is not None:
        mfd_kw["chunks"] = chunks
    try:
        return xr.open_mfdataset(ordered, **mfd_kw)
    except ValueError as exc:
        err = str(exc).lower()
        if "monotonic" not in err:
            raise
        with xr.open_dataset(ordered[0], decode_times=True) as ds0:
            time_dim = pick_time_dim_from_dataset(ds0, time_dim_candidates=cand)
        return _concat_era5_datasets_along_time(ordered, time_dim=time_dim, chunks=chunks)


def infer_core_dims(da: xr.DataArray, *, cfg: PreprocessConfig) -> dict[str, str]:
    time_dim = pick_coord_name(da, cfg.time_dim_candidates)
    lat_dim = pick_coord_name(da, cfg.lat_dim_candidates)
    lon_dim = pick_coord_name(da, cfg.lon_dim_candidates)
    return {"time": time_dim, "lat": lat_dim, "lon": lon_dim}


def select_mjjas(
    da: xr.DataArray, *, time_dim: str, months: tuple[int, ...] = (5, 6, 7, 8, 9)
) -> xr.DataArray:
    """Subset to the requested months (default MJJAS)."""
    return da.where(da[time_dim].dt.month.isin(months), drop=True)


def validate_mjjas_daily_coverage(
    da: xr.DataArray,
    *,
    time_dim: str,
    expected_unique_days: int = 153,
) -> None:
    """Raise if any calendar year does not have exactly ``expected_unique_days`` unique MJJAS days.

    Intended for use **after** daily aggregation (``resample(..., '1D')``) and MJJAS subsetting,
    i.e. on the same ``Z_mjjas`` object used in Step 2.
    """
    if time_dim not in da.dims and time_dim not in da.coords:
        raise KeyError(f"time_dim {time_dim!r} not in da.dims/coords")

    t = da[time_dim].values
    if t.size == 0:
        raise ValueError("validate_mjjas_daily_coverage: time axis is empty.")

    ts = pd.to_datetime(pd.Index(np.asarray(t)))
    floored = ts.floor("D")
    years = floored.year.to_numpy()

    for y in sorted(np.unique(years)):
        mask = years == y
        n_unique = int(pd.Index(floored[mask]).nunique())
        if n_unique != expected_unique_days:
            raise ValueError(
                f"MJJAS daily coverage incomplete for year {int(y)}: "
                f"expected {expected_unique_days} unique calendar days in MJJAS, got {n_unique}. "
                "Check raw ERA5 files for missing days or misaligned timestamps before Step 2."
            )


def day_of_mjjas(da: xr.DataArray, *, time_dim: str) -> xr.DataArray:
    """Season-day index 1..153 for MJJAS (May 1 -> 1, ..., Sep 30 -> 153).

    Leap-year safe: compute days since May 1 of the same year.
    Intended to be applied after selecting MJJAS only.
    """
    t = da[time_dim]
    try:
        months = t.dt.month
    except Exception as e:
        raise ValueError(
            f"Time coordinate does not appear to be datetime-like (cannot use .dt accessors). "
            f"coord={time_dim!r}, dtype={getattr(t, 'dtype', None)!r}"
        ) from e

    if int(months.min()) < 5 or int(months.max()) > 9:
        raise ValueError(
            f"Expected MJJAS-only times, got months {int(months.min())}..{int(months.max())}"
        )

    year = t.dt.year
    may1 = xr.DataArray(
        pd.to_datetime(year.astype(str) + "-05-01"),
        dims=t.dims,
        coords=t.coords,
        name="may1",
    )
    season_day = (t.dt.floor("D") - may1).dt.days + 1

    if int(season_day.min()) < 1 or int(season_day.max()) > 153:
        raise ValueError(
            "day_of_mjjas expected MJJAS-only times with season_day in 1..153; "
            f"got {int(season_day.min())}..{int(season_day.max())}"
        )
    return season_day.rename("season_day")


def subtract_mjjas_season_day_climatology(
    z_det: xr.DataArray,
    clim: xr.DataArray,
    *,
    time_dim: str,
) -> xr.DataArray:
    """Subtract ``clim(season_day, lat, lon)`` from ``z_det(time, lat, lon)`` without groupby arithmetic.

    ``z_det.groupby(season_day) - clim`` expands to a very wide lazy graph on long ERA5 stacks and
    large domains (notably North America), which routinely OOMs during ``.compute()`` / NetCDF writes.
    Vectorized ``sel`` along ``season_day`` keeps the graph linear.
    """
    sd = day_of_mjjas(z_det, time_dim=time_dim)
    clim_on_time = clim.sel(season_day=sd, drop=True)
    clim_on_time = clim_on_time.transpose(*z_det.dims)
    return z_det - clim_on_time


def mjjas_season_day_climatology_mean_unstack(
    z_det: xr.DataArray,
    *,
    time_dim: str,
) -> xr.DataArray:
    """Season-day climatology mean over years without ``groupby(time_dim)``.

    ``xarray`` + Dask ``groupby(...).mean(time)`` on long ERA5 stacks can explode memory during
    ``.compute()`` / NetCDF writes. If each calendar year contributes at most one sample per
    ``season_day`` (true for daily MJJAS-only series), reshape ``time -> (year, season_day)``
    via ``set_index`` / ``unstack`` and average over ``year``.
    """
    t = z_det[time_dim]
    years = t.dt.year.astype("int32")
    sd = day_of_mjjas(z_det, time_dim=time_dim).astype("int16")

    years_np = np.asarray(years.values).ravel()
    sd_np = np.asarray(sd.values).ravel()
    idx = pd.MultiIndex.from_arrays([years_np, sd_np])
    clim: xr.DataArray | None = None
    if not idx.has_duplicates:
        try:
            z_tmp = z_det.assign_coords(
                _year=(time_dim, years.values),
                _season_day=(time_dim, sd.values),
            )
            z_st = z_tmp.set_index(time_dim=["_year", "_season_day"]).unstack(time_dim)
            clim = z_st.mean(dim="_year", skipna=True).rename({"_season_day": "season_day"})
        except Exception:
            clim = None

    if clim is None:
        parts: list[xr.DataArray] = []
        for k in range(1, 154):
            sub = z_det.where(sd == k, drop=True)
            parts.append(sub.mean(time_dim, skipna=True).expand_dims(season_day=[int(k)]))
        clim = xr.concat(parts, dim="season_day")

    return clim.sortby("season_day").rename("z500_clim_raw")


def divide_by_mjjas_season_day_field(
    z_da: xr.DataArray,
    field: xr.DataArray,
    *,
    time_dim: str,
) -> xr.DataArray:
    """Divide ``z_da(time,...)`` by ``field(season_day,...)`` broadcast to time (no ``groupby`` divide)."""
    sd = day_of_mjjas(z_da, time_dim=time_dim)
    den = field.sel(season_day=sd, drop=True)
    if all(d in den.dims for d in z_da.dims) and set(den.dims) == set(z_da.dims):
        den = den.transpose(*z_da.dims)
    return z_da / den


def _step2_nan_check_time_slab_from_env() -> int:
    """Max number of timesteps to materialize at once when scanning ``z500_anom`` for NaNs."""
    raw = os.environ.get("STEP2_NANCHECK_TIME_SLAB")
    if raw is None or str(raw).strip() == "":
        return 256
    v = int(str(raw).strip())
    return max(1, v)


def _any_nan_time_slabwise(da: xr.DataArray, *, time_dim: str) -> bool:
    """Whether any element is NaN, evaluating ``time_dim`` in slabs to cap peak RAM."""
    n = int(da.sizes[time_dim])
    slab = _step2_nan_check_time_slab_from_env()
    for start in range(0, n, slab):
        stop = min(n, start + slab)
        sub = da.isel({time_dim: slice(start, stop)})
        if bool(np.asarray(sub.isnull().any().compute()).item()):
            return True
    return False


def detrend_linear(da: xr.DataArray, *, time_dim: str) -> xr.DataArray:
    """Remove linear trend at each grid point (least squares)."""
    t = xr.DataArray(
        np.arange(da.sizes[time_dim], dtype="float32"),
        dims=[time_dim],
        coords={time_dim: da[time_dim]},
        name="t_index",
    )

    t_mean = t.mean(time_dim)
    da_mean = da.mean(time_dim, skipna=True)
    t_anom = t - t_mean
    da_anom = da - da_mean

    cov = (t_anom * da_anom).mean(time_dim, skipna=True)
    var = (t_anom**2).mean(time_dim)
    slope = cov / var
    intercept = da_mean - slope * t_mean
    trend = slope * t + intercept

    out = da - trend
    out.attrs = dict(da.attrs)
    out.attrs["detrended"] = "linear trend removed at each grid point"
    return out


def _coslat_weights_1d(lat: xr.DataArray) -> xr.DataArray:
    lat_dim = str(lat.dims[0])
    w = np.cos(np.deg2rad(lat)).clip(0.0, 1.0)
    return xr.DataArray(w, dims=[lat_dim], coords={lat_dim: lat}, name="coslat")


def area_mean_cosphi(da: xr.DataArray, *, lat_dim: str, lon_dim: str) -> xr.DataArray:
    """Cos(latitude)-weighted area mean over lat/lon for regular lon grids."""
    wlat = _coslat_weights_1d(da[lat_dim])
    num = (da * wlat).sum(dim=(lat_dim, lon_dim), skipna=True)
    den = wlat.sum(dim=lat_dim, skipna=True) * int(da.sizes[lon_dim])
    return (num / den).rename("area_mean_cosphi")


def detrend_domain_mean(da: xr.DataArray, *, time_dim: str, lat_dim: str, lon_dim: str) -> xr.DataArray:
    """Remove linear trend of the daily cos(lat)-weighted domain-mean Z500 (scalar trend)."""
    ts = area_mean_cosphi(da, lat_dim=lat_dim, lon_dim=lon_dim)
    t = xr.DataArray(
        np.arange(ts.sizes[time_dim], dtype=np.float64),
        dims=[time_dim],
        coords={time_dim: ts[time_dim]},
        name="t_index",
    )
    slope, intercept = np.polyfit(np.asarray(t.values, dtype=np.float64), np.asarray(ts.values, dtype=np.float64), 1)
    trend = (slope * t + intercept).rename("domain_mean_trend")
    out = (da - trend).rename(da.name or "z500_det")
    out.attrs = dict(da.attrs)
    out.attrs["detrended"] = "linear trend removed from daily cos(lat)-weighted domain mean (scalar trend subtracted)"
    out.attrs["detrend_domain_mean_slope_per_index"] = float(slope)
    return out


def cyclic_running_mean(da: xr.DataArray, *, dim: str, window: int) -> xr.DataArray:
    """Centered running mean with cyclic wrap-around along `dim` (window must be odd)."""
    if window <= 0 or window % 2 != 1:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    half = window // 2
    left = da.isel({dim: slice(-half, None)})
    right = da.isel({dim: slice(0, half)})
    padded = xr.concat([left, da, right], dim=dim)
    rolled = padded.rolling({dim: window}, center=True).mean(skipna=True)
    return rolled.isel({dim: slice(half, half + da.sizes[dim])})


def infer_z500_var(ds: xr.Dataset) -> str:
    """Heuristic to pick the Z500 variable name from common ERA5 exports."""
    var_candidates = ["z", "geopotential", "Z", "z500", "gh", "hgt"]
    for v in var_candidates:
        if v in ds.data_vars:
            return v
    raise KeyError(f"Could not find a Z500-like variable. Available: {list(ds.data_vars)}")


def select_pressure_level(da: xr.DataArray, *, cfg: PreprocessConfig) -> xr.DataArray:
    """Select/squeeze pressure level if present, targeting cfg.target_pressure_hpa."""
    try:
        p_dim = pick_coord_name(da, cfg.pressure_dim_candidates)
    except KeyError:
        return da

    if p_dim not in da.dims:
        return da

    p_vals = da[p_dim].values
    if np.size(p_vals) == 1:
        out = da.isel({p_dim: 0}).drop_vars(p_dim, errors="ignore")
        return out

    target = cfg.target_pressure_hpa
    if np.any(p_vals == target):
        out = da.sel({p_dim: target})
    else:
        out = da.sel({p_dim: target}, method="nearest")
    return out.drop_vars(p_dim, errors="ignore")


def geopotential_to_height_if_needed(da: xr.DataArray, *, cfg: PreprocessConfig) -> xr.DataArray:
    """Convert ERA5 geopotential (m^2/s^2) to geopotential height (m) if units suggest it.

    ERA5/cfgrib write this as ``"m**2 s**-2"`` (double-star exponents), not ``"m^2"``/``"m2"``,
    so the substring check has to match that form too or the conversion silently never fires.
    """
    units = (da.attrs.get("units") or "").lower()
    if any(tok in units for tok in ("m2", "m^2", "m**2", "s-2", "s^-2", "s**-2")):
        out = da / cfg.g0
        out.attrs = dict(da.attrs)
        out.attrs["units"] = "m"
        out.attrs["long_name"] = "geopotential height"
        return out
    return da


def infer_netcdf_variable_chunk_sizes(path: Path, var_name: str) -> dict[str, int] | None:
    """Read HDF5/NetCDF4 chunk sizes for ``var_name`` from disk (dim_name -> chunk_length).

    When present, these values align ``xr.open_dataset(..., chunks=...)`` with on-disk layout and
    suppress xarray's "specified chunks separate the stored chunks" warnings, which often precede
    Dask rechunks / OOM on long ERA5 extracts.
    """
    try:
        import netCDF4 as nc4  # type: ignore
    except Exception:
        return None
    try:
        with nc4.Dataset(str(path), "r") as nc:
            if var_name not in nc.variables:
                return None
            v = nc.variables[var_name]
            ch = v.chunking()
            if ch is None or ch == "contiguous":
                return None
            dims = v.dimensions
            out: dict[str, int] = {}
            for d, c in zip(dims, ch):
                if isinstance(c, int) and c > 0:
                    out[str(d)] = int(c)
            return out or None
    except Exception:
        return None


def _step2_time_chunk_from_env_and_native(*, time_dim: str, native: dict[str, int] | None) -> int | None:
    """Resolve Step-2 time chunk size: env override, else on-disk chunk for ``time_dim``, else 128.

    ``STEP2_TIME_CHUNK=-1`` disables explicit chunking at open time (``chunks=None`` on the dataset).
    """
    raw = os.environ.get("STEP2_TIME_CHUNK")
    if raw is not None and str(raw).strip() == "-1":
        return None
    native = native or {}
    if raw is None or str(raw).strip() == "":
        if time_dim in native:
            return max(1, int(native[time_dim]))
        return 128
    v = int(str(raw).strip())
    if v <= 0:
        if time_dim in native:
            return max(1, int(native[time_dim]))
        return 128
    return int(v)


def _step2_lat_lon_chunks_from_env_and_native(
    *, native: dict[str, int] | None, lat_dim: str, lon_dim: str
) -> tuple[int | None, int | None]:
    """Lat/lon Step-2 chunk sizes: env override, else on-disk chunks, else (48, 80).

    ``STEP2_LAT_CHUNK=0`` / ``STEP2_LON_CHUNK=0`` disables chunking on that axis (``None``).
    """
    native = native or {}

    def pick(env_name: str, dim: str, fallback: int) -> int | None:
        raw = os.environ.get(env_name)
        if raw is None or str(raw).strip() == "":
            if dim in native:
                return max(1, int(native[dim]))
            return int(fallback)
        v = int(str(raw).strip())
        if v <= 0:
            return None
        return int(v)

    return pick("STEP2_LAT_CHUNK", lat_dim, 48), pick("STEP2_LON_CHUNK", lon_dim, 80)


def build_step2_products(
    *,
    in_files: list[Path],
    cfg: PreprocessConfig,
    bbox: dict[str, float] | None = None,
) -> tuple[xr.DataArray, dict[int, dict[str, xr.DataArray]]]:
    """Step 2: read raw ERA5, build detrended field + climatology/anomalies by window."""
    xr.set_options(keep_attrs=True)
    _step2_apply_dask_env()
    ds, Z_mjjas, time_dim, lat_dim, lon_dim = _step2_open_raw_to_mjjas(in_files=in_files, cfg=cfg, bbox=bbox)
    if cfg.yeo_johnson:
        Z_mjjas = apply_yeo_johnson(Z_mjjas)
    Z_det = _step2_mjjas_to_detrended(Z_mjjas, cfg=cfg, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim)
    results = _step2_detrended_to_results(Z_det, cfg=cfg, time_dim=time_dim)
    return Z_det, results


def compute_step3_anomaly(
    *,
    z_det: xr.DataArray,
    z_clim_raw: xr.DataArray,
    z_clim_smooth: xr.DataArray | None,
    cfg: PreprocessConfig,
) -> xr.DataArray:
    """Step 3: compute anomalies from detrended field and raw season-day climatology.

    Note: We keep a smoothed climatology product for diagnostics / σ smoothing, but subtracting
    the smoothed climatology leaves a residual mean seasonal structure equal to (clim_raw - clim_smooth).
    """
    dims = infer_core_dims(z_det, cfg=cfg)
    time_dim = dims["time"]

    if str(cfg.climatology_subtract).lower() in ("raw", "clim_raw", "unsmoothed"):
        clim_for_sub = z_clim_raw
        clim_desc = "raw"
    elif str(cfg.climatology_subtract).lower() in ("smooth", "smoothed", "clim_smooth"):
        if z_clim_smooth is None:
            raise ValueError("climatology_subtract='smooth' requested but z_clim_smooth is None")
        clim_for_sub = z_clim_smooth
        clim_desc = "smooth"
    else:
        raise ValueError(
            f"Unknown cfg.climatology_subtract={cfg.climatology_subtract!r}. Expected 'raw' or 'smooth'."
        )

    z_anom = subtract_mjjas_season_day_climatology(z_det, clim_for_sub, time_dim=time_dim).rename(
        "z500_anom"
    )
    z_anom.attrs = dict(z_det.attrs)
    z_anom.attrs.update(
        {
            "description": f"Z500 anomalies computed from detrended field using {clim_desc} MJJAS season-day climatology",
            "season": "MJJAS",
            "definition": f"z500_anom(time,lat,lon) = z500_det(time,lat,lon) - z500_clim_{clim_desc}(season_day,lat,lon)",
            "climatology_subtract": str(cfg.climatology_subtract),
        }
    )
    return z_anom


def compute_step4_sigma_and_norm(
    *,
    z_anom: xr.DataArray,
    window_days: int,
    epsilon_m: float,
    cfg: PreprocessConfig,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Step 4: build season-day σ (smoothed) and normalized anomalies."""
    dims = infer_core_dims(z_anom, cfg=cfg)
    time_dim = dims["time"]
    season_day = (
        z_anom["season_day"]
        if ("season_day" in z_anom.coords and time_dim in z_anom["season_day"].dims)
        else day_of_mjjas(z_anom, time_dim=time_dim)
    ).load()

    lat_dim = dims["lat"]
    lon_dim = dims["lon"]

    if str(cfg.sigma_mode).lower() in ("field", "grid", "gridpoint"):
        sigma_raw = z_anom.groupby(season_day).std(dim=time_dim, ddof=0, skipna=True).rename("sigma_raw")
        sigma_smooth = cyclic_running_mean(sigma_raw, dim="season_day", window=int(window_days)).rename(
            "sigma_smooth"
        )
        sigma = xr.where(sigma_smooth < epsilon_m, epsilon_m, sigma_smooth).rename("sigma_season_day")
        # xr.where drops attrs even under keep_attrs=True -- restore z_anom's physical units so
        # sigma (and anything reconstructed as z_norm * sigma) isn't left with no units at all.
        sigma.attrs["units"] = str(z_anom.attrs.get("units", "") or "")
        sigma.attrs["sigma_mode"] = "field"
        z_norm = divide_by_mjjas_season_day_field(z_anom, sigma, time_dim=time_dim).rename("z500_norm")
    elif str(cfg.sigma_mode).lower() in ("scalar", "area_mean", "domain_mean"):
        sigma_raw_field = z_anom.groupby(season_day).std(dim=time_dim, ddof=0, skipna=True).rename(
            "sigma_raw_field"
        )
        sigma_raw_scalar = area_mean_cosphi(sigma_raw_field, lat_dim=lat_dim, lon_dim=lon_dim).rename(
            "sigma_raw_scalar"
        )
        sigma_smooth = cyclic_running_mean(
            sigma_raw_scalar, dim="season_day", window=int(window_days)
        ).rename("sigma_smooth_scalar")
        sigma = xr.where(sigma_smooth < epsilon_m, epsilon_m, sigma_smooth).rename("sigma_season_day")
        # xr.where drops attrs even under keep_attrs=True -- restore z_anom's physical units so
        # sigma (and anything reconstructed as z_norm * sigma) isn't left with no units at all.
        sigma.attrs["units"] = str(z_anom.attrs.get("units", "") or "")
        sigma.attrs["sigma_mode"] = "scalar_area_mean_cosphi"
        z_norm = divide_by_mjjas_season_day_field(z_anom, sigma, time_dim=time_dim).rename("z500_norm")
    else:
        raise ValueError(f"Unknown cfg.sigma_mode={cfg.sigma_mode!r}. Expected 'field' or 'scalar'.")
    z_norm.attrs = dict(z_anom.attrs)
    z_norm.attrs.update(
        {
            "units": "1",
            "long_name": "standardized Z500 anomaly (z-score)",
            "description": "Z500 anomalies normalized by smoothed season-day σ (MJJAS)",
            "window_days": int(window_days),
            "epsilon_m": float(epsilon_m),
        }
    )
    return sigma, season_day.rename("season_day_of_time"), z_norm


def compute_step5_area_weighting(
    *,
    z_norm: xr.DataArray,
    cfg: PreprocessConfig,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Step 5: apply √cosφ latitude weighting."""
    dims = infer_core_dims(z_norm, cfg=cfg)
    lat_dim = dims["lat"]
    lat = z_norm[lat_dim].values
    cosphi = np.cos(np.deg2rad(lat)).clip(0.0, 1.0)
    w_lat = xr.DataArray(np.sqrt(cosphi), dims=[lat_dim], coords={lat_dim: lat}, name="w_lat")
    z_weighted = (z_norm * w_lat).rename("z500_weighted")
    z_weighted.attrs["units"] = "1"
    z_weighted.attrs["long_name"] = "area-weighted standardized Z500 anomaly (z-score * sqrt(cos(lat)))"
    return w_lat, z_weighted


def compute_step6_flatten(
    *,
    z_weighted: xr.DataArray,
    cfg: PreprocessConfig,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Step 6: flatten to `X_matrix(time, space)` and create mapping arrays."""
    dims = infer_core_dims(z_weighted, cfg=cfg)
    time_dim, lat_dim, lon_dim = dims["time"], dims["lat"], dims["lon"]

    if z_weighted.dims != (time_dim, lat_dim, lon_dim):
        z_weighted = z_weighted.transpose(time_dim, lat_dim, lon_dim)

    finite_frac = float(np.isfinite(z_weighted).mean())
    if finite_frac != 1.0:
        raise ValueError(f"z_weighted contains non-finite values (finite fraction: {finite_frac})")

    X_stacked = (
        z_weighted.stack(space=(lat_dim, lon_dim)).transpose(time_dim, "space").rename("X_matrix")
    )

    space_index = np.arange(X_stacked.sizes["space"], dtype=int)
    latitude_space = xr.DataArray(
        X_stacked[lat_dim].values,
        dims=["space"],
        coords={"space": space_index},
        name="latitude_space",
    )
    longitude_space = xr.DataArray(
        X_stacked[lon_dim].values,
        dims=["space"],
        coords={"space": space_index},
        name="longitude_space",
    )

    X_matrix = (
        X_stacked.drop_vars([lat_dim, lon_dim]).assign_coords(space=space_index).astype("float32")
    )
    return X_matrix, latitude_space, longitude_space


def build_step4_6_dataset(
    *,
    z_anom: xr.DataArray,
    window_days: int,
    epsilon_m: float,
    cfg: PreprocessConfig,
) -> xr.Dataset:
    sigma, season_day_of_time, z_norm = compute_step4_sigma_and_norm(
        z_anom=z_anom, window_days=window_days, epsilon_m=epsilon_m, cfg=cfg
    )
    w_lat, z_weighted = compute_step5_area_weighting(z_norm=z_norm, cfg=cfg)
    X_matrix, latitude_space, longitude_space = compute_step6_flatten(z_weighted=z_weighted, cfg=cfg)

    # Avoid coordinate-name collisions: time-wise `season_day` vs σ dimension `season_day`
    z_norm_out = z_norm.reset_coords("season_day", drop=True) if "season_day" in z_norm.coords else z_norm
    z_weighted_out = (
        z_weighted.reset_coords("season_day", drop=True) if "season_day" in z_weighted.coords else z_weighted
    )

    ds_out = xr.Dataset(
        data_vars={
            "z500_norm": z_norm_out.astype("float32"),
            "sigma_season_day": sigma.astype("float32"),
            "z500_weighted": z_weighted_out.astype("float32"),
            "w_lat": w_lat.astype("float32"),
            "season_day_of_time": season_day_of_time.astype("int16"),
            "X_matrix": X_matrix,
            "latitude_space": latitude_space.astype("float32"),
            "longitude_space": longitude_space.astype("float32"),
        },
        attrs=dict(z_anom.attrs),
    )
    ds_out.attrs.update(
        {
            "step": "04_05_06_preprocess_standardise_weighting_flattening",
            "window_days": int(window_days),
            "epsilon_m": float(epsilon_m),
            "definition_step4": "z500_norm(time,lat,lon) = z500_anom(time,lat,lon) / sigma(season_day,lat,lon)",
            "definition_step5": "z500_weighted(time,lat,lon) = z500_norm(time,lat,lon) * sqrt(cos(lat))",
            "sigma_definition": "std across years for each MJJAS season_day, then cyclic running-mean smoothed",
            "area_weighting": "sqrt(cos(phi)) applied after standardisation and before EOF/PCA",
            "X_matrix_space_mapping": "latitude_space(space), longitude_space(space)",
            "flatten_order": "lat outer, lon inner (xarray stack space=(lat,lon))",
        }
    )
    return ds_out


def infer_years_label_from_files(in_files: list[Path], *, cfg: PreprocessConfig) -> str:
    ds_meta = open_era5_z500_raw_mfdataset(in_files, cfg=cfg)
    try:
        time_dim = pick_time_dim_from_dataset(ds_meta, time_dim_candidates=cfg.time_dim_candidates)
        years = ds_meta[time_dim].dt.year.values.astype(int)
        y0 = int(years.min())
        y1 = int(years.max())
        return f"{y0}" if y0 == y1 else f"{y0}-{y1}"
    finally:
        ds_meta.close()


def parse_mjjas_year_bounds(years_label: str) -> tuple[int, int]:
    """Parse ``years_label`` like ``1979-2025`` or ``2020`` into inclusive calendar year bounds.

    Used for Step 3 (originally prototyped in the now-retired ``3_compute_anomalies.ipynb``): slice detrended MJJAS to
    May 1 … Sep 30 within each year and validate coverage.
    """
    s = years_label.strip()
    if not s:
        raise ValueError("years_label is empty")
    if "-" in s:
        a, b = s.split("-", 1)
        return int(a), int(b)
    y = int(s)
    return y, y


def mjjas_pipeline_global_attrs(
    years_label: str,
    *,
    climatology_window_days: int | None,
    preprocessing_step: str,
    region: str | None = None,
) -> dict[str, str | int | float]:
    """Standard global attributes for pipeline / Step 2 NetCDF products (year span + smoothing window).

    ``climatology_window_days=None`` marks products that do not encode a specific rolling window
    (shared detrended field and raw season-day climatology). In that case the attribute is set
    to ``-1`` and ``climatology_window_note`` explains the convention.
    """
    y0, y1 = parse_mjjas_year_bounds(years_label)
    out: dict[str, str | int | float] = {
        "year_span": years_label,
        "year_start": int(y0),
        "year_end": int(y1),
        "preprocessing_step": preprocessing_step,
        "season": "MJJAS",
    }
    if region:
        out["region"] = str(region)
    if climatology_window_days is None:
        out["climatology_window_days"] = -1
        out["climatology_window_note"] = (
            "Not applicable: this file is shared across smoothing windows. "
            "Use 02_z500_mjjas_climatology_anomalies_wXX_* for window-specific smoothed climatology."
        )
    else:
        out["climatology_window_days"] = int(climatology_window_days)
    return out


def raw_nc_filename_matches_span(filename: str, span: str) -> bool:
    """Whether ``filename`` looks like raw CDS Z500 for calendar span ``span`` (e.g. ``1979-2025`` or ``2020``)."""
    span = span.strip()
    if not span:
        return False
    if f"years_{span}" in filename or f"mjjas_{span}" in filename:
        return True
    # Newer naming produced by our sbatch downloader uses "..._<span>_..." without "years_" prefix,
    # e.g. "era5_z500_northamerica_m0506070809_1979-2025_1200UTC.nc".
    if f"_{span}_" in filename or filename.endswith(f"_{span}.nc"):
        return True
    if span.isdigit() and len(span) == 4:
        return (
            f"years_{span}_" in filename
            or f"years_{span}." in filename
            or f"mjjas_{span}_" in filename
            or f"_{span}_" in filename
        )
    return False


def select_raw_era5_nc_files(
    raw_dir: Path,
    *,
    years_span: str | None,
    raw_nc: Path | None = None,
) -> list[Path]:
    """
    Pick raw ERA5 NetCDF(s) under ``raw_dir``.

    If ``raw_nc`` is set, return that file only. Otherwise, if ``years_span`` is set, keep files whose
    names match (``years_<span>`` from ``cds_era5_z500_download.py``, or legacy ``mjjas_<span>``). If
    ``years_span`` is unset and exactly one ``*.nc`` exists, use it; if several exist without a span,
    raise with a clear message.
    """
    raw_dir = raw_dir.resolve()
    if raw_nc is not None:
        p = raw_nc.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--raw-nc not found or not a file: {p}")
        return [p]

    years_span = (years_span or "").strip()
    candidates = sorted(raw_dir.glob("*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No *.nc under {raw_dir}")

    if not years_span:
        if len(candidates) == 1:
            return candidates
        names = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"Multiple NetCDF files in {raw_dir} ({len(candidates)}): {names}. "
            "Set --raw-years-span to match one (e.g. 1979-2025 or 2020-2023), set env ERA5_RAW_YEARS_SPAN, "
            "use --raw-nc PATH, or leave only one file in that directory."
        )

    matched = [p for p in candidates if raw_nc_filename_matches_span(p.name, years_span)]
    if not matched:
        names = ", ".join(p.name for p in candidates)
        raise FileNotFoundError(
            f"No raw file name matches --raw-years-span={years_span!r}. "
            f"Files present: {names}. Expect substring years_<span> or mjjas_<span> (CDS output naming)."
        )
    if len(matched) > 1:
        names = ", ".join(p.name for p in matched)
        raise FileNotFoundError(
            f"Multiple files match --raw-years-span={years_span!r}: {names}. Use --raw-nc to pick one explicitly."
        )
    return matched


def build_default_paths(*, module_root: Path, region: str) -> tuple[Path, Path]:
    raw_dir = module_root / "data" / "raw" / "era5" / "z500" / region
    out_dir = module_root / "data" / "preprocessed" / region
    return raw_dir, out_dir


def build_default_figures_dir(*, module_root: Path, region: str) -> Path:
    return module_root / "figures" / region / "preprocessing"


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)


def save_map_figure(
    *,
    da: xr.DataArray,
    out_path: Path,
    title: str,
    robust: bool = True,
) -> None:
    """Save a simple 2D map diagnostic; optional Contextily basemap underlay."""
    if plt is None:
        return
    # Be forgiving: some diagnostics are 1D (e.g. σ(season_day) in scalar sigma_mode) or 0D
    # after selecting a single season_day. Figures are optional and must never crash the pipeline.
    try:
        if not np.issubdtype(np.asarray(da.dtype), np.number):
            return
    except Exception:
        return
    if int(getattr(da, "ndim", 0)) == 0:
        return
    if int(getattr(da, "ndim", 0)) == 1:
        save_timeseries_figure(da=da, out_path=out_path, title=title)
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        da.plot(ax=ax, robust=robust)
    except TypeError:
        # Most commonly: "No numeric data to plot." for non-2D / fully-masked arrays.
        plt.close(fig)
        return
    if add_contextily_basemap_under_spatial_raster is not None:
        try:
            add_contextily_basemap_under_spatial_raster(ax)
        except Exception:
            pass  # optional Contextily / tiles; never fail pipeline figures
    ax.set_title(title)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def save_timeseries_figure(
    *,
    da: xr.DataArray,
    out_path: Path,
    title: str,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(9, 3))
    da.plot(ax=ax)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def _is_netcdf_readable(path: Path) -> bool:
    """Return True if xarray can open the NetCDF file; False if file is corrupt/incomplete."""
    try:
        with xr.open_dataset(path, chunks=None) as _:
            return True
    except Exception:
        return False


def _atomic_to_netcdf(obj: xr.Dataset | xr.DataArray, path: Path, *, encoding: dict | None = None) -> None:
    """Write NetCDF to ``path`` via a temp file + ``os.replace`` so timeouts never leave a half-written target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        obj.to_netcdf(tmp, encoding=encoding)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    os.replace(tmp, path)


def _step2_apply_dask_env() -> None:
    try:
        import dask

        dask.config.set({"array.slicing.split_large_chunks": False})
        sched = os.environ.get("DASK_SCHEDULER")
        if sched is not None and str(sched).strip() != "":
            dask.config.set({"scheduler": str(sched).strip()})
    except Exception:
        pass


def _step2_open_raw_to_mjjas(
    *,
    in_files: list[Path],
    cfg: PreprocessConfig,
    bbox: dict[str, float] | None,
) -> tuple[xr.Dataset, xr.DataArray, str, str, str]:
    """Open raw ERA5, subset bbox, daily mean, MJJAS. Caller owns ``Dataset`` until closed."""
    if len(in_files) == 0:
        raise FileNotFoundError("No input NetCDF files provided.")

    path0 = Path(in_files[0]).resolve()
    with xr.open_dataset(path0, decode_times=True, chunks=None) as ds0:
        zvar0 = infer_z500_var(ds0)
        z0 = ds0[zvar0]
        tdim0 = pick_time_dim_from_dataset(ds0, time_dim_candidates=cfg.time_dim_candidates)
        lat0 = pick_coord_name(z0, cfg.lat_dim_candidates)
        lon0 = pick_coord_name(z0, cfg.lon_dim_candidates)

    native = infer_netcdf_variable_chunk_sizes(path0, zvar0)
    _tchunk = _step2_time_chunk_from_env_and_native(time_dim=tdim0, native=native)
    lat_c, lon_c = _step2_lat_lon_chunks_from_env_and_native(native=native, lat_dim=lat0, lon_dim=lon0)

    if _tchunk is None:
        ds = open_era5_z500_raw_mfdataset(in_files, cfg=cfg, chunks=None)
    else:
        chunks_open: dict[str, int] = {tdim0: int(_tchunk)}
        if lat_c is not None:
            chunks_open[lat0] = int(lat_c)
        if lon_c is not None:
            chunks_open[lon0] = int(lon_c)

        try:
            ds = open_era5_z500_raw_mfdataset(in_files, cfg=cfg, chunks=chunks_open)
        except ValueError as exc:
            err = str(exc).lower()
            if "chunk manager" in err or "unrecognized chunk" in err:
                warnings.warn(
                    "Step 2: opening raw ERA5 without Dask chunking (no usable xarray chunk manager). "
                    "For long extracts on HPC, install Dask (e.g. `conda install dask` or `pip install dask`). "
                    f"Falling back to chunks=None. Original error: {exc!r}",
                    UserWarning,
                    stacklevel=2,
                )
                ds = open_era5_z500_raw_mfdataset(in_files, cfg=cfg, chunks=None)
            else:
                raise

    zvar = infer_z500_var(ds)
    Z = ds[zvar]
    Z = select_pressure_level(Z, cfg=cfg)
    Z = geopotential_to_height_if_needed(Z, cfg=cfg)
    try:
        Z = Z.astype("float32")
    except Exception:
        pass
    Z.name = "z500"

    time_dim = pick_coord_name(Z, cfg.time_dim_candidates)
    lat_dim = pick_coord_name(Z, cfg.lat_dim_candidates)
    lon_dim = pick_coord_name(Z, cfg.lon_dim_candidates)

    if bbox is not None:
        b = _validate_bbox(bbox)
        lat_vals = np.asarray(Z[lat_dim].values)
        lat_desc = bool(lat_vals.size >= 2 and lat_vals[0] > lat_vals[-1])
        lat_slice = slice(b["N"], b["S"]) if lat_desc else slice(b["S"], b["N"])
        lon_slice = slice(b["W"], b["E"])
        Z = Z.sel({lat_dim: lat_slice, lon_dim: lon_slice})

        if int(Z.sizes.get(lat_dim, 0)) == 0 or int(Z.sizes.get(lon_dim, 0)) == 0:
            raise ValueError(f"BBox selection produced empty domain: bbox={b}")

    def _step2_rechunk_spatiotemporal(da: xr.DataArray) -> xr.DataArray:
        if _tchunk is None or time_dim not in da.dims:
            return da
        spec: dict[str, int] = {time_dim: int(_tchunk)}
        if lat_c is not None and lat_dim in da.dims:
            spec[lat_dim] = min(int(lat_c), int(da.sizes[lat_dim]))
        if lon_c is not None and lon_dim in da.dims:
            spec[lon_dim] = min(int(lon_c), int(da.sizes[lon_dim]))
        return da.chunk(spec)

    Z = _step2_rechunk_spatiotemporal(Z)

    try:
        Z_daily = Z.resample({time_dim: "1D"}).mean(skipna=True)
    except TypeError:
        Z_daily = Z.resample({time_dim: "1D"}).mean()
    Z_mjjas = select_mjjas(Z_daily, time_dim=time_dim, months=cfg.season_months)
    Z_mjjas = _step2_rechunk_spatiotemporal(Z_mjjas)
    validate_mjjas_daily_coverage(Z_mjjas, time_dim=time_dim)

    return ds, Z_mjjas, time_dim, lat_dim, lon_dim


def _step2_mjjas_to_detrended(
    Z_mjjas: xr.DataArray,
    *,
    cfg: PreprocessConfig,
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
) -> xr.DataArray:
    if str(cfg.detrend_method).lower() in ("gridpoint", "grid", "gp"):
        return detrend_linear(Z_mjjas, time_dim=time_dim)
    if str(cfg.detrend_method).lower() in ("domain_mean", "domain-mean", "domainmean", "dm"):
        return detrend_domain_mean(Z_mjjas, time_dim=time_dim, lat_dim=lat_dim, lon_dim=lon_dim)
    raise ValueError(
        f"Unknown cfg.detrend_method={cfg.detrend_method!r}. Expected 'gridpoint' or 'domain_mean'."
    )


def _step2_detrended_to_results(
    Z_det: xr.DataArray,
    *,
    cfg: PreprocessConfig,
    time_dim: str,
) -> dict[int, dict[str, xr.DataArray]]:
    Z_clim_raw = mjjas_season_day_climatology_mean_unstack(Z_det, time_dim=time_dim)

    results: dict[int, dict[str, xr.DataArray]] = {}
    for w in cfg.smooth_windows:
        clim_smooth = cyclic_running_mean(Z_clim_raw, dim="season_day", window=int(w))
        clim_smooth.attrs["smoothing"] = (
            f"{int(w)}-day centered running mean with cyclic wrap within MJJAS"
        )

        if str(cfg.climatology_subtract).lower() in ("raw", "clim_raw", "unsmoothed"):
            clim_for_sub = Z_clim_raw
            sub_note = "raw"
        elif str(cfg.climatology_subtract).lower() in ("smooth", "smoothed", "clim_smooth"):
            clim_for_sub = clim_smooth
            sub_note = "smooth"
        else:
            raise ValueError(
                f"Unknown cfg.climatology_subtract={cfg.climatology_subtract!r}. Expected 'raw' or 'smooth'."
            )

        anom = subtract_mjjas_season_day_climatology(Z_det, clim_for_sub, time_dim=time_dim)
        anom.name = "z500_anom"
        anom.attrs["description"] = (
            f"Detrended Z500 anomaly relative to {sub_note} MJJAS day-of-season climatology"
        )
        anom.attrs["climatology_window_days"] = int(w)
        anom.attrs["climatology_subtract"] = str(cfg.climatology_subtract)

        results[int(w)] = {"clim_raw": Z_clim_raw, "clim_smooth": clim_smooth, "anom": anom}

    for w, pack in results.items():
        has_nan = _any_nan_time_slabwise(pack["anom"], time_dim=time_dim)
        if has_nan:
            raise ValueError(
                f"Step 2 produced NaNs in z500_anom (smoothing window {w} d). "
                "Check raw missing values, MJJAS coverage, and climatology fields."
            )

    return results


def _step2_bbox_nc_attrs(
    bbox: dict[str, float] | None,
    *,
    bbox_step: str | None,
    bbox_config_path: Path,
    bbox_source: str | None,
) -> dict[str, str | float]:
    if bbox is None:
        return {}
    return {
        "bbox_definition": "N,W,S,E",
        "bbox_step": str(bbox_step),
        "bbox_config": str(bbox_config_path),
        "bbox_source": str(bbox_source),
        "bbox_N": float(bbox["N"]),
        "bbox_W": float(bbox["W"]),
        "bbox_S": float(bbox["S"]),
        "bbox_E": float(bbox["E"]),
        "bbox_note": "Subset applied to raw Z500 at Step 2 before resample/detrend/climatology.",
    }


def _raise_missing_prereqs_for_steps_7_8(
    *,
    stop_after: int,
    overwrite: bool,
    raw_inputs_present: bool,
    out_dir: Path,
    opt_label: str,
    prepro_path: Path,
    anom_path: Path,
    det_path: Path,
    clim_raw_path: Path,
    pack_path: Path,
    pca_path: Path,
    det_ok: bool,
    pack_ok: bool,
    clim_raw_ok: bool,
) -> None:
    """Fail fast when --stop-after-step is 7 or 8 but no Step 2–6 chain and no readable PCA yet."""
    if overwrite or stop_after < 7:
        return
    if raw_inputs_present:
        return
    prepro_ok = prepro_path.exists() and _is_netcdf_readable(prepro_path)
    anom_ok = anom_path.exists() and _is_netcdf_readable(anom_path)
    pca_ok = pca_path.exists() and _is_netcdf_readable(pca_path)
    can_feed_step7 = prepro_ok or anom_ok or (det_ok and pack_ok and clim_raw_ok)
    if pca_ok or can_feed_step7:
        return
    lines: list[str] = []
    if not det_ok:
        lines.append(f"Step 2 detrended Z500: {det_path}")
    if not clim_raw_ok:
        lines.append(f"Step 2 climatology (raw): {clim_raw_path}")
    if not pack_ok:
        lines.append(f"Step 2 anomaly pack (clim_smooth + anom): {pack_path}")
    if not anom_ok:
        lines.append(f"Step 3 anomaly: {anom_path}")
    if not prepro_ok:
        lines.append(f"Step 4–6 preprocessed: {prepro_path}")
    if not pca_ok:
        lines.append(f"Step 7 PCA: {pca_path}")
    raise FileNotFoundError(
        "Cannot run Steps 7–8: no raw ERA5 input was resolved for Step 2, and none of the prerequisite "
        f"NetCDF files are present or readable under {out_dir} for this preprocessing tag (`{opt_label}`). "
        "Copy Step 2–6 outputs (and Step 7 PCA if you only need to (re)run Step 8) from your HPC run into "
        "that folder, point --raw-nc / --raw-dir at local ERA5, or run the full pipeline where raw data "
        "is available (e.g. --stop-after-step 6 first, then 8).\n"
        "Missing or unreadable:\n  - " + "\n  - ".join(lines)
    )


def _step2_pack_has_finite_climatology(path: Path, *, max_nan_frac: float = 0.01) -> bool:
    """True only if z500_clim_smooth exists and is not mostly NaN.

    A readable NetCDF can still be useless after a partial write or a run that skipped saving
    real data; `_is_netcdf_readable` alone is not enough for pack_path gating.
    """
    try:
        with xr.open_dataset(path, chunks=None) as ds:
            if "z500_clim_smooth" not in ds:
                return False
            c = ds["z500_clim_smooth"]
            if int(c.size) == 0:
                return False
            nf = float(c.isnull().mean().compute())
            return nf <= float(max_nan_frac)
    except Exception:
        return False


def parse_pipeline_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline: run preprocessing Steps 2–6 (optionally Step 7 PCA and Step 8 HC).")
    p.add_argument("--region", type=str, default="europe", help="Region subfolder under data/*.")
    p.add_argument("--raw-dir", type=Path, default=None, help="Raw ERA5 input directory (override).")
    p.add_argument("--out-dir", type=Path, default=None, help="Preprocessed output directory (override).")
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory to save diagnostics figures (default: figures/<region>/preprocessing).",
    )
    p.add_argument(
        "--no-figs",
        action="store_true",
        help="Disable saving diagnostic figures.",
    )
    p.add_argument("--smooth-window", type=int, default=21, help="Odd smoothing window in days.")
    p.add_argument("--epsilon-m", type=float, default=20.0, help="σ floor (meters) for normalisation.")
    p.add_argument("--target-pressure-hpa", type=int, default=500, help="Pressure level (hPa) to select.")
    p.add_argument(
        "--detrend-method",
        type=str,
        default="gridpoint",
        choices=("gridpoint", "domain_mean"),
        help="Detrending: 'gridpoint' (default) or 'domain_mean' (remove trend of daily cos(lat)-weighted domain mean).",
    )
    p.add_argument(
        "--climatology-subtract",
        type=str,
        default="raw",
        choices=("raw", "smooth"),
        help="Anomalies: subtract raw season-day climatology ('raw', default) or smoothed climatology ('smooth').",
    )
    p.add_argument(
        "--sigma-mode",
        type=str,
        default="field",
        choices=("field", "scalar"),
        help="Step 4 σ normalisation: 'field' (σ(season_day,lat,lon), default) or 'scalar' (cos(lat)-weighted area-mean σ(season_day)).",
    )
    p.add_argument(
        "-f",
        "--overwrite",
        "--force",
        action="store_true",
        dest="overwrite",
        help="Recompute and replace existing outputs (Steps 2–6).",
    )
    p.add_argument(
        "--raw-years-span",
        type=str,
        default=(os.environ.get("ERA5_RAW_YEARS_SPAN") or "").strip() or None,
        help="When several *.nc exist in raw dir, pick the file whose name contains years_SPAN or "
        "mjjas_SPAN (e.g. 1979-2025 vs 2020-2023). Env: ERA5_RAW_YEARS_SPAN.",
    )
    p.add_argument(
        "--raw-nc",
        type=Path,
        default=None,
        help="Use exactly this NetCDF (overrides --raw-years-span).",
    )
    p.add_argument(
        "--bbox-step",
        type=str,
        default=(os.environ.get("BBOX_STEP") or "").strip() or None,
        help="Bbox step key (e.g. B0..B5) loaded from bbox-steps YAML. Env: BBOX_STEP.",
    )
    p.add_argument(
        "--bbox-config",
        type=Path,
        default=(Path(os.environ["BBOX_CONFIG"]) if os.environ.get("BBOX_CONFIG") else None),
        help="Path to bbox-steps YAML config (default: <module_root>/src/bbox_steps_europe.yaml). Env: BBOX_CONFIG.",
    )
    p.add_argument(
        "--bbox",
        type=str,
        default=(os.environ.get("BBOX") or "").strip() or None,
        help='Optional explicit bbox override "N,W,S,E" (CDS/EPE order). Env: BBOX.',
    )
    p.add_argument(
        "--stop-after-step",
        type=int,
        default=6,
        choices=(2, 3, 6, 7, 8),
        help=(
            "Stop after preprocessing step. "
            "2 = Step 2 only; "
            "3 = Steps 2–3; "
            "6 = Steps 2–6; "
            "7 = also run Step 7 PCA; "
            "8 = also run Step 8 hierarchical clustering."
        ),
    )
    p.add_argument(
        "--step2-stage",
        type=str,
        default=(os.environ.get("STEP2_STAGE") or "full").strip() or "full",
        choices=("full", "mjjas", "detrend", "clim"),
        help=(
            "Step 2 mode. 'full' (default): one pass, no 02a checkpoint. "
            "'mjjas': write 02a pre-detrend checkpoint only. 'detrend': checkpoint → detrended 02_*. "
            "'clim': detrended → clim raw + anomaly pack. Run mjjas→detrend→clim in order before "
            "--stop-after-step > 2. Env: STEP2_STAGE."
        ),
    )
    p.add_argument(
        "--yeo-johnson",
        action="store_true",
        dest="yeo_johnson",
        help="Apply Yeo-Johnson power transform to the raw daily MJJAS field before detrending (adds '_yj' to output filenames).",
    )
    p.add_argument(
        "--allow-recompute-steps-2-6",
        action="store_true",
        help=(
            "When --stop-after-step is 7 or 8: by default the pipeline does not re-run Steps 2–6 "
            "(expects existing 02–06 NetCDFs). Pass this flag to allow rebuilding them from raw ERA5 "
            "when missing (heavy). Incompatible with expecting a quick PCA/HC-only refresh."
        ),
    )
    p.add_argument(
        "--var-threshold",
        type=float,
        default=0.90,
        help="Step 7: cumulative explained variance threshold for selecting number of PCs.",
    )
    return p.parse_args()


def enforce_eof_sign_convention(
    eofs_space: np.ndarray, pcs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve EOF/PCA sign ambiguity deterministically.

    Convention: for each mode, ensure the element with maximum absolute loading is positive.
    This flips both EOF and the corresponding PC time series.
    """
    eofs = np.array(eofs_space, copy=True)
    pcs_out = np.array(pcs, copy=True)

    for k in range(int(eofs.shape[0])):
        j = int(np.argmax(np.abs(eofs[k])))
        if eofs[k, j] < 0:
            eofs[k] *= -1.0
            pcs_out[:, k] *= -1.0

    return eofs, pcs_out


def run_step7_pca(
    *,
    in_prepro_path: Path,
    out_pca_path: Path,
    var_threshold: float,
    trace_attrs: dict[str, str | int | float] | None = None,
) -> None:
    """Step 7: EOF/PCA decomposition on ``X_matrix(time, space)``.

    Mirrors the outputs of the now-retired `src/07_PCA.ipynb` prototype notebook:
    - PCs_reduced(time, mode)
    - EOFs_space(mode, space)
    - EOF_maps(mode, lat, lon)
    - explained_variance_ratio(mode)
    - cumulative_explained_variance(mode)
    plus pass-through mapping arrays latitude_space/longitude_space when present.
    """
    try:
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Step 7 requires scikit-learn (sklearn).") from e

    with xr.open_dataset(in_prepro_path, chunks=None) as ds:
        if "X_matrix" not in ds:
            raise KeyError(f"Expected 'X_matrix' in {in_prepro_path}, found: {list(ds.data_vars)}")
        X_da = ds["X_matrix"].load()
        time_dim = pick_coord_name(X_da, ("time", "valid_time"))
        time_coord = X_da[time_dim] if time_dim in X_da.coords else ds[time_dim]

        latitude_space = ds.get("latitude_space")
        longitude_space = ds.get("longitude_space")

        # Prefer lat/lon from the unflattened field if available (most robust)
        if "z500_weighted" in ds:
            dims_w = infer_core_dims(ds["z500_weighted"], cfg=PreprocessConfig())
            lat_dim = dims_w["lat"]
            lon_dim = dims_w["lon"]
            lat = ds["z500_weighted"][lat_dim].values
            lon = ds["z500_weighted"][lon_dim].values
        else:
            # Fallback if only X_matrix exists
            lat_dim = "lat"
            lon_dim = "lon"
            lat = ds.get("lat", ds.get("latitude"))
            lon = ds.get("lon", ds.get("longitude"))
            if lat is None or lon is None:
                raise KeyError("Could not infer lat/lon coordinates from dataset.")
            lat = np.asarray(lat)
            lon = np.asarray(lon)

        X = np.asarray(X_da.values, dtype=np.float32)
        finite_frac = float(np.isfinite(X).mean())
        if finite_frac != 1.0:
            raise ValueError(f"X_matrix contains non-finite values (finite fraction: {finite_frac})")

    pca = PCA()
    PCs = pca.fit_transform(X)  # (time, mode)
    EOFs_space = pca.components_  # (mode, space)
    explained_var = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained_var)

    k = int(np.searchsorted(cumulative, float(var_threshold)) + 1)
    k = max(1, min(k, PCs.shape[1]))
    PCs_reduced = PCs[:, :k]
    EOFs_reduced_space = EOFs_space[:k, :]

    # Deterministic sign convention (EOF/PC sign ambiguity)
    EOFs_reduced_space, PCs_reduced = enforce_eof_sign_convention(EOFs_reduced_space, PCs_reduced)

    n_lat = int(np.asarray(lat).shape[0])
    n_lon = int(np.asarray(lon).shape[0])
    EOF_maps = EOFs_reduced_space.reshape((k, n_lat, n_lon)).astype(np.float32)

    mode = np.arange(k, dtype=int)
    space_coord = (
        X_da["space"].values
        if "space" in X_da.coords
        else np.arange(int(np.asarray(X).shape[1]), dtype=int)
    )

    attrs: dict[str, str | int | float] = {
        "step": "07_PCA",
        "source_preprocessed": str(in_prepro_path),
        "var_threshold": float(var_threshold),
        "k": int(k),
        "pca_convention": "flip each EOF/PC so max(|EOF|) is positive",
        "flattening_order": "lat outer, lon inner (xarray stack space=(lat,lon))",
    }
    if trace_attrs:
        attrs.update(trace_attrs)

    ds_out = xr.Dataset(
        data_vars={
            "PCs_reduced": xr.DataArray(
                np.asarray(PCs_reduced, dtype=np.float32),
                dims=[time_dim, "mode"],
                coords={time_dim: time_coord, "mode": mode},
                name="PCs_reduced",
            ),
            "EOFs_space": xr.DataArray(
                np.asarray(EOFs_reduced_space, dtype=np.float32),
                dims=["mode", "space"],
                coords={"mode": mode, "space": space_coord},
                name="EOFs_space",
            ),
            "EOF_maps": xr.DataArray(
                EOF_maps,
                dims=["mode", lat_dim, lon_dim],
                coords={"mode": mode, lat_dim: np.asarray(lat), lon_dim: np.asarray(lon)},
                name="EOF_maps",
            ),
            "explained_variance_ratio": xr.DataArray(
                explained_var[:k].astype(np.float32),
                dims=["mode"],
                coords={"mode": mode},
                name="explained_variance_ratio",
            ),
            "cumulative_explained_variance": xr.DataArray(
                cumulative[:k].astype(np.float32),
                dims=["mode"],
                coords={"mode": mode},
                name="cumulative_explained_variance",
            ),
        },
        attrs=attrs,
    )

    if latitude_space is not None:
        ds_out["latitude_space"] = latitude_space.astype("float32")
    if longitude_space is not None:
        ds_out["longitude_space"] = longitude_space.astype("float32")

    out_pca_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_pca_path)


def run_step8_hierarchical_clustering(
    *,
    in_pca_path: Path,
    out_hc_path: Path,
    trace_attrs: dict[str, str | int | float] | None = None,
) -> None:
    """Step 8: Ward hierarchical clustering on PCs_reduced.

    Writes ``ward_linkage`` and candidate ``labels_k*`` (k=4…8), matching
    ``08_hierarchical_clustering.ipynb`` so summary PDFs do not need on-the-fly cuts.
    """
    try:
        from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Step 8 requires SciPy (scipy).") from e

    with xr.open_dataset(in_pca_path, chunks=None) as ds:
        if "PCs_reduced" not in ds:
            raise KeyError(f"Expected 'PCs_reduced' in {in_pca_path}, found: {list(ds.data_vars)}")
        PC_da = ds["PCs_reduced"].load()
        X_pca = np.asarray(PC_da.values, dtype=np.float64)

    Z = linkage(X_pca, method="ward")  # (n-1, 4)
    time_dims = [d for d in PC_da.dims if d != "mode"]
    if not time_dims:
        raise ValueError(f"PCs_reduced has no time dimension: dims={PC_da.dims}")
    time_dim = time_dims[0]
    time_coord = PC_da[time_dim].values

    attrs: dict[str, str | int | float] = {"step": "08_hierarchical_clustering", "source_pca": str(in_pca_path)}
    if trace_attrs:
        attrs.update(trace_attrs)
    data_vars: dict[str, xr.DataArray] = {
        "ward_linkage": xr.DataArray(
            Z.astype(np.float64),
            dims=["merge", "col"],
            coords={"merge": np.arange(Z.shape[0], dtype=int), "col": np.arange(4, dtype=int)},
            name="ward_linkage",
        )
    }
    n_samples = int(X_pca.shape[0])
    for kk in range(4, 9):
        if kk < 2 or kk > n_samples:
            continue
        labels = np.asarray(fcluster(Z, kk, criterion="maxclust"), dtype=np.int16)
        data_vars[f"labels_k{kk}"] = xr.DataArray(
            labels,
            dims=[time_dim],
            coords={time_dim: time_coord},
        )
    ds_out = xr.Dataset(data_vars=data_vars, attrs=attrs)
    out_hc_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_hc_path)


def run_pipeline_steps_2_to_6() -> None:
    args = parse_pipeline_args()
    t0 = time.perf_counter()

    def stamp(msg: str) -> None:
        dt = time.perf_counter() - t0
        stop = int(getattr(args, "stop_after_step", 6))
        if stop <= 2:
            label = "2"
        elif stop == 3:
            label = "2–3"
        elif stop <= 6:
            label = "2–6"
        elif stop == 7:
            label = "2–7"
        else:
            label = "2–8"
        print(f"[prepro {label} | +{dt:7.1f}s] {msg}")

    module_root = find_module_root()
    repo_root = module_root.parent

    # Default to Europe bbox-step config when none is specified (region-specific configs are
    # typically provided by the Slurm scripts via BBOX_CONFIG).
    bbox_config_path = (args.bbox_config or (module_root / "src" / "bbox_steps_europe.yaml")).resolve()
    bbox_step = (args.bbox_step or "").strip() or None
    bbox: dict[str, float] | None = None
    bbox_source: str | None = None
    if bbox_step:
        steps = _load_bbox_steps_yaml(bbox_config_path)
        if bbox_step not in steps:
            raise KeyError(
                f"Unknown --bbox-step={bbox_step!r}. Available: {sorted(steps)} (from {bbox_config_path})"
            )
        step_entry = steps[bbox_step]
        if not isinstance(step_entry, dict) or "bbox" not in step_entry:
            raise KeyError(f"bbox_steps entry for {bbox_step!r} has no 'bbox' field: {step_entry!r}")
        bbox = _validate_bbox(step_entry["bbox"])  # type: ignore[arg-type]
        bbox_source = f"config:{bbox_config_path}"
    if args.bbox:
        bbox = _validate_bbox(_parse_bbox_string(args.bbox))
        bbox_source = "cli:--bbox"
        bbox_step = bbox_step or "custom"

    raw_dir_default, out_dir_default = build_default_paths(module_root=module_root, region=args.region)
    raw_dir = args.raw_dir or raw_dir_default
    out_dir = args.out_dir or out_dir_default
    figures_dir = args.figures_dir or build_default_figures_dir(module_root=module_root, region=args.region)
    save_figs = (not args.no_figs) and (plt is not None)

    smooth_window = int(args.smooth_window)
    if smooth_window <= 0 or smooth_window % 2 != 1:
        raise ValueError(f"--smooth-window must be a positive odd integer. Got {smooth_window}")

    _yeo_johnson = bool(getattr(args, "yeo_johnson", False))
    _epsilon_m = float(args.epsilon_m)
    if _yeo_johnson and _epsilon_m >= 1.0:
        # After YJ transform output is ~N(0,1); physical epsilon (e.g. 20 m) would
        # always trigger the floor and flatten spatial sigma variability. Use a small
        # value that only guards against truly degenerate near-zero sigma cells.
        _epsilon_m = 0.01

    cfg = PreprocessConfig(
        smooth_windows=(smooth_window,),
        target_pressure_hpa=int(args.target_pressure_hpa),
        detrend_method=str(args.detrend_method),
        climatology_subtract=str(args.climatology_subtract),
        epsilon_m=_epsilon_m,
        sigma_mode=str(args.sigma_mode),
        yeo_johnson=_yeo_johnson,
    )

    stamp(f"Repo root:   {repo_root}")
    stamp(f"Module root: {module_root}")
    stamp(f"Raw dir:     {raw_dir}")
    stamp(f"Out dir:     {out_dir}")
    stamp(f"Region:      {args.region}")
    stamp(f"Window days: {smooth_window}")
    stamp(f"Epsilon (m): {float(args.epsilon_m)}")
    stamp(f"Figures dir: {figures_dir}" + ("" if save_figs else " (disabled)"))
    if bbox is not None:
        stamp(f"BBox step:   {bbox_step}")
        stamp(f"BBox:        {bbox} ({bbox_source})")

    stop_after = int(args.stop_after_step)
    # Steps 7–8 read Step 4–6 / PCA; do not silently start the huge Step 2 chain unless opted in.
    require_prepro_chain = stop_after >= 7 and (not bool(getattr(args, "allow_recompute_steps_2_6", False)))
    if require_prepro_chain and bool(args.overwrite):
        raise ValueError(
            "With --stop-after-step 7 or 8 the pipeline expects existing Step 2–6 files and does not "
            "recompute them; use of --overwrite / -f is not supported in that mode. "
            "Rebuild preprocessing with --stop-after-step 6 (or lower) without --overwrite on these "
            "outputs, or pass --allow-recompute-steps-2-6 if you intentionally want to recompute from raw."
        )

    in_files = select_raw_era5_nc_files(
        raw_dir, years_span=args.raw_years_span, raw_nc=args.raw_nc
    )
    stamp(f"Raw NetCDF input(s): {[p.name for p in in_files]}")

    years_label = infer_years_label_from_files(in_files, cfg=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    detr_label = "dm" if str(args.detrend_method) == "domain_mean" else "gp"
    climsub_label = "clims" if str(args.climatology_subtract) == "smooth" else "climr"
    sigma_label = "sigsc" if str(args.sigma_mode) == "scalar" else "sigf"
    opt_label = f"detr{detr_label}_{climsub_label}_{sigma_label}"
    if cfg.yeo_johnson:
        opt_label = f"{opt_label}_yj"
    if bbox is not None:
        opt_label = f"{opt_label}_step{bbox_step}_{bbox_slug(bbox)}"

    # Step 2 outputs (option-aware so we never overwrite different method variants)
    # Include wXX in filenames for consistent provenance across Steps 2–7 even when the product
    # itself is not mathematically window-dependent (detrended field + raw climatology).
    det_path = out_dir / f"02_z500_mjjas_detrended_w{smooth_window:02d}_{opt_label}_{years_label}.nc"
    clim_raw_path = out_dir / f"02_z500_mjjas_climatology_raw_w{smooth_window:02d}_{opt_label}_{years_label}.nc"
    pack_path = out_dir / f"02_z500_mjjas_climatology_anomalies_w{smooth_window:02d}_{opt_label}_{years_label}.nc"
    mjjas_ckpt_path = out_dir / f"02a_z500_mjjas_pre_detrend_w{smooth_window:02d}_{opt_label}_{years_label}.nc"

    # Step 3 output
    anom_path = out_dir / f"03_Z500_anomaly_w{smooth_window:02d}_{opt_label}_{years_label}.nc"

    # Step 4–6 output
    eps_label = int(round(float(args.epsilon_m)))
    prepro_path = out_dir / f"04_05_06_Z500_preprocessed_w{smooth_window:02d}_eps{eps_label:02d}_{opt_label}_{years_label}.nc"

    var_label = int(float(args.var_threshold) * 100)
    processed_dir = module_root / "data" / "processed" / args.region
    pca_path = processed_dir / (
        f"07_Z500_PCA_w{smooth_window:02d}_eps{eps_label:02d}_var{var_label:02d}_{opt_label}_{years_label}.nc"
    )
    hc_path = processed_dir / (
        f"08_Z500_HC_Ward_w{smooth_window:02d}_eps{eps_label:02d}_var{var_label:02d}_{opt_label}_{years_label}.nc"
    )

    # Step 2 compute/write
    step2_stage = str(getattr(args, "step2_stage", "full")).strip().lower()
    if step2_stage not in ("full", "mjjas", "detrend", "clim"):
        step2_stage = "full"

    det_ok = det_path.exists() and _is_netcdf_readable(det_path)
    clim_raw_ok = clim_raw_path.exists() and _is_netcdf_readable(clim_raw_path)
    pack_ok = (
        pack_path.exists()
        and _is_netcdf_readable(pack_path)
        and _step2_pack_has_finite_climatology(pack_path)
    )
    mjjas_ok = mjjas_ckpt_path.exists() and _is_netcdf_readable(mjjas_ckpt_path)

    _raise_missing_prereqs_for_steps_7_8(
        stop_after=stop_after,
        overwrite=bool(args.overwrite),
        raw_inputs_present=bool(in_files),
        out_dir=out_dir,
        opt_label=opt_label,
        prepro_path=prepro_path,
        anom_path=anom_path,
        det_path=det_path,
        clim_raw_path=clim_raw_path,
        pack_path=pack_path,
        pca_path=pca_path,
        det_ok=det_ok,
        pack_ok=pack_ok,
        clim_raw_ok=clim_raw_ok,
    )

    need_step2 = args.overwrite or (not det_ok) or (not clim_raw_ok) or (not pack_ok)
    if require_prepro_chain and need_step2 and step2_stage == "full":
        raise FileNotFoundError(
            "Steps 7–8 need existing Step 2 products; refusing to run Step 2 from this stop level. "
            "Copy or build 02/03/04_05_06 for this preprocessing tag, or run the pipeline with "
            "--allow-recompute-steps-2-6 to rebuild from raw (heavy).\n"
            f"  det_ok={det_ok}  -> {det_path}\n"
            f"  clim_raw_ok={clim_raw_ok}  -> {clim_raw_path}\n"
            f"  pack_ok={pack_ok}  -> {pack_path}"
        )

    _step2_opt_attrs = {
        "detrend_method": str(args.detrend_method),
        "climatology_subtract": str(args.climatology_subtract),
        "sigma_mode": str(args.sigma_mode),
    }
    _bbox_attrs = _step2_bbox_nc_attrs(
        bbox,
        bbox_step=bbox_step,
        bbox_config_path=bbox_config_path,
        bbox_source=bbox_source,
    )

    def _step2_stamp_diagnostic_figures() -> None:
        if not save_figs:
            return
        stamp("Step 2: figures ...")
        try:
            with xr.open_dataset(det_path, chunks=None) as ds_det_fig:
                z_det_fig = ds_det_fig["z500_det"]
                dims_det = infer_core_dims(z_det_fig, cfg=cfg)
                tdim = dims_det["time"]
                lat_dim = dims_det["lat"]
                lon_dim = dims_det["lon"]
                ts_det = z_det_fig.mean(dim=(lat_dim, lon_dim))
                save_timeseries_figure(
                    da=ts_det,
                    out_path=figures_dir
                    / f"step02_detrended_spatial_mean_{opt_label}_years_{years_label}.png",
                    title="Step 2: spatial mean of detrended Z500 (MJJAS)",
                )
            with xr.open_dataset(pack_path, chunks=None) as ds_pack_fig:
                clim = ds_pack_fig["z500_clim_smooth"]
                mid_day = int(clim["season_day"].values[len(clim["season_day"]) // 2])
                save_map_figure(
                    da=clim.sel(season_day=mid_day),
                    out_path=figures_dir
                    / f"step02_climatology_smooth_midday_w{smooth_window:02d}_{opt_label}_years_{years_label}.png",
                    title=f"Step 2: smoothed climatology (season_day={mid_day}, window={smooth_window}d)",
                )
                anom = ds_pack_fig["z500_anom"]
                save_map_figure(
                    da=anom.mean(dim=tdim),
                    out_path=figures_dir
                    / f"step02_anomaly_temporal_mean_w{smooth_window:02d}_{opt_label}_years_{years_label}.png",
                    title="Step 2: temporal mean of anomalies (should be ~0)",
                )
        except Exception as e:
            stamp(f"Step 2: figures skipped due to read/plot error: {type(e).__name__}: {e}")

    if step2_stage == "full":
        if need_step2:
            stamp("Step 2: computing detrended field + climatology/anomalies ...")
            Z_det, results = build_step2_products(in_files=in_files, cfg=cfg, bbox=bbox)
            stamp("Step 2: computation finished.")

            stamp("Step 2: writing outputs ...")
            if args.overwrite or (not det_ok):
                ds_det_w = Z_det.to_dataset(name="z500_det")
                ds_det_w.attrs.update(
                    mjjas_pipeline_global_attrs(
                        years_label,
                        climatology_window_days=None,
                        preprocessing_step="02_z500_mjjas_detrended",
                        region=args.region,
                    )
                )
                if _bbox_attrs:
                    ds_det_w.attrs.update(_bbox_attrs)
                ds_det_w.attrs.update(_step2_opt_attrs)
                _atomic_to_netcdf(ds_det_w, det_path)
                stamp(f"Wrote: {det_path.name}")
            else:
                stamp(f"Exists, skipping: {det_path.name}")

            if args.overwrite or (not clim_raw_ok):
                any_pack = results[smooth_window]
                ds_clim_raw = xr.Dataset({"z500_clim_raw": any_pack["clim_raw"]})
                ds_clim_raw.attrs.update(
                    mjjas_pipeline_global_attrs(
                        years_label,
                        climatology_window_days=None,
                        preprocessing_step="02_z500_mjjas_climatology_raw",
                        region=args.region,
                    )
                )
                if _bbox_attrs:
                    ds_clim_raw.attrs.update(_bbox_attrs)
                ds_clim_raw.attrs.update(_step2_opt_attrs)
                _atomic_to_netcdf(ds_clim_raw, clim_raw_path)
                stamp(f"Wrote: {clim_raw_path.name}")
            else:
                stamp(f"Exists, skipping: {clim_raw_path.name}")

            if args.overwrite or (not pack_ok):
                pack = results[smooth_window]
                ds_pack = xr.Dataset({"z500_clim_smooth": pack["clim_smooth"], "z500_anom": pack["anom"]})
                ds_pack.attrs.update(
                    mjjas_pipeline_global_attrs(
                        years_label,
                        climatology_window_days=smooth_window,
                        preprocessing_step="02_z500_mjjas_climatology_anomaly_pack",
                        region=args.region,
                    )
                )
                if _bbox_attrs:
                    ds_pack.attrs.update(_bbox_attrs)
                ds_pack.attrs["season"] = "MJJAS"
                ds_pack.attrs.update(_step2_opt_attrs)
                _atomic_to_netcdf(ds_pack, pack_path)
                stamp(f"Wrote: {pack_path.name}")
            else:
                stamp(f"Exists, skipping: {pack_path.name}")

            del Z_det, results
            gc.collect()
            _step2_stamp_diagnostic_figures()
        else:
            stamp("Step 2: outputs exist, skipping.")

    elif step2_stage == "mjjas":
        need_mjjas = args.overwrite or (not mjjas_ok)
        if need_mjjas:
            stamp("Step 2.1: MJJAS checkpoint (pre-detrend) ...")
            _step2_apply_dask_env()
            ds_m, Z_mjjas, _, _, _ = _step2_open_raw_to_mjjas(in_files=in_files, cfg=cfg, bbox=bbox)
            try:
                ds_ck = Z_mjjas.to_dataset(name="z500_mjjas")
                ds_ck.attrs.update(
                    mjjas_pipeline_global_attrs(
                        years_label,
                        climatology_window_days=None,
                        preprocessing_step="02a_z500_mjjas_pre_detrend",
                        region=args.region,
                    )
                )
                if _bbox_attrs:
                    ds_ck.attrs.update(_bbox_attrs)
                ds_ck.attrs.update(_step2_opt_attrs)
                _atomic_to_netcdf(ds_ck, mjjas_ckpt_path)
                stamp(f"Wrote: {mjjas_ckpt_path.name}")
            finally:
                try:
                    ds_m.close()
                except Exception:
                    pass
        else:
            stamp(f"Step 2.1: checkpoint exists, skipping: {mjjas_ckpt_path.name}")

    elif step2_stage == "detrend":
        need_det_stage = args.overwrite or (not det_ok)
        if need_det_stage:
            if not mjjas_ok:
                raise FileNotFoundError(
                    "Step 2.2 (detrend) requires the MJJAS checkpoint from Step 2.1. Missing or unreadable:\n"
                    f"  {mjjas_ckpt_path}\n"
                    "Run first with: --step2-stage=mjjas (or set STEP2_STAGE=mjjas)."
                )
            stamp("Step 2.2: detrend from checkpoint ...")
            ck = xr.open_dataset(mjjas_ckpt_path, chunks=None)
            try:
                Z_mjjas = ck["z500_mjjas"]
                dims_m = infer_core_dims(Z_mjjas, cfg=cfg)
                Z_det = _step2_mjjas_to_detrended(
                    Z_mjjas,
                    cfg=cfg,
                    time_dim=dims_m["time"],
                    lat_dim=dims_m["lat"],
                    lon_dim=dims_m["lon"],
                )
                ds_det_w = Z_det.to_dataset(name="z500_det")
                ds_det_w.attrs.update(
                    mjjas_pipeline_global_attrs(
                        years_label,
                        climatology_window_days=None,
                        preprocessing_step="02_z500_mjjas_detrended",
                        region=args.region,
                    )
                )
                if _bbox_attrs:
                    ds_det_w.attrs.update(_bbox_attrs)
                ds_det_w.attrs.update(_step2_opt_attrs)
                _atomic_to_netcdf(ds_det_w, det_path)
                stamp(f"Wrote: {det_path.name}")
            finally:
                ck.close()
        else:
            stamp(f"Step 2.2: detrended file exists, skipping: {det_path.name}")

    elif step2_stage == "clim":
        need_clim_stage = args.overwrite or (not clim_raw_ok) or (not pack_ok)
        if need_clim_stage:
            if not det_ok:
                raise FileNotFoundError(
                    "Step 2.3 (climatology + anomalies) requires the detrended Step 2 file. Missing or unreadable:\n"
                    f"  {det_path}\n"
                    "Run first with: --step2-stage=detrend (after --step2-stage=mjjas)."
                )
            stamp("Step 2.3: climatology + anomaly pack from detrended ...")
            ds_d = xr.open_dataset(det_path, chunks=None)
            try:
                Z_det = ds_d["z500_det"]
                dims_d = infer_core_dims(Z_det, cfg=cfg)
                results = _step2_detrended_to_results(Z_det, cfg=cfg, time_dim=dims_d["time"])
                if args.overwrite or (not clim_raw_ok):
                    any_pack = results[smooth_window]
                    ds_clim_raw = xr.Dataset({"z500_clim_raw": any_pack["clim_raw"]})
                    ds_clim_raw.attrs.update(
                        mjjas_pipeline_global_attrs(
                            years_label,
                            climatology_window_days=None,
                            preprocessing_step="02_z500_mjjas_climatology_raw",
                            region=args.region,
                        )
                    )
                    if _bbox_attrs:
                        ds_clim_raw.attrs.update(_bbox_attrs)
                    ds_clim_raw.attrs.update(_step2_opt_attrs)
                    _atomic_to_netcdf(ds_clim_raw, clim_raw_path)
                    stamp(f"Wrote: {clim_raw_path.name}")
                else:
                    stamp(f"Exists, skipping: {clim_raw_path.name}")

                if args.overwrite or (not pack_ok):
                    pack = results[smooth_window]
                    ds_pack = xr.Dataset({"z500_clim_smooth": pack["clim_smooth"], "z500_anom": pack["anom"]})
                    ds_pack.attrs.update(
                        mjjas_pipeline_global_attrs(
                            years_label,
                            climatology_window_days=smooth_window,
                            preprocessing_step="02_z500_mjjas_climatology_anomaly_pack",
                            region=args.region,
                        )
                    )
                    if _bbox_attrs:
                        ds_pack.attrs.update(_bbox_attrs)
                    ds_pack.attrs["season"] = "MJJAS"
                    ds_pack.attrs.update(_step2_opt_attrs)
                    _atomic_to_netcdf(ds_pack, pack_path)
                    stamp(f"Wrote: {pack_path.name}")
                else:
                    stamp(f"Exists, skipping: {pack_path.name}")
            finally:
                ds_d.close()
            _step2_stamp_diagnostic_figures()
        else:
            stamp("Step 2.3: climatology + pack exist, skipping.")

    det_ok = det_path.exists() and _is_netcdf_readable(det_path)
    clim_raw_ok = clim_raw_path.exists() and _is_netcdf_readable(clim_raw_path)
    pack_ok = (
        pack_path.exists()
        and _is_netcdf_readable(pack_path)
        and _step2_pack_has_finite_climatology(pack_path)
    )
    if stop_after > 2 and step2_stage != "full":
        if not (det_ok and clim_raw_ok and pack_ok):
            raise FileNotFoundError(
                f"--step2-stage={step2_stage!r} with --stop-after-step={stop_after} requires completed Step 2 "
                "outputs (detrended + clim raw + anomaly pack). "
                f"Current: det_ok={det_ok}, clim_raw_ok={clim_raw_ok}, pack_ok={pack_ok}. "
                "Run in order: --step2-stage=mjjas, then --step2-stage=detrend, then --step2-stage=clim "
                "(or use --step2-stage=full)."
            )

    if int(args.stop_after_step) <= 2:
        stamp("Done (stopped after Step 2).")
        return

    # Step 3 compute/write (from Step 2 detrended + climatology)
    anom_ok = anom_path.exists() and _is_netcdf_readable(anom_path)
    if require_prepro_chain and not anom_ok:
        raise FileNotFoundError(
            "Steps 7–8 need existing Step 3 anomaly file; missing or unreadable:\n" f"  {anom_path}"
        )
    if args.overwrite or (not anom_ok):
        if anom_path.exists() and not anom_ok:
            stamp(f"Step 3: unreadable/corrupt NetCDF, will replace: {anom_path.name}")
        stamp("Step 3: computing anomalies ...")
        ds_det = xr.open_dataset(det_path, chunks=None)
        ds_pack = xr.open_dataset(pack_path, chunks=None)
        ds_clim_raw = xr.open_dataset(clim_raw_path, chunks=None)
        try:
            z_det = ds_det["z500_det"]
            z_clim_raw = ds_clim_raw["z500_clim_raw"]

            # Matches the now-retired `3_compute_anomalies.ipynb`: enforce calendar span on the detrended axis
            # (May 1 … Sep 30 each year) and validate min/max years vs filenames / raw span.
            year_start, year_end = parse_mjjas_year_bounds(years_label)
            dims_det = infer_core_dims(z_det, cfg=cfg)
            time_dim = dims_det["time"]
            sel_start = f"{year_start}-05-01"
            sel_end = f"{year_end}-09-30"
            z_det = z_det.sel({time_dim: slice(sel_start, sel_end)})

            years_in_data = np.unique(z_det[time_dim].dt.year.values)
            if years_in_data.size == 0:
                raise ValueError(
                    f"No data left after selecting {sel_start}..{sel_end} on detrended field. "
                    f"Check raw year span vs {det_path}"
                )
            if int(years_in_data.min()) != year_start or int(years_in_data.max()) != year_end:
                raise ValueError(
                    "Detrended input years do not match the span inferred from raw inputs / filenames. "
                    f"Expected {year_start}-{year_end}; got years {years_in_data.tolist()}; file={det_path.name}"
                )

            feb29 = (z_det[time_dim].dt.month == 2) & (z_det[time_dim].dt.day == 29)
            if bool(np.asarray(feb29.any().values).item()):
                stamp("Step 3: WARNING: detrended times include Feb 29 (unexpected for MJJAS-only data).")

            z_clim_smooth = ds_pack["z500_clim_smooth"] if "z500_clim_smooth" in ds_pack.data_vars else None
            z_anom = compute_step3_anomaly(
                z_det=z_det, z_clim_raw=z_clim_raw, z_clim_smooth=z_clim_smooth, cfg=cfg
            ).load()

            clim_w = int(ds_pack.attrs.get("climatology_window_days", smooth_window))
            z_anom.attrs["climatology_window_days"] = clim_w
            z_anom.attrs.setdefault(
                "definition",
                "z500_anom(time,lat,lon) = z500_det(time,lat,lon) - z500_clim_raw(season_day(time),lat,lon)",
            )

            if "z500_anom" in ds_pack.data_vars:
                z2 = ds_pack["z500_anom"].sel({time_dim: slice(sel_start, sel_end)}).transpose(*z_anom.dims)
                max_abs_diff = float(np.abs(z_anom - z2).max().compute())
                stamp(f"Step 3: max |anomaly - pack z500_anom| = {max_abs_diff:.6g} (sanity vs Step 2 pack)")
        finally:
            ds_det.close()
            ds_pack.close()
            ds_clim_raw.close()

        nan_frac = float(z_anom.isnull().mean().compute())
        if nan_frac > 0.0:
            raise ValueError(
                f"Step 3 produced NaNs in z500_anom: NaN fraction = {nan_frac:g}. "
                "This usually indicates missing raw days/timestamps or a mismatch between detrended time axis "
                "and season-day climatology alignment."
            )

        encoding = {"z500_anom": {"zlib": True, "complevel": 4}}
        ds_out = xr.Dataset({"z500_anom": z_anom})
        ds_out.attrs.update(
            mjjas_pipeline_global_attrs(
                years_label,
                climatology_window_days=smooth_window,
                preprocessing_step="03_z500_mjjas_anomaly",
                region=args.region,
            )
        )
        if bbox is not None:
            ds_out.attrs.update(
                {
                    "bbox_definition": "N,W,S,E",
                    "bbox_step": str(bbox_step),
                    "bbox_config": str(bbox_config_path),
                    "bbox_source": str(bbox_source),
                    "bbox_N": float(bbox["N"]),
                    "bbox_W": float(bbox["W"]),
                    "bbox_S": float(bbox["S"]),
                    "bbox_E": float(bbox["E"]),
                    "bbox_note": "Subset applied to raw Z500 at Step 2 before resample/detrend/climatology.",
                }
            )
        ds_out.attrs.update(
            {
                "title": "ERA5 Z500 MJJAS anomalies",
                "history": "Step 3: anomaly computation from detrended Z500 and smoothed MJJAS climatology",
                "source_detrended": str(det_path.resolve()),
                "source_climatology_pack": str(pack_path.resolve()),
                "detrend_method": str(args.detrend_method),
                "climatology_subtract": str(args.climatology_subtract),
                "sigma_mode": str(args.sigma_mode),
            }
        )
        ds_out.to_netcdf(anom_path, encoding=encoding)
        stamp(f"Wrote: {anom_path.name}")

        # Figures (Step 3)
        if save_figs:
            dims_an = infer_core_dims(z_anom, cfg=cfg)
            tdim = dims_an["time"]
            lat_dim = dims_an["lat"]
            lon_dim = dims_an["lon"]
            save_map_figure(
                da=z_anom.mean(dim=tdim),
                out_path=figures_dir
                / f"step03_anomaly_temporal_mean_w{smooth_window:02d}_{opt_label}_years_{years_label}.png",
                title="Step 3: temporal mean of anomalies (should be ~0)",
            )
            ts_an = z_anom.mean(dim=(lat_dim, lon_dim))
            save_timeseries_figure(
                da=ts_an,
                out_path=figures_dir
                / f"step03_anomaly_spatial_mean_timeseries_w{smooth_window:02d}_{opt_label}_years_{years_label}.png",
                title="Step 3: spatial mean of anomalies (should be flat around 0)",
            )
    else:
        stamp(f"Step 3: exists, skipping: {anom_path.name}")

    if int(args.stop_after_step) <= 3:
        stamp("Done (stopped after Step 3).")
        return

    # Steps 4–6 compute/write
    prepro_ok = prepro_path.exists() and _is_netcdf_readable(prepro_path)
    if require_prepro_chain and not prepro_ok:
        raise FileNotFoundError(
            "Steps 7–8 need existing Step 4–6 preprocessed file; missing or unreadable:\n"
            f"  {prepro_path}"
        )
    if args.overwrite or (not prepro_ok):
        if prepro_path.exists() and not prepro_ok:
            stamp(f"Steps 4–6: unreadable/corrupt NetCDF, will replace: {prepro_path.name}")
        stamp("Steps 4–6: computing σ, normalisation, weighting, flattening ...")
        ds_in = xr.open_dataset(anom_path, chunks=None)
        try:
            z_anom = ds_in["z500_anom"].load()
        finally:
            ds_in.close()

        ds_prepro = build_step4_6_dataset(
            z_anom=z_anom, window_days=smooth_window, epsilon_m=float(args.epsilon_m), cfg=cfg
        )
        ds_prepro.attrs.update(
            mjjas_pipeline_global_attrs(
                years_label,
                climatology_window_days=smooth_window,
                preprocessing_step="04_05_06_Z500_preprocessed",
                region=args.region,
            )
        )
        if bbox is not None:
            ds_prepro.attrs.update(
                {
                    "bbox_definition": "N,W,S,E",
                    "bbox_step": str(bbox_step),
                    "bbox_config": str(bbox_config_path),
                    "bbox_source": str(bbox_source),
                    "bbox_N": float(bbox["N"]),
                    "bbox_W": float(bbox["W"]),
                    "bbox_S": float(bbox["S"]),
                    "bbox_E": float(bbox["E"]),
                    "bbox_note": "Subset applied to raw Z500 at Step 2 before resample/detrend/climatology.",
                }
            )
        ds_prepro.attrs.update(
            {
                "detrend_method": str(args.detrend_method),
                "climatology_subtract": str(args.climatology_subtract),
                "sigma_mode": str(args.sigma_mode),
            }
        )
        ds_prepro.to_netcdf(prepro_path)
        stamp(f"Wrote: {prepro_path.name}")

        # Figures (Steps 4–6)
        if save_figs:
            sigma = ds_prepro["sigma_season_day"]
            mid_day = int(sigma["season_day"].values[len(sigma["season_day"]) // 2])
            save_map_figure(
                da=sigma.sel(season_day=mid_day),
                out_path=figures_dir
                / f"step04_sigma_midday_w{smooth_window:02d}_eps{eps_label:02d}_{opt_label}_years_{years_label}.png",
                title=f"Step 4: σ(season_day={mid_day}) smoothed, ε={float(args.epsilon_m):g} m",
            )

            z_norm = ds_prepro["z500_norm"]
            dims_norm = infer_core_dims(z_norm, cfg=cfg)
            tdim = dims_norm["time"]
            save_map_figure(
                da=z_norm.var(dim=tdim),
                out_path=figures_dir
                / f"step04_z500_norm_temporal_variance_w{smooth_window:02d}_{opt_label}_years_{years_label}.png",
                title="Step 4: temporal variance of normalized anomalies",
            )

            z_weighted = ds_prepro["z500_weighted"]
            save_map_figure(
                da=z_weighted.var(dim=tdim),
                out_path=figures_dir
                / f"step05_z500_weighted_temporal_variance_w{smooth_window:02d}_{opt_label}_years_{years_label}.png",
                title="Step 5: temporal variance after √cosφ weighting",
            )
    else:
        stamp(f"Steps 4–6: exists, skipping: {prepro_path.name}")

    # Optional: Steps 7–8 (PCA + hierarchical clustering). Kept out of Steps 2–6 by default.
    if stop_after >= 7:
        stamp("Step 7: PCA ...")
        if args.overwrite or (not pca_path.exists()) or (not _is_netcdf_readable(pca_path)):
            run_step7_pca(
                in_prepro_path=prepro_path,
                out_pca_path=pca_path,
                var_threshold=float(args.var_threshold),
                trace_attrs=mjjas_pipeline_global_attrs(
                    years_label,
                    climatology_window_days=smooth_window,
                    preprocessing_step="07_Z500_PCA",
                    region=args.region,
                ),
            )
            stamp(f"Wrote: {pca_path.name}")
        else:
            stamp(f"Exists, skipping: {pca_path.name}")

    if stop_after >= 8:
        stamp("Step 8: hierarchical clustering (Ward) ...")
        if args.overwrite or (not hc_path.exists()) or (not _is_netcdf_readable(hc_path)):
            run_step8_hierarchical_clustering(
                in_pca_path=pca_path,
                out_hc_path=hc_path,
                trace_attrs=mjjas_pipeline_global_attrs(
                    years_label,
                    climatology_window_days=smooth_window,
                    preprocessing_step="08_Z500_HC_Ward",
                    region=args.region,
                ),
            )
            stamp(f"Wrote: {hc_path.name}")
        else:
            stamp(f"Exists, skipping: {hc_path.name}")

    stamp("Done.")


if __name__ == "__main__":
    run_pipeline_steps_2_to_6()

