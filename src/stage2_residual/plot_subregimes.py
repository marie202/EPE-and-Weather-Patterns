#!/usr/bin/env python3
"""
EPE flag maps and Z500×RR cross-product composites for residual regimes.

Saved as rr_subregimes_{region}_{bbox}.pdf next to the main report PDF.

  Page 1 — EPE flag-frequency map per global RR sub-regime: for each residual
            regime, the fraction of that regime's days on which each grid cell
            was flagged as a convective extreme (E_conv), shaded in Blues.
            Mirrors the per-Z500-regime EPE maps.

  Pages 2..N — One page per Z500 parent regime. Rows = the RR sub-regimes that
            occur within that Z500 regime (≥ min_days). Columns = signed
            composite anomaly of each field + an EPE flag-frequency map for that
            (Z500, RR) day-group.

EPE data: data/processed/{EPE_REGION_LABEL}/extreme_precip_events/
convective_duration_*.nc (default label 'germany'; override with env
EPE_REGION_LABEL). The per-cell daily flag is n_conv_hours_grid > 0, joined to
regime labels by calendar date. When several threshold files exist, the one
matching $EPE_DURATION_PREFER (default 'q999_sampleAll_cape0_cpFracMin01_lexmax_domainQ900' --
q999, all_hours sample, strict CAPE>0, cp/tp fraction >= 1%, domain_pctl_q=0.90, domain-agg=lexmax,
i.e. the pinned convective_duration_TEST_* file from
detect_convective_duration.py) is used; override the exact
file with $EPE_FILE / --epe-file. EPE maps are skipped if no file is found.

Usage:
    python -m stage2_residual.plot_subregimes \
        --region europe --bbox-step B5 --module-root /path/to/detect_weather
"""
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from common.colorbar_discretize import discretize_diverging, discretize_sequential_from_floor, style_discrete_ticks
from .plot_results import (
    _dw_root,
    _find_anomaly,
    _find_raw_z500,
    _find_rr_regimes,
    _find_z500_regimes,
    _load_labels,
)
from . import rr_geo
from .feature_categories import field_label

MIN_DAYS = 10
_EPE_CMAP = "Blues"
_G = 9.80665  # m/s² — divides raw ERA5 geopotential (m²/s²) to give geopotential metres (gpm)
# Per-cell daily flag variable in convective_duration_*.nc (NaN on non-extreme cells,
# hour-count where the cell was a convective extreme). A cell is "flagged" when > 0.
_EPE_FLAG_VAR = "n_conv_hours_grid"
# Preferred duration-file token when several thresholds are present (overridable via env
# EPE_DURATION_PREFER). Default (2026-08-03): the full pinned production definition — q999,
# all_hours sample (NOT the legacy wet-hours quantile sample), strict CAPE>0, cp/tp fraction >= 1%,
# domain-aggregated at the 0.90 percentile (lexmax), i.e. a convective_duration_TEST_* file from
# epe_01b_detect_convective_hourly_duration_test.py whose name contains this exact contiguous
# substring (matches the q_tag/sample_tag/cape_tag/cp_frac_tag/domain_agg_tag/domain_q_tag
# ordering in that script's filename template). A shorter substring like "lexmax_domainQ900" alone
# is NOT enough to disambiguate: several TEST_* files can share those two tokens while differing in
# their underlying point-wise definition (e.g. legacy q95_sampleWetmm01 vs this q999_sampleAll) —
# see the analogous path_matches_extremes_spec fix in summary_reports/extremes_spec.py.
_EPE_PREFER_DEFAULT = "q999_sampleAll_cape0_cpFracMin01_lexmax_domainQ900"


# ---------------------------------------------------------------------------
# Data containers + loaders
# ---------------------------------------------------------------------------

@dataclass
class FieldData:
    data: np.ndarray          # (time, lat, lon)
    dates: pd.DatetimeIndex   # normalised to day
    lats: np.ndarray
    lons: np.ndarray
    units: str = ""           # native physical units, read from the netCDF variable attrs


def _norm_dates(times) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(times))).normalize()


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    lat = next((d for d in da.dims if "lat" in d.lower()), None)
    lon = next((d for d in da.dims if "lon" in d.lower()), None)
    if lat is None or lon is None:
        raise ValueError(f"Could not find lat/lon dims in {da.dims}")
    return lat, lon


def _load_field(path: Path) -> FieldData | None:
    try:
        ds = xr.open_dataset(path)
        var = next((v for v in ds.data_vars if "anom" in v.lower()), list(ds.data_vars)[0])
        da = ds[var].load()
        ds.close()
    except Exception as e:
        print(f"[rr_subregimes]   WARNING: could not load {path.name}: {e}")
        return None
    lat_dim, lon_dim = _spatial_dims(da)
    time_dim = next((d for d in da.dims if d not in (lat_dim, lon_dim)), da.dims[0])
    da = da.transpose(time_dim, lat_dim, lon_dim)
    return FieldData(
        data=da.values,
        dates=_norm_dates(da[time_dim].values),
        lats=da[lat_dim].values,
        lons=da[lon_dim].values,
        units=str(da.attrs.get("units", "")).strip(),
    )


def _load_raw_z500_field(path: Path) -> FieldData | None:
    """Load raw ERA5 Z500 geopotential and convert to geopotential metres (gpm).

    The raw ERA5 variable ``z`` is in m²/s²; dividing by g=9.80665 gives gpm.
    Returns a FieldData with data shaped (time, lat, lon) in gpm.
    """
    try:
        ds = xr.open_dataset(path)
        var = "z" if "z" in ds.data_vars else next(iter(ds.data_vars))
        da = ds[var].load()
        ds.close()
    except Exception as e:
        print(f"[rr] WARNING: could not load raw Z500 {path.name}: {e}")
        return None
    if "pressure_level" in da.dims:
        da = da.isel(pressure_level=0, drop=True)
    lat_dim, lon_dim = _spatial_dims(da)
    time_dim = next((d for d in da.dims if d not in (lat_dim, lon_dim)), da.dims[0])
    da = da.transpose(time_dim, lat_dim, lon_dim)
    return FieldData(
        data=da.values.astype(float) / _G,
        dates=_norm_dates(da[time_dim].values),
        lats=da[lat_dim].values,
        lons=da[lon_dim].values,
        units="gpm",
    )


def _crop_field_to_bbox(fd: FieldData | None, bbox: dict[str, float] | None) -> FieldData | None:
    """Restrict a FieldData's grid (and data) to a bbox's N/S/W/E window.

    Raw Z500 is loaded once per REGION (not per bbox step) from a domain-wide file
    spanning the whole continent, so its value range is dominated by latitudes/areas
    far outside whatever small bbox is actually being plotted. Left uncropped, contour
    levels derived from ``_z500_raw_contour_levels`` (min/max over the WHOLE domain)
    end up spaced across a range the visible bbox window barely touches — producing
    just one or two isolines crammed in a corner, or none at all, instead of a well-
    spaced set of labelled isolines across the panel.
    """
    if fd is None or bbox is None:
        return fd
    lat_mask = (fd.lats >= bbox["S"]) & (fd.lats <= bbox["N"])
    lon_mask = (fd.lons >= bbox["W"]) & (fd.lons <= bbox["E"])
    if not lat_mask.any() or not lon_mask.any():
        return fd
    return FieldData(
        data=fd.data[:, lat_mask, :][:, :, lon_mask],
        dates=fd.dates,
        lats=fd.lats[lat_mask],
        lons=fd.lons[lon_mask],
        units=fd.units,
    )


_CONTOUR_STEP_CANDIDATES_GPM = (40.0, 20.0, 10.0, 5.0, 2.0, 1.0)


def _z500_raw_contour_levels(mean_gpm: np.ndarray, step: float = 40.0, min_levels: int = 3) -> np.ndarray:
    """Evenly-spaced contour levels for a raw Z500 field in gpm (typically 5000-5900).

    Starts at ``step`` gpm spacing and falls back to finer steps (20/10/5/2/1) if the
    field's own value range -- after ``_crop_field_to_bbox`` -- is too narrow for a fixed
    40 gpm step to produce at least ``min_levels`` isolines (a small bbox crop of a tightly
    clustered sub-regime/bin composite can have a range under 40 gpm on its own, collapsing
    a 40-step grid to a single line even though the field itself has real structure).
    """
    finite = mean_gpm[np.isfinite(mean_gpm)]
    if finite.size == 0:
        return np.arange(5100.0, 5900.0, step)
    data_min, data_max = float(finite.min()), float(finite.max())
    candidates = [step] + [s for s in _CONTOUR_STEP_CANDIDATES_GPM if s < step]
    chosen = candidates[-1]
    for cand in candidates:
        # Count multiples of cand that actually fall inside [data_min, data_max] --
        # a contour level outside the field's real range never gets drawn, so padding
        # the array with floor/ceil-derived levels beyond that range doesn't help.
        n_inside = int(np.floor(data_max / cand) - np.ceil(data_min / cand)) + 1
        chosen = cand
        if n_inside >= min_levels:
            break
    lo = np.floor(data_min / chosen) * chosen
    hi = np.ceil(data_max / chosen) * chosen + chosen
    return np.arange(lo, hi, chosen)


def _load_econv(path: Path) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Load the per-cell daily convective-flag mask from a duration file.

    Reads ``n_conv_hours_grid`` (NaN on non-extreme cells, hour-count otherwise)
    and returns a 0/1 float flag (cell flagged when > 0).
    """
    ds = xr.open_dataset(path)
    var = _EPE_FLAG_VAR if _EPE_FLAG_VAR in ds else None
    if var is None:  # fall back to a legacy E_conv mask if present
        var = "E_conv" if "E_conv" in ds else None
    if var is None:
        ds.close()
        raise KeyError(
            f"Neither '{_EPE_FLAG_VAR}' nor 'E_conv' in {path.name} (vars={list(ds.data_vars)})"
        )
    da = ds[var].load()
    ds.close()
    lat_dim, lon_dim = _spatial_dims(da)
    time_dim = next((d for d in da.dims if d not in (lat_dim, lon_dim)), da.dims[0])
    da = da.transpose(time_dim, lat_dim, lon_dim)
    flag = (np.nan_to_num(da.values, nan=0.0) > 0).astype(np.float32)
    return (flag, _norm_dates(da[time_dim].values), da[lat_dim].values, da[lon_dim].values)


def _prefer_match(matches: list[Path], prefer: str) -> Path:
    """From a list of duration files, return the one whose name contains ``prefer``
    (most specific). If nothing matches, raise instead of silently substituting some
    other threshold combination (e.g. a legacy wet-hours file when the pinned
    q999/all_hours/domain-lexmax/domainQ900 definition was wanted) — every EPE
    frequency map in this pipeline must be built from the domain q0.90 filter unless
    the caller has explicitly opted out. Set EPE_DURATION_PREFER to override; empty
    string disables the check (explicitly "any file is fine") and falls back to
    newest-sorted, with a warning."""
    if not matches:
        raise ValueError("empty matches")
    preferred = [m for m in matches if prefer in m.name]
    if preferred:
        return preferred[-1]
    if prefer:
        raise ValueError(
            f"[rr_subregimes] no convective_duration file matches EPE_DURATION_PREFER={prefer!r} "
            f"among {len(matches)} candidate(s) in {matches[0].parent}. Refusing to silently fall "
            f"back to newest-sorted ({matches[-1].name}), which is very likely NOT the intended EPE "
            f"definition (e.g. a legacy wet-hours file instead of the pinned domain q0.90 one). "
            f"Set EPE_FILE to the exact path, or set EPE_DURATION_PREFER='' to explicitly allow any "
            f"file."
        )
    print(f"[rr_subregimes] WARNING: EPE_DURATION_PREFER explicitly disabled — using newest-sorted "
          f"convective_duration file: {matches[-1].name}. This may not be the domain q0.90 EPE filter.")
    return matches[-1]


def resolve_epe_label(epe_region_label: str | None = None) -> str:
    return epe_region_label or os.environ.get("EPE_REGION_LABEL", "germany")


def epe_basis_label(path: "Path | None") -> str:
    """Human-readable extreme-day definition parsed from a duration file's OWN filename tokens
    (e.g. "...q999_..._domainQ900..." -> "domain q0.90, q99.9%"), so any panel title built from
    this always reflects whichever file ``_find_extremes_file`` actually resolved -- the pinned
    ``_EPE_PREFER_DEFAULT`` or an ``$EPE_FILE``/``$EPE_DURATION_PREFER`` override -- instead of a
    hardcoded assumption baked into the title text.
    """
    if path is None:
        return ""
    name = Path(path).name
    bits = []
    m = re.search(r"domainQ(\d{3})", name)
    if m:
        bits.append(f"domain q{int(m.group(1)) / 1000:.2f}")
    m = re.search(r"(?<![A-Za-z])q(\d{2,3})_", name)
    if m:
        q_raw = m.group(1)
        q = float(q_raw) / 10.0 if len(q_raw) == 3 else float(q_raw)
        bits.append(f"q{q:g}%")
    return ", ".join(bits)


def load_epe_day_stats(path: Path) -> pd.DataFrame:
    """Per-day extreme-event diagnostics from a convective_duration_*.nc file.

    Columns (when present): 'extreme_day' (1 if any cell flagged), 'n_cells'
    (number of flagged grid cells), 'conv_hours_anywhere' (sum of convective
    hours over the domain). Indexed by normalised date.
    """
    ds = xr.open_dataset(path)
    idx = _norm_dates(ds["time"].values)
    cols: dict[str, np.ndarray] = {}
    if "flagged_any_gridcell" in ds:
        cols["extreme_day"] = np.asarray(ds["flagged_any_gridcell"].values, dtype=float)
    if "n_conv_gridcells_any_hour_day" in ds:
        cols["n_cells"] = np.asarray(ds["n_conv_gridcells_any_hour_day"].values, dtype=float)
    if "n_conv_hours_anywhere" in ds:
        cols["conv_hours_anywhere"] = np.asarray(ds["n_conv_hours_anywhere"].values, dtype=float)
    ds.close()
    return pd.DataFrame(cols, index=idx)


def _find_extremes_file(dw: Path, region_label: str, module_root: Path | None = None) -> Path | None:
    """Locate a convective_duration_*.nc (per-cell flag) file.

    Search order:
      1. $EPE_FILE if set (explicit path)
      2. data/processed/{region_label}/extreme_precip_events/convective_duration_*.nc
      3. data/processed/*/extreme_precip_events/convective_duration_*.nc (any label)
      4. recursive search for convective_duration_*.nc under data/ (last resort)
    Within a directory the file whose name contains $EPE_DURATION_PREFER
    (default _EPE_PREFER_DEFAULT above) is chosen, loudly falling back to the
    newest-sorted file (see `_prefer_match`) if nothing matches. Candidate roots:
    dw, module_root, dw.parent.
    """
    env_file = os.environ.get("EPE_FILE")
    if env_file:
        p = Path(env_file).expanduser()
        if p.is_file():
            return p
        print(f"[rr_subregimes] EPE_FILE set but not found: {p}")

    prefer = os.environ.get("EPE_DURATION_PREFER", _EPE_PREFER_DEFAULT)
    pattern = "convective_duration_*.nc"

    roots: list[Path] = []
    for r in (dw, module_root, dw.parent if dw else None):
        if r is not None and Path(r) not in roots:
            roots.append(Path(r))

    # 2. labelled dir
    for root in roots:
        d = root / "data" / "processed" / region_label / "extreme_precip_events"
        if d.is_dir():
            matches = sorted(d.glob(pattern))
            if matches:
                return _prefer_match(matches, prefer)

    # 3. any label
    for root in roots:
        base = root / "data" / "processed"
        if base.is_dir():
            matches = sorted(base.glob(f"*/extreme_precip_events/{pattern}"))
            if matches:
                chosen = _prefer_match(matches, prefer)
                print(f"[rr_subregimes] EPE: '{region_label}' dir absent; using {chosen.parent.parent.name}")
                return chosen

    # 4. recursive last resort
    for root in roots:
        base = root / "data"
        if base.is_dir():
            matches = sorted(base.rglob(pattern))
            if matches:
                chosen = _prefer_match(matches, prefer)
                print(f"[rr_subregimes] EPE: found via recursive search: {chosen}")
                return chosen
    return None


# Public alias for reuse by plot_results (EPE statistics page).
find_duration_file = _find_extremes_file


# ---------------------------------------------------------------------------
# Composite / EPE computations over an arbitrary day-mask
# ---------------------------------------------------------------------------

def _composite_over_dates(fd: FieldData, group_dates: pd.DatetimeIndex) -> np.ndarray | None:
    sel = fd.dates.isin(group_dates)
    if sel.sum() == 0:
        return None
    return np.nanmean(fd.data[sel], axis=0)


def _epe_frequency_over_dates(
    econv: np.ndarray, edates: pd.DatetimeIndex, group_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray | None, int]:
    """Fraction of group days on which each cell was flagged. Returns (freq, n_days)."""
    sel = edates.isin(group_dates)
    n = int(sel.sum())
    if n == 0:
        return None, 0
    return np.nanmean(econv[sel], axis=0), n


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_map(ax, is_geo, lons, lats, data, *, cmap, norm=None, vmin=None, vmax=None):
    tr = rr_geo.data_transform() if is_geo else None
    kw = dict(cmap=cmap, shading="auto")
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"], kw["vmax"] = vmin, vmax
    if tr is not None:
        kw["transform"] = tr
    return ax.pcolormesh(lons, lats, data, **kw)


def _overlay_z500_isolines(ax, is_geo, lons, lats, field, *, levels=7, label_every: int = 2) -> None:
    """Overlay labelled Z500 (mean) contour lines on an existing map panel.

    label_every: label every Nth contour (1 = all, 2 = every second, etc.).
    Default 2 avoids clutter when ~20 levels at 40 gpm spacing are drawn.
    """
    if field is None:
        return
    tr = rr_geo.data_transform() if is_geo else None
    ckw = dict(colors="k", linewidths=0.6, levels=levels, zorder=15)
    if tr is not None:
        ckw["transform"] = tr
    try:
        cs = ax.contour(lons, lats, field, **ckw)
        label_lvls = cs.levels[::label_every] if label_every > 1 else cs.levels
        ax.clabel(cs, levels=label_lvls, inline=True, fontsize=8.5, fmt="%.0f")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Page 1: EPE flag-frequency per global RR sub-regime
# ---------------------------------------------------------------------------

def _page_epe_by_regime(
    pdf, rr_labels, rr_dates, n_rr, econv, edates, elats, elons,
    region, bbox, bbox_dict, module_root, epe_basis: str = "",
) -> None:
    extent = rr_geo.map_extent(None, elats, elons)
    fig = plt.figure(figsize=(max(3.4 * n_rr, 10), 4.4))
    basis_suffix = f" (basis: {epe_basis})" if epe_basis else ""
    fig.suptitle(
        f"Convective-extreme (EPE) flag frequency by residual regime — {region}  {bbox}\n"
        f"Fraction of each regime's days on which the cell was E_conv-flagged{basis_suffix}",
        fontsize=10, fontweight="bold",
    )
    # Shared colour scale across regimes for comparability
    freqs = []
    for r in range(1, n_rr + 1):
        gdates = rr_dates[rr_labels == r]
        f, n = _epe_frequency_over_dates(econv, edates, gdates)
        freqs.append((f, n))
    vmax = max((np.nanpercentile(f, 99) for f, _ in freqs if f is not None), default=1.0)
    vmax = max(float(vmax), 1e-3)

    for col in range(n_rr):
        f, n = freqs[col]
        ax, is_geo = rr_geo.make_map_axes(fig, (1, n_rr, col + 1), extent=extent)
        if f is not None:
            epe_cmap, epe_norm, epe_extend = discretize_sequential_from_floor(100 * vmax, _EPE_CMAP, vmin=0)
            im = _draw_map(ax, is_geo, elons, elats, 100 * f, cmap=epe_cmap, norm=epe_norm)
            style_discrete_ticks(
                plt.colorbar(im, ax=ax, shrink=0.72, pad=0.03, aspect=18, label="% of days", extend=epe_extend)
            )
        rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
        ax.set_title(f"SR{col+1}\n{n} days", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages 2..N: Z500 × RR cross-product composites + EPE
# ---------------------------------------------------------------------------

def _page_z500_subregimes(
    pdf, z500_val, z_all_dates, rr_present, group_dates_by_rr, fields_data,
    econv, edates, elats, elons, region, bbox, bbox_dict, module_root,
    z500_raw_fd: "FieldData | None" = None,
) -> None:
    """One page per Z500 parent regime.

    Columns: [Z500 anomaly mean (all regime days) | Z500 anomaly mean (sub-regime days) |
    <other field composites...> | EPE flag freq]. Black isolines show the mean
    **raw** Z500 geopotential (gpm) for the days of each (Z500, RR) sub-regime —
    one set of isolines per row, not shared across rows.
    Rows = RR sub-regimes within this Z500 regime.
    """
    have_epe = econv is not None
    other_fields = [f for f in fields_data if f != "z500"]
    has_z500 = "z500" in fields_data
    n_z500_cols = 2 if has_z500 else 0  # parent + sub-regime
    ncol = n_z500_cols + len(other_fields) + (1 if have_epe else 0)
    nrow = len(rr_present)
    if nrow == 0 or ncol == 0:
        return

    fig = plt.figure(figsize=(3.2 * ncol, 3.2 * nrow + 0.6))
    fig.suptitle(
        f"Z500-{z500_val}: sub-regime composites + EPE — {region}  {bbox}\n"
        "Black isolines = mean raw Z500 (gpm) of each Z500×RR sub-regime",
        fontsize=11, fontweight="bold", y=1.005,
    )

    # Parent Z500 anomaly mean (all days of this Z500 regime) — for the coloured panels.
    parent_z500 = None
    z_lats = z_lons = None
    if has_z500:
        fdz = fields_data["z500"]
        parent_z500 = _composite_over_dates(fdz, z_all_dates)
        z_lats, z_lons = fdz.lats, fdz.lons

    # Sub-regime Z500 anomaly composites + shared symmetric scale with the parent.
    z500_sub: dict[int, np.ndarray] = {}
    z_norm = z_cmap = z_extend = None
    if has_z500:
        for rr in rr_present:
            c = _composite_over_dates(fields_data["z500"], group_dates_by_rr[rr])
            if c is not None:
                z500_sub[rr] = c
        z_stack = [a for a in ([parent_z500] + list(z500_sub.values())) if a is not None]
        if z_stack:
            zv = max(float(np.nanpercentile(np.abs(np.stack(z_stack)), 98)), 1e-6)
            z_cmap, z_norm, z_extend = discretize_diverging(-zv, zv, "RdBu_r")

    # Raw Z500 mean per sub-regime for isolines (one set per row) + parent mean for col 0.
    z500_raw_sub: dict[int, np.ndarray] = {}
    z500_raw_parent: np.ndarray | None = None
    z_iso_parent: np.ndarray | int = 7
    z_lats_raw = z_lons_raw = None
    if z500_raw_fd is not None:
        z_lats_raw, z_lons_raw = z500_raw_fd.lats, z500_raw_fd.lons
        z500_raw_parent = _composite_over_dates(z500_raw_fd, z_all_dates)
        if z500_raw_parent is not None:
            z_iso_parent = _z500_raw_contour_levels(z500_raw_parent)
        for rr in rr_present:
            c = _composite_over_dates(z500_raw_fd, group_dates_by_rr[rr])
            if c is not None:
                z500_raw_sub[rr] = c

    # Other-field composites + per-field symmetric scale shared across rows.
    field_cmap: dict[str, object] = {}
    field_norm: dict[str, object] = {}
    field_extend: dict[str, str] = {}
    field_grid: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    field_comps: dict[str, dict[int, np.ndarray]] = {}
    for field in other_fields:
        fd = fields_data[field]
        comps = {rr: c for rr in rr_present if (c := _composite_over_dates(fd, group_dates_by_rr[rr])) is not None}
        field_comps[field] = comps
        field_grid[field] = (fd.lats, fd.lons)
        if comps:
            vmax = max(float(np.nanpercentile(np.abs(np.stack(list(comps.values()))), 95)), 1e-6)
            field_cmap[field], field_norm[field], field_extend[field] = discretize_diverging(-vmax, vmax, "RdBu_r")

    # EPE shared scale across rows.
    if have_epe:
        epe_freqs = {rr: _epe_frequency_over_dates(econv, edates, group_dates_by_rr[rr]) for rr in rr_present}
        epe_vmax = max((np.nanpercentile(f, 99) for f, _ in epe_freqs.values() if f is not None), default=1.0)
        epe_vmax = max(float(epe_vmax), 1e-3)
        epe_extent = rr_geo.map_extent(None, elats, elons)

    z_extent = rr_geo.map_extent(bbox_dict, z_lats, z_lons) if has_z500 else None

    for row, rr in enumerate(rr_present):
        n_days = len(group_dates_by_rr[rr])
        col = 0

        # Raw Z500 isolines for this row: mean over sub-regime days, in gpm.
        z_raw_rr = z500_raw_sub.get(rr)
        z_iso_levels = _z500_raw_contour_levels(z_raw_rr) if z_raw_rr is not None else 7

        if has_z500:
            # Col 0: parent Z500 anomaly mean (all regime days) + raw Z500 parent isolines.
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=z_extent)
            if parent_z500 is not None and z_norm is not None:
                im = _draw_map(ax, is_geo, z_lons, z_lats, parent_z500, cmap=z_cmap, norm=z_norm)
                style_discrete_ticks(plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=z_extend))
            if z500_raw_parent is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z500_raw_parent, levels=z_iso_parent)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title("Z500 (regime mean,\nall days)", fontsize=8)
            ax.set_ylabel(f"SR{rr}\n{n_days} d", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1

            # Col 1: Z500 anomaly mean for sub-regime days + raw Z500 sub-regime isolines.
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=z_extent)
            zc = z500_sub.get(rr)
            if zc is not None and z_norm is not None:
                im = _draw_map(ax, is_geo, z_lons, z_lats, zc, cmap=z_cmap, norm=z_norm)
                style_discrete_ticks(plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=z_extend))
            if z_raw_rr is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
            elif z500_raw_parent is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z500_raw_parent, levels=z_iso_parent)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title("Z500 (sub-regime\ndays)", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1

        # Other field composites, with per-sub-regime raw Z500 isolines overlaid.
        for field in other_fields:
            lats, lons = field_grid[field]
            extent = rr_geo.map_extent(bbox_dict, lats, lons)
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=extent)
            comp = field_comps[field].get(rr)
            if comp is not None and field in field_norm:
                im = _draw_map(ax, is_geo, lons, lats, comp, cmap=field_cmap[field], norm=field_norm[field])
                style_discrete_ticks(
                    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, extend=field_extend[field])
                )
            if z_raw_rr is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
            elif has_z500:
                _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, parent_z500)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title(field_label(field), fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            col += 1

        # EPE column, with per-sub-regime raw Z500 isolines overlaid.
        if have_epe:
            ax, is_geo = rr_geo.make_map_axes(fig, (nrow, ncol, row * ncol + col + 1), extent=epe_extent)
            f, n = epe_freqs[rr]
            if f is not None:
                epe_cmap, epe_norm, epe_extend = discretize_sequential_from_floor(100 * epe_vmax, _EPE_CMAP, vmin=0)
                im = _draw_map(ax, is_geo, elons, elats, 100 * f, cmap=epe_cmap, norm=epe_norm)
                style_discrete_ticks(
                    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.03, aspect=16, label="% days", extend=epe_extend)
                )
            if z_raw_rr is not None:
                _overlay_z500_isolines(ax, is_geo, z_lons_raw, z_lats_raw, z_raw_rr, levels=z_iso_levels)
            elif has_z500:
                _overlay_z500_isolines(ax, is_geo, z_lons, z_lats, parent_z500)
            rr_geo.add_boundaries(ax, is_geo=is_geo, module_root=module_root, region=region, bbox=bbox_dict)
            if row == 0:
                ax.set_title("EPE flag freq", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_subregime_plots(module_root, region, bbox_step, smooth_window=11,
                        epe_region_label=None, out_dir=None, mode=None,
                        z500_cluster_method="ward", area_weight_tag: str = "") -> Path:
    dw = _dw_root(module_root)
    epe_region_label = epe_region_label or os.environ.get("EPE_REGION_LABEL", "germany")

    rr_path = _find_rr_regimes(dw, region, bbox_step, module_root=module_root, mode=mode,
                                z500_cluster_method=z500_cluster_method)
    if rr_path is None:
        raise FileNotFoundError(
            f"No RR regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})"
        )
    z500_path = _find_z500_regimes(dw, region, bbox_step,
                                    cluster_method=z500_cluster_method or "kmeans")
    if z500_path is None:
        raise FileNotFoundError(
            f"No Z500 regime file for {region} {bbox_step} (z500_cluster_method={z500_cluster_method})"
        )
    print(f"[rr_subregimes] RR regimes:   {rr_path.name}")
    print(f"[rr_subregimes] Z500 regimes: {z500_path.name}")

    rr_labels, rr_times = _load_labels(rr_path)
    rr_labels = rr_labels.astype(int)
    rr_dates = _norm_dates(rr_times)
    n_rr = int(rr_labels.max())

    z500_labels_raw, z500_times = _load_labels(z500_path)
    z500_series = pd.Series(z500_labels_raw.astype(int), index=_norm_dates(z500_times))
    z500_on_rr = z500_series.reindex(rr_dates).values  # aligned to rr timeline
    z500_vals = sorted({int(v) for v in z500_on_rr if np.isfinite(v) and v > 0})
    print(f"[rr_subregimes] {n_rr} RR regimes, Z500 regimes: {z500_vals}, {len(rr_dates)} days")

    # Fields
    m = re.search(r"z500([^_]*)_w\d+", rr_path.name)
    fields = ["z500"] + ([f for f in m.group(1).split("+") if f] if (m and m.group(1)) else [])
    fields_data: dict[str, FieldData] = {}
    for field in fields:
        p = _find_anomaly(dw, region, field, bbox_step, smooth_window, area_weight_tag=area_weight_tag)
        fd = _load_field(p) if p else None
        if fd is not None:
            fields_data[field] = fd
            print(f"[rr_subregimes]   ✓ {field}: {p.name}")
        else:
            print(f"[rr_subregimes]   ✗ {field}: anomaly not found — skipping")
    if not fields_data:
        raise RuntimeError("No field anomalies loaded.")

    # Raw Z500 geopotential (optional — isolines fall back to anomaly contours if missing).
    raw_z500_path = _find_raw_z500(dw, region)
    z500_raw_fd = None
    if raw_z500_path is not None:
        print(f"[rr_subregimes] Raw Z500: {raw_z500_path.name}")
        z500_raw_fd = _load_raw_z500_field(raw_z500_path)
    else:
        print("[rr_subregimes] WARNING: raw Z500 not found; isolines will use anomaly field")

    # EPE data (optional — composites still render without it)
    ext_path = _find_extremes_file(dw, epe_region_label, module_root)
    econv = edates = elats = elons = None
    if ext_path is None:
        print(f"[rr_subregimes] WARNING: no extremes_*.nc found (label={epe_region_label}). "
              f"EPE maps will be skipped; set EPE_FILE or EPE_REGION_LABEL to enable them.")
    else:
        print(f"[rr_subregimes] EPE file: {ext_path}")
        econv, edates, elats, elons = _load_econv(ext_path)

    bbox_dict = rr_geo.load_bbox_for_step(module_root, region, bbox_step)
    # Raw Z500 was loaded from a domain-wide file (one per region, not per bbox step) — crop it to
    # this bbox's window now, otherwise its contour levels get spaced across the WHOLE domain's
    # value range and only 0-1 of them ever fall inside the visible bbox.
    z500_raw_fd = _crop_field_to_bbox(z500_raw_fd, bbox_dict)

    out_dir = out_dir or (dw / "results" / "stage2_residual" / "figures" / region)
    out_dir.mkdir(parents=True, exist_ok=True)
    ward_tag = "_zward" if z500_cluster_method == "ward" else ""
    out_pdf = out_dir / f"rr_subregimes_{region}_{bbox_step}{ward_tag}.pdf"

    have_epe = econv is not None
    with PdfPages(out_pdf) as pdf:
        # Page 1: global EPE-by-RR (only if EPE data available)
        if have_epe:
            _page_epe_by_regime(pdf, rr_labels, rr_dates, n_rr, econv, edates, elats, elons,
                                region, bbox_step, bbox_dict, module_root,
                                epe_basis=epe_basis_label(ext_path))

        # Pages: Z500 × RR cross-product
        for z in z500_vals:
            z_mask = (z500_on_rr == z)
            z_all_dates = rr_dates[z_mask]  # all days assigned to this Z500 regime
            rr_present = []
            group_dates_by_rr: dict[int, pd.DatetimeIndex] = {}
            for rr in range(1, n_rr + 1):
                g = z_mask & (rr_labels == rr)
                if int(g.sum()) >= MIN_DAYS:
                    rr_present.append(rr)
                    group_dates_by_rr[rr] = rr_dates[g]
            if not rr_present:
                print(f"[rr_subregimes] Z500-{z}: no RR sub-regime with >= {MIN_DAYS} days — skipping page")
                continue
            print(f"[rr_subregimes] Z500-{z}: sub-regimes {rr_present}")
            _page_z500_subregimes(pdf, z, z_all_dates, rr_present, group_dates_by_rr, fields_data,
                                  econv, edates, elats, elons,
                                  region, bbox_step, bbox_dict, module_root,
                                  z500_raw_fd=z500_raw_fd)

    print(f"[rr_subregimes] Saved: {out_pdf}")
    return out_pdf


def main(argv=None) -> int:
    import sys
    p = argparse.ArgumentParser(description="EPE flag maps + Z500×RR sub-regime composites.")
    p.add_argument("--region", required=True, choices=("europe", "northamerica"))
    p.add_argument("--bbox-step", required=True)
    p.add_argument("--module-root", type=Path, default=None)
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--epe-region-label", default=None,
                   help="Study-area label for extremes data (default: germany or $EPE_REGION_LABEL)")
    p.add_argument("--epe-file", default=None,
                   help="Explicit convective_duration_*.nc path (overrides discovery; also $EPE_FILE)")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--mode", choices=("bbox", "regime"), default=None,
                   help="Force which pipeline's RR regimes file to plot (rr_02 bbox-mode vs "
                        "rr_022 per-regime-mode). Default: auto-detect (prefers bbox-mode).")
    p.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="ward",
                   help="Restrict to RR/Z500 regime files built on this Z500 parent-regime "
                        "source ('ward' reads '_zward'-tagged files). Default: ward (pinned "
                        "production, matches plot_composites.py/plot_rgb_overview.py).")
    p.add_argument("--area-weight-tag", default="",
                   help="Must match the rr_00 run's --area-weight-tag (e.g. 'aw2') to resolve "
                        "that rerun's per-field anomaly files exactly, not a fuzzy glob (see "
                        "plot_results._find_anomaly) — has no effect on Z500, which lives "
                        "outside the rr_00 area-weighting rerun.")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.epe_file:
        os.environ["EPE_FILE"] = str(args.epe_file)

    if args.module_root is None:
        args.module_root = Path(__file__).resolve().parents[2]

    run_subregime_plots(
        module_root=args.module_root.expanduser().resolve(),
        region=args.region, bbox_step=args.bbox_step,
        smooth_window=args.smooth_window,
        epe_region_label=args.epe_region_label, out_dir=args.out_dir,
        mode=args.mode, z500_cluster_method=args.z500_cluster_method,
        area_weight_tag=args.area_weight_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
