from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

try:  # optional on HPC
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    from common.colorbar_discretize import discretize_diverging, style_discrete_ticks
except Exception:  # pragma: no cover
    plt = None


def _pick_name(candidates: Iterable[str], available: set[str], kind: str) -> str:
    for c in candidates:
        if c in available:
            return c
    raise KeyError(f"Could not infer {kind}. Tried {tuple(candidates)}, available={sorted(list(available))}")


def infer_time_lat_lon(da: xr.DataArray) -> tuple[str, str, str]:
    dims = set(da.dims) | set(da.coords)
    time_dim = _pick_name(("time", "valid_time"), dims, "time dim")
    lat_dim = _pick_name(("lat", "latitude"), dims, "lat dim")
    lon_dim = _pick_name(("lon", "longitude"), dims, "lon dim")
    return time_dim, lat_dim, lon_dim


def _resolve_step9_regimes(results_clustering_dir: Path, *, year_span: str) -> Path:
    """Pick the Step 9 regimes file matching year_span best."""
    patt = f"09_Z500_regimes_*_{year_span}.nc"
    hits = sorted(results_clustering_dir.glob(patt))
    if hits:
        return hits[0]
    # fallback: legacy without year span suffix
    hits2 = sorted(results_clustering_dir.glob("09_Z500_regimes_*.nc"))
    if hits2:
        return hits2[-1]
    raise FileNotFoundError(f"No Step 9 regimes file found under {results_clustering_dir} (pattern {patt!r})")


def _resolve_step3_anom(preprocessed_dir: Path, *, year_span: str) -> Path:
    """Pick the Step 3 anomaly file matching year_span best."""
    candidates = [
        preprocessed_dir / f"03_Z500_anomaly_w21_{year_span}.nc",
        preprocessed_dir / f"03_Z500_anomaly_{year_span}.nc",
        preprocessed_dir / f"03_z500_anomaly_{year_span}.nc",
        preprocessed_dir / f"03_z500_anom_{year_span}.nc",
        preprocessed_dir / f"03_Z500_anom_{year_span}.nc",
        preprocessed_dir / f"03_z500_mjjas_anom_{year_span}.nc",
    ]
    for p in candidates:
        if p.is_file():
            return p
    hits = sorted(preprocessed_dir.glob(f"*Z500*anom*{year_span}*.nc")) + sorted(preprocessed_dir.glob(f"*z500*anom*{year_span}*.nc"))
    if hits:
        return hits[0]
    hits2 = sorted(preprocessed_dir.glob("*Z500*anom*.nc")) + sorted(preprocessed_dir.glob("*z500*anom*.nc"))
    if hits2:
        return hits2[0]
    raise FileNotFoundError(f"No Z500 anomaly file found under {preprocessed_dir} for year_span={year_span!r}")


def _looks_like_raw_time_series(p: Path, *, allow_detrended: bool) -> bool:
    name = p.name.lower()
    if (
        "anom" in name
        or "anomal" in name
        or "climat" in name
        or "preprocessed" in name
        or "norm" in name
        or "weighted" in name
        or "eps" in name
    ):
        return False
    if ("detrend" in name or "detrended" in name) and not allow_detrended:
        return False
    return True


def _pick_raw_var(ds: xr.Dataset) -> tuple[str, str, str, str]:
    """Return (var_name, time_dim, lat_dim, lon_dim) for a raw Z500-like DataArray."""
    preferred = ["z500", "z", "geopotential", "z500_raw", "z_raw"]
    time_candidates = ["time", "valid_time"]
    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lon", "longitude"]

    def score(var: str) -> tuple[int, int]:
        v = var.lower()
        return (1 if v in preferred else 0, 1 if "z500" in v else 0)

    for var in sorted(ds.data_vars, key=score, reverse=True):
        v = var.lower()
        if any(s in v for s in ["anom", "anomal", "norm", "weighted", "weight", "clim", "climat", "std"]):
            continue
        da = ds[var]
        dims = set(da.dims) | set(da.coords)
        if any(t in dims for t in time_candidates) and any(la in dims for la in lat_candidates) and any(
            lo in dims for lo in lon_candidates
        ):
            t = next(t for t in time_candidates if t in dims)
            la = next(la for la in lat_candidates if la in dims)
            lo = next(lo for lo in lon_candidates if lo in dims)
            return var, t, la, lo

    raise KeyError(f"No raw Z500-like variable with (time, lat, lon) found. data_vars={list(ds.data_vars)}")


def _resolve_raw_z500(preprocessed_dir: Path, *, year_span: str) -> Path:
    candidates = [
        preprocessed_dir / f"01_z500_mjjas_{year_span}.nc",
        preprocessed_dir / "01_z500_mjjas.nc",
        preprocessed_dir / f"01_Z500_{year_span}.nc",
        preprocessed_dir / "01_Z500.nc",
        preprocessed_dir / f"02_z500_mjjas_{year_span}.nc",
        preprocessed_dir / "02_z500_mjjas.nc",
    ]

    p_any = (
        sorted(preprocessed_dir.glob(f"*z500*mjjas*{year_span}*.nc"))
        + sorted(preprocessed_dir.glob(f"*Z500*{year_span}*.nc"))
        + sorted(preprocessed_dir.glob(f"*z500*{year_span}*.nc"))
    )

    def scan(*, allow_detrended: bool) -> Path | None:
        for p in (*candidates, *p_any):
            if not p.exists() or not _looks_like_raw_time_series(p, allow_detrended=allow_detrended):
                continue
            try:
                with xr.open_dataset(p, chunks=None) as ds_try:
                    _pick_raw_var(ds_try)
            except Exception:
                continue
            return p
        return None

    p = scan(allow_detrended=False)
    if p is None:
        p = scan(allow_detrended=True)
        if p is not None:
            print(f"[regime_field] WARNING: no truly raw Z500 found; falling back to detrended file: {p.name}")

    if p is None:
        raise FileNotFoundError(
            "Could not find a Z500 time series (with a time dimension) in "
            f"{preprocessed_dir}. If you only have climatology/anomaly packs, regenerate the raw Z500 file (Step 1)."
        )
    return p


def composite_by_regime(
    field: xr.DataArray,
    *,
    labels: xr.DataArray,
    regimes: np.ndarray,
    time_dim: str,
) -> xr.DataArray:
    out = []
    for r in regimes:
        mask = labels == int(r)
        comp = field.where(mask).mean(dim=time_dim, skipna=True)
        out.append(comp)
    comp_all = xr.concat(out, dim=xr.IndexVariable("regime", regimes))
    comp_all.name = f"regime_composite_{field.name}" if field.name else "regime_composite"
    return comp_all


def composite_by_regime_std(
    field: xr.DataArray,
    *,
    labels: xr.DataArray,
    regimes: np.ndarray,
    time_dim: str,
) -> xr.DataArray:
    """Per-regime standard deviation across days (variability of the composite, not its mean)."""
    out = []
    for r in regimes:
        mask = labels == int(r)
        comp = field.where(mask).std(dim=time_dim, skipna=True)
        out.append(comp)
    comp_all = xr.concat(out, dim=xr.IndexVariable("regime", regimes))
    comp_all.name = f"regime_composite_std_{field.name}" if field.name else "regime_composite_std"
    return comp_all


def _plot_quicklook(
    composite: xr.DataArray,
    *,
    out_png: Path,
    lat_dim: str,
    lon_dim: str,
    cmap: str = "RdBu_r",
    diverging: bool = True,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib not available in this environment (cannot plot).")

    regimes = composite["regime"].values.astype(int)
    n_reg = int(regimes.size)
    ncol = min(4, n_reg)
    nrow = int(np.ceil(n_reg / ncol))

    v = composite.values
    if diverging:
        # Zero-centered anomaly (e.g. Z500 anomaly, RdBu_r): symmetric limits around 0.
        vabs = float(np.nanpercentile(np.abs(v), 98))
        vmin, vmax = (-vabs, vabs)
    else:
        # Raw field with no natural zero (e.g. raw Z500 geopotential, viridis): data-driven range.
        vmin, vmax = (float(np.nanpercentile(v, 2)), float(np.nanpercentile(v, 98)))
    disc_cmap, norm, extend = discretize_diverging(vmin, vmax, cmap)

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.6 * nrow), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    im = None
    for i, r in enumerate(regimes):
        ax = axes[i]
        m = composite.sel(regime=int(r))
        im = ax.pcolormesh(m[lon_dim], m[lat_dim], m, shading="auto", cmap=disc_cmap, norm=norm)
        ax.set_title(f"Regime {int(r)}")
        ax.set_xlabel(lon_dim)
        ax.set_ylabel(lat_dim)

    for j in range(n_reg, axes.size):
        axes[j].axis("off")

    if im is not None:
        units = str(composite.attrs.get("units", "") or "")
        base_label = composite.name or "Z500 anom"
        cb = fig.colorbar(
            im,
            ax=axes[:n_reg].tolist(),
            fraction=0.046,
            pad=0.02,
            extend=extend,
            label=f"{base_label} ({units})" if units else base_label,
        )
        style_discrete_ticks(cb)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute Z500 composites per regime (mean over all assigned days).",
    )
    ap.add_argument("--region", default=os.environ.get("REGION", "europe"))
    ap.add_argument("--year-span", default=os.environ.get("YEAR_SPAN", "1979-2025"))
    ap.add_argument(
        "--mode",
        choices=("anom", "raw"),
        default=os.environ.get("MODE", "anom"),
        help="Which field to composite: 'anom' (Step 3 anomalies) or 'raw' (raw Z500 time series).",
    )
    ap.add_argument("--module-root", type=Path, default=None, help="Path to 1_Detect_Weather_Patterns/ (optional).")
    ap.add_argument("--in-regimes", type=Path, default=None, help="Step 9 regimes NetCDF (contains regime_labels).")
    ap.add_argument("--in-anom", type=Path, default=None, help="Z500 anomaly NetCDF (contains z500_anom).")
    ap.add_argument("--in-raw", type=Path, default=None, help="Raw Z500 NetCDF (contains a raw Z500-like variable).")
    ap.add_argument("--out", type=Path, required=True, help="Output NetCDF path.")
    ap.add_argument("--var-anom", default=None, help="Variable name in anomaly file (default: z500_anom or first var).")
    ap.add_argument("--var-raw", default=None, help="Variable name in raw file (default: auto-detect).")
    ap.add_argument("--chunks-time", type=int, default=128, help="Dask time chunk size (0 disables explicit chunking).")
    ap.add_argument("--plot-png", type=Path, default=None, help="Optional quicklook PNG output (no Cartopy).")
    args = ap.parse_args()

    module_root = (args.module_root or Path(__file__).resolve().parents[1]).resolve()
    results_clustering_dir = module_root / "results" / "clustering" / args.region
    preprocessed_dir = module_root / "data" / "preprocessed" / args.region

    p_reg = (args.in_regimes or _resolve_step9_regimes(results_clustering_dir, year_span=args.year_span)).resolve()
    out_path = args.out.resolve()

    print("[regime_field] module_root:", module_root)
    print("[regime_field] region:", args.region)
    print("[regime_field] year_span:", args.year_span)
    print("[regime_field] mode:", args.mode)
    print("[regime_field] in_regimes:", p_reg)
    print("[regime_field] out:", out_path)

    ds_reg = xr.open_dataset(p_reg, chunks=None)
    if "regime_labels" not in ds_reg:
        raise KeyError(f"Expected 'regime_labels' in {p_reg}, found: {list(ds_reg.data_vars)}")
    labels = ds_reg["regime_labels"]
    label_time_dim = labels.dims[0]
    time_coord = labels[label_time_dim] if label_time_dim in labels.coords else ds_reg[label_time_dim]

    regimes = np.unique(labels.values.astype(int))
    regimes = regimes[np.isfinite(regimes)]
    regimes = regimes[regimes > 0]
    regimes = np.asarray(regimes, dtype=int)
    print("[regime_field] regimes:", regimes.tolist())
    print("[regime_field] n_time:", int(labels.sizes[label_time_dim]))

    chunks = None
    if args.chunks_time and args.chunks_time > 0:
        chunks = {label_time_dim: int(args.chunks_time)}

    if args.mode == "anom":
        p_field = (args.in_anom or _resolve_step3_anom(preprocessed_dir, year_span=args.year_span)).resolve()
        print("[regime_field] in_anom:", p_field)
        ds_field = xr.open_dataset(p_field, chunks=chunks)
        var = args.var_anom
        if var is None:
            var = "z500_anom" if "z500_anom" in ds_field.data_vars else next(iter(ds_field.data_vars))
        z = ds_field[var]
        field_label = "anom"
    else:
        p_field = (args.in_raw or _resolve_raw_z500(preprocessed_dir, year_span=args.year_span)).resolve()
        print("[regime_field] in_raw:", p_field)
        ds_field = xr.open_dataset(p_field, chunks=chunks)
        if args.var_raw is not None:
            z = ds_field[args.var_raw]
        else:
            var_raw, _, _, _ = _pick_raw_var(ds_field)
            z = ds_field[var_raw]
        field_label = "raw"

    t_dim, lat_dim, lon_dim = infer_time_lat_lon(z)
    if t_dim != label_time_dim:
        z = z.rename({t_dim: label_time_dim})
    z = z.sel({label_time_dim: time_coord})

    # Regime counts (days) for metadata / quick checks
    n_days = xr.concat([(labels == int(r)).sum(dim=label_time_dim) for r in regimes], dim=xr.IndexVariable("regime", regimes))
    n_days.name = "n_days"

    comp = composite_by_regime(z, labels=labels, regimes=regimes, time_dim=label_time_dim)
    comp.name = f"regime_composite_z500_{field_label}"

    ds_out = xr.Dataset(
        {
            "regime_labels": labels.astype(np.int16),
            "n_days": n_days.astype(np.int32),
            comp.name: comp.astype(np.float32),
        }
    )
    ds_out.attrs.update(
        {
            "region": str(args.region),
            "year_span": str(args.year_span),
            "source_regimes": str(p_reg),
            "mode": str(args.mode),
            "source_field": str(p_field),
            "note": "Composite is the mean field over all days assigned to each regime.",
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    print("[regime_field] wrote:", out_path)

    if args.plot_png is not None:
        is_anom = args.mode == "anom"
        cmap = "RdBu_r" if is_anom else "viridis"
        _plot_quicklook(
            ds_out[comp.name], out_png=args.plot_png, lat_dim=lat_dim, lon_dim=lon_dim,
            cmap=cmap, diverging=is_anom,
        )
        print("[regime_field] wrote:", args.plot_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

