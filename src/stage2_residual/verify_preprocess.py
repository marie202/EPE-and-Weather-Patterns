"""Verify rr_00 multivariate preprocessing outputs (anomaly/preprocessed/PCA files).

Checks, per (region, bbox_step, field):
  - the 03_anomaly / 04_05_06_preprocessed / 07_MF_PCA files exist
  - all three have an mtime at or after --since (i.e. were actually regenerated)
  - the preprocessed file's X_matrix is finite (sampled, not loaded in full)
  - the sqrt(cos(lat)) area weighting was actually applied (weighted_field /
    field_norm tracks sqrt(cos(lat)) rather than sitting flat near 1)
  - the PCA file's cumulative_explained_variance reaches var_threshold and
    EOF_maps/PCs_reduced contain no NaNs

Never materializes a full multi-decade (time, lat, lon) array in memory --
everything is either dask-lazy or reduced to a small time-sampled slice before
`.values`, so this is safe to run with modest --mem on a shared node. Even so,
run it via sbatch/srun, not directly on a login node.

Usage:
    python3 src/verify_rr00_preprocess.py \\
        --fields cape,deg0l,tcwv,qhum500,qhum700,qhum850,rh500,rh600,rh700,rh850,sst,td2m \\
        --regions europe,northamerica --bbox-steps B0,B1,B2,B3,B4,B5 \\
        --since 2026-07-28T00:00:00
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage2_residual.fields.config import MultivariateRunConfig
from stage2_residual.fields.paths import WorkflowPaths

# How many time steps to sample per file when checking finiteness/weighting.
# Kept small deliberately -- see module docstring.
N_TIME_SAMPLE = 30


def _find_lat_dim(da: xr.DataArray, exclude: tuple[str, ...]) -> str:
    for d in da.dims:
        if d in exclude or d not in da.coords:
            continue
        vals = np.asarray(da[d].values, dtype="float32")
        if vals.size >= 2 and np.nanmax(np.abs(vals)) <= 90.0 + 1e-6:
            return d
    raise KeyError(f"Could not find a latitude-like dim among {da.dims}")


def check_area_weighting(ds: xr.Dataset) -> str | None:
    """None if sqrt(cos(lat)) weighting looks correctly applied, else a failure note."""
    if "weighted_field" not in ds or "field_norm" not in ds or "w_lat" not in ds:
        return "missing weighted_field/field_norm/w_lat variables (pre-weighting-step file format?)"

    wf = ds["weighted_field"]
    fn = ds["field_norm"]
    w_lat = ds["w_lat"]
    time_like = tuple(d for d in wf.dims if d in ("time", "valid_time"))
    lat_dim = _find_lat_dim(wf, exclude=time_like)

    expected_w = np.sqrt(np.clip(np.cos(np.deg2rad(np.asarray(w_lat[lat_dim].values, dtype="float32"))), 0.0, 1.0))
    stored_w = np.asarray(w_lat.values, dtype="float32")
    if not np.allclose(stored_w, expected_w, atol=1e-4, rtol=1e-4):
        return f"w_lat does not match sqrt(cos(lat)) (max abs diff={np.max(np.abs(stored_w - expected_w)):.4g})"

    if time_like:
        t = time_like[0]
        n = wf.sizes[t]
        step = max(1, n // N_TIME_SAMPLE)
        wf = wf.isel({t: slice(0, n, step)})
        fn = fn.isel({t: slice(0, n, step)})

    wf_v = np.asarray(wf.values, dtype="float32")
    fn_v = np.asarray(fn.values, dtype="float32")
    lat_axis = wf.dims.index(lat_dim)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = wf_v / fn_v
    mask = np.isfinite(ratio) & (np.abs(fn_v) > 0.1)
    ratio = np.where(mask, ratio, np.nan)
    reduce_axes = tuple(i for i in range(ratio.ndim) if i != lat_axis)
    with np.errstate(invalid="ignore"):
        observed_w = np.nanmedian(ratio, axis=reduce_axes)

    valid = np.isfinite(observed_w)
    if valid.sum() < max(3, 0.5 * len(expected_w)):
        return f"too few valid lat rows to check weighting ({int(valid.sum())}/{len(expected_w)})"
    rel_err = np.abs(observed_w[valid] - expected_w[valid]) / np.clip(expected_w[valid], 1e-6, None)
    if np.nanmax(rel_err) > 0.10:
        flat = np.nanstd(observed_w[valid]) < 0.02
        reason = (
            "weighted_field/field_norm ratio is ~flat across latitude (weighting not applied)"
            if flat
            else "weighted_field/field_norm ratio does not track sqrt(cos(lat))"
        )
        return f"{reason} (max rel err={np.nanmax(rel_err):.2%})"
    return None


def check_one(region: str, bbox_step: str, field_key: str, since: datetime.datetime, area_weight_tag: str = "") -> dict:
    cfg = MultivariateRunConfig(
        region=region,
        bbox_step=bbox_step,
        smooth_window=11,
        detrend_method="domain_mean",
        climatology_subtract="smooth",
        sigma_mode="field",
        var_threshold=0.90,
        final_k=7,
        z500_epsilon=20.0,
        vo850_epsilon=1e-6,
        tcwv_epsilon=1e-3,
        deg0l_epsilon=50.0,
        vo850_smoothing_method="none",
        yeo_johnson=True,
        source_fields=(field_key,),
        area_weight_tag=area_weight_tag,
    )
    paths = WorkflowPaths(cfg)

    row: dict = {"region": region, "bbox_step": bbox_step, "field": field_key, "ok": True, "notes": []}

    file_paths = {
        "anomaly": paths.field_anomaly_path(field_key),
        "preprocessed": paths.field_preprocessed_path(field_key),
        "pca": paths.field_pca_path(field_key),
    }
    for label, path in file_paths.items():
        if not path.exists():
            row["ok"] = False
            row["notes"].append(f"MISSING {label}: {path}")
            continue
        mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
        row[f"{label}_mtime"] = mtime
        if mtime < since:
            row["ok"] = False
            row["notes"].append(f"STALE {label} (mtime={mtime} < since={since}): {path.name}")

    if row["ok"]:
        try:
            with xr.open_dataset(file_paths["preprocessed"], chunks="auto") as ds:
                x = ds["X_matrix"]
                t_dim = x.dims[0]
                n = x.sizes[t_dim]
                step = max(1, n // N_TIME_SAMPLE)
                sample = x.isel({t_dim: slice(0, n, step)}).values
                finite_frac = float(np.isfinite(sample).mean())
                weighting_note = check_area_weighting(ds)
            if finite_frac != 1.0:
                row["ok"] = False
                row["notes"].append(f"preprocessed X_matrix has non-finite values in sampled slice (finite_frac={finite_frac})")
            if weighting_note is not None:
                row["ok"] = False
                row["notes"].append(f"AREA WEIGHTING: {weighting_note}")
        except Exception as e:
            row["ok"] = False
            row["notes"].append(f"failed to open preprocessed file: {e}")

    if row["ok"]:
        try:
            with xr.open_dataset(file_paths["pca"]) as ds:
                k = int(ds.attrs.get("k", -1))
                cum = float(ds["cumulative_explained_variance"].values[-1])
                has_nan = bool(
                    np.isnan(ds["EOF_maps"].values).any() or np.isnan(ds["PCs_reduced"].values).any()
                )
            row["k"] = k
            row["cum_var"] = round(cum, 4)
            if cum < cfg.var_threshold - 1e-6:
                row["ok"] = False
                row["notes"].append(f"PCA cumulative_explained_variance={cum:.4f} < var_threshold={cfg.var_threshold}")
            if has_nan:
                row["ok"] = False
                row["notes"].append("PCA output contains NaNs in EOF_maps/PCs_reduced")
        except Exception as e:
            row["ok"] = False
            row["notes"].append(f"failed to open pca file: {e}")

    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fields", required=True, help="Comma-separated field keys to check.")
    p.add_argument("--regions", default="europe,northamerica")
    p.add_argument("--bbox-steps", default="B0,B1,B2,B3,B4,B5")
    p.add_argument("--since", required=True, help="ISO8601 timestamp; files older than this are flagged STALE.")
    p.add_argument(
        "--area-weight-tag",
        default="",
        help="Must match the rr_00 run's --area-weight-tag to check the tagged output "
        "(e.g. 'aw2') instead of the default untagged files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    bbox_steps = [b.strip() for b in args.bbox_steps.split(",") if b.strip()]
    since = datetime.datetime.fromisoformat(args.since)

    print(f"{'region':12s} {'bbox':4s} {'field':8s} {'status':6s} {'k':3s} {'cum_var':8s} pca_mtime", flush=True)

    results = []
    for region in regions:
        for bbox_step in bbox_steps:
            for field_key in fields:
                row = check_one(region, bbox_step, field_key, since, area_weight_tag=args.area_weight_tag)
                results.append(row)
                status = "OK" if row["ok"] else "FAIL"
                k = str(row.get("k", "-"))
                cum_var = str(row.get("cum_var", "-"))
                pca_mtime = row.get("pca_mtime", "-")
                print(
                    f"{row['region']:12s} {row['bbox_step']:4s} {row['field']:8s} {status:6s} {k:>3s} {cum_var:>8s} {pca_mtime}",
                    flush=True,
                )
                for note in row["notes"]:
                    print(f"    -> {note}", flush=True)

    n_ok = sum(1 for r in results if r["ok"])
    n_total = len(results)
    print()
    print(f"SUMMARY: {n_ok}/{n_total} combinations OK")
    if n_ok != n_total:
        print("FAILURES:")
        for r in results:
            if not r["ok"]:
                print(f"  {r['region']} {r['bbox_step']} {r['field']}: {'; '.join(r['notes'])}")
        sys.exit(1)


if __name__ == "__main__":
    main()