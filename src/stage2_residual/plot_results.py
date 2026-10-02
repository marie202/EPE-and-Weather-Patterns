#!/usr/bin/env python3
"""
Quick diagnostic plots for residual regime (rr_02) results.

Produces a multi-page PDF with:
  1. Regime frequency bar chart (overall + per Z500 regime)
  2. Cross-tabulation heatmap: Z500 regime × residual sub-regime
  3. Composite anomaly maps for each residual regime (one panel per field)
  4. Seasonal distribution of each regime (MJJAS months)

Usage:
    python -m stage2_residual.plot_results \
        --region europe --bbox-step B5 \
        --module-root /path/to/detect_weather
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from common.colorbar_discretize import discretize_bounded, discretize_diverging, style_discrete_ticks

from stage2_residual.feature_relevance import _filter_by_area_weight_tag


# ---------------------------------------------------------------------------
# File finders
# ---------------------------------------------------------------------------

def _dw_root(module_root: Path) -> Path:
    if (module_root / "results" / "stage2_residual").is_dir():
        return module_root
    parent = module_root.parent
    if (parent / "results" / "stage2_residual").is_dir():
        return parent
    return module_root


def _find_rr_regimes(
    dw: Path,
    region: str,
    bbox: str,
    module_root: Path | None = None,
    mode: str | None = None,
    z500_cluster_method: str | None = None,
) -> Path | None:
    """Locate the Stage-2 RR regimes file for region/bbox.

    `mode`: "bbox" or "regime" restricts to that pipeline's output only (rr_02
    vs rr_022 — both can leave files in the same directory, tagged "_perregime_"
    for the latter). Pass explicitly whenever the caller knows which pipeline it
    wants plotted; leave None only for best-effort auto-detection (prefers
    bbox-mode files when both are present, since that used to be the only mode).

    `z500_cluster_method`: "kmeans" or "ward" — restricts to RR runs built on that
    Z500 parent-regime source (`stage2_residual.paths.experiment_id` tags ward runs
    "_zward"; kmeans/the Ward-seeded hybrid stay untagged). Both kmeans- and ward-based
    RR runs can coexist in the same directory for the same region/bbox/mode — leave
    None only for best-effort auto-detection across both (mtime/field-match tie-break,
    no method preference).
    """
    d = dw / "results" / "stage2_residual" / "clustering" / region
    if not d.is_dir():
        return None
    # Prefer the post-processed (rare-regime-filtered) file when it exists
    filtered = sorted(d.glob(f"09b_RR_regimes_filtered_*step{bbox}*.nc"))
    all_matches = filtered if filtered else sorted(d.glob(f"09_RR_regimes_kmeans_*step{bbox}*.nc"))
    if not all_matches:
        return None

    if z500_cluster_method is not None:
        want_ward = z500_cluster_method == "ward"
        all_matches = [m for m in all_matches if ("_zward_" in m.name) == want_ward]
        if not all_matches:
            return None

    if mode == "regime":
        all_matches = [m for m in all_matches if "_perregime_" in m.name]
        if not all_matches:
            return None
        newest = max(all_matches, key=lambda p: p.stat().st_mtime)
        if len(all_matches) > 1:
            print(
                f"[rr_regimes] {len(all_matches)} per-regime files for {region}/{bbox}; "
                f"selected most recently modified: {newest.name}"
            )
        return newest

    if mode == "bbox":
        all_matches = [m for m in all_matches if "_perregime_" not in m.name]
        if not all_matches:
            return None

    if len(all_matches) == 1:
        return all_matches[0]

    # mode=None (auto-detect): per-regime-mode files (each Z500 regime clusters on its
    # own field subset, filename tagged "_perregime_") share the same glob as bbox-mode
    # files, so both pipelines' outputs can land in the same directory. Prefer bbox-mode
    # files whenever any exist — pass mode="regime" explicitly if you want the per-regime
    # output instead.
    non_perregime = [m for m in all_matches if "_perregime_" not in m.name]
    if non_perregime:
        all_matches = non_perregime
        if len(all_matches) == 1:
            return all_matches[0]
    else:
        # Only per-regime files present. Matching against the single region/bbox
        # bbox-mode field selection below can never apply (per-regime filenames encode
        # the UNION of every regime's own fields) — tie-break by mtime instead of
        # guessing from a mismatched tag.
        newest = max(all_matches, key=lambda p: p.stat().st_mtime)
        print(
            f"[rr_regimes] {len(all_matches)} per-regime files for {region}/{bbox}; "
            f"selected most recently modified: {newest.name}"
        )
        return newest

    # When multiple bbox-mode files exist, prefer the one whose name matches the rr_01 field selection.
    if module_root is not None:
        yaml_path = module_root / "src" / "common" / "rr01_field_selection.yaml"
        if yaml_path.exists():
            try:
                from .write_field_selection import read_fields_for
                fields_str = read_fields_for(yaml_path, region, bbox)
                if fields_str:
                    extra = "+".join(f for f in fields_str.split("+") if f != "z500")
                    tag = f"z500{extra}_w"
                    preferred = [m for m in all_matches if tag in m.name]
                    if preferred:
                        if len(preferred) < len(all_matches):
                            print(
                                f"[rr_regimes] {len(all_matches)} files for {region}/{bbox}; "
                                f"selected field-matched: {preferred[-1].name}"
                            )
                        return preferred[-1]
                    print(
                        f"[rr_regimes] WARNING: no {region}/{bbox} file matches "
                        f"field selection '{fields_str}' (tag='{tag}') — "
                        f"falling back to most recently modified: {max(all_matches, key=lambda p: p.stat().st_mtime).name}"
                    )
            except Exception as exc:
                print(f"[rr_regimes] WARNING: could not read field selection YAML: {exc}")

    return max(all_matches, key=lambda p: p.stat().st_mtime)


def _find_rr_logistic_merged(
    dw: Path,
    region: str,
    bbox: str,
    output_tag: str = "",
    z500_cluster_method: str | None = None,
) -> Path | None:
    """Locate the merged Stage-2 [LOGISTIC] continuous-index file
    (``stage2_logistic.merge_per_regime_logistic_outputs``'s
    ``09_RR_logistic_perregime_merged_...`` output) for region/bbox — the
    counterpart of `_find_rr_regimes` for the continuous
    `predicted_extreme_prob` index rather than discrete `regime_labels`.

    Unlike `_find_rr_regimes`, there is only ever one production mode here
    (stage2_logistic.py requires `--per-regime-mode`), so no bbox/regime mode
    disambiguation is needed — only `z500_cluster_method` (kmeans vs ward
    parent regimes) and `output_tag` (e.g. a domain_pctl_q sensitivity sweep
    tag) narrow the match.
    """
    import re

    d = dw / "results" / "stage2_residual" / "clustering" / region
    if not d.is_dir():
        return None
    matches = sorted(d.glob(f"09_RR_logistic_perregime_merged_*step{bbox}*.nc"))
    if output_tag:
        matches = [m for m in matches if m.name.endswith(f"_{output_tag}.nc")]
    else:
        # experiment_id() always ends "..._k{final_k:02d}" — an --output-tag (if any)
        # is appended after that. Without an explicit tag, only match filenames with
        # nothing after it, so a sensitivity-sweep variant (e.g. a domain_pctl_q
        # comparison run) is never silently picked up as "the" production file.
        matches = [m for m in matches if re.search(r"_k\d{2}\.nc$", m.name)]
    if not matches:
        return None
    if z500_cluster_method is not None:
        want_ward = z500_cluster_method == "ward"
        matches = [m for m in matches if ("_zward" in m.name) == want_ward]
        if not matches:
            return None
    if len(matches) == 1:
        return matches[0]
    newest = max(matches, key=lambda p: p.stat().st_mtime)
    print(f"[rr_logistic] {len(matches)} merged logistic files for {region}/{bbox}; "
          f"selected most recently modified: {newest.name}")
    return newest


def _find_z500_regimes(
    dw: Path,
    region: str,
    bbox: str,
    smooth_window: int = 11,
    z500_k: int | None = None,
    cluster_method: str = "kmeans",
) -> Path | None:
    """`cluster_method`: "kmeans" (default; also covers the Ward-seeded K-means hybrid,
    which keeps the "kmeans" tag) or "ward" (pure Ward HRC, z_03's CLUSTER_MODE=ward) —
    must match whichever Z500 parent-regime source this bbox's RR regimes (see
    `_find_rr_regimes`) were actually built from, or the Z500-regime x RR-subregime
    pairing shown in composites/statistics will be internally inconsistent."""
    d = dw / "results" / "clustering" / region
    if not d.is_dir():
        return None
    matches = sorted(d.glob(f"09_Z500_regimes_{cluster_method}_*step{bbox}*.nc"))
    if not matches:
        return None
    # Prefer file matching smooth_window (e.g. w11 over w21)
    w_tag = f"_w{smooth_window}_"
    preferred = [m for m in matches if w_tag in m.name]
    if preferred:
        matches = preferred
    # Further narrow by z500_k if provided
    if z500_k is not None:
        k_tag = f"_k{z500_k:02d}_"
        k_preferred = [m for m in matches if k_tag in m.name]
        if k_preferred:
            matches = k_preferred
    return matches[-1]


def _find_raw_z500(dw: Path, region: str) -> Path | None:
    """Locate the raw ERA5 Z500 geopotential file (not anomaly) for a region."""
    d = dw / "data" / "raw" / "era5" / "z500" / region
    if d.is_dir():
        matches = sorted(d.glob("era5_z500_*.nc"))
        if matches:
            return matches[-1]
    return None


def _find_anomaly(dw: Path, region: str, field: str, bbox: str, smooth_window: int,
                   area_weight_tag: str = "") -> Path | None:
    """``area_weight_tag`` is pinned exactly for non-Z500 fields, the same way
    ``stage2_residual.paths.resolve_mf_pca_path`` and
    ``feature_relevance._filter_by_area_weight_tag`` do — without this, the glob below
    would fall back to alphabetical-last tie-breaking, which only happens to prefer a
    tagged rerun (e.g. ``_aw2.nc``) over the untagged pre-fix file by accident of string
    ordering. Z500 lives outside the rr_00 area-weighting rerun and has no tag concept,
    so it is left unfiltered."""
    if field == "z500":
        d = dw / "data" / "preprocessed" / region
        if d.is_dir():
            matches = sorted(d.glob(f"*z500*anom*step{bbox}*.nc"))
            w_tag = f"_w{smooth_window}_"
            preferred = [m for m in matches if w_tag in m.name]
            return preferred[-1] if preferred else (matches[-1] if matches else None)
    else:
        d = dw / "data" / "multivariate" / "preprocessed" / region / field
        if d.is_dir():
            matches = sorted(d.glob(f"03_MF_{field}_anomaly_*step{bbox}*.nc"))
            matches = _filter_by_area_weight_tag(matches, area_weight_tag)
            w_tag = f"_w{smooth_window}_"
            preferred = [m for m in matches if w_tag in m.name]
            return preferred[-1] if preferred else (matches[-1] if matches else None)
    return None


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_labels(path: Path, label_var: str = "regime_labels") -> tuple[np.ndarray, pd.DatetimeIndex]:
    ds = xr.open_dataset(path)
    labels = ds[label_var].values
    time_coord = "valid_time" if "valid_time" in ds.coords else "time"
    times = pd.DatetimeIndex(ds[time_coord].values)
    ds.close()
    return labels, times


def _load_anomaly_composite(path: Path, labels: np.ndarray, n_regimes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load anomaly and compute per-regime composite. Returns (composites, lats, lons)."""
    try:
        ds = xr.open_dataset(path)
        var = next((v for v in ds.data_vars if "anom" in v.lower()), list(ds.data_vars)[0])
        da = ds[var].load()
        ds.close()
    except Exception as e:
        print(f"  WARNING: could not load {path.name}: {e}")
        return None

    lat_dim = next((d for d in da.dims if "lat" in d.lower()), None)
    lon_dim = next((d for d in da.dims if "lon" in d.lower()), None)
    if lat_dim is None or lon_dim is None:
        return None

    lats = da[lat_dim].values
    lons = da[lon_dim].values
    data = da.values  # (time, lat, lon)

    composites = np.full((n_regimes, len(lats), len(lons)), np.nan)
    for r in range(1, n_regimes + 1):
        mask = labels == r
        if mask.sum() > 0:
            composites[r - 1] = np.nanmean(data[mask], axis=0)

    return composites, lats, lons


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _plot_frequency(pdf: PdfPages, rr_labels: np.ndarray, z500_labels: np.ndarray,
                    n_rr: int, n_z500: int, region: str, bbox: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Residual Regime Frequencies — {region} {bbox}", fontsize=12, fontweight="bold")

    # Overall RR frequencies
    ax = axes[0]
    unique, counts = np.unique(rr_labels, return_counts=True)
    pcts = 100 * counts / len(rr_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    bars = ax.bar([f"SR{int(r)}" for r in unique], pcts, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)
    ax.set_ylabel("Frequency (%)")
    ax.set_title(f"Stage 2 Residual Regimes (K={n_rr})")
    ax.set_ylim(0, max(pcts) * 1.2)

    # Cross-tab heatmap: Z500 × RR
    ax = axes[1]
    crosstab = np.zeros((n_z500, n_rr))
    for zi in range(1, n_z500 + 1):
        z_mask = z500_labels == zi
        for ri in range(1, n_rr + 1):
            crosstab[zi - 1, ri - 1] = (rr_labels[z_mask] == ri).sum()
    # Normalize by Z500 regime size → conditional probability
    row_sums = crosstab.sum(axis=1, keepdims=True)
    crosstab_pct = 100 * crosstab / np.where(row_sums > 0, row_sums, 1)

    ct_cmap, ct_norm, ct_extend = discretize_bounded(0, 100, "YlOrRd")
    im = ax.imshow(crosstab_pct, cmap=ct_cmap, norm=ct_norm)
    style_discrete_ticks(plt.colorbar(im, ax=ax, label="% of Z500 regime days", extend=ct_extend))
    ax.set_xticks(range(n_rr))
    ax.set_xticklabels([f"SR{r+1}" for r in range(n_rr)])
    ax.set_yticks(range(n_z500))
    ax.set_yticklabels([f"Z500-{z+1}" for z in range(n_z500)])
    ax.set_xlabel("Residual Sub-Regime")
    ax.set_ylabel("Z500 Parent Regime")
    ax.set_title("Sub-regime distribution\nwithin each Z500 regime (%)")
    for zi in range(n_z500):
        for ri in range(n_rr):
            ax.text(ri, zi, f"{crosstab_pct[zi, ri]:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if crosstab_pct[zi, ri] > 60 else "black")

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_seasonal(pdf: PdfPages, rr_labels: np.ndarray, times: pd.DatetimeIndex,
                   n_rr: int, region: str, bbox: str) -> None:
    month_names = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
    months = times.month

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Seasonal Distribution of Residual Regimes — {region} {bbox}", fontsize=12, fontweight="bold")

    x = np.arange(5)
    width = 0.8 / n_rr
    colors = plt.cm.tab10(np.linspace(0, 1, n_rr))

    for ri in range(1, n_rr + 1):
        r_mask = rr_labels == ri
        month_pcts = [100 * ((months == m) & r_mask).sum() / (months == m).sum()
                      for m in [5, 6, 7, 8, 9]]
        offset = (ri - 1 - n_rr / 2 + 0.5) * width
        ax.bar(x + offset, month_pcts, width=width * 0.9,
               label=f"SR{ri}", color=colors[ri - 1])

    ax.set_xticks(x)
    ax.set_xticklabels(list(month_names.values()))
    ax.set_ylabel("% of month days in regime")
    ax.set_title("Seasonal occurrence by regime")
    ax.legend()
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _plot_composites(pdf: PdfPages, rr_labels: np.ndarray, n_rr: int,
                     anomaly_paths: dict[str, Path], region: str, bbox: str,
                     smooth_window: int) -> None:
    n_fields = len(anomaly_paths)
    if n_fields == 0:
        return

    for field, path in anomaly_paths.items():
        result = _load_anomaly_composite(path, rr_labels, n_rr)
        if result is None:
            continue
        composites, lats, lons = result

        fig, axes = plt.subplots(1, n_rr, figsize=(5 * n_rr, 4),
                                 subplot_kw={"projection": None})
        if n_rr == 1:
            axes = [axes]
        fig.suptitle(f"{field.upper()} composite anomalies by residual regime — {region} {bbox}",
                     fontsize=11, fontweight="bold")

        vmax = np.nanpercentile(np.abs(composites), 95)
        vmax = max(vmax, 1e-6)
        cmap, norm, extend = discretize_diverging(-vmax, vmax, "RdBu_r")

        for ri, ax in enumerate(axes):
            comp = composites[ri]
            count = (rr_labels == ri + 1).sum()
            pct = 100 * count / len(rr_labels)
            im = ax.pcolormesh(lons, lats, comp, cmap=cmap, norm=norm, shading="auto")
            ax.set_title(f"SR{ri+1}\n{count} days ({pct:.1f}%)", fontsize=9)
            ax.set_xlabel("Lon")
            if ri == 0:
                ax.set_ylabel("Lat")
            style_discrete_ticks(plt.colorbar(im, ax=ax, shrink=0.7, label=field, extend=extend))

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_plots(module_root: Path, region: str, bbox_step: str,
              smooth_window: int = 11, out_dir: Path | None = None) -> Path:
    dw = _dw_root(module_root)

    rr_path = _find_rr_regimes(dw, region, bbox_step, module_root=module_root)
    if rr_path is None:
        raise FileNotFoundError(f"No RR regime file found for {region} {bbox_step} under {dw}/results/residual_regimes/clustering/")

    z500_path = _find_z500_regimes(dw, region, bbox_step)
    if z500_path is None:
        raise FileNotFoundError(f"No Z500 regime file found for {region} {bbox_step}")

    print(f"[rr_plot] RR regimes:   {rr_path.name}")
    print(f"[rr_plot] Z500 regimes: {z500_path.name}")

    rr_labels, times = _load_labels(rr_path)
    z500_labels, _ = _load_labels(z500_path)
    n_rr = int(rr_labels.max())
    n_z500 = int(z500_labels.max())
    print(f"[rr_plot] {n_rr} residual regimes, {n_z500} Z500 regimes, {len(rr_labels)} days")

    # Find which fields were used (from filename)
    import re
    m = re.search(r"z500([^_]+)_w\d+", rr_path.name)
    fields = ["z500"]
    if m:
        extra = [f for f in m.group(1).split("+") if f]
        fields += extra
    print(f"[rr_plot] Fields: {fields}")

    anomaly_paths = {}
    for field in fields:
        p = _find_anomaly(dw, region, field, bbox_step, smooth_window)
        if p:
            anomaly_paths[field] = p
            print(f"[rr_plot]   ✓ {field}: {p.name}")
        else:
            print(f"[rr_plot]   ✗ {field}: anomaly not found")

    out_dir = out_dir or (dw / "results" / "stage2_residual" / "figures" / region)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"rr_results_{region}_{bbox_step}.pdf"

    with PdfPages(out_pdf) as pdf:
        _plot_frequency(pdf, rr_labels, z500_labels, n_rr, n_z500, region, bbox_step)
        _plot_seasonal(pdf, rr_labels, times, n_rr, region, bbox_step)
        _plot_composites(pdf, rr_labels, n_rr, anomaly_paths, region, bbox_step, smooth_window)

    print(f"[rr_plot] Saved: {out_pdf}")
    return out_pdf


def main(argv=None) -> int:
    import sys
    p = argparse.ArgumentParser(description="Plot residual regime results.")
    p.add_argument("--region", required=True, choices=("europe", "northamerica"))
    p.add_argument("--bbox-step", required=True)
    p.add_argument("--module-root", type=Path, default=None)
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.module_root is None:
        here = Path(__file__).resolve()
        args.module_root = here.parents[2]

    run_plots(
        module_root=args.module_root.expanduser().resolve(),
        region=args.region,
        bbox_step=args.bbox_step,
        smooth_window=args.smooth_window,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
