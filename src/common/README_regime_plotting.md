# `regime_plotting` — shared figures for notebooks and standalone scripts

This package holds **Matplotlib figure builders** used by:

- the **standalone Z500 composites / statistics scripts** (`scripts/plot_regime_z500_precip_composites.py`,
  `scripts/export_regime_z500_statistics_csv.py`), and
- the **Step 8 / 9 / 10–11 notebooks** (import the same functions so plots match those scripts).

`regime_plotting` is the **shared plotting layer**. It focuses on reusable figure construction, map
layout, Cartopy helpers, and data-to-figure transformations. It does **not** decide which bbox files
to open or how output filenames are built — that's `src/summary_reports/` (the **resolver /
shared-helper layer**: bbox YAML + `k_overrides` loading, Step 8/9/10–11 NetCDF resolution, extreme-event
spec matching, boxplot/statistics helpers). `summary_reports` used to also assemble a multi-page bbox
summary PDF; that report-assembly path (`cli.py`, `report_pdf.py`'s `PdfPages` orchestration) was removed
since PDF reporting is no longer needed for the Z500-only pipeline — see `src/summary_reports/README.md`.

So the two folders are related but not duplicates: `regime_plotting` builds figures, `summary_reports`
resolves inputs and provides shared rendering/labelling helpers.

---

## Prerequisites

1. **Working directory:** `cd` into `1_Detect_Weather_Patterns`.
2. **Python environment:** same env you use for `xarray` / `matplotlib` / (optional) Cartopy.
3. **NetCDF inputs** the resolver expects (names must match the preprocessing tag you pass):
   - **Step 8:** `data/processed/<region>/08_Z500_HC_Ward_w*_eps*_var*_..._<year-span>.nc`
   - **Step 9:** `results/clustering/<region>/09_Z500_regimes_kmeans_k*_..._<year-span>_stepB4_....nc` (example for bbox B4)
   - **Step 10–11:** `results/regimes/<region>/10_11_Z500_regime_reconstruction_characterisation_k*_..._<year-span>_stepB4_....nc`
4. **`--var-threshold`** is the **PCA variance** tag in Z500 filenames (`0.90` → `var90`, `0.99` → `var99`). It must match `07_*` / `08_*` on disk; it is **not** the precipitation exceedance percentile used in EPE extremes.
5. **EPE.4 (extreme precipitation boxplots):** NetCDF under
   `data/processed/<region>/extreme_precip_events/` or, with **`--study-area`**,
   `data/processed/<study_area>/extreme_precip_events/` (e.g. `germany`).
   Files like `extremes_*.nc` or `convective_duration_*.nc` must **overlap in time** with the Step 9 `regime_labels` for this bbox.

`k` per bbox step comes from `src/summary_reports/k_overrides_<region>_custom.yaml` unless you only use explicit `--*-nc` overrides (single step).

---

## Producing composites / statistics (NetCDF for steps 8–11 already on disk)

On TU HPC, once `z_03_run_regime_clustering_tu_hpc.sh` has produced the Step 9/10–11 NetCDF for a
region/bbox, run the standalone scripts directly:

```bash
sbatch scripts/z_04_run_regime_composites_tu_hpc.sh    # PNG composite mosaics + Z500 anomaly mosaics
sbatch scripts/z_05_run_regime_stats_csv_tu_hpc.sh      # stats CSV export
```

Locally / manually, the same underlying scripts:

```bash
cd "1_Detect_Weather_Patterns"

PYTHONPATH=src python3 scripts/plot_regime_z500_precip_composites.py \
  --region europe --bbox-step B4 --k 7 --study-area germany \
  --precip-metric flag_frequency --out-dir results/01_paper_plots/2_results/01_z500_weather_regimes/2_composites/europe

PYTHONPATH=src python3 scripts/export_regime_z500_statistics_csv.py \
  --region europe --bbox-step B4 --k 7 --study-area germany \
  --out-dir results/01_paper_plots/2_results/01_z500_weather_regimes/stats_csv
```

If auto-discovery fails, both scripts accept explicit `--*-nc` overrides (`--help` for the full list).

---

## Where `regime_plotting` fits

| Piece | Role |
|--------|------|
| `src/regime_plotting/` | Figure construction (`build_figure_*`, grids, Cartopy helpers, raw-composite path logic in `io_raw_composite.py`). |
| `scripts/plot_regime_z500_precip_composites.py` | CLI entrypoint → resolves inputs via `summary_reports.resolver` / `.extremes_spec`, builds figures via `regime_plotting`, writes PNG/PDF composite mosaics. |
| `scripts/export_regime_z500_statistics_csv.py` | CLI entrypoint → same resolver layer, writes per-bbox statistics CSVs (no figures). |

For more flags and defaults, see `python3 scripts/plot_regime_z500_precip_composites.py --help` /
`python3 scripts/export_regime_z500_statistics_csv.py --help`.
