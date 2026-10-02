# `summary_reports` — shared helpers for Z500 regime resolution / rendering

This package used to assemble a multi-page **bbox summary PDF** (`cli.py` + `report_pdf.py`,
driven by `scripts/generate_bbox_run_summary.py`). That PDF-report path has been **removed** —
PDF reporting for the Z500-only regime pipeline is no longer needed. What remains is the
**shared helper layer** several still-active scripts import from:

- **`resolver.py`** — bbox YAML / `k_overrides_*.yaml` loading, Step 8/9/10–11 NetCDF glob +
  disambiguation (`resolve_step_file`, `k_for_step`, `load_bbox_steps_yaml`, `load_k_overrides`).
- **`extremes_spec.py`** — the `ExtremesSpec` dataclass and CLI-argument helpers used to pin an
  EPE extreme-event definition (quantile, sample, CAPE threshold, domain-aggregated variant) and
  match it against `convective_duration_*.nc` filenames.
- **`page_layout.py`** — shared matplotlib typography/layout helpers.
- **`step0809_render.py` / `step1011_render.py`** — Step 8/9/10–11 plot builders (dendrogram,
  elbow/cumulative panels, composite maps, persistence/transition figures) from NetCDF.
- **`mf_cli.py`** — multivariate-pipeline label resolution helpers.
- **`epe_04_boxplot_render.py`** — EPE.4-style boxplot/statistics figure builders.
- **`report_pdf.py`** — low-level `PdfPages` assembly / figure-numbering helpers, still used
  where a script exports a standalone multi-page PDF (e.g. `--pdf-out` in
  `scripts/plot_regime_z500_precip_composites.py`); no longer used to build a bundled bbox report.

Current consumers of this package:

- `scripts/plot_regime_z500_precip_composites.py` (composite maps; run by
  `scripts/z_04_run_regime_composites_tu_hpc.sh`)
- `scripts/export_regime_z500_statistics_csv.py` (stats CSV export; run by
  `scripts/z_05_run_regime_stats_csv_tu_hpc.sh`)
- `src/run_regime_clustering.py` (resolves per-step `k` for notebooks 09/10_11)
- `src/residual_regimes/feature_relevance.py` and other RR-pipeline modules (resolve the
  Z500 parent-regime labels file for a given region/bbox_step)

Composites are produced exclusively on HPC via `scripts/z_04_run_regime_composites_tu_hpc.sh`
and downloaded into `results/01_paper_plots/2_results/01_z500_weather_regimes/2_composites/` —
there is no local notebook reproduction path anymore (`2_2_Results_Z500_regime_plots.ipynb` was
removed since it always ended up drifting out of sync with the HPC script's defaults).

The companion folder `src/regime_plotting/` is the **shared figure-building layer**: it holds the
actual `build_figure_*` functions. `summary_reports` resolves *which* NetCDF to read and provides
shared rendering/labelling helpers; `regime_plotting` draws the figures.

## Configuration files in this directory

| File | Purpose |
|------|---------|
| `k_overrides_{europe,northamerica}_custom.yaml` | `default_k` and per-step `k_by_step`, the single source of truth for clustering / file matching (z_03/z_04/z_05 and the RR pipeline all key off these). |
| `k_overrides_{europe,northamerica}_multivariate.yaml` | Same, for the multivariate pipeline. |
| `interpretation.yaml` | Short reading-guide text keyed by section id (legacy; no longer rendered anywhere since the PDF-report path was removed). |
| `regime_alignment_*.yaml` | Cross-run regime-label alignment tables. |

## Dependencies

- **Core:** Python 3, `matplotlib`, `numpy`, `pandas`, `xarray`, `PyYAML`.
- **Step 8 dendrogram:** `scipy` (optional; a text fallback is used if plotting fails).
- **Maps:** `cartopy` plus Natural Earth shapefiles (Cartopy may download these once; offline users can set `CARTOPY_DATA_DIR`).
- **EPE boxplots:** Imports `epe.epe_functions` from this repo; EPE **extremes** and/or **convective duration** NetCDF under the processed region folder.

## See also

- Bbox definitions: `src/bbox_steps_europe.yaml`, `src/bbox_steps_northamerica.yaml`.
- Filename suffix helper for pipeline outputs: `preprocessing_helpers.bbox_step_filename_suffix` (and related bbox slug utilities).
