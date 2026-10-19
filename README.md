# Weather regimes and extreme precipitation events

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23473383.svg)](https://doi.org/10.5281/zenodo.23473383)

Code accompanying:

> Eckert, M.-C., Rudolph, A., and Schielicke, L.:
> **On the scale-dependence of weather regimes describing extreme precipitation events.**

Weather regimes are detected from ERA5 geopotential height at 500 hPa (Z500) over a nested
sequence of bounding boxes, and related to convective extreme precipitation events (EPEs) in two
study areas: Germany and southern Ontario, Canada. A second stage refines each Z500 regime into
sub-regimes using 850 hPa relative vorticity, total column water vapour and the 0 °C isotherm
level, and tests whether that improves the separation of extreme from non-extreme days.

| | Produces |
|---|---|
| **EPE preprocessing** | Extreme-day labels from hourly ERA5 precipitation and CAPE. Independent of both stages; run first. |
| **Stage 1** | Z500-only weather regimes, per bounding-box step. |
| **Stage 2** | Residual sub-regimes within each Z500 regime. |

## Scope

This archive contains the analysis code: the numbered pipeline steps for EPE labelling, Stage 1
and Stage 2, the shared helpers they build on, the configuration files that pin the production
parameters, the environment definition, and the method notes under `docs/`. The steps write
NetCDF and CSV outputs.

Two retained modules keep figure-oriented names — `common/epe_figures.py` and
`common/epe_04_boxplot_render.py` — because the statistics functions the CSV exporters call live
in them.

## Layout

```
steps/                       Numbered entry points — how the analysis is run
  epe/     01 … 05
  stage1/  01 … 04
  stage2/  01 … 10  (+ optional_robustness_logregr.py)
src/
  epe_preprocessing/         Download → convective duration → extreme-day labels
  stage1_z500/               Z500 regimes
    notebooks/               Clustering steps 08–11, executed headlessly
  stage2_residual/           Relevance → logistic fit → bins → composites
    fields/                  Download, preprocessing and per-field PCA for the extra fields
  common/                    Shared helpers + the pinned override YAMLs
  bbox_steps_{europe,northamerica}.yaml    Bounding-box sequences B0–B5
configs/                     environment.yml, Dockerfile, run_configs.yaml
docs/                        Method notes
```

Two conventions:

- **Numbered files are entry points; named modules are libraries.** Python cannot import a module
  whose name starts with a digit, so step order lives in `steps/` and importable code keeps plain
  names under `src/`. Each `steps/` file puts `src/` on the path and calls one library entry point.
- **`src/stage1_z500/notebooks/*.ipynb` are pipeline steps, not figure notebooks.**
  `steps/stage1/02_cluster_regimes.py` executes them via `nbconvert`, passing configuration
  through environment variables. They are required.

## Installation

```bash
conda env create -f configs/environment.yml
conda activate weather-regimes
```

Run everything from the repository root; the `steps/` wrappers set the import path themselves.
Every step takes `--help`.

## Input data

ERA5 from the Copernicus Climate Data Store. Requires a CDS account and a `~/.cdsapirc` file
(<https://cds.climate.copernicus.eu/how-to-api>); the download steps fail fast with instructions
if credentials are missing. Period throughout: **1979–2025, May–September**.

| Field | Used by |
|---|---|
| Z500 (geopotential, 500 hPa) | Stage 1 |
| 850 hPa relative vorticity, total column water vapour, 0 °C isotherm level | Stage 2 |
| Hourly total precipitation, convective precipitation, CAPE | EPE preprocessing |

## Configuration

`configs/run_configs.yaml` is the canonical record of the production parameters. The commands
below pass them explicitly, so each invocation is self-contained.

| Parameter | Value |
|---|---|
| Period / season | 1979–2025, months 05–09 |
| Climatology smoothing window | 11 days |
| Detrending | domain mean |
| PCA variance threshold | 0.90 |
| Z500 regimes | 7 |
| Clustering | Ward hierarchical |
| Random seed | 42 |
| EPE percentile | 0.999 |
| EPE domain percentile / aggregation | `--domain-pctl-q 0.90`, `--domain-agg lexmax` |
| CAPE / convective fraction | CAPE > 0, cp ≥ 1 % of tp |
| Stage-2 sub-regime split | 2 bins, top 30 % of days by predicted probability |

Fitted per-region choices are pinned in `src/common/`: `k_overrides_<region>_custom.yaml` (Z500
regime count per bounding box), `rr_k_overrides_<region>_B*.yaml`,
`bin_overrides_<region>_domainQ900_lexmax.yaml` (Stage-2 bin edges),
`regime_alignment_<region>.yaml` (regime numbering aligned across bounding boxes) and
`rr01_field_selection.yaml`.

`REGION` is `europe` or `northamerica`; the matching study area / region label is `germany` or
`ontario_south`.

## Reproducing the analysis

Run EPE first, then Stage 1, then Stage 2 — Stage 2 consumes both the EPE labels and the Stage-1
regimes. Examples are for Europe; repeat with `--region northamerica` and
`--region-label ontario_south` / `--study-area ontario_south`.

### EPE preprocessing

```bash
# 1. Download hourly ERA5 tp / cp / CAPE
python steps/epe/01_download_era5.py \
  --region-label germany --years 1979-2025 --months 05,06,07,08,09 \
  --use-region-bbox --chunk-by year

# 2. Convective duration + domain-percentile extreme days (production detector)
python steps/epe/03_detect_convective_duration.py \
  --region-label germany --years 1979-2025 \
  --timezone Europe/Berlin --day-start-hour-local 6 \
  --hourly-precip-metric tp_hourly \
  --q 0.999 --quantile-sample all_hours --wet-threshold-mm 0.1 \
  --cape-threshold 0 --cp-frac-min 0.01 \
  --domain-agg lexmax --domain-pctl-q 0.90 --hourly-mask conv \
  --chunks-time 720 --chunks-lat 100 --chunks-lon 100 --persist
```

North America: `--region-label ontario_south --timezone America/Toronto`. The
`convective_duration_*.nc` file written here is Stage 2's `EXTREME_LABELS_PATH`.

Steps `02_aggregate_daily` and `04_detect_extremes` are the daily-aggregate route used for the
appendix comparison, not the production chain; `05_derive_domain_pctl_labels` turns a detection
file into labelled day sets.

### Stage 1 — Z500 weather regimes

```bash
# 1. Preprocessing (steps 2-6): climatology, detrending, anomalies, area weighting
for step in B0 B1 B2 B3 B4 B5; do
  python steps/stage1/01_preprocess.py \
    --region europe --bbox-step "$step" --raw-years-span 1979-2025 \
    --smooth-window 11 --detrend-method domain_mean \
    --climatology-subtract smooth --sigma-mode field \
    --var-threshold 0.90 --yeo-johnson --no-figs
done

# 2. Clustering (steps 8-11)
python steps/stage1/02_cluster_regimes.py \
  --region europe --steps "B0;B1;B2;B3;B4;B5" --cluster-mode ward \
  --year-span 1979-2025 --window-days 11 \
  --detrend-method domain_mean --climatology-subtract smooth \
  --sigma-mode field --var-threshold 0.90 --stop-after-step 10

# 3. Per-regime Z500 anomaly composites (NetCDF)
python steps/stage1/03_anom_composites.py --region europe --year-span 1979-2025
# Inputs/outputs can be pinned with --in-anom / --in-raw / --in-regimes / --out

# 4. Regime statistics (CSV)
python steps/stage1/04_export_statistics_csv.py \
  --region europe --bbox-step B5 --k 7 \
  --cluster-method ward --study-area germany \
  --extremes-q 0.999 --quantile-sample all_hours \
  --domain-pctl-q 0.90 --domain-agg lexmax
```

### Stage 2 — residual sub-regimes

```bash
export EXTREME_LABELS_PATH=/path/to/convective_duration_germany_..._domainQ900_....nc
export OUTPUT_TAG=domainQ900_lexmax
export BIN_OVERRIDES=src/common/bin_overrides_europe_${OUTPUT_TAG}.yaml

# 1-3. Download and preprocess the additional fields (per-field PCA), then verify
for field in z500 vo850 tcwv deg0l; do
  python steps/stage2/01_download_fields.py \
    --field-key "$field" --region-label europe \
    --years 1979-2025 --months 05,06,07,08,09
done

python steps/stage2/02_preprocess_fields.py \
  --region europe --bbox-step B5 \
  --year-span 1979-2025 --raw-years-span 1979-2025 \
  --smooth-window 11 --detrend-method domain_mean \
  --climatology-subtract smooth --sigma-mode field \
  --var-threshold 0.90 --max-modes-per-field 100 \
  --final-k 7 --random-state 42 \
  --source-fields z500,vo850,tcwv,deg0l \
  --z500-epsilon 20.0 --vo850-epsilon 1.0e-6 \
  --tcwv-epsilon 1.0e-3 --deg0l-epsilon 50.0 \
  --vo850-smoothing-method none

python steps/stage2/03_verify_preprocess.py --regions europe --bbox-steps B5

# 4-5. Per-regime field relevance, then the field-selection map
for step in B0 B1 B2 B3 B4 B5; do
  python steps/stage2/04_feature_relevance.py \
    --region europe --bbox-step "$step" \
    --mode regime --cluster-method ward \
    --field-representation domain_mean \
    --study-area germany --extreme-labels-path "$EXTREME_LABELS_PATH"
done

python steps/stage2/05_write_field_selection.py \
  --build-regime-map --region europe --bbox-step B5 --z500-k 7 \
  --z500-cluster-method ward --field-representation domain_mean --top-n-fields 3

# 6-7. Per-regime logistic fit, then sub-regime bin edges
python steps/stage2/06_logistic_fit.py \
  --region europe --bbox-step B5 --z500-k 7 \
  --z500-cluster-method ward --per-regime-mode \
  --regime-fields-map <path printed by step 5> \
  --predictor-representation domain_mean --include-z500 \
  --extreme-labels-path "$EXTREME_LABELS_PATH" \
  --output-tag "$OUTPUT_TAG" --year-span 1979-2025 --smooth-window 11

python steps/stage2/07_select_bins.py \
  --region europe --bbox-step B5 --z500-k 7 \
  --z500-cluster-method ward --per-regime-mode \
  --extreme-labels-path "$EXTREME_LABELS_PATH" \
  --output-tag "$OUTPUT_TAG" \
  --max-bins 2 --target-day-frac 0.3 --bin-overrides-path "$BIN_OVERRIDES"

# 8-10. Composites and statistics
python steps/stage2/08_composites.py \
  --region europe --bbox-step B5 --z500-cluster-method ward \
  --output-tag "$OUTPUT_TAG" --bin-overrides-path "$BIN_OVERRIDES"
python steps/stage2/09_export_statistics_csv.py \
  --region europe --bbox-step B5 --z500-cluster-method ward \
  --output-tag "$OUTPUT_TAG" --n-bins 2 --bin-overrides-path "$BIN_OVERRIDES"
python steps/stage2/10_export_subregime_persistence.py \
  --region europe --bbox-step B5 --z500-cluster-method ward \
  --output-tag "$OUTPUT_TAG" --n-bins 2 --bin-overrides-path "$BIN_OVERRIDES"
```

Field selection (step 4) and the logistic fit (step 6) use the same predictors. Each field enters
as its area-weighted (cos φ) domain-mean anomaly, and the Z500 domain-mean anomaly is included in
both: the greedy selection scores every candidate on top of it, and `--include-z500` keeps it in
the fit. A separate model is fitted per Z500 regime. `--field-representation pca_block` (step 4)
and `--predictor-representation pca` (step 6) are the alternative that uses each field's PCA
block (90 % variance, at most 100 modes); it overfits the few extreme days per regime.

`--target-day-frac 0.3` is the pinned setting: it fixes the high bin to a fraction of each
regime's days and so always yields two groups, unlike `--min-extreme-per-bin`, which collapses to
one bin for regimes with too few extreme days.
`steps/stage2/optional_robustness_logregr.py` adds coefficient-stability and regularisation
sweeps for the step-6 fit; not needed for the main results.

## Notes

- Paths are resolved from `MODULE_ROOT` (and `DETECT_WEATHER_ROOT` for the field downloads),
  defaulting to the repository root. Set them, or pass the explicit path arguments, if you run
  from elsewhere.
- The preprocessing and composite steps are memory-hungry (multi-decade hourly ERA5). Chunk sizes
  are exposed as `--chunks-time/-lat/-lon` where relevant.

## Licence

Code: MIT, see [LICENSE](LICENSE). ERA5 is subject to the Copernicus licence.

## Citing

See [CITATION.cff](CITATION.cff).

Results data (daily regime and sub-regime assignments): https://doi.org/10.5281/zenodo.23472858
