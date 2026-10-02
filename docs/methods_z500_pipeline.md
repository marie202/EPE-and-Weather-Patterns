# Workflow plan: ERA5 Z500 → weather regimes (Europe)

> **Note on paths in this document.** It was written against the pre-release layout and
> refers to modules and SLURM scripts by their old locations. The code has since been
> reorganised; see [`README.md`](../README.md) for the current structure. Mapping:
>
> | Old | New |
> |---|---|
> | `src/epe/` | `src/epe_preprocessing/` |
> | `src/z_pipeline/` | `src/stage1_z500/notebooks/` |
> | `src/residual_regimes/` | `src/stage2_residual/` |
> | `src/multivariate_regimes/` | `src/stage2_residual/fields/` |
> | `src/summary_reports/`, `src/regime_plotting/` | `src/common/` |
> | `scripts/*_tu_hpc.sh` (SLURM wrappers) | `steps/<stage>/NN_*.py` |
>
> The figure and PDF-report layer referenced here is not part of this archive.

This repo builds **summer (MJJAS) weather regimes** from **ERA5 geopotential height at 500 hPa (Z500)** for the **Europe (Euro-Atlantic) domain**, using the standard pipeline: **anomalies → normalisation/weighting → PCA/EOF → clustering → composites**.

- **Inputs**: ERA5 Z500 (for testing: **2022 only**; later: multi-year), plus (later) ERA5 precipitation and convective proxies (e.g. CAPE, shear, PWAT)
- **Core methods**: day-of-year climatology, anomalies, standardisation (σ), area weighting \( \sqrt{\cos\phi} \), PCA/EOF, hierarchical clustering (choose \(k\)), final clustering (K-means / GMM)
- **Primary outputs**: regime labels per day, composite Z500 anomaly maps per regime, regime metrics (frequency/persistence), links to precipitation extremes
- **Optional extension**: select representative regime days → CM1 case studies

## Folder structure (this module)

All paths below are **relative to** `1_Detect_Weather_Patterns/`.

- **`data/`**: strict separation by processing stage (keeps the workflow reproducible and debuggable)
  - **`data/raw/`**: untouched downloads
    - **`data/raw/era5/z500/europe/`**: ERA5 Z500 downloaded already **subset to Europe + MJJAS** (and currently 2022 for testing)
  - **`data/preprocessed/`**: steps **2–6** (still gridded, plus derived arrays needed for PCA)
    - detrended/climatology/anomalies (Steps 2–3)
    - standardised, area-weighted, flattened `X_matrix` + mapping arrays (Steps 4–6)
    - **`data/preprocessed/europe/`**
  - **`data/processed/`**: steps **7–8**
    - PCA/EOF outputs and hierarchical clustering diagnostics
    - **`data/processed/europe/`**
- **`configs/`**: config files controlling region bounds, season definition, PCA/clustering choices
- **`src/`**: reusable code (core logic should live here rather than in notebooks)
- **`notebooks/`**: exploration + plotting only (no “single source of truth” logic)
- **`results/`**: final analysis outputs, separated by region
  - **`results/clustering/europe/`**: step **9**
  - **`results/regimes/europe/`**: steps **10–11**
- **`figures/`**: paper-ready plots
  - **`figures/europe/`**
- **`logs/`**: run logs for long pipeline steps (useful for HPC / reruns)

## Pipeline summary (end-to-end)

```text
ERA5 Z500
↓
Subset region + season (MJJAS)
↓
Day-of-year climatology (within MJJAS)
↓
Anomalies (remove climatological mean)
↓
Standardisation (divide by season-day σ, smoothed)
↓
Area weighting (√cos φ)
↓
Flatten (grid → vector)
↓
PCA / EOF reduction
↓
Hierarchical clustering (choose k)
↓
K-means / GMM final regimes
↓
Regime composites + metrics
↓
Link regimes to precipitation extremes
↓
(Optional) CM1 case studies
```

## Scientific framing (why this workflow)

This creates a **multi-scale circulation-to-convection framework** linking:
- **Synoptic dynamics** (Z500)
- **Statistical regimes** (clustering in reduced space)
- **Extreme weather** (precipitation extremes and environments)
- **Process understanding** (optional CM1 simulations)

## 1) Data acquisition & subsetting

### Inputs
- **Variable**: geopotential height at 500 hPa (Z500)
- **Dataset**: ERA5
- **Time span**: for testing **2022 only** (later: multi-year, e.g. 1980–2024)
- **Season**: May–September (MJJAS)
- **Temporal resolution**: daily mean preferred (or 12 UTC consistently)

### Domain (current)

Weather regimes are defined on these **analysis domains** (ERA5 Z500 MJJAS):

- **North America**: **180°–30°W**, **20°–80°N** (equivalently longitude −180° to −30° in −180…180° convention)
- **Euro-Atlantic**: **80°W–40°E**, **30°–90°N** (equivalently longitude −80° to 40°)

#### Bounding boxes (paper / analysis definitions)

| Region | Longitude | Latitude |
|--------|-----------|----------|
| North America | 180°–30°W | 20°–80°N |
| Euro-Atlantic | 80°W–40°E | 30°–90°N |

#### CDS download note

The Slurm helper `scripts/01_sbatch_cds_era5_z500_download.sh` uses a **default** CDS `area` (**N, W, S, E**) of **90, −80, 30, 40** unless you override `AREA_*` at submit time. Match that (or your chosen subset) to the domain you want in the downloaded NetCDFs; the pipeline otherwise uses the spatial extent of the raw files you provide.

CDS API guide (local vs cluster): **`README_CDS_API.md`**.

## 2) Climatology (mean summer state)

Best practice is a **day-of-year climatology** within MJJAS:

\[
\overline{Z}(\text{doy}, x, y)
\]

Steps (implementation used here):
- **Open raw ERA5 files** from `data/raw/era5/z500/europe/` (NetCDF)
  - Handle ERA5 coordinate conventions (`time` vs `valid_time`)
  - If a pressure dimension exists (e.g. `pressure_level`), select **500 hPa**
- **Aggregate to daily means** (safe if the file already contains one time per day, e.g. 12 UTC)
- **Select MJJAS** only (May–September)
- **Linear detrending** (option-controlled)
  - **Default (paper-like)**: remove the linear trend of the **daily cos(lat)-weighted domain mean** (scalar trend subtracted from the full field)
  - **Alternative**: remove a linear trend at each grid point (gridpoint detrending)
- **Raw MJJAS day-of-season climatology**
  - Map each date to an MJJAS **season-day** \(d \in [1,153]\) (May 1 → 1, …, Sep 30 → 153)
  - Compute the mean field \(\overline{Z}(d,x,y)\) across years for each season-day
- **Moderate cyclic smoothing (key summer adaptation)**
  - Apply a centered running-mean smoothing **with wrap-around within MJJAS** (cyclic boundary)
  - Recommended default: **31-day** window (±15 days)
  - Sensitivity set: **21 / 41 / 61-day** windows (to document robustness of EOFs/clusters)

Rationale:
- Winter-focused studies often use very long smoothing (e.g. 90 days) because DJF evolves weakly and low-frequency variability dominates.
- For **MJJAS**, the seasonal cycle evolves strongly; overly long smoothing risks distorting early/late summer anomalies. A **21–31 day** window is a good compromise: it suppresses synoptic noise while preserving intra-summer evolution.

## 3) Anomalies (remove seasonal cycle)

\[
Z'(t,x,y) = Z(t,x,y) - \overline{Z}(\text{doy},x,y)
\]

In this workflow, anomalies can be computed from the **detrended** field using either the **raw** or the **smoothed** MJJAS season-day climatology (option-controlled). The **paper-like** choice is to subtract the **smoothed** climatology:

\[
Z'(t,x,y) = Z_{\mathrm{detrended}}(t,x,y) - \overline{Z}_{\mathrm{smooth}}(d(t),x,y)
\]

Removes:
- Seasonal cycle
- Mean flow bias

### Implementation (this repo)

**Inputs (from Step 2)** written to `data/preprocessed/<region>/` (consistent naming includes window + method label):
- `02_z500_mjjas_detrended_wXX_<method>_<years>.nc` containing `z500_det(time, lat, lon)`
- `02_z500_mjjas_climatology_raw_wXX_<method>_<years>.nc` containing `z500_clim_raw(season_day, lat, lon)`
- `02_z500_mjjas_climatology_anomalies_wXX_<method>_<years>.nc` containing `z500_clim_smooth(season_day, lat, lon)` (and `z500_anom` pack)

**Output (Step 3)**:
- `data/preprocessed/<region>/03_Z500_anomaly_wXX_<method>_<years>.nc` containing `z500_anom(time, lat, lon)`

Implemented in `src/preprocessing_helpers.py` / `src/pipeline_prepro.py` (the original prototyping notebook has been retired).

### Critical diagnostics (do not skip)

- **Mean ~ 0**: map of `mean_t(Z')` should be near-zero with no large-scale structure.
- **Seasonal cycle removed**: domain-mean `Z'` should be flat around 0 (compare to domain-mean `Z` or `Z_detrended`).
- **Variance is physical**: variance map should highlight midlatitude variability maxima (storm-track regions).
- **Visual sanity check**: for a few random days plot detrended field, matched climatology, and anomaly.
- **No edge artifacts**: inspect early May and late September for discontinuities/spikes.

## 4) Standardisation / normalisation

Goal: ensure each grid point contributes comparably to EOF/PCA and clustering (otherwise regimes tend to “rediscover” variance hotspots like storm tracks).

In this workflow, σ is computed **as a function of MJJAS season-day**, then **smoothed in time** with the **same cyclic running-mean window** used for the climatology. Two σ normalisation variants are supported (option-controlled):
- **Field σ**: \(\sigma(d,x,y)\) (gridpoint standardisation)
- **Scalar σ** (paper-like): \(\sigma(d)\), computed as cos(lat)-weighted **area mean** of the σ field, then used as a scalar divisor for all grid points at each season-day

\[
Z''(t,x,y) = \frac{Z'(t,x,y)}{\sigma(d(t),x,y)}
\]

Where:
- \(d(t)\) is the MJJAS season-day index for time \(t\) (May 1 → 1, …, Sep 30 → 153)
- \(\sigma(d,x,y)\) is the **standard deviation across years** at each grid point for each season-day \(d\)
- \(\sigma(d,x,y)\) is **cyclically smoothed within MJJAS** (wrap-around), using the same window length as in Step 2

Important implementation details:
- **Minimum variance floor**: avoid dividing by near-zero values in low-variance regions (esp. subtropics / summer interiors) via
  \(\sigma \leftarrow \max(\sigma,\epsilon)\), with a typical \(\epsilon \approx 10\)–\(20\) m.
- **Order matters**: apply **standardisation first**, then apply **area weighting** \( \sqrt{\cos(\phi)} \) in Step 5 (do not swap the order).

## 5) Area weighting (critical before PCA)

After standardisation, apply latitude-dependent **area weighting** to correct for spherical grid geometry (meridian convergence) before EOF/PCA.

Purpose:
- Prevent artificial dominance of high-latitude grid cells due to grid convergence
- Ensure the covariance structure in EOF space reflects **equal-area physical contributions**

Method:

Grid-cell area scales with \( \cos(\phi) \), so weight by:

\[
w(\phi) = \sqrt{\cos(\phi)}
\]

Apply:

\[
Z'''(t,x,y) = Z''(t,x,y)\, w(\phi)
\]

Implementation details:
- Weighting is applied **after anomaly standardisation** (Step 4) and **before EOF/PCA**
- Broadcasting is performed over spatial dimensions (lat × lon)
- Do **not** use full \( \cos(\phi) \) (overcorrects variance); use \( \sqrt{\cos(\phi)} \)

Output of Step 5:
- `Z_weighted(time, lat, lon)` ready to be flattened for EOF/PCA

![Area-weighting schematic](../assets/image_1776878410571_0.png)

## 6) Flatten the spatial field (grid → feature vector)

Convert:
- \([time, lat, lon] \rightarrow [time, features]\)

So each day becomes one high-dimensional vector of Z500 values across the domain.

In practice (EOF-ready matrix):
- `X_matrix(time, space)` where `space = lat × lon`
- **Deterministic ordering (critical)**: flatten with **lat as the outer index** and **lon as the inner index** (lon varies fastest). This ordering must be identical for training, projection, and reconstruction.
- **Reconstruction mapping (required)**: store `latitude_space(space)` and `longitude_space(space)` so each feature index `space` can be mapped back to a unique `(lat, lon)` grid point when reshaping EOFs/regimes back to maps.

## 7) Dimensionality reduction (EOF / PCA)

Fit PCA on daily vectors and retain enough components to explain ~80–95% variance (often 10–30 PCs).

Output per day:

```text
[PC1, PC2, PC3, ...]
```

## 8) Hierarchical clustering (exploration: choose k)

- **Method**: Ward linkage (on PCA scores)
- **Output**: dendrogram + candidate number of regimes
- **Typical range**: \(k \approx 4\)–\(8\)

Reference: `https://www.mltut.com/hierarchical-clustering-in-python-step-by-step-complete-guide/`

## Pipeline switches (paper-like vs default variants)

The preprocessing pipeline (`src/pipeline_prepro.py` → `src/preprocessing_helpers.py`) exposes these switches:

- **`--detrend-method {gridpoint, domain_mean}`**
  - `domain_mean` matches the “area-averaged detrending” approach (scalar trend removed from the full field)
- **`--climatology-subtract {raw, smooth}`**
  - `smooth` subtracts the running-mean climatology used in many regimes papers
- **`--sigma-mode {field, scalar}`**
  - `scalar` matches workflows that normalise by a seasonally varying scalar σ(day) to avoid seasonal weighting in clustering

### Recommended “paper-like” Slurm run scripts (no overwrite)

- **Europe**: `scripts/03_run_pipeline_europe_compare_options.sh`
- **North America**: `scripts/03_run_pipeline_northamerica_compare_options.sh`

Defaults in those scripts:
- `--smooth-window 21`
- `--detrend-method domain_mean`
- `--climatology-subtract smooth`
- `--sigma-mode scalar`

### Canonical Slurm submit (Levante)

Use these from the cluster directory that contains the scripts (e.g. `cd /work/bb1546/detect_weather/scripts`). Paths below are relative to `1_Detect_Weather_Patterns/` unless you keep the same layout on Levante.

**North America — reference run (B0, 11-day window, steps 2–8, paper-like options):**

```bash
sbatch --export=ALL,REGION=northamerica,STOP_AFTER_STEP=8,BBOX_STEP=B0,SMOOTH_WINDOW=11,DETREND_METHOD=domain_mean,CLIM_SUBTRACT=smooth,SIGMA_MODE=scalar \
  03_run_pipeline_northamerica_compare_options.sh
```

`REGION=northamerica` matches the script default but is kept explicit here so this line is self-contained. Change `BBOX_STEP` (e.g. `B1`, `B2`) or `SMOOTH_WINDOW` (must be **odd**) as needed. If the **compute** partition walltime is tight for a large bbox, run Step 2 in one job (`STOP_AFTER_STEP=2`) and then `STOP_AFTER_STEP=8` in a second job so Step 2 is skipped once outputs exist (see comments in the shell scripts).

**Europe — same pattern (raw years span + bbox step):**

```bash
sbatch --export=ALL,REGION=europe,ERA5_RAW_YEARS_SPAN=1979-2025,STOP_AFTER_STEP=8,BBOX_STEP=B0,SMOOTH_WINDOW=11,DETREND_METHOD=domain_mean,CLIM_SUBTRACT=smooth,SIGMA_MODE=scalar \
  03_run_pipeline_europe_compare_options.sh
```

## Method provenance / citations

- **Lee et al.** (weather regimes / domain choices & methodology): DOI `10.1175/JCLI-D-23-0214.1`
- **Messori & Dorrington** (related regime preprocessing choices / framing): DOI `10.1029/2023GL104696`

## 9) Final clustering (regime definition)

Run a final clustering algorithm on PCA scores:
- **Option A (standard)**: K-means (`KMeans(n_clusters=k)`)
- **Option B (probabilistic)**: Gaussian Mixture Model (GMM)

Output:
- Each day assigned to a **regime label**

## 10) Regime reconstruction (back to physical space)

For each regime:
- Select all days assigned to that regime
- Average the **original Z500 anomalies** (typically \(Z'\); optionally also show standardised anomalies)

Output:
- **Composite anomaly maps** (“weather patterns”)

Implementation (this repo):
- Notebook: `src/z_pipeline/z_10_11_regime_reconstruction_characterisation.ipynb`
- Reads:
  - Step 9 output: `data/processed/europe/Z500_regimes_kmeans_*.nc` (`regime_labels`)
  - Step 3 output: `data/processed/europe/Z500_anomaly.nc` (`z500_anom`)
  - Optional Step 4–6 output: `data/processed/europe/Z500_normalised_*.nc` (`z500_norm`, `z500_weighted`)
- Writes:
  - `data/processed/europe/Z500_regime_reconstruction_characterisation_kmeans_*.nc`
    - `regime_composite_anom(regime,lat,lon)` (+ optional `regime_composite_norm`, `regime_composite_weighted`)

## 11) Regime characterisation

Core metrics per regime:
- **Frequency** (% of days)
- **Persistence** (mean duration of consecutive days)
- **Seasonal distribution** (within MJJAS)

Implementation (this repo):
- Computed in `src/z_pipeline/z_10_11_regime_reconstruction_characterisation.ipynb`
- Adds:
  - `frequency(regime)`
  - run-length table: `run_regime(episode)`, `run_length(episode)`
  - `monthly_fraction(month, regime)` for May–Sep
  - transition diagnostics: `transition_counts`, `transition_prob`

## 12) Link regimes to extreme events (key application)

Define extremes (e.g. top 5% or 1% precipitation days from ERA5), then compute:

\[
P(\text{extreme} \mid \text{regime})
\]

Optional environmental diagnostics on extreme days:
- CAPE
- vertical wind shear
- moisture (PWAT)

More detail (including convective filtering, Germany observation validation, and case-study selection) is documented in:
- `README_EPE.md`
- `README_CDS_API.md` — Copernicus CDS API (local vs cluster; standalone, uses `YOUR_PROJECT` placeholders)
- `legacy/multivariate_pipeline/README_Multivariate.md` — archived standalone multivariate regime pipeline and VO850/TCWV CDS jobs (its underlying library, `src/multivariate_regimes/`, is still live and used by the residual_regimes pipeline)
- `README_bbox_steps.md` — bbox step IDs for preprocessing



## 14) Optional extension: CM1 case studies

Use representative regime days to:
- Initialise CM1 with thermodynamic profiles, wind shear, and moisture
- Study convective structure, rainfall efficiency, and storm organisation