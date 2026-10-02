# Domain-Percentile Threshold Selection: Complete Guide

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

## TL;DR

**The 7 thresholds being tested:**
- 0.999 (0.1% of days ~ 8 days total)
- 0.995 (0.5% of days ~ 36 days)
- 0.99 (1.0% of days ~ 72 days)
- 0.98 (2.0% of days ~ 145 days)
- 0.975 (2.5% of days ~ 181 days)
- 0.95 (5.0% of days ~ 362 days)
- **0.90 (10.0% of days ~ 724 days) ← PRODUCTION (updated 2026-08-03)**

**Plus 3 thresholds DENSER than production (added 2026-08), to defend 0.90 against "why not go
even looser" rather than only testing the sparser end above:**
- 0.86 (~14% of days)
- 0.82 (~18% of days)
- 0.79 (~21% of days)

See `results/summary_reports/regime_boxplot_domain_pctl_sensitivity*.csv`
(`src/epe/epe_04_boxplot_domain_pctl_sensitivity.py`) for how these are used to check whether
going denser than 0.90 buys any additional stability in the Z500-regime-conditional statistics
(duration/intensity/connectivity/spatial/excess), on top of the Stage-2 logistic-regression
comparison below.

**To run the test on both regions (Europe & North America):**
```bash
sbatch scripts/epe_domain_pctl_sensitivity_test_tu_hpc.sh
```

This will generate:
- TEST files for all 7 thresholds in both regions (14 total)
- Appendix table: `results/summary_reports/appendix_domain_pctl_sensitivity.csv`
- Pretty-printed summary for review

**Selection criterion:** Pick the threshold giving **60–100 flagged events per regime** (typically 0.95 with ~72 events/regime). **Production is now pinned at 0.90** (~145 events/regime) as of 2026-08-03 — outside this range, but confirmed as the more stable choice by a direct Stage-2 robustness comparison (see "Decision: 0.90 vs 0.95" below): the 60–100/regime target is a rule of thumb for statistical power, not a guarantee of CV stability, and 0.90 wins decisively on the latter.

### Decision: 0.90 vs 0.95 (confirmed 2026-08-06)

`domain_pctl_q = 0.90` is the final production choice. Rationale: `rr_021_stage2_LogRegr_Robustness` (block-bootstrap coefficient stability + blocked-CV/leave-one-year-out AUC) was run for every bbox (B0–B5) in both regions at both thresholds — 55 Z500-regime fits each, reproducible via `python3 src/residual_regimes/rr021_domain_pctl_sensitivity_summary.py`. Full-sample comparison:

| metric | q=0.90 | q=0.95 |
|---|---:|---:|
| mean events/regime | 157.9 | 79.0 |
| mean LOYO years used (of 47) | 32.0 | 19.9 |
| mean \|blocked-CV gap\| | 0.018 | 0.030 |
| regimes flagged BLOCKED-CV GAP | 1/55 (2%) | 11/55 (20%) |
| regimes verdict = ROBUST | 15/55 (27%) | 9/55 (16%) |

At 0.95, over half the study years contribute zero extreme days to the average regime (19.9/47 years used in LOYO), and 1 in 5 regime-fits show a >0.05 AUC swing between CV schemes. At 0.90 that instability rate is 10x lower. The narrower 60–100-events/regime target above optimizes for a *different* thing (avoiding "too many events dilutes signal") and doesn't account for per-regime CV stability once events are split across 4–7 Z500 regimes — 0.90's higher event count wins that tradeoff in practice. Caveat: "LOW SIGN-STABILITY" (coefficient sign flips under block-bootstrap) stays common at both thresholds — a separate, threshold-independent limitation worth noting in the paper's limitations section regardless of which q is used.

---

## Why This Matters

### The Problem

The old point-wise q999 threshold (applied naively across ~10,000 cell-hours) inflated the base rate to **17–54%** of days. The new domain-day approach is fundamentally different and requires its own calibration.

### The Solution

A properly-calibrated **continuous severity index** (`domain_extreme_pctl`) combined with a **tunable percentile threshold** (`domain_pctl_q`), selected based on downstream statistical power needs, not inherited from the old broken approach.

---

## How Extremes Are Selected Now

### The 5-Step Pipeline

**Step 1: Per-Cell Normalization**
- Each grid cell's daily convective precipitation ranked against its own 1979–2025 history
- Handles zero-inflation using average-rank ties
- Output: `cell_pctl_grid_day[lat,lon,day] ∈ (0,1]`

**Step 2: Domain Aggregation (Lexmax)**
- Combine per-cell percentiles into a single domain-day metric
- **Lexmax method**: Rank days by their single most-extreme cell, break ties with 2nd, 3rd, ... most-extreme cells
- **Why?** Fixes the pathology where unrelated events get identical scores
- Output: `domain_stat_raw[day]`

**Step 3: Recalibration**
- Rank `domain_stat_raw` against its own distribution across the full record
- Output: `domain_extreme_pctl[day] ∈ (0,1]` — continuous severity index
- By construction: top 5% of days have `domain_extreme_pctl ≥ 0.95`

**Step 4: Threshold Selection**
- Choose `domain_pctl_q` — a free design parameter (not mechanically tied to point q999)
- Selected based on **downstream statistical power needs** (target: 60–100 flagged events per Z500 regime)

**Step 5: Binary Label**
- `flagged_domain_pctl = (domain_extreme_pctl ≥ domain_pctl_q)`
- Example: `domain_pctl_q = 0.95` flags ~362 days (~5%) in a ~7,200-day MJJAS record

### What Changed from the Old Approach

| Problem (Old) | Solution (New) |
|---|---|
| OR-ing across ~10k cell-hours → 17–54% base rate | Proper aggregation → corrected base rate |
| Many days tied at identical max score | Lexmax ranks all days uniquely |
| Zero-inflation not handled properly | Average-rank ties in percentile calc |
| Binary label throws away severity info | Continuous `domain_extreme_pctl` available |
| Threshold inherited from unrelated point q999 | Free choice tuned to statistical power |

---

## Regional Symmetry Requirement

**Important:** Use the **same `domain_pctl_q` in both regions**.

Why?
- The threshold is a **percentile**, so it's scale-invariant (5% of days flagged in each region)
- Different absolute day counts are expected and correct (Europe vs North America have different sizes)
- Using the same threshold makes "Europe vs NA" comparisons valid
- If you use different thresholds per region, you're asking different questions in each region (breaks comparability)

---

## Decision Criteria

### Primary: Statistical Power for Regime-Conditional AUC

For **cross-validated logistic regression** to estimate stable regime-conditional AUC:
- **Too sparse** (<30 events/regime): high noise, overfitting risk → ❌
- **Optimal** (60–100 events/regime): stable AUC ± 0.03 → ✅
- **Too many** (>500 events/regime): dilutes signal, detects everything → ⚠️

### Secondary: Domain Coherence

Spot-check flagged days in regime composites to ensure they show:
- Wide spatial extent (not just one grid cell)
- High CAPE and organized convection
- Not just random noise

### Tertiary: Consistency

Pick the same threshold for both regions.

---

## Expected Results & Per-Regime Counts

Assuming **5 Z500 regimes**, divide total flagged days by 5:

| domain_pctl_q | Total Days | Per-Regime Events | Assessment |
|---|---|---|---|
| 0.999 | ~8 | ~1.6 | ❌ Too sparse |
| 0.995 | ~36 | ~7.2 | ❌ Too sparse |
| 0.99 | ~72 | ~14.4 | ⚠️ Marginal |
| **0.98** | **~145** | **~29** | ✅ Conservative |
| **0.975** | **~181** | **~36** | ✅ Good |
| 0.95 | ~362 | ~72 | ✅✅ In 60–100 target range |
| **0.90** | **~724** | **~145** | **⚠️ Loose, but PRODUCTION (updated 2026-08-03)** |

0.95 gives ~72 events per regime, right in the middle of the 60–100 target range. **0.90 is nonetheless the current pinned production choice** (~145 events/regime) — confirmed by the Stage-2 robustness comparison in "Decision: 0.90 vs 0.95" above (more LOYO years used, far fewer CV-instability caveats).

---

## How to Run the Test

### Step 1: Submit HPC Job (Both Regions)

```bash
cd /Users/marie-christineckert/Nextcloud_2/TU/phd_work/Paper3_canada/1_Detect_Weather_Patterns

sbatch scripts/epe_domain_pctl_sensitivity_test_tu_hpc.sh
```

This script:
- Runs `epe_01b_detect_convective_hourly_duration_test.py` 14 times (7 thresholds × 2 regions)
- Generates TEST NetCDF files in `data/processed/{region}/extreme_precip_events/`
- Creates appendix table in `results/summary_reports/appendix_domain_pctl_sensitivity.csv`
- Typical runtime: 8–16 hours

### Step 2: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Check for generated files
ls data/processed/europe/extreme_precip_events/convective_duration_TEST_*.nc | wc -l
ls data/processed/northamerica/extreme_precip_events/convective_duration_TEST_*.nc | wc -l
```

### Step 3: Review Appendix Table

Once complete, view the results:

```bash
# Pretty-printed table
cat results/summary_reports/appendix_domain_pctl_sensitivity.csv | column -t -s,

# Or in Python
python << 'EOF'
import pandas as pd

df = pd.read_csv('results/summary_reports/appendix_domain_pctl_sensitivity.csv')
print("\nRAW TABLE:")
print(df.to_string(index=False))

print("\n\nPER-REGIME ANALYSIS (assuming 5 Z500 regimes):")
df['Europe_Per_Regime'] = df['Europe_Days'] / 5
df['NA_Per_Regime'] = df['NorthAmerica_Days'] / 5

analysis = df[['domain_pctl_q', 'Europe_Days', 'Europe_Per_Regime', 
               'NorthAmerica_Days', 'NA_Per_Regime']].copy()
print(analysis.to_string(index=False))

print("\n\nRECOMMENDATION:")
target_min, target_max = 60, 100
for _, row in analysis.iterrows():
    eu_ok = target_min <= row['Europe_Per_Regime'] <= target_max
    na_ok = target_min <= row['NA_Per_Regime'] <= target_max
    if eu_ok and na_ok:
        print(f"✓ domain_pctl_q = {row['domain_pctl_q']:.2f} → {row['Europe_Per_Regime']:.1f} / {row['NA_Per_Regime']:.1f} per regime")
EOF
```

### Step 4: Select Threshold

Pick the row where both regions have **60–100 events per regime**. Typically this is **0.95** — production is pinned at **0.90** instead, per the Stage-2 robustness comparison in "Decision: 0.90 vs 0.95" above (see Step 5).

### Step 5: Use in Production

Point downstream scripts to the pinned production TEST file (domain_pctl_q = 0.90):

```bash
# Find the exact TEST file for the pinned threshold
ls data/processed/europe/extreme_precip_events/convective_duration_TEST_*domainQ90*.nc

# domain_pctl_q = 0.90 (pinned production, 2026-08-03)
EXTREME_LABELS_PATH="data/processed/europe/extreme_precip_events/convective_duration_TEST_europe_..._domainQ900.nc"
EXTREME_VAR="flagged_domain_pctl"

# Run regime analysis with the new threshold
sbatch --export=ALL,EXTREME_LABELS_PATH=$EXTREME_LABELS_PATH,EXTREME_VAR=$EXTREME_VAR \
  scripts/rr_01_feature_relevance_tu_hpc.sh

# Or for diagnostics
sbatch --export=ALL,EXTREME_LABELS_PATH=$EXTREME_LABELS_PATH,EXTREME_VAR=$EXTREME_VAR \
  scripts/rr_residual_extreme_signal_diag_tu_hpc_LOGISTIC.sh
```

---

## For the Paper Appendix (ipynb Integration)

After the test completes, the appendix table generator produces both:

1. **CSV file** (for Excel/paper): `results/summary_reports/appendix_domain_pctl_sensitivity.csv`
2. **Python DataFrame** (for notebook): Read directly in `src/01_paper_figures/4_Appendix.ipynb`

### In your notebook, use:

```python
import pandas as pd

# Read the sensitivity table
df = pd.read_csv('../../results/summary_reports/appendix_domain_pctl_sensitivity.csv')

# Create publication-ready table
print("\nAppendix Table: Sensitivity of Domain-Day Extreme Label to Percentile Threshold\n")
print(df.to_string(index=False))

# Add per-regime context
df['Europe_Per_Regime'] = df['Europe_Days'] / 5  # Adjust n_regimes as needed
df['NA_Per_Regime'] = df['NorthAmerica_Days'] / 5

# Format for LaTeX if needed
df_latex = df[['domain_pctl_q', 'Europe_Days', 'Europe_%', 
               'NorthAmerica_Days', 'NorthAmerica_%']].copy()
print(df_latex.to_latex(index=False))
```

---

## Calibration Guarantees

✓ By construction, `domain_extreme_pctl` is uniformly distributed (0,1]
✓ Percentages always match `domain_pctl_q` (e.g., 0.95 flags exactly 5% of days)
✓ No artificial ties (lexmax ensures uniqueness)
✓ Proper zero-inflation handling

---

## Validation Checklist

Before finalizing your choice, verify:

- [ ] TEST files exist for all 7 thresholds in both regions
- [ ] Appendix table generated successfully (`appendix_domain_pctl_sensitivity.csv`)
- [ ] Percentages match domain_pctl_q values (e.g., 0.95 → ~5%)
- [ ] Per-regime counts fall in 60–100 range for your selected threshold
- [ ] Flagged days appear spatially organized (check composites)
- [ ] Same threshold used in both regions

---

## Paper Methods Text (Template)

> To identify days with widespread, intense convective activity, we applied a domain-scale aggregation method. For each grid cell and day, we computed the percentile rank of that cell's daily convective precipitation relative to its own 1979–2025 history, handling zero-inflation with average-rank ties. We combined per-cell percentiles into a domain-day statistic using lexicographic ranking (lexmax): sorting each day's cell percentiles and ranking days by this tuple to avoid artificial score collapsing. We then recalibrated this domain-day statistic against its own distribution to obtain a continuous severity index (*domain_extreme_pctl* ∈ (0,1]). We flagged days in the top 10% of this index (*domain_pctl_q* = 0.90), yielding 724 extreme days across the ~7,200-day MJJAS record for both Europe and North America (~145 flagged days per regime across 5 Z500 regimes). The continuous index and domain-based aggregation improve upon the previous point-wise q999 approach, which inflated the base rate to 17–54% due to OR-ing across thousands of cell-hours.
>
> We selected *domain_pctl_q* = 0.90 over the narrower 60–100-events-per-regime target (favoring 0.95) based on a direct robustness comparison of the downstream Stage-2 per-regime logistic models: across all bounding boxes and both regions (55 Z500-regime fits per threshold), 0.90 yielded more than twice the extreme-day count per regime, leave-one-year-out cross-validation using 60% more study years on average (32 vs. 20 of 47), and a tenfold reduction in regime-fits showing a >0.05 AUC gap between random-fold and blocked/leave-one-year-out cross-validation (2% vs. 20% of fits). We therefore judged CV stability of the downstream model to be a more decisive criterion than the raw per-regime event-count target.

---

## Troubleshooting

### "No TEST files found"
```bash
# Check job status
squeue -u $USER

# Check error log
tail -20 slurm-*.out

# Verify hourly data exists
ls data/raw/era5/extreme_precip_events/europe/era5_single_levels_*.nc | head -3
```

### "Appendix table percentages look wrong"
- Check that `domain_extreme_pctl` has no repeated values (verify lexmax worked)
- Re-run: `PYTHONPATH=./src python src/epe/epe_01b_detect_convective_hourly_duration_test.py --domain-agg lexmax`

### "I want to test one more custom threshold (e.g., 0.92)"
```bash
PYTHONPATH=./src python src/epe/epe_01b_detect_convective_hourly_duration_test.py \
  --region-label europe \
  --timezone Europe/Berlin \
  --domain-pctl-q 0.92 \
  --domain-agg lexmax

# Regenerate table (auto-detects new file)
PYTHONPATH=./src python src/epe/epe_03_07_appendix_domain_pctl_sensitivity.py --regions europe northamerica
```

---

## Key Files

- **HPC submission script**: `scripts/epe_domain_pctl_sensitivity_test_tu_hpc.sh`
- **Test generator**: `src/epe/epe_01b_detect_convective_hourly_duration_test.py`
- **Appendix table generator**: `src/epe/epe_03_07_appendix_domain_pctl_sensitivity.py`
- **Notebook integration**: `src/01_paper_figures/4_Appendix.ipynb`
- **Output**: `results/summary_reports/appendix_domain_pctl_sensitivity.csv`

---

## Quick Reference Decision Tree

```
Do you want to run threshold sensitivity tests?
├─ YES
│  └─ sbatch scripts/epe_domain_pctl_sensitivity_test_tu_hpc.sh
│     (Wait 8–16 hours)
│     Review results/summary_reports/appendix_domain_pctl_sensitivity.csv
│     Pick threshold giving 60–100 per-regime events
│
└─ NO
   └─ Use pinned production default: domain_pctl_q = 0.90 (724 days, ~145 per regime)
```

---

## Done

You now have:
1. ✅ Complete overview of the 7 thresholds (0.999 → 0.90)
2. ✅ Single command to run tests on both regions
3. ✅ Clear selection criteria (60–100 events/regime)
4. ✅ Template for paper methods and appendix
5. ✅ Instructions for downstream integration

**Next:** `sbatch scripts/epe_domain_pctl_sensitivity_test_tu_hpc.sh`
