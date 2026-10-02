# Feature Relevance Diagnostic (rr_01)

## Overview

The feature relevance diagnostic evaluates which atmospheric fields are most useful for predicting extreme convective precipitation events. It uses two complementary evaluation approaches to inform field selection for the residual regime clustering pipeline.

## Two Evaluation Approaches

### Option A: Residual-Based Importance

**Question:** "Which fields explain what Z500 couldn't?"

**Methodology:**
1. Train a model with **Z500 only** → predict extreme events → AUC = 0.75
2. Get prediction errors (residuals) from that model
3. For each other field, test: **Can it predict those residuals?**
   - If CAPE can explain the residuals → CAPE captures structure Z500 missed
   - If W700 can explain the residuals → W700 captures structure Z500 missed

**Why it matters:**
- Directly aligned with residual clustering concept
- Shows which fields have signal in the Z500 residuals
- Important for understanding what the second clustering stage (residual regimes) will capture

**Output metric:** `delta_auc_residual`
- Measures how much each field improves over random guessing (0.5) when predicting Z500 residuals
- Higher value = field explains more residual structure

**Example interpretation:**
```
Field        Residual AUC   ΔAU (vs random)
-----        -----------    ---------------
CAPE         0.68           +0.18       ← strong residual signal
W700         0.62           +0.12       ← moderate residual signal
TCW          0.54           +0.04       ← weak residual signal
```

### Option B: Hierarchical/Additive Importance

**Question:** "Does CAPE add value when combined with Z500?"

**Methodology:**
1. Train model with **all fields** together → Baseline AUC = 0.82
2. Train model with **all fields except CAPE** → AUC = 0.79
3. Compare: ΔAUC = 0.82 - 0.79 = **+0.03**
4. Repeat for each field

**Why it matters:**
- Shows total predictive contribution when combined with other fields
- Standard feature importance metric (leave-one-out)
- Useful for understanding global predictive power

**Output metric:** `delta_auc`
- Measures how much the baseline AUC drops when that field is removed
- Higher value = field is more important for overall prediction

**Example interpretation:**
```
Field        AUC w/o field   ΔAU (global)
-----        ------          -----------
CAPE         0.8185          +0.0247     ← very predictive
W700         0.8331          +0.0100     ← moderately predictive
Z500         0.8426          +0.0006     ← redundant (already captured)
```

## Key Differences

| Aspect | Option A (Residual) | Option B (Hierarchical) |
|--------|---------------------|--------------------------|
| Baseline | Z500 alone | All fields together |
| Evaluates | Residual signal | Global predictive power |
| Question | "What does Z500 miss?" | "What predicts extremes best?" |
| Use case | Residual clustering | Overall field selection |
| Metric | AUC predicting residuals | Leave-one-out ΔAUC |

## Practical Interpretation Example

Imagine Z500 predicts 80% of extreme events correctly but misses some cases.

**Option A** asks:
- "In cases where Z500 failed, can CAPE explain the pattern?"
- If yes → CAPE has residual structure to explain → useful for residual clustering

**Option B** asks:
- "When CAPE is removed from the full model, does prediction quality drop?"
- If yes → CAPE is globally important → keep it in the full analysis

## Using Both Approaches

1. **Start with Option B (hierarchical)** to identify globally important fields
2. **Then check Option A (residual)** to see if those fields have structure in Z500 residuals
3. Fields that are important in BOTH approaches are strongest candidates for residual clustering

**Example:**
- CAPE: high in both → excellent for residual clustering
- W700: moderate in both → good for residual clustering  
- Z500: low in both → minimal added value
- TCW: high residual only → captures Z500 misses but globally weak predictor

## Running the Diagnostic

```bash
sbatch --export="ALL,\
  REGION=europe,\
  BBOX_STEP=B5,\
  EXTREME_LABELS_PATH=/path/to/convective_duration_...nc,\
  FIELDS=z500,vo850,tcwv,deg0l,cape,w700,rh700" \
  rr_01_feature_relevance_tu_hpc.sh
```

## Outputs

All filenames include the rr_01 **mode** tag (`bbox` or `regime`) so the two
approaches never overwrite each other:

**Bbox mode** (`MODE=bbox`, default — global selection per bounding box):

- `feature_relevance_sequential_<region>_<bbox>_bbox_optionB.csv` — primary for rr_02
- `feature_relevance_residual_<region>_<bbox>_bbox_optionA.csv`
- `feature_relevance_hierarchical_<region>_<bbox>_bbox.csv`
- `feature_relevance_bootstrap_<region>_<bbox>_bbox_stability.csv` (when `N_BOOTSTRAP>0`)

**Regime mode** (`MODE=regime` — per Z500 regime within bbox):

- `feature_relevance_hierarchical_<region>_<bbox>_regime.csv` — global Option C only
- `per_regime/feature_relevance_sequential_<region>_<bbox>_regime_R<N>_optionB.csv`
- `per_regime/feature_relevance_residual_<region>_<bbox>_regime_R<N>_optionA.csv`

Legacy bbox files without the `_bbox_` suffix are still read by
`write_field_selection.py` and plotting notebooks.

## Interpreting the CSV Results

### hierarchical CSV
```
field        baseline_auc  auc_without  delta_auc  evaluation_type
z500         0.8432        0.8426       0.0006     hierarchical
vo850        0.8432        0.8427       0.0005     hierarchical
cape         0.8432        0.8185       0.0247     hierarchical
w700         0.8432        0.8331       0.0100     hierarchical
```

**Action:** Keep fields with delta_auc > 0.01

### residual CSV
```
field        z500_auc  residual_auc  delta_auc_residual  evaluation_type
z500         0.75      NaN           NaN                 residual
cape         0.75      0.68          +0.18               residual
w700         0.75      0.62          +0.12               residual
vo850        0.75      0.54          +0.04               residual
```

**Action:** Fields with delta_auc_residual > 0.01 explain meaningful Z500 residuals

## Recommendations for Field Selection

1. **For residual clustering:** Prioritize fields high in BOTH Option A and Option B
2. **When unsure:** Use fields with delta_auc > 0.01 in hierarchical approach
3. **When adding new fields:** Check both metrics before including

## References

- Logistic regression with 5-fold stratified cross-validation
- ROC AUC: measures model's ability to rank extreme days higher than non-extreme days
- Balanced class weights: accounts for imbalanced classes (few extreme days)
