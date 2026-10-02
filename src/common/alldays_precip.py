"""Shared "all days, no extreme-day filter" daily precipitation loader plus a couple of simple
significance-test helpers, used by both ``export_regime_z500_statistics_csv.py`` and
``export_regime_rr_statistics_csv.py`` (via ``stage2_residual.plot_rr_statistics``) to add a
non-circular precip-amount baseline alongside the existing extreme-day-conditioned statistics.

Motivation: Stage-2 RR sub-regimes are trained on the domain-pctl-0.90 extreme-day label (see
README_DOMAIN_PCTL_Q.md), so any RR-sub-regime comparison that reuses that same label (EPE
occurrence, n_extreme) is close to circular -- the sub-regimes differ "by construction" on that
measure. This module supplies the domain-aggregated daily precip total for EVERY MJJAS day (from
``era5_daily_tpSum_...nc``, written by ``epe_01_aggregate_daily.py`` *before* any extreme-day
selection happens) as the non-circular alternative -- promoted from the
``scripts/adhoc_alldays_precip_test.py`` experiment (2026-08).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# Same mapping export_regime_z500_statistics_csv.py already pins: the AOI a precip file is
# written for (Germany / southern Ontario) is fixed and does NOT vary with bbox_step -- bbox_step
# only changes which Z500 circulation domain the regimes were clustered over.
STUDY_AREA_BY_REGION = {"europe": "germany", "northamerica": "ontario_south"}


def load_daily_tp_alldays(module_root: Path, region: str, study_area: str | None = None) -> pd.DataFrame:
    """Domain-mean/median/max daily total precip (mm), every MJJAS day, no selection at all.

    Index: calendar day (normalized, datetime64[ns]). Columns: ``tp_mean_mm``, ``tp_median_mm``,
    ``tp_max_mm`` -- spatial aggregations of ``tp_daily_sum`` over the region's fixed AOI grid.
    """
    study_area = study_area or STUDY_AREA_BY_REGION.get(region)
    if not study_area:
        raise ValueError(f"No default study_area for region={region!r}; pass study_area explicitly.")
    d = Path(module_root) / "data" / "processed" / study_area / "extreme_precip_events"
    hits = sorted(d.glob("era5_daily_tpSum_tpMax_cpSum_capeMax_*.nc"))
    if not hits:
        raise FileNotFoundError(
            f"No era5_daily_tpSum_*.nc under {d} -- run/copy epe_01_aggregate_daily.py's output "
            "for this study area first (see the README_ResidualRegimes.md all-days-precip note)."
        )
    path = hits[-1]
    with xr.open_dataset(path) as ds:
        tp = ds["tp_daily_sum"].astype("float64")
        spatial_dims = [dim for dim in tp.dims if dim != "time"]
        mean_mm = tp.mean(dim=spatial_dims, skipna=True) * 1000.0  # ERA5 tp is metres
        median_mm = tp.median(dim=spatial_dims, skipna=True) * 1000.0
        max_mm = tp.max(dim=spatial_dims, skipna=True) * 1000.0
        days = pd.DatetimeIndex(pd.to_datetime(tp["time"].values)).normalize()
    out = pd.DataFrame(
        {"tp_mean_mm": mean_mm.values, "tp_median_mm": median_mm.values, "tp_max_mm": max_mm.values},
        index=days,
    )
    return out[~out.index.duplicated(keep="first")]


def two_group_significance(values_a, values_b) -> tuple[str, float, float]:
    """Mann-Whitney U test (two independent groups, no normality assumption) -- the standard
    choice for "are bin1 and bin2 significantly different" on skewed precip data. Returns
    ``(test_name, statistic, p_value)``; NaNs are dropped first. NaN statistic/p if either group
    is empty after dropping NaNs.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return "mannwhitneyu", float("nan"), float("nan")
    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return "mannwhitneyu", float(stat), float(p)


def omnibus_significance(groups: list) -> tuple[str, float, float]:
    """Kruskal-Wallis omnibus test across >2 groups -- generalizes ``two_group_significance``
    for ``--n-bins`` > 2 (production default is 2, where the pairwise test above is used
    instead). Returns ``(test_name, statistic, p_value)``."""
    clean = [np.asarray(g, dtype=float) for g in groups]
    clean = [g[np.isfinite(g)] for g in clean]
    clean = [g for g in clean if g.size > 0]
    if len(clean) < 2:
        return "kruskal", float("nan"), float("nan")
    stat, p = stats.kruskal(*clean)
    return "kruskal", float(stat), float(p)
