"""Matplotlib figure builders for EPE.4 (extreme regimes) PDF / summary exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from common.boxplot_style import (
    PDF_FS_PANEL_AXIS,
    PDF_FS_PANEL_LEGEND,
    PDF_FS_PANEL_SUPTITLE,
    PDF_FS_PANEL_TICK,
    PDF_FS_PANEL_TITLE,
    regime_tab10_colors,
    style_boxplot,
)

_MJJAS_MONTHS = (5, 6, 7, 8, 9)
_MJJAS_LABELS = ("May", "Jun", "Jul", "Aug", "Sep")
_STRUCTURE8 = np.ones((3, 3), dtype=int)

PanelCIntensityMode = Literal["hourly_max", "daily_max"]


def build_connectivity_n_components_from_duration(dur_c: xr.Dataset) -> xr.DataArray | None:
    """Daily ``n_components`` (8-neighbour CC), matching epe_04 notebook section (C)."""
    try:
        from scipy import ndimage as ndi
    except Exception:
        return None

    n_t = int(dur_c.sizes["time"])
    coords = {"time": dur_c["time"].values}
    n_components = np.full((n_t,), np.nan, dtype=float)

    if "E_conv_hour" in dur_c:
        from collections import defaultdict

        e = dur_c["E_conv_hour"].astype(float)
        spatial_dims = tuple(d for d in e.dims if d != "time_hour")
        if len(spatial_dims) != 2:
            return None
        arr_h = np.nan_to_num(e.transpose("time_hour", *spatial_dims).values, nan=0.0)
        th = dur_c["time_hour"].values.astype("datetime64[h]")
        day_start = int(dur_c.attrs.get("day_start_hour_utc", 0))
        day_per_hour = (th - np.timedelta64(day_start, "h")).astype("datetime64[D]")
        days = dur_c["day"].values.astype("datetime64[D]")
        hours_for_day: dict = defaultdict(list)
        for hi in range(int(day_per_hour.shape[0])):
            hours_for_day[day_per_hour[hi]].append(hi)
        for t in range(n_t):
            hrs = hours_for_day.get(days[t], [])
            acc: list[float] = []
            for hi in hrs:
                mask = arr_h[hi] > 0.0
                if not mask.any():
                    continue
                _, nlab = ndi.label(mask, structure=_STRUCTURE8)
                acc.append(float(int(nlab)))
            if acc:
                n_components[t] = float(np.nanmean(acc))
    elif "n_conv_hours_grid" in dur_c:
        n_grid = dur_c["n_conv_hours_grid"].astype(float)
        spatial_dims = tuple(d for d in n_grid.dims if d != "time")
        if len(spatial_dims) != 2:
            return None
        arr = np.nan_to_num(n_grid.transpose("time", *spatial_dims).values, nan=0.0)
        for t in range(n_t):
            mask = arr[t] > 0.0
            if not mask.any():
                continue
            _, nlab = ndi.label(mask, structure=_STRUCTURE8)
            n_components[t] = float(int(nlab))
    else:
        return None

    return xr.DataArray(
        n_components.astype(float),
        coords=coords,
        dims=("time",),
        name="n_components",
    )


def build_extent_mean_patch_size_from_duration(dur_c: xr.Dataset) -> xr.DataArray | None:
    """Daily mean size (# grid cells) of each 8-connected flagged patch -- the typical EPE
    spatial-footprint size, as opposed to ``build_connectivity_n_components_from_duration``'s
    patch *count* or a plain domain-wide flagged-cell count. Same connected-component labeling
    (structure=_STRUCTURE8, diagonal touches count as connected) and dual E_conv_hour /
    n_conv_hours_grid source path as that function, so "connectivity" and "extent" are directly
    comparable per day.
    """
    try:
        from scipy import ndimage as ndi
    except Exception:
        return None

    n_t = int(dur_c.sizes["time"])
    coords = {"time": dur_c["time"].values}
    mean_patch_size = np.full((n_t,), np.nan, dtype=float)

    def _mean_component_size(mask: np.ndarray) -> float | None:
        if not mask.any():
            return None
        labeled, nlab = ndi.label(mask, structure=_STRUCTURE8)
        if nlab == 0:
            return None
        sizes = ndi.sum(mask, labeled, index=np.arange(1, nlab + 1))
        return float(np.mean(sizes))

    if "E_conv_hour" in dur_c:
        from collections import defaultdict

        e = dur_c["E_conv_hour"].astype(float)
        spatial_dims = tuple(d for d in e.dims if d != "time_hour")
        if len(spatial_dims) != 2:
            return None
        arr_h = np.nan_to_num(e.transpose("time_hour", *spatial_dims).values, nan=0.0)
        th = dur_c["time_hour"].values.astype("datetime64[h]")
        day_start = int(dur_c.attrs.get("day_start_hour_utc", 0))
        day_per_hour = (th - np.timedelta64(day_start, "h")).astype("datetime64[D]")
        days = dur_c["day"].values.astype("datetime64[D]")
        hours_for_day: dict = defaultdict(list)
        for hi in range(int(day_per_hour.shape[0])):
            hours_for_day[day_per_hour[hi]].append(hi)
        for t in range(n_t):
            hrs = hours_for_day.get(days[t], [])
            acc = [
                size for hi in hrs
                if (size := _mean_component_size(arr_h[hi] > 0.0)) is not None
            ]
            if acc:
                mean_patch_size[t] = float(np.nanmean(acc))
    elif "n_conv_hours_grid" in dur_c:
        n_grid = dur_c["n_conv_hours_grid"].astype(float)
        spatial_dims = tuple(d for d in n_grid.dims if d != "time")
        if len(spatial_dims) != 2:
            return None
        arr = np.nan_to_num(n_grid.transpose("time", *spatial_dims).values, nan=0.0)
        for t in range(n_t):
            size = _mean_component_size(arr[t] > 0.0)
            if size is not None:
                mean_patch_size[t] = size
    else:
        return None

    return xr.DataArray(
        mean_patch_size.astype(float),
        coords=coords,
        dims=("time",),
        name="mean_patch_size",
    )


def _precip_factor_from_units(units: str) -> float:
    u = str(units or "").lower()
    return 1000.0 if ("m" in u and "mm" not in u) else 1.0


def _parse_attr_token(attr_value: str, key: str) -> str:
    m = re.search(rf"{re.escape(key)}=([^;\s]+)", str(attr_value))
    return m.group(1).strip() if m else ""


def _resolve_hourly_source_path(
    duration_ds: xr.Dataset,
    *,
    module_root: Path | None,
    study_area: str | None,
) -> Path | None:
    """Best-effort path to hourly ERA5 inputs referenced by the duration NetCDF."""
    src = str(duration_ds.attrs.get("source_hourly_inputs", "") or "").strip()
    if src.startswith("auto:") and not duration_ds.attrs.get("hourly_raw_subdir"):
        parsed = _parse_attr_token(src, "raw_subdir")
        if parsed:
            duration_ds = duration_ds.copy()
            duration_ds.attrs = dict(duration_ds.attrs)
            duration_ds.attrs["hourly_raw_subdir"] = parsed
    if src and not src.startswith("auto"):
        p = Path(src).expanduser()
        if not p.is_absolute() and module_root is not None:
            for base in (module_root, module_root.parent):
                cand = (base / p).resolve()
                if cand.is_file():
                    return cand
        if p.is_file():
            return p.resolve()
    raw_sub = str(duration_ds.attrs.get("hourly_raw_subdir", "") or "").strip()
    if module_root is not None and raw_sub:
        cand = (Path(module_root) / "data/raw/era5/extreme_precip_events" / raw_sub).resolve()
        if cand.is_dir():
            return cand
    if module_root is not None and study_area:
        try:
            from epe_preprocessing.region_config import load_region_config

            cfg = load_region_config(step_root=module_root, region_id=str(study_area).strip())
            d = cfg.era5_hourly_dir
            if d.is_dir():
                return d
        except Exception:
            pass
    return None


def _daily_event_mask(
    *,
    ext_c: xr.Dataset | None,
    dur_c: xr.Dataset | None,
) -> xr.DataArray | None:
    """Day-level extreme mask (domain_extreme or duration flags)."""
    from epe_preprocessing.epe_functions import _daily_extreme_mask

    return _daily_extreme_mask(extremes_ds=ext_c, duration_ds=dur_c)


def _resolve_hourly_intensity_for_summary(
    dur_c: xr.Dataset | None,
    *,
    module_root: Path | None = None,
    study_area: str | None = None,
    skip_daily_mean_proxy: bool = False,
    agg: str = "mean",
) -> tuple[xr.DataArray | None, str]:
    """Mean (or median) hourly precip on wet convective-flagged cells, daily for panel (C).

    ``agg`` selects the spatial aggregation over wet flagged cells: ``"mean"`` or ``"median"``.
    When ``skip_daily_mean_proxy`` is True, do not fall back to the duration-file proxy.
    """
    if dur_c is None:
        return None, "no duration dataset"
    hourly_path: Path | None = None
    era5_dir: Path | None = None
    src = _resolve_hourly_source_path(dur_c, module_root=module_root, study_area=study_area)
    if src is not None:
        if src.is_file():
            hourly_path = src
        elif src.is_dir():
            era5_dir = src
    try:
        from epe_preprocessing.epe_functions import (
            _flagged_daily_intensity_proxy_max,
            _intensity_has_finite_values,
            flagged_hourly_precip_max_per_day,
        )

        da, metric = flagged_hourly_precip_max_per_day(
            dur_c,
            hourly_precip_path=hourly_path,
            era5_hourly_dir=era5_dir,
            allow_daily_mean_proxy=not skip_daily_mean_proxy,
            agg=agg,
        )
        if _intensity_has_finite_values(da):
            return da, metric

        if not skip_daily_mean_proxy:
            proxy_da, proxy_metric = _flagged_daily_intensity_proxy_max(dur_c, agg=agg)
            if _intensity_has_finite_values(proxy_da):
                return proxy_da, proxy_metric

        return None, metric or "hourly intensity unavailable"
    except Exception as e:
        if not skip_daily_mean_proxy:
            try:
                from epe_preprocessing.epe_functions import (
                    _flagged_daily_intensity_proxy_max,
                    _intensity_has_finite_values,
                )

                proxy_da, proxy_metric = _flagged_daily_intensity_proxy_max(dur_c, agg=agg)
                if _intensity_has_finite_values(proxy_da):
                    return proxy_da, proxy_metric
            except Exception:
                pass
        return None, str(e)


def _compute_flagged_daily_precip_max(ext_c: xr.Dataset) -> tuple[xr.DataArray | None, str]:
    """Daily max precip (mm) over ``E_conv``-flagged cells — epe_04 notebook panel (C)."""
    if "P" not in ext_c.data_vars:
        return None, "ext_c missing 'P'"
    if "E_conv" not in ext_c.data_vars:
        return None, "ext_c missing 'E_conv'"
    try:
        from epe_preprocessing.epe_functions import (
            _intensity_has_finite_values,
            guess_lat_lon,
            to_mm,
        )

        lat_e, lon_e = guess_lat_lon(ext_c)
        spatial = (lat_e, lon_e)
        p_day = to_mm(
            ext_c["P"].astype(float),
            str(ext_c["P"].attrs.get("units", "")),
        )
        e_day = ext_c["E_conv"].astype(int) > 0
        pmax = p_day.where(e_day).max(dim=spatial, skipna=True)
        if not _intensity_has_finite_values(pmax):
            return None, "no finite daily-max values on flagged cells"
        return pmax, "P_where_E_conv_max"
    except Exception as e:
        return None, str(e)


def _compute_flagged_daily_precip_spatial_stats(
    ext_c: xr.Dataset,
) -> tuple[xr.DataArray | None, xr.DataArray | None, str]:
    """Daily spatial mean and median of P over E_conv-flagged cells.

    Returns ``(mean_da, median_da, note)``.  Both are ``NaN`` on days with no
    flagged cells.  Units are converted to mm (same as
    ``_compute_flagged_daily_precip_max``).
    """
    if "P" not in ext_c.data_vars:
        return None, None, "ext_c missing 'P'"
    if "E_conv" not in ext_c.data_vars:
        return None, None, "ext_c missing 'E_conv'"
    try:
        from epe_preprocessing.epe_functions import (
            _intensity_has_finite_values,
            guess_lat_lon,
            to_mm,
        )

        lat_e, lon_e = guess_lat_lon(ext_c)
        spatial = (lat_e, lon_e)
        p_day = to_mm(
            ext_c["P"].astype(float),
            str(ext_c["P"].attrs.get("units", "")),
        )
        e_day = ext_c["E_conv"].astype(int) > 0
        p_masked = p_day.where(e_day)
        p_mean = p_masked.mean(dim=spatial, skipna=True)
        p_median = p_masked.median(dim=spatial, skipna=True)
        if not _intensity_has_finite_values(p_mean):
            return None, None, "no finite spatial-mean values on flagged cells"
        return p_mean, p_median, "P_where_E_conv_spatial_stats"
    except Exception as e:
        return None, None, str(e)


def _resolve_panel_c_intensity(
    *,
    ext_c: xr.Dataset,
    dur_c: xr.Dataset | None,
    module_root: Path | None,
    study_area: str | None,
    panel_c_intensity_mode: PanelCIntensityMode,
    panel_c_force_max_label: bool,
    p_flagged_max_day: xr.DataArray | None = None,
    agg: str = "mean",
) -> tuple[xr.DataArray | None, str, str, str, str]:
    """Panel (C) series and labels for the 2×2 summary.

    Returns ``(series, ylabel, title, empty_message, metric_note)``.
    ``agg`` is passed to the hourly resolver: ``"mean"`` or ``"median"``.
    """
    if panel_c_intensity_mode == "daily_max":
        pmax = p_flagged_max_day
        note = "precomputed P_flagged_max_day"
        if pmax is None:
            pmax, note = _compute_flagged_daily_precip_max(ext_c)
        return (
            pmax,
            "precip max over flagged cells (mm)",
            "(C) Intensity — daily max on flagged cells (extreme days)",
            "No daily-max intensity on extreme days",
            note,
        )

    pint: xr.DataArray | None = None
    note = ""
    if dur_c is not None:
        pint, note = _resolve_hourly_intensity_for_summary(
            dur_c,
            module_root=module_root,
            study_area=study_area,
            skip_daily_mean_proxy=panel_c_force_max_label,
            agg=agg,
        )
    agg_label = "median" if agg == "median" else "mean"
    if note and "proxy" in note and not panel_c_force_max_label:
        title = f"(C) Intensity — {agg_label} mm/h proxy on wet flagged cells (extreme days)"
    else:
        title = f"(C) Intensity — hourly {agg_label} on wet flagged cells (extreme days)"
    return (
        pint,
        f"hourly precip {agg_label} on wet flagged cells (mm/h)",
        title,
        "No hourly intensity on extreme days",
        note,
    )


def build_figure_epe_04_summary_occurrence_persistence_intensity_connectivity(
    *,
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    dur_c: xr.Dataset | None,
    labels_dur: xr.DataArray | None,
    n_components: xr.DataArray | None = None,
    p_flagged_max_day: xr.DataArray | None = None,
    lat_name: str | None = None,
    lon_name: str | None = None,
    module_root: Path | None = None,
    study_area: str | None = None,
    occurrence_all_days: bool = False,
    panel_c_force_max_label: bool = False,
    panel_c_intensity_mode: PanelCIntensityMode = "hourly_max",
) -> Figure:
    """2×2 summary: (A) occurrence, (B) persistence, (C) intensity, (D) connectivity (epe_04 notebook).

    Set ``occurrence_all_days=True`` to drop the extreme-day filter in panel (A) so the
    relative-occurrence stack uses every MJJAS day (regime frequency × monthly composition).
    Panels (B)–(D) always restrict to extreme days regardless of this flag.

    ``panel_c_intensity_mode="hourly_max"`` (default) uses ``flagged_hourly_precip_max_per_day``.
    ``panel_c_intensity_mode="daily_max"`` uses max daily ``ext_c['P']`` over ``E_conv`` cells
    (same metric as ``epe_04_extreme_regimes_statistics.ipynb`` panel C).

    ``panel_c_force_max_label=True`` (hourly mode only) keeps panel (C)'s title locked to the
    "hourly max" form even when the resolver falls back to the proxy.
    """
    from epe_preprocessing.epe_functions import guess_lat_lon

    reg_ext = np.unique(labels_ext.values.astype(int))
    reg_ext = reg_ext[reg_ext > 0]

    event_mask = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
    if event_mask is not None:
        event_mask = event_mask.astype(bool)

    # Larger canvas + explicit subplot padding so the bumped 16/14 pt fonts
    # (titles, axis labels, legend, sample-size tick labels) cannot collide
    # with each other or the plot interior. The figure is wider/taller than
    # A4 portrait but ``bbox_inches='tight'`` in savefig handles the page.
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), facecolor="white")
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    occ_scope_tag = "all days" if occurrence_all_days else "extreme days only"
    fig.suptitle(
        f"epe_04 summary — occurrence ({occ_scope_tag}), persistence, intensity, connectivity",
        fontsize=PDF_FS_PANEL_SUPTITLE,
        y=0.995,
    )

    # (A) Relative occurrence: MJJAS composition × regime share (extreme days only OR all days)
    month = pd.to_datetime(np.asarray(ext_c["time"].values)).month.astype(int)
    regime = labels_ext.values.astype(int)
    df_a = pd.DataFrame({"month": month, "regime": regime})
    if event_mask is not None and not occurrence_all_days:
        df_a = df_a.loc[np.asarray(event_mask.values, dtype=bool)]
    df_a = df_a[(df_a["regime"] > 0) & (df_a["month"].isin(_MJJAS_MONTHS))]
    n_per_regime_a: list[int] = []
    if not df_a.empty and reg_ext.size:
        ct_a = df_a.groupby(["regime", "month"]).size().unstack("month")
        ct_a = ct_a.reindex(columns=list(_MJJAS_MONTHS), fill_value=0).reindex(
            index=[int(r) for r in reg_ext],
            fill_value=0,
        )
        within_a = ct_a.div(ct_a.sum(axis=1), axis=0).fillna(0)
        within_a.columns = list(_MJJAS_LABELS)
        n_per_regime_a = [int((df_a["regime"] == int(r)).sum()) for r in reg_ext]
        n_ev = max(len(df_a), 1)
        freq_vec = np.array(
            [float(c) / n_ev for c in n_per_regime_a],
            dtype=float,
        )
        freq_s = pd.Series(freq_vec, index=[int(r) for r in reg_ext])
        wm_a = within_a.mul(freq_s, axis=0)
        blues = plt.cm.Blues(np.linspace(0.38, 0.92, len(wm_a.columns)))
        wm_a.plot(kind="bar", stacked=True, ax=ax_a, color=blues, legend=True)
        ax_a.legend(title="month", ncol=2, fontsize=PDF_FS_PANEL_LEGEND, loc="upper right", framealpha=0.92)
    ax_a.set_xlabel("regime", fontsize=PDF_FS_PANEL_AXIS, labelpad=8)
    ax_a.set_ylabel(
        "fraction of all days" if occurrence_all_days else "fraction of extreme days",
        fontsize=PDF_FS_PANEL_AXIS,
        labelpad=8,
    )
    ax_a.set_title(
        "(A) Relative occurrence (all days)"
        if occurrence_all_days
        else "(A) Relative occurrence (extreme days only)",
        fontsize=PDF_FS_PANEL_TITLE,
        pad=12,
    )
    ax_a.grid(True, axis="y", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=PDF_FS_PANEL_TICK)
    # Sample size below each x-tick: shows how often a regime occurred in the
    # underlying dataset (all MJJAS days vs extreme days only) so the bar height
    # can be read together with n. rotation=0 keeps the labels upright (no 90°
    # rotation that pandas applies to ``kind="bar"`` by default).
    if n_per_regime_a and len(n_per_regime_a) == len(reg_ext):
        tick_labels_a = [
            f"{int(r)}\nn={n}" for r, n in zip(reg_ext, n_per_regime_a, strict=False)
        ]
        ax_a.set_xticks(np.arange(len(reg_ext)))
        ax_a.set_xticklabels(tick_labels_a, rotation=0, ha="center")
    else:
        ax_a.tick_params(axis="x", rotation=0)
        for lab in ax_a.get_xticklabels():
            lab.set_rotation(0)
            lab.set_ha("center")

    # (B) Persistence — extreme days only (mean hours per flagged cell per day)
    data_b, lab_b, ids_b = [], [], []
    if dur_c is not None and labels_dur is not None and "n_conv_hours_grid" in dur_c:
        reg_dur = np.unique(labels_dur.values.astype(int))
        reg_dur = reg_dur[reg_dur > 0]
        n_grid_b = dur_c["n_conv_hours_grid"].astype(float)
        sp_b = tuple(d for d in n_grid_b.dims if d != "time")
        mh_flag = n_grid_b.where(n_grid_b > 0).mean(dim=sp_b, skipna=True)
        day_flag_b = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
        if day_flag_b is None:
            day_flag_b = mh_flag > 0
        for r in reg_dur:
            sel = (labels_dur == int(r)) & day_flag_b.astype(bool)
            v = mh_flag.where(sel, drop=True).values
            v = v[np.isfinite(v)]
            if v.size:
                data_b.append(v)
                lab_b.append(str(int(r)))
                ids_b.append(int(r))
    colors_b = [regime_tab10_colors(ids_b)[r] for r in ids_b] if ids_b else None
    style_boxplot(
        ax_b,
        data_b,
        lab_b,
        title="(B) Persistence — extreme days only",
        ylabel="hours / flagged cell / day",
        colors=colors_b,
        tier="panel",
    )

    # (C) Intensity — mean hourly precip on wet flagged cells (extreme days only)
    labels_c = labels_dur if labels_dur is not None else labels_ext
    pmax, ylab_c, title_c, empty_c, intensity_note = _resolve_panel_c_intensity(
        ext_c=ext_c,
        dur_c=dur_c,
        module_root=module_root,
        study_area=study_area,
        panel_c_intensity_mode=panel_c_intensity_mode,
        panel_c_force_max_label=panel_c_force_max_label,
        p_flagged_max_day=p_flagged_max_day,
        agg="mean",
    )
    data_c, lab_c, ids_c = [], [], []
    if pmax is not None:
        day_flag_c = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
        if day_flag_c is None:
            day_flag_c = pmax > 0
        pmax_use, labels_use = xr.align(pmax, labels_c, join="inner")
        day_flag_c, _ = xr.align(day_flag_c, labels_use, join="inner")
        for r in reg_ext:
            sel = (labels_use == int(r)) & day_flag_c.astype(bool)
            v = pmax_use.where(sel, drop=True).values
            v = v[np.isfinite(v)]
            if v.size:
                data_c.append(v)
                lab_c.append(str(int(r)))
                ids_c.append(int(r))
        if not data_c:
            ax_c.text(
                0.5,
                0.5,
                empty_c,
                ha="center",
                va="center",
                transform=ax_c.transAxes,
                fontsize=PDF_FS_PANEL_TITLE,
            )
    else:
        mode_tag = "Daily" if panel_c_intensity_mode == "daily_max" else "Hourly"
        ax_c.text(
            0.5,
            0.5,
            f"{mode_tag} intensity unavailable\n({intensity_note})",
            ha="center",
            va="center",
            transform=ax_c.transAxes,
            fontsize=PDF_FS_PANEL_TITLE,
        )
    colors_c = [regime_tab10_colors(ids_c)[r] for r in ids_c] if ids_c else None
    style_boxplot(
        ax_c,
        data_c,
        lab_c,
        title=title_c,
        ylabel=ylab_c,
        colors=colors_c,
        tier="panel",
    )

    # (D) Connectivity
    if n_components is None and dur_c is not None:
        n_components = build_connectivity_n_components_from_duration(dur_c)
    data_d, lab_d, ids_d = [], [], []
    if n_components is not None and labels_dur is not None:
        reg_dur = np.unique(labels_dur.values.astype(int))
        reg_dur = reg_dur[reg_dur > 0]
        day_flag_d = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
        if day_flag_d is None:
            day_flag_d = n_components > 0
        for r in reg_dur:
            sel = (labels_dur == int(r)) & day_flag_d.astype(bool)
            v = n_components.where(sel, drop=True).values
            v = v[np.isfinite(v)]
            if v.size:
                data_d.append(v)
                lab_d.append(str(int(r)))
                ids_d.append(int(r))
    else:
        ax_d.text(
            0.5,
            0.5,
            "Connectivity unavailable",
            ha="center",
            va="center",
            transform=ax_d.transAxes,
            fontsize=PDF_FS_PANEL_TITLE,
        )
    colors_d = [regime_tab10_colors(ids_d)[r] for r in ids_d] if ids_d else None
    style_boxplot(
        ax_d,
        data_d,
        lab_d,
        title="(D) Connectivity — extreme days only",
        ylabel="# components",
        colors=colors_d,
        tier="panel",
    )

    # Panel-tier fonts (10/9/7-8 pt) keep all four subplot titles/labels/ticks
    # legible at 2x2 density; h_pad/w_pad still give the suptitle and legend
    # on (A) room to breathe after tight_layout reflows the figure.
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.4, w_pad=1.8)
    return fig


@dataclass
class BboxRegimeMetricRow:
    """One bbox step × mapped regime id for :func:`build_figure_epe_04_summary_across_bbox`."""

    bbox_step: str
    regime_id: int
    labels_ext: xr.DataArray
    labels_dur: xr.DataArray | None = None
    k: int | None = None


def build_figure_epe_04_summary_across_bbox(
    *,
    ext_c: xr.Dataset,
    dur_c: xr.Dataset | None,
    rows: list[BboxRegimeMetricRow],
    n_components: xr.DataArray | None = None,
    module_root: Path | None = None,
    study_area: str | None = None,
    occurrence_all_days: bool = False,
    panel_c_force_max_label: bool = True,
    panel_c_intensity_mode: PanelCIntensityMode = "hourly_max",
    p_flagged_max_day: xr.DataArray | None = None,
    title_line: str | None = None,
) -> Figure:
    """2×2 epe_04-style summary with **bbox step** on the x-axis (one physical regime).

    Mirrors :func:`build_figure_epe_04_summary_occurrence_persistence_intensity_connectivity`
    (PDF Figure 15 styling) but each column position is a bounding-box step and shows the
    evolution of the *same* physical regime (possibly different regime ids per step).
    """
    if not rows:
        raise ValueError("Need at least one bbox row to build the across-bbox summary.")

    rows_sorted = sorted(rows, key=lambda r: str(r.bbox_step).upper())
    x_labels = [str(r.bbox_step).upper() for r in rows_sorted]

    event_mask = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
    if event_mask is not None:
        event_mask = event_mask.astype(bool)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), facecolor="white")
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    occ_scope_tag = "all days" if occurrence_all_days else "extreme days only"
    suptitle = title_line or (
        f"epe_04 summary across bbox — occurrence ({occ_scope_tag}), "
        "persistence, intensity, connectivity"
    )
    fig.suptitle(suptitle, fontsize=PDF_FS_PANEL_SUPTITLE, y=0.995)

    # (A) Relative occurrence — monthly stack for this regime at each bbox
    month_fracs: list[pd.Series] = []
    n_per_step_a: list[int] = []
    for row in rows_sorted:
        regime = int(row.regime_id)
        month = pd.to_datetime(np.asarray(ext_c["time"].values)).month.astype(int)
        regime_arr = row.labels_ext.values.astype(int)
        sel = (regime_arr == regime) & np.isin(month, _MJJAS_MONTHS)
        if event_mask is not None and not occurrence_all_days:
            sel = sel & np.asarray(event_mask.values, dtype=bool)
        n_sel = int(sel.sum())
        n_per_step_a.append(n_sel)
        if n_sel == 0:
            month_fracs.append(pd.Series(0.0, index=list(_MJJAS_LABELS)))
            continue
        ct = pd.Series(month[sel]).value_counts()
        ct = ct.reindex(list(_MJJAS_MONTHS), fill_value=0)
        month_fracs.append(ct.div(ct.sum()).fillna(0.0))
        month_fracs[-1].index = list(_MJJAS_LABELS)

    if month_fracs and any(n > 0 for n in n_per_step_a):
        wm_a = pd.DataFrame(month_fracs, index=x_labels).T
        blues = plt.cm.Blues(np.linspace(0.38, 0.92, len(wm_a.index)))
        bottom = np.zeros(len(x_labels))
        x_pos = np.arange(len(x_labels))
        for mi, month_lbl in enumerate(wm_a.index):
            vals = wm_a.loc[month_lbl].values.astype(float)
            ax_a.bar(x_pos, vals, bottom=bottom, color=blues[mi], label=month_lbl, width=0.72)
            bottom = bottom + vals
        ax_a.legend(title="month", ncol=2, fontsize=PDF_FS_PANEL_LEGEND, loc="upper right", framealpha=0.92)
    ax_a.set_xlabel("bounding box", fontsize=PDF_FS_PANEL_AXIS, labelpad=8)
    ax_a.set_ylabel(
        "fraction of all days" if occurrence_all_days else "fraction of extreme days",
        fontsize=PDF_FS_PANEL_AXIS,
        labelpad=8,
    )
    ax_a.set_title(
        "(A) Relative occurrence (all days)"
        if occurrence_all_days
        else "(A) Relative occurrence (extreme days only)",
        fontsize=PDF_FS_PANEL_TITLE,
        pad=12,
    )
    ax_a.grid(True, axis="y", alpha=0.25)
    ax_a.tick_params(axis="both", labelsize=PDF_FS_PANEL_TICK)
    tick_labels_a = [
        f"{lab}\nregime {int(row.regime_id)}\nn={n}"
        for lab, row, n in zip(x_labels, rows_sorted, n_per_step_a, strict=False)
    ]
    ax_a.set_xticks(np.arange(len(x_labels)))
    ax_a.set_xticklabels(tick_labels_a, rotation=0, ha="center")

    # (B) Persistence — extreme days only
    data_b, lab_b, ids_b = [], [], []
    if dur_c is not None and "n_conv_hours_grid" in dur_c:
        n_grid_b = dur_c["n_conv_hours_grid"].astype(float)
        sp_b = tuple(d for d in n_grid_b.dims if d != "time")
        mh_flag = n_grid_b.where(n_grid_b > 0).mean(dim=sp_b, skipna=True)
        for row in rows_sorted:
            labels_dur = row.labels_dur
            if labels_dur is None:
                continue
            day_flag_b = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
            if day_flag_b is None:
                day_flag_b = mh_flag > 0
            sel = (labels_dur == int(row.regime_id)) & day_flag_b.astype(bool)
            v = mh_flag.where(sel, drop=True).values
            v = v[np.isfinite(v)]
            if v.size:
                data_b.append(v)
                k_lbl = f" k={int(row.k)}" if row.k is not None else ""
                lab_b.append(f"{row.bbox_step.upper()}{k_lbl}")
                ids_b.append(int(row.regime_id))
    colors_b = [regime_tab10_colors(ids_b)[r] for r in ids_b] if ids_b else None
    style_boxplot(
        ax_b,
        data_b,
        lab_b,
        title="(B) Persistence — extreme days only",
        ylabel="hours / flagged cell / day",
        xlabel="bounding box",
        colors=colors_b,
        tier="panel",
    )

    # (C) Intensity — mean hourly precip on wet flagged cells (extreme days only)
    pmax, ylab_c, title_c, empty_c, intensity_note = _resolve_panel_c_intensity(
        ext_c=ext_c,
        dur_c=dur_c,
        module_root=module_root,
        study_area=study_area,
        panel_c_intensity_mode=panel_c_intensity_mode,
        panel_c_force_max_label=panel_c_force_max_label,
        p_flagged_max_day=p_flagged_max_day,
        agg="mean",
    )
    data_c, lab_c, ids_c = [], [], []
    if pmax is not None:
        day_flag_c = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
        if day_flag_c is None:
            day_flag_c = pmax > 0
        for row in rows_sorted:
            labels_c = row.labels_dur if row.labels_dur is not None else row.labels_ext
            pmax_use, labels_use = xr.align(pmax, labels_c, join="inner")
            day_c, _ = xr.align(day_flag_c, labels_use, join="inner")
            sel = (labels_use == int(row.regime_id)) & day_c.astype(bool)
            v = pmax_use.where(sel, drop=True).values
            v = v[np.isfinite(v)]
            if v.size:
                data_c.append(v)
                k_lbl = f" k={int(row.k)}" if row.k is not None else ""
                lab_c.append(f"{row.bbox_step.upper()}{k_lbl}")
                ids_c.append(int(row.regime_id))
        if not data_c:
            ax_c.text(
                0.5, 0.5, empty_c,
                ha="center", va="center", transform=ax_c.transAxes, fontsize=PDF_FS_PANEL_TITLE,
            )
    else:
        mode_tag = "Daily" if panel_c_intensity_mode == "daily_max" else "Hourly"
        ax_c.text(
            0.5, 0.5, f"{mode_tag} intensity unavailable\n({intensity_note})",
            ha="center", va="center", transform=ax_c.transAxes, fontsize=PDF_FS_PANEL_TITLE,
        )
    colors_c = [regime_tab10_colors(ids_c)[r] for r in ids_c] if ids_c else None
    style_boxplot(
        ax_c,
        data_c,
        lab_c,
        title=title_c,
        ylabel=ylab_c,
        xlabel="bounding box",
        colors=colors_c,
        tier="panel",
    )

    # (D) Connectivity
    if n_components is None and dur_c is not None:
        n_components = build_connectivity_n_components_from_duration(dur_c)
    data_d, lab_d, ids_d = [], [], []
    if n_components is not None:
        day_flag_d = _daily_event_mask(ext_c=ext_c, dur_c=dur_c)
        if day_flag_d is None:
            day_flag_d = n_components > 0
        for row in rows_sorted:
            labels_d = row.labels_dur
            if labels_d is None:
                continue
            sel = (labels_d == int(row.regime_id)) & day_flag_d.astype(bool)
            v = n_components.where(sel, drop=True).values
            v = v[np.isfinite(v)]
            if v.size:
                data_d.append(v)
                k_lbl = f" k={int(row.k)}" if row.k is not None else ""
                lab_d.append(f"{row.bbox_step.upper()}{k_lbl}")
                ids_d.append(int(row.regime_id))
        if not data_d:
            ax_d.text(
                0.5, 0.5, "Connectivity unavailable",
                ha="center", va="center", transform=ax_d.transAxes, fontsize=PDF_FS_PANEL_TITLE,
            )
    else:
        ax_d.text(
            0.5, 0.5, "Connectivity unavailable",
            ha="center", va="center", transform=ax_d.transAxes, fontsize=PDF_FS_PANEL_TITLE,
        )
    colors_d = [regime_tab10_colors(ids_d)[r] for r in ids_d] if ids_d else None
    style_boxplot(
        ax_d,
        data_d,
        lab_d,
        title="(D) Connectivity — extreme days only",
        ylabel="# components",
        xlabel="bounding box",
        colors=colors_d,
        tier="panel",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.4, w_pad=1.8)
    return fig
