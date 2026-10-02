"""EPE.4 style boxplots (``epe_04_extreme_regimes_statistics.ipynb``) for bbox summary PDFs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from common.page_layout import (
    PDF_FS_FIG_ANNOT,
    PDF_FS_FIG_AXIS,
    PDF_FS_FIG_SUPTITLE,
    PDF_FS_FIG_TICK,
    PDF_FS_FIG_TITLE,
    append_png_page_a4,
    filter_epe_summary_png_paths,
    pdf_next_figure_number_would_exceed_cap,
    savefig_a4,
)
from common.boxplot_style import regime_tab10_colors, style_boxplot
from common.epe_figures import (
    PanelCIntensityMode,
    build_connectivity_n_components_from_duration,
    build_extent_mean_patch_size_from_duration,
    build_figure_epe_04_summary_occurrence_persistence_intensity_connectivity,
)
from common.extremes_spec import (
    ExtremesSpec,
    _events_dirs,
    glob_convective_duration_nc,
    glob_extremes_nc,
)
try:
    # PDF-report helpers. Only the report-building functions in this module use them; the
    # statistics entry points (``load_epe_04_statistics_inputs``, ``compute_epe_04_core_metrics``)
    # do not, so a missing report layer must not make this module unimportable.
    from common.step1011_render import (  # type: ignore[import-not-found]
        _append_text_pdf_page,
        _bump_fig,
        _save_selected_report_png,
    )
except ImportError:  # report assembly is not part of this analysis-only distribution

    def _report_layer_missing(symbol: str):
        def _raise(*_args, **_kwargs):
            raise RuntimeError(
                f"{symbol}() needs the PDF report layer (step1011_render), which is not "
                "part of this analysis-only distribution. The statistics functions in this "
                "module do not require it."
            )

        return _raise

    _append_text_pdf_page = _report_layer_missing("_append_text_pdf_page")
    _bump_fig = _report_layer_missing("_bump_fig")
    _save_selected_report_png = _report_layer_missing("_save_selected_report_png")

EPE_04_SUMMARY_CONNECTIVITY_PNG_STEM = "epe_04_summary_occurrence_persistence_intensity_connectivity"
EPE_04_SUMMARY_PANEL_C_DAILY_MAX_SUFFIX = "_panel_c_daily_max"

_EPE_04_SECTION_A_PNG_ORDER_LOWER: tuple[str, ...] = (
    "001_regime_frequency_all_days_aligned_period.png",
    "002_within_each_regime_mjjas_composition_fractions_sum_to_1.png",
    "003_mjjas_composition__regime_frequency_bar_height__annual_regime_fraction.png",
    "004_flagged_event_days_per_regime.png",
    "005_domain-level_event_probability_by_regime.png",
    "031_a_relative_occurrence.png",
    "032_a_relative_occurrence.png",
    "033_a_occurrence_rate__mjjas_composition__regime_frequency.png",
    "033_a_relative_occurrence.png",
    "034_a_relative_occurrence.png",
)


def _append_duration_audit_pdf_page(
    pdf: PdfPages,
    duration_path: Path | None,
    *,
    regimes_source: str,
    extremes_source: str,
    cluster_tag: str,
) -> None:
    """Embed ``epe_duration_audit`` summary (same text as the CLI audit script)."""
    if duration_path is None or not duration_path.is_file():
        return
    try:
        from epe_preprocessing.epe_03_03_duration_audit import collect_duration_audit, format_duration_audit_body
    except ImportError as e:
        _append_text_pdf_page(
            pdf,
            "EPE — duration audit skipped",
            f"Could not import epe_03_03_duration_audit: {e}",
        )
        return

    report = collect_duration_audit(duration_path)
    header = (
        "Automated sanity check of the convective_duration NetCDF used for EPE.4 "
        f"(cluster labels: {cluster_tag}).\n"
        f"Regimes: {regimes_source}\n"
        f"Extremes alignment: {extremes_source}\n\n"
    )
    body = header + format_duration_audit_body(report)
    _append_text_pdf_page(pdf, "EPE — convective duration audit", body)


def _epe_04_subsection_intro(pdf: PdfPages, letter: str, heading: str, body: str) -> None:
    """Lettered subsection matching ``epe_04_extreme_regimes_statistics.ipynb`` roadmap (A)–(D)."""
    _append_text_pdf_page(
        pdf,
        f"EPE.4 — ({letter}) {heading}",
        body,
    )


def _calendar_day_coords_ns(times) -> np.ndarray:
    """Naive-datetime coords at nanosecond resolution (quiet xarray / pandas warnings)."""
    return np.asarray(times).astype("datetime64[D]").astype("datetime64[ns]")


def _load_regime_day_index(reg: xr.Dataset) -> xr.Dataset:
    from epe_preprocessing.epe_functions import regime_labels_by_day

    return regime_labels_by_day(reg)


def _align_extremes_to_regimes(
    ext: xr.Dataset,
    reg_day_ds: xr.Dataset,
) -> tuple[xr.Dataset, xr.DataArray]:
    ext_days = _calendar_day_coords_ns(ext["time"].values)
    common = np.intersect1d(ext_days, reg_day_ds["day"].values)
    if common.size == 0:
        raise RuntimeError("no overlapping days between extremes and regime labels")
    ext = ext.assign_coords(day=("time", ext_days))
    ext_c = ext.sel(time=ext["day"].isin(common))
    labels_day = reg_day_ds["regime_labels"].sel(day=ext_c["day"].values)
    labels_ext = xr.DataArray(
        labels_day.values.astype(int),
        coords={"time": ext_c["time"].values},
        dims=("time",),
    )
    return ext_c, labels_ext


def _try_align_extremes_from_path(
    path: Path,
    reg_day_ds: xr.Dataset,
) -> tuple[xr.Dataset, xr.DataArray] | None:
    from epe_preprocessing.epe_functions import dataset_rename_time

    try:
        with xr.open_dataset(path, chunks=None) as ds:
            ds2 = dataset_rename_time(ds)
            ext_c, labels = _align_extremes_to_regimes(ds2, reg_day_ds)
            return ext_c.load(), labels
    except Exception:
        return None


def _try_align_synthetic_from_duration(
    path: Path,
    reg_day_ds: xr.Dataset,
) -> tuple[xr.Dataset, xr.DataArray] | None:
    from epe_preprocessing.epe_functions import dataset_rename_time, synthetic_extremes_from_duration

    try:
        with xr.open_dataset(path, chunks=None) as dur:
            dur2 = dataset_rename_time(dur)
            ext = synthetic_extremes_from_duration(dur2.load())
        ext_c, labels = _align_extremes_to_regimes(ext, reg_day_ds)
        return ext_c.load(), labels
    except Exception:
        return None


def build_boxplot_figure(
    data: list[np.ndarray],
    tick_labels: list[str],
    *,
    title: str,
    ylabel: str,
    fig_size: tuple[float, float] = (10.0, 5.2),
    fig_counter: list[int] | None = None,
    regime_ids: list[int] | None = None,
) -> "plt.Figure | None":
    """Build one full-page boxplot using the shared Z500/RR box style (``tier="full"``).

    Pure figure construction — no saving/closing — so a notebook can call this repeatedly
    while tweaking styling; :func:`_boxplot_to_pdf` wraps this for the PDF/HPC path.

    ``regime_ids`` (one int per entry in ``data``/``tick_labels``) colours each box by
    regime via :func:`regime_tab10_colors`, matching the colouring used for the same
    regime id in the residual-regimes (RR) statistics PDF.
    """
    if not data:
        return None
    fig, ax = plt.subplots(figsize=fig_size, facecolor="white")
    n = _bump_fig(fig_counter)
    colors = None
    if regime_ids is not None:
        cmap = regime_tab10_colors(regime_ids)
        colors = [cmap[int(r)] for r in regime_ids]
    # Extra pad on the title keeps the larger 16 pt font from clipping into the
    # plot area; tight_layout below adds matching h_pad so it doesn't collide
    # with the axes spines after the figure reflows.
    style_boxplot(
        ax,
        data,
        tick_labels,
        title=f"Figure {n} — {title}" if n else title,
        ylabel=ylabel,
        colors=colors,
        tier="full",
    )
    fig.tight_layout(pad=1.5, h_pad=1.2)
    return fig


def _boxplot_to_pdf(
    pdf: PdfPages,
    data: list[np.ndarray],
    tick_labels: list[str],
    *,
    title: str,
    ylabel: str,
    fig_size: tuple[float, float],
    fig_counter: list[int] | None,
    regime_ids: list[int] | None = None,
) -> bool:
    """Render one full-page boxplot using the shared Z500/RR box style (``tier="full"``)."""
    fig = build_boxplot_figure(
        data, tick_labels, title=title, ylabel=ylabel, fig_size=fig_size,
        fig_counter=fig_counter, regime_ids=regime_ids,
    )
    if fig is None:
        return False
    savefig_a4(pdf, fig, tight=True, pad_inches=0.25)
    plt.close(fig)
    return True


def _precip_factor_from_units(units: str) -> float:
    u = (units or "").lower()
    if "mm" in u or "kg" in u:
        return 1.0
    if "m" in u and "mm" not in u:
        return 1000.0
    return 1.0


def _event_day_mask(ext_c: xr.Dataset) -> xr.DataArray | None:
    """Best-effort day-level event mask aligned to ``ext_c.time``."""
    if "domain_extreme" in ext_c.data_vars:
        mask = ext_c["domain_extreme"].astype(int) > 0
        return mask.astype(bool)
    if "E_conv" in ext_c.data_vars:
        event = ext_c["E_conv"].astype(int) > 0
        spatial_dims = tuple(d for d in event.dims if d != "time")
        if spatial_dims:
            event = event.any(dim=spatial_dims)
        return event.astype(bool)
    return None


def _build_lagged_precursor_regime_matrices(
    *,
    reg_day_ds: xr.Dataset,
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    lag_days: int,
) -> tuple[xr.DataArray, xr.DataArray] | None:
    """Return counts and column-normalised probabilities for precursor regimes on event days."""
    if lag_days < 1:
        raise ValueError("lag_days must be >= 1")

    event_mask = _event_day_mask(ext_c)
    if event_mask is None:
        return None

    event_mask = event_mask.astype(bool)
    if event_mask.sizes.get("time", 0) == 0 or not bool(event_mask.any().item()):
        return None

    event_days = np.asarray(ext_c["day"].where(event_mask, drop=True).values).astype("datetime64[ns]")
    event_regimes = np.asarray(labels_ext.where(event_mask, drop=True).values, dtype=int)
    if event_days.size == 0 or event_regimes.size == 0:
        return None

    reg_days = np.asarray(reg_day_ds["day"].values).astype("datetime64[ns]")
    reg_labels = np.asarray(reg_day_ds["regime_labels"].values, dtype=int)
    if reg_days.size == 0 or reg_labels.size == 0:
        return None

    regime_vals = np.unique(reg_labels[reg_labels > 0]).astype(int)
    if regime_vals.size == 0:
        return None
    regime_to_idx = {int(r): i for i, r in enumerate(regime_vals.tolist())}
    day_to_regime = {
        np.datetime64(day, "ns"): int(reg)
        for day, reg in zip(reg_days, reg_labels, strict=False)
    }

    lag = np.timedelta64(int(lag_days), "D")
    counts = np.zeros((regime_vals.size, regime_vals.size), dtype=float)
    matched = 0
    for day, event_regime in zip(event_days, event_regimes, strict=False):
        event_regime_i = int(event_regime)
        if event_regime_i <= 0:
            continue
        precursor_regime = int(day_to_regime.get(np.datetime64(day - lag, "ns"), 0))
        if precursor_regime <= 0:
            continue
        i = regime_to_idx.get(precursor_regime)
        j = regime_to_idx.get(event_regime_i)
        if i is None or j is None:
            continue
        counts[i, j] += 1.0
        matched += 1
    if matched == 0:
        return None

    col_sum = counts.sum(axis=0, keepdims=True)
    probs = np.full_like(counts, np.nan, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(counts, col_sum, where=col_sum > 0, out=probs)

    coords = {
        "regime_lag": regime_vals.astype(int),
        "regime_event": regime_vals.astype(int),
    }
    counts_da = xr.DataArray(counts, dims=("regime_lag", "regime_event"), coords=coords, name="precursor_counts")
    probs_da = xr.DataArray(
        probs,
        dims=("regime_lag", "regime_event"),
        coords=coords,
        name="precursor_prob",
        attrs={"normalisation": "column", "conditioning": "P(regime at t-lag | event-day regime)"},
    )
    return counts_da, probs_da


def _append_precursor_regime_heatmap_page(
    pdf: PdfPages,
    *,
    reg_day_ds: xr.Dataset,
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    lag_days: int,
    fig_counter: list[int] | None,
) -> bool:
    pair = _build_lagged_precursor_regime_matrices(
        reg_day_ds=reg_day_ds,
        ext_c=ext_c,
        labels_ext=labels_ext,
        lag_days=lag_days,
    )
    if pair is None:
        return False
    counts_da, probs_da = pair

    counts = np.asarray(counts_da.values, dtype=float)
    probs = np.asarray(probs_da.values, dtype=float)
    xlabels = [str(int(x)) for x in counts_da["regime_event"].values]
    ylabels = [str(int(y)) for y in counts_da["regime_lag"].values]

    def _draw_matrix(ax, arr: np.ndarray, *, title: str, cmap: str, annotate: str) -> None:
        finite = arr[np.isfinite(arr)]
        vmax = float(np.nanmax(finite)) if finite.size else 0.0
        vmax_safe = max(vmax, 1e-6)
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax_safe)
        ax.set_title(title, fontsize=PDF_FS_FIG_TITLE)
        ax.set_xlabel("event-day regime", fontsize=PDF_FS_FIG_AXIS)
        ax.set_ylabel(f"regime at t-{lag_days}", fontsize=PDF_FS_FIG_AXIS)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.tick_params(axis="both", labelsize=PDF_FS_FIG_TICK)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = float(arr[i, j])
                if not np.isfinite(v):
                    label = ""
                    color = "black"
                elif annotate == "int":
                    label = str(int(round(v)))
                    color = "white" if v > 0.5 * vmax_safe else "black"
                else:
                    label = f"{v:.2f}"
                    color = "white" if v > 0.6 * vmax_safe else "black"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=PDF_FS_FIG_ANNOT)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=PDF_FS_FIG_TICK)

    n = _bump_fig(fig_counter)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), facecolor="white")
    _draw_matrix(axes[0], counts, title="Counts", cmap="viridis", annotate="int")
    _draw_matrix(
        axes[1],
        probs,
        title=f"Probabilities: P(regime at t-{lag_days} | event-day regime)",
        cmap="magma",
        annotate="float",
    )
    st = (
        f"Figure {n} — Precursor regimes {lag_days} days before event days "
        "(counts | conditional probabilities)"
        if n
        else f"Precursor regimes {lag_days} days before event days (counts | conditional probabilities)"
    )
    fig.suptitle(st, fontsize=PDF_FS_FIG_TITLE, y=1.02)
    fig.tight_layout()
    savefig_a4(pdf, fig, tight=True, pad_inches=0.18)
    plt.close(fig)
    return True


def _collect_epe_notebook_png_paths(
    *,
    module_root: Path,
    region: str,
    study_area: str | None = None,
) -> list[Path]:
    """Collect EPE notebook-exported PNGs.

    Lookup order (first non-empty match wins):
      1. ``figures/extreme_precip_events/<study_area>/`` (e.g. ``germany``, ``ontario_south``).
         This is where ``epe_04_extreme_regimes_statistics.ipynb`` actually saves PNGs
         (``REGION_LABEL = cfg.region_id``), so prefer it whenever a study_area is provided.
      2. ``figures/extreme_precip_events/<region>/`` (e.g. ``europe``, ``northamerica``)
         for legacy notebook configurations that mirror the broader region label.
      3. Fallback: every immediate subdirectory of ``figures/extreme_precip_events/``.
    """
    root = module_root / "figures" / "extreme_precip_events"
    if not root.is_dir():
        return []

    def _subdir_pngs(subname: str | None) -> list[Path]:
        if not subname:
            return []
        sub = root / str(subname).strip()
        if not sub.is_dir():
            return []
        return [p for p in sorted(sub.glob("*.png")) if p.is_file()]

    pngs: list[Path] = []
    pngs.extend(_subdir_pngs(study_area))
    if not pngs:
        pngs.extend(_subdir_pngs(region))
    if not pngs:
        for sub in sorted(root.iterdir()):
            if sub.is_dir():
                pngs.extend(sorted(sub.glob("*.png")))
            elif sub.suffix.lower() == ".png":
                pngs.append(sub)

    uniq: dict[Path, Path] = {}
    for p in pngs:
        if p.is_file():
            uniq[p.resolve()] = p
    return list(uniq.values())


def _append_epe_04_section_a_png_pages(
    pdf: PdfPages,
    *,
    module_root: Path,
    region: str,
    fig_counter: list[int] | None,
    emitted_epe_once_basenames: set[str] | None = None,
    emitted_basenames: set[str] | None = None,
    k: int | None = None,
    study_area: str | None = None,
) -> int:
    """Restore notebook subsection (A) pages before the precursor-regime heatmap."""
    order = {name: idx for idx, name in enumerate(_EPE_04_SECTION_A_PNG_ORDER_LOWER)}
    pngs = _collect_epe_notebook_png_paths(
        module_root=module_root, region=region, study_area=study_area
    )
    if not pngs:
        return 0
    chosen = [p for p in pngs if p.name.lower() in order]
    chosen = filter_epe_summary_png_paths(
        sorted(chosen, key=lambda p: (order.get(p.name.lower(), 999), p.name.lower())),
        k=k,
        emitted_once_basenames=emitted_epe_once_basenames,
        include_excluded_basenames={"001_regime_frequency_all_days_aligned_period.png"},
    )
    appended = 0
    for png in chosen:
        if pdf_next_figure_number_would_exceed_cap(fig_counter):
            break
        try:
            if fig_counter is not None:
                fig_counter[0] += 1
                append_png_page_a4(pdf, png, fig_num=fig_counter[0])
            else:
                append_png_page_a4(pdf, png, fig_num=None)
            appended += 1
            if emitted_basenames is not None:
                emitted_basenames.add(png.name.lower())
        except Exception:
            if fig_counter is not None:
                fig_counter[0] -= 1
    return appended


def _append_epe_notebook_png_pages(
    pdf: PdfPages,
    *,
    module_root: Path,
    region: str,
    fig_counter: list[int] | None,
    emitted_epe_once_basenames: set[str] | None = None,
    skip_basenames: set[str] | None = None,
    k: int | None = None,
    limit: int = 45,
    study_area: str | None = None,
) -> None:
    """Pick up EPE.4 notebook exports (connectivity, heatmaps, summary panels) as PNG pages."""
    pngs = _collect_epe_notebook_png_paths(
        module_root=module_root, region=region, study_area=study_area
    )
    if not pngs:
        return

    def _prio(p: Path) -> tuple[int, str]:
        n = p.name.lower()
        score = 0
        for key, w in (
            ("connect", 5),
            ("component", 5),
            ("fragment", 4),
            ("summary", 4),
            ("footprint", 3),
            ("area", 2),
            ("heatmap", 2),
            ("conting", 2),
            ("epe_04", 3),
        ):
            if key in n:
                score += w
        return (-score, p.name)

    pngs = filter_epe_summary_png_paths(
        sorted(pngs, key=_prio),
        k=k,
        emitted_once_basenames=emitted_epe_once_basenames,
        skip_basenames=skip_basenames,
    )
    for png in pngs[:limit]:
        if pdf_next_figure_number_would_exceed_cap(fig_counter):
            break
        try:
            if fig_counter is not None:
                fig_counter[0] += 1
                append_png_page_a4(pdf, png, fig_num=fig_counter[0])
            else:
                append_png_page_a4(pdf, png, fig_num=None)
        except Exception:
            if fig_counter is not None:
                fig_counter[0] -= 1


def load_epe_04_aligned_extremes(
    module_root: Path,
    region: str,
    reg_day_ds: xr.Dataset,
    *,
    study_area: str | None = None,
    extremes_spec: ExtremesSpec | None = None,
) -> tuple[xr.Dataset, xr.DataArray] | None:
    """Align extremes (or synthetic from duration) to daily regime labels for one bbox step."""
    extremes_paths = glob_extremes_nc(
        module_root, region, study_area=study_area, spec=extremes_spec
    )
    for p in extremes_paths:
        pair = _try_align_extremes_from_path(p, reg_day_ds)
        if pair is not None:
            return pair
    duration_paths = glob_convective_duration_nc(
        module_root, region, study_area=study_area, spec=extremes_spec
    )
    for p in duration_paths:
        pair = _try_align_synthetic_from_duration(p, reg_day_ds)
        if pair is not None:
            return pair
    return None


def labels_dur_from_reg_day(dur_c: xr.Dataset, reg_day_ds: xr.Dataset) -> xr.DataArray | None:
    """Daily regime labels on ``dur_c`` time axis (same calendar days as duration)."""
    dur_days = _calendar_day_coords_ns(dur_c["time"].values)
    common_d = np.intersect1d(dur_days, reg_day_ds["day"].values)
    if common_d.size == 0:
        return None
    dur_w = dur_c.assign_coords(day=("time", dur_days))
    dur_sub = dur_w.sel(time=dur_w["day"].isin(common_d))
    ld = reg_day_ds["regime_labels"].sel(day=dur_sub["day"].values)
    return xr.DataArray(
        ld.values.astype(int),
        coords={"time": dur_sub["time"].values},
        dims=("time",),
    )


def _try_load_aligned_duration(
    module_root: Path,
    region: str,
    reg_day_ds: xr.Dataset,
    *,
    study_area: str | None = None,
    extremes_spec: ExtremesSpec | None = None,
) -> tuple[xr.Dataset, xr.DataArray] | None:
    """Align ``convective_duration_*.nc`` to regime daily labels (same logic as duration boxplots)."""
    from epe_preprocessing.epe_functions import dataset_rename_time

    duration_paths = glob_convective_duration_nc(
        module_root, region, study_area=study_area, spec=extremes_spec
    )

    for p in duration_paths:
        try:
            with xr.open_dataset(p, chunks=None) as dur_raw:
                dur_w = dataset_rename_time(dur_raw)
                dur_days = _calendar_day_coords_ns(dur_w["time"].values)
                common_d = np.intersect1d(dur_days, reg_day_ds["day"].values)
                if common_d.size == 0:
                    continue
                dur_w = dur_w.assign_coords(day=("time", dur_days))
                dur_c = dur_w.sel(time=dur_w["day"].isin(common_d)).load()
                ld = reg_day_ds["regime_labels"].sel(day=dur_c["day"].values)
                labels_dur = xr.DataArray(
                    ld.values.astype(int),
                    coords={"time": dur_c["time"].values},
                    dims=("time",),
                )
                return dur_c, labels_dur
        except Exception:
            continue
    return None


def build_epe_04_summary_figure(
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    reg_day_ds: xr.Dataset,
    *,
    module_root: Path | None,
    region: str | None,
    study_area: str | None = None,
    extremes_spec: ExtremesSpec | None = None,
    occurrence_all_days: bool = False,
    panel_c_force_max_label: bool = False,
    panel_c_intensity_mode: PanelCIntensityMode = "hourly_max",
) -> Figure | None:
    """Build the 2×2 epe_04 summary (occurrence, persistence, intensity, connectivity).

    Pure figure construction — no saving/closing/``PdfPages`` — so a notebook can call this
    directly, same as :func:`build_boxplot_figure`.

    ``occurrence_all_days`` selects panel (A)'s scope: ``False`` keeps the existing
    extreme-day filter, ``True`` builds the relative-occurrence stack from every MJJAS day.
    ``panel_c_intensity_mode`` selects hourly-max (default) vs daily-max-on-flagged-cells
    for panel (C). ``panel_c_force_max_label`` applies to hourly mode only.
    """
    try:
        from epe_preprocessing.epe_functions import guess_lat_lon

        dur_pair = _try_load_aligned_duration(
            module_root,
            region or "",
            reg_day_ds,
            study_area=study_area,
            extremes_spec=extremes_spec,
        ) if module_root is not None else None
        dur_c = labels_dur = None
        n_components = None
        if dur_pair is not None:
            dur_c, labels_dur = dur_pair
            n_components = build_connectivity_n_components_from_duration(dur_c)

        lat_e = lon_e = None
        try:
            lat_e, lon_e = guess_lat_lon(ext_c)
        except Exception:
            pass

        return build_figure_epe_04_summary_occurrence_persistence_intensity_connectivity(
            ext_c=ext_c,
            labels_ext=labels_ext,
            dur_c=dur_c,
            labels_dur=labels_dur,
            n_components=n_components,
            lat_name=lat_e,
            lon_name=lon_e,
            module_root=module_root,
            study_area=study_area,
            occurrence_all_days=occurrence_all_days,
            panel_c_force_max_label=panel_c_force_max_label,
            panel_c_intensity_mode=panel_c_intensity_mode,
        )
    except Exception as exc:
        print(f"[epe_04 summary figure] failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def _save_epe_04_summary_connectivity_png(
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    reg_day_ds: xr.Dataset,
    *,
    module_root: Path | None,
    region: str | None,
    bbox_step: str | None,
    k: int | None,
    cluster_tag: str,
    study_area: str | None = None,
    extremes_spec: ExtremesSpec | None = None,
    occurrence_all_days: bool = False,
    panel_c_intensity_mode: PanelCIntensityMode = "hourly_max",
) -> Path | None:
    """Export the 2×2 epe_04 summary PNG for one bbox step and cluster source (step8 / step9).

    ``occurrence_all_days=True`` writes the all-days panel (A) variant alongside the existing
    extreme-days export (different filename stem so both PNGs coexist on disk).
    ``panel_c_intensity_mode="daily_max"`` adds ``_panel_c_daily_max`` to the stem (notebook metric).
    """
    if module_root is None or region is None or bbox_step is None or k is None:
        return None
    fig = build_epe_04_summary_figure(
        ext_c,
        labels_ext,
        reg_day_ds,
        module_root=module_root,
        region=region,
        study_area=study_area,
        extremes_spec=extremes_spec,
        occurrence_all_days=occurrence_all_days,
        panel_c_intensity_mode=panel_c_intensity_mode,
    )
    if fig is None:
        return None
    base_stem = EPE_04_SUMMARY_CONNECTIVITY_PNG_STEM
    scope_tag = "_all_days" if occurrence_all_days else ""
    panel_c_tag = (
        EPE_04_SUMMARY_PANEL_C_DAILY_MAX_SUFFIX
        if panel_c_intensity_mode == "daily_max"
        else ""
    )
    stem = f"{cluster_tag}_{base_stem}{panel_c_tag}{scope_tag}"
    try:
        return _save_selected_report_png(
            fig,
            module_root=module_root,
            region=region,
            bbox_step=bbox_step,
            k=int(k),
            stem_suffix=stem,
        )
    finally:
        plt.close(fig)


def _append_epe_04_summary_figure_page(
    pdf: PdfPages,
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    reg_day_ds: xr.Dataset,
    *,
    module_root: Path,
    region: str,
    study_area: str | None,
    fig_counter: list[int] | None,
    cluster_tag: str,
    extremes_spec: ExtremesSpec | None = None,
    occurrence_all_days: bool = False,
    section_letter: str = "d",
    panel_c_force_max_label: bool = False,
    panel_c_intensity_mode: PanelCIntensityMode = "hourly_max",
    respect_figure_cap: bool = True,
) -> bool:
    """Embed the 2×2 summary in the PDF (and export ``epe_04_summary_*.png`` when possible).

    ``occurrence_all_days`` toggles panel (A) between the all-days and extreme-days variants;
    title and intro text are updated to reflect the active scope.
    ``panel_c_intensity_mode`` selects hourly-max vs daily-max-on-flagged-cells for panel (C).
    ``panel_c_force_max_label`` keeps panel (C)'s title on the "hourly max" wording even
    when the underlying metric falls back to the proxy (hourly mode only).
    ``respect_figure_cap=False`` always embeds the 2×2 summary (used for e1/e2/e3/e4).
    """
    fig = build_epe_04_summary_figure(
        ext_c,
        labels_ext,
        reg_day_ds,
        module_root=module_root,
        region=region,
        study_area=study_area,
        extremes_spec=extremes_spec,
        occurrence_all_days=occurrence_all_days,
        panel_c_force_max_label=panel_c_force_max_label,
        panel_c_intensity_mode=panel_c_intensity_mode,
    )
    if respect_figure_cap and pdf_next_figure_number_would_exceed_cap(fig_counter):
        if fig is not None:
            plt.close(fig)
        return False
    scope_tag = "all days" if occurrence_all_days else "extreme days only"
    occurrence_blurb = (
        "Panel (A) shows relative occurrence over ALL MJJAS days "
        "(regime frequency × monthly composition)."
        if occurrence_all_days
        else "Panel (A) shows relative occurrence on extreme days only."
    )
    if panel_c_intensity_mode == "daily_max":
        panel_c_note = (
            " Panel (C) uses the daily max of ``ext_c['P']`` over ``E_conv``-flagged cells "
            "(same as ``epe_04_extreme_regimes_statistics.ipynb``)."
        )
        intro_heading = f"Summary overview ({scope_tag}, panel C: daily max)"
        intensity_blurb = "daily-max precipitation intensity on flagged cells"
    else:
        panel_c_note = (
            " Panel (C) reports the same hourly-max metric as the singular section (b) "
            "boxplot (``flagged_hourly_precip_max_per_day``); the proxy fallback is kept "
            "but labelled as 'hourly max'."
            if panel_c_force_max_label
            else ""
        )
        intro_heading = (
            f"Summary overview ({scope_tag}, panel C: hourly max)"
            if panel_c_force_max_label
            else f"Summary overview ({scope_tag})"
        )
        intensity_blurb = "hourly-max precipitation intensity on flagged cells"
    _epe_04_subsection_intro(
        pdf,
        section_letter,
        intro_heading,
        f"Four-panel summary: relative occurrence, persistence, {intensity_blurb}, "
        "and connectivity. "
        f"{occurrence_blurb} Panels (B)–(D) are always restricted to extreme days."
        f"{panel_c_note}",
    )
    try:
        if fig_counter is not None:
            fig_counter[0] += 1
            n = fig_counter[0]
            fig.suptitle(
                f"Figure {n} — epe_04 summary — occurrence ({scope_tag}), "
                "persistence, intensity, connectivity",
                fontsize=PDF_FS_FIG_SUPTITLE,
                y=0.99,
            )
        savefig_a4(pdf, fig, tight=True, pad_inches=0.12)
        return True
    finally:
        plt.close(fig)


def _append_named_epe_png_page(
    pdf: PdfPages,
    *,
    module_root: Path,
    region: str,
    basename: str,
    fig_counter: list[int] | None,
    emitted_epe_once_basenames: set[str] | None = None,
    k: int | None = None,
    study_area: str | None = None,
) -> bool:
    """Append exactly one notebook-exported EPE PNG by basename."""
    pngs = _collect_epe_notebook_png_paths(
        module_root=module_root, region=region, study_area=study_area
    )
    chosen = [p for p in pngs if p.name.lower() == basename.lower()]
    chosen = filter_epe_summary_png_paths(
        chosen,
        k=k,
        emitted_once_basenames=emitted_epe_once_basenames,
    )
    if not chosen or pdf_next_figure_number_would_exceed_cap(fig_counter):
        return False
    png = chosen[0]
    try:
        if fig_counter is not None:
            fig_counter[0] += 1
            append_png_page_a4(pdf, png, fig_num=fig_counter[0])
        else:
            append_png_page_a4(pdf, png, fig_num=None)
        return True
    except Exception:
        if fig_counter is not None:
            fig_counter[0] -= 1
        return False


def load_epe_04_statistics_inputs(
    module_root: Path,
    region: str,
    *,
    regimes_nc_path: Path | None = None,
    regimes_ds: xr.Dataset | None = None,
    study_area: str | None = None,
    extremes_spec: ExtremesSpec | None = None,
    source_label: str | None = None,
) -> dict:
    """Notebook-friendly load step for the Z500 EPE.4 statistics.

    Resolves the regime-label day-index plus the best-aligning extremes/duration dataset —
    same resolution order as :func:`append_epe_04_boxplot_pages` (``extremes_*.nc`` first,
    then synthetic extremes from ``convective_duration_*.nc``) — with no ``PdfPages``
    involved. Raises instead of writing a PDF fallback page when nothing aligns; a notebook
    wants a traceback naming exactly what's missing, not a rendered text page.
    """
    from epe_preprocessing.epe_functions import dataset_rename_time

    reg_path = Path(regimes_nc_path) if regimes_nc_path is not None else None
    if reg_path is None and regimes_ds is None:
        raise ValueError("Provide either regimes_nc_path or regimes_ds.")
    if reg_path is not None and not reg_path.is_file() and regimes_ds is None:
        raise FileNotFoundError(f"regimes_nc_path not found: {reg_path}")

    if regimes_ds is not None:
        reg_day_ds = _load_regime_day_index(dataset_rename_time(regimes_ds))
        reg_source_name = source_label or "provided regimes dataset"
    else:
        assert reg_path is not None
        with xr.open_dataset(reg_path, chunks=None) as reg_ds:
            reg_day_ds = _load_regime_day_index(dataset_rename_time(reg_ds))
        reg_source_name = source_label or reg_path.name

    if study_area:
        study_area = str(study_area).strip()

    ext_c: xr.Dataset | None = None
    labels_ext: xr.DataArray | None = None
    source_msg = ""
    duration_audit_path: Path | None = None

    if extremes_spec is not None and extremes_spec.duration_nc is not None:
        _explicit = Path(extremes_spec.duration_nc)
        if _explicit.is_file():
            duration_audit_path = _explicit.resolve()

    extremes_paths = glob_extremes_nc(module_root, region, study_area=study_area, spec=extremes_spec)
    for p in extremes_paths:
        pair = _try_align_extremes_from_path(p, reg_day_ds)
        if pair is None:
            continue
        ext_c, labels_ext = pair
        source_msg = f"extremes file: {p.name}"
        break

    if ext_c is None:
        duration_paths = glob_convective_duration_nc(module_root, region, study_area=study_area, spec=extremes_spec)
        for p in duration_paths:
            pair = _try_align_synthetic_from_duration(p, reg_day_ds)
            if pair is None:
                continue
            ext_c, labels_ext = pair
            source_msg = f"synthetic extremes from duration: {p.name}"
            duration_audit_path = p.resolve()
            break

    if ext_c is None or labels_ext is None:
        raise RuntimeError(
            "No extremes_*.nc or convective_duration_*.nc under "
            "data/processed/*/extreme_precip_events/ shares any calendar day with "
            f"regime_labels from {reg_source_name}."
        )

    if duration_audit_path is None or not duration_audit_path.is_file():
        _candidates = glob_convective_duration_nc(module_root, region, study_area=study_area, spec=extremes_spec)
        if _candidates:
            duration_audit_path = _candidates[0].resolve()

    regimes = np.unique(labels_ext.values)
    regimes = regimes[regimes > 0]

    return dict(
        module_root=module_root,
        region=region,
        study_area=study_area,
        extremes_spec=extremes_spec,
        reg_day_ds=reg_day_ds,
        reg_source_name=reg_source_name,
        ext_c=ext_c,
        labels_ext=labels_ext,
        source_msg=source_msg,
        duration_audit_path=duration_audit_path,
        regimes=regimes,
    )


def compute_epe_04_core_metrics(
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    reg_day_ds: xr.Dataset,
    *,
    module_root: Path,
    region: str,
    study_area: str | None,
    extremes_spec: ExtremesSpec | None,
    regimes: np.ndarray,
) -> dict:
    """Pure-compute counterpart to :func:`_append_epe_04_core_metric_boxplots`.

    Same six per-regime metric arrays (intensity mean/median, spatial mean/median, duration,
    connectivity) as the PDF boxplot pages, returned as plain
    ``{key: {"data", "ticks", "regime_ids", "title", "ylabel"}}`` instead of drawn into a
    ``PdfPages``. Best-effort: a metric key is simply absent if its inputs aren't available
    (e.g. no ``convective_duration_*.nc`` for this region) rather than raising — callers
    (e.g. a notebook) check for the key before building a figure from it.
    """
    from epe_preprocessing.epe_functions import dataset_rename_time

    metrics: dict[str, dict] = {}

    dur_for_hourly = _try_load_aligned_duration(
        module_root, region, reg_day_ds, study_area=study_area, extremes_spec=extremes_spec,
    )
    if dur_for_hourly is not None:
        dur_c_h, labels_h = dur_for_hourly
        try:
            from common.epe_figures import _resolve_hourly_intensity_for_summary
            from epe_preprocessing.epe_functions import _daily_extreme_mask

            ev_mask_base = _daily_extreme_mask(extremes_ds=ext_c, duration_ds=dur_c_h)

            for agg, key in (("mean", "intensity_mean"), ("median", "intensity_median")):
                p_h, _note = _resolve_hourly_intensity_for_summary(
                    dur_c_h, module_root=module_root, study_area=study_area, agg=agg,
                )
                if p_h is None:
                    continue
                p_h, labels_h_aligned = xr.align(p_h, labels_h, join="inner")
                ev_mask = ev_mask_base
                if ev_mask is None:
                    if "flagged_any_gridcell" in dur_c_h:
                        ev_mask = dur_c_h["flagged_any_gridcell"].astype(int) > 0
                    else:
                        ev_mask = p_h > 0
                ev_mask, _ = xr.align(ev_mask.astype(bool), labels_h_aligned, join="inner")

                if "proxy" in str(_note):
                    ylab = f"precip rate proxy — wet-cell {agg} (mm/h)"
                    title_h = f"Precip rate proxy — wet-cell {agg} by regime (extreme days only)"
                else:
                    ylab = f"hourly precip — wet-cell {agg} (mm/h)"
                    title_h = f"Hourly precip — wet-cell {agg} on flagged cells by regime (extreme days only)"

                d, ticks, reg_ids = [], [], []
                for r in regimes:
                    sel = (labels_h_aligned == int(r)) & ev_mask
                    vals = p_h.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d.append(vals)
                    ticks.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids.append(int(r))
                if d:
                    metrics[key] = dict(data=d, ticks=ticks, regime_ids=reg_ids, title=title_h, ylabel=ylab)
        except Exception:
            pass

    try:
        from common.epe_figures import _compute_flagged_daily_precip_spatial_stats

        p_mean_day, p_median_day, _stat_note = _compute_flagged_daily_precip_spatial_stats(ext_c)
        if p_mean_day is not None:
            day_flag_b1 = _event_day_mask(ext_c)
            for stat_da, key, stat_label, stat_title in (
                (
                    p_mean_day,
                    "spatial_mean",
                    "spatial mean precip on flagged cells (mm)",
                    "Mean P over E_conv-flagged cells by regime (extreme days only)",
                ),
                (
                    p_median_day,
                    "spatial_median",
                    "spatial median precip on flagged cells (mm)",
                    "Median P over E_conv-flagged cells by regime (extreme days only)",
                ),
            ):
                if stat_da is None:
                    continue
                d_stat, ticks_stat, reg_ids_stat = [], [], []
                for r in regimes:
                    if day_flag_b1 is not None:
                        sel = (labels_ext == int(r)) & day_flag_b1.astype(bool)
                    else:
                        sel = labels_ext == int(r)
                    vals = stat_da.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d_stat.append(vals)
                    ticks_stat.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids_stat.append(int(r))
                if d_stat:
                    metrics[key] = dict(
                        data=d_stat, ticks=ticks_stat, regime_ids=reg_ids_stat,
                        title=stat_title, ylabel=stat_label,
                    )
    except Exception:
        pass

    try:
        from common.epe_figures import _compute_flagged_daily_precip_max

        p_max_day, _max_note = _compute_flagged_daily_precip_max(ext_c)
        if p_max_day is not None:
            day_flag_max = _event_day_mask(ext_c)
            d_max, ticks_max, reg_ids_max = [], [], []
            for r in regimes:
                if day_flag_max is not None:
                    sel = (labels_ext == int(r)) & day_flag_max.astype(bool)
                else:
                    sel = labels_ext == int(r)
                vals = p_max_day.where(sel, drop=True).values
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                d_max.append(vals)
                ticks_max.append(f"{int(r)}\n(n={int(vals.size)})")
                reg_ids_max.append(int(r))
            if d_max:
                metrics["spatial_max"] = dict(
                    data=d_max, ticks=ticks_max, regime_ids=reg_ids_max,
                    title="Max P over E_conv-flagged cells by regime (extreme days only)",
                    ylabel="spatial max precip on flagged cells (mm)",
                )
    except Exception:
        pass

    duration_paths = glob_convective_duration_nc(module_root, region, study_area=study_area, spec=extremes_spec)
    for p in duration_paths:
        try:
            with xr.open_dataset(p, chunks=None) as dur_raw:
                dur_w = dataset_rename_time(dur_raw)
                dur_days = _calendar_day_coords_ns(dur_w["time"].values)
                common_d = np.intersect1d(dur_days, reg_day_ds["day"].values)
                if common_d.size == 0:
                    continue
                dur_w = dur_w.assign_coords(day=("time", dur_days))
                dur_c = dur_w.sel(time=dur_w["day"].isin(common_d))
                ld = reg_day_ds["regime_labels"].sel(day=dur_c["day"].values)
                labels_dur = xr.DataArray(
                    ld.values.astype(int),
                    coords={"time": dur_c["time"].values},
                    dims=("time",),
                )
                regimes_d = np.unique(labels_dur.values)
                regimes_d = regimes_d[regimes_d > 0]

                if "n_conv_hours_grid" not in dur_c:
                    continue
                n_grid = dur_c["n_conv_hours_grid"].astype(float)
                spatial_dims_d = tuple(d for d in n_grid.dims if d != "time")
                is_conv = n_grid > 0
                mean_h = n_grid.where(is_conv).mean(dim=spatial_dims_d, skipna=True)
                day_flag = _event_day_mask(ext_c)
                if day_flag is None:
                    if "flagged_any_gridcell" in dur_c:
                        day_flag = dur_c["flagged_any_gridcell"].astype(int) > 0
                    else:
                        day_flag = mean_h > 0
                d2, ticks2, reg_ids2 = [], [], []
                for r in regimes_d:
                    sel = (labels_dur == int(r)) & day_flag.astype(bool)
                    vals = mean_h.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d2.append(vals)
                    ticks2.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids2.append(int(r))
                if d2:
                    metrics["duration"] = dict(
                        data=d2, ticks=ticks2, regime_ids=reg_ids2,
                        title="Convective duration per flagged grid cell (extreme days only)",
                        ylabel="hours per flagged cell per day",
                    )
                break
        except Exception:
            continue

    for p in duration_paths:
        try:
            with xr.open_dataset(p, chunks=None) as dur_raw:
                dur_w = dataset_rename_time(dur_raw)
                dur_days = _calendar_day_coords_ns(dur_w["time"].values)
                common_d = np.intersect1d(dur_days, reg_day_ds["day"].values)
                if common_d.size == 0:
                    continue
                dur_w = dur_w.assign_coords(day=("time", dur_days))
                dur_c3 = dur_w.sel(time=dur_w["day"].isin(common_d))
                ld = reg_day_ds["regime_labels"].sel(day=dur_c3["day"].values)
                labels_dur3 = xr.DataArray(
                    ld.values.astype(int),
                    coords={"time": dur_c3["time"].values},
                    dims=("time",),
                )
                n_components = build_connectivity_n_components_from_duration(dur_c3)
                if n_components is None:
                    continue
                regimes_c = np.unique(labels_dur3.values)
                regimes_c = regimes_c[regimes_c > 0]
                day_flag_c = _event_day_mask(ext_c)
                if day_flag_c is None:
                    if "flagged_any_gridcell" in dur_c3:
                        day_flag_c = dur_c3["flagged_any_gridcell"].astype(int) > 0
                    else:
                        day_flag_c = n_components > 0
                day_flag_c, labels_dur3, n_components = xr.align(
                    day_flag_c.astype(bool),
                    labels_dur3,
                    n_components,
                    join="inner",
                )
                d3, ticks3, reg_ids3 = [], [], []
                for r in regimes_c:
                    sel = (labels_dur3 == int(r)) & day_flag_c
                    vals = n_components.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d3.append(vals)
                    ticks3.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids3.append(int(r))
                if d3:
                    metrics["connectivity"] = dict(
                        data=d3, ticks=ticks3, regime_ids=reg_ids3,
                        title="Daily connectivity per regime — # of components (extreme days only)",
                        ylabel="# components / day",
                    )
                break
        except Exception:
            continue

    for p in duration_paths:
        try:
            with xr.open_dataset(p, chunks=None) as dur_raw:
                dur_w = dataset_rename_time(dur_raw)
                dur_days = _calendar_day_coords_ns(dur_w["time"].values)
                common_d = np.intersect1d(dur_days, reg_day_ds["day"].values)
                if common_d.size == 0:
                    continue
                dur_w = dur_w.assign_coords(day=("time", dur_days))
                dur_c4 = dur_w.sel(time=dur_w["day"].isin(common_d))
                ld = reg_day_ds["regime_labels"].sel(day=dur_c4["day"].values)
                labels_dur4 = xr.DataArray(
                    ld.values.astype(int),
                    coords={"time": dur_c4["time"].values},
                    dims=("time",),
                )
                patch_size = build_extent_mean_patch_size_from_duration(dur_c4)
                if patch_size is None:
                    continue
                regimes_e = np.unique(labels_dur4.values)
                regimes_e = regimes_e[regimes_e > 0]
                day_flag_e = _event_day_mask(ext_c)
                if day_flag_e is None:
                    if "flagged_any_gridcell" in dur_c4:
                        day_flag_e = dur_c4["flagged_any_gridcell"].astype(int) > 0
                    else:
                        day_flag_e = patch_size > 0
                day_flag_e, labels_dur4, patch_size = xr.align(
                    day_flag_e.astype(bool),
                    labels_dur4,
                    patch_size,
                    join="inner",
                )
                d4, ticks4, reg_ids4 = [], [], []
                for r in regimes_e:
                    sel = (labels_dur4 == int(r)) & day_flag_e
                    vals = patch_size.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d4.append(vals)
                    ticks4.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids4.append(int(r))
                if d4:
                    metrics["extent"] = dict(
                        data=d4, ticks=ticks4, regime_ids=reg_ids4,
                        title="EPE spatial-footprint size per regime — mean 8-connected patch size (extreme days only)",
                        ylabel="mean patch size (# grid cells, 8-connectivity)",
                    )
                break
        except Exception:
            continue

    return metrics


def _append_epe_04_core_metric_boxplots(
    pdf: PdfPages,
    *,
    ext_c: xr.Dataset,
    labels_ext: xr.DataArray,
    reg_day_ds: xr.Dataset,
    module_root: Path,
    region: str,
    study_area: str | None,
    extremes_spec: ExtremesSpec | None,
    regimes: np.ndarray,
    source_msg: str,
    fig_counter: list[int] | None,
) -> tuple[int, bool, bool, bool]:
    """Singular EPE.4 boxplots: (b) intensity, (c) duration, (d) connectivity.

    Rendered before optional occurrence PNGs so intensity / persistence / connectivity
    are always present in the PDF when the aligned duration product supports them.
    """
    from epe_preprocessing.epe_functions import dataset_rename_time

    plots = 0
    intensity_drawn = False
    persistence_drawn = False
    connectivity_drawn = False

    _epe_04_subsection_intro(
        pdf,
        "b",
        "Intensity of precipitation",
        "Mean and median hourly precipitation on wet convective-flagged cells (extreme days only). "
        "Wet-cell conditioning (P > 0) avoids diluting precipitation statistics with dry boundary cells. "
        "Statistics that average over non-flagged days are omitted.",
    )

    dur_for_hourly = _try_load_aligned_duration(
        module_root,
        region,
        reg_day_ds,
        study_area=study_area,
        extremes_spec=extremes_spec,
    )
    if dur_for_hourly is not None:
        dur_c_h, labels_h = dur_for_hourly
        try:
            from common.epe_figures import _resolve_hourly_intensity_for_summary
            from epe_preprocessing.epe_functions import _daily_extreme_mask

            ev_mask_base = _daily_extreme_mask(extremes_ds=ext_c, duration_ds=dur_c_h)

            for agg in ("mean", "median"):
                p_h, _note = _resolve_hourly_intensity_for_summary(
                    dur_c_h,
                    module_root=module_root,
                    study_area=study_area,
                    agg=agg,
                )
                if p_h is not None:
                    p_h, labels_h_aligned = xr.align(p_h, labels_h, join="inner")
                    ev_mask = ev_mask_base
                    if ev_mask is None:
                        if "flagged_any_gridcell" in dur_c_h:
                            ev_mask = dur_c_h["flagged_any_gridcell"].astype(int) > 0
                        else:
                            ev_mask = p_h > 0
                    ev_mask, _ = xr.align(ev_mask.astype(bool), labels_h_aligned, join="inner")

                    if "proxy" in str(_note):
                        ylab = f"precip rate proxy — wet-cell {agg} (mm/h)"
                        title_h = f"Precip rate proxy — wet-cell {agg} by regime (extreme days only)"
                    else:
                        ylab = f"hourly precip — wet-cell {agg} (mm/h)"
                        title_h = f"Hourly precip — wet-cell {agg} on flagged cells by regime (extreme days only)"

                    d, ticks, reg_ids = [], [], []
                    for r in regimes:
                        sel = (labels_h_aligned == int(r)) & ev_mask
                        vals = p_h.where(sel, drop=True).values
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        d.append(vals)
                        ticks.append(f"{int(r)}\n(n={int(vals.size)})")
                        reg_ids.append(int(r))
                    if _boxplot_to_pdf(
                        pdf,
                        d,
                        ticks,
                        title=title_h,
                        ylabel=ylab,
                        fig_size=(10.0, 5.2),
                        fig_counter=fig_counter,
                        regime_ids=reg_ids,
                    ):
                        plots += 1
                        intensity_drawn = True
                else:
                    _append_text_pdf_page(
                        pdf,
                        f"EPE.4 — (b) hourly intensity ({agg}) unavailable",
                        f"Could not resolve an hourly precipitation field for the aligned duration "
                        f"({source_msg}). Need ``E_conv_hour`` or a sibling hourly precip NetCDF.",
                    )
        except Exception as e:
            _append_text_pdf_page(
                pdf,
                "EPE.4 — (b) hourly intensity failed",
                f"{type(e).__name__}: {e}",
            )
    else:
        _append_text_pdf_page(
            pdf,
            "EPE.4 — (b) (no duration for hourly intensity)",
            f"Aligned dataset ({source_msg}) has no overlapping ``convective_duration`` file.",
        )

    _epe_04_subsection_intro(
        pdf,
        "b1",
        "Spatial mean / median precipitation on flagged cells",
        "Mean and median of daily precipitation (P) over E_conv-flagged cells on extreme event "
        "days only. Unlike the hourly max, these average over all flagged cells and are more "
        "robust to single-cell outliers — better for comparing typical event intensity across regimes.",
    )
    try:
        from common.epe_figures import _compute_flagged_daily_precip_spatial_stats

        p_mean_day, p_median_day, _stat_note = _compute_flagged_daily_precip_spatial_stats(ext_c)
        if p_mean_day is not None:
            day_flag_b1 = _event_day_mask(ext_c)
            for stat_da, stat_label, stat_title in (
                (
                    p_mean_day,
                    "spatial mean precip on flagged cells (mm)",
                    "Mean P over E_conv-flagged cells by regime (extreme days only)",
                ),
                (
                    p_median_day,
                    "spatial median precip on flagged cells (mm)",
                    "Median P over E_conv-flagged cells by regime (extreme days only)",
                ),
            ):
                if stat_da is None:
                    continue
                d_stat, ticks_stat, reg_ids_stat = [], [], []
                for r in regimes:
                    if day_flag_b1 is not None:
                        sel = (labels_ext == int(r)) & day_flag_b1.astype(bool)
                    else:
                        sel = labels_ext == int(r)
                    vals = stat_da.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d_stat.append(vals)
                    ticks_stat.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids_stat.append(int(r))
                if _boxplot_to_pdf(
                    pdf,
                    d_stat,
                    ticks_stat,
                    title=stat_title,
                    ylabel=stat_label,
                    fig_size=(10.0, 5.2),
                    fig_counter=fig_counter,
                    regime_ids=reg_ids_stat,
                ):
                    plots += 1
        else:
            _append_text_pdf_page(
                pdf,
                "EPE.4 — (b1) spatial stats unavailable",
                f"Could not compute spatial mean/median: {_stat_note}. "
                "Need 'P' and 'E_conv' in the aligned extremes dataset.",
            )
    except Exception as _e_b1:
        _append_text_pdf_page(
            pdf,
            "EPE.4 — (b1) spatial stats failed",
            f"{type(_e_b1).__name__}: {_e_b1}",
        )

    _epe_04_subsection_intro(
        pdf,
        "c",
        "Persistence and duration",
        "Notebook section (B): convective duration (hours per flagged grid cell on each day) "
        "versus regime. The boxplot below follows the duration-by-regime logic in epe_04.",
    )

    duration_paths = glob_convective_duration_nc(
        module_root, region, study_area=study_area, spec=extremes_spec
    )
    for p in duration_paths:
        try:
            with xr.open_dataset(p, chunks=None) as dur_raw:
                dur_w = dataset_rename_time(dur_raw)
                dur_days = _calendar_day_coords_ns(dur_w["time"].values)
                common_d = np.intersect1d(dur_days, reg_day_ds["day"].values)
                if common_d.size == 0:
                    continue
                dur_w = dur_w.assign_coords(day=("time", dur_days))
                dur_c = dur_w.sel(time=dur_w["day"].isin(common_d))
                ld = reg_day_ds["regime_labels"].sel(day=dur_c["day"].values)
                labels_dur = xr.DataArray(
                    ld.values.astype(int),
                    coords={"time": dur_c["time"].values},
                    dims=("time",),
                )
                regimes_d = np.unique(labels_dur.values)
                regimes_d = regimes_d[regimes_d > 0]

                if "n_conv_hours_grid" not in dur_c:
                    continue
                n_grid = dur_c["n_conv_hours_grid"].astype(float)
                spatial_dims = tuple(d for d in n_grid.dims if d != "time")
                is_conv = n_grid > 0
                mean_h = n_grid.where(is_conv).mean(dim=spatial_dims, skipna=True)
                day_flag = _event_day_mask(ext_c)
                if day_flag is None:
                    if "flagged_any_gridcell" in dur_c:
                        day_flag = dur_c["flagged_any_gridcell"].astype(int) > 0
                    else:
                        day_flag = mean_h > 0
                d2, ticks2, reg_ids2 = [], [], []
                for r in regimes_d:
                    sel = (labels_dur == int(r)) & day_flag.astype(bool)
                    vals = mean_h.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d2.append(vals)
                    ticks2.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids2.append(int(r))
                if _boxplot_to_pdf(
                    pdf,
                    d2,
                    ticks2,
                    title="Convective duration per flagged grid cell (extreme days only)",
                    ylabel="hours per flagged cell per day",
                    fig_size=(10.0, 5.2),
                    fig_counter=fig_counter,
                    regime_ids=reg_ids2,
                ):
                    plots += 1
                    persistence_drawn = True
                break
        except Exception:
            continue
    if not persistence_drawn:
        _append_text_pdf_page(
            pdf,
            "EPE.4 — (c) persistence boxplot unavailable",
            f"Aligned dataset ({source_msg}) has no overlapping ``convective_duration_*.nc`` "
            "with ``n_conv_hours_grid``. Re-run EPE.1b for this region / year span to "
            "produce the convective duration file, then rebuild the report.",
        )

    _epe_04_subsection_intro(
        pdf,
        "d",
        "Connectivity per regime",
        "Notebook section (D): daily count of connected components (8-neighbour) over the "
        "convective-flagged grid on extreme days. Higher counts indicate a more fragmented "
        "footprint, lower counts a more contiguous footprint.",
    )

    for p in duration_paths:
        try:
            with xr.open_dataset(p, chunks=None) as dur_raw:
                dur_w = dataset_rename_time(dur_raw)
                dur_days = _calendar_day_coords_ns(dur_w["time"].values)
                common_d = np.intersect1d(dur_days, reg_day_ds["day"].values)
                if common_d.size == 0:
                    continue
                dur_w = dur_w.assign_coords(day=("time", dur_days))
                dur_c3 = dur_w.sel(time=dur_w["day"].isin(common_d))
                ld = reg_day_ds["regime_labels"].sel(day=dur_c3["day"].values)
                labels_dur3 = xr.DataArray(
                    ld.values.astype(int),
                    coords={"time": dur_c3["time"].values},
                    dims=("time",),
                )
                n_components = build_connectivity_n_components_from_duration(dur_c3)
                if n_components is None:
                    continue
                regimes_c = np.unique(labels_dur3.values)
                regimes_c = regimes_c[regimes_c > 0]
                day_flag_c = _event_day_mask(ext_c)
                if day_flag_c is None:
                    if "flagged_any_gridcell" in dur_c3:
                        day_flag_c = dur_c3["flagged_any_gridcell"].astype(int) > 0
                    else:
                        day_flag_c = n_components > 0
                day_flag_c, labels_dur3, n_components = xr.align(
                    day_flag_c.astype(bool),
                    labels_dur3,
                    n_components,
                    join="inner",
                )
                d3, ticks3, reg_ids3 = [], [], []
                for r in regimes_c:
                    sel = (labels_dur3 == int(r)) & day_flag_c
                    vals = n_components.where(sel, drop=True).values
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    d3.append(vals)
                    ticks3.append(f"{int(r)}\n(n={int(vals.size)})")
                    reg_ids3.append(int(r))
                if _boxplot_to_pdf(
                    pdf,
                    d3,
                    ticks3,
                    title="Daily connectivity per regime — # of components (extreme days only)",
                    ylabel="# components / day",
                    regime_ids=reg_ids3,
                    fig_size=(10.0, 5.2),
                    fig_counter=fig_counter,
                ):
                    plots += 1
                    connectivity_drawn = True
                break
        except Exception:
            continue
    if not connectivity_drawn:
        _append_text_pdf_page(
            pdf,
            "EPE.4 — (d) connectivity boxplot unavailable",
            f"Aligned dataset ({source_msg}) has no overlapping ``convective_duration_*.nc`` "
            "with a usable hourly convective field (need ``E_conv_hour`` or per-hour info to "
            "compute daily connected components on flagged cells).",
        )

    return plots, intensity_drawn, persistence_drawn, connectivity_drawn


def append_epe_04_boxplot_pages(
    pdf: PdfPages,
    *,
    module_root: Path,
    region: str,
    regimes_nc_path: Path | None = None,
    regimes_ds: xr.Dataset | None = None,
    study_area: str | None = None,
    extremes_spec: ExtremesSpec | None = None,
    fig_counter: list[int] | None = None,
    emitted_epe_once_basenames: set[str] | None = None,
    k: int | None = None,
    bbox_step: str | None = None,
    include_notebook_pngs: bool = True,
    source_label: str | None = None,
    extra_summary_panel_c_max_label: bool = False,
    extra_summary_panel_c_daily_max: bool = True,
    include_summary_png: bool = False,
    save_summary_png: bool = False,
    summary_cluster_tag: str = "step9",
) -> None:
    """
    Add PDF pages mirroring key boxplots from ``epe_04_extreme_regimes_statistics.ipynb``.

    Resolves **extremes** NetCDF under ``data/processed/{region}/extreme_precip_events/`` and, if none
    overlap Step 9 daily labels in time, tries **other** processed subfolders (e.g. ``germany``) so
    summaries still work when EPE was run for a sub-domain label.

    Falls back to **synthetic** extremes from ``convective_duration_*.nc`` when no ``extremes_*.nc`` aligns.

    Optional ``extremes_spec`` matches epe_01b filename tags (``_q999_``, ``sampleWetmm01``, ``years_*``)
    instead of requiring ``--duration-nc``.

    ``extra_summary_panel_c_max_label=True`` (Step 8 / Ward) appends ``e3``: all-days panel (A),
    panel (C) labelled as hourly max regardless of proxy fallback.

    ``extra_summary_panel_c_daily_max=True`` (default) appends ``e4``: extreme-days panel (A),
    panel (C) = daily max of ``P`` over ``E_conv``-flagged cells (notebook metric). Also exports
    ``*_panel_c_daily_max*.png`` when ``save_summary_png=True``.
    """
    lag_days = 3
    try:
        from epe_preprocessing.epe_functions import (
            dataset_rename_time,
            guess_lat_lon,
        )
    except ImportError as e:
        _append_text_pdf_page(
            pdf,
            "EPE.4 — boxplots skipped",
            f"Could not import epe_preprocessing helpers: {e}",
        )
        return

    reg_path = Path(regimes_nc_path) if regimes_nc_path is not None else None
    if reg_path is None and regimes_ds is None:
        raise ValueError("Provide either regimes_nc_path or regimes_ds.")
    if reg_path is not None and not reg_path.is_file() and regimes_ds is None:
        return

    try:
        if regimes_ds is not None:
            reg_day_ds = _load_regime_day_index(dataset_rename_time(regimes_ds))
            reg_source_name = source_label or "provided regimes dataset"
        else:
            assert reg_path is not None
            with xr.open_dataset(reg_path, chunks=None) as reg_ds:
                reg_day_ds = _load_regime_day_index(dataset_rename_time(reg_ds))
            reg_source_name = source_label or reg_path.name
    except Exception as e:
        _append_text_pdf_page(
            pdf,
            "EPE.4 — regimes load failed",
            f"{type(e).__name__}: {e}\n"
            "Regimes data must expose ``regime_labels`` on a ``time`` or ``valid_time`` axis.",
        )
        return

    emitted_png_basenames: set[str] = set()
    plots = 0
    ext_c: xr.Dataset | None = None
    labels_ext: xr.DataArray | None = None
    source_msg = ""
    duration_audit_path: Path | None = None

    if study_area:
        study_area = str(study_area).strip()

    if extremes_spec is not None and extremes_spec.duration_nc is not None:
        _explicit = Path(extremes_spec.duration_nc)
        if _explicit.is_file():
            duration_audit_path = _explicit.resolve()

    spec_hint = extremes_spec.summary_label() if extremes_spec is not None else "(newest aligning file)"
    extremes_paths = glob_extremes_nc(
        module_root, region, study_area=study_area, spec=extremes_spec
    )

    for p in extremes_paths:
        pair = _try_align_extremes_from_path(p, reg_day_ds)
        if pair is None:
            continue
        ext_c, labels_ext = pair
        source_msg = f"extremes file: {p.name}"
        break

    if ext_c is None:
        duration_paths = glob_convective_duration_nc(
            module_root, region, study_area=study_area, spec=extremes_spec
        )

        for p in duration_paths:
            pair = _try_align_synthetic_from_duration(p, reg_day_ds)
            if pair is None:
                continue
            ext_c, labels_ext = pair
            source_msg = f"synthetic extremes from duration: {p.name}"
            duration_audit_path = p.resolve()
            break

    if ext_c is None or labels_ext is None:
        spec_detail = ""
        if extremes_spec is not None and extremes_spec.is_active():
            spec_detail = f"\nEPE selection: {spec_hint}\n"
            if extremes_spec.duration_nc is not None:
                spec_detail += f"  --duration-nc: {extremes_spec.duration_nc}\n"
            dirs = _events_dirs(module_root, study_area=study_area, region=region)
            if dirs:
                spec_detail += f"  searched: {', '.join(str(d) for d in dirs)}\n"
        _append_text_pdf_page(
            pdf,
            "EPE.4 — no aligned extremes/duration data",
            "No ``extremes_*.nc`` or ``convective_duration_*.nc`` under ``data/processed/*/extreme_precip_events/`` "
            "shares any calendar day with ``regime_labels`` from the regimes dataset used for this bbox step.\n"
            f"Regimes source: {reg_source_name}\n"
            f"{spec_detail}"
            "Run EPE.1b for the same region and year span, or relax / adjust --extremes-q, "
            "--quantile-sample, --extremes-year-span.",
        )
        return

    if duration_audit_path is None or not duration_audit_path.is_file():
        _candidates = glob_convective_duration_nc(
            module_root, region, study_area=study_area, spec=extremes_spec
        )
        if _candidates:
            duration_audit_path = _candidates[0].resolve()

    _append_duration_audit_pdf_page(
        pdf,
        duration_audit_path,
        regimes_source=reg_source_name,
        extremes_source=source_msg,
        cluster_tag=summary_cluster_tag,
    )

    if save_summary_png:
        _panel_c_modes: tuple[PanelCIntensityMode, ...] = ("hourly_max", "daily_max")
        for _scope_all_days in (True, False):
            for _panel_c_mode in _panel_c_modes:
                if _panel_c_mode == "daily_max" and not extra_summary_panel_c_daily_max:
                    continue
                _save_epe_04_summary_connectivity_png(
                    ext_c,
                    labels_ext,
                    reg_day_ds,
                    module_root=module_root,
                    region=region,
                    bbox_step=bbox_step,
                    k=k,
                    cluster_tag=summary_cluster_tag,
                    study_area=study_area,
                    extremes_spec=extremes_spec,
                    occurrence_all_days=_scope_all_days,
                    panel_c_intensity_mode=_panel_c_mode,
                )

    regimes = np.unique(labels_ext.values)
    regimes = regimes[regimes > 0]

    _core_plots, _intensity_ok, _persist_ok, _conn_ok = _append_epe_04_core_metric_boxplots(
        pdf,
        ext_c=ext_c,
        labels_ext=labels_ext,
        reg_day_ds=reg_day_ds,
        module_root=module_root,
        region=region,
        study_area=study_area,
        extremes_spec=extremes_spec,
        regimes=regimes,
        source_msg=source_msg,
        fig_counter=fig_counter,
    )
    plots += _core_plots

    _epe_04_subsection_intro(
        pdf,
        "a",
        "Relative occurrence",
        (
            "Notebook section (A): how often extremes co-occur with each weather regime. "
            "The original occurrence panels are included again below, followed by a precursor view "
            f"showing which regime was present {lag_days} days before each event day."
        )
        if include_notebook_pngs
        else (
            "Occurrence and precursor diagnostics conditioned on the supplied regime labels. "
            f"The heatmap below shows which regime was present {lag_days} days before each event day."
        ),
    )

    if include_notebook_pngs:
        plots += _append_epe_04_section_a_png_pages(
            pdf,
            module_root=module_root,
            region=region,
            fig_counter=fig_counter,
            emitted_epe_once_basenames=emitted_epe_once_basenames,
            emitted_basenames=emitted_png_basenames,
            k=k,
            study_area=study_area,
        )

    if _append_precursor_regime_heatmap_page(
        pdf,
        reg_day_ds=reg_day_ds,
        ext_c=ext_c,
        labels_ext=labels_ext,
        lag_days=lag_days,
        fig_counter=fig_counter,
    ):
        plots += 1
    else:
        _append_text_pdf_page(
            pdf,
            "EPE.4 — (a) precursor regimes unavailable",
            f"Could not build the t-{lag_days} precursor-regime heatmap from {source_msg}. "
            "The aligned dataset has no usable day-level event flag, or the Step 9 labels do not "
            f"extend {lag_days} days before any event day.",
        )

    if save_summary_png:
        # Print the 2×2 summary twice: first with panel (A) over ALL days, then with the
        # original extreme-days-only variant (panels B–D stay extreme-days in both).
        # (e1)/(e2) — singular connectivity now owns (d), so the in-PDF 2×2 moves down.
        for _scope_all_days, _section_letter in ((True, "e1"), (False, "e2")):
            _append_epe_04_summary_figure_page(
                pdf,
                ext_c,
                labels_ext,
                reg_day_ds,
                module_root=module_root,
                region=region,
                study_area=study_area,
                fig_counter=fig_counter,
                cluster_tag=summary_cluster_tag,
                extremes_spec=extremes_spec,
                occurrence_all_days=_scope_all_days,
                section_letter=_section_letter,
                respect_figure_cap=False,
            )
        # (e3) — additional all-days summary with panel (C) explicitly labelled
        # 'hourly max'. Currently emitted only when the caller opts in (Step 8 /
        # Ward in both the legacy and multivariate pipelines).
        if extra_summary_panel_c_max_label:
            _append_epe_04_summary_figure_page(
                pdf,
                ext_c,
                labels_ext,
                reg_day_ds,
                module_root=module_root,
                region=region,
                study_area=study_area,
                fig_counter=fig_counter,
                cluster_tag=summary_cluster_tag,
                extremes_spec=extremes_spec,
                occurrence_all_days=True,
                section_letter="e3",
                panel_c_force_max_label=True,
                respect_figure_cap=False,
            )
        if extra_summary_panel_c_daily_max:
            _append_epe_04_summary_figure_page(
                pdf,
                ext_c,
                labels_ext,
                reg_day_ds,
                module_root=module_root,
                region=region,
                study_area=study_area,
                fig_counter=fig_counter,
                cluster_tag=summary_cluster_tag,
                extremes_spec=extremes_spec,
                occurrence_all_days=False,
                section_letter="e4",
                panel_c_intensity_mode="daily_max",
                respect_figure_cap=False,
            )

    if include_notebook_pngs:
        _epe_04_subsection_intro(
            pdf,
            "f",
            "Affected area and connectivity",
            "Notebook section (D): spatial footprint of extremes, affected area, and connectivity / "
            "fragmentation metrics. PNGs saved by epe_04 (e.g. under figures/extreme_precip_events/) "
            "are appended below when present.",
        )

        _append_epe_notebook_png_pages(
            pdf,
            module_root=module_root,
            region=region,
            fig_counter=fig_counter,
            emitted_epe_once_basenames=emitted_epe_once_basenames,
            skip_basenames=emitted_png_basenames,
            k=k,
            study_area=study_area,
        )
    elif include_summary_png:
        _epe_04_subsection_intro(
            pdf,
            "f",
            "Summary overview",
            "Four-panel summary exported from the EPE.4 notebook: relative occurrence, persistence, "
            "precipitation intensity, and connectivity / affected-area diagnostics.",
        )
        _append_named_epe_png_page(
            pdf,
            module_root=module_root,
            region=region,
            basename="epe_04_summary_occurrence_persistence_intensity_connectivity.png",
            fig_counter=fig_counter,
            emitted_epe_once_basenames=emitted_epe_once_basenames,
            k=k,
            study_area=study_area,
        )

    if plots == 0 and not (_intensity_ok or _persist_ok or _conn_ok):
        _append_text_pdf_page(
            pdf,
            "EPE.4 — no boxplot figures emitted",
            f"Data source used for alignment: {source_msg}\n"
            "Intensity, persistence (duration), and connectivity boxplots could not be drawn. "
            "Check that ``convective_duration_*.nc`` overlaps regime labels and includes "
            "``n_conv_hours_grid`` and hourly convective fields (``E_conv_hour``) for connectivity.",
        )
