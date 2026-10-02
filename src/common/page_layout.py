"""Shared PDF page geometry (ISO A4 portrait, inches)."""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# ISO A4 portrait (inches) for matplotlib figure size / PDF mediabox consistency
A4_W_IN = 8.27
A4_H_IN = 11.69

FIG_A4_PORTRAIT: tuple[float, float] = (A4_W_IN, A4_H_IN)

# --- PDF typography (matplotlib ``fontsize`` in points) ---
# Single tuning point for bbox summary PDFs: text pages, tables, section breaks, and embedded figures.
# Plot typography (suptitle/title/axis/tick/clabel/annot/footer/legend/strip-panel)
# is intentionally uniform across every figure builder in the project:
#   - titles (figure suptitle + axes title): PDF_FS_FIG_TITLE_PT (16 pt)
#   - axis labels, ticks, legend, in-plot text (footer, clabel, annotations,
#     panel labels): PDF_FS_FIG_BODY_PT (14 pt)
# Text/table pages (cover, guides, section breaks) keep their own sizes.
PDF_FS_FIG_TITLE_PT = 16
PDF_FS_FIG_BODY_PT = 14

PDF_FS_BODY = 12
PDF_FS_GUIDE = 11
PDF_FS_SECTION_TITLE = 22
PDF_FS_SECTION_SUBTITLE = 13
PDF_FS_TABLE_TITLE = 13
PDF_FS_TABLE_CELL = 10
PDF_FS_TABLE_CELL_DENSE = 9
PDF_FS_COVER_META = 12
PDF_FS_MERGED_TITLE = 14
PDF_FS_PNG_CAPTION = PDF_FS_FIG_BODY_PT
# Matplotlib figures produced for the same PDFs (``regime_plotting`` builders).
# Titles use PDF_FS_FIG_TITLE_PT; everything else (axis labels, ticks, legend,
# contour labels, in-plot annotations, footers, panel headers) uses PDF_FS_FIG_BODY_PT.
PDF_FS_FIG_SUPTITLE = PDF_FS_FIG_TITLE_PT
PDF_FS_FIG_TITLE = PDF_FS_FIG_TITLE_PT
PDF_FS_FIG_AXIS = PDF_FS_FIG_BODY_PT
PDF_FS_FIG_TICK = PDF_FS_FIG_BODY_PT
PDF_FS_FIG_CLABEL = PDF_FS_FIG_BODY_PT
PDF_FS_FIG_ANNOT = PDF_FS_FIG_BODY_PT
PDF_FS_FIG_FOOTER = PDF_FS_FIG_BODY_PT
PDF_FS_FIG_LEGEND = PDF_FS_FIG_BODY_PT
PDF_FS_FIG_STRIP_PANEL = PDF_FS_FIG_TITLE_PT


# Preferred sans-serif chain for the project. Helvetica is the design target;
# Arial / Liberation Sans / DejaVu Sans are visually close fallbacks that ship
# with most Linux scientific stacks (TU HPC included).
PDF_FONT_SANS_SERIF: tuple[str, ...] = (
    "Helvetica",
    "Helvetica Neue",
    "Arial",
    "Liberation Sans",
    "Nimbus Sans",
    "DejaVu Sans",
    "sans-serif",
)


def configure_matplotlib_typography(
    *,
    title_pt: int = PDF_FS_FIG_TITLE_PT,
    body_pt: int = PDF_FS_FIG_BODY_PT,
    font_family: tuple[str, ...] = PDF_FONT_SANS_SERIF,
) -> None:
    """Apply project-wide matplotlib typography (Helvetica + uniform sizes).

    Called automatically on import so any ``Figure`` constructed afterwards by
    ``regime_plotting`` builders, summary-report PDF code, or epe_06 EPE
    composites inherits the same look. Notebooks that want the same defaults
    only need to ``from common.page_layout import …`` (any import of
    this module triggers the configuration) or call this function explicitly.

    Title text (figure suptitle + axes title) uses ``title_pt``; axis labels,
    tick labels, legends, contour labels, in-plot text and panel headers use
    ``body_pt``. PDF/PS font types are set to 42 so the embedded font is a
    TrueType subset (editable in Illustrator / Inkscape).
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(font_family),
            "font.size": body_pt,
            "axes.titlesize": title_pt,
            "axes.labelsize": body_pt,
            "xtick.labelsize": body_pt,
            "ytick.labelsize": body_pt,
            "legend.fontsize": body_pt,
            "legend.title_fontsize": body_pt,
            "figure.titlesize": title_pt,
            # math like $\mathrm{day}^{-1}$ inherits the sans-serif body font.
            "mathtext.default": "regular",
            "mathtext.fontset": "dejavusans",
            # TrueType embed (subsetted) so PDFs match on screen and in print.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


configure_matplotlib_typography()

# Optional Step-12 PNG pages: stop before this figure index (exclusive of duplicates past ~51).
PDF_MAX_APPEND_PNG_FIGURE_NUMBER = 51


def pdf_next_figure_number_would_exceed_cap(
    fig_counter: list[int] | None,
    *,
    cap: int | None = None,
) -> bool:
    """True if incrementing ``fig_counter`` would yield a figure number above ``cap``."""
    lim = int(cap if cap is not None else PDF_MAX_APPEND_PNG_FIGURE_NUMBER)
    if fig_counter is None:
        return False
    return int(fig_counter[0]) >= lim

# EPE PNG pages embedded from ``--s12-figures-dir`` or notebook exports: basename policy (lowercase keys).
_EPE_SUMMARY_PNG_EXCLUDE_BASENAMES_LOWER: frozenset[str] = frozenset(
    {
        "006_pe_conv1__r1.png",
        "001_regime_frequency_all_days_aligned_period.png",
        "007_daily_precip_domain_mean_by_regime_all_days.png",
        "008_daily_precip_domain_sum_by_regime_all_days.png",
        "009_daily_precip_over_flagged_cells_mean_e_conv0_by_regime_all_days.png",
        "011_daily_precip_over_flagged_cells_sum___flagged_cells_by_regime_all_days.png",
        "012_daily_precip_over_flagged_cells_max_e_conv0_by_regime_all_days.png",
        "032_a_relative_occurrence.png",
        "033_a_occurrence_rate__mjjas_composition__regime_frequency.png",
        "033_a_relative_occurrence.png",
        "034_a_relative_occurrence.png",
        # epe_03_01 QC only — not part of regime / extremes report narrative
        "epe_03_01_percentile_check_histograms.png",
        "epe_03_01_percentile_check_map.png",
    }
)
_EPE_SUMMARY_PNG_ONCE_BASENAMES_LOWER: frozenset[str] = frozenset(
    {
        "031_a_relative_occurrence.png",
        "epe_04_summary_occurrence_persistence_intensity_connectivity.png",
    }
)


def _path_matches_k_hint(path: Path, k: int) -> bool:
    """True if *path* (name or parents) suggests this asset was built for cluster count *k*."""
    blob = str(path.resolve()).lower()
    name = path.name.lower()
    for text in (name, blob):
        if re.search(rf"_k{k:02d}(?:_|\.|$)", text):
            return True
        if k < 10 and re.search(rf"_k{k}(?!\d)(?:_|\.|$)", text):
            return True
        if k < 10 and re.search(rf"(?<![0-9])k{k}(?![0-9])(?:_|\.|$)", text, re.I):
            return True
        if re.search(rf"k{k:02d}", text):
            return True
    return False


def _path_has_any_k_hint(path: Path) -> bool:
    blob = str(path.resolve()).lower()
    return re.search(r"(?:^|[_/\\\.-])k\d{1,2}(?:[_/\\\.-]|$)", blob) is not None


def _path_is_compatible_with_requested_k(path: Path, k: int | None) -> bool:
    """Allow generic paths and requested-k paths; reject paths explicitly tagged with another k."""
    if k is None:
        return True
    if _path_matches_k_hint(path, int(k)):
        return True
    return not _path_has_any_k_hint(path)


def _pick_best_png_for_k_group(candidates: list[Path], k: int | None) -> Path | None:
    compatible = [p for p in candidates if _path_is_compatible_with_requested_k(p, k)]
    if not compatible:
        return None
    if len(compatible) == 1:
        return compatible[0]
    if k is None:
        return sorted(compatible, key=lambda p: str(p.resolve()))[0]
    k_hits = [p for p in compatible if _path_matches_k_hint(p, int(k))]
    if k_hits:
        return max(k_hits, key=lambda p: p.stat().st_mtime)
    return max(compatible, key=lambda p: p.stat().st_mtime)


def filter_epe_summary_png_paths(
    paths: Iterable[Path],
    *,
    k: int | None = None,
    emitted_once_basenames: set[str] | None = None,
    skip_basenames: set[str] | None = None,
    include_excluded_basenames: set[str] | None = None,
) -> list[Path]:
    """Drop excluded PNGs; keep one page per \"once\" basename.

    When ``emitted_once_basenames`` is provided, once-only basenames are also suppressed across
    repeated calls, so merged reports or multiple Step-12 PNG sources still emit those pages once.
    """
    plist = [p for p in paths if _path_is_compatible_with_requested_k(p, k)]

    basename_chosen: dict[str, Path] = {}
    for lk in {p.name.lower() for p in plist}:
        cands = [p for p in plist if p.name.lower() == lk]
        if cands:
            chosen = _pick_best_png_for_k_group(cands, k)
            if chosen is not None:
                basename_chosen[lk] = chosen

    out: list[Path] = []
    emitted_once: set[str] = set()
    emitted_basenames: set[str] = set()
    for p in plist:
        key = p.name.lower()
        if skip_basenames is not None and key in skip_basenames:
            continue
        if (
            key in _EPE_SUMMARY_PNG_EXCLUDE_BASENAMES_LOWER
            and (include_excluded_basenames is None or key not in include_excluded_basenames)
        ):
            continue
        chosen = basename_chosen.get(key)
        if chosen is None:
            continue
        if p.resolve() != chosen.resolve():
            continue
        if key in emitted_basenames:
            continue
        if key in _EPE_SUMMARY_PNG_ONCE_BASENAMES_LOWER:
            if emitted_once_basenames is not None and key in emitted_once_basenames:
                continue
            if key in emitted_once:
                continue
            emitted_once.add(key)
            if emitted_once_basenames is not None:
                emitted_once_basenames.add(key)
        emitted_basenames.add(key)
        out.append(p)
    return out


def savefig_a4(
    pdf,
    fig: Figure,
    *,
    dpi: int = 120,
    tight: bool = True,
    pad_inches: float = 0.12,
) -> None:
    """Save figure to PdfPages with A4-friendly defaults."""
    if tight:
        pdf.savefig(fig, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=pad_inches)
    else:
        pdf.savefig(fig, dpi=dpi, facecolor="white", bbox_inches=None)


def notebook_markdown_to_plain(s: str) -> str:
    """Strip minimal Markdown so short notebook-style prose reads cleanly in PDF pages."""
    s = s.replace("**", "")
    lines_out: list[str] = []
    for line in s.splitlines():
        t = line.rstrip()
        if t.lstrip().startswith("###"):
            t = t.lstrip("#").strip()
        lines_out.append(t.rstrip())
    return "\n".join(lines_out).strip()


def _wrap_pdf_guide_text(body: str, *, width: int = 112) -> str:
    paras = [p.rstrip() for p in body.splitlines()]
    out: list[str] = []
    for p in paras:
        if not p.strip():
            out.append("")
            continue
        lead = len(p) - len(p.lstrip(" "))
        stripped = p.lstrip()
        if stripped.startswith("- "):
            wrapped = textwrap.fill(
                stripped[2:].strip(),
                width=max(32, width - 2 - lead),
                initial_indent=" " * lead + "- ",
                subsequent_indent=" " * lead + "  ",
            )
        else:
            wrapped = textwrap.fill(
                stripped,
                width=max(32, width - lead),
                initial_indent=" " * lead,
                subsequent_indent=" " * lead,
            )
        out.append(wrapped)
    return "\n".join(out).strip()


def save_figure_with_explanation_a4(
    pdf: PdfPages,
    plot_fig: Figure,
    explanation: str | None,
    *,
    dpi: int = 120,
    pad_inches: float = 0.12,
    tight: bool = True,
) -> None:
    """Save a figure on an A4 page with optional guide text above it."""
    body = notebook_markdown_to_plain(str(explanation or "").strip())
    if not body:
        savefig_a4(pdf, plot_fig, dpi=dpi, tight=tight, pad_inches=pad_inches)
        return

    plot_fig.canvas.draw()
    img = np.asarray(plot_fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]

    page = plt.figure(figsize=FIG_A4_PORTRAIT, facecolor="white")
    ax_text = page.add_axes([0.07, 0.79, 0.86, 0.15])
    ax_text.axis("off")
    ax_text.text(
        0,
        1.0,
        _wrap_pdf_guide_text(body),
        va="top",
        ha="left",
        fontsize=PDF_FS_GUIDE,
        linespacing=1.25,
    )
    ax_img = page.add_axes([0.05, 0.05, 0.90, 0.69])
    ax_img.imshow(img, aspect="equal", interpolation="bilinear")
    ax_img.axis("off")
    savefig_a4(pdf, page, dpi=dpi, tight=True, pad_inches=pad_inches)
    plt.close(page)


def load_png_as_rgb_uint8(path: Path) -> np.ndarray:
    """Load PNG for imshow; composite RGBA on white. Prefer Pillow (palette/RGBA-safe)."""
    try:
        from PIL import Image

        im = Image.open(path)
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return np.asarray(bg, dtype=np.uint8)
    except Exception:
        arr = np.asarray(plt.imread(str(path)))
        if arr.ndim == 2:
            x = np.clip(arr, 0.0, 1.0) if arr.dtype.kind == "f" else arr.astype(np.float64) / 255.0
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)
        rgb = arr[:, :, :3].astype(np.float64)
        if arr.shape[2] == 4:
            a = np.clip(arr[:, :, 3:4], 0.0, 1.0)
            rgb = rgb * a + (1.0 - a)
        hi = float(np.nanmax(rgb))
        if hi <= 1.0 + 1e-6:
            rgb = rgb * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)


def append_png_page_a4(pdf: PdfPages, png: Path, *, fig_num: int | None = None, dpi: int = 120) -> None:
    """One PDF page per raster image, A4 portrait canvas."""
    img = load_png_as_rgb_uint8(png)
    fig_img = plt.figure(figsize=FIG_A4_PORTRAIT, facecolor="white")
    ax = fig_img.add_axes([0.04, 0.05, 0.92, 0.90])
    ax.imshow(img, aspect="equal", interpolation="bilinear")
    ax.axis("off")
    title = f"Figure {fig_num} — {png.name}" if fig_num is not None else str(png.name)
    ax.set_title(title, fontsize=PDF_FS_PNG_CAPTION)
    savefig_a4(pdf, fig_img, dpi=dpi, tight=True, pad_inches=0.12)
    plt.close(fig_img)
