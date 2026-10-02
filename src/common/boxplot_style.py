"""Shared boxplot/bar styling for the Z500 (EPE.4) and residual-regimes (RR)
statistics PDFs, so both report families use identical box conventions, fonts,
and per-regime colors.

Two font tiers:
  - ``"full"``  — a boxplot alone on its own PDF page (EPE.4 singular
    boxplots; RR full-size pages). Matches ``PDF_FS_FIG_*`` (16/14/14 pt).
  - ``"panel"`` — a boxplot embedded as one of several panels in a multi-axes
    figure (the 2x2 occurrence/persistence/intensity/connectivity summaries in
    both pipelines). Smaller fonts keep four panels with ``n=`` sample-size
    tick labels legible on one page.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from common.page_layout import (
    PDF_FS_FIG_AXIS,
    PDF_FS_FIG_TICK,
    PDF_FS_FIG_TITLE,
)

# Panel tier: sized for a boxplot/bar chart that is one of several axes on the
# same page (e.g. a 2x2 occurrence/persistence/intensity/connectivity summary).
PDF_FS_PANEL_SUPTITLE = 11
PDF_FS_PANEL_TITLE = 10
PDF_FS_PANEL_AXIS = 9
PDF_FS_PANEL_TICK_X = 7
PDF_FS_PANEL_TICK_Y = 8
PDF_FS_PANEL_ANNOT = 7
# Convenience aliases for non-boxplot panel artists (legends, ``tick_params(axis="both")``)
# that don't need a separate x/y split.
PDF_FS_PANEL_LEGEND = PDF_FS_PANEL_AXIS
PDF_FS_PANEL_TICK = PDF_FS_PANEL_TICK_Y

_TIER_FONTS = {
    "full": {
        "title": PDF_FS_FIG_TITLE,
        "axis": PDF_FS_FIG_AXIS,
        "tick_x": PDF_FS_FIG_TICK,
        "tick_y": PDF_FS_FIG_TICK,
        "annot": PDF_FS_FIG_AXIS,
        "title_pad": 14,
        "label_pad": 10,
    },
    "panel": {
        "title": PDF_FS_PANEL_TITLE,
        "axis": PDF_FS_PANEL_AXIS,
        "tick_x": PDF_FS_PANEL_TICK_X,
        "tick_y": PDF_FS_PANEL_TICK_Y,
        "annot": PDF_FS_PANEL_ANNOT,
        "title_pad": 12,
        "label_pad": 8,
    },
}

_DEFAULT_BOX_COLOR = "#4C72B0"


def regime_tab10_colors(regime_ids: Sequence[int]) -> dict[int, tuple]:
    """Stable ``tab10`` colour per distinct integer regime id.

    Colours boxes/bars by regime (or, for RR Z500xRR groups, by the parent
    Z500 regime) so the same regime id maps to the same colour in every figure
    across both the Z500 and residual-regimes PDFs.
    """
    uniq = sorted({int(r) for r in regime_ids})
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1)))
    return {r: tuple(cmap[i]) for i, r in enumerate(uniq)}


# Region hue families for regime_region_colors(): Europe reads as purple, North America as
# green, everywhere a regime gets colored (Z500 boxplots, alluvial ribbons, RR boxplots,
# appendix dendrogram) -- a visual cue for which region a figure belongs to at a glance.
#
# A plain single-hue sequential colormap (e.g. plt.cm.Purples across 7 steps) reads as a
# near-gradient once you need more than ~3-4 steps: adjacent regimes become hard to tell apart
# because a sequential ramp only varies lightness, and a categorical variable like "regime id"
# has no ordering for lightness to usefully encode in the first place. These palettes instead
# vary TWO dimensions within each region's hue family -- e.g. Europe alternates a
# blue-violet lane (~hue 284-296 in OKLCH) with an orchid/magenta-purple lane (~312-336) as
# well as lightness, so neighboring regimes differ in both hue and shade, not lightness alone.
#
# Chosen and verified with the `dataviz` skill's color-formula method (adjacent-pair OKLab
# CVD deltaE, simulated protanopia/deuteranopia + unsimulated "normal-vision floor", both
# >= the skill's pass thresholds for every adjacent pair in this fixed order -- see
# scripts/validate_palette.js in that skill, or the ported Python check used to search this
# palette). Order is fixed (index 0 = canonical regime 1, etc.) -- do not re-sort or
# alphabetize; the adjacent-pair separation was optimized for exactly this order.
_REGION_PALETTES: dict[str, list[str]] = {
    "europe": [
        "#c371b0", "#5804d0", "#ce75ff", "#8c0779", "#8d44f9", "#e191d6", "#9851b2",
    ],
    "northamerica": [
        "#009800", "#14d621", "#186319", "#0fb89a",
    ],
}
# Fallback for a region/count this module has no hand-tuned palette for (kept as a single-hue
# sequential ramp -- not ideal for many categories, but a reasonable degrade).
_REGION_CMAPS = {"europe": "Purples", "northamerica": "Greens"}


def regime_region_colors(
    region: str,
    regime_ids: Sequence[int],
    *,
    n_total: int,
    lo: float = 0.35,
    hi: float = 0.92,
) -> dict[int, tuple]:
    """Per-region-hued shade for each *canonical* (cross-bbox-aligned) regime id.

    The shade for regime id ``r`` is fixed by its rank out of ``n_total`` (the region's total
    physical-regime count, e.g. from ``len(regime_alignment_<region>.yaml['groups'])``) -- NOT
    by the rank of whichever ids happen to be passed in this call. That's what makes the same
    physical regime get the same shade whether it's drawn alone, alongside every other regime,
    or at a bbox step where only some regimes are distinguishable (see the merge handling in
    ``regime_alignment.apply_regime_alignment``).

    ``regime_ids`` must already be canonical ids (1..n_total), not a bbox step's raw,
    mutually-incomparable cluster ids -- align them first. Ids outside ``1..n_total`` (e.g. the
    negative "unmapped" sentinels ``apply_regime_alignment`` produces) get a neutral grey so they
    stay visually distinct instead of colliding with a real regime's shade.

    Uses the hand-tuned, validated ``_REGION_PALETTES`` entry when one exists for
    ``(region, n_total)``; otherwise falls back to a single-hue sequential colormap (see
    ``_REGION_CMAPS``), and to :func:`regime_tab10_colors` for a region with no configured hue
    family at all.
    """
    unmapped = (0.6, 0.6, 0.6, 1.0)
    palette = _REGION_PALETTES.get(region)
    if palette is not None and len(palette) == n_total:
        shades = [matplotlib.colors.to_rgba(c) for c in palette]
    else:
        cmap_name = _REGION_CMAPS.get(region)
        if cmap_name is None or n_total <= 0:
            return regime_tab10_colors(regime_ids)
        shades = [tuple(c) for c in plt.get_cmap(cmap_name)(np.linspace(lo, hi, n_total))]
    out: dict[int, tuple] = {}
    for r in regime_ids:
        ri = int(r)
        out[ri] = shades[ri - 1] if 1 <= ri <= n_total else unmapped
    return out


def style_boxplot(
    ax,
    data: list[np.ndarray],
    tick_labels: list[str],
    *,
    title: str,
    ylabel: str,
    xlabel: str = "regime",
    colors: list | None = None,
    tier: str = "full",
):
    """Draw one boxplot with the shared notch/whisker/outlier/colour conventions.

    Every boxplot in both PDF pipelines uses ``showfliers=False`` (whiskers
    span the finite data range; outliers are hidden rather than drawn) and
    ``notch=True`` (95% CI notch around the median). Boxes are always
    ``patch_artist=True`` and shaded with ``colors`` (falls back to a single
    default colour) so single-regime and Z500xRR grouped boxplots look the
    same. Axis/title/grid styling is applied even when ``data`` is empty so
    callers don't need a separate "no data" branch.
    """
    fonts = _TIER_FONTS[tier]
    bp = None
    if data:
        bp = ax.boxplot(
            data, tick_labels=tick_labels, showfliers=False, notch=True, patch_artist=True,
            medianprops={"color": "black"},
        )
        box_colors = list(colors) if colors is not None else [_DEFAULT_BOX_COLOR] * len(bp["boxes"])
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
    ax.set_xlabel(xlabel, fontsize=fonts["axis"], labelpad=fonts["label_pad"])
    ax.set_ylabel(ylabel, fontsize=fonts["axis"], labelpad=fonts["label_pad"])
    ax.tick_params(axis="x", labelsize=fonts["tick_x"])
    ax.tick_params(axis="y", labelsize=fonts["tick_y"])
    ax.grid(axis="y", alpha=0.25)
    return bp


def style_bar(
    ax,
    values: Sequence[float],
    tick_labels: list[str],
    *,
    colors: Sequence,
    title: str,
    ylabel: str,
    tier: str = "panel",
    annotate_fmt: str | None = "%.1f%%",
):
    """Draw one bar chart with the shared colour/font conventions (occurrence panels)."""
    fonts = _TIER_FONTS[tier]
    bars = ax.bar(range(len(values)), values, color=list(colors))
    if annotate_fmt:
        ax.bar_label(bars, fmt=annotate_fmt, fontsize=fonts["annot"], padding=2)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(tick_labels, fontsize=fonts["tick_x"])
    ax.set_ylabel(ylabel, fontsize=fonts["axis"])
    ax.set_title(title, fontsize=fonts["title"])
    ax.tick_params(axis="y", labelsize=fonts["tick_y"])
    ax.grid(axis="y", alpha=0.25)
    return bars
