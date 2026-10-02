"""Shared helpers for discretized (binned) colorbars with arrow-shaped extend ends.

Used across the z500 and rr (residual regime) composite/variance/bbox-comparison
figures so every fill colorbar is built the same way: discrete bins instead of a
continuous gradient, with triangular "extend" ends where values can fall outside
the plotted range.

``discretize_diverging``/``discretize_sequential_from_floor`` round their bin edges
outward to "nice" 1/2/5x10^k steps (see ``_nice_step``) rather than slicing the raw
percentile-derived vmin/vmax into ``n_levels`` even pieces -- otherwise bin edges
(and the colorbar tick labels, which default to the bin boundaries) land on
arbitrary-precision floats like 12.847293. ``n_levels`` is therefore a *target*
bin count, not exact: rounding to a nice step can leave a range with fewer bins
than requested (e.g. a tight range rounding to just 6-8 bins). Whenever that
would drop to ``MIN_DISCRETE_LEVELS`` or fewer, the step is halved (doubling the
bin count) -- repeatedly if needed -- until there are more than
``MIN_DISCRETE_LEVELS`` bins, so every colorbar always ends up with more than
that many discrete classes. ``discretize_bounded``'s bin count is exact (no
"nice"-step rounding, since its bounds are already hard/round, e.g. [0, 1] or
[0, 100]) but is bumped the same way if ``n_levels`` alone wouldn't clear
``MIN_DISCRETE_LEVELS``.

Every caller that builds one of these colorbars should also call
``style_discrete_ticks(cb)`` right after ``fig.colorbar(...)``/``plt.colorbar(...)``
-- it picks the fewest decimal places (0, 1 or 2; 3 only if that's the only way to
keep every tick nonzero) and applies it to that colorbar's tick labels. 1 decimal
is preferred over 2 even when that means two adjacent ticks round to the same
label (e.g. a 0.05-wide bin edge at 0.45 is labelled "0.4", not "0.45"); decimals
are only added when fewer would round some nonzero boundary to "0" (e.g. 0.04
needs 2 decimals, since 1 decimal would show "0.0").
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Colormap
from matplotlib.ticker import FuncFormatter

N_DISCRETE_LEVELS = 10
# Every colorbar built here must end up with MORE than this many discrete bins.
MIN_DISCRETE_LEVELS = 10
# Tick labels use this many decimal places if possible, else DEFAULT_TICK_DECIMALS+1
# ("2 normally, 3 only if unavoidable").
DEFAULT_TICK_DECIMALS = 2


def _nice_step(raw_step: float) -> float:
    """Round a raw bin width up to a "nice" 1/2/5x10^k step (clean tick labels)."""
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 1.0
    exponent = np.floor(np.log10(raw_step))
    fraction = raw_step / (10.0**exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return float(nice_fraction * (10.0**exponent))


def discretize_diverging(
    vmin: float,
    vmax: float,
    cmap_name: str | Colormap,
    *,
    n_levels: int = N_DISCRETE_LEVELS,
    min_levels: int = MIN_DISCRETE_LEVELS,
) -> tuple[Colormap, BoundaryNorm, str]:
    """Binned cmap/norm for a diverging fill (e.g. anomalies straddling zero).

    Values can fall outside [vmin, vmax] on either side, so both colorbar ends
    are drawn as arrows. Bin edges are rounded outward to a "nice" step and kept
    symmetric about zero (so the diverging cmap's center stays at 0); the step is
    halved (doubling the bin count) until there are more than ``min_levels`` bins.
    """
    vabs = max(abs(float(vmin)), abs(float(vmax)), 1e-9)
    step = _nice_step(2.0 * vabs / max(int(n_levels), 1))
    vabs_nice = step * np.ceil(vabs / step)
    n_bins = max(int(round(2.0 * vabs_nice / step)), 1)
    while n_bins <= min_levels:
        step /= 2.0
        vabs_nice = step * np.ceil(vabs / step)
        n_bins = max(int(round(2.0 * vabs_nice / step)), 1)
    # Round away np.linspace's float64 noise (e.g. a boundary meant to be exactly
    # -0.05 landing as -0.049999999999999994) -- left alone, that noise makes
    # _tick_label_decimals think a nonzero boundary rounds to "0" and forces an
    # extra decimal place it doesn't actually need.
    boundaries = np.round(np.linspace(-vabs_nice, vabs_nice, n_bins + 1), 10)
    base = plt.get_cmap(cmap_name)
    cmap = plt.get_cmap(cmap_name, n_bins)
    cmap.set_over(base(1.0))
    cmap.set_under(base(0.0))
    norm = BoundaryNorm(boundaries, ncolors=n_bins)
    return cmap, norm, "both"


def discretize_sequential_from_floor(
    vmax: float,
    cmap_name: str | Colormap,
    *,
    vmin: float = 0.0,
    n_levels: int = N_DISCRETE_LEVELS,
    min_levels: int = MIN_DISCRETE_LEVELS,
) -> tuple[Colormap, BoundaryNorm, str]:
    """Binned cmap/norm for a sequential fill with a hard floor (e.g. variance, std,
    EPE % frequency) whose upper limit is percentile-clipped, so values can exceed
    vmax but never fall below vmin. Only the top colorbar end is drawn as an arrow.
    Bin edges are rounded outward (from ``vmin``) to a "nice" step; the step is
    halved (doubling the bin count) until there are more than ``min_levels`` bins.
    """
    vmin = float(vmin)
    span = max(float(vmax) - vmin, 1e-9)
    step = _nice_step(span / max(int(n_levels), 1))
    vmax_nice = vmin + step * np.ceil(span / step)
    n_bins = max(int(round((vmax_nice - vmin) / step)), 1)
    while n_bins <= min_levels:
        step /= 2.0
        vmax_nice = vmin + step * np.ceil(span / step)
        n_bins = max(int(round((vmax_nice - vmin) / step)), 1)
    # See the matching np.round(..., 10) note in discretize_diverging above.
    boundaries = np.round(np.linspace(vmin, vmax_nice, n_bins + 1), 10)
    base = plt.get_cmap(cmap_name)
    cmap = plt.get_cmap(cmap_name, n_bins)
    cmap.set_over(base(1.0))
    norm = BoundaryNorm(boundaries, ncolors=n_bins)
    return cmap, norm, "max"


def discretize_bounded(
    vmin: float,
    vmax: float,
    cmap_name: str | Colormap,
    *,
    n_levels: int = N_DISCRETE_LEVELS,
    min_levels: int = MIN_DISCRETE_LEVELS,
) -> tuple[Colormap, BoundaryNorm, str]:
    """Binned cmap/norm for a fill that is exactly bounded (e.g. probabilities in
    [0, 1]) so values cannot fall outside [vmin, vmax]. No extend arrows. ``n_levels``
    is doubled until there are more than ``min_levels`` bins.
    """
    n_levels = max(int(n_levels), 1)
    while n_levels <= min_levels:
        n_levels *= 2
    boundaries = np.linspace(vmin, vmax, n_levels + 1)
    cmap = plt.get_cmap(cmap_name, n_levels)
    norm = BoundaryNorm(boundaries, ncolors=n_levels)
    return cmap, norm, "neither"


def _round_half_up(v: float, decimals: int) -> float:
    """Round ``v`` to ``decimals`` places, half-up (0.075 -> 0.08 at 2 decimals) --
    unlike ``round()``/``np.round`` (round-half-to-even, plus float64 already stores
    0.075 as slightly under 0.075, so those would give 0.07). Goes through
    ``str(v)`` so the exact decimal digits typed/computed are what gets rounded,
    not float64's binary approximation of them.
    """
    if not np.isfinite(v):
        return v
    quantum = Decimal(1).scaleb(-decimals)
    return float(Decimal(str(v)).quantize(quantum, rounding=ROUND_HALF_UP))


def _tick_label_decimals(
    boundaries: np.ndarray, *, max_decimals: int = DEFAULT_TICK_DECIMALS, hard_max_decimals: int = 3
) -> int:
    """Fewest decimal places (0..``max_decimals``) that keep every nonzero boundary's
    label nonzero -- adjacent ticks rounding to the same label is fine (1 decimal is
    preferred even at the cost of that). Falls back to ``hard_max_decimals`` only if
    nothing up to ``max_decimals`` avoids a nonzero boundary rounding to "0".
    """
    vals = np.asarray(boundaries, dtype=float)
    for d in range(0, max_decimals + 1):
        rounded = np.array([_round_half_up(v, d) for v in vals])
        collapses_to_zero = np.any((vals != 0) & (rounded == 0))
        if not collapses_to_zero:
            return d
    return hard_max_decimals


def style_discrete_ticks(cb) -> None:
    """Format a discretized colorbar's tick labels with the fewest decimal places
    needed (see ``_tick_label_decimals``) -- e.g. a boundary at 0.075 is labelled
    "0.08", not "0.075" or "0.0750". Call this right after building any colorbar
    from a ``discretize_diverging``/``discretize_sequential_from_floor``/
    ``discretize_bounded`` norm.
    """
    boundaries = getattr(cb.norm, "boundaries", None)
    if boundaries is None:
        return
    decimals = _tick_label_decimals(np.asarray(boundaries))

    def _fmt(v: float, _pos, _d: int = decimals) -> str:
        return f"{_round_half_up(v, _d):.{_d}f}"

    formatter = FuncFormatter(_fmt)
    cb.ax.yaxis.set_major_formatter(formatter)
    cb.ax.xaxis.set_major_formatter(formatter)
