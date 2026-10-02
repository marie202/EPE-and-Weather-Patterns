"""Additive-RGB colour scale helpers for residual regime composite plots.

draw_rgb_scale(ax, channel_of, fields)
    Adds a colour key to an existing Axes object:
    - 3 fields  → equilateral RGB mixing triangle (full barycentric gradient)
    - 2 fields  → linear gradient bar
    - 1 or >3   → labelled colour swatches

The triangle shows:
  • Vertices   — pure single-field colours (R, G, B)
  • Edges      — 50/50 binary mixtures  (R+G=yellow, R+B=magenta, G+B=cyan)
  • Interior   — proportional additive blends
  • White      — no signal (all channels zero)
"""
from __future__ import annotations

import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch

from .feature_categories import field_label


# ---------------------------------------------------------------------------
# Field display names
# ---------------------------------------------------------------------------

FIELD_DISPLAY_NAMES: dict[str, str] = {
    # Geopotential / vorticity
    "z500":    "Geopotential\n500 hPa",
    "vo850":   "Rel. Vorticity\n850 hPa",
    "vo500":   "Rel. Vorticity\n500 hPa",
    # Moisture
    "tcwv":    "Total Column\nWater Vapour",
    "vimd":    "Moisture\nDivergence",
    "vidmf":   "Moisture Flux\nDivergence",
    "viwve":   "Water Vapour\nFlux E",
    "viwvn":   "Water Vapour\nFlux N",
    # Thermodynamic levels
    "deg0l":   "Freezing Level\nHeight",
    "cape":    "CAPE",
    "t2m":     "Temperature\n2 m",
    "td2m":    "Dewpoint\n2 m",
    "temp900": "Temperature\n900 hPa",
    "temp850": "Temperature\n850 hPa",
    "temp700": "Temperature\n700 hPa",
    "temp500": "Temperature\n500 hPa",
    "sst":     "Sea Surface\nTemperature",
    "tisr":    "Solar\nRadiation",
    # Vertical velocity
    "w850":    "Vert. Velocity\n850 hPa",
    "w700":    "Vert. Velocity\n700 hPa",
    "w500":    "Vert. Velocity\n500 hPa",
    # Relative humidity
    "rh850":   "Rel. Humidity\n850 hPa",
    "rh700":   "Rel. Humidity\n700 hPa",
    "rh600":   "Rel. Humidity\n600 hPa",
    "rh500":   "Rel. Humidity\n500 hPa",
    # Specific humidity
    "qhum900": "Spec. Humidity\n900 hPa",
    "qhum850": "Spec. Humidity\n850 hPa",
    "qhum700": "Spec. Humidity\n700 hPa",
    "qhum500": "Spec. Humidity\n500 hPa",
    # U-wind
    "uwnd925": "U-Wind\n925 hPa",
    "uwnd850": "U-Wind\n850 hPa",
    "uwnd700": "U-Wind\n700 hPa",
    "uwnd500": "U-Wind\n500 hPa",
    "uwnd250": "U-Wind\n250 hPa",
    # V-wind
    "vwnd925": "V-Wind\n925 hPa",
    "vwnd850": "V-Wind\n850 hPa",
    "vwnd700": "V-Wind\n700 hPa",
    "vwnd500": "V-Wind\n500 hPa",
    "vwnd250": "V-Wind\n250 hPa",
    # Legacy / alternative keys
    "u850":    "U-Wind\n850 hPa",
    "v850":    "V-Wind\n850 hPa",
    "t850":    "Temperature\n850 hPa",
    "tp":      "Precipitation",
    "precip":  "Precipitation",
}


def _field_display_name(key: str) -> str:
    """Human-readable label for an atmospheric field key.

    Falls back to key.upper() for any unknown field, so the legend always
    adapts to whatever fields are active in the current run.
    """
    return FIELD_DISPLAY_NAMES.get(key.lower(), key.upper())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUPERSCRIPTS = {"-1": "⁻¹", "-2": "⁻²", "-3": "⁻³", "2": "²", "3": "³"}


def _pretty_units(raw: str) -> str:
    """Turn cfgrib/netCDF exponent notation ('kg m**-2') into 'kg m⁻²'."""
    if not raw:
        return ""
    return re.sub(r"\*\*(-?\d+)", lambda m: _SUPERSCRIPTS.get(m.group(1), f"^{m.group(1)}"), raw.strip())


def _fmt_physical(v: float) -> str:
    """Adaptive-precision formatting for physical anomaly values spanning
    wildly different magnitudes across fields (e.g. CAPE in J/kg vs. specific
    humidity in kg/kg)."""
    av = abs(v)
    if av == 0:
        return "0"
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 0.1:
        return f"{v:.2f}"
    return f"{v:.2g}"


def _additive_over_white(rgb: np.ndarray) -> np.ndarray:
    """Map additive RGB (may exceed 1.0) onto a white background."""
    m = rgb.max(axis=-1, keepdims=True)
    return np.clip(rgb + (1.0 - m), 0.0, 1.0)


def _triangle_image(
    c0: tuple, c1: tuple, c2: tuple, res: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """Generate RGBA image of an equilateral RGB mixing triangle.

    Vertices (in image row/col space, y-up):
      v0 = top    → colour c0
      v1 = bottom-left  → colour c1
      v2 = bottom-right → colour c2

    Returns
    -------
    img : (res, res, 4) RGBA float32 image (transparent outside triangle)
    v   : (3, 2) vertex (x, y) in [0,1]^2 (y-up)
    """
    v = np.array([
        [0.50, 0.92],   # top
        [0.04, 0.08],   # bottom-left
        [0.96, 0.08],   # bottom-right
    ], dtype=np.float64)
    C = np.array([c0, c1, c2], dtype=np.float64)

    # Pixel grid in [0,1]^2 (y-up)
    xs = np.linspace(0.0, 1.0, res)
    ys = np.linspace(0.0, 1.0, res)
    px, py = np.meshgrid(xs, ys)   # (res, res)

    # Barycentric coordinates
    T = np.array([[v[0, 0] - v[2, 0], v[1, 0] - v[2, 0]],
                  [v[0, 1] - v[2, 1], v[1, 1] - v[2, 1]]])
    det = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    dx = px - v[2, 0]
    dy = py - v[2, 1]
    lam0 = (T[1, 1] * dx - T[0, 1] * dy) / det
    lam1 = (-T[1, 0] * dx + T[0, 0] * dy) / det
    lam2 = 1.0 - lam0 - lam1

    inside = (lam0 >= -1e-6) & (lam1 >= -1e-6) & (lam2 >= -1e-6)
    lam = np.stack([lam0, lam1, lam2], axis=-1).clip(0, 1)  # (res, res, 3)

    rgb = np.einsum("ijc,cd->ijd", lam, C)                  # (res, res, 3)
    rgb = _additive_over_white(rgb)

    img = np.ones((res, res, 4), dtype=np.float32)
    img[..., :3] = rgb
    img[~inside, 3] = 0.0   # transparent outside

    return img[::-1], v     # flip y for imshow (top=high y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_rgb_scale(
    ax: plt.Axes,
    channel_of: dict[str, str],   # field_key -> "R" | "G" | "B"
    channel_rgb: dict[str, tuple], # "R" -> (r,g,b), …
    fields: list[str],
    fontsize: float = 8.0,
    tick_fontsize: float | None = None,
    title: str = "Colour key\n(additive RGB)",
    physical_scales: dict[str, float] | None = None,
    units: dict[str, str] | None = None,
) -> None:
    """Draw an additive-RGB colour key inside *ax*.

    Parameters
    ----------
    ax          : target Axes (will be cleared and used as the legend panel)
    channel_of  : mapping field_key → channel name ("R"/"G"/"B")
    channel_rgb : mapping channel name → (r, g, b) tuple in [0,1]
    fields      : ordered list of impact fields to show
    fontsize    : vertex-label (and title, swatch/gradient-bar) fontsize
    tick_fontsize : fontsize for the ternary triangle's edge tick numbers
                    (3-field layout only); defaults to ``fontsize`` if not given
    physical_scales : optional mapping field_key → physical value (native units)
                       that corresponds to the "1.0"/full-signal end of that
                       field's axis (typically the same 95th-percentile scale
                       used to normalise the RGB composite). When given (only
                       used by the 3-field ternary layout), edge tick numbers
                       are shown in physical units instead of the 0..1 fraction.
    units       : optional mapping field_key → unit string, shown next to the
                   vertex labels when physical_scales is also given.
    title       : caption drawn below the key; pass "" to omit it.
    """
    ax.set_axis_off()

    n = len(fields)

    if n == 3:
        _draw_triangle(ax, channel_of, channel_rgb, fields, fontsize, title,
                       tick_fontsize=tick_fontsize, physical_scales=physical_scales, units=units)
    elif n == 2:
        _draw_gradient_bar(ax, channel_of, channel_rgb, fields, fontsize, title)
    else:
        _draw_swatches(ax, channel_of, channel_rgb, fields, fontsize, title)


# ---------------------------------------------------------------------------
# Three-field: equilateral RGB triangle
# ---------------------------------------------------------------------------

def _draw_triangle(ax, channel_of, channel_rgb, fields, fontsize, title,
                    tick_fontsize=None, physical_scales=None, units=None):
    c = [channel_rgb[channel_of[f]] for f in fields]
    img, v = _triangle_image(c[0], c[1], c[2])

    ax.imshow(img, origin="upper", extent=[0, 1, 0, 1],
              aspect="equal", interpolation="bilinear")

    # ax.set_ylim(0,1) below makes data coords y-up (y=0 bottom, y=1 top),
    # matching the y-up vertex positions returned by _triangle_image.
    vd = np.array(v)   # (3, 2): v[0]=top, v[1]=bottom-left, v[2]=bottom-right
    centroid_d = vd.mean(axis=0)

    # --- Vertex labels: short code (+ unit) only, no full atmospheric field name ---
    # offsets push labels away from each vertex in the y-up system
    v_offsets = [(0.0, 0.07), (-0.05, -0.05), (0.05, -0.05)]
    v_ha = ["center", "right", "left"]
    v_va = ["bottom", "top", "top"]
    for i, field in enumerate(fields):
        dx, dy = v_offsets[i]
        label = field_label(field)
        unit = _pretty_units(units.get(field, "")) if units else ""
        if physical_scales is not None and unit:
            label += f"\n[{unit}]"
        ax.text(vd[i, 0] + dx, vd[i, 1] + dy, label,
                ha=v_ha[i], va=v_va[i], fontsize=fontsize, fontweight="bold",
                color=_darken(c[i], 0.6), transform=ax.transData, clip_on=False)

    # --- Tick marks and numbers along each edge ---
    _draw_ternary_ticks(ax, vd, centroid_d, tick_fontsize if tick_fontsize is not None else fontsize,
                        fields=fields, physical_scales=physical_scales)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title placed below the triangle to avoid overlapping the top vertex label
    if title:
        ax.text(0.5, -0.08, title, ha="center", va="top",
                fontsize=fontsize, transform=ax.transAxes)


def _draw_ternary_ticks(ax, vd, centroid_d, tick_fontsize, fields=None, physical_scales=None):
    """Tick marks and numeric labels on the 3 edges of the ternary triangle.

    vd          : (3, 2) vertex data coordinates (y-up: y=0 at bottom, y=1 at top)
    centroid_d  : (2,) centroid data coordinates
    fields      : ordered field keys matching vd's vertex order (needed when
                  physical_scales is given, to know which field's scale
                  applies to the "destination" end of each edge)
    physical_scales : optional field_key → physical value at the 1.0 end of
                       that field's axis; when given, tick labels show
                       ``t * physical_scales[to_field]`` instead of the raw
                       fraction ``t``.
    """
    tick_len = 0.06
    label_dist = 0.11
    tick_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_fontsize = tick_fontsize

    # Each edge: (from_vertex_idx, to_vertex_idx)
    # Numbers run 0 → 1 from the "from" vertex to the "to" vertex.
    edges = [(0, 1), (0, 2), (1, 2)]

    for i, j in edges:
        pi = vd[i]
        pj = vd[j]
        edge_vec = pj - pi
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-10:
            continue
        edge_dir = edge_vec / edge_len

        # 90° CW in y-up (standard Cartesian): (a, b) → (b, -a)
        perp = np.array([edge_dir[1], -edge_dir[0]])
        mid = (pi + pj) * 0.5
        # Ensure perp points OUTWARD (away from centroid)
        if np.dot(perp, mid - centroid_d) < 0:
            perp = -perp

        if physical_scales is not None and fields is not None:
            to_scale = physical_scales.get(fields[j], 1.0)
        else:
            to_scale = None

        for t in tick_vals:
            p = pi + t * edge_vec
            ax.plot(
                [p[0], p[0] + tick_len * perp[0]],
                [p[1], p[1] + tick_len * perp[1]],
                color="0.35", lw=0.8, transform=ax.transData, clip_on=False,
            )
            lx = p[0] + label_dist * perp[0]
            ly = p[1] + label_dist * perp[1]
            text = _fmt_physical(t * to_scale) if to_scale is not None else f"{t:.2g}"
            ax.text(
                lx, ly, text,
                ha="center", va="center",
                fontsize=num_fontsize, color="0.35",
                transform=ax.transData, clip_on=False,
            )


def _mix(c0, c1):
    return _additive_over_white(np.array([(np.array(c0) + np.array(c1)) / 2]))[0]


def _darken(c, frac=0.7):
    return tuple(x * frac for x in c)


# ---------------------------------------------------------------------------
# Two-field: horizontal gradient bar
# ---------------------------------------------------------------------------

def _draw_gradient_bar(ax, channel_of, channel_rgb, fields, fontsize, title):
    c0 = np.array(channel_rgb[channel_of[fields[0]]])
    c1 = np.array(channel_rgb[channel_of[fields[1]]])

    n = 256
    gradient = np.linspace(0, 1, n)
    rgb = np.outer(gradient, c1) + np.outer(1 - gradient, c0)
    rgb = _additive_over_white(rgb.reshape(1, n, 3)).reshape(1, n, 3)

    ax.imshow(rgb, aspect="auto", origin="upper", extent=[0, 1, 0, 1])
    ax.text(-0.02, 0.5, field_label(fields[0]), ha="right", va="center",
            fontsize=fontsize, fontweight="bold",
            color=_darken(tuple(c0)), transform=ax.transAxes)
    ax.text(1.02, 0.5, field_label(fields[1]), ha="left", va="center",
            fontsize=fontsize, fontweight="bold",
            color=_darken(tuple(c1)), transform=ax.transAxes)
    ax.text(0.5, -0.15, "white = no signal", ha="center", va="top",
            fontsize=fontsize - 1, color="0.5", style="italic",
            transform=ax.transAxes)
    ax.set_title(title, fontsize=fontsize, pad=4)
    ax.set_axis_off()


# ---------------------------------------------------------------------------
# Other field counts: colour swatches
# ---------------------------------------------------------------------------

def _draw_swatches(ax, channel_of, channel_rgb, fields, fontsize, title):
    ax.set_title(title, fontsize=fontsize, pad=4)
    n = len(fields)
    for i, field in enumerate(fields):
        ch = channel_of.get(field)
        col = channel_rgb.get(ch, (0.5, 0.5, 0.5)) if ch else (0.5, 0.5, 0.5)
        y = 1.0 - (i + 0.5) / n
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.02, y - 0.35 / n), 0.18, 0.65 / n,
            boxstyle="round,pad=0.01", fc=col, ec="0.3", lw=0.5,
            transform=ax.transAxes,
        ))
        ax.text(0.26, y, field_label(field), va="center", ha="left",
                fontsize=fontsize, color=_darken(col, 0.6),
                transform=ax.transAxes)
    ax.text(0.5, -0.05, "additive; white = no signal",
            ha="center", va="top", fontsize=fontsize - 1,
            color="0.5", style="italic", transform=ax.transAxes)
