"""Feature category + pressure-level colour coding, shared across residual-regime
feature-importance plots.

Colour encodes the process family (dynamic / thermodynamic / convective /
boundary forcing / the Z500 target itself); lightness encodes the pressure
level, lightest at the surface and darkest at 250 hPa, so a heatmap built from
these colours lets you read off both "which process family" and "how high up"
at a glance. Category assignments and level tiers live in
feature_categories.yaml (single source of truth — also mirrors the
_PROCESS_CATEGORY mapping used for the input-features LaTeX table in
01_paper_figures/1_Intro_Methods.ipynb).
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "feature_categories.yaml"


def load_feature_categories(path: Path | None = None) -> dict:
    with open(path or _CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _hex_to_hls(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h, min(1.0, max(0.0, l)), s)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def shade_for_level(
    base_hex: str,
    level: str,
    level_order: list[str],
    l_light: float = 0.82,
    l_dark: float = 0.30,
) -> str:
    """Lighten/darken `base_hex` by where `level` sits in `level_order`.

    Hue and saturation come from `base_hex`; only lightness moves, linearly
    from `l_light` (level_order[0], e.g. "surface") to `l_dark`
    (level_order[-1], e.g. "250"). Fixed tier *positions* (not just fixed
    endpoints) mean the same level lands at the same lightness in every
    category, so lightness is comparable across colour families too.
    """
    h, _, s = _hex_to_hls(base_hex)
    idx = level_order.index(level)
    frac = idx / max(1, len(level_order) - 1)
    lightness = l_light + frac * (l_dark - l_light)
    return _hls_to_hex(h, lightness, s)


def text_color_for(hex_color: str) -> str:
    """Black text on light fills, white text on dark fills."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#000000" if luminance > 0.55 else "#ffffff"


def field_label(field: str, cfg: dict | None = None) -> str:
    """Short display label for a field key (e.g. "uwnd250" -> "U250", "qhum700" -> "QH700").

    Single source of truth for the abbreviations used on composite-panel titles and legends
    (plot_composites.py, plot_subregimes.py, plot_rgb_scale.py), so they always match the
    feature-importance heatmap (field_style, below) and the paper's input-features table.
    Falls back to `field.upper()` for any field not listed in feature_categories.yaml (e.g. a
    field only ever used for file discovery, never added to the curated category/label list).
    """
    cfg = cfg or load_feature_categories()
    entry = cfg["fields"].get(field.strip().lower())
    return entry["label"] if entry else field.upper()


def field_style(field_combination: str, cfg: dict | None = None) -> dict:
    """Return {"category", "level", "label", "color", "text_color"} for one field.

    `field_combination` is matched case-insensitively against the keys in
    feature_categories.yaml (e.g. "Z500 baseline", "uwnd250", "cape").
    """
    cfg = cfg or load_feature_categories()
    key = field_combination.strip().lower()
    try:
        entry = cfg["fields"][key]
    except KeyError:
        raise KeyError(
            f"Unknown field '{field_combination}' -- add it to feature_categories.yaml"
        ) from None

    category = entry["category"]
    level = entry["level"]
    base_hex = cfg["category_colors"][category]
    if category == "target":
        color = base_hex
    else:
        bounds = cfg.get("level_shading", {})
        band = bounds.get(category, bounds.get("default", {}))
        color = shade_for_level(
            base_hex,
            level,
            cfg["level_order"],
            l_light=band.get("light", 0.82),
            l_dark=band.get("dark", 0.30),
        )

    return {
        "category": category,
        "level": level,
        "label": entry["label"],
        "color": color,
        "text_color": text_color_for(color),
    }
