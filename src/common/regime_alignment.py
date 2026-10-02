"""Cross-bbox-step physical-regime alignment, shared by every consumer of
``regime_alignment_<region>.yaml`` (``src/common/regime_alignment_europe.yaml`` /
``regime_alignment_northamerica.yaml``).

A Ward cut at one bbox step is an INDEPENDENT clustering from another step's -- "regime 3" in
B1 has no guaranteed relationship to "regime 3" in B2. The alignment yaml hand-assigns each
physical regime a stable ``canonical_id`` (1..N, N = the region's total physical-regime count)
and lists, per bbox step, which of that step's raw cluster ids it corresponds to. This module
turns that yaml into raw-id -> canonical-id lookups so callers can color/label by canonical id
and get the same color/name for the same physical regime everywhere -- the Z500 boxplots and
alluvial diagrams (``01_paper_figures/2_3_Results_Z500_Stats.ipynb``), the RR sub-regime
boxplots (``residual_regimes/plot_rr_statistics.py``), and the appendix.

Not every bbox step has as many distinguishable clusters as there are physical regimes (e.g.
Europe has 7 physical regimes but only k=4 at B1) -- when that happens, two or more canonical
ids legitimately share one raw cluster id (see ``build_bbox_step_relabel_maps``'s ``merges``
return value). That's expected, not an authoring error.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_regime_alignment(path: Path) -> tuple[str, list[dict]]:
    """Parse a ``regime_alignment_<region>.yaml``.

    Returns ``(region, groups)``, each group a dict with ``canonical_id`` (the group's
    1-indexed position -- stable across renames, unlike matching on ``name``), ``name``,
    ``short``, and ``members`` (``{bbox_step: raw_regime_id}``).
    """
    data = yaml.safe_load(path.read_text())
    region = str(data.get("region", "")).strip()
    groups_raw = data.get("groups") or []
    if not isinstance(groups_raw, list) or not groups_raw:
        raise ValueError(f"{path}: 'groups' must be a non-empty list")
    groups: list[dict] = []
    for idx, g in enumerate(groups_raw, start=1):
        members_raw = g.get("members") or {}
        members = {str(step).strip().upper(): int(regime) for step, regime in members_raw.items()}
        groups.append({
            "canonical_id": idx,
            "name": str(g.get("name") or f"Regime {idx}"),
            "short": str(g.get("short") or f"Z{idx}"),
            "members": members,
        })
    return region, groups


def short_name_for_canonical(groups: list[dict] | None, canonical_id: int | None) -> str | None:
    """The group's ``short`` label (e.g. ``"Z3"``) for a given canonical id, or ``None`` if
    ``groups``/``canonical_id`` is missing or the id isn't found (kept ``None`` rather than a
    guessed ``f"Z{canonical_id}"`` so callers can fall back to their own default).
    """
    if not groups or canonical_id is None:
        return None
    for g in groups:
        if g["canonical_id"] == int(canonical_id):
            return str(g["short"])
    return None


def build_bbox_step_relabel_maps(
    groups: list[dict], stage_cols: list[str],
) -> tuple[dict[str, dict[int, int]], dict[str, dict[int, list[int]]]]:
    """Per bbox step: ``{raw_regime_id: canonical_id}``, plus a same-shaped ``merges`` map of
    ``{canonical_id: [canonical_id, ...]}`` wherever more than one canonical (physical) regime
    claims the same raw cluster id at that step (see module docstring). The representative id
    used in ``maps`` is the lowest canonical id among the merged group.
    """
    claims: dict[str, dict[int, list[int]]] = {step: {} for step in stage_cols}
    for g in groups:
        for step, raw_id in g["members"].items():
            if step not in claims:
                continue
            claims[step].setdefault(int(raw_id), []).append(g["canonical_id"])

    maps: dict[str, dict[int, int]] = {step: {} for step in stage_cols}
    merges: dict[str, dict[int, list[int]]] = {step: {} for step in stage_cols}
    for step, raw_map in claims.items():
        for raw_id, canon_ids in raw_map.items():
            canon_ids = sorted(canon_ids)
            rep = canon_ids[0]
            maps[step][raw_id] = rep
            if len(canon_ids) > 1:
                merges[step][rep] = canon_ids
    return maps, merges


def relabel_ids(raw_ids, groups: list[dict], bbox_step: str) -> dict[int, int]:
    """Single-bbox-step convenience wrapper: ``{raw_id: canonical_id}`` for whichever raw ids
    are actually present, falling back to a negative sentinel (``-raw_id``) for any raw id the
    yaml's ``members`` doesn't cover at this step (kept distinct rather than silently colliding
    with a real canonical id -- add it to the yaml once you've identified what it corresponds to).
    """
    step = bbox_step.strip().upper()
    maps, _merges = build_bbox_step_relabel_maps(groups, [step])
    raw_to_canon = maps[step]
    return {int(r): raw_to_canon.get(int(r), -int(r)) for r in raw_ids}


def canonical_order(raw_ids, groups: list[dict], bbox_step: str) -> list[int]:
    """``raw_ids`` sorted by their canonical (physical) id, so "regime 1" sorts first, "regime
    2" second, etc. in every chart -- regardless of which raw cluster number this bbox step
    happened to assign it. Ids the yaml doesn't cover at this step sort by their (negative)
    sentinel, i.e. after every real regime, in raw-id order among themselves.
    """
    raw_to_canon = relabel_ids(raw_ids, groups, bbox_step)
    return sorted({int(r) for r in raw_ids}, key=lambda r: raw_to_canon[r])


def apply_regime_alignment(
    day_df: pd.DataFrame, stage_cols: list[str], groups: list[dict],
) -> tuple[pd.DataFrame, dict[str, dict[int, list[int]]]]:
    """Relabel each stage column's raw (per-bbox-step, mutually incomparable) regime id onto
    the alignment yaml's canonical id, so the same physical regime carries the same id (and
    color) at every bbox step.

    Raw ids the yaml doesn't cover at a given bbox step are kept as negative sentinels
    (``-raw_id``) so they render distinctly instead of silently colliding with a real canonical
    id -- add them to the yaml's ``members`` once you've identified what they correspond to.

    Returns ``(relabeled_day_df, merges)`` -- see :func:`build_bbox_step_relabel_maps`'s
    ``merges`` return value.
    """
    relabel_maps, merges = build_bbox_step_relabel_maps(groups, stage_cols)
    out = day_df.copy()
    for step in stage_cols:
        raw_to_canon = relabel_maps[step]
        unmapped = sorted(set(out[step].unique()) - set(raw_to_canon))
        if unmapped:
            print(
                f"[alignment] WARNING: {step} raw regime id(s) {unmapped} not covered by any "
                "group's members in the alignment yaml -- shown as unmapped (negative id)."
            )
        if merges[step]:
            merge_desc = ", ".join(
                "+".join(f"Z{cid}" for cid in canon_ids) for canon_ids in merges[step].values()
            )
            print(
                f"[alignment] {step}: k is smaller than the physical-regime count -- "
                f"{merge_desc} share one raw cluster here (shown as one merged node)."
            )
        out[step] = out[step].map(lambda r: raw_to_canon.get(int(r), -int(r)))
    return out, merges
