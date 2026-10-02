#!/usr/bin/env python3
"""Write rr_01 field selection results to a YAML file.

Reads feature_relevance_sequential_{region}_{bbox}_{mode}_optionB.csv files produced by
rr_01, applies the two-criterion selection rule, and writes the selected fields
per region/bbox to src/common/rr01_field_selection.yaml.

Selection rule
--------------
  Criterion 1: include fields until cumulative AUC >= AUC_FLOOR (default 0.80)
  Criterion 2: additionally include any field with marginal gain >= MARGINAL_THRESHOLD (default 0.01)

The resulting YAML is read by rr_02 to set RESIDUAL_FIELDS automatically.

Usage
-----
  # From HPC scripts/ dir (after rr_01 jobs complete):
  python -m stage2_residual.write_field_selection \\
      --results-dir results/residual_regimes/feature_relevance \\
      --module-root .
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from stage2_residual.rr01_paths import resolve_rr01_csv, resolve_rr01_per_regime_csv


def select_fields(
    csv_path: Path,
    auc_floor: float,
    marginal_threshold: float,
    max_fields: int | None = None,
) -> list[str]:
    """Apply two-criterion rule and return selected field list.

    `max_fields`, if given, caps the number of non-z500 fields returned (keeping
    the first `max_fields` in greedy AUC order) — the AUC-floor/marginal-gain
    rule can still stop earlier for a regime whose AUC saturates fast, but never
    returns more than `max_fields`. Use this for an "optimal, but capped" mode
    that stays comparable in max dimensionality to `select_top_n_fields`
    without forcing every regime up to the same fixed count.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    fields = df["field_combination"].tolist()
    auc = df["cumulative_auc"].tolist()
    gains = [auc[0]] + [auc[i] - auc[i - 1] for i in range(1, len(auc))]

    selected, floor_reached = [], False
    for field, gain, cum in zip(fields, gains, auc):
        if (not floor_reached) or gain >= marginal_threshold:
            selected.append(field.replace("Z500 baseline", "z500"))
        if cum >= auc_floor:
            floor_reached = True
    if max_fields is not None and selected:
        head, rest = selected[0], selected[1:]
        selected = [head] + rest[:max_fields]
    return selected


def select_fields_decline_stop(csv_path: Path, auc_floor: float, min_fields: int) -> list[str]:
    """Greedy order from rr_01, with a hard decline-stop: z500 is always kept, then
    non-z500 fields are added in rr_01's own greedy AUC order until BOTH
    ``min_fields`` are selected AND cumulative AUC >= ``auc_floor`` — but the very
    first field whose addition makes cumulative AUC go DOWN relative to the
    previous step ends selection immediately, keeping the previous (higher-AUC)
    field set. This decline-stop check overrides both the ``min_fields`` and
    ``auc_floor`` requirements — even if neither has been satisfied yet, a decline
    still stops selection (e.g. fields 1-5 reach 0.67, still under the 0.80 floor,
    but field 6 drops it to 0.65: selection stops at 5, 0.67 is reported, the
    floor is never forced).

    Differs from ``select_fields``'s AUC-floor + marginal-gain rule, which keeps
    adding every remaining field unconditionally (gain sign included) until the
    floor is first reached — that rule has no mechanism to stop once AUC has
    started declining, which is what motivated this variant.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    field_col = df["field_combination"].tolist()
    auc_col = df["cumulative_auc"].tolist()
    has_baseline = bool(field_col) and field_col[0] == "Z500 baseline"
    fields = field_col[1:] if has_baseline else field_col
    auc = auc_col[1:] if has_baseline else auc_col
    prev_auc = auc_col[0] if has_baseline else None

    selected = ["z500"]
    for field, cum in zip(fields, auc):
        if prev_auc is not None and cum < prev_auc:
            break
        selected.append(field)
        prev_auc = cum
        if (len(selected) - 1) >= min_fields and cum >= auc_floor:
            break
    return selected


def select_top_n_fields(csv_path: Path, n: int) -> list[str]:
    """Fixed-count field selection: z500 plus the top `n` non-z500 fields, in the
    greedy sequential order rr_01 already computed (each field was added at the
    stage where it gave the best marginal AUC gain among what remained).

    Unlike `select_fields`, there is no AUC-floor or marginal-gain-threshold
    filter — every regime gets exactly `n` fields regardless of how quickly AUC
    saturates for that regime. Use this when the goal is a controlled comparison
    across Z500 regimes (e.g. does subregime stability improve with more
    features?) where held-fixed dimensionality matters more than each regime
    picking its own "natural" stopping point.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    fields = [f for f in df["field_combination"].tolist() if f != "Z500 baseline"]
    return ["z500"] + fields[:n]


def build_regime_fields_map(
    results_dir: Path,
    region: str,
    bbox_step: str,
    z500_k: int,
    *,
    auc_floor: float = 0.80,
    marginal_threshold: float = 0.01,
    top_n: int | None = None,
    max_fields: int | None = None,
    decline_stop_min_fields: int | None = None,
    tag_suffix: str = "",
    z500_cluster_method: str = "kmeans",
    field_representation: str = "domain_mean",
) -> dict[int, str]:
    """Build ``{z500_regime_id: '+'-joined field string}`` from rr_01 MODE=regime
    per-regime CSVs, independently per Z500 regime.

    ``z500_cluster_method``: must match whatever rr_01 MODE=regime was run with
    ("kmeans"/hybrid default, or "ward" — reads the "_zward"-tagged CSVs). Passing
    the wrong value here silently selects the OTHER method's field set — it does
    not raise, since kmeans-tagged files are the untagged default and always
    "exist" once any kmeans run has completed.

    ``field_representation``: must match whatever rr_01 MODE=regime was actually run
    with for this region/bbox ("domain_mean", production default, or "pca_block" —
    see ``feature_relevance.py``'s module docstring). Unlike ``z500_cluster_method``,
    both representations are now always explicitly tagged in the CSV filename (see
    ``rr01_paths.rr01_tag``), so passing the wrong one raises ``FileNotFoundError``
    below rather than silently reading the other representation's file.

    Four selection rules, in precedence order (``top_n`` beats
    ``decline_stop_min_fields`` beats ``max_fields``):
      - ``top_n=N`` ("fixed3" mode): `select_top_n_fields` — every regime gets
        exactly N fields (plus z500), fixed count regardless of AUC floor. Use
        for controlled comparisons where held-fixed dimensionality matters more
        than each regime's own natural stopping point.
      - ``decline_stop_min_fields=N`` ("decline-stop" mode): `select_fields_decline_stop`
        — z500 plus fields in rr_01's own greedy order until at least N fields are
        selected AND cumulative AUC >= ``auc_floor``, but stopping immediately (and
        discarding the field that caused it) the moment a field's addition makes
        cumulative AUC go down — this decline-stop check overrides both the N-field
        minimum and the AUC floor, even if neither is satisfied yet.
      - ``max_fields=N`` ("optimal, capped" mode): the two-criterion
        `select_fields` rule (AUC floor + marginal-gain threshold), but capped
        at N non-z500 fields — a regime whose AUC saturates fast can still get
        fewer, just never more than N.
      - Default (all None): the uncapped two-criterion `select_fields` rule —
        each regime stops wherever its own AUC curve says to, with no ceiling.

    Single source of truth for "which fields does each Z500 regime cluster on" —
    shared by rr_022 (production per-regime clustering) and rr_021 (subregime-k
    selection), which both need the identical field subset per regime (pass the
    same mode/params to both — see FIELD_MODE in their respective HPC scripts).
    Previously duplicated as a bash `for z in ...` loop with one container-exec'd
    python heredoc per regime in rr_02_submit_per_regime_from_rr01.sh (now
    rr_05_clustering_test_KMEANS_perregime.sh); now a single call.
    """
    field_map: dict[int, str] = {}
    missing: list[int] = []
    for z in range(1, int(z500_k) + 1):
        csv = resolve_rr01_per_regime_csv(
            results_dir, region, bbox_step, z,
            tag_suffix=tag_suffix, z500_cluster_method=z500_cluster_method,
            field_representation=field_representation,
        )
        if csv is None:
            missing.append(z)
            continue
        if top_n is not None:
            fields = select_top_n_fields(csv, top_n)
        elif decline_stop_min_fields is not None:
            fields = select_fields_decline_stop(csv, auc_floor, decline_stop_min_fields)
        else:
            fields = select_fields(csv, auc_floor, marginal_threshold, max_fields=max_fields)
        field_map[z] = "+".join(fields)
    if missing:
        raise FileNotFoundError(
            f"rr_01 MODE=regime output missing for Z500 regime(s) {missing} "
            f"({region}/{bbox_step}, tag_suffix={tag_suffix!r}); run rr_01 with MODE=regime first."
        )
    return field_map


def write_yaml(
    results_dir: Path,
    out_yaml: Path,
    auc_floor: float = 0.80,
    marginal_threshold: float = 0.01,
    regions: tuple[str, ...] = ("europe", "northamerica"),
    bbox_steps: tuple[str, ...] = ("B0", "B1", "B2", "B3", "B4", "B5"),
    mode: str = "bbox",
    field_representation: str = "domain_mean",
) -> Path:
    """Read all rr_01 CSVs and write field selection YAML."""
    selection: dict[str, dict[str, str]] = {}
    missing: list[str] = []

    for region in regions:
        selection[region] = {}
        for bbox in bbox_steps:
            csv = resolve_rr01_csv(results_dir, region, bbox, "sequential_optionB", mode=mode,
                                    field_representation=field_representation)
            if csv is None:
                missing.append(f"{region}/{bbox}")
                continue
            fields = select_fields(csv, auc_floor, marginal_threshold)
            selection[region][bbox] = "+".join(fields)

    if missing:
        print(f"[write_field_selection] WARNING: CSVs not yet available for: {', '.join(missing)}")

    lines = [
        "# Auto-generated by rr_01 — do not edit manually.",
        "# Regenerate after re-running rr_01:",
        "#   python -m stage2_residual.write_field_selection",
        "#",
        "# Selection criteria:",
        f"#   Criterion 1: cumulative AUC >= {auc_floor:.0%}",
        f"#   Criterion 2: marginal AUC gain >= {marginal_threshold:.0%}",
        "#",
        f'generated: "{date.today()}"',
        f"auc_floor: {auc_floor}",
        f"marginal_threshold: {marginal_threshold}",
        "",
        "field_selection:",
    ]
    for region in regions:
        lines.append(f"  {region}:")
        for bbox in bbox_steps:
            if bbox in selection[region]:
                lines.append(f'    {bbox}: "{selection[region][bbox]}"')
            else:
                lines.append(f'    {bbox}: ""  # not yet computed')

    # Atomic write: write to a temp file then rename so parallel jobs never
    # leave the YAML in a partially-written state.
    import os, tempfile
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=out_yaml.parent, delete=False, suffix=".tmp", encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    os.replace(tmp_path, out_yaml)  # atomic on POSIX (Linux/HPC)
    print(f"[write_field_selection] Written: {out_yaml}")
    for region in regions:
        print(f"  {region}:")
        for bbox, fields in selection[region].items():
            print(f"    {bbox}: {fields}")
    return out_yaml


def read_fields_for(yaml_path: Path, region: str, bbox_step: str) -> str | None:
    """Read RESIDUAL_FIELDS for a given region/bbox from the YAML.

    Returns the '+'-separated field string, or None if not found.
    Used by rr_02 bash scripts via: python -m stage2_residual.write_field_selection --get
    """
    import re

    if not yaml_path.exists():
        return None
    txt = yaml_path.read_text()
    # Parse the field_selection block
    region_block = re.search(
        rf"^\s+{re.escape(region)}:\n((?:\s+\S.*\n)*)",
        txt,
        re.MULTILINE,
    )
    if not region_block:
        return None
    m = re.search(
        rf'^\s+{re.escape(bbox_step)}:\s+"?([^"\n]+)"?',
        region_block.group(0),
        re.MULTILINE,
    )
    if not m or not m.group(1).strip():
        return None
    return m.group(1).strip()


def main() -> int:
    p = argparse.ArgumentParser(description="Write or query rr_01 field selection YAML.")
    p.add_argument("--module-root", type=Path, default=None,
                   help="Module root (auto-detected if omitted).")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="Path to feature_relevance results dir "
                        "(default: <module-root>/results/residual_regimes/feature_relevance).")
    p.add_argument("--out-yaml", type=Path, default=None,
                   help="Output YAML path (default: <module-root>/src/common/rr01_field_selection.yaml).")
    p.add_argument("--auc-floor", type=float, default=0.80)
    p.add_argument("--marginal-threshold", type=float, default=0.01)
    p.add_argument("--mode", choices=("bbox", "regime"), default="bbox",
                   help="rr_01 output mode tag in CSV filenames (default: bbox for rr_02).")
    # Query mode: read a single field set for a given region/bbox
    p.add_argument("--get", action="store_true",
                   help="Query mode: print RESIDUAL_FIELDS for --region/--bbox-step and exit.")
    p.add_argument("--region", default=None)
    p.add_argument("--bbox-step", default=None)
    # Per-regime field map mode: build {z500_regime_id: fields} from rr_01 MODE=regime CSVs
    p.add_argument("--build-regime-map", action="store_true",
                   help="Print JSON {z500_regime_id: '+'-joined fields} for --region/--bbox-step/"
                        "--z500-k, built from rr_01 MODE=regime per-regime CSVs, and exit.")
    p.add_argument("--z500-k", type=int, default=None)
    p.add_argument("--top-n-fields", type=int, default=None,
                   help="Fixed-count field selection ('fixed3' mode): every Z500 regime gets "
                        "exactly this many non-z500 fields (by greedy AUC-contribution order), "
                        "ignoring --auc-floor/--marginal-threshold/--max-fields. For controlled "
                        "comparisons across regimes with held-fixed dimensionality.")
    p.add_argument("--max-fields", type=int, default=None,
                   help="Cap on non-z500 fields for the AUC-floor/marginal-gain rule ('optimal, "
                        "capped' mode) — a regime can still get fewer if its own AUC saturates "
                        "fast, but never more than this. Ignored if --top-n-fields or "
                        "--decline-stop-min-fields is set.")
    p.add_argument("--decline-stop-min-fields", type=int, default=None,
                   help="'decline-stop' mode: z500 plus fields in rr_01's own greedy order "
                        "until at least this many non-z500 fields are selected AND cumulative "
                        "AUC >= --auc-floor, but stopping immediately (keeping the previous, "
                        "higher-AUC field set) the moment a field's addition makes cumulative "
                        "AUC decrease — overrides both the field-count minimum and the AUC "
                        "floor. Ignored if --top-n-fields is set; takes precedence over "
                        "--max-fields.")
    p.add_argument("--tag-suffix", default="",
                   help="Cross-bbox diagnostic tag suffix (matches rr_01's TAG_SUFFIX), if any.")
    p.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="kmeans",
                   help="--build-regime-map only: which rr_01 MODE=regime per-regime CSVs to "
                        "read (must match the Z500 parent-regime source rr_01 was run against — "
                        "'ward' reads the '_zward'-tagged CSVs). Ignored otherwise.")
    p.add_argument("--field-representation", choices=("domain_mean", "pca_block"), default="domain_mean",
                   help="Which rr_01 field-representation CSVs to read (must match how rr_01 was "
                        "actually run for this region/bbox/mode — see feature_relevance.py's "
                        "--field-representation). Default 'domain_mean' matches production. Both "
                        "representations are always explicitly tagged in rr_01's filenames (see "
                        "rr01_paths.rr01_tag), so passing the wrong one here raises "
                        "FileNotFoundError instead of silently reading the other one's results.")
    args = p.parse_args()

    # Resolve module root
    if args.module_root is None:
        try:
            from common.preprocessing_helpers import find_module_root
            module_root = find_module_root()
        except Exception:
            module_root = Path(__file__).resolve().parents[2]
    else:
        module_root = args.module_root.resolve()

    out_yaml = args.out_yaml or (module_root / "src" / "common" / "rr01_field_selection.yaml")

    # Per-regime field map mode: print JSON {z500_regime_id: fields} and exit.
    if args.build_regime_map:
        if not args.region or not args.bbox_step or not args.z500_k:
            print("ERROR: --build-regime-map requires --region, --bbox-step, and --z500-k", file=sys.stderr)
            return 1
        results_dir = args.results_dir or (module_root / "results" / "stage2_residual" / "feature_relevance")
        try:
            field_map = build_regime_fields_map(
                results_dir, args.region, args.bbox_step, args.z500_k,
                auc_floor=args.auc_floor, marginal_threshold=args.marginal_threshold,
                top_n=args.top_n_fields, max_fields=args.max_fields,
                decline_stop_min_fields=args.decline_stop_min_fields,
                tag_suffix=args.tag_suffix, z500_cluster_method=args.z500_cluster_method,
                field_representation=args.field_representation,
            )
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        import json
        print(json.dumps(field_map))
        return 0

    # Query mode: just print the fields for one region/bbox
    if args.get:
        if not args.region or not args.bbox_step:
            print("ERROR: --get requires --region and --bbox-step", file=sys.stderr)
            return 1
        fields = read_fields_for(out_yaml, args.region, args.bbox_step)
        if fields:
            print(fields)
            return 0
        else:
            print(f"[write_field_selection] No entry for {args.region}/{args.bbox_step} in {out_yaml}",
                  file=sys.stderr)
            return 1

    # Write mode
    results_dir = args.results_dir or (module_root / "results" / "stage2_residual" / "feature_relevance")
    write_yaml(
        results_dir=results_dir,
        out_yaml=out_yaml,
        auc_floor=args.auc_floor,
        marginal_threshold=args.marginal_threshold,
        mode=args.mode,
        field_representation=args.field_representation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
