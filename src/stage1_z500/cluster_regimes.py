#!/usr/bin/env python3
"""
Run Steps 8 → 11 (Z500 regime clustering) in one go.

CLI entrypoint: ``steps/stage1/02_cluster_regimes.py``.

This module is the missing glue between:
  - `src/pipeline_prepro.py` (Steps 1–8) and
  - the notebooks for Step 9 and Step 10/11.

It executes the notebooks *headlessly* via ``python -m nbconvert --execute`` while injecting
configuration through environment variables (REGION, BBOX_STEP, FINAL_K, etc.).

Clustering-only: no PDF/report assembly happens here (that legacy path — bbox summary PDFs via
``src/common`` — was removed; ``the removed z_04 composites job`` and
``the removed z_05 stats job`` produce the current PNG/CSV outputs from the
NetCDFs this script writes).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _ensure_src_on_path(module_root: Path) -> None:
    src = module_root / "src"
    s = str(src.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def _parse_steps(arg: str, steps_yaml: dict) -> list[str]:
    arg = arg.strip().lower()
    if arg in ("all", "*"):
        return sorted(steps_yaml.keys(), key=lambda x: (len(x), x))
    out: list[str] = []
    for tok in arg.split(","):
        t = tok.strip().upper()
        if not t:
            continue
        if t not in steps_yaml:
            raise SystemExit(f"Unknown bbox step {t!r}. Available: {sorted(steps_yaml)}")
        out.append(t)
    return out


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), env=env, check=False, text=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _normalize_cluster_mode(value: str) -> str:
    """'ward' (pure Ward HRC, Step-8 linkage cut at k) or 'ward_kmeans' (Ward-seeded K-means hybrid,
    the previous default behaviour whenever Step-8 exists). Mirrors the notebook-09 normalizer and
    ``_normalize_cluster_method`` in the removed Z500/precip composite script."""
    v = str(value).strip().lower().replace("-", "").replace("_", "")
    if v in ("ward", "w", "hrc", "wardonly"):
        return "ward"
    if v in ("ward+kmeans", "wardkmeans", "wardkm", "hybrid", "wk"):
        return "ward_kmeans"
    raise SystemExit(f"--cluster-mode must be 'ward' or 'ward+kmeans', got {value!r}")


def _method_tag(cluster_mode: str) -> str:
    """On-disk Step-9/10-11 filename family: pure Ward gets its own 'ward' tag; the Ward-seeded
    K-means hybrid keeps the existing 'kmeans' tag (it IS a K-means result, just seeded
    differently), so previously-computed hybrid runs stay valid without recomputation."""
    return "ward" if cluster_mode == "ward" else "kmeans"


def _opt_label(module_root: Path, region: str, bbox_step: str, args: argparse.Namespace) -> str:
    """Same ``detr*_clim*_*`` token as ``pipeline_prepro`` / notebooks (``preprocessing_helpers``)."""
    detr_label = "dm" if args.detrend_method == "domain_mean" else "gp"
    climsub_label = "clims" if args.climatology_subtract == "smooth" else "climr"
    sigma_label = "sigsc" if args.sigma_mode == "scalar" else "sigf"
    base = f"detr{detr_label}_{climsub_label}_{sigma_label}"
    if bool(getattr(args, "yeo_johnson", True)):
        base = f"{base}_yj"
    from common.preprocessing_helpers import bbox_step_filename_suffix  # noqa: WPS433

    return base + bbox_step_filename_suffix(module_root, region, bbox_step)


def _nc_readable(path: Path) -> bool:
    from common.preprocessing_helpers import _is_netcdf_readable  # noqa: WPS433

    return path.is_file() and _is_netcdf_readable(path)


def _paths_for_step(
    module_root: Path,
    region: str,
    bbox_step: str,
    *,
    k: int,
    args: argparse.Namespace,
) -> dict[str, Path]:
    w, eps_i, var_i = int(args.window_days), int(round(float(args.epsilon_m))), int(round(float(args.var_threshold) * 100))
    years = str(args.year_span)
    opt = _opt_label(module_root, region, bbox_step, args)
    from common.preprocessing_helpers import bbox_step_filename_suffix  # noqa: WPS433

    bbox_tag = bbox_step_filename_suffix(module_root, region, bbox_step)
    # Step 9 / 10-11 outputs must carry the same _yj disambiguation tag as Step 7/8
    # (via `opt`), otherwise a YJ rerun's "skip: output already exists" cache check
    # can't tell a pre-YJ result from a YJ one at the same path.
    yj_tag = "_yj" if bool(getattr(args, "yeo_johnson", True)) else ""
    method_tag = _method_tag(args.cluster_mode)
    processed = module_root / "data" / "processed" / region
    pca = processed / f"07_Z500_PCA_w{w:02d}_eps{eps_i:02d}_var{var_i:02d}_{opt}_{years}.nc"
    hc = processed / f"08_Z500_HC_Ward_w{w:02d}_eps{eps_i:02d}_var{var_i:02d}_{opt}_{years}.nc"
    reg = module_root / "results" / "clustering" / region / (
        f"09_Z500_regimes_{method_tag}_k{k:02d}_w{w:02d}_eps{eps_i:02d}_var{var_i:02d}{yj_tag}_{years}{bbox_tag}.nc"
    )
    s1011 = module_root / "results" / "regimes" / region / (
        f"10_11_Z500_regime_reconstruction_characterisation_{method_tag}_k{k:02d}_w{w:02d}_eps{eps_i:02d}_"
        f"var{var_i:02d}{yj_tag}_{years}{bbox_tag}.nc"
    )
    return {"pca": pca, "hc": hc, "regimes": reg, "step1011": s1011}


# ``09_final_clustering_regimes.ipynb`` has a manual "North America" section (cell id below)
# with hardcoded REGION and legacy filename globs. Headless runs must not execute it.
_NB09_NA_SECTION_CELL_ID = "na_section_header"


def _headless_notebook_for_execution(nb_path: Path, *, module_root: Path) -> Path:
    """Trim dual-region notebooks so automated runs only execute the env-driven block."""
    if nb_path.name != "09_final_clustering_regimes.ipynb":
        return nb_path
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells") or []
    trimmed: list[dict] = []
    for cell in cells:
        if cell.get("id") == _NB09_NA_SECTION_CELL_ID:
            break
        trimmed.append(cell)
    if len(trimmed) == len(cells):
        return nb_path
    # Keep under src/ so interactive ``Path("..")`` still works; batch runs use find_module_root().
    out = module_root / "src" / "_headless" / nb_path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    nb_out = dict(nb)
    nb_out["cells"] = trimmed
    out.write_text(json.dumps(nb_out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(
        f"[run_regime] headless notebook: dropped {len(cells) - len(trimmed)} "
        f"North-America-only cells from {nb_path.name}",
        flush=True,
    )
    return out


def _execute_notebook(
    nb_path: Path,
    *,
    out_nb: Path,
    cwd: Path,
    env: dict[str, str],
    timeout_s: int = -1,
    module_root: Path | None = None,
) -> None:
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    exec_nb = nb_path
    if module_root is not None:
        exec_nb = _headless_notebook_for_execution(nb_path, module_root=module_root)
    # Use ``python -m nbconvert`` so only the ``nbconvert`` package is required; ``jupyter nbconvert``
    # needs the full Jupyter CLI shim (often missing if only ``pip install nbconvert`` was done).
    cmd = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={int(timeout_s)}",
        "--output",
        out_nb.name,
        "--output-dir",
        str(out_nb.parent),
        str(exec_nb),
    ]
    _run(cmd, cwd=cwd, env=env)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    here = Path(__file__).resolve()
    module_root_hint = here.parents[1]  # .../1_Detect_Weather_Patterns (this file lives in src/)
    _ensure_src_on_path(module_root_hint)

    from common.preprocessing_helpers import find_module_root  # noqa: WPS433
    from common.resolver import k_for_step, load_bbox_steps_yaml, load_k_overrides  # noqa: WPS433

    p = argparse.ArgumentParser(
        description=(
            "Run Step 8/9/10_11 (Z500 regime clustering) for bbox steps. "
            "Skips pipeline Step 7/8 and notebooks 09 / 10–11 when the expected NetCDFs already exist "
            "(unless --force-prepro / --force-notebooks)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--region", required=True, choices=("europe", "northamerica"))
    p.add_argument("--steps", default="all", help='Comma-separated B0..B5 or "all".')
    p.add_argument("--bbox-yaml", type=Path, default=None)
    p.add_argument("--k-config", type=Path, default=None)
    p.add_argument("--window-days", type=int, default=11)
    p.add_argument("--epsilon-m", type=float, default=20.0)
    p.add_argument(
        "--var-threshold",
        type=float,
        default=0.90,
        help=(
            "PCA variance fraction for Step 7 filenames (0.90 → var90 in NetCDF names). "
            "Must match existing 07_* files; this is not the precipitation exceedance percentile."
        ),
    )
    p.add_argument("--year-span", type=str, default="1979-2025")
    # Must match filenames under data/preprocessed/<region>/ (see 03_run_pipeline_*_compare_options.sh).
    # pipeline_prepro defaults are gridpoint+raw+field (detrgp_climr_sigf); HPC "paper" runs use:
    p.add_argument(
        "--detrend-method",
        choices=("gridpoint", "domain_mean"),
        default="domain_mean",
        help="Forwarded to pipeline_prepro (drives detrgp vs detrdm in filenames).",
    )
    p.add_argument(
        "--climatology-subtract",
        choices=("raw", "smooth"),
        default="smooth",
        help="Forwarded to pipeline_prepro (climr vs clims in filenames).",
    )
    p.add_argument(
        "--sigma-mode",
        choices=("field", "scalar"),
        default="scalar",
        help="Forwarded to pipeline_prepro (sigf vs sigsc in filenames).",
    )
    yj = p.add_mutually_exclusive_group()
    yj.add_argument(
        "--yeo-johnson",
        dest="yeo_johnson",
        action="store_true",
        default=None,
        help="Use preprocessing NetCDFs with _yj_ in filenames (default: on).",
    )
    yj.add_argument(
        "--no-yeo-johnson",
        dest="yeo_johnson",
        action="store_false",
        help="Use preprocessing without Yeo-Johnson (_yj_) files.",
    )
    p.add_argument(
        "--stop-after-step",
        type=int,
        default=8,
        help="How far to run `pipeline_prepro.py` (7=PCA, 8=PCA+HC). Steps 2–6 must already exist (default pipeline behaviour).",
    )
    p.add_argument(
        "--skip-prepro",
        action="store_true",
        help="Skip running pipeline_prepro.py (assume Step 8 NetCDF already exists).",
    )
    p.add_argument(
        "--skip-notebooks",
        action="store_true",
        help="Skip running notebooks 09 and 10_11 (assume outputs already exist).",
    )
    p.add_argument(
        "--force-prepro",
        action="store_true",
        help="Run pipeline_prepro even if the target Step 7/8 NetCDF for this tag already exists.",
    )
    p.add_argument(
        "--force-notebooks",
        action="store_true",
        help="Execute notebooks 09 and 10_11 even if their output NetCDFs already exist.",
    )
    p.add_argument(
        "--cluster-mode",
        default="ward",
        help=(
            "Stage-1 Z500 regime clustering used by notebook 09: 'ward' = pure Ward hierarchical "
            "clustering (Step-8 linkage cut at k, no K-means at all); 'ward+kmeans' = Ward-seeded "
            "K-means (Step-8 centroids as K-means init — the previous default/hybrid behaviour, "
            "still a K-means fit_predict, just no longer randomly initialised). Sets CLUSTER_MODE "
            "for notebooks 09/10_11 and picks the 'ward' vs 'kmeans' tag in their output filenames "
            "(the ward+kmeans hybrid keeps the existing 'kmeans' tag so already-computed hybrid runs "
            "stay valid). 'ward' requires the Step-8 08_Z500_HC_Ward_*.nc to exist for each bbox/region."
        ),
    )
    p.add_argument(
        "--execute-notebook-08",
        action="store_true",
        help=(
            "After pipeline for each bbox step, also execute stage1_z500/notebooks/08_hierarchical_clustering.ipynb via nbconvert "
            "(when Step 8 HC NetCDF exists). Optional; pipeline already writes HC without this notebook."
        ),
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=-1,
        help="Notebook execution timeout in seconds (-1 = no timeout).",
    )
    p.add_argument(
        "--module-root",
        type=Path,
        default=None,
        help="1_Detect_Weather_Patterns root (auto-detected if omitted).",
    )
    args = p.parse_args(argv)

    from common.preprocessing_helpers import yeo_johnson_enabled  # noqa: WPS433

    if args.yeo_johnson is None:
        args.yeo_johnson = yeo_johnson_enabled()
    args.cluster_mode = _normalize_cluster_mode(args.cluster_mode)

    module_root = (
        args.module_root.expanduser().resolve()
        if args.module_root is not None
        else find_module_root(module_root_hint)
    )
    _ensure_src_on_path(module_root)

    bbox_yaml = args.bbox_yaml or (module_root / "src" / f"bbox_steps_{args.region}.yaml")
    # Default to the per-region production k-overrides, matching what Stage 2's
    # feature_relevance resolves; a bare "k_overrides.yaml" has never existed.
    k_config_path = args.k_config or (
        module_root / "src" / "common" / f"k_overrides_{args.region}_custom.yaml"
    )

    steps_cfg = load_bbox_steps_yaml(bbox_yaml.resolve())
    k_cfg = load_k_overrides(k_config_path.resolve())
    step_ids = _parse_steps(args.steps, steps_cfg)

    base_env = os.environ.copy()
    base_env["MPLBACKEND"] = base_env.get("MPLBACKEND", "Agg")
    base_env["PYTHONUNBUFFERED"] = "1"
    # nbconvert runs a fresh Python interpreter (cwd=src/); src/ must be on PYTHONPATH.
    _src_dir = str((module_root / "src").resolve())
    _prev_pp = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = f"{_src_dir}:{_prev_pp}" if _prev_pp else _src_dir
    base_env["MODULE_ROOT"] = str(module_root.resolve())
    base_env["REGION"] = args.region
    base_env["WINDOW_DAYS"] = str(int(args.window_days))
    base_env["EPSILON_M"] = str(float(args.epsilon_m))
    base_env["VAR_THRESHOLD"] = str(float(args.var_threshold))
    base_env["YEAR_SPAN"] = str(args.year_span)
    base_env["SIGMA_MODE"] = str(args.sigma_mode)
    base_env["DETREND_METHOD"] = str(args.detrend_method)
    base_env["CLIM_SUBTRACT"] = str(args.climatology_subtract)
    base_env["YEO_JOHNSON"] = "1" if args.yeo_johnson else "0"
    base_env["CLUSTER_MODE"] = args.cluster_mode
    print(f"[run_regime] CLUSTER_MODE={args.cluster_mode!r}", flush=True)

    nb09 = module_root / "src" / "stage1_z500" / "notebooks" / "09_final_clustering_regimes.ipynb"
    nb1011 = module_root / "src" / "stage1_z500" / "notebooks" / "10_11_regime_reconstruction_characterisation.ipynb"
    nb08 = module_root / "src" / "stage1_z500" / "notebooks" / "08_hierarchical_clustering.ipynb"
    if not nb09.is_file() or not nb1011.is_file():
        raise SystemExit("Missing notebooks under src/stage1_z500/notebooks/: 09_final_clustering_regimes.ipynb and/or 10_11_regime_reconstruction_characterisation.ipynb")

    exec_out_dir = module_root / "results" / "executed_notebooks" / args.region

    for step in step_ids:
        k = k_for_step(k_cfg, step)
        env = dict(base_env)
        env["BBOX_STEP"] = step
        env["FINAL_K"] = str(int(k))

        paths = _paths_for_step(module_root, args.region, step, k=k, args=args)

        if not args.skip_prepro:
            stop_i = int(args.stop_after_step)
            skip_pre = False
            if not args.force_prepro:
                if stop_i >= 8 and _nc_readable(paths["hc"]):
                    skip_pre = True
                    print(f"[run_regime] skip pipeline: Step 8 HC exists: {paths['hc']}", flush=True)
                elif stop_i == 7 and _nc_readable(paths["pca"]):
                    skip_pre = True
                    print(f"[run_regime] skip pipeline: Step 7 PCA exists: {paths['pca']}", flush=True)
            if not skip_pre:
                cmd_prepro = [
                    sys.executable,
                    str(module_root / "src" / "pipeline_prepro.py"),
                    "--region",
                    args.region,
                    "--smooth-window",
                    str(int(args.window_days)),
                    "--epsilon-m",
                    str(float(args.epsilon_m)),
                    "--var-threshold",
                    str(float(args.var_threshold)),
                    "--detrend-method",
                    str(args.detrend_method),
                    "--climatology-subtract",
                    str(args.climatology_subtract),
                    "--sigma-mode",
                    str(args.sigma_mode),
                    # pipeline_prepro / preprocessing_helpers use --raw-years-span (not --year-span)
                    "--raw-years-span",
                    str(args.year_span),
                    "--bbox-step",
                    step,
                    "--stop-after-step",
                    str(int(args.stop_after_step)),
                    "--bbox-config",
                    str(bbox_yaml.resolve()),
                ]
                if args.yeo_johnson:
                    cmd_prepro.append("--yeo-johnson")
                _run(cmd_prepro, cwd=module_root, env=env)

        if args.execute_notebook_08 and nb08.is_file() and _nc_readable(paths["hc"]):
            _execute_notebook(
                nb08,
                out_nb=exec_out_dir / f"08_hierarchical_clustering__{step}.ipynb",
                cwd=module_root / "src",
                env=env,
                timeout_s=args.timeout_s,
                module_root=module_root,
            )

        if not args.skip_notebooks:
            if not args.force_notebooks and _nc_readable(paths["regimes"]):
                print(f"[run_regime] skip notebook 09: exists: {paths['regimes']}", flush=True)
            else:
                nb09_env = dict(env)
                if _nc_readable(paths["hc"]):
                    nb09_env["IN_STEP8_HC"] = str(paths["hc"].resolve())
                elif args.cluster_mode == "ward":
                    raise SystemExit(
                        f"[run_regime] --cluster-mode=ward requires Step 8 HC NetCDF: {paths['hc']} "
                        "(not found/readable). Run Step 8 first (pipeline_prepro / z_03 with "
                        "--stop-after-step 8), or pass --cluster-mode ward+kmeans."
                    )
                _execute_notebook(
                    nb09,
                    out_nb=exec_out_dir / f"09_final_clustering_regimes__{step}.ipynb",
                    cwd=module_root / "src",
                    env=nb09_env,
                    timeout_s=args.timeout_s,
                    module_root=module_root,
                )
            if not args.force_notebooks and _nc_readable(paths["step1011"]):
                print(f"[run_regime] skip notebook 10_11: exists: {paths['step1011']}", flush=True)
            else:
                nb_env = dict(env)
                if _nc_readable(paths["regimes"]):
                    nb_env["IN_STEP9_REGIMES"] = str(paths["regimes"].resolve())
                _execute_notebook(
                    nb1011,
                    out_nb=exec_out_dir / f"10_11_regime_reconstruction_characterisation__{step}.ipynb",
                    cwd=module_root / "src",
                    env=nb_env,
                    timeout_s=args.timeout_s,
                    module_root=module_root,
                )

    print("[run_regime_clustering] SUCCESS: pipeline finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

