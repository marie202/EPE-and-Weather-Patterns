from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from common.preprocessing_helpers import find_module_root

from stage2_residual.fields.field_specs import default_field_order


def _parse_bbox_string(bbox: str | None) -> str | None:
    raw = (bbox or "").strip()
    return raw or None


def _parse_plus_separated_fields(fields_str: str) -> tuple[str, ...]:
    """Parse plus-separated or comma-separated field list."""
    sep = "+" if "+" in fields_str else ","
    return tuple(f.strip().lower() for f in fields_str.split(sep) if f.strip())


def _load_json_or_file(value: str) -> dict:
    """Parse a CLI value that may be an inline JSON string or a path to a JSON/YAML file."""
    path = Path(value).expanduser()
    if path.is_file():
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(text)
        return json.loads(text)
    return json.loads(value)


def _parse_regime_fields_map(value: str) -> dict[int, tuple[str, ...]]:
    raw = _load_json_or_file(value)
    return {int(z): _parse_plus_separated_fields(str(fields)) for z, fields in raw.items()}


def _parse_regime_k_spec(value: str) -> dict[int, int | str]:
    raw = _load_json_or_file(value)
    out: dict[int, int | str] = {}
    for z, k in raw.items():
        out[int(z)] = k if isinstance(k, str) else int(k)
    return out


def _parse_final_k_range(value: str) -> tuple[int, ...]:
    return tuple(int(k.strip()) for k in value.split(",") if k.strip())


@dataclass(frozen=True)
class ResidualRunConfig:
    """Configuration for residual regimes two-stage clustering.

    Stage 1: Cluster Z500 to define circulation regimes (z500_k).
    Bridge: Compute residuals for each impact field (residual_fields).
    Stage 2: Cluster residual PCs with k-means (final_k).
    """

    module_root: Path = field(default_factory=find_module_root)
    region: str = "europe"
    bbox_step: str = "B5"
    bbox_config: Path | None = None
    bbox: str | None = None
    raw_years_span: str = "1979-2025"
    year_span: str = "1979-2025"
    smooth_window: int = 11
    detrend_method: str = "domain_mean"
    climatology_subtract: str = "smooth"
    sigma_mode: str = "field"
    yeo_johnson: bool = True  # Stage 2 PCA is reused directly from rr_00's 07_MF_PCA files
        # (see _find_mf_pca_path in auxiliary.py), which are always produced with
        # SIGMA_MODE=field + YEO_JOHNSON=1 (see run_configs.yaml) — these defaults must
        # match or the lookup will report the file as missing.
    var_threshold: float = 0.90
    max_modes_per_field: int | None = 100  # Caps each field's retained PCA modes at this many
        # (leading modes, already ordered by descending explained variance) even if
        # var_threshold would otherwise retain more — see per_field_pca.run_per_field_pca's
        # max_modes param and MultivariateRunConfig.max_modes_per_field (must match rr_00's
        # setting for resolve_mf_pca_path lookups to hit the same files). None disables the cap.
    z500_k: int = 5  # Stage 1: number of Z500 circulation regimes
    z500_cluster_method: str = "kmeans"  # On-disk Stage 1 Z500 regimes filename family to
        # auto-discover when z500_regimes_override is not set: "kmeans" (default; also covers
        # the Ward-seeded K-means hybrid, z_03's CLUSTER_MODE=ward+kmeans, which keeps the
        # "kmeans" tag) or "ward" (pure Ward HRC, z_03's CLUSTER_MODE=ward). Ignored when
        # z500_regimes_override is set.
    z500_regimes_override: Path | None = None  # Explicit Stage 1 Z500 regimes NetCDF, bypassing
        # the usual bbox_step-based legacy-file search — for cross-bbox diagnostics (e.g.
        # classifying this bbox's residual fields using ANOTHER bbox's Z500 circulation
        # regimes/k, to isolate whether the Z500 regime DEFINITION itself changes the
        # resulting subregimes). z500_k must still be set to match the override file's actual
        # regime count (it is NOT read from the file). Mirrors rr_01's --z500-regimes-path.
    final_k: int = 7  # Stage 2: number of final k-means clusters (global mode).
        # In per_regime_mode this becomes a path-naming placeholder only — it has
        # no effect on clustering, since each Z500 regime picks its own k via
        # per_regime_k_spec (explicit or "auto").
    per_regime_mode: bool = False  # Stage 2: cluster residuals independently per Z500 regime,
        # each with its own field subset (per_regime_field_map) and own k (per_regime_k_spec),
        # instead of one global joint clustering. Additive — default False preserves today's
        # global-mode behavior exactly.
    per_regime_field_map: dict[int, tuple[str, ...]] | None = None  # {z500_regime_id: (fields,...)}
    per_regime_k_spec: dict[int, int | str] | None = None  # {z500_regime_id: k or "auto"}
    residualize: bool = False  # Stage 2 [per-regime only]: Option 2 of the three Stage-2
        # conditioning modes (README_ResidualRegimes.md) — subtract EACH Z500 regime's own
        # day-subset mean from its joint features before clustering, i.e.
        # residual[t] = anomaly[t] - mean(anomaly | regime[t]). Requires per_regime_mode=True
        # (see validation below) — global mode is deliberately left as the unconditioned
        # null-baseline (Option 1) and is not affected by this flag. Additive — default False
        # preserves the pre-Jul-2026 (non-residualized) per-regime behavior exactly. Folded
        # into per_regime_fields_signature/per_regime_signature (paths.py) so residualized and
        # non-residualized runs never collide on cached joint-features/Ward-HC/regimes files.
    final_k_range: tuple[int, ...] = (3, 4, 5, 6, 7, 8)  # candidate k's for per-regime "auto" silhouette sweep
    random_state: int = 42
    stop_after_step: int = 10
    z500_epsilon: float = 20.0
    vo850_epsilon: float = 1.0e-6
    tcwv_epsilon: float = 1.0e-3
    deg0l_epsilon: float = 50.0
    cape_epsilon: float = 1.0
    w700_epsilon: float = 0.01
    rh700_epsilon: float = 0.1
    vo850_smoothing_method: str = "none"
    vo850_gaussian_sigma: float = 1.0
    vo850_coarsen_factor: int = 0
    residual_fields: tuple[str, ...] = field(default_factory=lambda: ("z500", "vo850", "tcwv", "deg0l", "cape", "w700", "rh700"))
    overwrite: bool = False
    dry_run: bool = False
    area_weight_tag: str = ""  # Must match the rr_00 MultivariateRunConfig.area_weight_tag used
        # to produce the 07_MF_PCA files this pipeline reuses (see resolve_mf_pca_path in
        # paths.py) — when set, PCA lookup requires this tag in the filename instead of
        # accepting whichever file the glob happens to match, so Stage 2 doesn't silently
        # pick up a rerun before it's been verified. Empty (default) preserves today's
        # loose-glob behavior against untagged files.
    z500_raw_nc: Path | None = None
    vo850_raw_nc: Path | None = None
    tcwv_raw_nc: Path | None = None
    deg0l_raw_nc: Path | None = None
    cape_raw_nc: Path | None = None
    w700_raw_nc: Path | None = None
    rh700_raw_nc: Path | None = None

    def __post_init__(self) -> None:
        module_root = Path(self.module_root).expanduser().resolve()
        object.__setattr__(self, "module_root", module_root)

        if self.region not in {"europe", "northamerica"}:
            raise ValueError(f"Unsupported region={self.region!r}")
        if self.smooth_window <= 0 or self.smooth_window % 2 != 1:
            raise ValueError(f"smooth_window must be a positive odd integer, got {self.smooth_window}")
        if self.detrend_method not in {"gridpoint", "domain_mean"}:
            raise ValueError(f"Unsupported detrend_method={self.detrend_method!r}")
        if self.climatology_subtract not in {"raw", "smooth"}:
            raise ValueError(f"Unsupported climatology_subtract={self.climatology_subtract!r}")
        if self.sigma_mode not in {"field", "scalar"}:
            raise ValueError(f"Unsupported sigma_mode={self.sigma_mode!r}")
        if not (0.0 < float(self.var_threshold) <= 1.0):
            raise ValueError(f"var_threshold must be in (0, 1], got {self.var_threshold}")
        if self.max_modes_per_field is not None and int(self.max_modes_per_field) < 1:
            raise ValueError(f"max_modes_per_field must be >= 1 or None, got {self.max_modes_per_field}")
        if int(self.z500_k) < 2:
            raise ValueError(f"z500_k must be >= 2, got {self.z500_k}")
        if int(self.final_k) < 2:
            raise ValueError(f"final_k must be >= 2, got {self.final_k}")
        if int(self.stop_after_step) not in {6, 7, 8, 9, 10}:
            raise ValueError("stop_after_step must be one of {6, 7, 8, 9, 10}")
        if self.vo850_smoothing_method not in {"none", "gaussian", "coarsen"}:
            raise ValueError(f"Unsupported vo850_smoothing_method={self.vo850_smoothing_method!r}")
        if float(self.vo850_gaussian_sigma) < 0.0:
            raise ValueError("vo850_gaussian_sigma must be >= 0")
        if int(self.vo850_coarsen_factor) < 0:
            raise ValueError("vo850_coarsen_factor must be >= 0")

        if self.residualize and not self.per_regime_mode:
            raise ValueError(
                "residualize=True requires per_regime_mode=True — regime-conditional mean "
                "subtraction only makes sense within a single Z500 regime's own day-subset "
                "(see README_ResidualRegimes.md, 'Three Stage-2 conditioning modes'). Global "
                "mode is the deliberate unconditioned baseline and is not residualized."
            )

        if self.per_regime_mode:
            if not self.per_regime_field_map:
                raise ValueError("per_regime_mode=True requires a non-empty per_regime_field_map")
            expected_ids = set(range(1, int(self.z500_k) + 1))
            actual_ids = set(self.per_regime_field_map.keys())
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(
                    f"per_regime_field_map must have exactly one entry per Z500 regime "
                    f"1..{self.z500_k}; missing={missing} extra={extra}"
                )
            normalized_field_map: dict[int, tuple[str, ...]] = {}
            for z, fields_for_z in self.per_regime_field_map.items():
                fields_for_z = tuple(str(f).strip().lower() for f in fields_for_z if str(f).strip())
                if "z500" not in fields_for_z:
                    raise ValueError(f"per_regime_field_map[{z}] must include 'z500', got {fields_for_z}")
                normalized_field_map[int(z)] = fields_for_z
            object.__setattr__(self, "per_regime_field_map", normalized_field_map)

            k_spec: dict[int, int | str] = dict(self.per_regime_k_spec or {})
            for z in expected_ids:
                k_spec.setdefault(z, "auto")
            object.__setattr__(self, "per_regime_k_spec", k_spec)

            if not self.final_k_range:
                raise ValueError("final_k_range must not be empty")

            # residual_fields is overridden to the union of all per-regime selections so the
            # existing (unmodified) anomaly-check and per-field-PCA loops in auxiliary.py
            # compute each field's PCA exactly once, shared across all Z500 regimes that use
            # it — see per_regime_mode note on `final_k` above for the analogous final_k caveat.
            union_fields = tuple(sorted({f for fs in normalized_field_map.values() for f in fs}))
            object.__setattr__(self, "residual_fields", union_fields)

        fields = tuple(str(v).strip().lower() for v in self.residual_fields if str(v).strip())
        if not fields:
            raise ValueError("residual_fields must not be empty")
        if "z500" not in fields:
            raise ValueError("residual_fields must include 'z500' for Stage 1 circulation regimes")
        object.__setattr__(self, "residual_fields", fields)
        object.__setattr__(self, "bbox", _parse_bbox_string(self.bbox))

        if self.bbox_config is not None:
            object.__setattr__(self, "bbox_config", Path(self.bbox_config).expanduser().resolve())
        if self.z500_regimes_override is not None:
            object.__setattr__(
                self, "z500_regimes_override", Path(self.z500_regimes_override).expanduser().resolve()
            )
        if self.z500_raw_nc is not None:
            object.__setattr__(self, "z500_raw_nc", Path(self.z500_raw_nc).expanduser().resolve())
        if self.vo850_raw_nc is not None:
            object.__setattr__(self, "vo850_raw_nc", Path(self.vo850_raw_nc).expanduser().resolve())
        if self.tcwv_raw_nc is not None:
            object.__setattr__(self, "tcwv_raw_nc", Path(self.tcwv_raw_nc).expanduser().resolve())
        if self.deg0l_raw_nc is not None:
            object.__setattr__(self, "deg0l_raw_nc", Path(self.deg0l_raw_nc).expanduser().resolve())
        if self.cape_raw_nc is not None:
            object.__setattr__(self, "cape_raw_nc", Path(self.cape_raw_nc).expanduser().resolve())
        if self.w700_raw_nc is not None:
            object.__setattr__(self, "w700_raw_nc", Path(self.w700_raw_nc).expanduser().resolve())
        if self.rh700_raw_nc is not None:
            object.__setattr__(self, "rh700_raw_nc", Path(self.rh700_raw_nc).expanduser().resolve())

    @property
    def regime_set_label(self) -> str:
        return f"Residual regime set (Z500_K={self.z500_k}, fields={'+'.join(f.upper() for f in self.residual_fields)})"

    @property
    def regime_set_id(self) -> str:
        return f"rr_z500k{self.z500_k}_{'-'.join(self.residual_fields)}"

    @property
    def bbox_config_path(self) -> Path:
        if self.bbox_config is not None:
            return self.bbox_config
        cfg_name = (
            "bbox_steps_northamerica.yaml" if self.region == "northamerica" else "bbox_steps_europe.yaml"
        )
        return (self.module_root / "src" / cfg_name).resolve()

    @property
    def bbox_id(self) -> str:
        return self.bbox_step if not self.bbox else "custom"

    @property
    def z500_reuse_enabled(self) -> bool:
        return True


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Residual regimes two-stage clustering pipeline. "
            "Stage 1: Cluster Z500 to define circulation regimes. "
            "Stage 2: Cluster residual anomalies (within-regime variability) to find sub-regimes. "
            "This isolates the moisture/instability patterns that vary within the same Z500 regime."
        )
    )
    p.add_argument("--region", default="europe", choices=("europe", "northamerica"))
    p.add_argument("--bbox-step", default="B5", help="Bbox step key, e.g. B0..B5.")
    p.add_argument("--bbox-config", type=Path, default=None)
    p.add_argument("--bbox", default=None, help='Optional explicit bbox override "N,W,S,E".')
    p.add_argument("--raw-years-span", default="1979-2025")
    p.add_argument("--year-span", default="1979-2025")
    p.add_argument("--smooth-window", type=int, default=11)
    p.add_argument("--detrend-method", choices=("gridpoint", "domain_mean"), default="domain_mean")
    p.add_argument("--climatology-subtract", choices=("raw", "smooth"), default="smooth")
    p.add_argument("--sigma-mode", choices=("field", "scalar"), default="field")
    p.add_argument("--yeo-johnson", action=argparse.BooleanOptionalAction, default=True,
                   help="Match rr_00's 07_MF_PCA files, which are always produced with "
                        "Yeo-Johnson preprocessing (YEO_JOHNSON=1 in run_configs.yaml). "
                        "Must match rr_00's setting or Stage 2's MF-PCA lookup will report "
                        "the file as missing. Use --no-yeo-johnson only if rr_00 ran without it.")
    p.add_argument("--var-threshold", type=float, default=0.90)
    p.add_argument("--max-modes-per-field", type=int, default=100,
                   help="Cap each field's retained PCA modes at this many (leading modes, "
                        "already ordered by descending explained variance), even if "
                        "--var-threshold would otherwise retain more. Pass <=0 to disable "
                        "the cap. Must match rr_00's MultivariateRunConfig setting for "
                        "resolve_mf_pca_path lookups to hit the same files.")
    p.add_argument("--z500-k", type=int, default=5, help="Number of Stage 1 Z500 circulation regimes. "
                   "When --z500-regimes-path is set, must match that file's actual regime count.")
    p.add_argument("--z500-cluster-method", choices=("kmeans", "ward"), default="kmeans",
                   help="On-disk Stage 1 Z500 regimes filename family to auto-discover when "
                        "--z500-regimes-path is not set: 'kmeans' (default; also covers the "
                        "Ward-seeded K-means hybrid) or 'ward' (pure Ward HRC, z_03's "
                        "CLUSTER_MODE=ward). Ignored when --z500-regimes-path is set.")
    p.add_argument("--z500-regimes-path", type=Path, default=None,
                   help="Explicit Stage 1 Z500 regimes NetCDF, bypassing the usual bbox_step-based "
                        "legacy-file search — for cross-bbox diagnostics (e.g. classifying THIS "
                        "bbox's residual fields using ANOTHER bbox's Z500 circulation regimes). "
                        "Mirrors rr_01's --z500-regimes-path.")
    p.add_argument("--final-k", type=int, default=7, help="Number of Stage 2 k-means clusters (global mode; "
                   "path-naming placeholder only when --per-regime-mode is set).")
    p.add_argument("--per-regime-mode", action="store_true",
                   help="Cluster Stage-2 residuals independently per Z500 regime, each with its own field "
                        "subset (--regime-fields-map) and own k (--regime-k-spec, default auto).")
    p.add_argument("--regime-fields-map", default=None,
                   help='Per-Z500-regime residual field selection: inline JSON (e.g. '
                        '\'{"1":"z500+cape+w700","2":"z500+vo850"}\') or a path to a .json/.yaml file '
                        "with the same structure. Required when --per-regime-mode is set.")
    p.add_argument("--regime-k-spec", default=None,
                   help='Per-Z500-regime subregime count: inline JSON (e.g. \'{"1":3,"2":"auto"}\') or a '
                        "file path. Regimes omitted default to \"auto\" (silhouette sweep).")
    p.add_argument("--residualize", action="store_true",
                   help="Stage 2 [per-regime only]: subtract each Z500 regime's own day-subset "
                        "mean from its joint features before clustering -- "
                        "residual[t] = anomaly[t] - mean(anomaly | regime[t]), Option 2 of the "
                        "three Stage-2 conditioning modes in README_ResidualRegimes.md. Requires "
                        "--per-regime-mode. Default off preserves the pre-Jul-2026 behavior.")
    p.add_argument("--final-k-range", default="3,4,5,6,7,8",
                   help="Comma-separated candidate k values for per-regime \"auto\" silhouette sweep.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--stop-after-step", type=int, choices=(6, 7, 8, 9, 10), default=10)
    p.add_argument("--z500-epsilon", type=float, default=20.0)
    p.add_argument("--vo850-epsilon", type=float, default=1.0e-6)
    p.add_argument("--tcwv-epsilon", type=float, default=1.0e-3)
    p.add_argument("--deg0l-epsilon", type=float, default=50.0)
    p.add_argument("--cape-epsilon", type=float, default=1.0)
    p.add_argument("--w700-epsilon", type=float, default=0.01)
    p.add_argument("--rh700-epsilon", type=float, default=0.1)
    p.add_argument("--vo850-smoothing-method", choices=("none", "gaussian", "coarsen"), default="none")
    p.add_argument("--vo850-gaussian-sigma", type=float, default=1.0)
    p.add_argument("--vo850-coarsen-factor", type=int, default=0)
    p.add_argument(
        "--residual-fields",
        default="z500+vo850+tcwv+deg0l+cape+w700+rh700",
        help="Plus-separated or comma-separated field keys for residual clustering. Must include z500.",
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute outputs even when targets already exist.")
    p.add_argument("--dry-run", action="store_true", help="Check inputs and legacy files without running any computation.")
    p.add_argument(
        "--area-weight-tag",
        default="",
        help="Must match the rr_00 run's --area-weight-tag to require that tag when resolving "
        "the 07_MF_PCA files this pipeline reuses. Default '' accepts untagged (pre-fix) files.",
    )
    p.add_argument("--z500-raw-nc", type=Path, default=None)
    p.add_argument("--vo850-raw-nc", type=Path, default=None)
    p.add_argument("--tcwv-raw-nc", type=Path, default=None)
    p.add_argument("--deg0l-raw-nc", type=Path, default=None)
    p.add_argument("--cape-raw-nc", type=Path, default=None)
    p.add_argument("--w700-raw-nc", type=Path, default=None)
    p.add_argument("--rh700-raw-nc", type=Path, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> ResidualRunConfig:
    fields = _parse_plus_separated_fields(str(args.residual_fields))
    regime_fields_map = (
        _parse_regime_fields_map(str(args.regime_fields_map))
        if getattr(args, "regime_fields_map", None)
        else None
    )
    regime_k_spec = (
        _parse_regime_k_spec(str(args.regime_k_spec))
        if getattr(args, "regime_k_spec", None)
        else None
    )
    final_k_range = _parse_final_k_range(str(args.final_k_range))
    return ResidualRunConfig(
        region=str(args.region),
        bbox_step=str(args.bbox_step),
        bbox_config=args.bbox_config,
        bbox=args.bbox,
        raw_years_span=str(args.raw_years_span),
        year_span=str(args.year_span),
        smooth_window=int(args.smooth_window),
        detrend_method=str(args.detrend_method),
        climatology_subtract=str(args.climatology_subtract),
        sigma_mode=str(args.sigma_mode),
        yeo_johnson=bool(args.yeo_johnson),
        var_threshold=float(args.var_threshold),
        max_modes_per_field=(
            int(args.max_modes_per_field)
            if getattr(args, "max_modes_per_field", 100) is not None
            and int(getattr(args, "max_modes_per_field", 100)) > 0
            else None
        ),
        z500_k=int(args.z500_k),
        z500_cluster_method=str(args.z500_cluster_method),
        z500_regimes_override=args.z500_regimes_path,
        final_k=int(args.final_k),
        per_regime_mode=bool(args.per_regime_mode),
        per_regime_field_map=regime_fields_map,
        per_regime_k_spec=regime_k_spec,
        residualize=bool(getattr(args, "residualize", False)),
        final_k_range=final_k_range,
        random_state=int(args.random_state),
        stop_after_step=int(args.stop_after_step),
        z500_epsilon=float(args.z500_epsilon),
        vo850_epsilon=float(args.vo850_epsilon),
        tcwv_epsilon=float(args.tcwv_epsilon),
        deg0l_epsilon=float(args.deg0l_epsilon),
        cape_epsilon=float(args.cape_epsilon),
        w700_epsilon=float(args.w700_epsilon),
        rh700_epsilon=float(args.rh700_epsilon),
        vo850_smoothing_method=str(args.vo850_smoothing_method),
        vo850_gaussian_sigma=float(args.vo850_gaussian_sigma),
        vo850_coarsen_factor=int(args.vo850_coarsen_factor),
        residual_fields=fields,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        area_weight_tag=str(getattr(args, "area_weight_tag", "") or ""),
        z500_raw_nc=args.z500_raw_nc,
        vo850_raw_nc=args.vo850_raw_nc,
        tcwv_raw_nc=args.tcwv_raw_nc,
        deg0l_raw_nc=args.deg0l_raw_nc,
        cape_raw_nc=args.cape_raw_nc,
        w700_raw_nc=args.w700_raw_nc,
        rh700_raw_nc=args.rh700_raw_nc,
    )
