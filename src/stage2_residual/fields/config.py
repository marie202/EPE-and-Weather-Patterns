from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from common.preprocessing_helpers import find_module_root

from .field_specs import (
    STANDARD_MULTIVARIATE_REGIME_SET,
    STANDARD_MULTIVARIATE_REGIME_SET_ID,
    default_field_order,
    is_standard_multivariate_field_set,
)


def _parse_bbox_string(bbox: str | None) -> str | None:
    raw = (bbox or "").strip()
    return raw or None


@dataclass(frozen=True)
class MultivariateRunConfig:
    """Configuration for one multivariate run."""

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
    sigma_mode: str = "scalar"
    var_threshold: float = 0.90
    max_modes_per_field: int | None = 100  # Caps each field's retained PCA modes at this many
        # (leading modes, already ordered by descending explained variance) even if
        # var_threshold would otherwise retain more — see per_field_pca.run_per_field_pca's
        # max_modes param. Some fields (e.g. w500/w700/w850, vo850, cape) need hundreds of
        # modes to satisfy var_threshold=0.90, which is a small-n/large-p setup for the
        # downstream per-regime Stage 2 fits. None disables the cap.
    final_k: int = 5
    random_state: int = 42
    stop_after_step: int = 10
    z500_epsilon: float = 20.0
    vo850_epsilon: float = 1.0e-6
    tcwv_epsilon: float = 1.0e-3
    deg0l_epsilon: float = 50.0
    vo850_smoothing_method: str = "none"
    vo850_gaussian_sigma: float = 1.0
    vo850_coarsen_factor: int = 0
    source_fields: tuple[str, ...] = field(default_factory=default_field_order)
    overwrite: bool = False
    yeo_johnson: bool = False
    area_weight_tag: str = ""  # Appended to per_field_experiment_id() (paths.py) so a
        # rerun after a preprocessing fix lands under a distinct filename instead of
        # silently overwriting/mixing with pre-fix outputs still in production use.
        # Empty (default) reproduces today's filenames exactly.
    z500_raw_nc: Path | None = None
    vo850_raw_nc: Path | None = None
    tcwv_raw_nc: Path | None = None
    deg0l_raw_nc: Path | None = None

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
        if int(self.final_k) < 2:
            raise ValueError(f"final_k must be >= 2, got {self.final_k}")
        if int(self.stop_after_step) not in {6, 7, 8, 9, 10}:
            raise ValueError("stop_after_step must be one of {6, 7, 8, 9, 10}")
        if self.vo850_smoothing_method not in {"none", "gaussian", "coarsen"}:
            raise ValueError(
                f"Unsupported vo850_smoothing_method={self.vo850_smoothing_method!r}"
            )
        if float(self.vo850_gaussian_sigma) < 0.0:
            raise ValueError("vo850_gaussian_sigma must be >= 0")
        if int(self.vo850_coarsen_factor) < 0:
            raise ValueError("vo850_coarsen_factor must be >= 0")

        fields = tuple(str(v).strip().lower() for v in self.source_fields if str(v).strip())
        if not fields:
            raise ValueError("source_fields must not be empty")
        object.__setattr__(self, "source_fields", fields)
        object.__setattr__(self, "bbox", _parse_bbox_string(self.bbox))

        if self.bbox_config is not None:
            object.__setattr__(self, "bbox_config", Path(self.bbox_config).expanduser().resolve())
        if self.z500_raw_nc is not None:
            object.__setattr__(self, "z500_raw_nc", Path(self.z500_raw_nc).expanduser().resolve())
        if self.vo850_raw_nc is not None:
            object.__setattr__(self, "vo850_raw_nc", Path(self.vo850_raw_nc).expanduser().resolve())
        if self.tcwv_raw_nc is not None:
            object.__setattr__(self, "tcwv_raw_nc", Path(self.tcwv_raw_nc).expanduser().resolve())
        if self.deg0l_raw_nc is not None:
            object.__setattr__(self, "deg0l_raw_nc", Path(self.deg0l_raw_nc).expanduser().resolve())

    @property
    def regime_set_label(self) -> str:
        if is_standard_multivariate_field_set(self.source_fields):
            return STANDARD_MULTIVARIATE_REGIME_SET
        return f"Multivariate regime set ({'+'.join(f.upper() for f in self.source_fields)})"

    @property
    def regime_set_id(self) -> str:
        if is_standard_multivariate_field_set(self.source_fields):
            return STANDARD_MULTIVARIATE_REGIME_SET_ID
        return "-".join(self.source_fields)

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
            "Parallel multivariate regime workflow (Pipeline C). Standard input is "
            "Z500 + smoothed VO850 + TCWV with per-field preprocessing/PCA before joint clustering. "
            "Outputs are isolated from the legacy Z500-only pipeline."
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
    p.add_argument("--sigma-mode", choices=("field", "scalar"), default="scalar")
    p.add_argument("--var-threshold", type=float, default=0.90)
    p.add_argument("--max-modes-per-field", type=int, default=100,
                   help="Cap each field's retained PCA modes at this many (leading modes, "
                        "already ordered by descending explained variance), even if "
                        "--var-threshold would otherwise retain more. Pass <=0 to disable "
                        "the cap.")
    p.add_argument("--final-k", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--stop-after-step", type=int, choices=(6, 7, 8, 9, 10), default=10)
    p.add_argument("--z500-epsilon", type=float, default=20.0)
    p.add_argument("--vo850-epsilon", type=float, default=1.0e-6)
    p.add_argument("--tcwv-epsilon", type=float, default=1.0e-3)
    p.add_argument("--deg0l-epsilon", type=float, default=50.0)
    p.add_argument("--vo850-smoothing-method", choices=("none", "gaussian", "coarsen"), default="none")
    p.add_argument("--vo850-gaussian-sigma", type=float, default=1.0)
    p.add_argument("--vo850-coarsen-factor", type=int, default=0)
    p.add_argument(
        "--source-fields",
        default="z500,vo850,tcwv",
        help="Comma-separated field keys in joint-feature order. Default: z500,vo850,tcwv",
    )
    p.add_argument("--overwrite", action="store_true", help="Recompute outputs even when targets already exist.")
    p.add_argument("--yeo-johnson", action="store_true", dest="yeo_johnson", help="Apply Yeo-Johnson transform to each raw daily field before detrending.")
    p.add_argument(
        "--area-weight-tag",
        default="",
        help="Suffix appended to per-field output filenames (03/04_05_06/07). Use a distinct "
        "tag (e.g. 'aw2') to reprocess after a preprocessing fix without overwriting or "
        "mixing with existing files still relied on downstream. Default '' keeps today's names.",
    )
    p.add_argument("--z500-raw-nc", type=Path, default=None)
    p.add_argument("--vo850-raw-nc", type=Path, default=None)
    p.add_argument("--tcwv-raw-nc", type=Path, default=None)
    p.add_argument("--deg0l-raw-nc", type=Path, default=None)
    return p


def config_from_args(args: argparse.Namespace) -> MultivariateRunConfig:
    fields = tuple(tok.strip().lower() for tok in str(args.source_fields).split(",") if tok.strip())
    return MultivariateRunConfig(
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
        var_threshold=float(args.var_threshold),
        max_modes_per_field=(
            int(args.max_modes_per_field)
            if getattr(args, "max_modes_per_field", 100) is not None
            and int(getattr(args, "max_modes_per_field", 100)) > 0
            else None
        ),
        final_k=int(args.final_k),
        random_state=int(args.random_state),
        stop_after_step=int(args.stop_after_step),
        z500_epsilon=float(args.z500_epsilon),
        vo850_epsilon=float(args.vo850_epsilon),
        tcwv_epsilon=float(args.tcwv_epsilon),
        deg0l_epsilon=float(args.deg0l_epsilon),
        vo850_smoothing_method=str(args.vo850_smoothing_method),
        vo850_gaussian_sigma=float(args.vo850_gaussian_sigma),
        vo850_coarsen_factor=int(args.vo850_coarsen_factor),
        source_fields=fields,
        overwrite=bool(args.overwrite),
        yeo_johnson=bool(getattr(args, "yeo_johnson", False)),
        area_weight_tag=str(getattr(args, "area_weight_tag", "") or ""),
        z500_raw_nc=args.z500_raw_nc,
        vo850_raw_nc=args.vo850_raw_nc,
        tcwv_raw_nc=args.tcwv_raw_nc,
        deg0l_raw_nc=args.deg0l_raw_nc,
    )
