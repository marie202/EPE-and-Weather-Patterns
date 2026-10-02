"""Parallel multivariate weather-regime workflow.

This package intentionally lives beside the legacy Z500-only workflow and
keeps all multifield logic under a separate namespace.
"""

from .auxiliary import main, run_pipeline
from .config import MultivariateRunConfig
from .field_specs import (
    ERA5_TCWV,
    ERA5_VO500,
    ERA5_VO850,
    ERA5_Z500,
    STANDARD_MULTIVARIATE_REGIME_SET,
    FieldSpec,
    get_field_spec,
)

__all__ = [
    "ERA5_TCWV",
    "ERA5_VO500",
    "ERA5_VO850",
    "ERA5_Z500",
    "STANDARD_MULTIVARIATE_REGIME_SET",
    "FieldSpec",
    "MultivariateRunConfig",
    "get_field_spec",
    "main",
    "run_pipeline",
]
