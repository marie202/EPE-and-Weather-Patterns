#!/usr/bin/env python3
"""Stage 2 step 6 — per-regime logistic sub-regime model.

Fits the per-Z500-regime logistic model and writes the merged NetCDF carrying the
predicted extreme probability per day.

Run ``python steps/stage2/06_logistic_fit.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.stage2_logistic import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
