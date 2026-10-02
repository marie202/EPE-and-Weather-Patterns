#!/usr/bin/env python3
"""Stage 2 step 4 — per-regime field relevance scoring.

Scores each additional field's ability to separate extreme from non-extreme days
within each Z500 regime, as cross-validated AUC over PCA blocks.

Run ``python steps/stage2/04_feature_relevance.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.feature_relevance import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
