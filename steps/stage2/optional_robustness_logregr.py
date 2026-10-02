#!/usr/bin/env python3
"""Stage 2 (optional) — logistic-fit robustness diagnostic.

Coefficient stability and regularisation sweep for the step-6 model. Not required
to reproduce the main results.

Run ``python steps/stage2/optional_robustness_logregr.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.robustness_logregr import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
