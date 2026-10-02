#!/usr/bin/env python3
"""Stage 2 step 8 — sub-regime composites.

Composite fields per sub-regime, using the bin edges from step 7.

Run ``python steps/stage2/08_composites.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.plot_composites import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
