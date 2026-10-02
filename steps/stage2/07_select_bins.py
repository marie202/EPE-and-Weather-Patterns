#!/usr/bin/env python3
"""Stage 2 step 7 — sub-regime bin edges.

Splits each regime's days into sub-regimes by predicted probability. Production
setting is ``--max-bins 2 --target-day-frac 0.3``.

Run ``python steps/stage2/07_select_bins.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.select_subregime_bins import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
