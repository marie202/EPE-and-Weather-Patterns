#!/usr/bin/env python3
"""EPE step 4 — daily-aggregate extreme detection (appendix route).

Percentile-based extreme-day flags from the daily aggregates of step 2.

Run ``python steps/epe/04_detect_extremes.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from epe_preprocessing.detect_extremes import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
