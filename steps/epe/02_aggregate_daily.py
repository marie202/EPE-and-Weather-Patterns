#!/usr/bin/env python3
"""EPE step 2 — aggregate hourly ERA5 to local analysis days.

Optional daily-aggregate route, used by the appendix comparison rather than by the
production label chain (which goes straight from step 1 to step 3).

Run ``python steps/epe/02_aggregate_daily.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from epe_preprocessing.aggregate_daily import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
