#!/usr/bin/env python3
"""Stage 1 step 4 — export Z500 regime statistics to CSV.

Regime occurrence, persistence and EPE-conditional statistics as CSV tables.

Run ``python steps/stage1/04_export_statistics_csv.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage1_z500.export_statistics_csv import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
