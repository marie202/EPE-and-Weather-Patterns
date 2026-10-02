#!/usr/bin/env python3
"""Stage 2 step 9 — export sub-regime statistics to CSV.

Sub-regime occurrence, enrichment and recall statistics as CSV tables.

Run ``python steps/stage2/09_export_statistics_csv.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.export_statistics_csv import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
