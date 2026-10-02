#!/usr/bin/env python3
"""Stage 1 step 3 — Z500 anomaly composites per regime.

Writes the per-regime Z500 anomaly composite NetCDFs that the statistics export
and the Stage-2 composites build on.

Run ``python steps/stage1/03_anom_composites.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage1_z500.anom_composites import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
