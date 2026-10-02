#!/usr/bin/env python3
"""Stage 2 step 10 — export sub-regime persistence to CSV.

Persistence and transition statistics for the sub-regimes.

Run ``python steps/stage2/10_export_subregime_persistence.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.export_subregime_persistence_csv import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
