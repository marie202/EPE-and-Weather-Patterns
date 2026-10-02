#!/usr/bin/env python3
"""EPE step 3 — convective duration and domain-percentile extreme days.

Production extreme-day detector. Writes the ``convective_duration_*.nc`` label file
that Stage 2 consumes. Pinned at ``--domain-pctl-q 0.90 --domain-agg lexmax``.

Run ``python steps/epe/03_detect_convective_duration.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from epe_preprocessing.detect_convective_duration import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
