#!/usr/bin/env python3
"""Stage 1 step 2 — Z500 regime clustering (pipeline steps 8-11).

Ward hierarchical clustering, final regime assignment, reconstruction and
characterisation. Executes the notebooks under ``src/stage1_z500/notebooks/``
headlessly via nbconvert.

Run ``python steps/stage1/02_cluster_regimes.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage1_z500.cluster_regimes import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
