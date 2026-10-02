#!/usr/bin/env python3
"""Stage 1 step 1 — Z500 preprocessing (pipeline steps 2-6).

Climatology, detrending, anomalies, standardisation and area weighting for Z500,
for one region and bounding-box step.

Run ``python steps/stage1/01_preprocess.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.preprocessing_helpers import run_pipeline_steps_2_to_6  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_pipeline_steps_2_to_6())
