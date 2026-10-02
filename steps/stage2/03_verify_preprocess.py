#!/usr/bin/env python3
"""Stage 2 step 3 — verify the preprocessing outputs.

Consistency checks on the preprocessed field files before the Stage-2 fit.

Run ``python steps/stage2/03_verify_preprocess.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.verify_preprocess import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
