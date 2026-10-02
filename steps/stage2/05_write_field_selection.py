#!/usr/bin/env python3
"""Stage 2 step 5 — build the per-regime field-selection map.

Condenses the step-4 rankings into the regime to fields map the logistic fit reads.

Run ``python steps/stage2/05_write_field_selection.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.write_field_selection import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
