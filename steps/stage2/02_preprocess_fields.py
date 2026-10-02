#!/usr/bin/env python3
"""Stage 2 step 2 — preprocess the additional fields and run per-field PCA.

Climatology, detrending, Yeo-Johnson transform and per-field PCA for the
multivariate field set.

Run ``python steps/stage2/02_preprocess_fields.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.fields.auxiliary import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
