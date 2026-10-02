#!/usr/bin/env python3
"""Stage 2 step 1 — download the additional ERA5 fields.

Downloads z500, vo850, tcwv and deg0l for one region, one field per invocation.

Run ``python steps/stage2/01_download_fields.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from stage2_residual.fields.download import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
