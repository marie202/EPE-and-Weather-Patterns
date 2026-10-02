#!/usr/bin/env python3
"""EPE step 1 — download hourly ERA5 precipitation and CAPE.

Fetches hourly total precipitation, convective precipitation and CAPE from the
Copernicus CDS for one region label (``germany`` or ``ontario_south``), MJJAS.

Run ``python steps/epe/01_download_era5.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from epe_preprocessing.download_era5 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
