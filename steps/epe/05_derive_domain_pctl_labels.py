#!/usr/bin/env python3
"""EPE step 5 — derive domain-percentile label sets.

Turns a detection NetCDF into the labelled extreme-day sets used downstream.

Run ``python steps/epe/05_derive_domain_pctl_labels.py --help`` for the full argument list.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from epe_preprocessing.derive_domain_pctl_labels import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
