"""Mutable figure index for PDF / notebook titles (``Figure N — …``)."""

from __future__ import annotations


def bump_figure_number(n_ref: list[int] | None) -> int:
    if n_ref is None:
        return 0
    n_ref[0] += 1
    return n_ref[0]
