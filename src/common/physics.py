from __future__ import annotations

import numpy as np
import xarray as xr

G0 = 9.80665


def is_geopotential_units(units: str) -> bool:
    u = (units or "").lower()
    return ("m**2" in u) or ("m2" in u) or ("m^2" in u) or ("s**-2" in u) or ("s^-2" in u)


def z500_anomaly_units_look_physical(da: xr.DataArray, units: str) -> bool:
    """``is_geopotential_units(units)``, plus a magnitude sanity check for Z500 regime-composite
    anomalies specifically: a real composite is never sub-gpm (empirically tens of gpm -- a
    coherent circulation regime's day-averaged anomaly). Guards against a normalized/
    Yeo-Johnson-transformed anomaly (~O(1), see ``preprocessing_helpers.apply_yeo_johnson``)
    that kept a stale physical ``units`` attr from before the transform -- without this check,
    such an array gets silently divided by g0 a second time and mislabeled "(gpm)" at roughly
    1/500th its already-wrong scale (confirmed in production: a ~0.6 composite ended up
    displayed as ~0.06 "gpm").
    """
    if not is_geopotential_units(units):
        return False
    try:
        peak = float(np.nanmax(np.abs(np.asarray(da.values, dtype=float))))
    except (TypeError, ValueError):
        return True
    return np.isfinite(peak) and (peak / G0) >= 2.0


def maybe_geopotential_to_gpm(da: xr.DataArray) -> tuple[xr.DataArray, str]:
    """Divide by g0 and return (converted_da, "gpm") when ``da`` is still raw geopotential.

    Stamps ``units="gpm"`` on the returned array (xarray's ``keep_attrs`` otherwise carries the
    pre-division "m**2 s**-2" forward unchanged) so a caller that re-checks
    ``is_geopotential_units`` on the result won't divide by g0 a second time.
    """
    u = str(da.attrs.get("units", "") or "").lower()
    ln = str(da.attrs.get("long_name", "") or "").lower()
    if "m**2" in u or "m^2" in u or "m2 s-2" in u or "geopotential" in ln:
        out = da / G0
        out.attrs["units"] = "gpm"
        return out, "gpm"
    if u:
        return da, u
    return da, ""
