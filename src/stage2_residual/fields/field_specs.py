from __future__ import annotations

from dataclasses import dataclass


STANDARD_MULTIVARIATE_REGIME_SET = "Z500+VO850+TCWV multivariate regime set"
STANDARD_MULTIVARIATE_REGIME_SET_ID = "z500_vo850_tcwv"


@dataclass(frozen=True)
class FieldSpec:
    """Definition of one ERA5 field used by the multivariate workflow."""

    key: str
    cds_variable: str
    pressure_level_hpa: int | None
    raw_dir_name: str
    candidate_var_names: tuple[str, ...]
    output_prefix: str
    long_name: str
    cds_dataset: str = "reanalysis-era5-pressure-levels"
    convert_geopotential_to_height: bool = False
    default_epsilon: float = 0.0
    default_smoothing_method: str = "none"
    # For ocean-only fields (e.g. SST) land cells are NaN. Filling the standardised
    # anomaly with 0 over those cells lets them enter the flatten/PCA as zero-variance
    # columns: they contribute nothing to the covariance (PC time series match a masked
    # ocean EOF) while keeping the full lat×lon grid intact for EOF-map reconstruction.
    fill_nonfinite_with_zero: bool = False


ERA5_Z500 = FieldSpec(
    key="z500",
    cds_variable="geopotential",
    pressure_level_hpa=500,
    raw_dir_name="z500",
    candidate_var_names=("z", "geopotential", "Z", "z500", "gh", "hgt"),
    output_prefix="Z500",
    long_name="ERA5 geopotential height at 500 hPa",
    convert_geopotential_to_height=True,
    default_epsilon=20.0,
    default_smoothing_method="none",
)

ERA5_VO850 = FieldSpec(
    key="vo850",
    cds_variable="vorticity",
    pressure_level_hpa=850,
    raw_dir_name="vo850",
    candidate_var_names=("vo", "vorticity", "VO", "relative_vorticity"),
    output_prefix="VO850",
    long_name="ERA5 vorticity at 850 hPa",
    default_epsilon=1.0e-6,
    default_smoothing_method="none",
)

ERA5_TCWV = FieldSpec(
    key="tcwv",
    cds_variable="total_column_water_vapour",
    pressure_level_hpa=None,
    raw_dir_name="tcwv",
    candidate_var_names=(
        "tcwv",
        "total_column_water_vapour",
        "TCWV",
        "tcw",
        "total_column_water",
    ),
    output_prefix="TCWV",
    long_name="ERA5 total column water vapour",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=1.0e-3,
    default_smoothing_method="none",
)

ERA5_DEG0L = FieldSpec(
    key="deg0l",
    cds_variable="zero_degree_level",
    pressure_level_hpa=None,
    raw_dir_name="deg0l",
    candidate_var_names=("deg0l", "zero_degree_level", "zdl", "h0c"),
    output_prefix="DEG0L",
    long_name="ERA5 zero degree level (0°C isotherm height)",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=50.0,
    default_smoothing_method="none",
)

ERA5_VO500 = FieldSpec(
    key="vo500",
    cds_variable="vorticity",
    pressure_level_hpa=500,
    raw_dir_name="vo500",
    candidate_var_names=("vo", "vorticity", "VO", "relative_vorticity"),
    output_prefix="VO500",
    long_name="ERA5 vorticity at 500 hPa",
    default_epsilon=1.0e-6,
    default_smoothing_method="none",
)

ERA5_CAPE = FieldSpec(
    key="cape",
    cds_variable="convective_available_potential_energy",
    pressure_level_hpa=None,
    raw_dir_name="cape",
    candidate_var_names=("cape", "convective_available_potential_energy", "CAPE"),
    output_prefix="CAPE",
    long_name="ERA5 convective available potential energy",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=1.0,
    default_smoothing_method="none",
)

ERA5_W700 = FieldSpec(
    key="w700",
    cds_variable="vertical_velocity",
    pressure_level_hpa=700,
    raw_dir_name="w700",
    candidate_var_names=("w", "omega", "vertical_velocity", "w700"),
    output_prefix="W700",
    long_name="ERA5 vertical velocity (omega) at 700 hPa",
    default_epsilon=0.01,
    default_smoothing_method="none",
)

ERA5_RH700 = FieldSpec(
    key="rh700",
    cds_variable="relative_humidity",
    pressure_level_hpa=700,
    raw_dir_name="rh700",
    candidate_var_names=("r", "relative_humidity", "rh", "rh700"),
    output_prefix="RH700",
    long_name="ERA5 relative humidity at 700 hPa",
    default_epsilon=0.5,
    default_smoothing_method="none",
)

# ============================================================================
# Table 1: Additional candidate fields for extreme event discrimination
# ============================================================================

# Moisture flux fields
ERA5_VIMD = FieldSpec(
    key="vimd",
    cds_variable="vertical_integral_of_divergence_of_moisture_flux",
    pressure_level_hpa=None,
    raw_dir_name="vimd",
    candidate_var_names=("vimd", "vertical_integral_of_divergence_of_moisture_flux", "VIMD"),
    output_prefix="VIMD",
    long_name="ERA5 vertically integrated moisture divergence",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_VIDMF = FieldSpec(
    key="vidmf",
    cds_variable="vertical_integral_of_divergence_of_moisture_flux",
    pressure_level_hpa=None,
    raw_dir_name="vidmf",
    candidate_var_names=("vidmf", "vertical_integral_of_divergence_of_moisture_flux"),
    output_prefix="VIDMF",
    long_name="ERA5 vertical integral divergence moisture flux (post-hoc)",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_VIWVE = FieldSpec(
    key="viwve",
    cds_variable="vertical_integral_of_eastward_water_vapour_flux",
    pressure_level_hpa=None,
    raw_dir_name="viwve",
    candidate_var_names=("viwve", "vertical_integral_of_eastward_water_vapour_flux", "IVT_E"),
    output_prefix="VIWVE",
    long_name="ERA5 vertical integral eastward water vapour flux",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=100.0,
    default_smoothing_method="none",
)

ERA5_VIWVN = FieldSpec(
    key="viwvn",
    cds_variable="vertical_integral_of_northward_water_vapour_flux",
    pressure_level_hpa=None,
    raw_dir_name="viwvn",
    candidate_var_names=("viwvn", "vertical_integral_of_northward_water_vapour_flux", "IVT_N"),
    output_prefix="VIWVN",
    long_name="ERA5 vertical integral northward water vapour flux",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=100.0,
    default_smoothing_method="none",
)

# Humidity fields
ERA5_TD2M = FieldSpec(
    key="td2m",
    cds_variable="2m_dewpoint_temperature",
    pressure_level_hpa=None,
    raw_dir_name="td2m",
    candidate_var_names=("td2m", "2m_dewpoint_temperature", "d2m"),
    output_prefix="TD2M",
    long_name="ERA5 2m dewpoint temperature",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_QHUM900 = FieldSpec(
    key="qhum900",
    cds_variable="specific_humidity",
    pressure_level_hpa=900,
    raw_dir_name="qhum900",
    candidate_var_names=("qhum", "specific_humidity", "q"),
    output_prefix="QHUM900",
    long_name="ERA5 specific humidity at 900 hPa",
    default_epsilon=1.0e-5,
    default_smoothing_method="none",
)

ERA5_QHUM850 = FieldSpec(
    key="qhum850",
    cds_variable="specific_humidity",
    pressure_level_hpa=850,
    raw_dir_name="qhum850",
    candidate_var_names=("qhum", "specific_humidity", "q"),
    output_prefix="QHUM850",
    long_name="ERA5 specific humidity at 850 hPa",
    default_epsilon=1.0e-5,
    default_smoothing_method="none",
)

ERA5_QHUM700 = FieldSpec(
    key="qhum700",
    cds_variable="specific_humidity",
    pressure_level_hpa=700,
    raw_dir_name="qhum700",
    candidate_var_names=("qhum", "specific_humidity", "q"),
    output_prefix="QHUM700",
    long_name="ERA5 specific humidity at 700 hPa",
    default_epsilon=1.0e-5,
    default_smoothing_method="none",
)

ERA5_QHUM500 = FieldSpec(
    key="qhum500",
    cds_variable="specific_humidity",
    pressure_level_hpa=500,
    raw_dir_name="qhum500",
    candidate_var_names=("qhum", "specific_humidity", "q"),
    output_prefix="QHUM500",
    long_name="ERA5 specific humidity at 500 hPa",
    default_epsilon=1.0e-5,
    default_smoothing_method="none",
)

ERA5_RH850 = FieldSpec(
    key="rh850",
    cds_variable="relative_humidity",
    pressure_level_hpa=850,
    raw_dir_name="rh850",
    candidate_var_names=("r", "rh", "relative_humidity", "rh850"),
    output_prefix="RH850",
    long_name="ERA5 relative humidity at 850 hPa",
    default_epsilon=0.5,
    default_smoothing_method="none",
)

ERA5_RH600 = FieldSpec(
    key="rh600",
    cds_variable="relative_humidity",
    pressure_level_hpa=600,
    raw_dir_name="rh600",
    candidate_var_names=("r", "rh", "relative_humidity", "rh600"),
    output_prefix="RH600",
    long_name="ERA5 relative humidity at 600 hPa",
    default_epsilon=0.5,
    default_smoothing_method="none",
)

ERA5_RH500 = FieldSpec(
    key="rh500",
    cds_variable="relative_humidity",
    pressure_level_hpa=500,
    raw_dir_name="rh500",
    candidate_var_names=("r", "rh", "relative_humidity", "rh500"),
    output_prefix="RH500",
    long_name="ERA5 relative humidity at 500 hPa",
    default_epsilon=0.5,
    default_smoothing_method="none",
)

# Temperature fields
ERA5_TEMP900 = FieldSpec(
    key="temp900",
    cds_variable="temperature",
    pressure_level_hpa=900,
    raw_dir_name="temp900",
    candidate_var_names=("temp", "temperature", "t"),
    output_prefix="TEMP900",
    long_name="ERA5 temperature at 900 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_TEMP850 = FieldSpec(
    key="temp850",
    cds_variable="temperature",
    pressure_level_hpa=850,
    raw_dir_name="temp850",
    candidate_var_names=("temp", "temperature", "t"),
    output_prefix="TEMP850",
    long_name="ERA5 temperature at 850 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_TEMP700 = FieldSpec(
    key="temp700",
    cds_variable="temperature",
    pressure_level_hpa=700,
    raw_dir_name="temp700",
    candidate_var_names=("temp", "temperature", "t"),
    output_prefix="TEMP700",
    long_name="ERA5 temperature at 700 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_TEMP500 = FieldSpec(
    key="temp500",
    cds_variable="temperature",
    pressure_level_hpa=500,
    raw_dir_name="temp500",
    candidate_var_names=("temp", "temperature", "t"),
    output_prefix="TEMP500",
    long_name="ERA5 temperature at 500 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_T2M = FieldSpec(
    key="t2m",
    cds_variable="2m_temperature",
    pressure_level_hpa=None,
    raw_dir_name="t2m",
    candidate_var_names=("t2m", "2m_temperature"),
    output_prefix="T2M",
    long_name="ERA5 2m temperature",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

# Vertical velocity fields
ERA5_W850 = FieldSpec(
    key="w850",
    cds_variable="vertical_velocity",
    pressure_level_hpa=850,
    raw_dir_name="w850",
    candidate_var_names=("w", "omega", "vertical_velocity"),
    output_prefix="W850",
    long_name="ERA5 vertical velocity (omega) at 850 hPa",
    default_epsilon=0.01,
    default_smoothing_method="none",
)

ERA5_W500 = FieldSpec(
    key="w500",
    cds_variable="vertical_velocity",
    pressure_level_hpa=500,
    raw_dir_name="w500",
    candidate_var_names=("w", "omega", "vertical_velocity"),
    output_prefix="W500",
    long_name="ERA5 vertical velocity (omega) at 500 hPa",
    default_epsilon=0.01,
    default_smoothing_method="none",
)

# Wind fields — u component
ERA5_UWND925 = FieldSpec(
    key="uwnd925",
    cds_variable="u_component_of_wind",
    pressure_level_hpa=925,
    raw_dir_name="uwnd925",
    candidate_var_names=("uwnd", "u_component_of_wind", "u"),
    output_prefix="UWND925",
    long_name="ERA5 eastward wind at 925 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_UWND850 = FieldSpec(
    key="uwnd850",
    cds_variable="u_component_of_wind",
    pressure_level_hpa=850,
    raw_dir_name="uwnd850",
    candidate_var_names=("uwnd", "u_component_of_wind", "u"),
    output_prefix="UWND850",
    long_name="ERA5 eastward wind at 850 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_UWND700 = FieldSpec(
    key="uwnd700",
    cds_variable="u_component_of_wind",
    pressure_level_hpa=700,
    raw_dir_name="uwnd700",
    candidate_var_names=("uwnd", "u_component_of_wind", "u"),
    output_prefix="UWND700",
    long_name="ERA5 eastward wind at 700 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_UWND500 = FieldSpec(
    key="uwnd500",
    cds_variable="u_component_of_wind",
    pressure_level_hpa=500,
    raw_dir_name="uwnd500",
    candidate_var_names=("uwnd", "u_component_of_wind", "u"),
    output_prefix="UWND500",
    long_name="ERA5 eastward wind at 500 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_UWND250 = FieldSpec(
    key="uwnd250",
    cds_variable="u_component_of_wind",
    pressure_level_hpa=250,
    raw_dir_name="uwnd250",
    candidate_var_names=("uwnd", "u_component_of_wind", "u"),
    output_prefix="UWND250",
    long_name="ERA5 eastward wind at 250 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

# Wind fields — v component
ERA5_VWND925 = FieldSpec(
    key="vwnd925",
    cds_variable="v_component_of_wind",
    pressure_level_hpa=925,
    raw_dir_name="vwnd925",
    candidate_var_names=("vwnd", "v_component_of_wind", "v"),
    output_prefix="VWND925",
    long_name="ERA5 northward wind at 925 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_VWND850 = FieldSpec(
    key="vwnd850",
    cds_variable="v_component_of_wind",
    pressure_level_hpa=850,
    raw_dir_name="vwnd850",
    candidate_var_names=("vwnd", "v_component_of_wind", "v"),
    output_prefix="VWND850",
    long_name="ERA5 northward wind at 850 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_VWND700 = FieldSpec(
    key="vwnd700",
    cds_variable="v_component_of_wind",
    pressure_level_hpa=700,
    raw_dir_name="vwnd700",
    candidate_var_names=("vwnd", "v_component_of_wind", "v"),
    output_prefix="VWND700",
    long_name="ERA5 northward wind at 700 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_VWND500 = FieldSpec(
    key="vwnd500",
    cds_variable="v_component_of_wind",
    pressure_level_hpa=500,
    raw_dir_name="vwnd500",
    candidate_var_names=("vwnd", "v_component_of_wind", "v"),
    output_prefix="VWND500",
    long_name="ERA5 northward wind at 500 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

ERA5_VWND250 = FieldSpec(
    key="vwnd250",
    cds_variable="v_component_of_wind",
    pressure_level_hpa=250,
    raw_dir_name="vwnd250",
    candidate_var_names=("vwnd", "v_component_of_wind", "v"),
    output_prefix="VWND250",
    long_name="ERA5 northward wind at 250 hPa",
    default_epsilon=0.1,
    default_smoothing_method="none",
)

# Surface & integrated fields
ERA5_SST = FieldSpec(
    key="sst",
    cds_variable="sea_surface_temperature",
    pressure_level_hpa=None,
    raw_dir_name="sst",
    candidate_var_names=("sst", "sea_surface_temperature"),
    output_prefix="SST",
    long_name="ERA5 sea surface temperature",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=0.1,
    default_smoothing_method="none",
    fill_nonfinite_with_zero=True,  # land cells are NaN — zero-fill standardised anomaly
)

ERA5_TISR = FieldSpec(
    key="tisr",
    cds_variable="total_solar_radiation",
    pressure_level_hpa=None,
    raw_dir_name="tisr",
    candidate_var_names=("tisr", "total_solar_radiation"),
    output_prefix="TISR",
    long_name="ERA5 total incoming solar radiation",
    cds_dataset="reanalysis-era5-single-levels",
    default_epsilon=100.0,
    default_smoothing_method="none",
)

_FIELD_SPECS: dict[str, FieldSpec] = {
    # Core fields (existing)
    ERA5_Z500.key: ERA5_Z500,
    ERA5_VO850.key: ERA5_VO850,
    ERA5_TCWV.key: ERA5_TCWV,
    ERA5_DEG0L.key: ERA5_DEG0L,
    ERA5_VO500.key: ERA5_VO500,
    ERA5_CAPE.key: ERA5_CAPE,
    ERA5_W700.key: ERA5_W700,
    ERA5_RH700.key: ERA5_RH700,
    # Table 1: Moisture flux fields (4)
    ERA5_VIMD.key: ERA5_VIMD,
    ERA5_VIDMF.key: ERA5_VIDMF,
    ERA5_VIWVE.key: ERA5_VIWVE,
    ERA5_VIWVN.key: ERA5_VIWVN,
    # Table 1: Specific humidity (4)
    ERA5_QHUM900.key: ERA5_QHUM900,
    ERA5_QHUM850.key: ERA5_QHUM850,
    ERA5_QHUM700.key: ERA5_QHUM700,
    ERA5_QHUM500.key: ERA5_QHUM500,
    # Table 1: Relative humidity (3, + RH700 from core)
    ERA5_RH850.key: ERA5_RH850,
    ERA5_RH600.key: ERA5_RH600,
    ERA5_RH500.key: ERA5_RH500,
    # Table 1: Temperature (5, + existing T2M)
    ERA5_TEMP900.key: ERA5_TEMP900,
    ERA5_TEMP850.key: ERA5_TEMP850,
    ERA5_TEMP700.key: ERA5_TEMP700,
    ERA5_TEMP500.key: ERA5_TEMP500,
    ERA5_T2M.key: ERA5_T2M,
    # Table 1: Dewpoint
    ERA5_TD2M.key: ERA5_TD2M,
    # Table 1: Vertical velocity (2, + W700 from core)
    ERA5_W850.key: ERA5_W850,
    ERA5_W500.key: ERA5_W500,
    # Table 1: Wind — u component (5)
    ERA5_UWND925.key: ERA5_UWND925,
    ERA5_UWND850.key: ERA5_UWND850,
    ERA5_UWND700.key: ERA5_UWND700,
    ERA5_UWND500.key: ERA5_UWND500,
    ERA5_UWND250.key: ERA5_UWND250,
    # Table 1: Wind — v component (5)
    ERA5_VWND925.key: ERA5_VWND925,
    ERA5_VWND850.key: ERA5_VWND850,
    ERA5_VWND700.key: ERA5_VWND700,
    ERA5_VWND500.key: ERA5_VWND500,
    ERA5_VWND250.key: ERA5_VWND250,
    # Table 1: Surface & integrated fields
    ERA5_SST.key: ERA5_SST,
    ERA5_TISR.key: ERA5_TISR,
}


def get_field_spec(field_key: str) -> FieldSpec:
    key = str(field_key).strip().lower()
    if key not in _FIELD_SPECS:
        raise KeyError(f"Unknown field_key={field_key!r}. Available: {sorted(_FIELD_SPECS)}")
    return _FIELD_SPECS[key]


def default_field_order() -> tuple[str, ...]:
    """Standard Pipeline C (HRC) multivariate input: Z500 + VO850 + TCWV."""

    return (ERA5_Z500.key, ERA5_VO850.key, ERA5_TCWV.key)


def is_standard_multivariate_field_set(field_keys: tuple[str, ...]) -> bool:
    return tuple(field_keys) == default_field_order()
