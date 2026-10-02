"""Resolve Step-12 extremes / convective_duration NetCDF by threshold options (not full paths)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from epe_preprocessing.epe_functions import parse_duration_thresholds_from_name


def q_filename_tag(q: float) -> str:
    """Encode quantile in epe_01b output stem: 0.99 -> q99, 0.999 -> q999."""
    qf = normalize_q_fraction(q)
    pct1000 = int(round(qf * 1000.0))
    if pct1000 % 10 == 0:
        return f"q{pct1000 // 10:02d}"
    return f"q{pct1000:03d}"


def quantile_sample_filename_tag(sample: str, *, wet_threshold_mm: float) -> str:
    s = str(sample).strip().lower()
    if s in {"wet_hours", "wet", "wetdays", "wet_days"}:
        return f"sampleWetmm{int(round(float(wet_threshold_mm) * 10)):02d}"
    return "sampleAll"


def cape_filename_tag(cape_threshold: float) -> str:
    """Encode CAPE threshold in epe_01b output stem: 0 -> ``cape0`` (strict CAPE>0)."""
    return f"cape{int(round(float(cape_threshold)))}"


def cp_frac_filename_tag(cp_frac_min: float) -> str:
    """Encode CP fraction filter in epe_01b output stem: 0.0 -> ``cpFracMin00``."""
    return f"cpFracMin{int(round(float(cp_frac_min) * 100)):02d}"


def region_to_study_area(region: str, *, override: str | None = None) -> str:
    """Map clustering region (europe/northamerica) to processed-data folder name."""
    if override and str(override).strip():
        return str(override).strip()
    return {"europe": "germany", "northamerica": "ontario_south"}.get(region, region)


def normalize_q_fraction(q: float) -> float:
    """Accept 0.999 or 99.9 (percent) as exceedance quantile."""
    v = float(q)
    if v > 1.0:
        return v / 100.0
    return v


def domain_pctl_filename_tag(domain_pctl_q: float) -> str:
    """Encode domain-aggregation percentile in epe_01b_*_test output stem:
    0.95 -> domainQ950 (matches ``domain_q_tag`` in epe_01b_detect_convective_hourly_duration_test.py)."""
    qf = normalize_q_fraction(domain_pctl_q)
    return f"domainQ{int(round(qf * 1000)):03d}"


@dataclass(frozen=True)
class ExtremesSpec:
    """Step-12 file selection for summary PDFs (matches epe_01b filename tokens)."""

    q: float | None = None
    quantile_sample: str | None = None
    wet_threshold_mm: float | None = None
    cape_threshold: float | None = None
    cp_frac_min: float | None = None
    year_span: str | None = None
    duration_nc: Path | None = None
    # Domain-aggregated extreme-day definition (epe_01b_detect_convective_hourly_duration_test.py
    # TEST-variant output, e.g. domain_pctl_q=0.95, domain_agg="lexmax" -> "..._lexmax_domainQ950_...").
    # When set, these two tokens narrow the match (see path_matches_extremes_spec); any of the
    # legacy q/quantile_sample/cape/cp_frac/year_span fields left unset are simply not enforced,
    # same as when domain_pctl_q/domain_agg are absent — leave those unset for "domain tokens
    # alone select the file" (pre-2026-07-31 behavior), or set them too to also pin the underlying
    # point-wise definition among several TEST_* files that share the same domain tokens.
    domain_pctl_q: float | None = None
    domain_agg: str | None = None

    def q_fraction(self) -> float | None:
        if self.q is None:
            return None
        return normalize_q_fraction(self.q)

    def summary_label(self) -> str:
        parts: list[str] = []
        if self.duration_nc is not None:
            parts.append(f"duration_nc={self.duration_nc.name}")
        if self.domain_pctl_q is not None or self.domain_agg is not None:
            dq = f"{normalize_q_fraction(self.domain_pctl_q) * 100:g}th" if self.domain_pctl_q is not None else "?"
            parts.append(f"domain_agg={self.domain_agg or '?'}, domain_pctl_q={dq} percentile")
        qf = self.q_fraction()
        if qf is not None:
            parts.append(f"q={qf * 100:g}th percentile")
        if self.quantile_sample:
            qs = str(self.quantile_sample).strip().lower()
            if qs in {"wet_hours", "wet"}:
                wt = self.wet_threshold_mm if self.wet_threshold_mm is not None else 0.1
                parts.append(f"sample=wet_hours (tp>{wt:g} mm/h)")
            else:
                parts.append("sample=all_hours")
        if self.cape_threshold is not None:
            ct = float(self.cape_threshold)
            parts.append("CAPE>0" if ct <= 0.0 else f"CAPE>{ct:g} J/kg")
        if self.cp_frac_min is not None:
            cf = float(self.cp_frac_min)
            parts.append("no CP-frac filter" if cf <= 0.0 else f"CP frac >= {cf:.0%}")
        if self.year_span:
            parts.append(f"years={self.year_span}")
        return ", ".join(parts) if parts else "(newest aligning file)"

    def is_active(self) -> bool:
        return any(
            (
                self.q is not None,
                self.quantile_sample is not None,
                self.wet_threshold_mm is not None,
                self.cape_threshold is not None,
                self.cp_frac_min is not None,
                self.year_span is not None,
                self.duration_nc is not None,
                self.domain_pctl_q is not None,
                self.domain_agg is not None,
            )
        )


def default_rr01_extremes_spec() -> ExtremesSpec:
    """Production Step-12.01b settings used by rr_01 (matches run_configs.yaml)."""
    return ExtremesSpec(
        q=0.999,
        quantile_sample="all_hours",
        wet_threshold_mm=0.1,
        cape_threshold=0.0,
        cp_frac_min=0.01,
        year_span="1979-2025",
    )


def default_domain_lexmax_extremes_spec() -> ExtremesSpec:
    """Domain-aggregated extreme-day definition (domain_pctl_q=0.90, lexmax), matching the
    epe_01b_detect_convective_hourly_duration_test.py TEST-variant output currently used as the
    default extreme-event definition for composite plotting / statistics generation."""
    return ExtremesSpec(domain_pctl_q=0.90, domain_agg="lexmax")


def bbox_summary_pdf_extremes_suffix(
    *,
    study_area: str | None = None,
    extremes_spec: ExtremesSpec | None = None,
) -> str:
    """Filename suffix for summary PDFs (EPE.4 choices; matches epe_01b tokens)."""
    tokens: list[str] = []
    if study_area and str(study_area).strip():
        tokens.append(str(study_area).strip())
    if extremes_spec is not None:
        if extremes_spec.duration_nc is not None:
            tokens.append("durationExplicit")
        if extremes_spec.domain_agg:
            tokens.append(str(extremes_spec.domain_agg).strip())
        if extremes_spec.domain_pctl_q is not None:
            tokens.append(domain_pctl_filename_tag(extremes_spec.domain_pctl_q))
        qf = extremes_spec.q_fraction()
        if qf is not None:
            tokens.append(q_filename_tag(qf))
        if extremes_spec.quantile_sample:
            wet_mm = (
                extremes_spec.wet_threshold_mm
                if extremes_spec.wet_threshold_mm is not None
                else 0.1
            )
            tokens.append(
                quantile_sample_filename_tag(
                    extremes_spec.quantile_sample,
                    wet_threshold_mm=wet_mm,
                )
            )
        if extremes_spec.cape_threshold is not None:
            tokens.append(cape_filename_tag(extremes_spec.cape_threshold))
        if extremes_spec.cp_frac_min is not None:
            tokens.append(cp_frac_filename_tag(extremes_spec.cp_frac_min))
        if extremes_spec.year_span:
            tokens.append(f"years_{extremes_spec.year_span.strip()}")
    return ("_" + "_".join(tokens)) if tokens else ""


def extremes_spec_from_args(args: argparse.Namespace) -> ExtremesSpec | None:
    duration_nc = getattr(args, "duration_nc", None)
    path = Path(duration_nc).expanduser().resolve() if duration_nc else None
    q = getattr(args, "extremes_q", None)
    sample = getattr(args, "quantile_sample", None)
    wet = getattr(args, "extremes_wet_threshold_mm", None)
    cape_thr = getattr(args, "extremes_cape_threshold", None)
    cp_frac = getattr(args, "extremes_cp_frac_min", None)
    years = getattr(args, "extremes_year_span", None)
    domain_pctl_q = getattr(args, "domain_pctl_q", None)
    domain_agg = getattr(args, "domain_agg", None)
    if (
        q is None
        and sample is None
        and wet is None
        and cape_thr is None
        and cp_frac is None
        and years is None
        and path is None
        and domain_pctl_q is None
        and domain_agg is None
    ):
        return None
    return ExtremesSpec(
        q=float(q) if q is not None else None,
        quantile_sample=str(sample).strip() if sample else None,
        wet_threshold_mm=float(wet) if wet is not None else None,
        cape_threshold=float(cape_thr) if cape_thr is not None else None,
        cp_frac_min=float(cp_frac) if cp_frac is not None else None,
        year_span=str(years).strip() if years else None,
        duration_nc=path,
        domain_pctl_q=float(domain_pctl_q) if domain_pctl_q is not None else None,
        domain_agg=str(domain_agg).strip() if domain_agg else None,
    )


def extremes_spec_argv_extensions(spec: ExtremesSpec | None) -> list[str]:
    """CLI tokens for ``plot_regime_z500_precip_composites`` / ``export_regime_z500_statistics_csv``."""
    if spec is None:
        return []
    out: list[str] = []
    if spec.q is not None:
        out.extend(["--extremes-q", str(spec.q)])
    if spec.quantile_sample:
        out.extend(["--quantile-sample", str(spec.quantile_sample)])
    if spec.wet_threshold_mm is not None:
        out.extend(["--extremes-wet-threshold-mm", str(spec.wet_threshold_mm)])
    if spec.cape_threshold is not None:
        out.extend(["--extremes-cape-threshold", str(spec.cape_threshold)])
    if spec.cp_frac_min is not None:
        out.extend(["--extremes-cp-frac-min", str(spec.cp_frac_min)])
    if spec.year_span:
        out.extend(["--extremes-year-span", str(spec.year_span)])
    if spec.duration_nc is not None:
        out.extend(["--duration-nc", str(spec.duration_nc)])
    if spec.domain_pctl_q is not None:
        out.extend(["--domain-pctl-q", str(spec.domain_pctl_q)])
    if spec.domain_agg:
        out.extend(["--domain-agg", str(spec.domain_agg)])
    return out


def add_extremes_spec_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group(
        "EPE.4 extremes (optional)",
        "Select convective_duration / extremes NetCDF by epe_01b filename tags instead of a full path.",
    )
    g.add_argument(
        "--extremes-q",
        type=float,
        default=None,
        metavar="Q",
        help="Precipitation exceedance quantile as fraction (0.999) or percent (99.9). Matches _q999_ in filenames.",
    )
    g.add_argument(
        "--quantile-sample",
        choices=("wet_hours", "all_hours"),
        default=None,
        help="Hourly sample for P_q: wet_hours -> sampleWetmm*; all_hours -> sampleAll.",
    )
    g.add_argument(
        "--extremes-wet-threshold-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Wet-hour threshold (mm/h) when --quantile-sample=wet_hours (default 0.1 -> sampleWetmm01).",
    )
    g.add_argument(
        "--extremes-cape-threshold",
        type=float,
        default=None,
        metavar="JKG",
        help=(
            "CAPE filter encoded in convective_duration filename as _cape<int>_ "
            "(0 means strictly CAPE>0, the epe_01b production default)."
        ),
    )
    g.add_argument(
        "--extremes-cp-frac-min",
        type=float,
        default=None,
        metavar="FRAC",
        help=(
            "cp/tp fraction filter encoded as cpFracMin<int> "
            "(0.0 -> cpFracMin00 = no fraction filter). "
            "NOT the same as hourly cp>0; that is not a separate token in epe_01b filenames."
        ),
    )
    g.add_argument(
        "--extremes-year-span",
        default=None,
        metavar="SPAN",
        help="Match years_* in convective_duration filename (e.g. 1979-2025).",
    )
    g.add_argument(
        "--duration-nc",
        type=Path,
        default=None,
        help="Explicit convective_duration or extremes NetCDF (overrides tag matching).",
    )
    g.add_argument(
        "--domain-pctl-q",
        type=float,
        default=None,
        metavar="Q",
        help=(
            "Domain-aggregated extreme-day percentile as fraction (0.95) or percent (95). "
            "Matches _domainQ950_ in convective_duration_TEST_* filenames "
            "(epe_01b_detect_convective_hourly_duration_test.py). Set together with --domain-agg; "
            "these two tokens narrow the match but do not replace the legacy --extremes-q/"
            "--quantile-sample/--extremes-cape-threshold/--extremes-cp-frac-min/--extremes-year-span "
            "flags — set those too if several TEST_* files share the same domain tokens but differ "
            "in their underlying point-wise definition, or leave them unset for domain tokens alone."
        ),
    )
    g.add_argument(
        "--domain-agg",
        default=None,
        metavar="AGG",
        help="Domain aggregation method (e.g. 'lexmax', 'max', 'topk_mean'); matches _<agg>_ in filenames.",
    )


def path_matches_extremes_spec(path: Path, spec: ExtremesSpec) -> bool:
    if spec.duration_nc is not None:
        try:
            return path.resolve() == spec.duration_nc.resolve()
        except OSError:
            return path == spec.duration_nc

    name = path.name

    # Domain-aggregated tokens narrow the match but do NOT short-circuit past the legacy
    # q/quantile_sample/cape/cp_frac/year_span checks below: several convective_duration_TEST_*
    # files can share the same domain_pctl_q/domain_agg tokens while differing in their underlying
    # point-wise definition (e.g. q95_sampleWetmm01 vs q999_sampleAll_cape0_cpFracMin01) — if the
    # caller ALSO set those fields (not just domain_pctl_q/domain_agg), they must match too, or
    # glob_convective_duration_nc's mtime tie-break silently picks whichever such file was written
    # most recently, which need not be the intended one. A caller wanting domain tokens alone to
    # select the file (the pre-fix behavior) simply leaves the other spec fields unset, which the
    # per-field `is not None`/truthy guards below already treat as "no constraint".
    if spec.domain_pctl_q is not None and domain_pctl_filename_tag(spec.domain_pctl_q) not in name:
        return False
    if spec.domain_agg and f"_{str(spec.domain_agg).strip()}_" not in name:
        return False

    qf = spec.q_fraction()
    if qf is not None:
        tag = q_filename_tag(qf)
        if f"_{tag}_" not in name:
            parsed = parse_duration_thresholds_from_name(path)
            pq = parsed.get("q")
            if pq is None or abs(float(pq) / 100.0 - qf) > 1e-3:
                return False

    if spec.quantile_sample is not None:
        wet_mm = spec.wet_threshold_mm if spec.wet_threshold_mm is not None else 0.1
        sample_tag = quantile_sample_filename_tag(spec.quantile_sample, wet_threshold_mm=wet_mm)
        if sample_tag not in name:
            return False

    if spec.cape_threshold is not None:
        cape_tag = cape_filename_tag(spec.cape_threshold)
        if f"_{cape_tag}_" not in name:
            return False

    if spec.cp_frac_min is not None:
        cp_tag = cp_frac_filename_tag(spec.cp_frac_min)
        if cp_tag not in name:
            return False

    if spec.year_span:
        needle = f"years_{spec.year_span.strip()}"
        if needle not in name:
            return False

    return True


def _events_dirs(
    module_root: Path,
    *,
    study_area: str | None,
    region: str,
) -> list[Path]:
    base = module_root / "data" / "processed"
    if study_area:
        d = base / str(study_area).strip() / "extreme_precip_events"
        return [d] if d.is_dir() else []
    dirs: list[Path] = []
    primary = base / region / "extreme_precip_events"
    if primary.is_dir():
        dirs.append(primary)
    if base.is_dir():
        for sub in sorted(base.iterdir()):
            ep = sub / "extreme_precip_events"
            if ep.is_dir() and ep not in dirs:
                dirs.append(ep)
    return dirs


def glob_extremes_nc(
    module_root: Path,
    region: str,
    *,
    study_area: str | None = None,
    spec: ExtremesSpec | None = None,
) -> list[Path]:
    hits: list[Path] = []
    for d in _events_dirs(module_root, study_area=study_area, region=region):
        hits.extend(d.glob("extremes_*.nc"))
    uniq: dict[Path, Path] = {p.resolve(): p for p in hits}
    paths = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    if spec is not None and spec.is_active():
        paths = [p for p in paths if path_matches_extremes_spec(p, spec)]
    return paths


def glob_convective_duration_nc(
    module_root: Path,
    region: str,
    *,
    study_area: str | None = None,
    spec: ExtremesSpec | None = None,
) -> list[Path]:
    if spec is not None and spec.duration_nc is not None:
        p = spec.duration_nc.expanduser().resolve()
        return [p] if p.is_file() else []

    hits: list[Path] = []
    for d in _events_dirs(module_root, study_area=study_area, region=region):
        hits.extend(d.glob("convective_duration_*.nc"))
    uniq: dict[Path, Path] = {p.resolve(): p for p in hits}
    paths = sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)
    if spec is not None and spec.is_active():
        matched = [p for p in paths if path_matches_extremes_spec(p, spec)]
        if matched:
            return matched
        # Explicit tags requested but none matched — return empty so caller can report clearly
        # (rather than silently falling through to "return paths" below and picking some
        # unrelated file by mtime, e.g. a plain q999 file when domain_pctl_q/domain_agg was
        # explicitly requested and no domainQ950_lexmax file exists yet).
        if (
            spec.q is not None
            or spec.quantile_sample
            or spec.cape_threshold is not None
            or spec.cp_frac_min is not None
            or spec.year_span
            or spec.domain_pctl_q is not None
            or spec.domain_agg
        ):
            return []
    return paths


def resolve_convective_duration_path(
    module_root: Path,
    region: str,
    *,
    study_area: str | None = None,
    spec: ExtremesSpec | None = None,
    legacy_path: Path | None = None,
) -> Path:
    """Resolve a single convective_duration NetCDF for rr_01 / summary reports."""
    if legacy_path is not None:
        p = legacy_path.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Legacy extremes path not found: {p}")
        print(f"[extremes_spec] Using legacy path: {p.name}")
        return p

    area = region_to_study_area(region, override=study_area)
    active_spec = spec if spec is not None else default_rr01_extremes_spec()
    paths = glob_convective_duration_nc(
        module_root,
        region,
        study_area=area,
        spec=active_spec,
    )
    if not paths:
        raise FileNotFoundError(
            f"No convective_duration file matching {active_spec.summary_label()!r} "
            f"under data/processed/{area}/extreme_precip_events/. "
            f"Run epe_01b first or pass --extreme-labels-path (legacy)."
        )
    chosen = paths[0]
    print(f"[extremes_spec] Resolved extremes ({area}): {chosen.name}")
    return chosen
