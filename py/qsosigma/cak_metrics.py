"""
Ca II K stellar absorption template fitting and uncertainty estimation.

Stellar template files live in ``data/stellar_templates/``. See that
directory's README.md for their origin and format. 
empirical spectra 

Dispersion metrics
------------------
``CAK_STELLAR_DISP`` (σ*) is the fitted stellar / kinematic Gaussian broadening
applied on top of the template. The forward model already includes the DESI
instrumental LSF in quadrature with σ*, so **σ* is the quantity to use** when
comparing stacks or validating injected redshift errors via
√(σ₀² + σ_zerr²).

``CAK_STELLAR_DISP_TOTAL`` is the total Gaussian kernel width applied to the
template, √(σ*² + σ_inst² + σ_template_LSF²). It is useful as a diagnostic of
the full line-spread kernel, but it is **not** the LSF-corrected stellar
dispersion and should not be used for redshift-error quadrature tests.

The continuum is initialized from sideband power-law fits and refined jointly
with (v, σ*, depth). Hard-freezing the continuum underestimates σ* on QSO host
stacks.

Point estimates use a locked template (explicit name or lowest χ²). Uncertainties
are the 16–84 percentile half-range over enabled templates that pass a failed-fit
cull (σ* near the active bounds, or depth below a floor), always retaining the
locked template. If only the locked template remains, errors are NaN. Drive Ca K
fits through ``run_cakfit.py``.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize

import qsosigma
from qsosigma import template_tools as ttools

from qsosigma.desi_resolution import (
    combine_velocity_sigmas,
    desi_spectro_instrumental_sigma_kms_from_rest,
)

repo_root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())

C_KMS = 2.99792458e5
CAK_LAB_WAVE = 3933.663
CAH_LAB_WAVE = 3968.463
MIN_PIXELS = 4
MAX_CONT_ITER = 3
DEFAULT_N_BOOT = 30
MANIFEST_FILENAME = 'templates.manifest.csv'

# Rest-frame windows (Angstrom). Continuum sidebands avoid Ca K and Ca H absorption.
LINE_HALF_EXCL = 12.0
CONT_BLUE_LO = 3888.663
CONT_BLUE_HI = 3921.663
CONT_RED_LO = 3980.463
CONT_RED_HI = 4008.663
FIT_HALF = 28.0
PLOT_WAVE_LO = 3863.663
PLOT_WAVE_HI = 4008.463
PLOT_TEMPLATE_SIG_KMS = 200.0
LWRNG_HALF = 12.0

# MILES library spectral resolution (FWHM in Angstrom); see stellar_templates/README.md.
MILES_FWHM_A = 2.5
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Gaussian core weighting in velocity space; sigma scales with the current sigv.
CORE_WEIGHT_SIG_FLOOR_KMS = 20.0
CORE_WEIGHT_FLOOR = 0.2

# Free-parameter bounds for stellar dispersion (km/s).
SIGV_MIN = 20.0
SIGV_MAX = 750.0
SIGV_BOUND_TOL = 5.0
# Ensemble members with depth at or below this are treated as failed fits.
DEPTH_FAIL_MAX = 0.02

# Peak search window used when aligning templates to CaK_LAB_WAVE.
CAK_ALIGN_HALF_A = 20.0

# Multi-start grid for sigv (km/s) to avoid shallow local minima after template alignment.
SIGV_START_GRID = (80.0, 120.0, 180.0, 250.0, 350.0, 500.0)

CAK_METRIC_SUFFIXES = ('CENTROID', 'STELLAR_DISP', 'STELLAR_DISP_TOTAL', 'DEPTH')
CAK_PREFIX = 'CAK'


@dataclass(frozen=True)
class CaKTemplate:
    """One Ca K absorption template CSV (see stellar_templates/README.md)."""

    name: str
    filename: str
    label: str
    spectral_type: str = ''
    fe_h: str = ''
    source: str = ''
    enabled: bool = True


def _parse_manifest_bool(value) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y')


def _template_from_manifest_row(row) -> CaKTemplate:
    filename = str(row['filename']).strip()
    return CaKTemplate(
        name=str(row['name']).strip(),
        filename=filename.replace('\\', '/'),
        label=str(row.get('label', row['name'])).strip(),
        spectral_type=str(row.get('spectral_type', '') or '').strip(),
        fe_h=str(row.get('fe_h', '') or '').strip(),
        source=str(row.get('source', '') or '').strip(),
        enabled=_parse_manifest_bool(row.get('enabled')),
    )


def load_cak_template_catalog(
    template_dir=None,
    enabled_only=True,
):
    """
    Load Ca K templates from templates.manifest.csv.

    If ``enabled_only`` is True (default), only rows with enabled=true are
    returned. 
    """
    template_dir = template_dir or default_stellar_template_dir()
    manifest_path = os.path.join(template_dir, MANIFEST_FILENAME)
    templates: List[CaKTemplate] = []

    if os.path.isfile(manifest_path):
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            filename = str(row.get('filename', '') or '').strip()
            if not filename:
                continue
            template = _template_from_manifest_row(row)
            if enabled_only and not template.enabled:
                continue
            templates.append(template)

    if templates:
        return templates

    if not enabled_only:
        return templates

    return templates


def find_cak_template_by_name(template_dir, name):
    """
    Find a manifest template by name, including disabled rows.

    Returns (CaKTemplate, None) on success, or (None, error_message) on failure.
    """
    template_dir = template_dir or default_stellar_template_dir()
    name = str(name).strip()
    if not name:
        return None, 'Ca K template name is empty.'

    all_templates = load_cak_template_catalog(
        template_dir, enabled_only=False,
    )
    matches = [t for t in all_templates if t.name == name]
    if not matches:
        available = ', '.join(t.name for t in all_templates) or '(none)'
        return None, (
            'Ca K template %r not found in %s. Available: %s'
            % (name, os.path.join(template_dir, MANIFEST_FILENAME), available)
        )
    return matches[0], None


def _n_pixels(lbd, lo, hi):
    return int(np.sum((lbd >= lo) & (lbd <= hi)))


def build_cak_windows(
    lbd_min,
    lbd_max,
    line_half_excl=None,
    fit_half=None,
):
    """Build fit, metric, and continuum windows for Ca K."""
    line_half = LINE_HALF_EXCL if line_half_excl is None else line_half_excl
    fit_half = FIT_HALF if fit_half is None else fit_half

    fit_lo = max(lbd_min, CAK_LAB_WAVE - fit_half)
    fit_hi = min(lbd_max, CAK_LAB_WAVE + fit_half)

    blue_lo = max(lbd_min, CONT_BLUE_LO)
    blue_hi = max(lbd_min, min(CONT_BLUE_HI, CAK_LAB_WAVE - line_half))
    red_lo = min(lbd_max, max(CONT_RED_LO, CAK_LAB_WAVE + line_half))
    red_hi = min(lbd_max, CONT_RED_HI)

    lwrng = [
        max(lbd_min, CAK_LAB_WAVE - LWRNG_HALF),
        min(lbd_max, CAK_LAB_WAVE + LWRNG_HALF),
    ]
    rngline = [fit_lo, fit_hi]
    cntrng = np.array([blue_lo, blue_hi, red_lo, red_hi], dtype=float)
    return rngline, lwrng, cntrng


def build_cak_plot_window(lbd_min, lbd_max):
    """Return the wide wavelength window used for Ca K diagnostic plots."""
    return max(lbd_min, PLOT_WAVE_LO), min(lbd_max, PLOT_WAVE_HI)


def _cak_core_weights(
    lbd,
    rest_wave=CAK_LAB_WAVE,
    sigma_kms=CORE_WEIGHT_SIG_FLOOR_KMS,
    floor=CORE_WEIGHT_FLOOR,
):
    """Return chi2 weights that emphasize pixels near the Ca K line core."""
    dv = C_KMS * (np.asarray(lbd, dtype=float) / rest_wave - 1.0)
    sigma = max(float(sigma_kms), CORE_WEIGHT_SIG_FLOOR_KMS)
    core = np.exp(-0.5 * (dv / sigma) ** 2)
    return floor + (1.0 - floor) * core


def cak_is_measurable(spres, min_pixels=MIN_PIXELS):
    """Return True if Ca K can be fit in the spectrum."""
    lbd = np.asarray(spres['lbd'], dtype=float)
    if lbd.size == 0:
        return False
    lbd_min, lbd_max = float(np.min(lbd)), float(np.max(lbd))
    if CAK_LAB_WAVE < lbd_min or CAK_LAB_WAVE > lbd_max:
        return False
    rngline, lwrng, cntrng = build_cak_windows(lbd_min, lbd_max)
    if rngline[0] >= rngline[1] or _n_pixels(lbd, rngline[0], rngline[1]) < min_pixels:
        return False
    if lwrng[0] >= lwrng[1] or _n_pixels(lbd, lwrng[0], lwrng[1]) < min_pixels:
        return False
    blue_ok = (
        cntrng[0] < cntrng[1]
        and _n_pixels(lbd, cntrng[0], cntrng[1]) >= min_pixels
    )
    red_ok = (
        cntrng[2] < cntrng[3]
        and _n_pixels(lbd, cntrng[2], cntrng[3]) >= min_pixels
    )
    return blue_ok or red_ok


def default_stellar_template_dir():
    return os.path.join(
        repo_root, 'data', 'stellar_templates',
    )


def _sanitize_template_arrays(lbdtpl, ttpl):
    """Drop non-finite wavelengths and replace non-finite absorption with 0."""
    lbdtpl = np.asarray(lbdtpl, dtype=float)
    ttpl = np.asarray(ttpl, dtype=float)
    finite_wave = np.isfinite(lbdtpl)
    lbdtpl = lbdtpl[finite_wave]
    ttpl = ttpl[finite_wave]
    ttpl = np.where(np.isfinite(ttpl), ttpl, 0.0)
    return lbdtpl, ttpl


def _cak_peak_wave(lbdtpl, ttpl, lab_wave=CAK_LAB_WAVE, half_window=CAK_ALIGN_HALF_A):
    """Return the wavelength of the peak Ca K absorption near lab_wave."""
    lbdtpl = np.asarray(lbdtpl, dtype=float)
    ttpl = np.asarray(ttpl, dtype=float)
    mask = (lbdtpl >= lab_wave - half_window) & (lbdtpl <= lab_wave + half_window)
    if not np.any(mask) or float(np.max(ttpl[mask])) <= 0.0:
        return np.nan
    return float(lbdtpl[mask][np.argmax(ttpl[mask])])


def _align_template_cak_peak(lbdtpl, ttpl, lab_wave=CAK_LAB_WAVE, half_window=CAK_ALIGN_HALF_A):
    """
    Multiplicatively shift the template so the Ca K absorption peak is at lab_wave.

    Returns (lbdtpl_shifted, peak_wave_before, shift_factor).
    """
    peak_wave = _cak_peak_wave(lbdtpl, ttpl, lab_wave=lab_wave, half_window=half_window)
    if not np.isfinite(peak_wave) or peak_wave <= 0.0:
        return np.asarray(lbdtpl, dtype=float), np.nan, 1.0
    shift_factor = float(lab_wave / peak_wave)
    return np.asarray(lbdtpl, dtype=float) * shift_factor, peak_wave, shift_factor


def cak_peak_velocity_offset_kms(lbdtpl, ttpl, lab_wave=CAK_LAB_WAVE):
    """
    Velocity that recenters the template Ca K peak on lab_wave via shift_template.

    Positive values move template features to longer wavelength.
    """
    peak_wave = _cak_peak_wave(lbdtpl, ttpl, lab_wave=lab_wave)
    if not np.isfinite(peak_wave) or peak_wave <= 0.0:
        return 0.0
    return float(C_KMS * np.log(lab_wave / peak_wave))


def _resample_template_log(lbdtpl, ttpl):
    """Resample a template onto a uniform ln(wavelength) grid for velocity convolution."""
    lbdtpl = np.asarray(lbdtpl, dtype=float)
    ttpl = np.asarray(ttpl, dtype=float)
    if lbdtpl.size < 2:
        return lbdtpl, ttpl
    lnlbd = np.log(lbdtpl)
    dln = float(np.median(np.diff(lnlbd)))
    if not np.isfinite(dln) or dln <= 0.0:
        return lbdtpl, ttpl
    n = int(np.floor((lnlbd[-1] - lnlbd[0]) / dln)) + 1
    if n < 2:
        return lbdtpl, ttpl
    lnlbd_u = lnlbd[0] + dln * np.arange(n)
    lbdtpl_u = np.exp(lnlbd_u)
    ttpl_u = np.interp(lbdtpl_u, lbdtpl, ttpl, left=0.0, right=0.0)
    return lbdtpl_u, ttpl_u


def load_cak_template(template_dir, template: CaKTemplate, align_peak=True):
    """
    Load a Ca K template CSV, sanitize NaNs, optionally align the Ca K peak to
    ``CAK_LAB_WAVE``, and resample onto a uniform ln(wavelength) grid.

    Peak alignment is a multiplicative wavelength shift so the absorption peak
    sits at 3933.663 Å before fitting. The template is then resampled in
    ``ln(λ)`` for velocity-space convolution with ``broaden_template``.
    """
    path = os.path.join(template_dir, template.filename)
    if not os.path.isfile(path):
        raise FileNotFoundError('Ca K template not found: %s' % path)
    df = pd.read_csv(path)
    lbdtpl = np.asarray(df['wavelength'], dtype=float)
    ttpl = np.asarray(df['absorption'], dtype=float)
    order = np.argsort(lbdtpl)
    lbdtpl, ttpl = lbdtpl[order], ttpl[order]
    lbdtpl, ttpl = _sanitize_template_arrays(lbdtpl, ttpl)
    if align_peak:
        lbdtpl, _peak_wave, _shift = _align_template_cak_peak(lbdtpl, ttpl)
    lbdtpl, ttpl = _resample_template_log(lbdtpl, ttpl)
    return lbdtpl, ttpl


def _template_pix_kms(lbdtpl):
    dln = abs(np.log(lbdtpl[1]) - np.log(lbdtpl[0]))
    return dln * C_KMS


def fwhm_angstrom_to_sigma_kms(fwhm_angstrom, wavelength_angstrom):
    """Convert a Gaussian FWHM in Angstrom to sigma in km/s at rest wavelength."""
    if fwhm_angstrom is None or fwhm_angstrom <= 0:
        return 0.0
    sigma_angstrom = float(fwhm_angstrom) / FWHM_TO_SIGMA
    return float(C_KMS * sigma_angstrom / float(wavelength_angstrom))


def infer_template_fwhm_angstrom(template: Optional[CaKTemplate]) -> float:
    """Return the template LSF FWHM (Angstrom), or 0 when baked into the profile."""
    if template is None:
        return 0.0
    source = (template.source or '').upper()
    if 'MILES' in source:
        return MILES_FWHM_A
    if 'UVES' in source:
        return 0.0
    return 0.0


def template_lsf_sigma_kms(
    wavelength_angstrom,
    template: Optional[CaKTemplate] = None,
    fwhm_angstrom=None,
):
    """Gaussian sigma (km/s) of the stellar template's native spectral resolution."""
    if fwhm_angstrom is None:
        fwhm_angstrom = infer_template_fwhm_angstrom(template)
    return fwhm_angstrom_to_sigma_kms(fwhm_angstrom, wavelength_angstrom)


def stellar_disp_total_kms(
    sig_fit_kms,
    sig_inst_kms=0.0,
    sig_template_lsf_kms=0.0,
):
    """
    Total Gaussian kernel width applied to the template (km/s).

    This is √(σ*² + σ_inst² + σ_template_LSF²). Prefer ``CAK_STELLAR_DISP`` (σ*)
    for stellar / redshift-error science; see module docstring.
    """
    terms = [
        float(sig_fit_kms) ** 2,
        float(sig_inst_kms) ** 2,
        float(sig_template_lsf_kms) ** 2,
    ]
    return float(np.sqrt(np.sum(terms)))


def _propagate_total_disp_err(sig_fit_kms, err_fit_kms, sig_inst_kms, sig_template_lsf_kms):
    total = stellar_disp_total_kms(sig_fit_kms, sig_inst_kms, sig_template_lsf_kms)
    if total <= 0 or not np.isfinite(err_fit_kms):
        return np.nan
    return float(abs(sig_fit_kms) / total * err_fit_kms)


def _powerlaw(lbd, scnt, acnt, ref=CAK_LAB_WAVE):
    return np.exp(scnt) * (np.asarray(lbd, dtype=float) / ref) ** acnt


def _fit_powerlaw(lbd, flux, ferr, ref=CAK_LAB_WAVE):
    positive = flux[flux > 0]
    amp = float(np.median(positive)) if positive.size else 1.0
    amp = max(amp, 1e-30)
    pini = [np.log(amp), 0.0]
    plim = [(-20.0, 20.0), (-5.0, 5.0)]

    def nll(param):
        model = _powerlaw(lbd, param[0], param[1], ref)
        return 0.5 * np.sum(((flux - model) / ferr) ** 2)

    res = scipy.optimize.minimize(nll, pini, method='L-BFGS-B', bounds=plim)
    return float(res.x[0]), float(res.x[1])


def _absorption_profile(
    lbdtpl, ttpl, v_shift, sigv, lbd_target, brdspace='log', sig_inst_kms=0.0,
):
    """Shift, broaden, and interpolate the stellar absorption template."""
    sig_broad = combine_velocity_sigmas(sigv, sig_inst_kms)
    lbdtpl_shf = ttools.shift_template(lbdtpl, v_shift, shfspace='log')
    if sig_broad <= 0:
        ttpl_brd = np.asarray(ttpl, dtype=float)
    else:
        pix_kms = _template_pix_kms(lbdtpl)
        ttpl_brd = ttools.broaden_template(lbdtpl_shf, ttpl, sig_broad, pixtpl=pix_kms, brdspace=brdspace)
    return np.interp(lbd_target, lbdtpl_shf, ttpl_brd, left=0.0, right=0.0)


def _cak_model(lbd, scnt, acnt, v_shift, sigv, depth, lbdtpl, ttpl, sig_inst_kms=0.0):
    cont = _powerlaw(lbd, scnt, acnt)
    absorption = _absorption_profile(
        lbdtpl, ttpl, v_shift, sigv, lbd, sig_inst_kms=sig_inst_kms,
    )
    return cont * (1.0 - depth * absorption)


def _resolve_sigv_bounds(sigv_min=None, sigv_max=None):
    """Return active (sigv_min, sigv_max), clamped to the global fit limits."""
    lo = float(SIGV_MIN if sigv_min is None else sigv_min)
    hi = float(SIGV_MAX if sigv_max is None else sigv_max)
    lo = max(float(SIGV_MIN), lo)
    hi = min(float(SIGV_MAX), hi)
    if hi < lo:
        hi = lo
    return lo, hi


def _sigv_at_bound(sigv, sigv_min=None, sigv_max=None, tol=None):
    lo, hi = _resolve_sigv_bounds(sigv_min, sigv_max)
    if tol is None:
        tol = SIGV_BOUND_TOL
    return bool(sigv <= lo + tol or sigv >= hi - tol)


def _clamp_sigv(sigv, sigv_min=None, sigv_max=None):
    lo, hi = _resolve_sigv_bounds(sigv_min, sigv_max)
    return float(np.clip(float(sigv), lo, hi))


def _ensemble_member_ok(fit, template_name, locked_name, sigv_min=None, sigv_max=None):
    """
    Return True if ``fit`` should enter the template-ensemble uncertainty sample.

    The locked template is always kept. Other members are dropped when σ* is
    within ``SIGV_BOUND_TOL`` of the active bounds or depth ≤ ``DEPTH_FAIL_MAX``.
    """
    if template_name == locked_name:
        return True
    if float(fit.get('depth', 0.0)) <= DEPTH_FAIL_MAX:
        return False
    if _sigv_at_bound(float(fit['sigv']), sigv_min=sigv_min, sigv_max=sigv_max):
        return False
    return True


def _full_p0_from_start(p0_start, scnt, acnt, default_v=0.0, default_sigv=120.0, default_depth=0.15):
    """Build a 5-parameter warm start [scnt, acnt, v, sigv, depth]."""
    if p0_start is None:
        return [float(scnt), float(acnt), default_v, default_sigv, default_depth]
    p0_start = list(p0_start)
    if len(p0_start) >= 5:
        return [float(x) for x in p0_start[:5]]
    if len(p0_start) >= 3:
        return [
            float(scnt), float(acnt),
            float(p0_start[0]), float(p0_start[1]), float(p0_start[2]),
        ]
    return [float(scnt), float(acnt), default_v, default_sigv, default_depth]


def _fit_cak_once(
    lbd, flux, ferr, lbdtpl, ttpl, p0=None, sig_inst_kms=0.0, sigv_min=None,
):
    """
    ML fit for one template on a fixed wavelength segment.

    Continuum parameters are initialized from the sideband power law and refined
    jointly with (v_shift, sigv, depth). A hard continuum freeze biases σ* low on
    QSO stacks; joint refinement with a single locked template is preferred.
    Core weights scale with the current sigv at each likelihood evaluation.
    """
    lo, hi = _resolve_sigv_bounds(sigv_min)
    if p0 is None:
        p0 = [0.0, 0.0, 0.0, _clamp_sigv(120.0, lo, hi), 0.15]
    else:
        p0 = list(p0)
        p0[3] = _clamp_sigv(p0[3], lo, hi)
    bounds = [
        (-20.0, 20.0),
        (-5.0, 5.0),
        (-800.0, 800.0),
        (lo, hi),
        (1e-4, 0.99),
    ]

    def nll(param):
        scnt, acnt, v_shift, sigv, depth = param
        if sigv <= 0 or depth <= 0 or depth >= 1:
            return 1e30
        weights = _cak_core_weights(lbd, sigma_kms=sigv)
        model = _cak_model(
            lbd, scnt, acnt, v_shift, sigv, depth, lbdtpl, ttpl, sig_inst_kms,
        )
        return 0.5 * np.sum(weights * ((flux - model) / ferr) ** 2)

    res = scipy.optimize.minimize(nll, p0, method='L-BFGS-B', bounds=bounds)
    if not res.success:
        return None

    scnt, acnt, v_shift, sigv, depth = res.x
    weights = _cak_core_weights(lbd, sigma_kms=sigv)
    model = _cak_model(
        lbd, scnt, acnt, v_shift, sigv, depth, lbdtpl, ttpl, sig_inst_kms,
    )
    chi2 = float(np.sum(weights * ((flux - model) / ferr) ** 2))
    return {
        'scnt': float(scnt),
        'acnt': float(acnt),
        'v_shift': float(v_shift),
        'sigv': float(sigv),
        'depth': float(depth),
        'sig_inst_kms': float(sig_inst_kms),
        'sigv_min': lo,
        'sigv_max': hi,
        'chi2': chi2,
        'ndof': float(max(len(lbd) - 5, 1)),
        'at_bound': _sigv_at_bound(float(sigv), sigv_min=lo, sigv_max=hi),
        'model': model,
        'continuum': _powerlaw(lbd, scnt, acnt),
        'absorption': _absorption_profile(
            lbdtpl, ttpl, v_shift, sigv, lbd, sig_inst_kms=sig_inst_kms,
        ),
    }


def _fit_cak_multistart(
    lbd, flux, ferr, lbdtpl, ttpl, scnt, acnt, p0_base=None, sig_inst_kms=0.0,
    sigv_min=None,
):
    """Run ``_fit_cak_once`` from several sigv starting values; keep lowest chi2."""
    lo, hi = _resolve_sigv_bounds(sigv_min)
    base = _full_p0_from_start(p0_base, scnt, acnt)
    base[3] = _clamp_sigv(base[3], lo, hi)
    starts = []
    for sig0 in [base[3], *SIGV_START_GRID]:
        sig_clamped = _clamp_sigv(sig0, lo, hi)
        if all(abs(sig_clamped - s) > 15.0 for s in starts):
            starts.append(sig_clamped)

    best = None
    for sig0 in starts:
        p0 = [base[0], base[1], base[2], float(sig0), base[4]]
        fit = _fit_cak_once(
            lbd, flux, ferr, lbdtpl, ttpl, p0=p0, sig_inst_kms=sig_inst_kms,
            sigv_min=lo,
        )
        if fit is None:
            continue
        if best is None or fit['chi2'] < best['chi2']:
            best = fit
    return best


def _refine_line_half_excl(sigv_kms, rest_wave, current_half):
    sigma_lbd = rest_wave * sigv_kms / C_KMS
    target = max(2.0 * sigma_lbd, LINE_HALF_EXCL * 0.5)
    return max(current_half, min(target, 20.0))


def fit_cak_with_template(
    spres, lbdtpl, ttpl, max_cont_iter=MAX_CONT_ITER, z=None, p0_start=None,
    sigv_min=None,
):
    """
    Fit Ca K with iterative continuum sideband placement for one stellar template.

    Sideband power-law coefficients initialize the continuum; continuum amplitude
    and slope are then refined jointly with (v_shift, sigv, depth). Multiple
    sigv starting values are tried to avoid shallow local minima. Returns fit
    dict with model arrays on the line-fit window, or None.

    ``sigv_min`` raises the lower bound on σ* (default ``SIGV_MIN``). Validation
    uses ``max(SIGV_MIN, verr)`` so the fit cannot go below the injected error.
    """
    lbd_all = np.asarray(spres['lbd'], dtype=float)
    flux_all = np.asarray(spres['f'], dtype=float)
    ferr_all = np.asarray(spres['ferr'], dtype=float)
    lbd_min, lbd_max = float(np.min(lbd_all)), float(np.max(lbd_all))
    sig_inst_kms = 0.0
    if z is not None:
        sig_inst_kms = desi_spectro_instrumental_sigma_kms_from_rest(CAK_LAB_WAVE, z)
    lo, _hi = _resolve_sigv_bounds(sigv_min)

    line_half = LINE_HALF_EXCL
    best = None
    p0 = list(p0_start) if p0_start is not None else None
    use_multistart = p0_start is None

    for _ in range(max_cont_iter):
        rngline, lwrng, cntrng = build_cak_windows(
            lbd_min, lbd_max, line_half_excl=line_half,
        )

        blue_mask = (lbd_all >= cntrng[0]) & (lbd_all <= cntrng[1])
        red_mask = (lbd_all >= cntrng[2]) & (lbd_all <= cntrng[3])
        cont_mask = blue_mask | red_mask
        if int(np.sum(cont_mask)) < MIN_PIXELS:
            return None

        scnt, acnt = _fit_powerlaw(
            lbd_all[cont_mask], flux_all[cont_mask], ferr_all[cont_mask],
        )

        line_mask = (lbd_all >= rngline[0]) & (lbd_all <= rngline[1])
        lbd = lbd_all[line_mask]
        flux = flux_all[line_mask]
        ferr = ferr_all[line_mask]
        if len(lbd) < MIN_PIXELS:
            return None

        if use_multistart:
            fit = _fit_cak_multistart(
                lbd, flux, ferr, lbdtpl, ttpl, scnt, acnt,
                p0_base=p0, sig_inst_kms=sig_inst_kms, sigv_min=lo,
            )
            use_multistart = False
        else:
            fit = _fit_cak_once(
                lbd, flux, ferr, lbdtpl, ttpl,
                p0=_full_p0_from_start(p0, scnt, acnt),
                sig_inst_kms=sig_inst_kms,
                sigv_min=lo,
            )
        if fit is None:
            return None

        p0 = [fit['scnt'], fit['acnt'], fit['v_shift'], fit['sigv'], fit['depth']]
        best = fit
        best.update({
            'lbd': lbd,
            'flux': flux,
            'ferr': ferr,
            'rngline': rngline,
            'lwrng': lwrng,
            'cntrng': cntrng,
        })

        new_half = _refine_line_half_excl(fit['sigv'], CAK_LAB_WAVE, line_half)
        if abs(new_half - line_half) < 0.5:
            break
        line_half = new_half

    if best is None:
        return None
    best['residuals'] = best['flux'] - best['model']
    return best


def build_cak_plot_data(spres, fit, lbdtpl, ttpl, z=None):
    """Build Ca K plot arrays on a wide window that includes Ca H."""
    lbd_all = np.asarray(spres['lbd'], dtype=float)
    flux_all = np.asarray(spres['f'], dtype=float)
    ferr_all = np.asarray(spres['ferr'], dtype=float)
    lbd_min, lbd_max = float(np.min(lbd_all)), float(np.max(lbd_all))

    plot_lo, plot_hi = build_cak_plot_window(lbd_min, lbd_max)
    mask = (lbd_all >= plot_lo) & (lbd_all <= plot_hi)
    lbd = lbd_all[mask]
    flux = flux_all[mask]
    ferr = ferr_all[mask]
    if lbd.size < MIN_PIXELS:
        return None

    scnt = fit['scnt']
    acnt = fit['acnt']
    v_shift = fit['v_shift']
    sigv = fit['sigv']
    depth = fit['depth']
    sig_inst_kms = fit.get('sig_inst_kms', 0.0)
    if z is not None and (sig_inst_kms is None or sig_inst_kms <= 0):
        sig_inst_kms = desi_spectro_instrumental_sigma_kms_from_rest(CAK_LAB_WAVE, z)

    continuum = _powerlaw(lbd, scnt, acnt)
    absorption = _absorption_profile(
        lbdtpl, ttpl, v_shift, sigv, lbd, sig_inst_kms=sig_inst_kms,
    )
    model = continuum * (1.0 - depth * absorption)
    absorption_template = _absorption_profile(
        lbdtpl, ttpl, v_shift, PLOT_TEMPLATE_SIG_KMS, lbd, sig_inst_kms=0.0,
    )
    template_broad = continuum * (1.0 - depth * absorption_template)
    fit_range = fit.get('rngline', [CAK_LAB_WAVE - FIT_HALF, CAK_LAB_WAVE + FIT_HALF])

    return {
        'lbd': lbd,
        'flux': flux,
        'ferr': ferr,
        'continuum': continuum,
        'model': model,
        'template_broad': template_broad,
        'absorption': absorption,
        'absorption_template': absorption_template,
        'residuals': flux - model,
        'rest_wave': CAK_LAB_WAVE,
        'cah_wave': CAH_LAB_WAVE,
        'label': 'Ca II K',
        'template_sig_kms': float(PLOT_TEMPLATE_SIG_KMS),
        'fit_range': [float(fit_range[0]), float(fit_range[1])],
        'sig_inst_kms': float(sig_inst_kms),
    }


def _ivar_from_err(err):
    if not np.isfinite(err) or err <= 0:
        return 0.0
    return float(1.0 / err ** 2)


def template_percentile_half_range(values):
    """
    Return the (84th - 16th percentile) / 2 scatter of ``values``.

    Used for Ca K uncertainties from the stellar-template ensemble. With fewer
    than two finite values the scatter is 0.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    if finite.size == 1:
        return 0.0
    return float((np.percentile(finite, 84) - np.percentile(finite, 16)) / 2.0)


def _emit_cak_warnings(warnings):
    for message in warnings:
        print('WARNING: %s' % message)


def _collect_cak_templates(template_dir, locked_template_name=None):
    """
    Build the template list for an ensemble fit.

    All enabled manifest templates are included. If ``locked_template_name`` is
    set and not already present, that template is added even when disabled.
    """
    template_dir = template_dir or default_stellar_template_dir()
    warnings: List[str] = []
    templates = list(load_cak_template_catalog(template_dir))
    names = {t.name for t in templates}
    template_disabled = False

    if locked_template_name is not None:
        locked, err = find_cak_template_by_name(template_dir, locked_template_name)
        if locked is None:
            return None, None, [err]
        if locked.name not in names:
            templates.append(locked)
        if not locked.enabled:
            template_disabled = True
            warnings.append(
                'Ca K template %r is disabled in the manifest but was selected '
                'as the locked / reporting template.' % locked.name
            )

    if not templates:
        warnings.append('No Ca K templates found in %s' % template_dir)
        return [], warnings, None

    return templates, warnings, template_disabled


def fit_cak_all_templates(
    spres,
    template_dir=None,
    z=None,
    template_name=None,
    locked_template_name=None,
    sigv_min=None,
):
    """
    Fit Ca K with the stellar-template ensemble.

    All enabled templates are fit. The reported point estimate comes from the
    locked template: ``locked_template_name`` or ``template_name`` if given,
    otherwise the lowest-χ² template. Uncertainties are the 16–84 percentile
    half-range over ensemble members that pass the failed-fit cull (σ* near the
    active bounds or depth ≤ ``DEPTH_FAIL_MAX``), always including the locked
    template. If only the locked template remains, errors are NaN. Bootstrap is
    not used.

    Returns dict with metrics, best-fit plot data, per-template arrays, and metadata.
    """
    template_dir = template_dir or default_stellar_template_dir()
    lock_name = locked_template_name if locked_template_name is not None else template_name
    lo, hi = _resolve_sigv_bounds(sigv_min)

    templates, warnings, template_disabled = _collect_cak_templates(
        template_dir, locked_template_name=lock_name,
    )
    if templates is None:
        print('ERROR: %s' % warnings[0])
        return None
    if not templates:
        for message in warnings:
            print('WARNING: %s' % message)
        return None

    template_fits = []
    for template in templates:
        lbdtpl, ttpl = load_cak_template(template_dir, template)
        sig_lsf_kms = template_lsf_sigma_kms(CAK_LAB_WAVE, template)
        fit = fit_cak_with_template(spres, lbdtpl, ttpl, z=z, sigv_min=lo)
        if fit is None:
            continue
        template_fits.append({
            'template': template,
            'lbdtpl': lbdtpl,
            'ttpl': ttpl,
            'sig_lsf_kms': sig_lsf_kms,
            'fit': fit,
        })

    if not template_fits:
        return None

    if lock_name is not None:
        locked_items = [
            item for item in template_fits if item['template'].name == lock_name
        ]
        if not locked_items:
            print('ERROR: Locked Ca K template %r failed to fit.' % lock_name)
            return None
        locked = locked_items[0]
    else:
        locked = min(template_fits, key=lambda item: item['fit']['chi2'])
    locked_name = locked['template'].name

    v_shifts = [item['fit']['v_shift'] for item in template_fits]
    sigvs = [item['fit']['sigv'] for item in template_fits]
    sigvs_total = [
        stellar_disp_total_kms(
            item['fit']['sigv'],
            item['fit'].get('sig_inst_kms', 0.0),
            item['sig_lsf_kms'],
        )
        for item in template_fits
    ]
    depths = [item['fit']['depth'] for item in template_fits]
    template_names = [item['template'].name for item in template_fits]

    ensemble_ok = [
        item for item in template_fits
        if _ensemble_member_ok(
            item['fit'], item['template'].name, locked_name,
            sigv_min=lo, sigv_max=hi,
        )
    ]
    n_culled = len(template_fits) - len(ensemble_ok)
    if len(ensemble_ok) < 2:
        err_centroid = np.nan
        err_disp = np.nan
        err_disp_total = np.nan
        err_depth = np.nan
        warnings.append(
            'Ca K ensemble uncertainty undefined: %d/%d templates culled '
            '(σ* near bounds [%.0f, %.0f] km/s or depth ≤ %.3g); only locked '
            'template %r remains.'
            % (n_culled, len(template_fits), lo, hi, DEPTH_FAIL_MAX, locked_name)
        )
    else:
        err_centroid = template_percentile_half_range(
            [item['fit']['v_shift'] for item in ensemble_ok]
        )
        err_disp = template_percentile_half_range(
            [item['fit']['sigv'] for item in ensemble_ok]
        )
        err_disp_total = template_percentile_half_range([
            stellar_disp_total_kms(
                item['fit']['sigv'],
                item['fit'].get('sig_inst_kms', 0.0),
                item['sig_lsf_kms'],
            )
            for item in ensemble_ok
        ])
        err_depth = template_percentile_half_range(
            [item['fit']['depth'] for item in ensemble_ok]
        )

    fit = locked['fit']
    sig_inst_kms = float(fit.get('sig_inst_kms', 0.0))
    sig_lsf_kms = float(locked['sig_lsf_kms'])
    sigv_total = stellar_disp_total_kms(fit['sigv'], sig_inst_kms, sig_lsf_kms)
    if not np.isfinite(err_disp_total) and np.isfinite(err_disp):
        err_disp_total = _propagate_total_disp_err(
            fit['sigv'], err_disp, sig_inst_kms, sig_lsf_kms,
        )

    at_bound = bool(
        fit.get('at_bound', _sigv_at_bound(fit['sigv'], sigv_min=lo, sigv_max=hi))
    )
    if at_bound:
        warnings.append(
            'CAK_STELLAR_DISP = %.1f km/s is at the fit bound (%.0f–%.0f km/s).'
            % (fit['sigv'], lo, hi)
        )

    _emit_cak_warnings(warnings)

    plot = build_cak_plot_data(spres, fit, locked['lbdtpl'], locked['ttpl'], z=z)
    if plot is None:
        absorption_template = _absorption_profile(
            locked['lbdtpl'], locked['ttpl'], fit['v_shift'], PLOT_TEMPLATE_SIG_KMS,
            fit['lbd'], sig_inst_kms=0.0,
        )
        fit_range = fit.get('rngline', [CAK_LAB_WAVE - FIT_HALF, CAK_LAB_WAVE + FIT_HALF])
        plot = {
            'lbd': fit['lbd'],
            'flux': fit['flux'],
            'ferr': fit['ferr'],
            'continuum': fit['continuum'],
            'model': fit['model'],
            'template_broad': fit['continuum'] * (1.0 - fit['depth'] * absorption_template),
            'absorption': fit['absorption'],
            'residuals': fit['residuals'],
            'rest_wave': CAK_LAB_WAVE,
            'cah_wave': CAH_LAB_WAVE,
            'label': 'Ca II K',
            'template_sig_kms': float(PLOT_TEMPLATE_SIG_KMS),
            'fit_range': [float(fit_range[0]), float(fit_range[1])],
            'sig_inst_kms': sig_inst_kms,
        }

    metrics = {
        'CAK_CENTROID': fit['v_shift'],
        'CAK_STELLAR_DISP': fit['sigv'],
        'CAK_STELLAR_DISP_TOTAL': sigv_total,
        'CAK_DEPTH': fit['depth'],
        'CAK_CHI2': fit['chi2'],
        'CAK_CHI2_DOF': fit['ndof'],
        'CAK_AT_BOUND': 1.0 if at_bound else 0.0,
        'CAK_CENTROID_IVAR': _ivar_from_err(err_centroid),
        'CAK_STELLAR_DISP_IVAR': _ivar_from_err(err_disp),
        'CAK_STELLAR_DISP_TOTAL_IVAR': _ivar_from_err(err_disp_total),
        'CAK_DEPTH_IVAR': _ivar_from_err(err_depth),
        'CAK_STELLAR_DISP_ERR': err_disp,
        'CAK_CENTROID_ERR': err_centroid,
        'CAK_DEPTH_ERR': err_depth,
        'CAK_STELLAR_DISP_TOTAL_ERR': err_disp_total,
    }

    return {
        'metrics': metrics,
        'plot': plot,
        'best_template': locked_name,
        'locked_template': locked_name,
        'template_systematics': {
            'centroid_kms': err_centroid,
            'stellar_disp_kms': err_disp,
            'stellar_disp_total_kms': err_disp_total,
            'depth': err_depth,
        },
        'all_templates': template_names,
        'ensemble_templates': [item['template'].name for item in ensemble_ok],
        'template_sigvs': np.asarray(sigvs, dtype=float),
        'template_sigvs_total': np.asarray(sigvs_total, dtype=float),
        'template_centroids': np.asarray(v_shifts, dtype=float),
        'template_depths': np.asarray(depths, dtype=float),
        'template_lsf_kms': sig_lsf_kms,
        'instrumental_sigma_kms': sig_inst_kms,
        'sigv_min': lo,
        'sigv_max': hi,
        'n_ensemble': len(ensemble_ok),
        'n_culled': n_culled,
        'at_bound': at_bound,
        'template_disabled': bool(template_disabled),
        'warnings': warnings,
    }


def measure_cak_absorption(
    spres,
    template_dir=None,
    z=None,
    template_name=None,
    locked_template_name=None,
    n_boot=None,
    sigv_min=None,
):
    """
    Public entry point for Ca K measurement.

    ``template_name`` / ``locked_template_name`` select the reporting template;
    enabled templates are fit and culled for 16–84 uncertainties. ``sigv_min``
    raises the lower bound on σ* (validation: ``max(SIGV_MIN, verr)``).
    ``n_boot`` is accepted for backward compatibility and ignored.
    """
    if n_boot is not None:
        pass  # bootstrap uncertainties retired; template ensemble scatter only
    if not cak_is_measurable(spres):
        return None
    return fit_cak_all_templates(
        spres,
        template_dir=template_dir,
        z=z,
        template_name=template_name,
        locked_template_name=locked_template_name,
        sigv_min=sigv_min,
    )


def expected_dispersion_kms(sigma0_kms, verr_kms):
    """Quadrature expectation √(σ₀² + σ_verr²)."""
    return float(np.sqrt(float(sigma0_kms) ** 2 + float(verr_kms) ** 2))
