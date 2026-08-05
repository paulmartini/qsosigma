"""Ca K fit-result helpers: plot FITS extensions and console summaries."""

import numpy as np
from astropy.io import fits
from astropy.table import Table

from qsosigma.cak_metrics import (
    CAH_LAB_WAVE,
    CAK_LAB_WAVE,
    CAK_METRIC_SUFFIXES,
)

FITS_STR_MAX = 68  # FITS header string value limit (characters)


def _format_sigfig(value, n=3):
    """Format a float with ``n`` significant figures."""
    value = float(value)
    if not np.isfinite(value):
        return 'nan'
    if value == 0.0:
        return '0'
    return ('%#.*g' % (n, value)).rstrip('.')


def _format_value_err(value, err, unit, n=3):
    """Format value +/- error with n significant figures."""
    unit = (' ' + unit) if unit else ''
    if not np.isfinite(value):
        return 'nan%s' % unit
    if not np.isfinite(err) or err <= 0:
        return '%s%s' % (_format_sigfig(value, n), unit)
    return '%s +/- %s%s' % (
        _format_sigfig(value, n), _format_sigfig(err, n), unit,
    )


def build_cak_plot_hdu(plot, pixspec=None, flux_unit=None, name='CAK_PLOT'):
    """
    Build a Ca K plot binary-table extension from a plot dict.

    Required keys: ``lbd``, ``flux``, ``ferr``, ``continuum``, ``model``,
    ``template_broad``, ``residuals``. Optional: ``fit_range``,
    ``template_sig_kms``, ``rest_wave``, ``cah_wave``. Writes header keywords
    ``CAKFITLO/HI``, ``CAKTPLSG``, ``CAKRESTW``, ``CAKHWAVE``, ``CAKPIXSP``,
    ``FLUXUNIT`` when available.
    """
    if plot is None:
        return None

    table = Table({
        'WAVE': np.asarray(plot['lbd'], dtype=float),
        'FLUX': np.asarray(plot['flux'], dtype=float),
        'FERR': np.asarray(plot['ferr'], dtype=float),
        'CONTINUUM': np.asarray(plot['continuum'], dtype=float),
        'MODEL': np.asarray(plot['model'], dtype=float),
        'TEMPLATE': np.asarray(plot['template_broad'], dtype=float),
        'RESIDUAL': np.asarray(plot['residuals'], dtype=float),
    })
    hdu = fits.BinTableHDU(table, name=str(name))
    header = hdu.header
    fit_range = plot.get('fit_range')
    if fit_range is not None:
        header['CAKFITLO'] = float(fit_range[0])
        header['CAKFITHI'] = float(fit_range[1])
    if plot.get('template_sig_kms') is not None and np.isfinite(plot['template_sig_kms']):
        header['CAKTPLSG'] = float(plot['template_sig_kms'])
    header['CAKRESTW'] = float(plot.get('rest_wave', CAK_LAB_WAVE))
    header['CAKHWAVE'] = float(plot.get('cah_wave', CAH_LAB_WAVE))
    if pixspec is not None and np.isfinite(pixspec):
        header['CAKPIXSP'] = float(pixspec)
    if flux_unit:
        header['FLUXUNIT'] = str(flux_unit)[:FITS_STR_MAX]
    return hdu


def print_cak_summary(results, cak_meta=None):
    """Print Ca K metrics; uncertainties are template-ensemble 16–84 half-ranges."""
    print('CAK (Ca II K stellar absorption):')
    print(
        '  CAK_STELLAR_DISP (σ*) is stellar/kinematic broadening with the DESI '
        'instrumental LSF in the forward model (not template-LSF corrected).'
    )
    print(
        '  CAK_STELLAR_DISP_TOTAL is √(σ*² + σ_inst² + σ_template_LSF²) '
        '(diagnostic only).'
    )
    print(
        '  Uncertainties are the 16–84 half-range over the culled template '
        'ensemble (NaN if only the locked template remains).'
    )
    for suffix in CAK_METRIC_SUFFIXES:
        key = 'CAK_%s' % suffix
        value = results.get(key, np.nan)
        err_key = '%s_ERR' % key
        if err_key in results and np.isfinite(results.get(err_key, np.nan)):
            err = float(results[err_key])
        else:
            ivar = results.get('%s_IVAR' % key, 0.0)
            err = (1.0 / np.sqrt(ivar)) if ivar > 0 else np.nan
        if suffix == 'DEPTH':
            print('  %s: %s' % (key, _format_value_err(value, err, '', n=3)))
        else:
            print('  %s: %s' % (key, _format_value_err(value, err, 'km/s', n=3)))
    print('  CAK_CHI2: %.2f' % results.get('CAK_CHI2', np.nan))
    print('  CAK_CHI2_DOF: %.1f' % results.get('CAK_CHI2_DOF', np.nan))
    at_bound = results.get('CAK_AT_BOUND', 0.0)
    if np.isfinite(at_bound):
        print('  CAK_AT_BOUND: %s' % ('yes' if at_bound >= 0.5 else 'no'))
    if cak_meta is not None:
        locked = cak_meta.get('locked_template') or cak_meta.get('best_template', '')
        print('  locked template: %s' % locked)
        n_tpl = len(cak_meta.get('all_templates') or [])
        if n_tpl:
            print('  template ensemble: %d' % n_tpl)
        for message in cak_meta.get('warnings') or []:
            print('  WARNING: %s' % message)
