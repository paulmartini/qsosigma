"""Write combined emission-line fit results to FITS and print summaries."""
    
import os
    
import numpy as np
from astropy.io import fits
from astropy.table import Table

from cak_metrics import (
    CAH_LAB_WAVE,
    CAK_LAB_WAVE,
    CAK_METRIC_SUFFIXES,
    CAK_PREFIX,
)   
    
FITS_STR_MAX = 68

def _set_header_str(header, keyword, value, continuation=None):
    """Write a string header value, truncating to the FITS 68-char limit."""
    text = str(value)
    header[keyword] = text[:FITS_STR_MAX]
    if continuation and len(text) > FITS_STR_MAX:
        header[continuation] = text[FITS_STR_MAX:2 * FITS_STR_MAX]

def _apply_lines_meta_header(header, meta, fe_params=None):
    """Write shared metadata to the LINES extension header."""
    header['OBSERVER'] = str(meta.get('user', ''))[:68]
    header['DATE'] = str(meta.get('date', ''))[:68]
    input_file = str(meta.get('input_file', ''))
    header['INFILE'] = os.path.basename(input_file)[:68]
    if input_file:
        dirpart = os.path.dirname(input_file)
        header['INFILDIR'] = dirpart[-68:]
    if meta.get('uncertainty_floor') is not None and np.isfinite(meta['uncertainty_floor']):
        header['HIERARCH IRONFIT UNCERTFLR'] = float(meta['uncertainty_floor'])
    if meta.get('z') is not None:
        header['Z'] = float(meta['z'])
    fe_template = meta.get('fe_template')
    if fe_template is not None:
        header['FETMPL'] = str(fe_template)[:68]
    params = fe_params if fe_params is not None else meta.get('fe_params')
    _fe_param_header(header, params, fe_template)

    fit_lines = meta.get('fit_lines', [])
    if fit_lines:
        _set_header_str(
            header, 'FITLINES', ', '.join(fit_lines), continuation='FITLN2',
        )
    if meta.get('cak_template') is not None:
        header['CAKTPL'] = str(meta['cak_template'])[:FITS_STR_MAX]
    if meta.get('cak_templates') is not None:
        _set_header_str(
            header, 'CAKTPLS', ', '.join(meta['cak_templates']), continuation='CAKTPL2',
        )
    if meta.get('cak_boot') is not None:
        header['HIERARCH CAK NBOOT'] = int(meta['cak_boot'])
    if meta.get('cak_template_lsf_kms') is not None and np.isfinite(meta['cak_template_lsf_kms']):
        header['HIERARCH CAK TPL LSF'] = float(meta['cak_template_lsf_kms'])
    if meta.get('cak_inst_sigma_kms') is not None and np.isfinite(meta['cak_inst_sigma_kms']):
        header['HIERARCH CAK INST SIG'] = float(meta['cak_inst_sigma_kms'])
    if meta.get('cak_at_bound') is not None:
        header['HIERARCH CAK AT BOUND'] = bool(meta['cak_at_bound'])
    if meta.get('cak_template_disabled') is not None:
        header['HIERARCH CAK TPL DISAB'] = bool(meta['cak_template_disabled'])
    warnings = meta.get('cak_warnings') or []
    for i, message in enumerate(warnings[:4]):
        key = 'CAKWARN%d' % (i + 1)
        header[key] = str(message)[:FITS_STR_MAX]
    for key in ('centroid', 'stellar_disp', 'depth'):
        val = meta.get('cak_sys_%s' % key)
        if val is not None and np.isfinite(val):
            header['HIERARCH CAK SYS %s' % key.upper()] = float(val)
    val = meta.get('cak_sys_stellar_disp_total')
    if val is not None and np.isfinite(val):
        header['HIERARCH CAK SYS SDTOT'] = float(val)

def build_cak_plot_hdu(plot, pixspec=None, flux_unit=None, name='CAK_PLOT'):
    """Build a Ca K plot binary-table extension from a plot dict."""
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

def write_fit_results_fits(
    path,
    results,
    meta=None,
    fe_params=None,
    cak_plot=None,
    pixspec=None,
    flux_unit=None,
):
    """
    Write one-row LINES table and optional CAK_PLOT extension to FITS.
    """
    row = {}
    for key in METRIC_IVAR_COLUMNS:
        row[key] = results.get(key, np.nan)
        row['%s_IVAR' % key] = results.get('%s_IVAR' % key, 0.0)

    for line in EMISSION_LINES:
        prefix = line.prefix
        for suffix in SINGLE_LINE_METRIC_SUFFIXES:
            col = '%s_%s' % (prefix, suffix)
            row[col] = results.get(col, np.nan)
            row['%s_IVAR' % col] = results.get('%s_IVAR' % col, 0.0)
        row['%s_CHI2' % prefix] = results.get('%s_CHI2' % prefix, np.nan)
        row['%s_CHI2_DOF' % prefix] = results.get('%s_CHI2_DOF' % prefix, np.nan)

    for suffix in CAK_METRIC_SUFFIXES:
        col = '%s_%s' % (CAK_PREFIX, suffix)
        row[col] = results.get(col, np.nan)
        row['%s_IVAR' % col] = results.get('%s_IVAR' % col, 0.0)
    row['CAK_CHI2'] = results.get('CAK_CHI2', np.nan)
    row['CAK_CHI2_DOF'] = results.get('CAK_CHI2_DOF', np.nan)
    row['CAK_AT_BOUND'] = float(results.get('CAK_AT_BOUND', 0.0))

    table = Table({col: [row[col]] for col in row})
    lines_hdu = fits.BinTableHDU(table, name='LINES')

    if meta is not None:
        _apply_lines_meta_header(lines_hdu.header, meta, fe_params=fe_params)

    hdul = fits.HDUList([fits.PrimaryHDU(), lines_hdu])
    cak_hdu = build_cak_plot_hdu(cak_plot, pixspec=pixspec, flux_unit=flux_unit)
    if cak_hdu is not None:
        hdul.append(cak_hdu)
    hdul.writeto(path, overwrite=True)

def print_cak_summary(results, cak_meta=None):
    print('CAK (Ca II K stellar absorption):')
    print('  CAK_STELLAR_DISP is the LSF-corrected stellar / kinematic sigma*.')
    print('  CAK_STELLAR_DISP_TOTAL is the full kernel width (diagnostic only).')
    print('  Uncertainties are the 16–84 half-range over the template ensemble.')
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

def print_all_summaries(results, emission_fits, cak_result=None):
    for fit in emission_fits:
        line = fit['line']
        print_emission_line_summary(line.prefix, line.name, results)
    if cak_result is not None:
        print_cak_summary(cak_result.get('metrics', results), cak_meta=cak_result)  

