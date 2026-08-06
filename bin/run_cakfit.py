#!/usr/bin/env python3
"""
Fit Ca II K stellar absorption velocity dispersion.

Single-spectrum mode
--------------------
  python bin/run_cakfit.py stack.fits
  python bin/run_cakfit.py stack.fits --cak-template hd138688
  python bin/run_cakfit.py stack.fits --uncertainty-floor 0.002 --clobber

Fits all enabled stellar templates. The reported CAK_STELLAR_DISP comes from
the locked template (``--cak-template``, or lowest-χ² if omitted). Uncertainties
are the 16–84 percentile half-range over the culled template ensemble
(failed fits near σ* bounds or with tiny depth are dropped; locked is kept).
Default FITS output: ``cak_fitresults_z{zlo}_z{zhi}.fits`` when the stack name
contains a redshift bin, otherwise ``cak_fitresults_{stem}.fits``. The matching
PNG is ``cak_fitresults_*.png`` (disable with ``--no-plot``).

Multi-stack (``--validate``) mode
---------------------------------
  python bin/run_cakfit.py --validate \\
      --verr-root /path/to/verrtests \\
      --zlo 0.05 --zhi 0.10

Same default FITS filename as single-spectrum mode for that redshift bin. In
addition to the verr0 stack, discovers ``verr100/``, … stacks, fits each, and
writes ``VERR*`` / ``TPL*`` / ``CAKPLOT*`` extensions for every level. Also
writes ``cak_verr_diagnostic_z{zlo}_z{zhi}.png``. For each stack the σ* lower
fit bound is raised to ``max(20, verr)`` km/s.

Ca K templates: data/stellar_templates/ (see that directory's README.md).
"""

from __future__ import annotations

import argparse
import getpass
import glob
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow ``python bin/run_cakfit.py`` from a checkout without PYTHONPATH set.
_PY_DIR = Path(__file__).resolve().parents[1] / 'py'
if _PY_DIR.is_dir() and str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

import numpy as np
from astropy.io import fits
from astropy.table import Table

from qsosigma.cak_metrics import (
    CAK_LAB_WAVE,
    SIGV_MIN,
    cak_is_measurable,
    default_stellar_template_dir,
    expected_dispersion_kms,
    measure_cak_absorption,
)
from qsosigma.cak_plots import (
    parse_redshift_bin_from_name,
    plot_cak,
    plot_cak_verr_diagnostic,
    redshift_bin_tag,
)
from qsosigma.fit_results import build_cak_plot_hdu, print_cak_summary
from qsosigma.spectrum_io import (
    DEFAULT_UNCERTAINTY_FLOOR,
    flux_label,
    infer_fscale,
    load_spectrum,
    spectrum_stem,
)


def _cakplot_extname(verr_kms):
    """FITS extension name for a verr plot snapshot (e.g. CAKPLOT100)."""
    return 'CAKPLOT%d' % int(round(float(verr_kms)))

FITS_STR_MAX = 68  # FITS header string value limit (characters)
VERR_DIR_RE = re.compile(r'^verr(\d+)$')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Fit Ca II K stellar absorption velocity dispersion.',
    )
    parser.add_argument(
        'spectrum',
        nargs='?',
        default=None,
        help='Input spectrum (single-spectrum mode)',
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help=(
            'Also fit verr-injected stacks under --verr-root for the redshift '
            'bin and write VERR*/TPL*/CAKPLOT* extensions for each level'
        ),
    )
    parser.add_argument(
        '--verr-root',
        default=None,
        help='Root directory containing verr0/, verr100/, … (--validate mode)',
    )
    parser.add_argument(
        '--zlo', type=float, default=None,
        help='Redshift bin lower edge (--validate mode)',
    )
    parser.add_argument(
        '--zhi', type=float, default=None,
        help='Redshift bin upper edge (--validate mode)',
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help=(
            'Output FITS path. Default: cak_fitresults_z{zlo}_z{zhi}.fits '
            'when a redshift bin is known, else cak_fitresults_{stem}.fits'
        ),
    )
    parser.add_argument('--z', type=float, default=None,
                        help='Redshift override (single-spectrum mode)')
    parser.add_argument(
        '--output-dir', default='.',
        help='Directory for default output paths (default: .)',
    )
    parser.add_argument(
        '--cak-template', default=None,
        help='Locked reporting template (manifest name). Default: best χ².',
    )
    parser.add_argument(
        '--stellar-template-dir',
        default=None,
        help='Directory with Ca K templates and templates.manifest.csv',
    )
    parser.add_argument(
        '--uncertainty-floor', type=float, default=DEFAULT_UNCERTAINTY_FLOOR,
        help=(
            'Fractional uncertainty floor per pixel (default: %g)'
            % DEFAULT_UNCERTAINTY_FLOOR
        ),
    )
    parser.add_argument(
        '--clobber',
        action='store_true',
        help='Overwrite existing output files',
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not write Ca K PNG output(s)',
    )
    return parser.parse_args(argv)


def stellar_template_dir(args_dir):
    if args_dir is not None:
        return os.path.abspath(args_dir)
    return default_stellar_template_dir()


def default_fitresults_filename(zlo=None, zhi=None, stem=None):
    """Return the default Ca K fit-results FITS basename."""
    if zlo is not None and zhi is not None:
        return 'cak_fitresults_%s.fits' % redshift_bin_tag(zlo, zhi)
    if stem:
        return 'cak_fitresults_%s.fits' % stem
    raise ValueError('Need zlo/zhi or stem for default fit-results filename')


def discover_verr_directories(verr_root):
    """Return sorted list of (verr_kms, directory_path) under verr_root."""
    verr_root = os.path.abspath(verr_root)
    found = []
    if not os.path.isdir(verr_root):
        return found
    for name in os.listdir(verr_root):
        match = VERR_DIR_RE.match(name)
        if not match:
            continue
        path = os.path.join(verr_root, name)
        if os.path.isdir(path):
            found.append((float(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    return found


def find_stack_in_directory(directory, zlo, zhi):
    """
    Find the stacked spectrum FITS for a redshift bin inside a verr directory.

    Matches ``stack_*z{zlo}_z{zhi}*dark.fits`` (dark-time coadds only), so
    other products with the same redshift tag are excluded.
    """
    tag = redshift_bin_tag(zlo, zhi)
    pattern = os.path.join(directory, 'stack_*%s*dark.fits' % tag)
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(
            'WARNING: Multiple stacks match %s in %s; using %s'
            % (tag, directory, os.path.basename(candidates[0]))
        )
    return candidates[0]


def discover_validation_stacks(verr_root, zlo, zhi):
    """
    Return list of dicts ``{verr_kms, path}`` for available verr directories.

    Requires verr0. Other missing verr levels are skipped with a warning.
    """
    dirs = discover_verr_directories(verr_root)
    if not dirs:
        raise FileNotFoundError('No verr* directories found under %s' % verr_root)

    stacks = []
    for verr_kms, directory in dirs:
        path = find_stack_in_directory(directory, zlo, zhi)
        if path is None:
            print(
                'WARNING: No stack for %s in %s; skipping verr=%.0f'
                % (redshift_bin_tag(zlo, zhi), directory, verr_kms)
            )
            continue
        stacks.append({'verr_kms': verr_kms, 'path': path})

    if not stacks or stacks[0]['verr_kms'] != 0.0:
        # allow verr0 not first if somehow ordered wrong
        zero = [s for s in stacks if s['verr_kms'] == 0.0]
        if not zero:
            raise FileNotFoundError(
                'Reference verr0 stack not found for %s under %s'
                % (redshift_bin_tag(zlo, zhi), verr_root)
            )
    return stacks


def _set_header_str(header, keyword, value, continuation=None):
    text = str(value)
    header[keyword] = text[:FITS_STR_MAX]
    if continuation and len(text) > FITS_STR_MAX:
        header[continuation] = text[FITS_STR_MAX:2 * FITS_STR_MAX]


def result_row(cak_result, verr_kms=None, expected_disp=None, input_file=None):
    """Build a flat dict of columns for one CaK fit result."""
    metrics = cak_result['metrics']
    names = list(cak_result.get('all_templates') or [])
    sigvs = np.asarray(cak_result.get('template_sigvs', []), dtype=float)
    row = {
        'VERR_KMS': np.nan if verr_kms is None else float(verr_kms),
        'EXPECTED_DISP': np.nan if expected_disp is None else float(expected_disp),
        'LOCKED_TEMPLATE': str(cak_result.get('locked_template')
                               or cak_result.get('best_template') or ''),
        'CAK_STELLAR_DISP': float(metrics['CAK_STELLAR_DISP']),
        'CAK_STELLAR_DISP_ERR': float(metrics.get(
            'CAK_STELLAR_DISP_ERR',
            (1.0 / np.sqrt(metrics['CAK_STELLAR_DISP_IVAR']))
            if metrics.get('CAK_STELLAR_DISP_IVAR', 0) > 0 else np.nan,
        )),
        'CAK_STELLAR_DISP_TOTAL': float(metrics['CAK_STELLAR_DISP_TOTAL']),
        'CAK_STELLAR_DISP_TOTAL_ERR': float(metrics.get(
            'CAK_STELLAR_DISP_TOTAL_ERR', np.nan,
        )),
        'CAK_CENTROID': float(metrics['CAK_CENTROID']),
        'CAK_CENTROID_ERR': float(metrics.get('CAK_CENTROID_ERR', np.nan)),
        'CAK_DEPTH': float(metrics['CAK_DEPTH']),
        'CAK_DEPTH_ERR': float(metrics.get('CAK_DEPTH_ERR', np.nan)),
        'CAK_CHI2': float(metrics['CAK_CHI2']),
        'CAK_CHI2_DOF': float(metrics['CAK_CHI2_DOF']),
        'CAK_AT_BOUND': float(metrics.get('CAK_AT_BOUND', 0.0)),
        'CAK_INST_SIG': float(cak_result.get('instrumental_sigma_kms', np.nan)),
        'CAK_TPL_LSF': float(cak_result.get('template_lsf_kms', np.nan)),
        'N_TEMPLATES': float(len(names)),
        'INPUT_FILE': '' if input_file is None else os.path.basename(input_file),
    }
    # Variable-length columns for per-template diagnostics
    row['TEMPLATE_NAMES'] = np.array(names, dtype='U64')
    row['TEMPLATE_SIGV'] = sigvs
    row['TEMPLATE_SIGV_TOTAL'] = np.asarray(
        cak_result.get('template_sigvs_total', []), dtype=float,
    )
    row['TEMPLATE_CENTROID'] = np.asarray(
        cak_result.get('template_centroids', []), dtype=float,
    )
    row['TEMPLATE_DEPTH'] = np.asarray(
        cak_result.get('template_depths', []), dtype=float,
    )
    return row


def write_single_cakfit_fits(path, cak_result, meta, pixspec=None, flux_unit=None):
    """Write single-spectrum CaK results (CAKFIT + optional CAK_PLOT)."""
    row = result_row(
        cak_result,
        input_file=meta.get('input_file'),
    )
    # Store scalar columns only in the main table; arrays go in a second extension
    scalar = {
        key: [row[key]] for key in row
        if key not in (
            'TEMPLATE_NAMES', 'TEMPLATE_SIGV', 'TEMPLATE_SIGV_TOTAL',
            'TEMPLATE_CENTROID', 'TEMPLATE_DEPTH',
        )
    }
    hdu = fits.BinTableHDU(Table(scalar), name='CAKFIT')
    hdr = hdu.header
    hdr['OBSERVER'] = str(meta.get('user', ''))[:FITS_STR_MAX]
    hdr['DATE'] = str(meta.get('date', ''))[:FITS_STR_MAX]
    if meta.get('z') is not None:
        hdr['Z'] = float(meta['z'])
    if meta.get('uncertainty_floor') is not None:
        hdr['HIERARCH CAK UNCERTFLR'] = float(meta['uncertainty_floor'])
    hdr['CAKTPL'] = str(cak_result.get('locked_template', ''))[:FITS_STR_MAX]
    _set_header_str(
        hdr, 'CAKTPLS', ', '.join(cak_result.get('all_templates') or []),
        continuation='CAKTPL2',
    )
    hdr['HIERARCH CAK AT BOUND'] = bool(cak_result.get('at_bound', False))
    if cak_result.get('instrumental_sigma_kms') is not None:
        hdr['HIERARCH CAK INST SIG'] = float(cak_result['instrumental_sigma_kms'])
    if meta.get('input_file'):
        hdr['INFILE'] = os.path.basename(meta['input_file'])[:FITS_STR_MAX]

    tpl = Table({
        'NAME': np.asarray(row['TEMPLATE_NAMES']),
        'STELLAR_DISP': np.asarray(row['TEMPLATE_SIGV'], dtype=float),
        'STELLAR_DISP_TOTAL': np.asarray(row['TEMPLATE_SIGV_TOTAL'], dtype=float),
        'CENTROID': np.asarray(row['TEMPLATE_CENTROID'], dtype=float),
        'DEPTH': np.asarray(row['TEMPLATE_DEPTH'], dtype=float),
    })
    tpl_hdu = fits.BinTableHDU(tpl, name='TEMPLATES')

    hdul = [fits.PrimaryHDU(), hdu, tpl_hdu]
    plot_hdu = build_cak_plot_hdu(
        cak_result.get('plot'), pixspec=pixspec, flux_unit=flux_unit,
    )
    if plot_hdu is not None:
        hdul.append(plot_hdu)
    fits.HDUList(hdul).writeto(path, overwrite=True)


def write_validation_cakfit_fits(path, entries, meta):
    """
    Write validation results: one binary-table extension per verr level.

    ``entries`` is a list of dicts with keys verr_kms, cak_result, expected_disp,
    path, and optionally z, pixspec, flux_unit. Each verr level also gets a
    ``CAKPLOT{N}`` spectrum/model snapshot for diagnostic plotting.
    """
    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr['OBSERVER'] = str(meta.get('user', ''))[:FITS_STR_MAX]
    hdr['DATE'] = str(meta.get('date', ''))[:FITS_STR_MAX]
    hdr['HIERARCH CAK ZLO'] = float(meta['zlo'])
    hdr['HIERARCH CAK ZHI'] = float(meta['zhi'])
    hdr['HIERARCH CAK VERRROOT'] = str(meta['verr_root'])[-FITS_STR_MAX:]
    hdr['CAKTPL'] = str(meta.get('locked_template', ''))[:FITS_STR_MAX]
    if meta.get('uncertainty_floor') is not None:
        hdr['HIERARCH CAK UNCERTFLR'] = float(meta['uncertainty_floor'])
    hdr['HIERARCH CAK SIG0'] = float(meta.get('sigma0_kms', np.nan))
    _set_header_str(
        hdr, 'CAKTPLS', ', '.join(meta.get('all_templates') or []),
        continuation='CAKTPL2',
    )

    hdul = [primary]
    for entry in entries:
        verr = float(entry['verr_kms'])
        row = result_row(
            entry['cak_result'],
            verr_kms=verr,
            expected_disp=entry.get('expected_disp'),
            input_file=entry.get('path'),
        )
        scalar = {
            key: [row[key]] for key in row
            if key not in (
                'TEMPLATE_NAMES', 'TEMPLATE_SIGV', 'TEMPLATE_SIGV_TOTAL',
                'TEMPLATE_CENTROID', 'TEMPLATE_DEPTH',
            )
        }
        extname = 'VERR%d' % int(round(verr))
        table_hdu = fits.BinTableHDU(Table(scalar), name=extname)
        table_hdu.header['VERR'] = verr
        table_hdu.header['CAKTPL'] = row['LOCKED_TEMPLATE'][:FITS_STR_MAX]
        table_hdu.header['INFILE'] = row['INPUT_FILE'][:FITS_STR_MAX]
        if entry.get('z') is not None and np.isfinite(entry['z']):
            table_hdu.header['Z'] = float(entry['z'])
        hdul.append(table_hdu)

        tpl = Table({
            'NAME': np.asarray(row['TEMPLATE_NAMES']),
            'STELLAR_DISP': np.asarray(row['TEMPLATE_SIGV'], dtype=float),
            'STELLAR_DISP_TOTAL': np.asarray(row['TEMPLATE_SIGV_TOTAL'], dtype=float),
            'CENTROID': np.asarray(row['TEMPLATE_CENTROID'], dtype=float),
            'DEPTH': np.asarray(row['TEMPLATE_DEPTH'], dtype=float),
        })
        tpl_hdu = fits.BinTableHDU(tpl, name='TPL%d' % int(round(verr)))
        hdul.append(tpl_hdu)

        plot_hdu = build_cak_plot_hdu(
            entry['cak_result'].get('plot'),
            pixspec=entry.get('pixspec'),
            flux_unit=entry.get('flux_unit'),
            name=_cakplot_extname(verr),
        )
        if plot_hdu is not None:
            plot_hdu.header['VERR'] = verr
            if entry.get('z') is not None and np.isfinite(entry['z']):
                plot_hdu.header['Z'] = float(entry['z'])
            if row['INPUT_FILE']:
                plot_hdu.header['INFILE'] = row['INPUT_FILE'][:FITS_STR_MAX]
            hdul.append(plot_hdu)

    fits.HDUList(hdul).writeto(path, overwrite=True)


def run_single(args):
    if args.spectrum is None:
        print('ERROR: spectrum file required in single-spectrum mode.', file=sys.stderr)
        return 2

    spectrum = os.path.abspath(args.spectrum)
    if not os.path.isfile(spectrum):
        print('ERROR: Spectrum not found: %s' % spectrum, file=sys.stderr)
        return 2

    tpldir = stellar_template_dir(args.stellar_template_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stem = spectrum_stem(spectrum)
    zlo_name, zhi_name = parse_redshift_bin_from_name(spectrum)
    out_fits = (
        os.path.abspath(args.output)
        if args.output
        else os.path.join(
            output_dir,
            default_fitresults_filename(zlo=zlo_name, zhi=zhi_name, stem=stem),
        )
    )
    if zlo_name is not None and zhi_name is not None:
        out_png = os.path.join(
            output_dir, 'cak_fitresults_%s.png' % redshift_bin_tag(zlo_name, zhi_name),
        )
    else:
        out_png = os.path.join(output_dir, 'cak_fitresults_%s.png' % stem)

    if not args.clobber and os.path.isfile(out_fits):
        print('Output exists (use --clobber): %s' % out_fits)
        return 0

    spres, z, pixspec, input_format = load_spectrum(
        spectrum, z_override=args.z, uncertainty_floor=args.uncertainty_floor,
    )
    print('Start Ca K analysis.')
    print('spectra file name: %s' % os.path.basename(spectrum))
    print('input format: %s' % input_format)
    print('z: %f' % z)
    print('rest-frame pixel size: %.4f A' % pixspec)
    print('uncertainty floor: %g' % args.uncertainty_floor)
    print('template directory: %s' % tpldir)
    if args.cak_template:
        print('locked template: %s' % args.cak_template)
    else:
        print('locked template: auto (best χ²)')

    if not cak_is_measurable(spres):
        print(
            'ERROR: Ca K is not covered (requires rest-frame coverage near %.1f A).'
            % CAK_LAB_WAVE
        )
        return 1

    cak_result = measure_cak_absorption(
        spres,
        template_dir=tpldir,
        z=z,
        locked_template_name=args.cak_template,
    )
    if cak_result is None:
        print('ERROR: Ca K fit failed.')
        return 1

    print(
        'Ca K fit complete (locked template: %s; ensemble n=%d)'
        % (cak_result['locked_template'], len(cak_result['all_templates']))
    )
    print_cak_summary(cak_result['metrics'], cak_meta=cak_result)

    meta = {
        'user': getpass.getuser(),
        'date': datetime.now(timezone.utc).isoformat(),
        'input_file': spectrum,
        'uncertainty_floor': args.uncertainty_floor,
        'z': z,
    }
    ylabel = flux_label(infer_fscale(spres))
    write_single_cakfit_fits(
        out_fits, cak_result, meta, pixspec=pixspec, flux_unit=ylabel,
    )
    print('Ca K fit results saved: %s' % out_fits)

    if not args.no_plot:
        plot_cak(cak_result, ylabel, pixspec, out_png)
    return 0


def run_validate(args):
    if args.verr_root is None or args.zlo is None or args.zhi is None:
        print(
            'ERROR: --validate requires --verr-root, --zlo, and --zhi.',
            file=sys.stderr,
        )
        return 2
    if args.zlo >= args.zhi:
        print('ERROR: --zlo must be < --zhi.', file=sys.stderr)
        return 2

    verr_root = os.path.abspath(args.verr_root)
    tpldir = stellar_template_dir(args.stellar_template_dir)
    tag = redshift_bin_tag(args.zlo, args.zhi)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    out_fits = (
        os.path.abspath(args.output)
        if args.output
        else os.path.join(
            output_dir,
            default_fitresults_filename(zlo=args.zlo, zhi=args.zhi),
        )
    )
    if not args.clobber and os.path.isfile(out_fits):
        print('Output exists (use --clobber): %s' % out_fits)
        return 0

    try:
        stacks = discover_validation_stacks(verr_root, args.zlo, args.zhi)
    except FileNotFoundError as exc:
        print('ERROR: %s' % exc, file=sys.stderr)
        return 1

    print('Start Ca K validation.')
    print('verr root: %s' % verr_root)
    print('redshift bin: %s' % tag)
    print('uncertainty floor: %g' % args.uncertainty_floor)
    print('template directory: %s' % tpldir)
    print('stacks:')
    for item in stacks:
        print('  verr=%5.0f  %s' % (item['verr_kms'], item['path']))

    # Reference (verr0): auto-pick best template unless --cak-template given
    ref = next(item for item in stacks if item['verr_kms'] == 0.0)
    spres0, z0, pixspec0, _fmt = load_spectrum(
        ref['path'], uncertainty_floor=args.uncertainty_floor,
    )
    if not cak_is_measurable(spres0):
        print('ERROR: Ca K not measurable in reference stack.', file=sys.stderr)
        return 1

    print('Fitting reference verr0 to lock template…')
    ref_result = measure_cak_absorption(
        spres0,
        template_dir=tpldir,
        z=z0,
        locked_template_name=args.cak_template,
        sigv_min=SIGV_MIN,
    )
    if ref_result is None:
        print('ERROR: Ca K fit failed on reference stack.', file=sys.stderr)
        return 1

    locked = ref_result['locked_template']
    sigma0 = float(ref_result['metrics']['CAK_STELLAR_DISP'])
    err0 = float(ref_result['metrics'].get('CAK_STELLAR_DISP_ERR', np.nan))
    chi2_0 = float(ref_result['metrics'].get('CAK_CHI2', np.nan))
    n_tpl0 = len(ref_result.get('all_templates') or [])
    n_ens0 = int(ref_result.get('n_ensemble', n_tpl0))
    print(
        'Locked template: %s  (σ0 = %.1f +/- %s km/s   χ² = %.1f   n_tpl = %d, %d ensemble)'
        % (
            locked, sigma0,
            ('%.1f' % err0) if np.isfinite(err0) else 'nan',
            chi2_0, n_tpl0, n_ens0,
        )
    )

    entries = []
    for item in stacks:
        verr = float(item['verr_kms'])
        sigv_min = max(float(SIGV_MIN), verr)
        print('Fitting verr=%.0f …  (σ* lower bound = %.0f km/s)' % (verr, sigv_min))
        if verr == 0.0:
            cak_result = ref_result
            z = z0
            pixspec = pixspec0
            spres = spres0
        else:
            spres, z, pixspec, _fmt = load_spectrum(
                item['path'], uncertainty_floor=args.uncertainty_floor,
            )
            if not cak_is_measurable(spres):
                print('WARNING: Ca K not measurable for verr=%.0f; skipping.' % verr)
                continue
            cak_result = measure_cak_absorption(
                spres,
                template_dir=tpldir,
                z=z,
                locked_template_name=locked,
                sigv_min=sigv_min,
            )
            if cak_result is None:
                print('WARNING: Ca K fit failed for verr=%.0f; skipping.' % verr)
                continue

        expected = expected_dispersion_kms(sigma0, verr)
        disp = float(cak_result['metrics']['CAK_STELLAR_DISP'])
        err = float(cak_result['metrics'].get('CAK_STELLAR_DISP_ERR', np.nan))
        err_txt = ('%.1f' % err) if np.isfinite(err) else 'nan'
        chi2 = float(cak_result['metrics'].get('CAK_CHI2', np.nan))
        n_tpl = len(cak_result.get('all_templates') or [])
        n_ens = int(cak_result.get('n_ensemble', n_tpl))
        print(
            '  locked σ* = %.1f +/- %s km/s   expected = %.1f km/s'
            '   χ² = %.1f   n_tpl = %d, %d ensemble'
            % (disp, err_txt, expected, chi2, n_tpl, n_ens)
        )
        entries.append({
            'verr_kms': verr,
            'path': item['path'],
            'cak_result': cak_result,
            'expected_disp': expected,
            'z': float(z),
            'pixspec': float(pixspec),
            'flux_unit': flux_label(infer_fscale(spres)),
        })

    if not entries:
        print('ERROR: No successful Ca K fits in validation series.', file=sys.stderr)
        return 1

    meta = {
        'user': getpass.getuser(),
        'date': datetime.now(timezone.utc).isoformat(),
        'zlo': float(args.zlo),
        'zhi': float(args.zhi),
        'verr_root': verr_root,
        'locked_template': locked,
        'all_templates': ref_result.get('all_templates'),
        'uncertainty_floor': args.uncertainty_floor,
        'sigma0_kms': sigma0,
    }
    write_validation_cakfit_fits(out_fits, entries, meta)
    print('Fit results saved: %s' % out_fits)
    print('Extensions: %s' % ', '.join(
        'VERR%d,TPL%d,%s' % (
            int(round(e['verr_kms'])),
            int(round(e['verr_kms'])),
            _cakplot_extname(e['verr_kms']),
        )
        for e in entries
    ))

    if not args.no_plot:
        out_png = os.path.join(output_dir, 'cak_verr_diagnostic_%s.png' % tag)
        plot_entries = []
        for entry in entries:
            cak_result = entry['cak_result']
            metrics = dict(cak_result['metrics'])
            n_tpl = len(cak_result.get('all_templates') or [])
            if n_tpl > 0:
                metrics.setdefault('N_TEMPLATES', float(n_tpl))
            if entry.get('expected_disp') is not None:
                metrics.setdefault('EXPECTED_DISP', float(entry['expected_disp']))
            plot_entries.append({
                'verr_kms': float(entry['verr_kms']),
                'snapshot': {
                    'plot': cak_result['plot'],
                    'metrics': metrics,
                    'z': float(entry['z']),
                    'zlo': float(args.zlo),
                    'zhi': float(args.zhi),
                    'pixspec': float(entry.get('pixspec', np.nan)),
                    'flux_unit': entry.get('flux_unit'),
                },
                'path': entry.get('path') or out_fits,
            })
        plot_cak_verr_diagnostic(
            plot_entries,
            out_png,
            ylabel='Relative Flux and Fit Residual',
        )
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.validate:
        return run_validate(args)
    return run_single(args)


if __name__ == '__main__':
    sys.exit(main())
