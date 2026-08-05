#!/usr/bin/env python3
"""
Build Ca II K stellar absorption templates 

Reads a rest-frame stellar spectrum (ASCII or FITS), divides out a pseudo-continuum
fit to line-free regions blueward of Ca K and redward of Ca H, and writes a CSV
template plus an entry in templates.manifest.csv.

FITS support includes DESI-style WAVE/FLUX HDUs, binary tables (including UVES-POP
SPECTRUM extensions), and MILES-style PRIMARY HDUs with CRVAL1/CDELT1/CRPIX1 WCS
keywords. High-resolution inputs are read only over the Ca K region and resampled
before writing compact CSV templates.

Example
-------
  python build_stellar_templates.py miles_spectrum.fits \\
      --name miles_1234_K3III --label "MILES 1234 K3III" \\
      --spectral-type K3III --fe-h 0.0 --source "MILES v3.1 ID 1234"

  python build_stellar_templates.py --batch stars.csv

Batch CSV columns: input_path,name,label,spectral_type,fe_h,source
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

_PY_DIR = Path(__file__).resolve().parents[1] / 'py'
if _PY_DIR.is_dir() and str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table

from qsosigma.cak_metrics import (
    CAH_LAB_WAVE,
    CAK_LAB_WAVE,
    MANIFEST_FILENAME,
    PLOT_WAVE_HI,
    PLOT_WAVE_LO,
    default_stellar_template_dir,
    load_cak_template_catalog,
)

# Wavelength range stored in template CSVs: Ca K fit/plot window plus a small margin.
DEFAULT_TEMPLATE_PAD_A = 10.0
DEFAULT_WAVE_RANGE = (
    PLOT_WAVE_LO - DEFAULT_TEMPLATE_PAD_A,
    PLOT_WAVE_HI + DEFAULT_TEMPLATE_PAD_A,
)
DEFAULT_BLUE_CONT = (3820.0, 3880.0)
DEFAULT_RED_CONT = (3990.0, 4050.0)
DEFAULT_READ_PAD_A = 5.0
DEFAULT_OUTPUT_STEP_A = 0.2


def _parse_range(text: str) -> Tuple[float, float]:
    parts = [float(x.strip()) for x in text.split(',')]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise argparse.ArgumentTypeError('Expected lo,hi with lo < hi, got %r' % text)
    return parts[0], parts[1]


def _is_fits(path: str) -> bool:
    with open(path, 'rb') as handle:
        return handle.read(6) == b'SIMPLE'


def _as_1d(data) -> np.ndarray:
    return np.asarray(data, dtype=float).reshape(-1)


def _squeeze_flux(data) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and 1 in arr.shape:
        return arr.reshape(-1)
    raise ValueError('Expected a 1D spectrum or a 2D array with one axis length 1.')


def _wavelength_from_wcs_header(header, n_pix: int) -> Optional[np.ndarray]:
    """Build a linear wavelength axis from standard FITS WCS keywords."""
    crval = header.get('CRVAL1')
    cdelt = header.get('CDELT1')
    if crval is None or cdelt is None:
        return None
    crpix = float(header.get('CRPIX1', 1.0))
    index = np.arange(n_pix, dtype=float)
    return crval + (index + 1.0 - crpix) * float(cdelt)


def _wcs_pixel_bounds(header, wave_lo: float, wave_hi: float) -> Tuple[int, int]:
    """Return inclusive pixel indices covering a wavelength interval."""
    crval = float(header['CRVAL1'])
    cdelt = float(header['CDELT1'])
    crpix = float(header.get('CRPIX1', 1.0))
    if cdelt == 0:
        raise ValueError('CDELT1 is zero in FITS WCS header.')
    i0 = int(np.floor((wave_lo - crval) / cdelt + crpix - 1.0))
    i1 = int(np.ceil((wave_hi - crval) / cdelt + crpix - 1.0))
    naxis = int(header.get('NAXIS1', i1 + 1))
    i0 = max(0, i0)
    i1 = min(naxis - 1, i1)
    if i1 < i0:
        raise ValueError(
            'Requested wavelength range %.3f–%.3f A is outside the FITS WCS coverage.'
            % (wave_lo, wave_hi)
        )
    return i0, i1


def _wavelength_from_wcs_slice(header, i0: int, i1: int) -> np.ndarray:
    crval = float(header['CRVAL1'])
    cdelt = float(header['CDELT1'])
    crpix = float(header.get('CRPIX1', 1.0))
    index = np.arange(i0, i1 + 1, dtype=float)
    return crval + (index + 1.0 - crpix) * cdelt


def _read_range_for_build(
    wave_range: Sequence[float],
    blue_range: Sequence[float],
    red_range: Sequence[float],
    read_pad: float = DEFAULT_READ_PAD_A,
) -> Tuple[float, float]:
    """Return the wavelength interval to load before pseudo-continuum fitting."""
    return (
        min(wave_range[0], blue_range[0]) - read_pad,
        max(wave_range[1], red_range[1]) + read_pad,
    )


def _resample_uniform(
    wave: np.ndarray,
    values: np.ndarray,
    step_angstrom: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample onto a uniform wavelength grid if step_angstrom is set and finer than input."""
    wave = np.asarray(wave, dtype=float)
    values = np.asarray(values, dtype=float)
    if step_angstrom is None or step_angstrom <= 0:
        return wave, values
    if wave.size < 2:
        return wave, values
    step = float(step_angstrom)
    native_step = float(np.median(np.diff(wave)))
    if not np.isfinite(native_step) or native_step >= step:
        return wave, values
    grid = np.arange(wave[0], wave[-1] + 0.5 * step, step)
    if grid.size < 2:
        return wave, values
    resampled = np.interp(grid, wave, values, left=np.nan, right=np.nan)
    return grid, resampled


def _read_primary_wcs_spectrum(hdu, read_range: Optional[Sequence[float]] = None):
    flux_all = _squeeze_flux(hdu.data)
    header = hdu.header
    if read_range is None:
        wave = _wavelength_from_wcs_header(header, len(flux_all))
        if wave is None:
            return None
        return wave, flux_all

    i0, i1 = _wcs_pixel_bounds(header, read_range[0], read_range[1])
    if hasattr(hdu, 'section'):
        flux = np.asarray(hdu.section[i0:i1 + 1], dtype=float)
    else:
        flux = flux_all[i0:i1 + 1]
    wave = _wavelength_from_wcs_slice(header, i0, i1)
    return wave, flux


def _read_binary_table_spectrum(hdu, read_range: Optional[Sequence[float]] = None):
    if hdu.data is None or not getattr(hdu.data, 'dtype', None) or not hdu.data.dtype.names:
        return None
    names = hdu.data.dtype.names
    wave_key = next(
        (k for k in names if k.upper() in ('WAVE', 'WAVELENGTH', 'LAMBDA', 'LBD')),
        None,
    )
    flux_key = next(
        (k for k in names if k.upper() in ('FLUX', 'F', 'INTENSITY', 'FLUX_DENSITY')),
        None,
    )
    if wave_key is None or flux_key is None:
        return None

    wave_all = _as_1d(hdu.data[wave_key][0])
    flux_all = _as_1d(hdu.data[flux_key][0])
    if read_range is None:
        return wave_all, flux_all

    mask = (wave_all >= read_range[0]) & (wave_all <= read_range[1])
    if int(np.sum(mask)) < 2:
        raise ValueError(
            'Requested wavelength range %.3f–%.3f A is outside the binary-table spectrum.'
            % (read_range[0], read_range[1])
        )
    return wave_all[mask], flux_all[mask]


def _read_fits_spectrum(hdul, read_range: Optional[Sequence[float]] = None):
    """Read wavelength and flux arrays from a FITS file."""
    if 'WAVE' in hdul and 'FLUX' in hdul:
        wave = _as_1d(hdul['WAVE'].data)
        flux = _as_1d(hdul['FLUX'].data)
        if read_range is not None:
            mask = (wave >= read_range[0]) & (wave <= read_range[1])
            if int(np.sum(mask)) < 2:
                raise ValueError(
                    'Requested wavelength range %.3f–%.3f A is outside WAVE/FLUX HDUs.'
                    % (read_range[0], read_range[1])
                )
            wave, flux = wave[mask], flux[mask]
        return wave, flux

    primary = hdul[0]
    if primary.data is not None:
        primary_spec = _read_primary_wcs_spectrum(primary, read_range=read_range)
        if primary_spec is not None:
            return primary_spec

    for hdu in hdul:
        if hdu is primary:
            continue
        table_spec = _read_binary_table_spectrum(hdu, read_range=read_range)
        if table_spec is not None:
            return table_spec

    raise ValueError(
        'Could not find spectral data. Expected WAVE/FLUX HDUs, a binary table with '
        'wavelength and flux columns, or a PRIMARY HDU with CRVAL1/CDELT1 WCS keywords '
        '(MILES / UVES-POP format).'
    )


def _read_fits_metadata(path: str) -> dict:
    """Extract provenance metadata from known high-resolution library formats."""
    meta = {}
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header
        if header.get('OBJECT'):
            meta['object'] = str(header['OBJECT']).strip()
        if header.get('INSTRUME'):
            meta['instrument'] = str(header['INSTRUME']).strip()
        if 'SPECTRUM' in hdul and hdul['SPECTRUM'].data is not None:
            names = hdul['SPECTRUM'].data.dtype.names or ()
            row_index = 0

            def _field(name):
                if name not in names:
                    return None
                value = hdul['SPECTRUM'].data[name][row_index]
                if isinstance(value, (bytes, np.bytes_)):
                    value = value.decode('utf-8', errors='ignore').strip()
                if isinstance(value, str):
                    value = value.strip('\x00').strip()
                return value

            spclass = _field('SPCLASS')
            if spclass:
                meta['spectral_type'] = str(spclass)
            fe_h = _field('FE_H')
            if fe_h is not None and np.isfinite(float(fe_h)):
                meta['fe_h'] = '%.2f' % float(fe_h)
            resolution = _field('RESOLUTION')
            if resolution is not None and np.isfinite(float(resolution)):
                meta['resolution'] = float(resolution)
            objname = _field('OBJNAME')
            if objname:
                meta['object'] = str(objname)
    return meta


def _default_source_string(path: str, meta: dict) -> str:
    instrument = meta.get('instrument', '')
    obj = meta.get('object', os.path.splitext(os.path.basename(path))[0])
    resolution = meta.get('resolution')
    if instrument.upper() == 'UVES' or 'R80K' in os.path.basename(path).upper():
        if resolution is not None and np.isfinite(resolution):
            return 'UVES-POP %s (R~%.0f)' % (obj, resolution)
        return 'UVES-POP %s' % obj
    return os.path.basename(path)


def read_stellar_spectrum(
    path: str,
    read_range: Optional[Sequence[float]] = None,
) -> Table:
    """Load a rest-frame stellar spectrum with columns lbd, f."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError('Spectrum not found: %s' % path)

    if _is_fits(path):
        with fits.open(path, memmap=True) as hdul:
            wave, flux = _read_fits_spectrum(hdul, read_range=read_range)
    else:
        try:
            table = ascii.read(path, format='no_header', comment='#')
        except Exception:
            table = ascii.read(path, format='csv', comment='#')
        if len(table.colnames) < 2:
            raise ValueError(
                'ASCII spectrum must have at least wavelength and flux columns.'
            )
        wave = np.asarray(table.columns[0], dtype=float)
        flux = np.asarray(table.columns[1], dtype=float)
        if read_range is not None:
            mask = (wave >= read_range[0]) & (wave <= read_range[1])
            wave, flux = wave[mask], flux[mask]

    order = np.argsort(wave)
    return Table({
        'lbd': np.asarray(wave[order], dtype=float),
        'f': np.asarray(flux[order], dtype=float),
    })


def fit_pseudo_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    blue_range: Sequence[float],
    red_range: Sequence[float],
) -> np.ndarray:
    """Fit a power law to line-free blue and red continuum regions."""
    cont_mask = (
        ((wave >= blue_range[0]) & (wave <= blue_range[1]))
        | ((wave >= red_range[0]) & (wave <= red_range[1]))
    )
    if int(np.sum(cont_mask)) < 4:
        raise ValueError(
            'Not enough continuum pixels in blue %s–%s or red %s–%s.'
            % (blue_range[0], blue_range[1], red_range[0], red_range[1])
        )
    w = wave[cont_mask]
    f = flux[cont_mask]
    positive = f > 0
    if int(np.sum(positive)) < 4:
        raise ValueError('Continuum regions must contain positive flux values.')

    logw = np.log(w[positive])
    logf = np.log(f[positive])
    slope, intercept = np.polyfit(logw, logf, 1)
    continuum = np.exp(intercept + slope * np.log(np.asarray(wave, dtype=float)))
    continuum = np.where(continuum > 0, continuum, np.nan)
    return continuum


def build_absorption_template(
    wave: np.ndarray,
    flux: np.ndarray,
    wave_range: Sequence[float] = DEFAULT_WAVE_RANGE,
    blue_range: Sequence[float] = DEFAULT_BLUE_CONT,
    red_range: Sequence[float] = DEFAULT_RED_CONT,
    output_step: Optional[float] = DEFAULT_OUTPUT_STEP_A,
) -> Table:
    """Convert a stellar spectrum to absorption = 1 - flux / pseudo-continuum."""
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if wave.size < 10:
        raise ValueError('Input spectrum has too few pixels to build a template.')
    if wave[0] > blue_range[0] or wave[-1] < red_range[1]:
        raise ValueError(
            'Input spectrum (%.3f–%.3f A) must cover the pseudo-continuum regions '
            'blue %.1f–%.1f and red %.1f–%.1f A.'
            % (wave[0], wave[-1], blue_range[0], blue_range[1], red_range[0], red_range[1])
        )

    continuum = fit_pseudo_continuum(wave, flux, blue_range, red_range)
    save_mask = (wave >= wave_range[0]) & (wave <= wave_range[1])
    wave = wave[save_mask]
    flux = flux[save_mask]
    continuum = continuum[save_mask]
    if wave.size < 10:
        raise ValueError(
            'Input spectrum does not cover the requested template range %.1f–%.1f A.'
            % (wave_range[0], wave_range[1])
        )

    valid = np.isfinite(continuum) & (continuum > 0) & (flux > 0)
    absorption = np.zeros_like(flux)
    absorption[valid] = 1.0 - flux[valid] / continuum[valid]
    absorption = np.clip(absorption, 0.0, 1.0)
    wave, absorption = _resample_uniform(wave, absorption, output_step)
    return Table({
        'wavelength': np.asarray(wave, dtype=float),
        'absorption': np.asarray(absorption, dtype=float),
    })


def manifest_path(template_dir: str) -> str:
    return os.path.join(template_dir, MANIFEST_FILENAME)


def read_manifest(template_dir: str) -> List[dict]:
    path = manifest_path(template_dir)
    if not os.path.isfile(path):
        return []
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def write_manifest(template_dir: str, rows: Iterable[dict]) -> None:
    fieldnames = [
        'name', 'filename', 'label', 'spectral_type', 'fe_h', 'source', 'enabled',
    ]
    path = manifest_path(template_dir)
    os.makedirs(template_dir, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def upsert_manifest_entry(template_dir: str, entry: dict) -> None:
    rows = read_manifest(template_dir)
    replaced = False
    for row in rows:
        if row.get('name') == entry['name']:
            row.update(entry)
            replaced = True
            break
    if not replaced:
        rows.append(entry)
    write_manifest(template_dir, rows)


def build_one_template(
    input_path: str,
    name: str,
    label: str,
    template_dir: str,
    spectral_type: str = '',
    fe_h: str = '',
    source: str = '',
    wave_range: Sequence[float] = DEFAULT_WAVE_RANGE,
    blue_range: Sequence[float] = DEFAULT_BLUE_CONT,
    red_range: Sequence[float] = DEFAULT_RED_CONT,
    read_pad: float = DEFAULT_READ_PAD_A,
    output_step: Optional[float] = DEFAULT_OUTPUT_STEP_A,
    subdir: str = 'empirical',
    enabled: bool = True,
) -> str:
    input_path = os.path.abspath(input_path)
    file_meta = _read_fits_metadata(input_path) if _is_fits(input_path) else {}
    if not spectral_type:
        spectral_type = str(file_meta.get('spectral_type', '') or '')
    if not fe_h:
        fe_h = str(file_meta.get('fe_h', '') or '')
    if not source:
        source = _default_source_string(input_path, file_meta)

    read_range = _read_range_for_build(
        wave_range, blue_range, red_range, read_pad=read_pad,
    )
    sp = read_stellar_spectrum(input_path, read_range=read_range)
    template = build_absorption_template(
        np.asarray(sp['lbd'], dtype=float),
        np.asarray(sp['f'], dtype=float),
        wave_range=wave_range,
        blue_range=blue_range,
        red_range=red_range,
        output_step=output_step,
    )

    rel_dir = subdir.strip('/') if subdir else ''
    filename = os.path.join(rel_dir, '%s.csv' % name) if rel_dir else '%s.csv' % name
    out_path = os.path.join(template_dir, filename)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ascii.write(template, out_path, format='csv', overwrite=True)

    upsert_manifest_entry(template_dir, {
        'name': name,
        'filename': filename.replace('\\', '/'),
        'label': label,
        'spectral_type': spectral_type,
        'fe_h': fe_h,
        'source': source,
        'enabled': 'true' if enabled else 'false',
    })
    return out_path


def _batch_row_get(row, key, default=''):
    colnames = getattr(row, 'colnames', ())
    if key not in colnames:
        return default
    value = row[key]
    if value is None or np.ma.is_masked(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none', '--'):
        return default
    return text


def build_from_batch_table(batch_path: str, template_dir: str, **kwargs) -> List[str]:
    table = ascii.read(batch_path, format='csv')
    required = {'input_path', 'name', 'label'}
    missing = required - set(table.colnames)
    if missing:
        raise ValueError('Batch table missing columns: %s' % ', '.join(sorted(missing)))

    outputs = []
    default_subdir = kwargs.get('subdir', 'empirical')
    for row in table:
        outputs.append(build_one_template(
            input_path=_batch_row_get(row, 'input_path'),
            name=_batch_row_get(row, 'name'),
            label=_batch_row_get(row, 'label'),
            template_dir=template_dir,
            spectral_type=_batch_row_get(row, 'spectral_type'),
            fe_h=_batch_row_get(row, 'fe_h'),
            source=_batch_row_get(row, 'source'),
            subdir=_batch_row_get(row, 'subdir', default_subdir) or 'empirical',
            enabled=_batch_row_get(row, 'enabled', 'true').lower() in (
                '1', 'true', 'yes', 'y',
            ),
            wave_range=kwargs.get('wave_range', DEFAULT_WAVE_RANGE),
            blue_range=kwargs.get('blue_range', DEFAULT_BLUE_CONT),
            red_range=kwargs.get('red_range', DEFAULT_RED_CONT),
            read_pad=kwargs.get('read_pad', DEFAULT_READ_PAD_A),
            output_step=kwargs.get('output_step', DEFAULT_OUTPUT_STEP_A),
        ))
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build Ca II K stellar absorption templates.',
    )
    parser.add_argument(
        'input_spectrum',
        nargs='?',
        help='Rest-frame stellar spectrum (FITS or ASCII: wavelength flux)',
    )
    parser.add_argument('--name', help='Template identifier used in outputs and manifest')
    parser.add_argument('--label', help='Human-readable template label')
    parser.add_argument('--spectral-type', default='', help='Spectral type, e.g. K3III')
    parser.add_argument('--fe-h', default='', help='Metallicity [Fe/H]')
    parser.add_argument('--source', default='', help='Provenance string, e.g. MILES ID 1234')
    parser.add_argument(
        '--output-dir',
        default=default_stellar_template_dir(),
        help='Template directory (default: data/stellar_templates)',
    )
    parser.add_argument(
        '--subdir',
        default='empirical',
        help='Subdirectory under output-dir for the CSV (default: empirical)',
    )
    parser.add_argument(
        '--wave-range',
        type=_parse_range,
        default=DEFAULT_WAVE_RANGE,
        help='Template wavelength range stored in the CSV (default: Ca K plot window +/- %.0f A)'
        % DEFAULT_TEMPLATE_PAD_A,
    )
    parser.add_argument(
        '--blue-cont',
        type=_parse_range,
        default=DEFAULT_BLUE_CONT,
        help='Blue pseudo-continuum region, blueward of Ca K (default: 3820,3880)',
    )
    parser.add_argument(
        '--red-cont',
        type=_parse_range,
        default=DEFAULT_RED_CONT,
        help='Red pseudo-continuum region, redward of Ca H (default: 3990,4050)',
    )
    parser.add_argument(
        '--read-pad',
        type=float,
        default=DEFAULT_READ_PAD_A,
        help='Extra Angstrom padding when reading high-resolution input spectra (default: %.1f)'
        % DEFAULT_READ_PAD_A,
    )
    parser.add_argument(
        '--output-step',
        type=float,
        default=DEFAULT_OUTPUT_STEP_A,
        help='Resample stored templates to this uniform Angstrom step (default: %.2f; 0 keeps native step)'
        % DEFAULT_OUTPUT_STEP_A,
    )
    parser.add_argument(
        '--batch',
        help='CSV with columns input_path,name,label[,spectral_type,fe_h,source,subdir,enabled]',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List templates currently enabled for run_cakfit.py and exit',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    template_dir = os.path.abspath(args.output_dir)

    if args.list:
        templates = load_cak_template_catalog(template_dir)
        if not templates:
            print('No enabled templates found in %s' % template_dir)
            return 1
        for template in templates:
            print('%s  %s  (%s)' % (template.name, template.filename, template.label))
        return 0

    if args.batch:
        outputs = build_from_batch_table(
            args.batch,
            template_dir,
            subdir=args.subdir,
            wave_range=args.wave_range,
            blue_range=args.blue_cont,
            red_range=args.red_cont,
            read_pad=args.read_pad,
            output_step=None if args.output_step <= 0 else args.output_step,
        )
        for path in outputs:
            print('Wrote %s' % path)
        print('Updated manifest: %s' % manifest_path(template_dir))
        return 0

    if args.input_spectrum is None or args.name is None or args.label is None:
        print('ERROR: provide input_spectrum, --name, and --label (or use --batch).', file=sys.stderr)
        return 2

    out_path = build_one_template(
        input_path=args.input_spectrum,
        name=args.name,
        label=args.label,
        template_dir=template_dir,
        spectral_type=args.spectral_type,
        fe_h=args.fe_h,
        source=args.source,
        wave_range=args.wave_range,
        blue_range=args.blue_cont,
        red_range=args.red_cont,
        read_pad=args.read_pad,
        output_step=None if args.output_step <= 0 else args.output_step,
        subdir=args.subdir,
        enabled=True,
    )
    print('Wrote %s (%d pixels, %.3f–%.3f A)' % (
        out_path,
        len(ascii.read(out_path, format='csv')),
        args.wave_range[0],
        args.wave_range[1],
    ))
    print('Ca K lab wavelength: %.3f A' % CAK_LAB_WAVE)
    print('Ca H lab wavelength: %.3f A (included in template, excluded from fit windows)' % CAH_LAB_WAVE)
    print('Updated manifest: %s' % manifest_path(template_dir))
    print('Enabled templates:')
    for template in load_cak_template_catalog(template_dir):
        print('  %s (%s)' % (template.name, template.filename))
    return 0


if __name__ == '__main__':
    sys.exit(main())
