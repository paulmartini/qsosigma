"""Read input spectra from FITS or ASCII files."""

import os

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table

REDSHIFT_KEYS = ('Z', 'REDSHIFT', 'Z_OBJ', 'Z_QSO', 'ZSTACK', 'HELIOZ', 'BLSZ')
STACKINFO_Z_KEYS = ('Z', 'ZREF', 'ZMEAN')

def load_spectrum(path, z_override=None, uncertainty_floor=0.01):
    """
    Load a spectrum from FITS or ASCII.

    Parameters
    ----------
    uncertainty_floor : float
        Lower limit on the fractional uncertainty per pixel. Combined in
        quadrature with the formal error (default: 0.01).

    Returns
    -------
    spres : astropy.table.Table
        Columns lbd, f, ferr in the rest frame.
    z : float
        Redshift used for the conversion (ASCII) or metadata (FITS).
    pixspec : float
        Median rest-frame pixel size in Angstroms.
    fmt : str
        'FITS' or 'ASCII'.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError('Spectrum file not found: %s' % path)

    if is_fits_file(path):
        fmt = 'FITS'
        with fits.open(path) as hdul:
            z_meta = read_redshift_from_fits(hdul)
            z = z_override if z_override is not None else z_meta
            spres, z = _read_spectrum_from_hdul(hdul, z)
    else:
        fmt = 'ASCII'
        spres, z = read_spectrum_ascii(path, z_override)

    spres['ferr'] = apply_uncertainty_floor(
        spres['f'], spres['ferr'], uncertainty_floor,
    )

    order = np.argsort(np.asarray(spres['lbd'], dtype=float))
    spres = spres[order]
    pixspec = float(np.median(np.diff(np.asarray(spres['lbd'], dtype=float))))
    return spres, z, pixspec, fmt


def is_fits_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.gz':
        ext = os.path.splitext(path[:-3])[1].lower()
    if ext in ('.fits', '.fit'):
        return True
    with open(path, 'rb') as fhandle:
        return fhandle.read(6) == b'SIMPLE'


def spectrum_stem(path):
    """Return the input filename without path or common spectrum extensions."""
    name = os.path.basename(path)
    for suffix in ('.fits.gz', '.fit.gz', '.fits', '.fit', '.txt', '.dat', '.gz'):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def output_paths(stem, output_dir='.'):
    """Build output file paths from the spectrum stem and output directory."""
    return {
        'results': os.path.join(output_dir, '%s_linefit.fits' % stem),
        'line': os.path.join(output_dir, '%s_line.png' % stem),
        'cak': os.path.join(output_dir, '%s_cak.png' % stem),
    }

def infer_fscale(spres):
    """Choose plot flux scaling based on typical flux level."""
    lbd = np.asarray(spres['lbd'], dtype=float)
    cakmask = (lbd >= 3900) & (lbd <= 4000)
    ref = spres[cakmask] if np.any(cakmask) else spres
    median_flux = float(np.median(np.asarray(ref['f'], dtype=float)))
    if median_flux > 1.0:
        return 1.0
    return 1e17


def flux_label(fscale):
    if fscale == 1.0:
        return r'Flux (erg/s/cm$^2$/$\AA$)'
    return r'Flux ($10^{-17}$ erg/s/cm$^2$/$\AA$)'


def apply_uncertainty_floor(flux, ferr, uncertainty_floor):
    """
    Apply a lower limit on the fractional uncertainty per pixel.

    The floor uncertainty is combined in quadrature with the formal error:
    ferr = sqrt(ferr^2 + (uncertainty_floor * |flux|)^2)
    """
    if uncertainty_floor is None or uncertainty_floor <= 0:
        return ferr
    floor_err = uncertainty_floor * np.abs(np.asarray(flux, dtype=float))
    formal_err = np.asarray(ferr, dtype=float)
    return np.sqrt(formal_err ** 2 + floor_err ** 2)


def _header_redshift(header):
    for key in REDSHIFT_KEYS:
        if key in header:
            return float(header[key])
    return None


def _table_redshift(hdul):
    for extname in ('STACKINFO', 'SPECOBJ', 'FIBERMAP'):
        if extname not in hdul:
            continue
        data = hdul[extname].data
        if data is None:
            continue
        names = data.dtype.names or ()
        for key in STACKINFO_Z_KEYS:
            if key in names:
                value = data[key][0]
                if hasattr(value, 'item'):
                    value = value.item()
                return float(value)
    return None


def read_redshift_from_fits(hdul):
    for hdu in hdul:
        if not hasattr(hdu, 'header'):
            continue
        z = _header_redshift(hdu.header)
        if z is not None:
            return z
    return _table_redshift(hdul)


def _as_1d_array(data):
    return np.asarray(data, dtype=float).reshape(-1)


def _flux_from_hdu(hdul):
    if 'FLUX' in hdul:
        return _as_1d_array(hdul['FLUX'].data)
    for hdu in hdul:
        if hdu.data is None:
            continue
        if getattr(hdu, 'name', '') in ('PRIMARY', ''):
            arr = np.asarray(hdu.data, dtype=float)
            if arr.ndim >= 1 and arr.size > 1:
                return _as_1d_array(arr)
    raise ValueError('Could not find flux data in FITS file.')


def _ivar_from_hdu(hdul, nflux):
    if 'IVAR' in hdul:
        ivar = _as_1d_array(hdul['IVAR'].data)
        return np.where(ivar > 0, 1.0 / np.sqrt(ivar), np.nan)
    if 'SIGMA' in hdul:
        sigma = _as_1d_array(hdul['SIGMA'].data)
        return np.where(sigma > 0, sigma, np.nan)
    if 'ERR' in hdul:
        err = _as_1d_array(hdul['ERR'].data)
        return np.where(err > 0, err, np.nan)
    raise ValueError(
        'Could not find flux uncertainty in FITS file (expected IVAR, SIGMA, or ERR).'
    )


def _wave_from_hdu(hdul, nflux):
    if 'WAVE' in hdul:
        wave = _as_1d_array(hdul['WAVE'].data)
        if wave.size == nflux:
            return wave

    for extname in ('WAVE', 'WAVELENGTH', 'LAMBDA'):
        if extname not in hdul:
            continue
        wave = _as_1d_array(hdul[extname].data)
        if wave.size == nflux:
            return wave

    for hdu in hdul:
        if not hasattr(hdu, 'header'):
            continue
        header = hdu.header
        if 'CRVAL1' in header and 'CDELT1' in header and 'NAXIS1' in header:
            naxis1 = int(header['NAXIS1'])
            if naxis1 == nflux:
                crval1 = float(header['CRVAL1'])
                cdelt1 = float(header['CDELT1'])
                crpix1 = float(header.get('CRPIX1', 1))
                index = np.arange(nflux, dtype=float)
                return crval1 + (index + 1 - crpix1) * cdelt1

    raise ValueError('Could not determine wavelength array from FITS file.')


def _read_spectrum_from_hdul(hdul, z):
    if z is None:
        raise ValueError(
            'Redshift not found in FITS metadata. Supply it with --z.'
        )

    flux = _flux_from_hdu(hdul)
    ferr = _ivar_from_hdu(hdul, len(flux))
    wave_obs = _wave_from_hdu(hdul, len(flux))

    wave_rest = wave_obs / (1.0 + z)
    valid = np.isfinite(wave_rest) & np.isfinite(flux) & np.isfinite(ferr)
    if not np.any(valid):
        raise ValueError('No valid spectral pixels found in FITS file.')

    spres = Table({
        'lbd': np.asarray(wave_rest[valid], dtype=float),
        'f': np.asarray(flux[valid], dtype=float),
        'ferr': np.asarray(ferr[valid], dtype=float),
    })
    return spres, z


def read_spectrum_ascii(path, z):
    if z is None:
        raise ValueError('ASCII spectra are rest-frame; supply redshift with --z.')

    spres = ascii.read(
        path,
        format='no_header',
        names=['lbd', 'f', 'ferr'],
        comment='#',
    )
    if len(spres) == 0:
        raise ValueError('ASCII spectrum file is empty.')
    return spres, z
