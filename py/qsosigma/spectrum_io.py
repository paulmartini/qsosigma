"""Read input spectra from FITS or ASCII files."""

import os

import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table

REDSHIFT_KEYS = ('Z', 'REDSHIFT', 'Z_OBJ', 'Z_QSO', 'ZSTACK', 'HELIOZ', 'BLSZ')
STACKINFO_Z_KEYS = ('Z', 'ZREF', 'ZMEAN')

# Match run_cakfit.py --uncertainty-floor default.
DEFAULT_UNCERTAINTY_FLOOR = 0.002
# Lower clip for ferr so chi2 never divides by zero / subnormals.
FERR_EPSILON = float(np.finfo(np.float64).tiny)


def sanitize_ferr(ferr):
    """Clip ``ferr`` to be at least ``FERR_EPSILON`` (finite values only)."""
    ferr = np.asarray(ferr, dtype=float)
    out = np.array(ferr, dtype=float, copy=True)
    finite = np.isfinite(out)
    out[finite] = np.maximum(out[finite], FERR_EPSILON)
    return out


def load_spectrum(path, z_override=None, uncertainty_floor=DEFAULT_UNCERTAINTY_FLOOR):
    """
    Load a spectrum from FITS or ASCII into the rest frame.

    Parameters
    ----------
    path : str
        Input spectrum path.
    z_override : float, optional
        Redshift. Required for ASCII (already rest-frame wavelengths; ``z`` is
        metadata). For FITS, overrides header/table redshift; observed-frame
        wavelengths are divided by ``(1 + z)``.
    uncertainty_floor : float
        Fractional uncertainty floor combined in quadrature with the formal
        error (default: 0.002).

    Returns
    -------
    spres : astropy.table.Table
        Columns ``lbd``, ``f``, ``ferr`` in the rest frame.
    z : float
        Redshift used.
    pixspec : float
        Median rest-frame pixel size in Angstroms.
    fmt : str
        ``'FITS'`` or ``'ASCII'``.
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

    spres['ferr'] = sanitize_ferr(apply_uncertainty_floor(
        spres['f'], spres['ferr'], uncertainty_floor,
    ))

    order = np.argsort(np.asarray(spres['lbd'], dtype=float))
    spres = spres[order]
    pixspec = float(np.median(np.diff(np.asarray(spres['lbd'], dtype=float))))
    return spres, z, pixspec, fmt


def is_fits_file(path):
    """Return True if ``path`` looks like a FITS file (extension or SIMPLE card)."""
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


def infer_fscale(spres):
    """
    Choose plot flux scaling from the median near Ca K (3900–4000 Å).

    Returns 1.0 if the median flux is > 1, else 1e17 (for 10^-17 erg units).
    """
    lbd = np.asarray(spres['lbd'], dtype=float)
    cakmask = (lbd >= 3900) & (lbd <= 4000)
    ref = spres[cakmask] if np.any(cakmask) else spres
    median_flux = float(np.median(np.asarray(ref['f'], dtype=float)))
    if median_flux > 1.0:
        return 1.0
    return 1e17


def flux_label(fscale):
    """Axis label string for the chosen flux scale."""
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
    """Read redshift from known header keywords, or None."""
    for key in REDSHIFT_KEYS:
        if key in header:
            return float(header[key])
    return None


def _table_redshift(hdul):
    """Read redshift from STACKINFO / SPECOBJ / FIBERMAP tables, or None."""
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
    """Return redshift from FITS headers or known tables, or None."""
    for hdu in hdul:
        if not hasattr(hdu, 'header'):
            continue
        z = _header_redshift(hdu.header)
        if z is not None:
            return z
    return _table_redshift(hdul)


def _as_1d_array(data):
    """Flatten array-like data to 1-D float."""
    return np.asarray(data, dtype=float).reshape(-1)


def _flux_from_hdu(hdul):
    """Extract flux from a FLUX HDU or PRIMARY image."""
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
    """
    Extract per-pixel flux *error* (not IVAR) from IVAR, SIGMA, or ERR.

    IVAR > 0 is converted as ``1/sqrt(IVAR)``; non-positive entries become NaN.
    """
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
    """Extract observed-frame wavelength from WAVE HDUs or WCS keywords."""
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
    """Build a rest-frame spectrum Table from an open FITS HDUList."""
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
    """
    Read a rest-frame ASCII spectrum: whitespace columns ``lbd f ferr``.

    ``z`` is required metadata (wavelengths are not de-redshifted).
    """
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
