"""Tools to broaden and shift stellar absorption templates.

Originally from IronFit by Zhefu Yu.
"""

import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve

C_KMS = 2.99792458e5


def broaden_template(
    wavelength, values, width, pixel_size=1.0, space='linear',
):
    """
    Broaden a template with a Gaussian kernel.

    Parameters
    ----------
    wavelength, values : array-like
        Template wavelength and absorption/flux arrays.
    width : float
        Gaussian smoothing width. In wavelength units when ``space='linear'``;
        in km/s when ``space='log'``.
    pixel_size : float
        Template pixel size in wavelength units (used only for
        ``space='linear'``). For ``space='log'``, the ln(λ) spacing is taken
        from ``wavelength``.
    space : {'linear', 'log'}
        Broaden in wavelength space (``linear``) or velocity / ln(λ) space
        (``log``; natural log, not log10). For ``linear``, ``wavelength``
        should be evenly sampled in λ; for ``log``, evenly sampled in ln(λ).

    Returns
    -------
    broadened : ndarray
        Broadened template on the same sampling as the input.
    """
    # Gaussian1DKernel expects stddev in pixels, not physical units.
    if space == 'linear':
        width_pix = width / pixel_size
        kernel = Gaussian1DKernel(stddev=width_pix)
        return convolve(values, kernel)

    if space == 'log':
        ln_wavelength = np.log(wavelength)
        dln = abs(ln_wavelength[1] - ln_wavelength[0])
        width_pix = (width / C_KMS) / dln
        kernel = Gaussian1DKernel(stddev=width_pix)
        # Convolve λ * F(λ) vs. ln(λ), then divide by λ.
        return convolve(wavelength * values, kernel) / wavelength

    raise ValueError('Unknown space for broadening: %r' % space)


def shift_template(wavelength, shift, space='linear'):
    """
    Shift a template in wavelength or velocity space.

    Parameters
    ----------
    wavelength : array-like
        Template wavelength array.
    shift : float
        Displacement in wavelength units when ``space='linear'``, or in km/s
        when ``space='log'``.
    space : {'linear', 'log'}
        Shift in wavelength space (``linear``) or velocity / ln(λ) space
        (``log``; natural log, not log10).

    Returns
    -------
    wavelength_shifted : ndarray
        Shifted wavelength array (values unchanged).
    """
    if space == 'linear':
        return wavelength + shift

    if space == 'log':
        dln = shift / C_KMS
        return np.exp(np.log(wavelength) + dln)

    raise ValueError('Unknown space for shifting: %r' % space)
