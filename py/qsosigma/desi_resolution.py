"""
DESI spectrograph resolution vs. wavelength.

Distilled from ``res-sm5-{b,r,z}-new.csv`` in the fastspecfit calibration
directory: one file per channel, wavelength in column 1 and resolution in
30 focal-plane columns. Each channel is summarized as a linear fit to the
median resolution across those positions:

    R(lambda_obs) = slope * lambda_obs + intercept

where ``lambda_obs`` is in Angstroms and R is the resolving power (FWHM).

Instrumental Gaussian sigma in km/s (for use in template broadening):

    sigma_inst = c / (R * FWHM_TO_SIGMA)
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

C_KMS = 2.99792458e5
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

# Provenance only (coefficients below are baked in; this path is not read at runtime).
# Median-over-fibers linear fits from res-sm5-*-new.csv (2026-07).
RESOLUTION_SOURCE = (
    '/global/cfs/cdirs/desi/users/martini/redshifts/fastspecfit/res-sm5-*-new.csv'
)


@dataclass(frozen=True)
class DesiSm5Channel:
    """Linear R(lambda) for one DESI SPECTRO channel."""

    name: str
    wave_min: float
    wave_max: float
    slope: float
    intercept: float

    def R(self, wavelength_obs_angstrom: float) -> float:
        return self.slope * float(wavelength_obs_angstrom) + self.intercept

    def sigma_kms(self, wavelength_obs_angstrom: float) -> float:
        r = self.R(wavelength_obs_angstrom)
        if r <= 0:
            return np.nan
        return C_KMS / (r * FWHM_TO_SIGMA)


DESI_SPECTRO_CHANNELS: Tuple[DesiSm5Channel, ...] = (
    DesiSm5Channel(
        name='b',
        wave_min=3523.239747882707,
        wave_max=5998.417253672886,
        slope=0.5926116823599326,
        intercept=-106.85288122989122,
    ),
    DesiSm5Channel(
        name='r',
        wave_min=5558.550246005832,
        wave_max=7812.318918666259,
        slope=0.6548773275913595,
        intercept=-294.7156988062869,
    ),
    DesiSm5Channel(
        name='z',
        wave_min=7354.5152224177955,
        wave_max=9921.598590183241,
        slope=0.5362712999526501,
        intercept=-106.88254359580827,
    ),
)

DESI_SPECTRO_BY_NAME = {ch.name: ch for ch in DESI_SPECTRO_CHANNELS}


def channels_covering(wavelength_obs_angstrom: float) -> Tuple[DesiSm5Channel, ...]:
    """Return SPECTRO channels whose tabulated wavelength range includes lambda_obs. Note that out-of-range wavelengths fall back to the nearest channel."""
    wave = float(wavelength_obs_angstrom)
    covering = tuple(
        ch for ch in DESI_SPECTRO_CHANNELS
        if ch.wave_min <= wave <= ch.wave_max
    )
    if covering:
        return covering
    nearest = min(DESI_SPECTRO_CHANNELS, key=lambda ch: _distance_to_range(wave, ch))
    return (nearest,)


def _distance_to_range(wave: float, channel: DesiSm5Channel) -> float:
    if wave < channel.wave_min:
        return channel.wave_min - wave
    if wave > channel.wave_max:
        return wave - channel.wave_max
    return 0.0


def desi_spectro_R(
    wavelength_obs_angstrom: float,
    channels: Optional[Iterable[DesiSm5Channel]] = None,
) -> float:
    """Resolving power R at observed-frame wavelength (Angstrom)."""
    if channels is None:
        channels = channels_covering(wavelength_obs_angstrom)
    else:
        channels = tuple(channels)
    if not channels:
        return np.nan
    return float(np.mean([ch.R(wavelength_obs_angstrom) for ch in channels]))


def desi_spectro_R_from_rest(
    wavelength_rest_angstrom: float,
    z: float,
    channels: Optional[Iterable[DesiSm5Channel]] = None,
) -> float:
    """Resolving power R for a rest-frame wavelength and redshift."""
    wave_obs = float(wavelength_rest_angstrom) * (1.0 + float(z))
    if channels is None:
        channels = channels_covering(wave_obs)
    return desi_spectro_R(wave_obs, channels=channels)


def desi_spectro_instrumental_sigma_kms(
    wavelength_obs_angstrom: float,
    channels: Optional[Iterable[DesiSm5Channel]] = None,
) -> float:
    """Instrumental Gaussian sigma (km/s) at observed-frame wavelength."""
    if channels is None:
        channels = channels_covering(wavelength_obs_angstrom)
    else:
        channels = tuple(channels)
    if not channels:
        return np.nan
    sigmas = [ch.sigma_kms(wavelength_obs_angstrom) for ch in channels]
    return float(np.mean(sigmas))


def desi_spectro_instrumental_sigma_kms_from_rest(
    wavelength_rest_angstrom: float,
    z: float,
    channels: Optional[Iterable[DesiSm5Channel]] = None,
) -> float:
    """Instrumental Gaussian sigma (km/s) for a rest-frame wavelength."""
    wave_obs = float(wavelength_rest_angstrom) * (1.0 + float(z))
    if channels is None:
        channels = channels_covering(wave_obs)
    return desi_spectro_instrumental_sigma_kms(wave_obs, channels=channels)


def combine_velocity_sigmas(stellar_sigma_kms: float, instrumental_sigma_kms: float) -> float:
    """Combine stellar and instrumental broadening in quadrature (km/s)."""
    return float(np.sqrt(stellar_sigma_kms ** 2 + instrumental_sigma_kms ** 2))


def desi_spectro_sigma_lbd_from_rest(
    wavelength_rest_angstrom: float,
    z: float,
    ref_wave: Optional[float] = None,
) -> float:
    """Instrumental Gaussian sigma in Angstrom at rest-frame wavelength."""
    ref = float(wavelength_rest_angstrom if ref_wave is None else ref_wave)
    sig_kms = desi_spectro_instrumental_sigma_kms_from_rest(wavelength_rest_angstrom, z)
    return ref * sig_kms / C_KMS
