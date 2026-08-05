"""Lightweight unit tests for qsosigma helpers (no full spectrum fits)."""

from __future__ import annotations

import os

import numpy as np
import pytest
from qsosigma.cak_metrics import (
    CAK_LAB_WAVE,
    DEPTH_FAIL_MAX,
    SIGV_MIN,
    _ensemble_member_ok,
    _positive_ferr,
    build_cak_windows,
    clear_cak_template_cache,
    default_stellar_template_dir,
    expected_dispersion_kms,
    load_cak_template,
    load_cak_template_catalog,
)
from qsosigma.fit_results import _format_value_err
from qsosigma.spectrum_io import (
    DEFAULT_UNCERTAINTY_FLOOR,
    FERR_EPSILON,
    apply_uncertainty_floor,
    sanitize_ferr,
)


def test_default_template_dir_exists():
    path = default_stellar_template_dir()
    assert os.path.isdir(path)
    assert os.path.isfile(os.path.join(path, 'templates.manifest.csv'))


def test_uncertainty_floor_default_matches_cli():
    assert DEFAULT_UNCERTAINTY_FLOOR == 0.002


def test_sanitize_ferr_clips_nonpositive():
    ferr = np.array([0.0, -1.0, 0.05, np.nan])
    out = sanitize_ferr(ferr)
    assert out[0] == FERR_EPSILON
    assert out[1] == FERR_EPSILON
    assert out[2] == pytest.approx(0.05)
    assert np.isnan(out[3])


def test_positive_ferr_matches_sanitize():
    ferr = np.array([0.0, 1e-3])
    assert np.allclose(_positive_ferr(ferr), sanitize_ferr(ferr))


def test_apply_uncertainty_floor_quadrature():
    flux = np.array([1.0, 2.0])
    ferr = np.array([0.0, 0.0])
    out = apply_uncertainty_floor(flux, ferr, 0.01)
    assert out[0] == pytest.approx(0.01)
    assert out[1] == pytest.approx(0.02)


def test_build_cak_windows_ordering():
    rngline, lwrng, cntrng = build_cak_windows(3800.0, 4100.0)
    assert rngline[0] < CAK_LAB_WAVE < rngline[1]
    assert lwrng[0] < CAK_LAB_WAVE < lwrng[1]
    assert cntrng[0] < cntrng[1] < CAK_LAB_WAVE < cntrng[2] < cntrng[3]


def test_ensemble_member_ok_cull_rules():
    locked = 'locked'
    good = {'sigv': 120.0, 'depth': 0.2}
    shallow = {'sigv': 120.0, 'depth': DEPTH_FAIL_MAX}
    at_lo = {'sigv': SIGV_MIN, 'depth': 0.2}
    assert _ensemble_member_ok(good, 'other', locked)
    assert not _ensemble_member_ok(shallow, 'other', locked)
    assert not _ensemble_member_ok(at_lo, 'other', locked)
    assert _ensemble_member_ok(at_lo, locked, locked)


def test_expected_dispersion_kms():
    assert expected_dispersion_kms(0.0, 100.0) == pytest.approx(100.0)
    assert expected_dispersion_kms(120.0, 160.0) == pytest.approx(
        np.sqrt(120.0 ** 2 + 160.0 ** 2)
    )


def test_format_value_err():
    text = _format_value_err(120.0, 15.0, 'km/s', n=3)
    assert '120' in text and '15' in text and 'km/s' in text
    assert _format_value_err(np.nan, 1.0, 'km/s').startswith('nan')


def test_template_catalog_and_cache():
    clear_cak_template_cache()
    tpldir = default_stellar_template_dir()
    enabled = load_cak_template_catalog(tpldir, enabled_only=True)
    assert len(enabled) >= 1
    a = load_cak_template(tpldir, enabled[0])
    b = load_cak_template(tpldir, enabled[0])
    assert a[0] is b[0] and a[1] is b[1]
    assert a[0].size > 10
    assert np.all(np.isfinite(a[0]))
    clear_cak_template_cache()
