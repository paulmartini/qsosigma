#!/usr/bin/env python3
"""
Plot Ca II K validation residuals versus redshift.

For each multi-stack ``cak_fitresults_*.fits`` file, compare measured
``CAK_STELLAR_DISP`` to ``EXPECTED_DISP`` at injected sigma_verr = 100, 200,
300, and 400 km/s. Points that fail the same criterion as the verr-diagnostic
``FAILED`` flag (measured σ* more than 2σ above expected) are omitted.

Example
-------
  python bin/plot_cak_validation.py cak_fitresults*.fits -o cak_validation.png
  python bin/plot_cak_validation.py "cak_fitresults_*.fits" -o cak_validation.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parents[1] / 'py'
if _PY_DIR.is_dir() and str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

import matplotlib.pyplot as plt
import numpy as np

from qsosigma.cak_plots import (
    discover_results_files,
    figure_output_path,
    is_cak_fitresults_fits,
    is_verr_bad_fit,
    load_cak_fitresults_verr_entries,
    stem_from_results,
)

# Injected verr levels and marker styles for the summary figure.
VERR_STYLES = (
    (100.0, dict(fmt='o', color='k', label=r'$\sigma_{\rm verr} = 100$ km/s')),
    (200.0, dict(fmt='s', color='b', label=r'$\sigma_{\rm verr} = 200$ km/s')),
    (300.0, dict(fmt='^', color='g', label=r'$\sigma_{\rm verr} = 300$ km/s')),
    (400.0, dict(fmt='*', color='r', label=r'$\sigma_{\rm verr} = 400$ km/s')),
)
VERR_LEVELS = tuple(verr for verr, _style in VERR_STYLES)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Plot Ca II K validation residuals (measured minus expected '
            'dispersion) versus redshift from Ca K fit-results FITS files.'
        ),
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help=(
            'Input FITS files and/or glob patterns, e.g. cak_fitresults*.fits '
            'or "cak_fitresults_*.fits"'
        ),
    )
    parser.add_argument(
        '-o', '--output',
        default='cak_validation.png',
        help='Output figure path (default: cak_validation.png)',
    )
    parser.add_argument(
        '--pdf',
        action='store_true',
        help='Write PDF instead of PNG (adjusts the output extension)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI (default: 150)',
    )
    return parser.parse_args()


def _dispersion_error(metrics):
    """Return CAK_STELLAR_DISP_ERR, synthesizing from IVAR if needed."""
    err = metrics.get('CAK_STELLAR_DISP_ERR', np.nan)
    if np.isfinite(err) and err > 0:
        return float(err)
    ivar = metrics.get('CAK_STELLAR_DISP_IVAR', 0.0)
    if ivar > 0:
        return float(1.0 / np.sqrt(ivar))
    return np.nan


def _central_redshift(snapshot):
    z = snapshot.get('z', np.nan)
    if np.isfinite(z):
        return float(z)
    zlo = snapshot.get('zlo', np.nan)
    zhi = snapshot.get('zhi', np.nan)
    if np.isfinite(zlo) and np.isfinite(zhi):
        return float(0.5 * (zlo + zhi))
    return np.nan


def load_validation_points(paths):
    """
    Load validation residual points from multi-stack fit-results files.

    Returns ``(points, skipped_files, n_failed)`` where each point has
    ``z``, ``verr``, ``residual``, ``err``, ``stem``, and ``path``.
    Failed points (same criterion as verr-diagnostic FAILED) are counted in
    ``n_failed`` and not included in ``points``.
    """
    points = []
    skipped = []
    n_failed = 0
    for path in paths:
        if not is_cak_fitresults_fits(path):
            skipped.append(path)
            continue
        entries = load_cak_fitresults_verr_entries(path)
        if not entries:
            skipped.append(path)
            continue
        stem = stem_from_results(path)
        for entry in entries:
            verr = float(entry['verr_kms'])
            if int(round(verr)) not in {int(v) for v in VERR_LEVELS}:
                continue
            snapshot = entry['snapshot']
            metrics = snapshot.get('metrics', {})
            z = _central_redshift(snapshot)
            sig = metrics.get('CAK_STELLAR_DISP', np.nan)
            expected = metrics.get('EXPECTED_DISP', np.nan)
            err = _dispersion_error(metrics)
            if not (
                np.isfinite(z)
                and np.isfinite(sig)
                and np.isfinite(expected)
                and np.isfinite(err)
                and err > 0
            ):
                continue
            if is_verr_bad_fit(snapshot):
                n_failed += 1
                continue
            points.append({
                'z': float(z),
                'verr': float(verr),
                'residual': float(sig - expected),
                'err': float(err),
                'stem': stem,
                'path': path,
            })

    points.sort(key=lambda item: (item['z'], item['verr']))
    return points, skipped, n_failed


def plot_cak_validation(points, output_path, dpi=150):
    """Save validation residual versus redshift for each sigma_verr series."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    plotted = 0
    for verr, style in VERR_STYLES:
        series = [p for p in points if int(round(p['verr'])) == int(round(verr))]
        if not series:
            continue
        z = np.asarray([p['z'] for p in series], dtype=float)
        residual = np.asarray([p['residual'] for p in series], dtype=float)
        err = np.asarray([p['err'] for p in series], dtype=float)
        ax.errorbar(
            z, residual, yerr=err,
            fmt=style['fmt'], color=style['color'], ecolor=style['color'],
            elinewidth=1.2, capsize=3, markersize=7,
            markerfacecolor=style['color'], markeredgecolor=style['color'],
            linestyle='none', label=style['label'],
        )
        plotted += len(series)

    ax.axhline(0.0, color='0.5', linewidth=1.0, linestyle='--')
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('Validation Residual (km/s)', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    if plotted:
        z_all = np.asarray([p['z'] for p in points], dtype=float)
        ax.set_xlim(left=max(0.0, float(np.min(z_all)) - 0.02))
    ax.legend(loc='best', fontsize=11, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(
        'Ca K validation figure saved: %s (%d points)'
        % (output_path, plotted)
    )


def main():
    args = parse_args()
    paths = discover_results_files(args.inputs)
    if not paths:
        print(
            'ERROR: No files matched: %s' % ' '.join(args.inputs),
            file=sys.stderr,
        )
        return 2

    points, skipped, n_failed = load_validation_points(paths)
    for path in skipped:
        print(
            'WARNING: Skipping %s (not a multi-stack fit-results file '
            'with VERR*/CAKPLOT* extensions)' % path,
        )
    if n_failed:
        print(
            'Omitting %d FAILED validation point(s) '
            '(CAK_STELLAR_DISP > EXPECTED_DISP + 2*ERR).' % n_failed,
        )
    if not points:
        print('ERROR: No passing Ca K validation points found.', file=sys.stderr)
        return 1

    print('Plotting %d Ca K validation residual(s):' % len(points))
    for point in points:
        print(
            '  z=%.3f  verr=%.0f  residual=%+.1f +/- %.1f km/s  (%s)'
            % (
                point['z'], point['verr'], point['residual'],
                point['err'], point['stem'],
            )
        )

    plot_cak_validation(
        points,
        figure_output_path(os.path.abspath(args.output), pdf=args.pdf),
        dpi=args.dpi,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
