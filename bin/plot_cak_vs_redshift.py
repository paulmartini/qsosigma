#!/usr/bin/env python3
"""
Plot Ca II K stellar velocity dispersion versus central stack redshift.

Accepts ``cak_fitresults_*.fits`` / legacy ``*_cakfit.fits`` /
``*_linefit.fits`` files. Multi-stack fit-results files use the verr0
(``VERR0`` / ``CAKPLOT0``) metrics. Points are sorted by increasing redshift.

Example
-------
  python bin/plot_cak_vs_redshift.py cak_fitresults*.fits -o cak_vs_redshift.png
  python bin/plot_cak_vs_redshift.py "cak_fitresults_*.fits" -o cak_vs_redshift.png
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
    AXIS_LABEL_FONTSIZE,
    discover_results_files,
    figure_output_path,
    load_cak_plot_snapshot,
    stem_from_results,
)

# Match the heavier verr-diagnostic axis typography.
TICK_FONTSIZE = 14


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Plot Ca II K stellar velocity dispersion versus redshift '
            'from one or more Ca K results FITS files.'
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
        default='cak_vs_redshift.png',
        help='Output figure path (default: cak_vs_redshift.png)',
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


def load_dispersion_points(paths):
    """
    Load ``(z, sigma_*, err, stem, path)`` rows from results files.

    Multi-stack files use the verr0 snapshot (default of
    :func:`load_cak_plot_snapshot`). Returns ``(points, skipped)``.
    """
    points = []
    skipped = []
    for path in paths:
        snapshot = load_cak_plot_snapshot(path)
        if snapshot is None:
            skipped.append(path)
            continue
        metrics = snapshot.get('metrics', {})
        z = snapshot.get('z', np.nan)
        if not np.isfinite(z):
            zlo = snapshot.get('zlo', np.nan)
            zhi = snapshot.get('zhi', np.nan)
            if np.isfinite(zlo) and np.isfinite(zhi):
                z = 0.5 * (zlo + zhi)
        sig = metrics.get('CAK_STELLAR_DISP', np.nan)
        if not (np.isfinite(z) and np.isfinite(sig)):
            skipped.append(path)
            continue
        points.append({
            'z': float(z),
            'sig': float(sig),
            'err': _dispersion_error(metrics),
            'stem': stem_from_results(path),
            'path': path,
        })

    points.sort(key=lambda item: item['z'])
    return points, skipped


def plot_cak_vs_redshift(
    points, output_path, dpi=150,
    label_fontsize=AXIS_LABEL_FONTSIZE,
    tick_fontsize=TICK_FONTSIZE,
):
    """Save sigma_* versus redshift with black points and error bars."""
    z = np.asarray([p['z'] for p in points], dtype=float)
    sig = np.asarray([p['sig'] for p in points], dtype=float)
    err = np.asarray([p['err'] for p in points], dtype=float)
    yerr = np.where(np.isfinite(err) & (err > 0), err, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.errorbar(
        z, sig, yerr=yerr, fmt='o', color='k', ecolor='k',
        elinewidth=1.2, capsize=3, markersize=6, markerfacecolor='k',
        markeredgecolor='k', linestyle='none',
    )
    ax.set_xlabel('Redshift', fontsize=label_fontsize)
    ax.set_ylabel('Stellar Velocity Dispersion (km/s)', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_xlim(left=max(0.0, float(np.min(z)) - 0.02))
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print('Ca K vs redshift figure saved: %s (%d points)' % (output_path, len(points)))


def main():
    args = parse_args()
    paths = discover_results_files(args.inputs)
    if not paths:
        print(
            'ERROR: No files matched: %s' % ' '.join(args.inputs),
            file=sys.stderr,
        )
        return 2

    points, skipped = load_dispersion_points(paths)
    for path in skipped:
        print(
            'WARNING: Skipping %s (no usable verr0 Ca K metrics / redshift)'
            % path,
        )
    if not points:
        print('ERROR: No Ca K dispersion measurements found.', file=sys.stderr)
        return 1

    print('Plotting %d Ca K dispersion point(s):' % len(points))
    for point in points:
        err = point['err']
        err_txt = ('%.1f' % err) if np.isfinite(err) else 'nan'
        print(
            '  z=%.3f  sigma_*=%.1f +/- %s km/s  (%s)'
            % (point['z'], point['sig'], err_txt, point['stem'])
        )

    plot_cak_vs_redshift(
        points,
        figure_output_path(os.path.abspath(args.output), pdf=args.pdf),
        dpi=args.dpi,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
