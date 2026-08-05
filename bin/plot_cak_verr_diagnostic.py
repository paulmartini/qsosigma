#!/usr/bin/env python3
"""
Build a 1-column Ca II K diagnostic figure for sigma_verr stacking tests.

Preferred input is one multi-stack ``cak_fitresults_z*.fits`` file from
``run_cakfit.py --validate`` (contains CAKPLOT* snapshots for each verr
level; legacy ``cak_validate_*.fits`` names still work). Legacy mode still
accepts five separate results FITS files with CAK_PLOT extensions.

Example
-------
  python plot_cak_verr_diagnostic.py \\
    /path/to/cak_fitresults_z0.250_z0.300.fits \\
    -o cak_verr_diagnostic.png \\
    --title "QSO stack, z = 0.25–0.30"

  python plot_cak_verr_diagnostic.py \\
    verr0/stack_cakfit.fits \\
    verr100/stack_cakfit.fits \\
    verr200/stack_cakfit.fits \\
    verr300/stack_cakfit.fits \\
    verr400/stack_cakfit.fits \\
    -o cak_verr_diagnostic.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from cak_plots import (
    is_cak_fitresults_fits,
    load_cak_fitresults_verr_entries,
    load_cak_plot_snapshot,
    plot_cak_verr_diagnostic,
)

DEFAULT_VERR_KMS = (0, 100, 200, 300, 400)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Plot Ca II K fits for a sigma_verr stacking test '
            '(panels increasing verr top to bottom).'
        ),
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help=(
            'One cak_fitresults_*.fits file (multi-stack), or five results '
            'FITS files in order of increasing sigma_verr'
        ),
    )
    parser.add_argument(
        '-o', '--output',
        default='cak_verr_diagnostic.png',
        help='Output PNG path (default: cak_verr_diagnostic.png)',
    )
    parser.add_argument(
        '--verr',
        type=float,
        nargs=5,
        default=list(DEFAULT_VERR_KMS),
        metavar='KM/S',
        help=(
            'Injected sigma_verr for each panel when giving five separate '
            'files (default: 0 100 200 300 400). Ignored for multi-stack input.'
        ),
    )
    parser.add_argument(
        '--title',
        default=None,
        help='Optional figure title (e.g. redshift range of the stack)',
    )
    parser.add_argument(
        '--ylabel',
        default='Relative Flux',
        help='Shared y-axis label (default: Relative Flux)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI (default: 150)',
    )
    return parser.parse_args()


def load_verr_entries_from_files(paths, verr_values):
    entries = []
    for path, verr_kms in zip(paths, verr_values):
        path = os.path.abspath(path)
        snapshot = load_cak_plot_snapshot(path)
        if snapshot is None:
            print('ERROR: No CAK_PLOT / CAKPLOT* extension in %s' % path, file=sys.stderr)
            return None
        entries.append({
            'verr_kms': float(verr_kms),
            'snapshot': snapshot,
            'path': path,
        })
    return entries


def _default_title_from_fitresults(path):
    base = os.path.basename(path)
    for prefix in ('cak_fitresults_', 'cak_validate_'):
        if base.startswith(prefix) and base.endswith('.fits'):
            tag = base[len(prefix):-len('.fits')]
            return 'Ca II K fit results, %s' % tag.replace('_', ' ')
    return None


def main():
    args = parse_args()
    paths = [os.path.abspath(path) for path in args.inputs]
    for path in paths:
        if not os.path.isfile(path):
            print('ERROR: File not found: %s' % path, file=sys.stderr)
            return 2

    title = args.title
    if len(paths) == 1 and is_cak_fitresults_fits(paths[0]):
        entries = load_cak_fitresults_verr_entries(paths[0])
        if not entries:
            print(
                'ERROR: No CAKPLOT* extensions in %s. Re-run with '
                'run_cakfit.py --validate to write plot snapshots.'
                % paths[0],
                file=sys.stderr,
            )
            return 1
        if title is None:
            title = _default_title_from_fitresults(paths[0])
    elif len(paths) == 5:
        entries = load_verr_entries_from_files(paths, args.verr)
        if entries is None:
            return 1
    else:
        print(
            'ERROR: Provide one multi-stack cak_fitresults_*.fits file or '
            'exactly five results FITS files (got %d).' % len(paths),
            file=sys.stderr,
        )
        return 2

    print('Plotting Ca K verr diagnostic:')
    for entry in entries:
        sig = entry['snapshot']['metrics'].get('CAK_STELLAR_DISP', np.nan)
        print('  sigma_verr=%3.0f km/s  sigma_*=%.1f km/s  (%s)' % (
            entry['verr_kms'],
            sig,
            os.path.basename(entry['path']),
        ))

    plot_cak_verr_diagnostic(
        entries,
        os.path.abspath(args.output),
        title=title,
        ylabel=args.ylabel,
        dpi=args.dpi,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
