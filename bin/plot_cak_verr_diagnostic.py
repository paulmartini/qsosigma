#!/usr/bin/env python3
"""
Build a 1-column Ca II K diagnostic figure for sigma_verr stacking tests.

Preferred input is one multi-stack ``cak_fitresults_z*.fits`` file from
``run_cakfit.py --validate`` (any number of ``CAKPLOT*`` panels; legacy
``cak_validate_*.fits`` names still work). Alternate mode accepts separate
results FITS files with ``CAK_PLOT`` extensions (one per verr level; default
labels 0/100/200/300/400 km/s).

Example
-------
  python bin/plot_cak_verr_diagnostic.py \\
    /path/to/cak_fitresults_z0.250_z0.300.fits

  python bin/plot_cak_verr_diagnostic.py \\
    verr0/stack_cakfit.fits \\
    verr100/stack_cakfit.fits \\
    verr200/stack_cakfit.fits \\
    verr300/stack_cakfit.fits \\
    verr400/stack_cakfit.fits \\
    -o cak_verr_diagnostic_z0.250_z0.300.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parents[1] / 'py'
if _PY_DIR.is_dir() and str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

import numpy as np

from qsosigma.cak_plots import (
    is_cak_fitresults_fits,
    load_cak_fitresults_verr_entries,
    load_cak_plot_snapshot,
    plot_cak_verr_diagnostic,
)

DEFAULT_VERR_KMS = (0, 100, 200, 300, 400)
STACK_ZBIN_RE = re.compile(r'_z(\d+\.\d+)_z(\d+\.\d+)(?:_|$|\.)')
DEFAULT_YLABEL = 'Relative Flux and Fit Residual'


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
        default=None,
        help=(
            'Output PNG path (default: '
            'cak_verr_diagnostic_z{zlo}_z{zhi}.png when the redshift bin '
            'is known, else cak_verr_diagnostic.png)'
        ),
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
        help='Optional figure title (none by default)',
    )
    parser.add_argument(
        '--ylabel',
        default=DEFAULT_YLABEL,
        help='Shared y-axis label (default: %s)' % DEFAULT_YLABEL,
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI (default: 150)',
    )
    return parser.parse_args()


def redshift_bin_tag(zlo, zhi):
    return 'z%.3f_z%.3f' % (float(zlo), float(zhi))


def parse_redshift_bin_from_name(path):
    """Return ``(zlo, zhi)`` from a filename tag, or ``(None, None)``."""
    match = STACK_ZBIN_RE.search(os.path.basename(path))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def resolve_redshift_bin(paths, entries):
    """Best-effort ``(zlo, zhi)`` from FITS snapshots or input filenames."""
    for entry in entries:
        snapshot = entry.get('snapshot') or {}
        zlo = snapshot.get('zlo', np.nan)
        zhi = snapshot.get('zhi', np.nan)
        if np.isfinite(zlo) and np.isfinite(zhi):
            return float(zlo), float(zhi)
    for path in paths:
        zlo, zhi = parse_redshift_bin_from_name(path)
        if zlo is not None and zhi is not None:
            return zlo, zhi
    return None, None


def default_output_path(zlo, zhi):
    if zlo is not None and zhi is not None:
        return 'cak_verr_diagnostic_%s.png' % redshift_bin_tag(zlo, zhi)
    return 'cak_verr_diagnostic.png'


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


def main():
    args = parse_args()
    paths = [os.path.abspath(path) for path in args.inputs]
    for path in paths:
        if not os.path.isfile(path):
            print('ERROR: File not found: %s' % path, file=sys.stderr)
            return 2

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

    zlo, zhi = resolve_redshift_bin(paths, entries)
    output = args.output if args.output is not None else default_output_path(zlo, zhi)

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
        os.path.abspath(output),
        title=args.title,
        ylabel=args.ylabel,
        dpi=args.dpi,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
