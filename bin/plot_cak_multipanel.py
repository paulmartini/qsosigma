#!/usr/bin/env python3
"""
Build multi-panel Ca II K figures from Ca K results FITS files.

Accepts ``cak_fitresults_*.fits`` / legacy ``*_cakfit.fits`` /
``*_linefit.fits`` files with a ``CAK_PLOT`` or ``CAKPLOT*`` extension.
Multi-stack fit-results files use the verr0 ``CAKPLOT0`` snapshot. Panels
are sorted by increasing redshift and laid out left-to-right, then
top-to-bottom by row. Large sets are split across multiple output files
(at most 5 rows by 3 columns per file). The first file reserves the
lower-right cell for the legend (14 data panels maximum); later files
contain data panels only.

For a single redshift bin's verr series, prefer ``plot_cak_verr_diagnostic.py``.

Example
-------
  python bin/plot_cak_multipanel.py cak_fitresults*.fits -o cak_stacks.png
  python bin/plot_cak_multipanel.py "cak_fitresults_*.fits" -o cak_stacks.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PY_DIR = Path(__file__).resolve().parents[1] / 'py'
if _PY_DIR.is_dir() and str(_PY_DIR) not in sys.path:
    sys.path.insert(0, str(_PY_DIR))

import numpy as np

from qsosigma.cak_plots import (
    discover_results_files,
    format_multipanel_label,
    load_cak_plot_snapshot,
    plot_cak_multipanel,
    stem_from_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot Ca II K fits from one or more Ca K results FITS files.',
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
        default='cak_stacks.png',
        help='Output PNG path (default: cak_stacks.png; numbered if split)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Figure DPI (default: 150)',
    )
    return parser.parse_args()


def load_snapshots(paths):
    """Load plot snapshots from results files; return (snapshots, skipped)."""
    snapshots = []
    skipped = []
    for path in paths:
        snapshot = load_cak_plot_snapshot(path)
        if snapshot is None:
            skipped.append(path)
            continue
        stem = stem_from_results(path)
        snapshot['panel_label'] = format_multipanel_label(snapshot)
        snapshot['stem'] = stem
        snapshot['path'] = path
        snapshots.append(snapshot)

    snapshots.sort(key=lambda item: (
        np.inf if not np.isfinite(item.get('z', np.nan)) else float(item['z']),
        item.get('stem', ''),
    ))
    return snapshots, skipped


def main():
    args = parse_args()
    paths = discover_results_files(args.inputs)
    if not paths:
        print(
            'ERROR: No files matched: %s' % ' '.join(args.inputs),
            file=sys.stderr,
        )
        return 2

    snapshots, skipped = load_snapshots(paths)
    for path in skipped:
        print('WARNING: Skipping %s (no CAK_PLOT / CAKPLOT* extension)' % path)
    if not snapshots:
        print('ERROR: No Ca K plot snapshots found in matched files.', file=sys.stderr)
        return 1

    print('Plotting %d Ca K panel(s):' % len(snapshots))
    for snapshot in snapshots:
        z_text = 'z=%.3f' % snapshot['z'] if np.isfinite(snapshot.get('z', np.nan)) else 'z=nan'
        print('  %s (%s)' % (snapshot['stem'], z_text))

    # Shared label for flux and residual rows across heterogeneous stacks.
    plot_cak_multipanel(
        snapshots,
        os.path.abspath(args.output),
        dpi=args.dpi,
        ylabel='Relative Flux and Fit Residual',
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
