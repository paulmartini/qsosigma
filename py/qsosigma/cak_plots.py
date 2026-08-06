"""Ca II K diagnostic plotting and CAK_PLOT snapshot helpers."""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator
import numpy as np
from astropy.io import fits

from qsosigma.cak_metrics import CAH_LAB_WAVE, CAK_LAB_WAVE, CAK_METRIC_SUFFIXES, CAK_PREFIX, C_KMS

CAKPLOT_EXT_RE = re.compile(r'^CAKPLOT(\d+)$', re.IGNORECASE)
VERR_EXT_RE = re.compile(r'^VERR(\d+)$', re.IGNORECASE)

# Shared panel styling.
CAK_WAVE_TICKS = (3870.0, 3900.0, 3930.0, 3960.0, 3990.0)
TICK_LABELSIZE = 12
LEGEND_FONTSIZE = 12
MULTIPANEL_LEGEND_FONTSIZE = 14
PANEL_LABEL_FONTSIZE = 14


def velocity_to_wavelength(v_kms, ref):
    """Velocity offset (km/s) to rest wavelength (Angstrom) relative to ref."""
    return ref * (1.0 + v_kms / C_KMS)


def _mark_centroid(ax, v_kms, rest_wave, label=None):
    """Draw a vertical line at the Ca K centroid velocity."""
    if not np.isfinite(v_kms):
        return
    wave = velocity_to_wavelength(v_kms, rest_wave)
    ax.axvline(wave, color='r', linestyle='--', linewidth=1.0, label=label)


def _panel_label(ax, text, fontsize=PANEL_LABEL_FONTSIZE):
    """Draw an upper-left panel annotation."""
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes,
        ha='left', va='top', fontsize=fontsize,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'),
    )


def shade_cak_excluded_regions(ax, plot):
    """Mark plot-window pixels outside the Ca K chi2 fit range."""
    fit_range = plot.get('fit_range')
    if fit_range is None:
        return
    fit_lo, fit_hi = float(fit_range[0]), float(fit_range[1])
    plot_lo = float(np.min(plot['lbd']))
    plot_hi = float(np.max(plot['lbd']))
    shade_kw = dict(color='0.90', alpha=0.55, zorder=0, linewidth=0)
    if plot_lo < fit_lo:
        ax.axvspan(plot_lo, fit_lo, **shade_kw)
    if fit_hi < plot_hi:
        ax.axvspan(fit_hi, plot_hi, **shade_kw)


def format_cak_wavelength_axis(ax, labelsize=TICK_LABELSIZE):
    """Apply fixed Angstrom ticks (3870, 3900, 3930, 3960, 3990)."""
    ax.xaxis.set_major_locator(FixedLocator(CAK_WAVE_TICKS))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _pos: '%.0f' % x))
    ax.tick_params(axis='both', labelsize=labelsize)


def plot_cak_panels(
    ax_data, ax_res, cak_result, ylabel, pixspec,
    panel_label='Ca II K', show_legend=True, show_ylabel=True,
    xlim=None, show_reference_template=False, show_centroid=False,
):
    """Draw Ca K data/model and residual panels on existing axes."""
    plot = cak_result['plot']
    metrics = cak_result.get('metrics', {})
    shade_cak_excluded_regions(ax_data, plot)
    shade_cak_excluded_regions(ax_res, plot)
    PlotSpec(
        ax_data, plot['lbd'], plot['flux'], plot['ferr'],
        PLOTERR=False, pixspec=pixspec, linewidth=0.4, erralpha=0.2,
        YLABEL=show_ylabel, XLABEL=False,
    )
    ax_data.plot(plot['lbd'], plot['continuum'], '--', color='0.5', linewidth=1.0,
                 label='Power Law Continuum')
    if show_reference_template and 'template_broad' in plot:
        sig_tpl = plot.get('template_sig_kms', np.nan)
        tpl_label = ' ({:.0f} km/s)'.format(sig_tpl) if np.isfinite(sig_tpl) else ''
        ax_data.plot(
            plot['lbd'], plot['template_broad'], '--', color='b', linewidth=1.0,
            label='Reference' + tpl_label,
        )
    ax_data.plot(plot['lbd'], plot['model'], '-', color='r', linewidth=1.2,
                 label='Best Fit Template')
    rest_wave = plot.get('rest_wave', CAK_LAB_WAVE)
    ax_data.axvline(rest_wave, color='k', linestyle=':', linewidth=0.8,
                    label='Ca II K')
    cah_wave = plot.get('cah_wave', CAH_LAB_WAVE)
    ax_data.axvline(cah_wave, color='0.5', linestyle=':', linewidth=0.8,
                    label=r'Ca II H + H$\epsilon$')
    if show_centroid:
        _mark_centroid(
            ax_data, metrics.get('CAK_CENTROID', np.nan), rest_wave,
            label='Ca II K Centroid',
        )
    if show_ylabel:
        ax_data.set_ylabel(ylabel, fontsize=10)
    else:
        ax_data.set_ylabel('')
    ax_data.set_title('')
    _panel_label(ax_data, panel_label)
    if xlim is not None:
        ax_data.set_xlim(xlim)
    else:
        ax_data.set_xlim((plot['lbd'].min(), plot['lbd'].max()))
    ax_data.tick_params(axis='both', labelsize=TICK_LABELSIZE, labelbottom=False)
    if show_legend:
        ax_data.legend(loc='lower right', fontsize=LEGEND_FONTSIZE)

    ax_res.errorbar(plot['lbd'], plot['residuals'], yerr=plot['ferr'], fmt='none',
                    ecolor='0.7', elinewidth=0.4, alpha=0.5)
    ax_res.plot(plot['lbd'], plot['residuals'], 'k-', linewidth=0.4)
    ax_res.axhline(0.0, color='k', linewidth=0.8)
    if xlim is not None:
        ax_res.set_xlim(xlim)
    format_cak_wavelength_axis(ax_res)
    ax_res.tick_params(axis='both', labelsize=TICK_LABELSIZE)


def plot_cak(cak_result, ylabel, pixspec, output_path):
    """Save a dedicated Ca II K figure with data, model, and residuals."""
    fig = plt.figure(figsize=(8, 4))
    gs = GridSpec(2, 1, figure=fig, hspace=0.08, height_ratios=[3, 1])
    ax_data = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_data)
    plot_cak_panels(ax_data, ax_res, cak_result, ylabel, pixspec)
    ax_res.set_xlabel(r'Wavelength ($\mathrm{\AA}$)', fontsize=12)
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('Ca K profile figure saved: %s' % output_path)


def _plot_dict_from_cak_plot_hdu(hdu) -> Dict:
    """Convert a CAK_PLOT FITS extension into a plot dict."""
    header = hdu.header
    return {
        'lbd': np.asarray(hdu.data['WAVE'], dtype=float),
        'flux': np.asarray(hdu.data['FLUX'], dtype=float),
        'ferr': np.asarray(hdu.data['FERR'], dtype=float),
        'continuum': np.asarray(hdu.data['CONTINUUM'], dtype=float),
        'model': np.asarray(hdu.data['MODEL'], dtype=float),
        'template_broad': np.asarray(hdu.data['TEMPLATE'], dtype=float),
        'residuals': np.asarray(hdu.data['RESIDUAL'], dtype=float),
        'fit_range': [
            float(header['CAKFITLO']),
            float(header['CAKFITHI']),
        ],
        'template_sig_kms': float(header.get('CAKTPLSG', np.nan)),
        'rest_wave': float(header.get('CAKRESTW', CAK_LAB_WAVE)),
        'cah_wave': float(header.get('CAKHWAVE', CAH_LAB_WAVE)),
    }


def _metrics_from_table_hdu(hdu) -> Dict:
    """Read Ca K scalar metrics from a one-row table HDU (LINES/CAKFIT/VERR*)."""
    if hdu is None or hdu.data is None or len(hdu.data) == 0:
        return {}
    names = hdu.data.dtype.names or ()
    row = hdu.data[0]
    metrics = {}
    for suffix in CAK_METRIC_SUFFIXES:
        key = '%s_%s' % (CAK_PREFIX, suffix)
        if key in names:
            metrics[key] = float(row[key])
        ivar_key = '%s_IVAR' % key
        if ivar_key in names:
            metrics[ivar_key] = float(row[ivar_key])
        err_key = '%s_ERR' % key
        if err_key in names:
            metrics[err_key] = float(row[err_key])
    for key in ('CAK_CHI2', 'CAK_CHI2_DOF', 'CAK_AT_BOUND', 'EXPECTED_DISP', 'VERR_KMS'):
        if key in names:
            metrics[key] = float(row[key])

    # Prefer explicit IVAR; otherwise synthesize from ERR for label formatting.
    for suffix in CAK_METRIC_SUFFIXES:
        key = '%s_%s' % (CAK_PREFIX, suffix)
        ivar_key = '%s_IVAR' % key
        err_key = '%s_ERR' % key
        if metrics.get(ivar_key, 0.0) > 0:
            continue
        err = metrics.get(err_key, np.nan)
        if np.isfinite(err) and err > 0:
            metrics[ivar_key] = float(1.0 / err ** 2)
        else:
            metrics.setdefault(ivar_key, 0.0)
    return metrics


def _list_cakplot_extensions(hdul) -> List[Tuple[float, str]]:
    """Return sorted ``(verr_kms, extname)`` for CAKPLOT* extensions."""
    found = []
    for hdu in hdul:
        name = str(getattr(hdu, 'name', '') or '')
        match = CAKPLOT_EXT_RE.match(name)
        if match:
            found.append((float(match.group(1)), name))
            continue
        if name.upper() == 'CAK_PLOT':
            verr = hdu.header.get('VERR', 0.0)
            found.append((float(verr) if verr is not None else 0.0, name))
    found.sort(key=lambda item: item[0])
    return found


def _metrics_hdu_for_verr(hdul, verr_kms: Optional[float] = None):
    """Pick the table HDU holding Ca K scalars for a snapshot."""
    if verr_kms is not None:
        name = 'VERR%d' % int(round(float(verr_kms)))
        if name in hdul:
            return hdul[name]
    for key in ('LINES', 'CAKFIT', 'VERR0'):
        if key in hdul:
            return hdul[key]
    # First VERR* extension, if any
    for hdu in hdul:
        if VERR_EXT_RE.match(str(getattr(hdu, 'name', '') or '')):
            return hdu
    return hdul[0]


def _plot_hdu_for_verr(hdul, verr_kms: Optional[float] = None):
    """Pick the CAK_PLOT / CAKPLOT* HDU for a snapshot."""
    if verr_kms is not None:
        name = 'CAKPLOT%d' % int(round(float(verr_kms)))
        if name in hdul:
            return hdul[name]
    if 'CAK_PLOT' in hdul:
        return hdul['CAK_PLOT']
    plots = _list_cakplot_extensions(hdul)
    if not plots:
        return None
    if verr_kms is None:
        return hdul[plots[0][1]]
    # Nearest verr extension
    verr_kms = float(verr_kms)
    best = min(plots, key=lambda item: abs(item[0] - verr_kms))
    return hdul[best[1]]


def _snapshot_from_hdus(plot_hdu, metrics_hdu, primary_header=None) -> Dict:
    """Assemble a plotting snapshot dict from plot and metrics HDUs."""
    plot = _plot_dict_from_cak_plot_hdu(plot_hdu)
    metrics = _metrics_from_table_hdu(metrics_hdu)
    cak_header = plot_hdu.header
    metrics_header = metrics_hdu.header if metrics_hdu is not None else {}
    z = np.nan
    for header in (cak_header, metrics_header, primary_header or {}):
        if header is not None and 'Z' in header:
            z = float(header['Z'])
            break
        if header is not None and 'CAK ZLO' in header and 'CAK ZHI' in header:
            z = 0.5 * (float(header['CAK ZLO']) + float(header['CAK ZHI']))
            break
    infile = str(
        cak_header.get('INFILE', metrics_header.get('INFILE', ''))
    ).strip()
    flux_unit = str(cak_header.get('FLUXUNIT', '')).strip() or None
    pixspec = float(cak_header.get('CAKPIXSP', np.nan))
    verr = cak_header.get('VERR', metrics.get('VERR_KMS', np.nan))
    try:
        verr = float(verr)
    except (TypeError, ValueError):
        verr = np.nan
    return {
        'plot': plot,
        'metrics': metrics,
        'z': z,
        'infile': infile,
        'flux_unit': flux_unit,
        'pixspec': pixspec,
        'verr_kms': verr,
    }


def load_cak_plot_snapshot(path: str, verr_kms: Optional[float] = None) -> Optional[Dict]:
    """
    Load a Ca K plotting snapshot from a results FITS file.

    Supported layouts:
      - ``LINES`` / ``CAKFIT`` + ``CAK_PLOT`` (single-spectrum / legacy linefit)
      - multi-stack ``cak_fitresults_*.fits`` with ``VERR*`` + ``CAKPLOT*``
        (legacy ``cak_validate_*.fits`` names are also recognized)

    For multi-stack files, ``verr_kms`` selects the panel (default: lowest
    available, typically verr0).

    Returns a dict with ``plot``, ``metrics``, ``z``, ``flux_unit``,
    ``infile``, ``pixspec``, and ``verr_kms`` (when available), suitable for
    :func:`plot_cak_panels`.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError('Results file not found: %s' % path)

    with fits.open(path) as hdul:
        plot_hdu = _plot_hdu_for_verr(hdul, verr_kms=verr_kms)
        if plot_hdu is None:
            return None
        selected_verr = verr_kms
        if selected_verr is None and 'VERR' in plot_hdu.header:
            selected_verr = plot_hdu.header['VERR']
        metrics_hdu = _metrics_hdu_for_verr(hdul, verr_kms=selected_verr)
        return _snapshot_from_hdus(plot_hdu, metrics_hdu, primary_header=hdul[0].header)


def load_cak_fitresults_verr_entries(path: str) -> List[Dict]:
    """
    Load all verr panels from a multi-stack Ca K fit-results FITS file.

    Returns a list of dicts with ``verr_kms``, ``snapshot``, and ``path`` keys,
    sorted by increasing injected velocity error.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError('Results file not found: %s' % path)

    entries = []
    with fits.open(path) as hdul:
        plot_exts = _list_cakplot_extensions(hdul)
        if not plot_exts:
            return []
        primary_header = hdul[0].header
        for verr, extname in plot_exts:
            plot_hdu = hdul[extname]
            metrics_hdu = _metrics_hdu_for_verr(hdul, verr_kms=verr)
            snapshot = _snapshot_from_hdus(
                plot_hdu, metrics_hdu, primary_header=primary_header,
            )
            entries.append({
                'verr_kms': float(verr),
                'snapshot': snapshot,
                'path': path,
            })
    return entries


# Backward-compatible alias
load_cak_validate_verr_entries = load_cak_fitresults_verr_entries


def is_cak_fitresults_fits(path: str) -> bool:
    """
    Return True if ``path`` is a multi-stack Ca K fit-results FITS file
    (``VERR*`` + ``CAKPLOT*``). Single-spectrum ``cak_fitresults_*.fits``
    files with only ``CAK_PLOT`` return False.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return False
    with fits.open(path) as hdul:
        names = {str(h.name).upper() for h in hdul}
        return any(n.startswith('CAKPLOT') for n in names) and any(
            n.startswith('VERR') for n in names
        )


# Backward-compatible alias
is_cak_validate_fits = is_cak_fitresults_fits


def _default_panel_label(snapshot: Dict) -> str:
    """Fallback panel label from redshift."""
    z = snapshot.get('z', np.nan)
    if np.isfinite(z):
        return 'z = %.3f' % z
    return 'Ca II K'


def format_multipanel_label(snapshot: Dict) -> str:
    """Build upper-left annotation with z and stellar dispersion (CAK_STELLAR_DISP)."""
    metrics = snapshot.get('metrics', {})
    z = snapshot.get('z', np.nan)
    sig = metrics.get('CAK_STELLAR_DISP', np.nan)
    ivar = metrics.get('CAK_STELLAR_DISP_IVAR', 0.0)
    err = (1.0 / np.sqrt(ivar)) if ivar > 0 else np.nan

    parts = []
    if np.isfinite(z):
        parts.append('z = %.3f' % z)
    if np.isfinite(sig):
        sig_i = int(round(sig))
        if np.isfinite(err):
            err_i = int(round(err))
            parts.append(r'$\sigma_* = %d \pm %d$ km/s' % (sig_i, err_i))
        else:
            parts.append(r'$\sigma_* = %d$ km/s' % sig_i)
    return '\n'.join(parts) if parts else 'Ca II K'


def format_verr_panel_label(verr_kms, snapshot: Dict) -> str:
    """Build upper-left annotation with injected verr and measured stellar dispersion."""
    metrics = snapshot.get('metrics', {})
    sig = metrics.get('CAK_STELLAR_DISP', np.nan)
    ivar = metrics.get('CAK_STELLAR_DISP_IVAR', 0.0)
    err = (1.0 / np.sqrt(ivar)) if ivar > 0 else np.nan

    parts = [r'$\sigma_{\rm verr} = %d$ km/s' % int(round(float(verr_kms)))]
    if np.isfinite(sig):
        sig_i = int(round(sig))
        if np.isfinite(err):
            err_i = int(round(err))
            parts.append(r'$\sigma_* = %d \pm %d$ km/s' % (sig_i, err_i))
        else:
            parts.append(r'$\sigma_* = %d$ km/s' % sig_i)
    chi2 = metrics.get('CAK_CHI2', np.nan)
    chi2_dof = metrics.get('CAK_CHI2_DOF', np.nan)
    if np.isfinite(chi2) and np.isfinite(chi2_dof) and chi2_dof > 0:
        parts.append(r'$\chi^2/\mathrm{dof} = %.1f$' % (chi2 / chi2_dof))
    return '\n'.join(parts)


def _snapshot_cak_result(snapshot: Dict) -> Dict:
    """Minimal ``{plot, metrics}`` dict for :func:`plot_cak_panels`."""
    return {'plot': snapshot['plot'], 'metrics': snapshot.get('metrics', {})}


def _legend_handles_from_snapshot(snapshot: Dict, ylabel: str, show_reference_template=False):
    """Extract legend handles/labels by drawing one panel off-screen."""
    pixspec = snapshot.get('pixspec', np.nan)
    if not np.isfinite(pixspec):
        lbd = snapshot['plot']['lbd']
        pixspec = float(np.median(np.diff(lbd))) if len(lbd) > 1 else 1.0
    tmp_fig, tmp_axes = plt.subplots(2, 1)
    try:
        plot_cak_panels(
            tmp_axes[0], tmp_axes[1],
            _snapshot_cak_result(snapshot), ylabel, pixspec,
            show_legend=True,
            show_ylabel=False,
            show_reference_template=show_reference_template,
            show_centroid=False,
        )
        return tmp_axes[0].get_legend_handles_labels()
    finally:
        plt.close(tmp_fig)


def plot_cak_verr_diagnostic(
    entries,
    output_path,
    title=None,
    ylabel='Relative Flux',
    dpi=150,
    panel_width=8.0,
    panel_height=3.2,
):
    """
    Save a 1-column Ca K figure comparing fits at increasing sigma_verr.

    Each entry is a dict with ``verr_kms`` and ``snapshot`` keys. Panels are
    drawn top-to-bottom in entry order. A shared legend is placed below the
    stack; sigma_verr and measured sigma_* annotations appear in each panel.

    Returns the last panel's ``{plot, metrics}`` (for optional reuse).
    """
    entries = list(entries)
    if not entries:
        raise ValueError('No Ca K verr diagnostic entries to display.')

    snapshots = [entry['snapshot'] for entry in entries]
    xlim = _global_plot_xlim(snapshots)
    n_panels = len(entries)

    fig = plt.figure(figsize=(panel_width, panel_height * n_panels))
    outer = GridSpec(n_panels, 1, figure=fig)

    ref_ax_data = None
    ref_ax_res = None
    res_axes_by_row = {}
    data_axes = []

    for index, entry in enumerate(entries):
        snapshot = entry['snapshot']
        inner = outer[index, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        if ref_ax_data is None:
            ax_data = fig.add_subplot(inner[0, 0])
            ax_res = fig.add_subplot(inner[1, 0], sharex=ax_data)
            ref_ax_data = ax_data
            ref_ax_res = ax_res
        else:
            ax_data = fig.add_subplot(inner[0, 0], sharex=ref_ax_data)
            ax_res = fig.add_subplot(inner[1, 0], sharex=ref_ax_res)

        pixspec = snapshot.get('pixspec', np.nan)
        if not np.isfinite(pixspec):
            lbd = snapshot['plot']['lbd']
            pixspec = float(np.median(np.diff(lbd))) if len(lbd) > 1 else 1.0

        plot_cak_panels(
            ax_data, ax_res, _snapshot_cak_result(snapshot), ylabel, pixspec,
            panel_label=format_verr_panel_label(entry['verr_kms'], snapshot),
            show_legend=False,
            show_ylabel=False,
            xlim=xlim,
            show_reference_template=False,
        )
        data_axes.append(ax_data)
        res_axes_by_row.setdefault(index, []).append(ax_res)

    for ax in data_axes:
        ax.set_ylabel('')

    for row, axes in res_axes_by_row.items():
        for ax in axes:
            ax.set_xlim(xlim)
            ax.set_xlabel('')
            format_cak_wavelength_axis(ax)
            if row < n_panels - 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis='both', labelsize=TICK_LABELSIZE)

    if ref_ax_data is not None:
        ref_ax_data.set_xlim(xlim)

    legend_handles, legend_labels = _legend_handles_from_snapshot(
        entries[0]['snapshot'], ylabel, show_reference_template=False,
    )
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center', bbox_to_anchor=(0.5, 0.02),
            ncol=2, fontsize=MULTIPANEL_LEGEND_FONTSIZE, frameon=True,
        )

    if title:
        fig.suptitle(title, fontsize=16, y=0.995)

    fig.tight_layout(rect=[0.06, 0.05, 1.0, 0.98 if title else 1.0])
    fig.supxlabel(r'Wavelength ($\mathrm{\AA}$)', fontsize=16)
    fig.supylabel(ylabel, fontsize=16)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print('Ca K verr diagnostic figure saved: %s (%d panels)' % (output_path, n_panels))

    return {'plot': snapshot['plot'], 'metrics': snapshot.get('metrics', {})}


def _global_plot_xlim(snapshots):
    """Shared wavelength limits spanning all snapshot plot windows."""
    xmin = np.inf
    xmax = -np.inf
    for snapshot in snapshots:
        lbd = snapshot['plot']['lbd']
        xmin = min(xmin, float(np.min(lbd)))
        xmax = max(xmax, float(np.max(lbd)))
    return (xmin, xmax)


# Pagination limits for :func:`plot_cak_multipanel` (rows x cols grid).
MULTIPANEL_MAX_ROWS = 5
MULTIPANEL_MAX_COLS = 3
MULTIPANEL_FIRST_PAGE_MAX_DATA = 14  # one cell reserved for the legend
MULTIPANEL_OTHER_PAGE_MAX_DATA = 15


def _multipanel_output_paths(output_path: str, n_pages: int):
    """Numbered output paths when multiple pages are written."""
    if n_pages <= 1:
        return [output_path]
    root, ext = os.path.splitext(output_path)
    return ['%s%d%s' % (root, page + 1, ext) for page in range(n_pages)]


def paginate_cak_snapshots(snapshots):
    """
    Split snapshots into pages for multi-panel Ca K figures.

    The first page reserves one grid slot for the legend (lower right), so it
    holds at most 14 data panels. Later pages hold up to 15 data panels each.
    """
    snapshots = list(snapshots)
    pages = []
    offset = 0
    page_index = 0
    while offset < len(snapshots):
        if page_index == 0:
            chunk = min(MULTIPANEL_FIRST_PAGE_MAX_DATA, len(snapshots) - offset)
            include_legend = True
        else:
            chunk = min(MULTIPANEL_OTHER_PAGE_MAX_DATA, len(snapshots) - offset)
            include_legend = False
        pages.append({
            'snapshots': snapshots[offset:offset + chunk],
            'include_legend': include_legend,
        })
        offset += chunk
        page_index += 1
    return pages


def _multipanel_page_layout(n_data, include_legend):
    """Grid shape and legend slot (lower-right cell) for one page."""
    n_slots = n_data + (1 if include_legend else 0)
    ncols = min(MULTIPANEL_MAX_COLS, max(1, n_slots))
    nrows = min(MULTIPANEL_MAX_ROWS, int(np.ceil(n_slots / ncols)))
    legend_slot = ((nrows - 1) * ncols + (ncols - 1)) if include_legend else None
    return nrows, ncols, legend_slot


def _multipanel_data_slots(n_data, nrows, ncols, legend_slot):
    """Row-major grid slots for data panels, skipping the legend cell."""
    slots = []
    for slot in range(nrows * ncols):
        if legend_slot is not None and slot == legend_slot:
            continue
        if len(slots) >= n_data:
            break
        slots.append(slot)
    return slots


def _plot_cak_multipanel_page(
    page_snapshots,
    output_path,
    include_legend,
    ylabel,
    xlim,
    legend_handles,
    legend_labels,
    dpi=150,
    panel_width=4.0,
    panel_height=3.2,
):
    """Render and save one paginated Ca K multi-panel figure."""
    page_snapshots = list(page_snapshots)
    n_data = len(page_snapshots)
    nrows, ncols, legend_slot = _multipanel_page_layout(n_data, include_legend)
    data_slots = _multipanel_data_slots(n_data, nrows, ncols, legend_slot)

    fig = plt.figure(figsize=(panel_width * ncols, panel_height * nrows))
    outer = GridSpec(nrows, ncols, figure=fig)

    ref_ax_data = None
    ref_ax_res = None
    panel_axes = []  # (ax_data, ax_res, row, col)

    for index, snapshot in enumerate(page_snapshots):
        slot = data_slots[index]
        row = slot // ncols
        col = slot % ncols
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        if ref_ax_data is None:
            ax_data = fig.add_subplot(inner[0, 0])
            ax_res = fig.add_subplot(inner[1, 0], sharex=ax_data)
            ref_ax_data = ax_data
            ref_ax_res = ax_res
        else:
            ax_data = fig.add_subplot(
                inner[0, 0], sharex=ref_ax_data, sharey=ref_ax_data,
            )
            ax_res = fig.add_subplot(
                inner[1, 0], sharex=ref_ax_data, sharey=ref_ax_res,
            )

        pixspec = snapshot.get('pixspec', np.nan)
        if not np.isfinite(pixspec):
            lbd = snapshot['plot']['lbd']
            pixspec = float(np.median(np.diff(lbd))) if len(lbd) > 1 else 1.0

        plot_cak_panels(
            ax_data, ax_res, _snapshot_cak_result(snapshot), ylabel, pixspec,
            panel_label=snapshot.get('panel_label', format_multipanel_label(snapshot)),
            show_legend=False,
            show_ylabel=False,
            xlim=xlim,
        )
        panel_axes.append((ax_data, ax_res, row, col))

    if ref_ax_data is not None:
        ref_ax_data.set_xlim(xlim)

    for ax_data, ax_res, row, col in panel_axes:
        ax_data.set_ylabel('')
        ax_res.set_xlabel('')
        format_cak_wavelength_axis(ax_res)
        # X labels only on the bottom grid row; y labels only on the left column.
        # Use labelbottom/labelleft (not set_*ticklabels) so sharex/sharey stay intact.
        show_x = row == nrows - 1
        show_y = col == 0
        ax_data.tick_params(
            axis='both', labelsize=TICK_LABELSIZE,
            labelbottom=False, labelleft=show_y,
        )
        ax_res.tick_params(
            axis='both', labelsize=TICK_LABELSIZE,
            labelbottom=show_x, labelleft=show_y,
        )

    if include_legend and legend_slot is not None and legend_handles:
        legend_row = legend_slot // ncols
        legend_col = legend_slot % ncols
        inner = outer[legend_row, legend_col].subgridspec(1, 1)
        ax_legend = fig.add_subplot(inner[0, 0])
        ax_legend.axis('off')
        ax_legend.legend(
            legend_handles, legend_labels, loc='center',
            fontsize=MULTIPANEL_LEGEND_FONTSIZE,
        )

    fig.tight_layout(rect=[0.05, 0.04, 1.0, 1.0])
    fig.supxlabel(r'Wavelength ($\mathrm{\AA}$)', fontsize=16)
    fig.supylabel(ylabel, fontsize=16)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_cak_multipanel(
    snapshots,
    output_path,
    ncols=None,
    dpi=150,
    panel_width=4.0,
    panel_height=3.2,
    ylabel=None,
):
    """
    Save one or more multi-panel Ca K figures from loaded plot snapshots.

    Large sets are split across multiple output files with at most 5 rows and
    3 columns per figure. The first figure reserves the lower-right cell for
    the legend (14 data panels maximum); later figures contain data panels only.

    When more than one file is written, numbered suffixes are inserted before
    the extension (e.g. ``cak_stacks.png`` -> ``cak_stacks1.png``,
    ``cak_stacks2.png``).

    Panels are ordered left-to-right, then top-to-bottom, in the caller's
    snapshot order (typically increasing redshift).

    ``ncols`` is accepted for backward compatibility and ignored; layout is
    paginated with at most 3 columns.

    Returns the list of output paths written.
    """
    snapshots = list(snapshots)
    if not snapshots:
        raise ValueError('No Ca K plot snapshots to display.')

    if ncols is not None:
        pass  # ignored; see docstring

    ylabel = ylabel or next(
        (snap.get('flux_unit') for snap in snapshots if snap.get('flux_unit')),
        r'Flux',
    )
    xlim = _global_plot_xlim(snapshots)
    pages = paginate_cak_snapshots(snapshots)
    output_paths = _multipanel_output_paths(output_path, len(pages))

    legend_handles = None
    legend_labels = None
    if pages and pages[0]['snapshots']:
        first = pages[0]['snapshots'][0]
        pixspec = first.get('pixspec', np.nan)
        if not np.isfinite(pixspec):
            lbd = first['plot']['lbd']
            pixspec = float(np.median(np.diff(lbd))) if len(lbd) > 1 else 1.0
        tmp_fig, tmp_axes = plt.subplots(2, 1)
        try:
            plot_cak_panels(
                tmp_axes[0], tmp_axes[1],
                _snapshot_cak_result(first), ylabel, pixspec,
                show_legend=True,
                show_ylabel=False,
                show_reference_template=False,
                show_centroid=False,
            )
            legend_handles, legend_labels = tmp_axes[0].get_legend_handles_labels()
        finally:
            plt.close(tmp_fig)

    written = []
    for page, path in zip(pages, output_paths):
        _plot_cak_multipanel_page(
            page['snapshots'],
            path,
            include_legend=page['include_legend'],
            ylabel=ylabel,
            xlim=xlim,
            legend_handles=legend_handles,
            legend_labels=legend_labels,
            dpi=dpi,
            panel_width=panel_width,
            panel_height=panel_height,
        )
        written.append(path)
        print(
            'Ca K multi-panel figure saved: %s (%d panel%s%s)' % (
                path,
                len(page['snapshots']),
                '' if len(page['snapshots']) == 1 else 's',
                ', with legend' if page['include_legend'] else '',
            ),
        )
    return written

def PlotSpec(
    ax, lbd, f, ferr=None, pixspec=None,
    oprpix=None, PLOTERR=True,
    XLABEL=True, YLABEL=True,
    fscale=1., lbdscale=1., fmt='k-', label='',
    linewidth=0.3, erralpha=0.2,
):
    """
    Plot a spectrum (and optional error band) on ``ax``.

    Legacy IronFit helper used by Ca K panels. Arrays should not contain NaNs.
    If ``pixspec`` is set, the error fill skips wavelength gaps larger than one
    pixel. ``oprpix`` may mark interpolated pixels (highlighted in orange).
    """
    ax.plot(lbd, f, fmt, linewidth=linewidth, label=label)
    if PLOTERR:
        wherefill = None
        if pixspec is not None:
            wherefill = np.concatenate((abs(np.diff(lbd) - pixspec) < 1e-3, [True]))
        if ferr is not None:
            ax.fill_between(
                lbd, f - ferr, f + ferr, where=wherefill, alpha=erralpha,
                facecolor=fmt[0], edgecolor=fmt[0],
            )
        if oprpix is not None:
            ax.fill_between(
                lbd, f - ferr, f + ferr, where=(np.array(oprpix) == 'interp'),
                alpha=0.2, facecolor='tab:orange', edgecolor='tab:orange',
            )

    if XLABEL:
        ax.set_xlabel(r'Wavelength (%.0e $\AA$)' % (1. / lbdscale), fontsize=12)
    if YLABEL:
        ax.set_ylabel(r'%.0e erg/s/cm$^2$/$\AA$' % (1. / fscale), fontsize=12)


