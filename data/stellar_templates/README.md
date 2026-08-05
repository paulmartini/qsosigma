# Ca II K stellar absorption templates

These files supply **rest-frame stellar absorption templates** for the Ca K fit in
`run_cakfit.py` (single-spectrum and verr validation modes).

## Directory layout

```
stellar_templates/
  templates.manifest.csv   # which templates are active
  README.md
  empirical/               # real templates built with build_stellar_templates.py
```

## Template format

Each CSV has two columns:

| column | meaning |
|--------|---------|
| `wavelength` | Rest wavelength (Angstrom) |
| `absorption` | Fractional absorption, 0–1, relative to a pseudo-continuum |

The fit model is:

```text
F(λ) = C(λ) × [1 − depth × T(λ; v, σ*)]
```

`C(λ)` is a power-law continuum initialized from the sidebands and refined
jointly with velocity shift, stellar dispersion σ*, and depth. (A hard continuum
freeze systematically underestimates σ* on QSO stacks.)

On load, templates are:

1. Sanitized (non-finite absorption → 0)
2. Shifted so the Ca K absorption peak lies at **3933.663 Å**
3. Resampled onto a uniform `ln(λ)` grid for velocity-space convolution

Templates should span **3800–4100 Å** so they include **Ca II K (3933.663 Å)**,
**Ca II H (3968.463 Å)**, and nearby features. The fitter uses **Ca K only** for
χ², but the wide plot window shows Ca H and other lines.

## Dispersion metrics

| column | meaning |
|--------|---------|
| `CAK_STELLAR_DISP` (σ*) | Fitted stellar / kinematic Gaussian broadening. The DESI instrumental LSF is included in the **forward model**, so σ* is LSF-corrected. **Use this** for science and for redshift-error tests √(σ₀² + σ_zerr²). |
| `CAK_STELLAR_DISP_TOTAL` | Total kernel width √(σ*² + σ_inst² + σ_template_LSF²). Diagnostic only; not the quantity for quadrature redshift-error tests. |
| `CAK_AT_BOUND` | 1 if σ* sits at the fit bound (20–750 km/s), else 0. Warnings are also printed and written to the FITS header. |

## Running Ca K fits

See the repository root [README.md](../../README.md) for how to run
`run_cakfit.py`, validation mode, and plotting scripts.

## Building real templates

Use `build_stellar_templates.py` at the repository root.

Most of the existing templates originate from the UVES-POP program. To add more, download original resolution files from https://sl.voxastro.org/library/UVES-POP. 

Here is how to create a template from a spectrum

```bash
python build_stellar_templates.py /HD107446_R80k.fits \
  --name hd107446 \
  --label "UVES-POP HD107446 K3III" \
  --spectral-type K3III \
  --fe-h -0.398 \
  --source "UVES-POP HD107446 (R~80000)"
```

Batch mode (`stars.csv` columns: `input_path,name,label[,spectral_type,fe_h,source,subdir,enabled]`):

```bash
python build_stellar_templates.py --batch stars.csv
```

List enabled templates:

```bash
python build_stellar_templates.py --list
```

### Pseudo-continuum regions (defaults)

| region | wavelength (Å) | purpose |
|--------|----------------|---------|
| Blue | 3820–3880 | line-free continuum blueward of Ca K |
| Red | 3990–4050 | line-free continuum redward of Ca H |
| Template range | 3800–4100 | saved CSV coverage |

Override with `--blue-cont`, `--red-cont`, and `--wave-range`.

Input spectra must be **rest frame** (e.g. MILES). Supported formats:

- **MILES FITS**: flux in `PRIMARY`, wavelength from `CRVAL1` / `CDELT1` / `CRPIX1`
- **DESI-style FITS**: separate `WAVE` and `FLUX` HDUs
- **Table FITS**: binary table columns such as `wavelength` and `flux`
- **ASCII**: whitespace columns `wavelength flux`

## Manifest

`templates.manifest.csv` controls which templates are fit. Columns:

- `name` — unique identifier
- `filename` — path relative to this directory
- `label` — descriptive label
- `spectral_type`, `fe_h`, `source` — metadata (optional)
- `enabled` — `true` / `false`

Only rows with `enabled=true` are used in the ensemble by default.  Use
`--cak-template` (or validation locking) to force a reporting template,
including disabled rows.

## Fit vs plot windows

| window | approximate range (Å) | use |
|--------|----------------------|-----|
| Ca K fit | 3906–3962 | χ² for σ*, centroid, depth |
| Blue continuum | 3889–3922 | power-law continuum |
| Red continuum | 3980–4008 | power-law continuum (after Ca H) |
| Plot | 3864–4008 | `{stem}_cak.png` diagnostics |

Ca H is **not** included in the continuum sidebands or the primary fit window.
