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

```bash
# Single spectrum (auto-pick best χ² template for the point estimate)
python run_cakfit.py qsospec.fits

# Lock the reporting template; ensemble still used for 16–84 uncertainties
python run_cakfit.py qsospec.fits --cak-template hd138688

# Fit verr0 plus verr-injected stacks for one redshift bin
# Default output: cak_fitresults_z0.050_z0.100.fits (override with -o)
python run_cakfit.py --validate \
  --verr-root /path/to/verrtests --zlo 0.05 --zhi 0.10
```

`--cak-template` / the locked template may be disabled in the manifest; a
warning is issued. Uncertainties are the **16–84 percentile half-range**
over enabled templates after culling failed fits (σ* within 5 km/s of the
active bounds, or depth ≤ 0.02). The locked template is always kept. If only
it remains, the error is NaN and a warning is printed.

Default products are named `cak_fitresults_z{zlo}_z{zhi}.fits`. With
`--validate`, the code also fits verr-injected stacks, writes
`VERR*` / `TPL*` / `CAKPLOT*` extensions for each level, and saves
`cak_verr_diagnostic_z{zlo}_z{zhi}.png` (use `--no-plot` to skip). The σ* lower
bound is raised to `max(20, verr)` km/s on those stacks.

```bash
python plot_cak_verr_diagnostic.py cak_fitresults_z0.250_z0.300.fits
python plot_cak_multipanel.py "cak_fitresults_*.fits" -o cak_spectra.png
```

(`plot_cak_multipanel.py` uses the verr0 panel from each multi-stack file.)

## Building real templates

Use `build_stellar_templates.py` at the repository root.

Single spectrum:

```bash
python build_stellar_templates.py /path/to/miles_star.fits \
  --name miles_1234_K3III \
  --label "MILES 1234 K3III" \
  --spectral-type K3III \
  --fe-h 0.0 \
  --source "MILES v3.1 ID 1234"
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
