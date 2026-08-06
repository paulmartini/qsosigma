qsosigma
========

Introduction
------------

Tools to measure the stellar velocity dispersion of a quasar host galaxy.

The code focuses on the Ca II K absorption line and has been tested with stacked 
DESI quasar spectra. 

There is a validation mode that will run on multiple stacks of the same quasar sample 
that have had different amounts of redshift errors added at the catalog level before stacking. 
The validation tests if the code recovers the expected increase in velocity dispersion due to 
the artificially increased redshift errors. 

The quasar stacking code is not part of this repository.

There are a number of plotting scripts.

Dependencies
------------

- Python >= 3.9
- numpy, scipy, astropy, matplotlib

Installation
------------

From a clone of this repository (recommended editable install)::

    pip install -e .

    # optional: test extras
    pip install -e ".[test]"

This installs the ``qsosigma`` package from ``py/``. CLI scripts live in
``bin/``; add that directory to your ``$PATH``, or run them as
``python bin/run_cakfit.py ...``. The scripts add the checkout ``py/`` directory
to ``sys.path`` automatically when needed.

Alternatively, without pip::

    export PYTHONPATH="/path/to/qsosigma/py:$PYTHONPATH"
    export PATH="/path/to/qsosigma/bin:$PATH"

Stellar templates are under ``data/stellar_templates/`` (see that directory's
README). Override the location with ``--stellar-template-dir`` if needed.

Running Ca K fits
-----------------

The main script is **bin/run_cakfit.py**::

    # Single spectrum (auto-pick best χ² template for the point estimate)
    python bin/run_cakfit.py qsospec.fits

    # Lock the reporting template; ensemble still used for 16–84 uncertainties
    python bin/run_cakfit.py qsospec.fits --cak-template hd138688

    # Fit verr0 plus verr-injected stacks for one redshift bin
    # Default output: cak_fitresults_z0.050_z0.100.fits (override with -o)
    python bin/run_cakfit.py --validate \
      --verr-root /path/to/verrtests --zlo 0.05 --zhi 0.10

    python bin/run_cakfit.py --help

``--cak-template`` / the locked template may be disabled in the manifest; a
warning is issued. Uncertainties are the **16–84 percentile half-range**
over enabled templates after culling failed fits (σ* within 5 km/s of the
active bounds, or depth ≤ 0.02). The locked template is always kept. If only
it remains, the error is NaN and a warning is printed.

Default products are named ``cak_fitresults_z{zlo}_z{zhi}.fits``. With
``--validate``, the code also fits verr-injected stacks, writes
``VERR*`` / ``TPL*`` / ``CAKPLOT*`` extensions for each level, and saves
``cak_verr_diagnostic_z{zlo}_z{zhi}.png`` (use ``--no-plot`` to skip). The σ*
lower bound is raised to ``max(20, verr)`` km/s on those stacks.::

    python bin/plot_cak_verr_diagnostic.py cak_fitresults_z0.250_z0.300.fits
    python bin/plot_cak_multipanel.py cak_fitresults*.fits -o cak_spectra.png

(``plot_cak_multipanel.py`` uses the verr0 panel from each multi-stack file.)

Template format, manifest, and building new templates:
``data/stellar_templates/README.md``.

Tour of the Code
----------------

**bin/** : command-line scripts

**py/qsosigma** : importable package

**data/stellar_templates** : stellar templates and manifest

**tests/** : lightweight unit tests (``pytest``)

Acknowledgements
----------------

Parts of the code were constructed with help from the Cursor AI-powered code editor and GitHub Copilot. The template tools were originally developed for ``IronFit`` [see Yu, Martini, et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.3771Y/abstract).

