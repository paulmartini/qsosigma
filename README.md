========
qsosigma
========

Introduction
------------

Tools to measure the stellar velocity dispersion of quasar host galaxies.

The code currently focuses on the Ca II K absorption line and has been tested
with stacked DESI quasar spectra. Validation tools and plotting scripts are
included.

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

Getting Started
---------------

The main script is **bin/run_cakfit.py**::

    python bin/run_cakfit.py stack.fits --cak-template hd138688
    python bin/run_cakfit.py --help

Ca K template documentation and further examples:
``data/stellar_templates/README.md``.

Tour of the Code
----------------

**bin/** : command-line scripts

**py/qsosigma** : importable package

**data/stellar_templates** : stellar templates and manifest

**tests/** : lightweight unit tests (``pytest``)

Acknowledgements
----------------

Parts of the code were constructed with help from the Cursor AI-powered code editor and GitHub Copilot.
