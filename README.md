qsosigma
========

Introduction
------------

Tools to measure the stellar velocity dispersion of quasar host galaxies. 

The code currently focuses on just the CaK absorption line and has been tested with stacked DESI quasar spectra. 

There are also validation tools and plotting scripts.

Installation
------------

To install, clone the repository, add the "py" directory to your $PYTHONPATH, and the "bin" directory to your $PATH. 

Getting Started
---------------

The main script to measure the CaK absorption line is **bin/run_cakfit.py**. 

Use the **--help** option for further documentation. 

Tour of the Code
----------------

Here is a brief description of the repository layout

**bin/** : command-line scripts

**py/qsosigma** : packages

**data/stellar_templates** : stellar templates


Acknowledgements
----------------

Parts of the code were constructed with help from the Cursor AI-powered code editor and github Copilot. 

