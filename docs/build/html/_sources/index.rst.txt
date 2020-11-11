.. araucaria documentation master file, created by
   sphinx-quickstart on Tue Nov 19 09:26:07 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for araucaria
===========================

``araucaria`` is a Python library to read, process and analyze X-ray absorption fine structure 
(XAFS) spectra. The library is designed to be modular, transparent, and light-weight, allowing 
the development of routines that are reproducible, exchangeable, and readily extensible.

The library is under active develeopment, but in its current state allows to perform both
routine and advanced tasks on XAFS spectra such as linear combination fitting (LCF). 
Additional functionality will be added on a regular basis.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   install

.. toctree::
   :maxdepth: 2
   :caption: Modules

   main_module
   io_module
   xas_module
   fit_module
   plot_module
   utils_module
   testdata_module

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
