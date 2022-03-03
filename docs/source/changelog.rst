Changelog
=========

ver 0.1.12
----------
- :mod:`~.fit.feffit` submodule added: support for Feffit on EXAFS spectra (:mod:`~araucaria.fit` module).
- binning functions moved to (:mod:`~araucaria.xas.xasutils` module).
- added :func:`~.utils.check_dictkeys` function (:mod:`~araucaria.utils` module).
- added :func:`~.utils.check_maxminval` and :func:`~.utils.check_minmaxval` functions (:mod:`~araucaria.utils` module).
- added :func:`~.utils.count_decimals` function (:mod:`~araucaria.utils` module).
- updated labels for EXAFS plots in :func:`~.plot.template.fig_xas_template` function (:mod:`~.plot` module).
- fixed bug in exception of :func:`~.utils.check_dictkeys` function (:mod:`~araucaria.utils` module).


ver 0.1.11
----------
- :mod:`~.stats.signal` submodule added: functions to filter and analyze univariate data.
- :mod:`~.linalg` module added: linear algebra operations on collections.
- :mod:`~.xrdb` module added: basic support for x-ray databases.
- added :func:`~.utils.read_fdicts` function (:mod:`~araucaria.utils` module).
- fixed bug in derivative computation of :func:`~.xasutils.get_mapped_data` function (thanks to M. Desmau).
- updated tutorials.