#!/usr/bin/env python
"""
The :mod:`~araucaria.fit` module offers functions to analyze XAFS spectra by means of
linear combination fitting (LCF) with known reference scans, or by 
non-linear least-squares fitting (LSF) considering ab-initio models (e.g. Feffit).

The following submodules are currently implemented:

- The :class:`~araucaria.fit.lcfit` module contains routines to perform LCF analysis.
"""
from .lcfit import lcf, lcf_report, sum_references, residuals
