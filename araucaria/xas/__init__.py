#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xas` module contains the main functions to manipulate XAFS spectra.

The following submodules are currently implemented:

- The :mod:`~araucaria.xas.merge` module contains functions to pre-process and merge spectra.
- The :mod:`~araucaria.xas.deglitch` module contains an algorithm to automatically deglitch a spectrum.
- The :mod:`~araucaria.xas.normalize` module contains functions to normalize a spectrum.
- The :mod:`~araucaria.xas.autobk` module contains the Autobk algorithm for background removal of a spectrum.
- The :mod:`~araucaria.xas.xasft` module contains functions to perform Fourier transforms on a spectrum.
- The :mod:`~araucaria.xas.xasutils` module contains utility functions to assist manipulation of spectra.
"""

from .merge import calibrate, align, merge
from .deglitch import deglitch
from .normalize import find_e0, guess_edge, pre_edge
from .autobk import autobk
from .xasft import ftwindow, xftf, xftr, xftf_kwin, xftr_kwin
from .xasutils import etok, ktoe, get_mapped_data
