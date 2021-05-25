#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.plot` module offers classes and functions to preset ``Matplotlib`` axes for
plotting XAS spectra. The following functions are currently implemented:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~araucaria.plot.template.fig_xas_template`
     - Presets a plot of XAS spectra.
   * - :func:`~araucaria.plot.fig_merge.fig_merge`
     - Plot the results of a merge operation.
   * - :func:`~araucaria.plot.fig_pre_edge.fig_pre_edge`
     - Plot the results of pre-edge substratction and normalization.
   * - :func:`~araucaria.plot.fig_autobk.fig_autobk`
     - Plot the results of background removal.
"""
from .template import FigPars, fig_xas_template
from .fig_merge import fig_merge
from .fig_pre_edge import fig_pre_edge
from .fig_autobk import fig_autobk
from .fig_lcf import fig_lcf
from .fig_lsf import fig_lsf