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
   * - :func:`~araucaria.plot.fig_xas.fig_merge`
     - Plot the results of a merge operation.
   * - :func:`~araucaria.plot.fig_xas_edge.fig_pre_edge`
     - Plot the results of pre-edge substraction and normalization.
   * - :func:`~araucaria.plot.fig_xas.fig_autobk`
     - Plot the results of background removal.
   * - :func:`~araucaria.plot.fig_stats.fig_cluster`
     - Plots the dendrogram of a hierarchical clustering.
   * - :func:`~araucaria.plot.fig_stats.fig_pca`
     - Plots the results of principal component analysis.
   * - :func:`~araucaria.plot.fig_stats.fig_target_transform`
     - Plots the results of target transformation.
   * - :func:`~araucaria.plot.fig_lcf.fig_lcf`
     - Plot the results of a linear combination fit.
"""
from .template import FigPars, fig_xas_template
from .fig_xas import fig_merge, fig_pre_edge, fig_autobk
from .fig_lcf import fig_lcf
from .fig_lsf import fig_lsf
from .fig_stats import fig_cluster, fig_pca, fig_target_transform