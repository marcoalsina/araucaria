#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats` module contains functions to perform data filtering
and exploratory data analysis.

The following submodules are currently implemented:

- The :mod:`~araucaria.stats.genesd` module contains functions to detect outliers in a data array.
- The :mod:`~araucaria.stats.cluster` module contains functions to perform clustering.
- The :mod:`~araucaria.stats.pca` module contains classes and functions to perform principal component analysis.
- The :mod:`~araucaria.stats.signal` module contains functions to filter and analyze univariate data.
"""
from .genesd import genesd, find_ri, find_critvals
from .cluster import get_mapped_data, cluster
from .pca import PCAModel, pca, target_transform
from .signal import compute_bins, rebin, roll_med
