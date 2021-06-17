#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats` module contains functions to perform data filtering
and exploratory data analysis.

The following submodules are currently implemented:

- The :mod:`~araucaria.stats.genesd` module contains functions to detect outliers in a data array.
- The :mod:`~araucaria.stats.window` module contains functions to filter a data array.
- The :mod:`~araucaria.stats.cluster` module contains functions to perform clustering.
- The :mod:`~araucaria.stats.pca` module contains classes and functions to perform principal component analysis.
"""
from .genesd import genesd, find_ri, find_critvals
from .window import roll_med
from .cluster import get_mapped_data, cluster
from .pca import PCAModel, pca, target_transform
