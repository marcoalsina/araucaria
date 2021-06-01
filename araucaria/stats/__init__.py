#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats` module contains functions to perform data filtering
and exploratory data analysis.

The following submodules are currently implemented:

- The :mod:`~araucaria.stats.genesd` module contains functions to detect outliers in a data array.
- The :mod:`~araucaria.stats.window` module contains functions to filter a data array.
- The :mod:`~araucaria.stats.eda` module contains functions to perform exploratory data analysis.
"""
from .genesd import genesd, find_ri, find_critvals
from .window import roll_med
from .eda import cluster
