#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats` module contains functions to perform exploratory data analysis.

The following submodules are currently implemented:

- The :mod:`~araucaria.stats.genesd` module contains functions to detect outliers in a data array.
- The :mod:`~araucaria.stats.smooth` module contains low-pass filter functions.
"""
from .genesd import genesd, find_ri, find_critvals
from .smooth import roll_med
