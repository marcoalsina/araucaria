#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.main` module contains the core classes of the library:

- The :class:`~araucaria.main.group.Group` class stores a dataset from a single XAFS scan.
- The :class:`~araucaria.main.collection.Collection` class is stores and operate on multiple groups.
- The :class:`~araucaria.main.dataset.Dataset` class stores a dataset from analysis of XAFS spectra.
- The :class:`~araucaria.main.report.Report` class  provides a simple framework to print and manipulate dataset information.
"""
from .group import Group
from .dataset import Dataset
from .report import Report
from .collection import Collection
