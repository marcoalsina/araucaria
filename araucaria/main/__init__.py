#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.main` module contains the core classes of the library:

- The :class:`~araucaria.main.group.Group` class is designed to store and operate a dataset from a single XAFS scan.
- The :class:`~araucaria.main.group.DatGroup` class is designed to store a dataset from analysis of XAFS spectra.
- The :class:`~araucaria.main.group.FitGroup` class is designed to store a dataset from a fitted XAFS scan.
- The :class:`~araucaria.main.collection.Collection` class is designed to store and operate on multiple group datasets.
- The :class:`~araucaria.main.report.Report` class  provides a simple framework to print dataset information to ``sys.stdout``.
"""
from .group import Group, DatGroup, FitGroup
from .report import Report
from .collection import Collection
