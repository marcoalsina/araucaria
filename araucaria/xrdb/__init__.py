#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xrdb` module offers routines to access chemical and x-ray databases.

The following submodules are currently implemented:

- The :mod:`~araucaria.xrdb.chem` module allows access to chemical data and formula parser.
- The :mod:`~araucaria.xrdb.xray` module allows acces to x-ray data.
"""
from .chem import ztosym, symtoz, at_weight, formula_parser, formula_weight
from .xray import edge_energy, nearest_edge