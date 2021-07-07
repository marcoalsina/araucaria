#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.linalg` module offers the following routines to
perform linear algebra operations:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~araucaria.linalg.la.cond_num`
     - Computes the condition number of a collection.
   * - :func:`~araucaria.linalg.la.matrix_rank`
     - Computes the matrix rank of a collection.
   * - :func:`~araucaria.linalg.la.imd`
     - Computes the interpolative matrix decomposition of a collection.
"""
from .la import cond_num, matrix_rank, imd
