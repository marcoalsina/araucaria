#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xrdb.xray` module offers the following 
routines to access x-ray tabulated data:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~araucaria.xrdb.xray.edge_energy`
     - Returns the x-ray edge energy for a given element.
   * - :func:`~araucaria.xrdb.xray.nearest_edge`
     - Returns nearest x-ray edge for a given energy.

Absorption edge energies are taken from data compiled by Elam et al. [1]_
    
References
----------
.. [1] W.T. Elam, B.D. Ravel, J.R. Sieber (2002) 
   "A new atomic database for X-ray spectroscopic calculations",
   Radiation Physics and Chemistry 63(2): pp 121-128, 
   `DOI:10.1016/S0969-806X(01)00227-4 <https://doi.org/10.1016/S0969-806X(01)00227-4>`_.
"""
from typing import List, Tuple
from os import pardir
from os.path import join, dirname
from sqlite3 import connect
from .chem import symtoz

# FILE PATH FOR ELAM DATABASE
# ---------------------------
dbpath = join(dirname(__file__), 'data', 'elam.db') 

def edge_energy(sym: str, edge: str='K') ->float:
    """Returns the x-ray edge energy for a given element.

    Parameters
    ----------
    sym
        Atomic symbol.
    edge
        Absorption edge in Siegbanh notation.
        The default is 'K'.

    Returns
    -------
    :
        X-ray edge energy in eV.

    Raises
    ------
    NameError
        If ``sym`` is not available.
    NameError
        If ``edge`` is not available for a given ``sym``.

    Example
    -------
    >>> from araucaria.xrdb import edge_energy
    >>> syms = ['Fe', 'Cu', 'Zn', 'As']
    >>> for s in syms:
    ...     energy = edge_energy(s, edge='K')
    ...     print('{0:2} K-edge: {1:7} eV'.format(s, energy))
    Fe K-edge:  7112.0 eV
    Cu K-edge:  8979.0 eV
    Zn K-edge:  9659.0 eV
    As K-edge: 11867.0 eV
    """
    # connecting to database
    conn   = connect(dbpath)
    cur    = conn.cursor()

    # executing search
    exe_str= "SELECT energy FROM edges WHERE symbol==? and edge==?"
    cur.execute(exe_str, (sym, edge))

    # saving search result
    result = cur.fetchone()
    if result is None:
        raise NameError('symbol: %s not available.' % sym)
    else:
        result = result[0]

    # closing database and returning value
    conn.close()
    return result

def nearest_edge(energy: float)-> Tuple[str, str]:
    """Return nearest x-ray edge for a given energy.

    Parameters
    ----------
    energy
        X-ray energy in eV.

    Returns
    -------
    :
        Absorption element.
    :
        Absorption edge.

    Raises
    ------
    ValueError
        If ``energy`` is not a number.

    Example
    -------
    >>> from araucaria.xrdb import nearest_edge
    >>> energies = [7115, 8980, 9660, 11870]
    >>> for ener in energies:
    ...     print(nearest_edge(ener))
    ('Fe', 'K')
    ('Cu', 'K')
    ('Zn', 'K')
    ('As', 'K')
    """
    # validating number
    try:
        float(energy)
    except:
        raise ValueError('energy: %s is not a number.' % energy)
    
    # connecting to database
    conn   = connect(dbpath)
    cur    = conn.cursor()

    # executing search
    exe_str= "SELECT symbol, edge FROM edges ORDER BY ABS(? - energy);"
    cur.execute(exe_str, [energy])

    # saving search result
    result = cur.fetchone()
    if result is None:
        raise ValueError('edge near energy: %s not available.' % energy)
    else:
        pass

    # closing database and returning value
    conn.close()
    return result

if __name__ == '__main__':
    import doctest
    doctest.testmod()