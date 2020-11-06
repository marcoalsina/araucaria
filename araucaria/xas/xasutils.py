#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xas.xasutils` module offers the following XAFS utility functions :

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`etok`
     - Converts energies to wavenumbers.
   * - :func:`ktoe`
     - Converts wavenumbers to energies.
"""

from numpy import ndarray
from scipy.constants import hbar    # reduced planck constant
from scipy.constants import m_e, e  # electron mass and and elementary charge

# constants
k2e = 1.e20 * hbar**2 / (2 * m_e * e)
e2k = 1/k2e

def etok(energy: ndarray) -> ndarray:
    """Converts photo-electron energies to wavenumbers.
    
    Parameters
    ----------
    energy
        Array of photo-electron energies.
    
    Returns
    -------
    :
        Arary of photo-electron wavenumbers.
    
    Example
    -------
    >>> from araucaria.xas import etok
    >>> e = 400      # eV
    >>> k = etok(e)  # A^{-1}
    >>> print('%1.5f' % k)
    10.24633
    """
    from numpy import sqrt
    return sqrt(energy/k2e)

def ktoe(k: ndarray) -> ndarray:
    """Converts photo-electron wavenumbers to energies.
    
    Parameters
    ----------
    k
        Array with photo-electron wavenumbers.
    
    Returns
    -------
    :
        Array with photo-electron energies.

    Example
    -------
    >>> from araucaria.xas import ktoe
    >>> k = 10      # A^{-1}
    >>> e = ktoe(k)  # eV
    >>> print('%1.5f' % e)
    380.99821
    """
    return k**2*k2e
