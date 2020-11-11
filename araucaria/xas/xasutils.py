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
     - Converts photo-electron energy to wavenumber.
   * - :func:`ktoe`
     - Converts photo-electron wavenumber to energy.
"""

from numpy import ndarray
from scipy.constants import hbar  # reduced planck constant
from scipy.constants import m_e   # electron mass
from scipy.constants import eV    # electron volt in joules

# constants
# 1e10 converts from 1/meter to 1/angstrom
k2e = (1e10 * hbar)**2 / (2 * m_e * eV)

def etok(energy: ndarray) -> ndarray:
    """Converts photo-electron energy to wavenumber.

    Parameters
    ----------
    energy
        Array of photo-electron energies (eV).
    
    Returns
    -------
    :
        Array of photo-electron wavenumbers (:math:`Å^{-1}`).

    Notes
    -----
    Conversion is performed considering the non-relativistic
    approximation:

    .. math::

        k = \\frac{\sqrt{2mE}}{\hbar}  

    Where

    - :math:`k`: photo-electron wavenumber.
    - :math:`E`: kinetic energy of the photo-electron.
    - :math:`m`: electron mass.
    - :math:`\hbar`: reduced Planck constant.
    
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
    """Converts photo-electron wavenumber to energy.
    
    Parameters
    ----------
    k
        Array with photo-electron wavenumbers (:math:`Å^{-1}`).
    
    Returns
    -------
    :
        Array with photo-electron energies (eV).

    See also
    --------
    etok : Converts photo-electron energy to wavenumber.

    Example
    -------
    >>> from araucaria.xas import ktoe
    >>> k = 10      # A^{-1}
    >>> e = ktoe(k)  # eV
    >>> print('%1.5f' % e)
    380.99821
    """
    return k**2*k2e
