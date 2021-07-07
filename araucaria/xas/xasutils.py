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
   * - :func:`get_mapped_data`
     - Returns data mapped to a common domain in a collection.
"""
from scipy.constants import hbar  # reduced planck constant
from scipy.constants import m_e   # electron mass
from scipy.constants import eV    # electron volt in joules

from typing import List, Tuple
from numpy import ndarray, array, std, gradient, inf
from scipy.interpolate import interp1d
from .. import Group, Collection
from ..utils import check_objattrs, index_xrange

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

def get_mapped_data(collection: Collection, taglist: List[str]=['all'],
                    region: str='xanes', domain: ndarray=None,
                    range: list=[-inf,inf], kweight: int=2)-> Tuple[ndarray, ndarray]:
    """Returns data mapped to a common domain in a collection.

    Parameters
    ----------
    collection
        Collection with group datasets.
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].
    region
        XAS region to perform mapping. Accepted values are 'dxanes',
        'xanes', 'exafs'. The default is 'xanes'.
    domain
        Domain values to perform mapping. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'. It overrides the ``range`` parameter.
    range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    kweight
        Exponent for weighting chi(k) by k^kweight.
        Only valid for ``region='exafs'``.
        The default is 2.  

    Returns
    -------
    :
        1-D array containing the domain values for mapping.
    :
        Array containing in each column the mapped values of groups.
    Raises
    ------
    TypeError
        If ``collection`` is not a valid Collection instance.
    AttributeError
        If ``collection`` has no ``tags`` attribute.
    AttributeError
        If groups in ``collection`` have no ``energy`` or ``norm`` attribute.
        Only verified if ``region=dxanes`` or ``region=xanes``.
    AttributeError
        If groups in ``collection`` have no ``k`` or ``chi`` attribute.
        Only verified if ``region='exafs'``.
    KeyError
        If items in ``taglist`` are not keys of the ``tags`` attribute.
    ValueError
        If ``region`` is not recognized.
    ValueError
        If ``range`` or ``domain`` is outside the domain of a group 
        in ``collection``.

    Notes
    -----
    Exploratory data analysis assumes that data is mapped to a common domain range.
    However, datasets in a Collection are often mapped to different domain ranges.

    :func:`get_mapped_data` establishes a common domain based on the specified 
    ``region`` and ``range`` variables,  and returns an  array with the interpolated 
    values.
    
    Alternatively, if a ``domain`` array is provided, :func:`get_mapped_data` returns
    an array with the interpolated values.

    Example
    -------
    >>> from numpy import allclose
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.xas import pre_edge
    >>> from araucaria.stats import get_mapped_data
    >>> from araucaria.io import read_collection_hdf5
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> collection.apply(pre_edge)
    >>> ener, data = get_mapped_data(collection, region='xanes')
    >>> allclose(ener.shape[0], data.shape[0])
    True
    
    >>> # passing a domain argument
    >>> nener, ndata = get_mapped_data(collection, region='xanes', 
    ...                                domain=ener)
    >>> allclose(data, ndata)
    True
    """
    # checking class and attributes
    check_objattrs(collection, Collection, attrlist=['tags'], exceptions=True)

   # verifying region type
    region_valid = ['dxanes', 'xanes','exafs']
    if region not in region_valid:
        raise ValueError('region %s not recognized.'% region)

    # list of groupnames
    listgroups = collection.get_names(taglist=taglist)

    # checking if domain was provided as argument
    # if not, it will be constructued.
    if domain is None:
        # report parameters for exafs mapping
        # since k-range may not have been computed, the first group is used
        if region == 'exafs':
            dummy = collection.get_group(listgroups[0]).copy()
            check_objattrs(dummy, Group, attrlist=['k', 'chi'], exceptions=True)
            xvals = dummy.k
        # report paramters for xanes mapping
        else:
            xvals = collection.get_mcer(taglist=taglist)

        # computing xvalues
        index = index_xrange(range, xvals)
        xvals = xvals[index]

    else:
        xvals = domain

    # data container
    matrix = []

    # reading and processing spectra
    for i, name in enumerate(listgroups):
        group = collection.get_group(name).copy()
        if region == 'exafs':
            check_objattrs(group, Group, attrlist=['k', 'chi'], exceptions=True)
            s = interp1d(group.k, group.k**kweight*group.chi, kind='cubic')
        else:
            # region == 'xanes' or 'dxanes'
            check_objattrs(group, Group, attrlist=['energy', 'norm'], exceptions=True)
            # spline interpolation
            if region =='xanes':
                s = interp1d(group.energy, group.norm, kind='cubic')
            else:
                s = interp1d(group.energy, gradient(group.norm)/gradient(group.energy), kind='cubic')

        # interpolating in the fit range
        try:
            yvals = s(xvals)
        except:
            raise ValueError('requested domain is outside the values of group %s' % name)
        
        # saving yvals in data
        matrix.append(yvals)

    # converting list to array
    matrix = array(matrix).T
    return (xvals, matrix)

if __name__ == '__main__':
    import doctest
    doctest.testmod()