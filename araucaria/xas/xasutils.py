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
   * - :func:`compute_bins`
     - Computes bin sequence for an energy array.
   * - :func:`rebin`
     - Rebins XAFS spectra in a group.
   * - :func:`roll_med`
     - Rolling median of a 1-D array.
"""
from scipy.constants import hbar  # reduced planck constant
from scipy.constants import m_e   # electron mass
from scipy.constants import eV    # electron volt in joules

from typing import List, Tuple, Union
from numpy import (ndarray, array, arange, nan, isnan, std, append, concatenate, 
                   gradient, inf, delete, count_nonzero, nanmedian, ediff1d)
from scipy.stats import binned_statistic
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

def compute_bins(ref_energy: float, e_step: float=0.5, bkg_pars: list=[-300, -20, 5], 
                 exafs_pars: list=[3, 15, 0.05], ndigits: int=4) -> ndarray:
    """Computes bin sequence for an energy array.

    Parameters
    ----------
    ref_energy:
        Reference energy for computing bins (eV). Both background range
        and exafs range are computed with respect to this value.
    e_step:
        Energy increment step for XANES bins (eV).
        The default is 0.5.
    bkg_pars:
        Parameters for background bins (eV). Should include
        initial energy value, final energy value, and energy increment step.
        The detault is [-300, -20, 5]
    exafs_pars:
        Parameters for EXAFS bins (inverse angstrom). Should include
        initial k value, final k value, and k increment step.
        The default is [3, 15, 0.05].
    ndigits:
        Number of decimal places to round bins.
        The default is 4.

    Returns
    -------
    :
        Array with bin edges.

    Raises
    ------
    ValueError
        If len of ``bkg_pars`` or ``exafs_pars`` is smaller than 3.

    See also
    --------
    :func:`rebin`: Rebins spectra in a group.

    Notes
    -----
    Computation of bins is performed considering 3 regions:

    - background region: defined by ``bkg_pars``.
    - XANES region: defined between ``bkg_pars``, ``exafs_pars``, and ``e_step``.
    - EXAFS region: defined by ``exafs_pars``

    Bin sizes are adjusted between regions in order to follow the
    energy limits established by ``bkg_pars`` and ``exafs_ pars``.

    Example
    -------
    >>> from numpy import around, allclose
    >>> from araucaria.xas import compute_bins, ktoe
    >>> ndigits    = 4
    >>> edge       = 7112
    >>> bkg_pars   = [-300, -50, 10]
    >>> exafs_pars = [2, 10, 0.05]
    >>> bin_edges  = compute_bins(ref_energy=edge, bkg_pars=bkg_pars, 
    ...                           exafs_pars=exafs_pars, ndigits=ndigits)
    >>> # verifying bin edges 
    >>> minval = edge + bkg_pars[0] - bkg_pars[2]/2           # minimum bin edge value
    >>> maxval = edge + ktoe(exafs_pars[1] + exafs_pars[2]/2) # maximum bin edge value
    >>> vals   = around((minval, maxval), ndigits)            # rounding to ndigits
    >>> allclose((bin_edges[0], bin_edges[-1]), vals)
    True
    """
    if len(bkg_pars) < 3:
        raise ValueError("'bkg_pars' should provide at least 3 values.")
    if len(exafs_pars) < 3:
        raise ValueError("'exafs_pars should provide at least 3 values.")

    # computing bin edges for background region
    ival      = ref_energy + bkg_pars[0] - bkg_pars[2]/2
    fval      = ref_energy + bkg_pars[1] + bkg_pars[2]/2
    bkg_edges = arange(ival, fval, bkg_pars[2])
    bkg_edges = bkg_edges.round(ndigits)

    # computing bin edges at the edge    
    ival      = bkg_pars[1] + e_step/2 + ref_energy
    fval      = ktoe(exafs_pars[0]) + ref_energy + e_step/2
    ene_edges = arange(ival, fval, e_step)
    ene_edges = ene_edges.round(ndigits)

    # computing bind esges for exafs region
    step = exafs_pars[2]
    ival = exafs_pars[0] + step/2
    fval = exafs_pars[1] + step
    exafs_edges = []
    for kval in arange(ival, fval, step):
        exafs_edges.append( ktoe(kval) + ref_energy )
    exafs_edges = array(exafs_edges).round(ndigits)

    # concatenating bin edges
    bin_edges = concatenate((bkg_edges, ene_edges, exafs_edges))
    return bin_edges

def rebin(group: Group, statistic: str='mean',
          bins: Union[int, list]=10, remove_nans: bool=True,
          update: bool=False) -> dict:
    """Rebins XAFS spectra in a group.

    Parameters
    ----------
    group:
         Group containing the spectrum to rebin.
    statistic:
        The statistic to compute for rebinning. See
        :func:`~scipy.stats.binned_statistic` for valid names.
        The default is 'mean'.
    bins:
        If :class:`int`, it defines the number of equal-width bins in the 
        given range. If a sequence, it defines the bin edges, including 
        the rightmost edge, allowing for non-uniform bin widths.
        The default is 10.
    remove_nans:
        Indicates if bins with :data:`~numpy.nan` values should be removed.
        The default is True.
    update:
         Indicates if the group should be updated with the rebin attributes.
         The default is False.

    Returns
    -------
    :
        Dictionary with the following parameters:
        
        - ``energy``     : rebinned energy values.
        - ``mu``         : rebinned transmission :math:`\mu(E)`, 
          if ``mu`` attribute exists in the group.
        - ``fluo``       : rebinned fluorescence :math:`\mu(E)`, 
          if ``fluo`` attribute exists in the group.
        - ``mu_ref``     : rebinned reference :math:`\mu(E)`, 
          if ``mu_ref`` attribute exists in the group.
        - ``rebin_stats``: additional rebin statistics. 

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``energy`` does not exist in ``group``.

    Important
    ---------
    Bins with no data are removed by default to prevent
    :data:`~numpy.nan` values in the rebinned group arrays.
    As a consecuence , the total number of bins might be 
    smaller than originally specified, and bins will exhibit 
    varying width.
    
    You can override this default behavior by specifying ``remove_nans=False``.

    Example
    -------
    .. plot::
        :context: reset

        >>> from araucaria import Group
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_xmu
        >>> from araucaria.xas import rebin
        >>> from araucaria.utils import check_objattrs
        >>> fpath = get_testpath('xmu_testfile.xmu')
        >>> # extracting mu and mu_ref scans
        >>> group_mu = read_xmu(fpath, scan='mu')
        >>> bins    = 600             # number of bins
        >>> regroup = group_mu.copy() # rebinning copy of group
        >>> rebin   = rebin(regroup, bins=bins, update=True)
        >>> attrs   = ['energy', 'mu', 'mu_ref', 'rebin_stats']
        >>> check_objattrs(regroup, Group, attrs)
        [True, True, True, True]

        >>> # plotting rebinned spectrum
        >>> from araucaria.plot import fig_xas_template
        >>> import matplotlib.pyplot as plt
        >>> figpars = {'e_range' : (11850, 11900)}   # energy range
        >>> fig, ax = fig_xas_template(panels='x', fig_pars=figpars)
        >>> stdev = regroup.rebin_stats['mu_std']    # std of rebinned mu
        >>> line  = ax.plot(group_mu.energy, group_mu.mu, label='original')
        >>> line  = ax.errorbar(regroup.energy, regroup.mu, yerr=stdev, marker='o',
        ...                     capsize=3.0, label='rebinned')
        >>> leg   = ax.legend(edgecolor='k')
        >>> lab   = ax.set_ylabel('abs [a.u]')
        >>> plt.show(block=False)
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['energy'], exceptions=True)

    # storing energy and mu as indepedent arrays
    energy = group.energy
    mode   = group.get_mode()
    mu     = getattr(group, mode)

    # rebining mu
    rebin_mu, bin_edges, binnumber = binned_statistic(energy, mu, statistic=statistic, bins=bins)
    rebin_energy = ediff1d(bin_edges)/2 + bin_edges[:-1]

    # computing std
    stdev = []
    for i in range(len(rebin_mu)):
        index = (binnumber == i+1)
        if not any(index):
            # testing if all indexes are false
            stdev.append(nan)
        else:
            stdev.append( std( mu[index] ) )
    stdev = array(stdev)

    if remove_nans:
        # removing nans
        nindex       = isnan(rebin_mu)
        edges_nindex = append(nindex.copy(), False)
        rebin_mu     = delete(rebin_mu, nindex)
        bin_edges    = delete(bin_edges, edges_nindex)
        rebin_energy = delete(rebin_energy, nindex)
        stdev        = delete(stdev,nindex)

    # storing results
    bin_std     = mode + '_std'
    rebin_stats = {'energy_edges': bin_edges,
                   'binnumber'   : binnumber,
                   bin_std       : stdev}
    content     = {mode          : rebin_mu,
                   'energy'      : rebin_energy}

    if group.has_ref and mode in ('mu', 'fluo'):
        # rebinning mu_ref (if exists)
        mu_ref = group.mu_ref
        rebin_mu_ref, bin_edges, binnumber = \
        binned_statistic(energy, mu_ref, statistic=statistic, bins=bins)

        # computing ref_std
        stdev = []
        for i in range(len(rebin_mu_ref)):
            index = (binnumber == i+1)
            if not any(index):
                stdev.append(nan)
            else:
                stdev.append( std( mu_ref[index] ) )
        stdev = array(stdev)

        if remove_nans:
            # removing nans
            nindex       = isnan(rebin_mu_ref)
            rebin_mu_ref = delete(rebin_mu_ref, nindex)
            stdev        = delete(stdev,nindex)

        # storing mu_ref results
        content['mu_ref']               = rebin_mu_ref
        rebin_stats['mu_ref_std']       = stdev

    # storing rebin statistics
    content['rebin_stats'] = rebin_stats

    # updating group
    if update:
        group.add_content(content)
    return content

def roll_med(data: ndarray, window: int, min_samples: int=2, 
             edgemethod: str='nan') -> ndarray:
    """Computes the rolling median of a 1-D array.
    
    Parameters
    ----------
    data:
        Array to compute the rolling median.
    window:
        Size of the rolling window for analysis.
    min_samples:
        Minimum sample points to calculate the median in each window.
        The default is 2.
    edgemethod :
        Dictates how medians are calculated at the edges of the array.
        Options are 'nan', 'calc' and 'extend'. See the Notes for further details.
        The default is 'nan'.

    Returns
    -------
    :
        Rolling median of the array.

    Raises
    ------
    ValueError
        If ``window`` is not an odd value.
    ValueError
        If ``window`` is smaller or equal than 3.
    TypeError
        If ``window`` is not an integer.
    ValueError
        If ``edgemethod`` is not recognized.

    Notes
    -----
    This function calculates the median of a moving window. Results are returned in the 
    index corresponding to the center of the window. The function ignores :data:`~numpy.nan` 
    values in the array.

    - ``edgemethod='nan'`` uses :data:`~numpy.nan` values for missing values at the edges. 
    - ``edgemethod='calc'`` uses an abbreviated window at the edges 
      (e.g. the first sample will have (window/2)+1 points in the calculation).
    - ``edgemethod='extend'`` uses the nearest calculated value for missing values at the edges.

    Warning
    -------
    If ``window`` is less than ``min_samples``, :data:`~numpy.nan` is given as the median.

    Example
    -------
    .. plot::
        :context: reset

        >>> from numpy import pi, sin, linspace
        >>> from araucaria.xas import roll_med
        >>> import matplotlib.pyplot as plt
        >>> # generating a signal and its rolling median
        >>> f1   = 0.2 # frequency
        >>> t    = linspace(0,10)
        >>> y    = sin(2*pi*f1*t)
        >>> line = plt.plot(t,y, label='signal')
        >>> for method in ['calc', 'extend', 'nan']:
        ...    fy   = roll_med(y, window=25, edgemethod=method)
        ...    line = plt.plot(t, fy, marker='o', label=method)
        >>> lab = plt.xlabel('t')
        >>> lab =plt.ylabel('y')
        >>> leg = plt.legend()
        >>> plt.show(block=False)
    """
    if window % 2 == 0:
        raise ValueError('window length must be an odd value.')
    elif window < 3 or type(window)!=int:
        raise ValueError('window length must be larger than 3.')

    validEdgeMethods = ['nan', 'extend', 'calc']     
    if edgemethod not in validEdgeMethods:
        raise ValueError('please choose a valid edgemethod.')

    # calculating points on either side of the point of interest in the window
    movement  = int((window - 1) / 2) 
    med_array = array([nan for point in data])
    
    for i, point in enumerate(data[ : -movement]):
        if i>=movement:
            if count_nonzero(isnan(data[i - movement : i + 1 + movement]) == False) >= min_samples:
                med_array[i] = nanmedian(data[i - movement : i + 1 + movement])

    if edgemethod == 'nan':
        return med_array

    for i, point in enumerate(data[ : movement]):
        if edgemethod == 'calc':
            if count_nonzero(isnan(data[0 : i + 1 + movement]) == False) >= min_samples:
                med_array[i] = nanmedian(data[0 : i + 1 + movement])
        elif edgemethod == 'extend':
            med_array[i] = med_array[movement]

    for i, point in enumerate(data[-movement : ]):
        if edgemethod == 'calc':
            if count_nonzero(isnan(data[(-2 * movement) + i : ]) == False) >= min_samples:
                med_array[-movement + i] = nanmedian(data[(-2 * movement) + i : ])
        elif edgemethod == 'extend':
            med_array[-movement + i] = med_array[-movement - 1]

    return med_array

if __name__ == '__main__':
    import doctest
    doctest.testmod()