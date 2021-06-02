#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats.eda` module offers the following 
functions to perform exploratory data analysis:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`get_mapped_data`
     - Returns mapped data for common domain in a colleciton.
   * - :func:`cluster`
     - Performs hierarchical clustering on a collection.
"""
from typing import List, Tuple
from numpy import ndarray, array, inf
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage
from .. import Group, DatGroup, Collection
from ..utils import check_objattrs, index_xrange
from araucaria import xas

def get_mapped_data(collection: Collection, taglist: List[str]=['all'],
                    region: str='xanes', range: list=[-inf,inf], 
                    kweight: int=2, pre_edge_kws: dict=None,
                    autobk_kws: dict=None) -> Tuple[ndarray, ndarray]:
    """Returns mapped data for common domain in a collection.

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
    range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    kweight
        Exponent for weighting chi(k) by k^kweight.
        Only valid for ``region='exafs'``.
        The default is 2.
    pre_edge_kws
        Dictionary with parameters for :func:`~araucaria.xas.normalize.pre_edge`.
        The default is None, indicating that this step will be skipped.
    autobk_kws
        Dictionary with parameters :func:`~araucaria.xas.autobk.autobk`.
        Only valid for ``region='exafs'``.
        The default is None, indicating that this step will be skipped.    

    Returns
    -------
    :
        Array containing the domain values for mapping.
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
        Only verified if ``pre_edge_kws=None``.
    AttributeError
        If groups in ``collection`` have no ``k`` or ``chi`` attribute.
        Only verified if ``autobk_kws=None`` and ``region='exafs'``.
    KeyError
        If items in ``taglist`` are not keys of the ``tags`` attribute.
    ValueError
        If ``region`` is not recognized.
    ValueError
        If ``range`` is outside the domain of a group in ``collection``.

    Warning
    -------
    If given, ``pre_edge_kws`` or ``autobk_kws`` will only be used to 
    compute the mapped array. Results from normalization and background removal 
    will not be written in ``collection``.

    Notes
    -----
    Exploratory data analysis requires data that is mapped to a common domain range.
    However, datasets in a Collection are often mapped to different domain ranges.
    
    :func:`get_mapped_data` establishes a common domain based on the specified 
    ``region`` and ``range`` variables,  and returns an  array with the interpolated 
    values.

    Example
    -------
    >>> from numpy import allclose
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import DatGroup
    >>> from araucaria.stats import get_mapped_data
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> ener, data = get_mapped_data(collection, region='xanes', pre_edge_kws={})
    >>> allclose(ener.shape[0], data.shape[0])
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

    # report parameters for exafs mapping
    # since k-range may not have been computed, the first group is used
    if region == 'exafs':
        dummy = collection.get_group(listgroups[0])
        if pre_edge_kws is None:
            pass
        else:
            dictpars = xas.pre_edge(dummy, update=True, **pre_edge_kws)
        if autobk_kws is None:
            pass
        else:
            dictpars = xas.autobk(dummy, update=True, **autobk_kws)
        xvals = dummy.k
    # report paramters for xanes mapping
    else:
        xvals = collection.get_mcer(taglist=taglist)

    # computing xvalues
    index = index_xrange(range, xvals)
    xvals = xvals[index]

    # data container
    matrix = []

    # reading and processing spectra
    for i, name in enumerate(listgroups):
        group = collection.get_group(name)

        if region == 'exafs':
            # spectrum normalization
            if pre_edge_kws is None:
                pass
            else:
                dictpars = xas.pre_edge(group, update=True, **pre_edge_kws)
            # background removal
            if autobk_kws is None:
                check_objattrs(group, Group, attrlist=['k', 'chi'], exceptions=True)
            else:
                dictpars = xas.autobk(group, update=True, **autobk_kws)
            # spline interpolation
            s = interp1d(group.k, group.k**kweight*group.chi, kind='cubic')
        else:
            # region == 'xanes' or 'dxanes'
            if pre_edge_kws is None:
                check_objattrs(group, Group, attrlist=['energy', 'norm'], exceptions=True)
            else:
                dictpars = xas.pre_edge(group, update=True, **pre_edge_kws)
            # spline interpolation
            if region =='xanes':
                s = interp1d(group.energy, group.norm, kind='cubic')
            else:
                s = interp1d(group.energy, gradient(group.norm)/gradient(group.energy), kind='cubic')

        # interpolating in the fit range
        try:
            yvals = s(xvals)
        except:
            raise ValueError('range is outside the domain of group %s' % name)
        
        # saving yvals in data
        matrix.append(yvals)

    # converting list to array
    matrix = array(matrix).T
    return (xvals, matrix)

def cluster(collection: Collection, taglist: List[str]=['all'],
            cluster_region: str='xanes', cluster_range: list=[-inf,inf], 
            method: str='single', metric: str='euclidean',  kweight: int=2,
            pre_edge_kws: dict=None, autobk_kws: dict=None) -> DatGroup:
    """Performs cluster analysis on a collection.

    Parameters
    ----------
    collection
        Collection with the groups for clustering.
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].
    cluster_region
        XAFS region to perform clustering. Accepted values are 'dxanes',
        'xanes', or 'exafs'. The default is 'xanes'.
    cluster_range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    method
        Likage method to compute the distance between clusters.
        See the :func:`~scipy.cluster.hierarchy.linkage` function
        of ``scipy`` for a list of valid method names.
        The default is 'single'.
    metric
        The distance metric. See the :func:`~scipy.spatial.distance.pdist`
        function of ``scipy`` for a list of valid distance metrics.
        The default is 'euclidean'.
    kweight
        Exponent for weighting chi(k) by k^kweight.
        Only valid for ``cluster_region='exafs'``.
        The default is 2.
    pre_edge_kws
        Dictionary with parameters for :func:`~araucaria.xas.normalize.pre_edge`.
        The default is None, indicating that this step will be skipped.
    autobk_kws
        Dictionary with parameters :func:`~araucaria.xas.autobk.autobk`.
        Only valid for ``cluster_region='exafs'``.
        The default is None, indicating that this step will be skipped.

    Returns
    -------
    :
        Data group with the following arguments:

        - ``Z``            : hierarchical clustering encoded as a linkage matrix.
        - ``groupnames``   : list with names of clustered groups.
        - ``energy``       : array with energy values. Returned only if
          ``cluster_region='xanes`` or ``cluster_region=dxanes``.
        - ``k``            : array with wavenumber values. Returned only if
          ``cluster_region='exafs'``.
        - ``matrix``       : array with observed values for groups in ``cluster_range``.
        - ``cluster_pars`` : dictionary with cluster parameters.

    Raises
    ------
    TypeError
        If ``collection`` is not a valid Collection instance.
    AttributeError
        If ``collection`` has no ``tags`` attribute.
    AttributeError
        If groups have no ``energy`` or ``norm`` attribute.
        Only verified if ``cluster_region='dxanes'`` or  ``cluster_region='xanes'``.
    AttributeError
        If groups have no ``k`` or ``chi`` attribute.
        Only verified if ``cluster_region='exafs'``.
    KeyError
        If ``taglist`` are not keys of the ``tags`` attribute.
    ValueError
        If ``cluster_region`` is not recognized.
    ValueError
        If ``cluster_range`` is outside the domain of a group.

    Warning
    -------
    If given, ``pre_edge_kws`` or ``autobk_kws`` will only be used to 
    perform clustering. Results from normalization and background removal 
    will not be written in ``collection``.

    See also
    --------
    :func:`~araucaria.plot.fig_cluster.fig_cluster` : Plots the dendrogram of a hierarchical clustering.

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import DatGroup
    >>> from araucaria.stats import cluster
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> datgroup   = cluster(collection, cluster_region='xanes', pre_edge_kws={})
    >>> attrs      = ['groupnames', 'energy', 'matrix', 'Z', 'cluster_pars']
    >>> check_objattrs(datgroup, DatGroup, attrs)
    [True, True, True, True, True]

    >>> # exafs clustering
    >>> datgroup   = cluster(collection, cluster_region='exafs', cluster_range=[0,10],
    ... pre_edge_kws={}, autobk_kws={})
    >>> attrs      = ['groupnames', 'k', 'matrix', 'Z', 'cluster_pars']
    >>> check_objattrs(datgroup, DatGroup, attrs)
    [True, True, True, True, True]
    """
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=cluster_region, 
                                    range=cluster_range, kweight=kweight,
                                    pre_edge_kws=pre_edge_kws, autobk_kws=autobk_kws)

    # linkage matrix
    # matrix is transposed to follow the m by n convention with m observation vectors
    Z = linkage(matrix.T, method=method, metric=metric)

    # storing cluster parameters
    cluster_pars = {'cluster_region': cluster_region, 
                    'cluster_range' : cluster_range,
                    'method'        : method,
                    'metric'        : metric,}

    # additional cluster parameters
    if cluster_region == 'exafs':
        xvar  = 'k'    # x-variable
        cluster_pars['kweight'] = kweight
        if pre_edge_kws is None:
            cluster_pars['pre_edge_kws'] = None
        else:
            cluster_pars['pre_edge_kws'] = pre_edge_kws
        if autobk_kws is None:
            cluster_pars['autobk_kws'] = None
        else:
            cluster_pars['autobk_kws'] = autobk_kws
    # xanes/dxanes clustering
    else:
        xvar  = 'energy'    # x-variable
        if pre_edge_kws is None:
            cluster_pars['pre_edge_kws'] = None
        else:
            cluster_pars['pre_edge_kws'] = pre_edge_kws

    # storing cluster results
    content = {'groupnames'   : collection.get_names(taglist=taglist),
               xvar           : xvals,
               'matrix'       : matrix,
               'Z'            : Z,
               'cluster_pars' : cluster_pars,}

    eda_group = DatGroup(**content)
    return eda_group

if __name__ == '__main__':
    import doctest
    doctest.testmod()