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
   * - :func:`cluster`
     - Performs hierarchical clustering on a collection.
"""
from typing import List
from numpy import inf
from scipy.interpolate import interp1d
from scipy.cluster.hierarchy import linkage
from .. import Group, DatGroup, Collection
from ..utils import check_objattrs, index_xrange
from araucaria import xas

def cluster(collection: Collection, cluster_region: str='xanes', 
            cluster_range: list=[-inf,inf], method: str='single',
            metric: str='euclidean', taglist: List[str]=['all'],
            kweight: int=2, pre_edge_kws: dict=None,
            autobk_kws: dict=None) -> DatGroup:
    """Performs cluster analysis on a collection.

    Parameters
    ----------
    collection
        Collection with the groups for clustering.
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
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].
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
        If ``scantag`` or ``refttag``  are not keys of the ``tags`` attribute.
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

    Example
    -------
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
    # checking class and attributes
    check_objattrs(collection, Collection, attrlist=['tags'], exceptions=True)

   # verifying cluster type
    cluster_valid = ['dxanes', 'xanes','exafs']
    if cluster_region not in cluster_valid:
        raise ValueError('cluster_region %s not recognized.'%cluster_region)

    # list of groupnames
    listgroups = collection.get_names(taglist=taglist)

    # storing report parameters
    cluster_pars = {'cluster_region':cluster_region, 'cluster_range':cluster_range}

    # report parameters for exafs clustering
    # since k-range may not have been computed, the first group is used
    if cluster_region == 'exafs':
        xvar  = 'k'    # x-variable
        dummy = collection.get_group(listgroups[0])
        cluster_pars['kweight'] = kweight

        if pre_edge_kws is None:
            cluster_pars['pre_edge_kws'] = None
        else:
            cluster_pars['pre_edge_kws'] = pre_edge_kws
            dictpars = xas.pre_edge(dummy, update=True, **pre_edge_kws)
        if autobk_kws is None:
            cluster_pars['autobk_kws'] = None
            check_objattrs(dummy, Group, attrlist=['k', 'chi'], exceptions=True)
        else:
            cluster_pars['autobk_kws'] = autobk_kws
            dictpars = xas.autobk(dummy, update=True, **autobk_kws)
        xvals = dummy.k
    # report paramters for xanes clustering
    else:
        xvar  = 'energy'    # x-variable
        xvals = collection.get_mcer(taglist=taglist)
        if pre_edge_kws is None:
            cluster_pars['pre_edge_kws'] = None
        else:
            cluster_pars['pre_edge_kws'] = pre_edge_kws

    # computing xvalues
    index = index_xrange(cluster_range, xvals)
    xvals = xvals[index]

    # containers
    content = {'groupnames': listgroups}
    matrix  = []

    # reading and processing spectra
    for i, name in enumerate(listgroups):
        group = collection.get_group(name)

        if cluster_region == 'exafs':
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
            # cluster_region == 'xanes' or 'dxanes'
            if pre_edge_kws is None:
                check_objattrs(group, Group, attrlist=['energy', 'norm'], exceptions=True)
            else:
                dictpars = xas.pre_edge(group, update=True, **pre_edge_kws)
            # spline interpolation
            if cluster_region =='xanes':
                s = interp1d(group.energy, group.norm, kind='cubic')
            else:
                s = interp1d(group.energy, gradient(group.norm)/gradient(group.energy), kind='cubic')

        # interpolating in the fit range
        try:
            yvals = s(xvals)
        except:
            raise ValueError('cluster_range is outside the domain of group %s' % name)
        
        # saving yvals in data
        matrix.append(yvals)

    # setting xvar as an attribute of datgroup
    content[xvar]     = xvals
    content['matrix'] = matrix

    # final cluster parameters
    cluster_pars['method'] = method
    cluster_pars['metric'] = metric
    
    # linkage matrix
    Z = linkage(matrix, method=method, metric=metric)
    
    # storing cluster data, parameters, and results
    content['Z'] = Z
    content['cluster_pars'] = cluster_pars

    eda_group = DatGroup(**content)

    return eda_group

if __name__ == '__main__':
    import doctest
    doctest.testmod()