#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats.cluster` module offers the following 
functions to perform clustering:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`cluster`
     - Performs hierarchical clustering on a collection.
"""
from typing import List, Tuple
from numpy import inf
from scipy.cluster.hierarchy import linkage
from .. import Dataset, Collection
from ..xas.xasutils import get_mapped_data

def cluster(collection: Collection, taglist: List[str]=['all'],
            cluster_region: str='xanes', cluster_range: list=[-inf,inf], 
            method: str='single', metric: str='euclidean',
            kweight: int=2) -> Dataset:
    """Performs hierarchical clustering on a collection.

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

    Returns
    -------
    :
        Dataset with the following arguments:

        - ``Z``            : hierarchical clustering encoded as a linkage matrix.
        - ``groupnames``   : list with names of clustered groups.
        - ``energy``       : array with energy values. Returned only if
          ``cluster_region='xanes`` or ``cluster_region=dxanes``.
        - ``k``            : array with wavenumber values. Returned only if
          ``cluster_region='exafs'``.
        - ``matrix``       : array with observed values for groups in ``cluster_range``.
        - ``cluster_pars`` : dictionary with cluster parameters.

    See also
    --------
    :func:`~araucaria.plot.fig_cluster.fig_cluster` : Plots the dendrogram of a hierarchical clustering.

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Dataset
    >>> from araucaria.xas import pre_edge, autobk
    >>> from araucaria.stats import cluster
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> collection.apply(pre_edge)
    >>> out        = cluster(collection, cluster_region='xanes')
    >>> attrs      = ['groupnames', 'energy', 'matrix', 'Z', 'cluster_pars']
    >>> check_objattrs(out, Dataset, attrs)
    [True, True, True, True, True]

    >>> # exafs clustering
    >>> collection.apply(autobk)
    >>> out   = cluster(collection, cluster_region='exafs', cluster_range=[0,10])
    >>> attrs = ['groupnames', 'k', 'matrix', 'Z', 'cluster_pars']
    >>> check_objattrs(out, Dataset, attrs)
    [True, True, True, True, True]
    """
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=cluster_region, 
                                    range=cluster_range, kweight=kweight)

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
    else:
        # xanes/dxanes clustering
        xvar  = 'energy'    # x-variable

    # storing cluster results
    content = {'groupnames'   : collection.get_names(taglist=taglist),
               xvar           : xvals,
               'matrix'       : matrix,
               'Z'            : Z,
               'cluster_pars' : cluster_pars,}

    out = Dataset(**content)
    return out

if __name__ == '__main__':
    import doctest
    doctest.testmod()