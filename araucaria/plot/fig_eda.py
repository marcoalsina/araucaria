#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
from .. import Dataset
from ..utils import check_objattrs

def fig_cluster(out: Dataset, fontsize: float=8, **fig_kws) -> Tuple[Figure,Axes]:
    """Plots the dendrogram of a hierarchical clustering.

    Parameters
    ----------
    out
        Valid Dataset from :func:`~araucaria.stats.eda.cluster`.
    fontsize
        Font size for labels.
        The default is 8.
    fig_kws
        Additional arguments to pass to the :meth:`~matplotlib.figure.Figure.subplots` 
        routine of ``Matplotlib``.

    Returns
    -------
    figure
        Matplolib figure object.
    axes
        Matplotlib axes object. 
   
    Raises
    ------
    TypeError
        If ``group`` is not a valid Dictionary instance.
    KeyError
        If keys ``groupnames``, ``cluster_pars`` does not exist in ``out``.

    See also
    --------
    :func:`~araucaria.stats.eda.cluster` : Performs cluster analysis on a collection.

    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.stats import cluster
        >>> from araucaria.io import read_collection_hdf5
        >>> from araucaria.plot import fig_cluster
        >>> fpath      = get_testpath('Fe_database.h5')
        >>> collection = read_collection_hdf5(fpath)
        >>> datgroup   = cluster(collection, cluster_region='xanes', pre_edge_kws={})
        >>> fig, ax    = fig_cluster(datgroup)
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    check_objattrs(out, Dataset, attrlist=['groupnames', 'Z', 'cluster_pars'], 
    exceptions=True)

    # plotting the results
    fig, ax = plt.subplots(1,1, **fig_kws)
    hierarchy.set_link_color_palette(['c', 'm', 'y', 'k'])
    dn = hierarchy.dendrogram(out.Z, ax=ax, orientation='right', leaf_font_size= fontsize,
                    above_threshold_color='gray', labels=out.groupnames)
    ax.set_title(out.cluster_pars['cluster_region'].upper()+' dendrogram')
    return (fig, ax)

def fig_pca(out: Dataset, fontsize: float=8, **fig_kws) -> Tuple[Figure,Axes]:
    """Plots the results of principal component analysis (PCA).

    Parameters
    ----------
    out
        Valid dataset from :func:`~araucaria.stats.eda.pca`.
    fontsize
        Font size for labels.
        The default is 8.
    fig_kws
        Additional arguments to pass to the :meth:`~matplotlib.figure.Figure.subplots` 
        routine of ``Matplotlib``.

    Returns
    -------
    figure
        Matplolib figure object.
    axes
        Matplotlib axes object. 
   
    Raises
    ------
    TypeError
        If ``group`` is not a valid Dictionary instance.
    KeyError
        If keys ``groupnames``, ``cluster_pars`` does not exist in ``out``.

    See also
    --------
    :func:`~araucaria.stats.eda.cluster` : Performs cluster analysis on a collection.

    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.stats import cluster
        >>> from araucaria.io import read_collection_hdf5
        >>> from araucaria.plot import fig_cluster
        >>> fpath      = get_testpath('Fe_database.h5')
        >>> collection = read_collection_hdf5(fpath)
        >>> datgroup   = cluster(collection, cluster_region='xanes', pre_edge_kws={})
        >>> fig, ax    = fig_cluster(datgroup)
        >>> plt.show(block=False)
    """
    check_objattrs(out, Dataset, attrlist=['groupnames', 'Z', 'cluster_pars'], 
    exceptions=True)

    # plotting the results
    fig, ax = plt.subplots(1,1, **fig_kws)
    hierarchy.set_link_color_palette(['c', 'm', 'y', 'k'])
    dn = hierarchy.dendrogram(out.Z, ax=ax, orientation='right', leaf_font_size= fontsize,
                    above_threshold_color='gray', labels=out.groupnames)
    ax.set_title(out.cluster_pars['cluster_region'].upper()+' dendrogram')
    return (fig, ax)

if __name__ == '__main__':
    import doctest
    doctest.testmod()