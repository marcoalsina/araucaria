#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from numpy import cumsum, ptp, ceil, sqrt, ravel, argsort
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
from .. import Dataset
from .template import fig_xas_template
from ..stats import PCAModel
from ..utils import check_objattrs

def fig_cluster(out: Dataset, fontsize: float=8, 
                **fig_kws: dict) -> Tuple[Figure,Axes]:
    """Plots the dendrogram of a hierarchical clustering.

    Parameters
    ----------
    out
        Valid Dataset from :func:`~araucaria.stats.cluster.cluster`.
    fontsize
        Font size for labels.
        The default is 8.
    fig_kws
        Additional arguments to pass to the :meth:`~matplotlib.figure.Figure.subplots` 
        routine of ``Matplotlib``.

    Returns
    -------
    figure
        ``Matplolib`` figure object.
    axes
        ``Matplotlib`` axes object. 
   
    Raises
    ------
    TypeError
        If ``out`` is not a valid Dataset instance.
    KeyError
        If attributes from :func:`~araucaria.stats.cluster.cluster` 
        do not exist in ``out``.

    See also
    --------
    :func:`~araucaria.stats.cluster.cluster` : Performs hierarchical clustering on a collection.

    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.xas import pre_edge
        >>> from araucaria.stats import cluster
        >>> from araucaria.io import read_collection_hdf5
        >>> from araucaria.plot import fig_cluster
        >>> fpath      = get_testpath('Fe_database.h5')
        >>> collection = read_collection_hdf5(fpath)
        >>> collection.apply(pre_edge)
        >>> datgroup   = cluster(collection, cluster_region='xanes')
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

def fig_pca(model: PCAModel, fontsize: float=8, **fig_kws: dict) -> Tuple[Figure,Axes]:
    """Plots the results of principal component analysis (PCA) on a collection.

    Parameters
    ----------
    model
        Valid PCAModel from :func:`~araucaria.stats.pca.pca`.
    fontsize
        Font size for labels.
        The default is 8.
    fig_kws
        Additional arguments to pass to the :meth:`~matplotlib.figure.Figure.subplots` 
        routine of ``Matplotlib``.

    Returns
    -------
    figure
        ``Matplolib`` figure object.
    axes
        ``Matplotlib`` axes object. 
   
    Raises
    ------
    TypeError
        If ``model`` is not a valid PCAModel instance
    KeyError
        If attributes from :func:`~araucaria.stats.pca.pca`
        do not exist in ``model``.

    See also
    --------
    :func:`~araucaria.stats.pca.pca` : Performs principal component analysis on a collection.

    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.xas import pre_edge
        >>> from araucaria.stats import pca
        >>> from araucaria.io import read_collection_hdf5
        >>> from araucaria.plot import fig_pca
        >>> fpath      = get_testpath('Fe_database.h5')
        >>> collection = read_collection_hdf5(fpath)
        >>> collection.apply(pre_edge)
        >>> model      = pca(collection, pca_region='xanes', 
        ...                  pca_range=[7050, 7300])
        >>> fig, axes = fig_pca(model, fontsize=8)
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    check_objattrs(model, PCAModel, attrlist=['groupnames', 'matrix', 
    'variance', 'pca_pars'], exceptions=True)

    assert model.ncomps > 1, "ncomps must be at least 2."

    # copy of principal components
    pcomps = model.components.copy()

    # setting labels
    labels = ['PC%i' % (i+1) for i in range(model.ncomps)]

    # setting panels based on pca region
    region = (model.pca_pars['pca_region'])
    if region == 'xanes':
        xvar     = 'energy'
        pan      = 'x'
        fig_pars = {}
        # translatting PCs to start from zero
        for i in range(model.ncomps):
            pcomps[:,i] = pcomps[:,i] - pcomps[1,i]

    elif region == 'exafs':
        xvar = 'k'
        pan  = 'e'
        fig_pars = {'kweight' : model.pca_pars['kweight']}
    else:
        xvar     = 'energy'
        pan      = 'd'
        fig_pars = {}

    # ensuring a reasonable figsize
    if fig_kws is None:
        fig_kws = {'figsize' : (8,3)}
    elif 'figsize' not in fig_kws:
        fig_kws['figsize'] = (8, 3)

    # initializing figure and axes
    panels = 'uu' + pan
    fig, axes = fig_xas_template(panels, fig_pars=fig_pars, **fig_kws)

    # cumulative variance
    cumvar = cumsum(model.variance)

    # computing scores to project on PC basis
    scores = model.transform(model.matrix)

    # first panel: variance
    axes[0].bar(labels, model.variance)
    axes[0].plot(range(model.ncomps), cumvar, marker='o', color='tomato')

    # second panel: projection onto first 2 PC
    axes[1].scatter(scores[0,:], scores[1,:])

    # annotating with group names and arrows
    for i, label in enumerate(model.groupnames):
        axes[1].annotate('', (scores[:,i][0], scores[:,i][1]), xytext=(0,0),
                         arrowprops=dict({'arrowstyle':"->", 'color':'tomato'}))
        axes[1].annotate(label, (scores[:,i][0], scores[:,i][1]), ha='center', fontsize=fontsize)

    # third panel: principal components
    for i in range(model.ncomps):
        axes[2].plot(getattr(model, xvar), pcomps[:,i], label='PC%i' % (i+1))

    # axes decorators
    axes[0].set_ylabel('Variance')
    axes[0].axhline(1,0,1, ls='--', color='0.6', zorder=-3)
    axes[1].axvline(0,0,1, ls='--', color='0.6', zorder=-3)
    
    axes[1].axhline(0,0,1, ls='--', color='0.6', zorder=-3)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')

    # expanding axes limits 
    xlim = axes[1].get_xlim()
    ylim = axes[1].get_ylim()
    axes[1].set_xlim((xlim[0]-0.1*ptp(xlim), xlim[1] + 0.2*ptp(xlim)))
    axes[1].set_ylim((ylim[0]-0.1*ptp(ylim), ylim[1] + 0.1*ptp(ylim)))

    axes[2].legend(edgecolor='k')
    axes[2].set_yticks([])

    return (fig, axes)

def fig_target_transform(out: Dataset, model: PCAModel, sorted: bool=True,
                         fontsize: float=8, **fig_kws: dict) -> Tuple[Figure,Axes]:
    """Plots the results of target transformation on a collection.

    Parameters
    ----------
    out
        Valid Dataset from :func:`~araucaria.stats.pca.target_transform`.
    model
        Valid PCA model from :func:`~araucaria.stats.pca.pca`.
    sorted
        Conditional for plotting target transformations from lowest to highest
        chi-square.
        The default is True.
    fontsize
        Font size for labels.
        The default is 8.
    fig_kws
        Additional arguments to pass to the :meth:`~matplotlib.figure.Figure.subplots` 
        routine of ``Matplotlib``.

    Returns
    -------
    figure
        ``Matplolib`` figure object.
    axes
        ``Matplotlib`` axes object. 
   
    Raises
    ------
    TypeError
        If ``model`` is not a valid PCAModel instance.
    TypeError
        If ``out`` is not a valid Dataset instance.
    KeyError
        If attributes from :func:`~araucaria.stats.pca.pca`
        do not exist in ``model``.
    KeyError
        If attributes from :func:`~araucaria.stats.pca.target_transform`
        do not exist in ``out``.

    See also
    --------
    :func:`~araucaria.stats.pca.pca` : Performs principal component analysis on a collection.
    :func:`~araucaria.stats.pca.target_transform` : Performs target transformation on a collection.

    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria import Dataset
        >>> from araucaria.io import read_collection_hdf5
        >>> from araucaria.xas import pre_edge
        >>> from araucaria.stats import pca, target_transform
        >>> from araucaria.plot import fig_target_transform
        >>> fpath      = get_testpath('Fe_database.h5')
        >>> collection = read_collection_hdf5(fpath)
        >>> collection.apply(pre_edge)
        >>> model      = pca(collection, pca_region='xanes', ncomps=3,
        ...                  pca_range=[7050,7300])
        >>> data       = target_transform(model, collection)
        >>> fig, axes  = fig_target_transform(data, model)
        >>> legend     = axes[0,0].legend(loc='upper right', edgecolor='k')
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    check_objattrs(model, PCAModel, attrlist=['groupnames', 'matrix', 
                   'variance', 'pca_pars'], exceptions=True)

    check_objattrs(out, Dataset, attrlist=['groupnames', 'matrix', 
                   'tmatrix', 'scores', 'chi2'], exceptions=True)

    # original and transformed matrix array
    matrix  = model.matrix
    tmatrix = out.tmatrix
    scores  = out.scores
    names   = out.groupnames

    # setting panels based on pca region
    region = (model.pca_pars['pca_region'])
    if region == 'xanes':
        xvar     = 'energy'
        pan      = 'x'
        fig_pars = {}

    elif region == 'exafs':
        xvar = 'k'
        pan  = 'e'
        fig_pars = {'kweight' : model.pca_pars['kweight']}

    else:
        xvar     = 'energy'
        pan      = 'd'
        fig_pars = {}

    # calculating chi square
    # standards are sorted according to this metric
    chi2   = out.chi2
    
    if sorted:
        porder  = argsort(chi2)[::-1]
        matrix  = matrix[:, porder]
        tmatrix = tmatrix[:, porder]
        scores  = scores[:, porder]
        names   = [out.groupnames[i] for i in porder]

    # calculating number of panels
    n = len(names)
    if n > 3:
        l = int(ceil(sqrt(n)))  # panel length
        h = int(ceil(n/l))      # panel height
        panels = l*pan
        for i in range(1,h):
            panels += '/' + l*pan
    else:
        l = n    # panel length
        h = 1    # panel height
        panels = n*pan

    # ensuring a reasonable figsize
    if fig_kws is None:
        fig_kws = {'figsize' : (3*l, 3*h)}
    elif 'figsize' not in fig_kws:
        fig_kws['figsize'] = (3*l, 3*h)

    # initializing figure and axes
    fig, axes = fig_xas_template(panels, fig_pars=fig_pars, **fig_kws)
    raxes     = ravel(axes, order='F')
    for i, name in enumerate(names):
        raxes[i].plot(getattr(out, xvar), matrix[:,i] , label='original')
        raxes[i].plot(getattr(out, xvar), tmatrix[:,i], label='target')
        raxes[i].set_title(name)

        # decorations
        chi2_text = r'$\chi^2$=$%1.4f$' % chi2[i]
        tt_text   = '\nTT coefficients:'
        for j in range(model.ncomps):
            tt_text += '\nPC%i: %1.3f' % (j+1, scores[j,i])

        # locating text
        xlim = raxes[i].get_xlim()
        ylim = raxes[i].get_ylim()
        raxes[i].text(xlim[0]+0.5*ptp(xlim),
                      ylim[0]+0.5*ptp(ylim), chi2_text+tt_text, ha='left', va='top', fontsize=fontsize)

    return (fig, axes)

if __name__ == '__main__':
    import doctest
    doctest.testmod()