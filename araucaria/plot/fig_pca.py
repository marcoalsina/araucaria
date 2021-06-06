#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from numpy import cumsum, ptp
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
from .template import fig_xas_template
from ..stats import PCAModel
from ..utils import check_objattrs

def fig_pca(out: PCAModel, fontsize: float=8, **fig_kws) -> Tuple[Figure,Axes]:
    """Plots the results of principal component analysis (PCA).

    Parameters
    ----------
    out
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
        If ``out`` is not a valid PCAModel instance
    KeyError
        If attributes from :func:`~araucaria.stats.pca.pca`
        do not exist in ``out``.

    See also
    --------
    :func:`~araucaria.stats.pca.pca` : Performs principal component analysis on a collection.

    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.stats import pca
        >>> from araucaria.io import read_collection_hdf5
        >>> from araucaria.plot import fig_pca
        >>> fpath      = get_testpath('Fe_database.h5')
        >>> collection = read_collection_hdf5(fpath)
        >>> out        = pca(collection, pca_region='xanes', 
        ...                  pca_range=[7050, 7300], pre_edge_kws={})
        >>> fig, axes = fig_pca(out, fontsize=8)
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    check_objattrs(out, PCAModel, attrlist=['groupnames', 'matrix', 
    'variance', 'pca_pars'], exceptions=True)

    assert out.ncomps > 1, "ncomps must be at least 2."

    # copy of principal components
    pcomps = out.components.copy()

    # setting labels
    labels = ['PC%i' % (i+1) for i in range(out.ncomps)]

    # setting panels based on pca region
    region = (out.pca_pars['pca_region'])
    if region == 'xanes':
        xvar     = 'energy'
        pan      = 'x'
        fig_pars = {}
        # translatting PCs to start from zero
        for i in range(out.ncomps):
            pcomps[:,i] = pcomps[:,i] - pcomps[1,i]

    elif region == 'exafs':
        xvar = 'k'
        pan  = 'e'
        fig_pars = {'kweight' : out.pca_pars['kweight']}
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
    cumvar = cumsum(out.variance)

    # computing scores to project on PC basis
    scores = out.transform(out.matrix)

    # first panel: variance
    axes[0].bar(labels, out.variance)
    axes[0].plot(range(out.ncomps), cumvar, marker='o', color='tomato')

    # second panel: projection onto first 2 PC
    axes[1].scatter(scores[0,:], scores[1,:])

    # annotating with group names and arrows
    for i, label in enumerate(out.groupnames):
        axes[1].annotate('', (scores[:,i][0], scores[:,i][1]), xytext=(0,0),
                         arrowprops=dict({'arrowstyle':"->", 'color':'tomato'}))
        axes[1].annotate(label, (scores[:,i][0], scores[:,i][1]), fontsize=fontsize)

    # third panel: principal components
    for i in range(out.ncomps):
        axes[2].plot(getattr(out, xvar), pcomps[:,i], label='PC%i' % (i+1))

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
    axes[1].set_xlim((xlim[0]-0.1*ptp(xlim), xlim[1] + 0.5*ptp(xlim)))
    axes[1].set_ylim((ylim[0]-0.1*ptp(ylim), ylim[1] + 0.1*ptp(ylim)))

    axes[2].legend(edgecolor='k', fontsize=fontsize)
    axes[2].set_yticks([])

    return (fig, axes)

if __name__ == '__main__':
    import doctest
    doctest.testmod()