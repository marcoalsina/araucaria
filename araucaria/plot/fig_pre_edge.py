#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from matplotlib.pyplot import Axes, Figure
from .. import Group
from .template import fig_xas_template, FigPars
from ..utils import check_objattrs, index_nearest

def fig_pre_edge(group: Group, show_pre_edge: bool=True, 
                 show_post_edge: bool=True, fig_pars: FigPars=None, 
                 **fig_kws) -> Tuple[Figure,Axes]:
    """Plots the results of pre-edge substraction and normalization of a scan.

    Parameters
    ----------
    group
        Group with the normalized scan.
    show_pre_edge
        Indicates if the pre-edge fit curve should be plotted.
        The detault is True.
    show_post_edge
        Indicates if the post-edge fit curve should be plotted.
        The detault is True.
    fig_pars
        Dictionary arguments for the figure.
        See :class:`~araucaria.plot.template.FigPars` for details.    
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
        If ``group`` is not a valid Group instance.
    AttributeError
        If attributes from :func:`~araucaria.xas.normalize.pre_edge` 
        do not exist in ``group``.

    Notes
    -----
    The returned plot contains the original :math:`\mu(E)`, the
    adjusted pre-edge and post-edge curves (optional), and the
    normalized and flattened :math:`\mu(E)`.

    See also 
    --------
	:func:`~araucaria.xas.normalize.pre_edge`: Pre-edge substaction and normalization of a scan.
    
    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_dnd
        >>> from araucaria.xas import pre_edge
        >>> from araucaria.plot import fig_pre_edge
        >>> fpath   = get_testpath('dnd_testfile.dat')
        >>> group   = read_dnd(fpath, scan='mu')
        >>> pre     = pre_edge(group, update=True)
        >>> fig, ax = fig_pre_edge(group)
        >>> plt.show(block=False)
    """
    # checking class and attributes
    attrlist = ['e0', 'norm', 'flat', 'pre_edge', 'post_edge', 'pre_edge_pars']
    check_objattrs(group, Group, attrlist=attrlist, exceptions=True)

    # get scan mode
    mode = group.get_mode()
    e0   = group.e0
    ie0  = index_nearest(group.energy, e0)

    # plot data
    fig, axes = fig_xas_template(panels='xx', fig_pars=fig_pars, **fig_kws)
    axes[0].plot(group.energy, getattr(group, mode), label=mode)
    axes[0].scatter(group.energy[ie0], getattr(group, mode)[ie0], color='firebrick', label='$E_0$')
    
    if show_pre_edge:
        pre_range = group.pre_edge_pars['pre_range']
        pre_index    = [0,-1]
        pre_index[0] = index_nearest(group.energy, pre_range[0] + e0, kind='lower')
        pre_index[1] = index_nearest(group.energy, pre_range[1] + e0)
        
        axes[0].plot(group.energy, group.pre_edge, color='gray', ls='--', label='pre-edge')
        for val in pre_index:
            axes[0].scatter(group.energy[val], getattr(group, mode)[val], color='magenta')
        
    if show_post_edge:
        post_range = group.pre_edge_pars['post_range']
        post_index    = [0,-1]
        post_index[0] = index_nearest(group.energy, post_range[0] + e0, kind='lower')
        post_index[1] = index_nearest(group.energy, post_range[1] + e0)
        
        axes[0].plot(group.energy, group.post_edge, color='gray', ls=':', label='post-edge')
        for val in post_index:
            axes[0].scatter(group.energy[val], getattr(group, mode)[val], color='magenta')

    axes[1].plot(group.energy, group.norm, label='norm')
    axes[1].plot(group.energy, group.flat, label='flat')

    for ax in axes:
        ax.legend()
        if ax == axes[0]:
            ax.set_title('Absorption')
            ax.set_ylabel('Absorbtion [a.u.]')
        else:
            ax.set_title('Normalized absorption')

    return (fig, axes)

if __name__ == '__main__':
    import doctest
    doctest.testmod()