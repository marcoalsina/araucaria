#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from numpy import gradient
from matplotlib.pyplot import Axes, Figure
from .. import Group, Collection
from ..xas import pre_edge, autobk
from ..xas.xasft import ftwindow
from .template import fig_xas_template, FigPars
from ..utils import check_objattrs, index_nearest

def fig_merge(merge: Group, collection: Collection, 
              pre_edge_kws: dict=None, autobk_kws:dict =None,
              fig_pars: FigPars=None, **fig_kws: dict) -> Tuple[Figure,Axes]:
    """Plots the results of a merge operation in a collection.

    Parameters
    ----------
    merge
        Group with the merged scan.
    collection
        Collection with the merged groups.
    pre_edge_kws
        Dictionary with arguments for normalization.
    autobk_kws
        Dictionary with arguments for background removal.
    fig_pars
        Dictionary arguments for the figure.
        See :class:`~araucaria.plot.template.FigPars` for details.    
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
        If ``collection`` is not a valid Collection instance.
    TypeError
        If ``merge`` is not a valid Group instance.
    AttributeError
        If attribute ``merged_scans`` does not exist in ``merge``.
    AttributeError
        If attribute ``merged_scans`` does not exist in any of the 
        groups in ``collection``.

    Notes
    -----
    The returned plot contains the original signals and merged signal in both
    normalized :math:`\mu(E)` and  :math:`\chi(k)`.

    Important
    ---------
    Optional arguments such as ``pre_edge_kws`` and ``autobk_kws`` are used 
    to calculate the normalized :math:`\mu(E)` and :math:`\chi(k)`  
    signals. Such parameters have no effect on the merge operation.
    
    By default legends are not included in the figure. However they can be
    requested for any axis (see Example).

    See also 
    --------
	:func:`~araucaria.xas.merge.merge` : Merge groups in a collection.
    
    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria import Collection
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_dnd
        >>> from araucaria.xas import merge
        >>> from araucaria.plot import fig_merge
        >>> collection = Collection()
        >>> files = ['dnd_testfile1.dat' , 'dnd_testfile2.dat', 'dnd_testfile3.dat']
        >>> for file in files:
        ...     fpath = get_testpath(file)
        ...     group_mu = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
        ...     collection.add_group(group_mu)         # adding group to collection
        >>> report, merge = merge(collection)
        >>> fig, ax = fig_merge(merge, collection)
        >>> leg     = ax[0].legend(fontsize=8)
        >>> plt.show(block=False)
    """
    # checking class and attributes
    check_objattrs(merge, Group, attrlist=['merged_scans'], exceptions=True)
    merged_scans = merge.merged_scans
    check_objattrs(collection, Collection, attrlist=merged_scans, exceptions=True)

    # need to check if only mu_ref must be plotted
    if merge.get_mode() == 'mu_ref':
        mu_ref = True
    else:
        mu_ref = False      

    # setting kweight for EXAFS plot
    try:
        kweight = fig_pars['kweight']
    except:
        kweight = 2

    # checking normalization and backgroud removal parameters
    if pre_edge_kws is None:
        pre_edge_kws = {}
    if autobk_kws is None:
        autobk_kws = {}
        
    # ensuring a reasonable figsize
    if fig_kws is None:
        fig_kws = {'figsize' : (8,3)}
    elif 'figsize' not in fig_kws:
        fig_kws['figsize'] = (8, 3)

    # plot decorations
    fig, axes = fig_xas_template(panels='dxe', fig_pars=fig_pars, **fig_kws)
    for ax in axes:
        if ax == axes[0]:
            ax.set_title('Deriv. normalized absorption')
        elif ax == axes[1]:
            ax.set_title('Normalized absorption')
        else:
            ax.set_title('Extended fine structure')

    # processing original scans
    for i, item in enumerate(merged_scans):
        group = collection.get_group(item)
        if mu_ref:
            group = Group(**{'energy': group.energy, 'mu_ref': group.mu_ref})
        pre_edge(group, update=True, **pre_edge_kws)
        autobk(group, update=True, **autobk_kws)
        
        # calculating first derivative
        dmude = gradient(group.norm)/gradient(group.energy)
        
        # plotting scans
        axes[0].plot(group.energy, dmude,  label=item)
        axes[1].plot(group.energy, group.norm, label=item)
        axes[2].plot(group.k, group.k**kweight*group.chi, label=item)

    # processing merged spectra (copy)
    merge_copy = merge.copy()
    pre_edge(merge_copy, update=True, **pre_edge_kws)
    autobk(merge_copy, update=True, **autobk_kws)
    
    # calculating first derivative
    dmude = gradient(merge_copy.norm)/gradient(merge_copy.energy)

    # plotting spectra
    axes[0].plot(merge_copy.energy, dmude, label='merge')
    axes[1].plot(merge_copy.energy, merge_copy.norm, label='merge')
    axes[2].plot(merge_copy.k, merge_copy.k**kweight*merge_copy.chi, label='merge')

    return (fig, axes)

def fig_pre_edge(group: Group, show_pre_edge: bool=True, 
                 show_post_edge: bool=True, fig_pars: FigPars=None, 
                 **fig_kws: dict) -> Tuple[Figure,Axes]:
    """Plots the results of pre-edge substraction and normalization of a data group.

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
        ``Matplolib`` figure object.
    axes
        ``Matplotlib`` axes object. 

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
        >>> fpath   = get_testpath('dnd_testfile1.dat')
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

def fig_autobk(group: Group, show_window: bool=True, 
               fig_pars: FigPars=None, **fig_kws: dict) -> Tuple[Figure,Axes]:
    """Plots the results of background removal for a data group.

    Parameters
    ----------
    group
        Group with the normalized scan.
    show_window
        Indicates if the FT-window should be plotted.
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
        ``Matplolib`` figure object.
    axes
        ``Matplotlib`` axes object. 

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attributes from :func:`~araucaria.xas.autobk.autobk` 
        do not exist in ``group``.

    See also 
    --------
	:func:`~araucaria.xas.autobk.autobk`: Background removal of a scan.

    Notes
    -----
    The returned plot contains the original :math:`\mu(E)` and the
    background function, as well as the resulting :math:`\chi(k)`.

    Example
    -------
    .. plot::
        :context: reset

        >>> import matplotlib.pyplot as plt
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_dnd
        >>> from araucaria.xas import pre_edge, autobk
        >>> from araucaria.plot import fig_autobk
        >>> fpath   = get_testpath('dnd_testfile1.dat')
        >>> group   = read_dnd(fpath, scan='mu')
        >>> pre     = pre_edge(group, update=True)
        >>> bkg     = autobk(group, update=True)
        >>> fig, ax = fig_autobk(group, show_window=False)
        >>> plt.show(block=False)
    """
    # checking class and attributes
    attrlist = ['bkg', 'chie', 'chi', 'k', 'autobk_pars']
    check_objattrs(group, Group, attrlist=attrlist, exceptions=True)

    # get scan mode
    mode = group.get_mode()
    e0   = group.e0
    ie0  = index_nearest(group.energy, e0)

    # get kweight from autobk pars
    kw   = group.autobk_pars['kweight']
    if fig_pars is None:
        fig_pars = {'kweight' : kw}
    else:
        fig_pars['kweight'] = kw

    # plot absorbance data
    fig, axes = fig_xas_template(panels='xe', fig_pars=fig_pars, **fig_kws)
    axes[0].plot(group.energy, getattr(group, mode), label=mode)
    axes[0].plot(group.energy, group.bkg, label='bkg', zorder=-1)

    # plot exafs
    if kw == 0:
        label = '$\chi(k)$'
    elif kw == 1:
        label = '$k\chi(k)$'
    else:
        label = '$k^{%s}\chi(k)$'%kw
    axes[1].plot(group.k, group.k**kw*group.chi, label=label)

    if show_window:
        win    = group.autobk_pars['win']
        krange = group.autobk_pars['k_range']
        dk     = group.autobk_pars['dk']
        kwin   = ftwindow(group.k, x_range=krange, win=win, dx1=dk)
        axes[1].plot(group.k, kwin, color='gray', ls='--', label='win')

    for ax in axes:
        ax.legend()
    return (fig, axes)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
