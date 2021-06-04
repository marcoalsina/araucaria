#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from numpy import gradient
from matplotlib.pyplot import Axes, Figure
from .. import Group, Collection
from ..xas import pre_edge, autobk
from .template import fig_xas_template, FigPars
from ..utils import check_objattrs

def fig_merge(merge: Group, collection: Collection, 
              pre_edge_kws: dict=None, autobk_kws:dict =None,
              fig_pars: FigPars=None, **fig_kws) -> Tuple[Figure,Axes]:
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
        Matplolib figure object.
    axes
        Matplotlib axes object. 

    Raises
    ------
    TypeError
        If ``collection`` is not a valid Collection instance.
    TypeError
        If ``merge`` is not a valid Group instance.
    AttributeError
        If attribute ``merged_scans`` does not exist in ``merge``.
    AttributeError
        If any attribute listed ``merged_scans`` does not exist in ``collection``.

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

if __name__ == '__main__':
    import doctest
    doctest.testmod()