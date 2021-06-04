#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from matplotlib.pyplot import Axes, Figure
from .. import Group
from .template import fig_xas_template, FigPars
from ..xas.xasft import ftwindow
from ..utils import check_objattrs, index_nearest

def fig_autobk(group: Group, show_window: bool=True, 
               fig_pars: FigPars=None, **fig_kws) -> Tuple[Figure,Axes]:
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
        Matplolib figure object.
    axes
        Matplotlib axes object. 

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