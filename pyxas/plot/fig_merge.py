#!/usr/bin/python
# -*- coding: utf-8 -*-

def fig_merge(group, merge, pre_edge_kws=None, 
              autobk_kws=None, fig_pars=None, **fig_kws):
    """
    This function plots the merged scans.
    --------------
    Required input:
    group : list of Larch groups to be merged in xmu.
    merge : Larch group containing the merged xmu scans.
    fix_e0 [float]: Fix edge energy.
    pre_edge_kws [dict]: dictionary with arguments for normalization.
    autobk_kws [dict]: dictionary with arguments for normalization.
    fig_pars [dict]: dictionary with arguments for normalization.
    fig_kws [dict]: dictionary with arguments for normalization.
    """
    import numpy as np    
    import matplotlib.pyplot as plt
    import larch
    from larch.xafs import find_e0, pre_edge, autobk
    from pyxas import get_scan_type
    from pyxas.plot import fig_xas_template

    # larch parameters
    session = larch.Interpreter(with_plugins=False)

    # setting k-mult for EXAFS plot
    try:
        k_mult = fig_pars['k_mult']
    except:
        k_mult = 2

    # plot decorations
    fig, axes = fig_xas_template(panels='dxe', fig_pars=fig_pars, **fig_kws)
    for ax in axes:
        ax.grid()
        if ax == axes[0]:
            ax.set_title('Reference channel alignment')
        elif ax == axes[1]:
            ax.set_title('Normalized absorption')
        else:
            ax.set_title('Extended fine structure')
    
    # processing group spectra
    for i, data in enumerate(group):
        scantype = get_scan_type(data)
        if pre_edge_kws is None:
            pre_edge_kws = {}
        if autobk_kws is None:
            autobk_kws = {}
        autobk(data.energy,   getattr(data, scantype), group=data, _larch=session, **autobk_kws)
        pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session, **pre_edge_kws)
 
        axes[0].plot(data.energy, np.gradient(getattr(data, scantype))/np.gradient(data.energy))
        axes[1].plot(data.energy, data.flat, label=data.name[:-4])
        axes[2].plot(data.k, data.k**k_mult*data.chi)
    
    # processing merged spectra
    if pre_edge_kws is None:
        pre_edge(merge.energy, getattr(merge, scantype), group=merge, _larch=session)
    else:
        pre_edge(merge.energy, getattr(merge, scantype), group=merge, _larch=session, **pre_edge_kws)
    if autobk_kws is None:
        autobk(merge.energy,   getattr(merge, scantype), group=merge, _larch=session, **autobk_kws)
    else:
        autobk(merge.energy,   getattr(merge, scantype), group=merge, _larch=session)
    
    # plotting spectra
    axes[0].plot(merge.energy, np.gradient(getattr(merge, scantype))/np.gradient(merge.energy))
    axes[1].plot(merge.energy, merge.flat, label='merge')
    axes[2].plot(merge.k, merge.k**k_mult*merge.chi)

    axes[1].legend(loc=0, fontsize=8, edgecolor='k')            
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    
    return (fig, axes)
