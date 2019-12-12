#!/usr/bin/python
# -*- coding: utf-8 -*-

def fig_merge(group, merge, e0, pre_edge_kws=None, 
              autobk_kws=None, fig_pars=None, **fig_kws):
    """
    This function plots the merged scans.
    --------------
    Required input:
    group : list of Larch groups to be merged in xmu.
    merge : Larch group containing the merged xmu scans.
    edge [float]: Edge energy.
    pre_edge_kws [dict]: dictionary with arguments for normalization
    """
    import numpy as np    
    import matplotlib.pyplot as plt
    import larch
    from larch.xafs import find_e0, pre_edge, autobk
    from pyxas import get_scan_type
    from pyxas.plot import fig_xas_template

    # larch parameters
    session = larch.Interpreter(with_plugins=False)

    if pre_edge_kws == {}:
        # using default parameters
        pre_edge_kws={'pre1':-150, 'pre2':-50, 'nnorm':3, 'norm1':150, 'e0':edge}

    autobk_kws={'rbkg':1.0, 'kweight':2, 'dk':0}

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
    
    for i, data in enumerate(group):
        scantype = get_scan_type(data)
        #e0 = find_e0(data.energy, getattr(data, scan), _larch=session)
        pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session, **pre_edge_kws)
        autobk(data.energy,   getattr(data, scantype), group=data, _larch=session, **autobk_kws)
        try:
            data.e_offset
        except:
            data.e_offset = 0.0
         
        axes[0].plot(data.energy, np.gradient(getattr(data, scan))/np.gradient(data.energy), lw=lw, label=data.name[:-4])
        axes[1].plot(data.energy, data.flat, lw=lw)
        axes[2].plot(data.k, data.k**2*data.chi, lw=lw)
    
    #e0 = find_e0(merge.energy, getattr(merge, scan), _larch=session)
    pre_edge(merge.energy, getattr(merge, scan), group=merge, _larch=session, **pre_edge_kws)
    autobk(merge.energy,   getattr(merge, scan), group=merge, _larch=session, **autobk_kws)
    
    axes[0].plot(merge.energy, np.gradient(getattr(merge, scan))/np.gradient(merge.energy), 
                 lw=lw, label='Merge', color='firebrick')
    axes[0].legend(fontsize=8, loc='lower left', edgecolor='k')
    axes[1].plot(merge.energy, merge.flat, lw=lw, color='firebrick')
    axes[2].plot(merge.k, merge.k**2*merge.chi, lw=lw, color='firebrick')
            
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.2)
    
    return(fig, axes)