#!/usr/bin/env python
'''
filename: plot.py

Collection of functions to plot XAS data.

Implemented functions:
    fig_xas_template
    plot_merged_scans
'''

def fig_xas_template(panels='xx', fig_pars=None, **fig_kws):
    '''
    This funtion returns a Matpoltlib figure and
    axes object based on user specification regarding 
    the XAFS spectra to plot.
    Panel elements (axes) are indexed in row-major, i.e.
    C-style order.
    --------------
    Required input:
    panels [str] : characters defining the type of 
                   panels (axes) to plot.
                   Valid arguments are as follows:
                   'd': derivative of XANES spectra.
                   'x': XANES spectra.
                   'e': EXAFS spectra.
                   'r': FT EXAFS spectra.
                   '/': designates a new row.
                   characters can be concatenated to
                   produce multiple panels and rows.
                   Examples: 'dxe', 'xx', 'xer', 'xx/xx'.
    fig_pars [dict]: optional arguments for the figure.
    Valid arguments include:
        prop_cycle[list]: list of prop_cycle dictionaries for each
                          panel. List is cycled if the elements of 
                          the list are less than the number of 
                          panels.
        e_range   [list]: XANES energy range.
        e_ticks   [list]: XANES energy tick marks.
        mu_range  [list]: XANES norm absorption range.
        mu_ticks  [list]: XANES norm absorption tick marks.
        dmu_range [list]: XANES deriv norm absorption range.
        dmu_ticks [list]: XANES deriv norm absorption tick marks.
        k_mult   [float]: EXAFS k-multiplier
        k_range   [list]: EXAFS wavenumber range.
        k_ticks   [list]: EXAFS wavenumber tick marks.
        chi_range [list]: EXAFS chi range.
        chi_ticks [list]: EXAFS chi range
        r_range   [list]: FT EXAFS r range.
        r_ticks   [list]: FT EXAFS r tick marks.
        mag_range [list]: FT EXAFS magnitude range.
        mag_ticks [list]: FT EXAFS magnitude tick marks.
    fig_kws [dict]: arguments to pass to the Matplotlib
                    subplots instance.  
    --------------
    Output:
    fig :  Matplolib figure object.
    ax:    Matplotlib axes object.
    '''
    from itertools import cycle
    from numpy import ravel
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # valid axis type
    valid_panel_types = ['d', 'x', 'e', 'r']

    # counting the number of rows
    rows  = panels.split('/')
    nrows = len(rows)

    # veryfing that each row has the same number of elements
    if nrows == 1:
        ncols = len(panels)
    else:
        ncols = len(rows[0])
        for cols in rows[1:]:
            if len(cols) != ncols:
                raise ValueError('number of columns does not match between rows')

    # converting panel argument to list of strings
    panels = ''.join(rows)    # merging all rows in a single string
    panels = list(panels)     # separating string in characters

    try:    
        for panel in panels:
            panel in valid_panel_types
    except ValueError:
        print('%s panel type not recognized!' % panel)

    # creating figure and axes
    fig, axes = plt.subplots(nrows, ncols, **fig_kws)

    # formatting axes
    try:
        prop_cycler = cycle(fig_pars['prop_cycle'])
        prop_elem   = next(prop_cycler)
    except:
        pass
    
    for i, ax in enumerate(ravel(axes)):
        # prop_cycles
        try:
            ax.set_prop_cycle(**prop_elem)
            prop_elem   = next(prop_cycler)
        except:
            pass
 
        # XANES derivative axis
        if panels[i] == 'd':
            ax.set_xlabel(r'Energy [$eV$]')
            ax.set_ylabel('Deriv. abs. [a.u.]')
            try:
                ax.set_xlim(fig_pars['e_range'])
            except:
                pass
            try:
                ax.set_ylim(fig_pars['dmu_range'])
            except:
                pass
            try:
                ax.set_xticks(fig_pars['e_ticks'])
            except:
                pass
            try:
                ax.set_yticks(fig_pars['dmu_ticks'])
            except:
                pass

        # XANES axis
        elif panels[i] == 'x':
            ax.set_xlabel(r'Energy [$eV$]')
            ax.set_ylabel('Norm. abs. [a.u.]')
            try:
                ax.set_xlim(fig_pars['e_range'])
            except:
                pass
            try:
                ax.set_ylim(fig_pars['mu_range'])
            except:
                pass
            try:
                ax.set_xticks(fig_pars['e_ticks'])
            except:
                pass
            try:
                ax.set_yticks(fig_pars['mu_ticks'])
            except:
                pass

        # EXAFS axis
        elif panels[i] == 'e':
            try:
                k = fig_pars['k_mult']
            except:
                k = 2
            ax.set_xlabel(r'$k$ [$\AA^{-1}$]')
            ax.set_ylabel(r'$k^%i\chi(k)$' % k)
            try:
                ax.set_xlim(fig_pars['k_range'])
            except:
                pass
            try:
                ax.set_ylim(fig_pars['chi_range'])
            except:
                pass
            try:
                ax.set_xticks(fig_pars['k_ticks'])
            except:
                pass
            try:
                ax.set_yticks(fig_pars['chi_ticks'])
            except:
                pass

        # FT EXAFS axis
        if panels[i] == 'r':
            try:
                k = fig_pars['k_mult']
            except:
                k = 2
            ax.set_xlabel(r'$R$ [$\AA$]')
            ax.set_ylabel(r'|$\chi(R)$|  [$\AA^{-%i}$]' % (k+1))
            try:
                ax.set_xlim(fig_pars['r_range'])
            except:
                pass
            try:
                ax.set_ylim(fig_pars['mag_range'])
            except:
                pass
            try:
                ax.set_xticks(fig_pars['r_ticks'])
            except:
                pass
            try:
                ax.set_yticks(fig_pars['mag_ticks'])
            except:
                pass

    return (fig, axes)

def plot_merged_scans(group, merge, edge, scan='mu', **pre_edge_kws):
    '''
    This function plots the merged scans.
    --------------
    Required input:
    group : list of Larch groups to be merged in xmu.
    merge : Larch group containing the merged xmu scans.
    edge [float]: Edge energy.
    pre_edge_kws [dict]: dictionary with arguments for normalization
    '''
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
    
    # plot parameters
    lw = 1.2                                       # line width
    fig_pars = {'e_range' : [edge-50,edge+75],     # plotting window energy
                'mu_range': [-0.1,1.5],            # plotting window mu
                'k_mult'  : 2,                     # k multiplier
                'k_range' : [0,16]                 # plotting window k-space
                }
    fig_kws = {'figsize': (12,4)}                  # image with and height
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
    
    separator = 63*'='
    print (separator)
    print ('{0:3}{1:30}{2:10}{3:10}{4:10}'.format('id', 'name', 'type', 'e_offset', 'e0'))
    print (separator)
    
    for i, data in enumerate(group):
        scan = get_scan_type(data)
        #e0 = find_e0(data.energy, getattr(data, scan), _larch=session)
        pre_edge(data.energy, getattr(data, scan), group=data, _larch=session, **pre_edge_kws)
        autobk(data.energy,   getattr(data, scan), group=data, _larch=session, **autobk_kws)
        try:
            data.e_offset
        except:
            data.e_offset = 0.0
        print ('{0:<3}{1:30}{2:10}{3:<10.3f}{4:<10.3f}'.format(i+1, data.name, scan, data.e_offset, data.e0))
         
        axes[0].plot(data.energy, np.gradient(getattr(data, scan))/np.gradient(data.energy), lw=lw, label=data.name[:-4])
        axes[1].plot(data.energy, data.flat, lw=lw)
        axes[2].plot(data.k, data.k**2*data.chi, lw=lw)
    
    print (separator)
    #e0 = find_e0(merge.energy, getattr(merge, scan), _larch=session)
    pre_edge(merge.energy, getattr(merge, scan), group=merge, _larch=session, **pre_edge_kws)
    autobk(merge.energy,   getattr(merge, scan), group=merge, _larch=session, **autobk_kws)
    print ('{0:<3}{1:30}{2:10}{3:<10.3f}{4:<10.3f}'.format('','Merge', scan, 0, merge.e0))
    print (separator)
    
    axes[0].plot(merge.energy, np.gradient(getattr(merge, scan))/np.gradient(merge.energy), 
                 lw=lw, label='Merge', color='firebrick')
    axes[0].legend(fontsize=8, loc='lower left', edgecolor='k')
    axes[1].plot(merge.energy, merge.flat, lw=lw, color='firebrick')
    axes[2].plot(merge.k, merge.k**2*merge.chi, lw=lw, color='firebrick')
            
    plt.tight_layout()
    plt.show()
