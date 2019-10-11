#!/usr/bin/env python
'''
filename: plot.py

Collection of functions to plot XAS data.

Implemented functions:
    fig_xas_template
    plot_merged_scans

Marco A. Alsina
08/23/2019
'''

def fig_xas_template(panels='xx', **fig_pars):
    '''
    This funtion returns a Matpoltlib figure and
    axes object based on user specification regarding 
    the XAFS spectra to plot.
    --------------
    Required input:
    panels [str] : characters defining the XAFS spectra
                   to plot. Valid arguments are as follows:
                   'd': derivative of XANES spectra.
                   'x': XANES spectra.
                   'e': EXAFS spectra.
                   'r': FT EXAFS spectra.
                   characters can be concatenated to
                   produce multiple panels.
                   Examples: 'dxe', 'xx', 'xer'.
    fig_pars: dictionary with arguments for the figure.
    Valid arguments include:
        fig_w    [float]: figure width in inches.
        fig_h    [float]: figure height in inches.
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
        
    --------------
    Output:
    fig :  Matplolib figure object.
    ax:    Matplotlib axes object.
    '''
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # valid axis type
    valid_panel_types = ['d', 'x', 'e', 'r']

    # converting argument to list of strings
    panels = list(panels)

    try:    
        for panel in panels:
            panel in valid_panel_types
    except ValueError:
        print ('Error: %s panel type not recognized!' % panel)

    # creating figure and axes
    try:
        fig, axes = plt.subplots(1, len(panels), figsize=(fig_pars['fig_w'], fig_pars['fig_h']))
    except:
        fig, axes = plt.subplots(1, len(panels))

    # formatting panels (axes)
    for i, panel in enumerate(panels):
        # XANES derivative axis
        if panel == 'd':
            axes[i].set_xlabel('Energy [eV]')
            axes[i].set_ylabel('Deriv. abs. [a.u.]')

            try:
                axes[i].set_xlim(fig_pars['e_range'])
            except:
                pass
            try:
                axes[i].set_ylim(fig_pars['dmu_range'])
            except:
                pass
            try:
                axes[i].set_xticks(fig_pars['e_ticks'])
            except:
                pass
            try:
                axes[i].set_yticks(fig_pars['dmu_ticks'])
            except:
                pass

        # XANES axis
        elif panel == 'x':
            axes[i].set_xlabel('Energy [eV]')
            axes[i].set_ylabel('Norm. abs. [a.u.]')
            try:
                axes[i].set_xlim(fig_pars['e_range'])
            except:
                pass
            try:
                axes[i].set_ylim(fig_pars['mu_range'])
            except:
                pass
            try:
                axes[i].set_xticks(fig_pars['e_ticks'])
            except:
                pass
            try:
                axes[i].set_yticks(fig_pars['mu_ticks'])
            except:
                pass

        # EXAFS axis
        elif panel == 'e':
            try:
                k = fig_pars['k_mult']
            except:
                k = 2
            axes[i].set_xlabel(r'$k$ [$\AA^{-1}$]')
            axes[i].set_ylabel(r'$k^%i\chi(k)$' % k)
            try:
                axes[i].set_xlim(fig_pars['k_range'])
            except:
                pass
            try:
                axes[i].set_ylim(fig_pars['chi_range'])
            except:
                pass
            try:
                axes[i].set_xticks(fig_pars['k_ticks'])
            except:
                pass
            try:
                axes[i].set_yticks(fig_pars['chi_ticks'])
            except:
                pass

        # FT EXAFS axis
        if panel == 'r':
            try:
                k = fig_pars['k_mult']
            except:
                k = 2
            axes[i].set_xlabel(r'$R$ [$\AA$]')
            axes[i].set_ylabel(r'|$\chi(R)$|  [$\AA^{-%i}$]' % (k+1))
            try:
                axes[i].set_xlim(fig_pars['r_range'])
            except:
                pass
            try:
                axes[i].set_ylim(fig_pars['mag_range'])
            except:
                pass
            try:
                axes[i].set_xticks(fig_pars['r_ticks'])
            except:
                pass
            try:
                axes[i].set_yticks(fig_pars['mag_ticks'])
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
    fig_pars = {'fig_w'   : 12.0,                  # image width
                'fig_h'   : 4.00,                  # image height
                'e_range' : [edge-50,edge+75],     # plotting window energy
                'mu_range': [-0.1,1.5],            # plotting window mu
                'k_mult'  : 2,                     # k multiplier
                'k_range' : [0,16]                 # plotting window k-space
                }
    
    # plot decorations
    fig, axes = fig_xas_template(panels='dxe', **fig_pars)
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
