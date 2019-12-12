#!/usr/bin/python
# -*- coding: utf-8 -*-

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
    if 'prop_cycle' in fig_pars:
        prop_cycler = cycle(fig_pars['prop_cycle'])
    
    for i, ax in enumerate(ravel(axes)):
        # prop_cycles
        try:
            prop_elem   = next(prop_cycler)
            ax.set_prop_cycle(**prop_elem)
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