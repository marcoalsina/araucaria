#!/usr/bin/python
# -*- coding: utf-8 -*-

def fig_ecf(out, annotate=True, fontsize=8, step=0.5, fig_pars=None, **fig_kws):
    '''
    This funtion returns a Matpoltlib figure and
    axes object containing the plotted results of an ECF analysis.
    --------------
    Required input:
    out [obj]       : valid LMFIT object for the LCF.
    annotate [bool] : if 'true' it annotates the fit results
                      on the figure.
    fontsize [float]: font size for legends and annotations.
    step [float]    : vertical separation step between plots.
    fig_pars [dict] : optional arguments for the figure.
                      Check the function 'fig_xas_template'
                      for valid arguments.
    fig_kws [dict] : arguments to pass to the Matplotlib
                     subplots instance.  
    --------------
    Output:
    fig :  Matplolib figure object.
    axes:  Matplotlib axes object.
    '''
    from numpy import gradient, ptp
    import larch
    from larch import Group
    from larch.xafs import pre_edge, autobk
    import matplotlib.pyplot as plt
    from pyxas import get_scan_type
    from pyxas.io import read_hdf5
    from pyxas.plot import fig_xas_template
    
    # verifying the out object
    try:
        out.pars_kws
        out.data_kws
    except ValueError:
        print('Object is not a valid lmift object from ECF.')
    
    # setting the figure type
    if out.pars_kws['fit_type'] == 'exafs':
        panels = 'ee'
    elif out.pars_kws['fit_type'] == 'xanes':
        panels = 'xx'
    else:
        panels = 'dd'
    
    fig, axes = fig_xas_template(panels=panels, fig_pars=fig_pars, **fig_kws)
    
    # reading original spectra
    names = []
    paths = []
    for item in out.data_kws:
        if 'name' in item:
            names.append(out.data_kws[item])
        elif 'path' in item:
            paths.append(out.data_kws[item])
    
    # processing original spectra
    session = larch.Interpreter(with_plugins=False)
    for i, name in enumerate(names):
        data = Group(**read_hdf5(paths[i], name=name))
        scantype = get_scan_type(data)
    
        if out.pars_kws['pre_edge_kws'] == 'default':
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session)
        else:
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session, 
                     **out.pars_kws['pre_edge_kws'])
    
        # plotting original spectra
        xloc = axes[0].set_xlim()[0] + 0.98*ptp(axes[0].set_xlim())
        # EXAFS spectra
        if out.pars_kws['fit_type'] == 'exafs':
            if out.pars_kws['autobk_kws'] == 'default':
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session)
            else:
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session, 
                       **out.pars_kws['autobk_kws'])
        
            axes[0].plot(data.k, i*step + data.k**out.pars_kws['k_mult']*data.chi)
            axes[0].text(xloc, (0.5 + i)*step, name, fontsize=fontsize, ha='right')
    
        # XANES spectra
        elif out.pars_kws['fit_type'] == 'xanes':
            axes[0].plot(data.energy, i*step + data.norm)
            axes[0].text(xloc, 1 + (0.25 + i)*step, name, fontsize=fontsize, ha='right')
    
        # DXANES spectra
        else:
            axes[0].plot(data.energy, i*step + gradient(data.norm))
            axes[0].text(xloc, (0.25 + i)*step, name, fontsize=fontsize, ha='right')

    # plotting fitted data
    ncomps   = out.pars_kws['ncomps']
    nspectra = len(names)
    ncols    = int(nspectra*(nspectra-1)/2)
    
    # EXAFS fit
    if out.pars_kws['fit_type'] == 'exafs':
        for i in range(ncomps):
            axes[1].plot(out.data_group.k, out.data_group.comps['x%s_mean'%(i+1)], label='x%s'%(i+1))
            for j in range(ncols):
                axes[1].plot(out.data_group.k, out.data_group.comps['x%s'%(i+1)][:,j], zorder=-1)
        
    
    # XANES or DXANES fit
    else:
        for i in range(ncomps):
            axes[1].plot(out.data_group.energy, 0.5*step + \
                         out.data_group.comps['x%s_mean'%(i+1)], label='x%s'%(i+1))
    
    axes[1].legend(loc='upper right', edgecolor='k', fontsize=fontsize)
    
    # increasing y-lim to include legend
    yloc = axes[1].get_ylim()
    axes[1].set_ylim(yloc[0], 1.3*yloc[1])
    
    #if annotate:
        # summary results for plot
        #summary = r'$\chi^2$ = %1.4f' % out.chisqr +'\n'
        #for i in range(1,len(out.params)+1):
            #val = out.params['amp'+str(i)].value
            #err = out.params['amp'+str(i)].stderr
            #summary += names[i]+r': %1.2f$\pm$%1.2f' % (val, err)
            #summary += '\n'

        #if out.pars_kws['fit_type']   == 'dxanes':
            #axes[1].text(xloc, yloc[0] + 0.6*ptp(yloc), summary, ha='right', va='center', fontsize=fontsize)
        #elif out.pars_kws['fit_type'] == 'xanes':
            #axes[1].text(xloc, yloc[0] + 0.4*ptp(yloc), summary, ha='right', va='center', fontsize=fontsize)
        #else:
            #axes[1].text(xloc, yloc[0] + 0.5*ptp(yloc), summary, ha='right', va='top', fontsize=fontsize)

    # axes decorators
    axline_kws = {'color': 'lightgray', 'dashes': [4,2], 'linewidth': 1.0, 'zorder':-2} 
    for ax in axes:
        ax.set_yticks([])
        ax.axvline(out.pars_kws['fit_window'][0], **axline_kws)
        ax.axvline(out.pars_kws['fit_window'][1], **axline_kws)
   
    return(fig, axes)
