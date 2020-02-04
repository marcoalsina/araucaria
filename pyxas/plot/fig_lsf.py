#!/usr/bin/python
# -*- coding: utf-8 -*-

def fig_lsf(out, annotate=True, fontsize=8, fig_pars=None, **fig_kws):
    """
    This funtion returns a Matpoltlib figure and
    axes objects containing the plotted results of 
    a feff least-squares fit (feffit) on an EXAFS 
    spectrum.
    --------------
    Required input:
    out [obj]      : valid feffit object.
    annotate [bool]: if 'true' it annotates the results of
                     the first shell fitting on the figure.
    fontsize [int] : font size for legend and annotations.
    fig_pars [dict]: optional arguments for the figure.
                     Check the function ´fig_xas_template´
                     for valid arguments.
    fig_kws [dict] : arguments to pass to the Matplotlib
                     subplots instance.  
    --------------
    Output:
    fig :  Matplolib figure object.
    axes:  Matplotlib axes object.
    """
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
    except:
        raise ValueError("'%s' is not a valid feffit object."%out)
    
    # setting the figure type
    fig, axes = fig_xas_template(panels='er', fig_pars=fig_pars, **fig_kws)
    
    # plotting data and fit result
    dset    = out.datasets[0].data
    model   = out.datasets[0].model
    pardir  = out.datasets[0].pathlist[0].path_paramvals()
    reff    = out.datasets[0].pathlist[0].reff
    try:
        k_mult = fig_pars['k_mult']
    except:
        k_mult = out.pars_kws['k_mult']

    axes[0].plot(dset.k, dset.k**k_mult*dset.chi, label=out.data_kws['spectrum_name'])
    axes[0].plot(model.k, model.k**k_mult*model.chi, label='fit')

    # calculating phase-corrected FT
    #xftf_pha(dset, path1, k_mult)
    axes[1].plot(dset.r, dset.chir_mag, label=out.data_kws['spectrum_name'])
    axes[1].plot(model.r, model.chir_mag, label='fit')
    
    if annotate:
        # summary results for plot
        summary = 'First shell fit:\n'\
        r'$s_0^2$ = %1.1f' '\n'\
        r'$N$ = %1.2f' '\n'\
        r'$R$ = %1.3f $\AA$' '\n'\
        r'$\sigma^2$ = %1.3f $\AA^2$' '\n'\
        r'$\Delta E_0$ = %1.2f $eV$'%(
            pardir['s02'],
            pardir['degen'],
            pardir['deltar'] + reff,
            pardir['sigma2'],
            pardir['e0'])

        axes[1].text(2.5,0.2, summary, fontsize=fontsize)

    # ax legend
    axes[0].legend(loc='lower right', edgecolor='k', fontsize=fontsize, numpoints=1)

    return(fig, axes)
