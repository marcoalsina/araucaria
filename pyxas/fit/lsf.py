#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Routine to perform a feffit least squares fit (LSF) on a XAS spectrum.
"""

def lsf(data_kws, pars, path_pars, k_mult=2, pre_edge_kws=None, 
        autobk_kws=None, xftf_kws=None):
    """Least squares fit on a XAS spectrum.
    
    This function performs a least squares fit
    on a spectrum given a FEFF calculation.
    --------------
    Required input:
    data_kws [dict]    : dictionary containing the filepaths of
                         the database containing the spectrum and
                         feffpaths
                         It requires at least the following keys:
                         'spectrum_path',
                         'spectrum_name',
                         'feff1_path'.
    pars [group]       : group of fit parameters.
    path_pars[list]    : list containing dictionaries of variables
                         for each path.
    k_mult [int]       : multiplier for wavenumber k. Only used
                         for 'exafs' fit. Default value is 2.
    pre_edge_kws [dict]: dictionary with pre-edge parameters.
    autobk_kws [dict]  : dictionary with autobk parameters.
    xftf_kws [dict]    : dictionary with xftf parameters.
    --------------
    Output:
    out [obj]: Fit object containing the results of the
               linear combination fit.
    """
    import os
    import types
    import larch
    from larch import Group
    from larch.fitting import param
    from larch.xafs import pre_edge, autobk
    from larch.xafs.feffdat import feffpath
    from larch.xafs.feffit import feffit, feffit_transform, feffit_dataset
    from pyxas import get_scan_type
    from pyxas.io import read_hdf5
    from pyxas.fit import lsf_report, save_lsf_report, save_lsf_data
    
    # counting the number of feffpaths
    nfpaths = 0
    for key in data_kws:
        if 'feff' in key:
            nfpaths += 1
        
        # checking if filepaths exist
        if 'path' in key:
            if os.path.isfile(data_kws[key]):
                True
            else:
                raise IOError('File %s does not exist.' % data_kws[key])
    
    # required datasets
    # at least a spectrum and a single feff path must be provided
    req_keys =['spectrum_path', 'spectrum_name', 'feff1_path']
    for i in range(nfpaths-1):
        req_keys.append('feff%i_path' % (i+1))
    
    for key in req_keys:
        if key not in data_kws:
            raise ValueError("Argument '%s' is missing." % key)
            
    # required fit parameters
    req_path_pars = ['s02', 'e0', 'sigma2', 'deltar', 'degen']
    for path_pars_kws in path_pars:
        for key in req_path_pars:
            if key not in path_pars_kws:
                raise ValueError("Parameter '%s' is missing in %s." % (key, path_pars_kws))
    
    # dict for report parameters
    pars_kws = {}
    
    # reading and processing spectrum
    session = larch.Interpreter(with_plugins=False)
    datgroup = Group()   # container for spectra to perform LCF analysis
    
    data = Group(**read_hdf5(data_kws[req_keys[0]], name=data_kws[req_keys[1]]))
    scantype = get_scan_type(data)
        
    if pre_edge_kws is None:
        pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session)
        pars_kws['pre_edge_kws'] = 'default'
    else:
        pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session, **pre_edge_kws)
        pars_kws['pre_edge_kws'] = pre_edge_kws
        
    if autobk_kws is None:
        autobk(data.energy, getattr(data, scantype), group=data, _larch=session)
        pars_kws['autobk_kws'] = 'default'
    else:
        autobk(data.energy, getattr(data, scantype), group=data, _larch=session, **autobk_kws)
        pars_kws['autobk_kws'] = autobk_kws
    
    # set transform/fit ranges
    if xftf_kws is None:
        trans = feffit_transform(_larch=session)
        pars_kws['xftf_kws'] = 'default'
    else:
        trans = feffit_transform(_larch=session, **xftf_kws)    
        pars_kws['xftf_kws'] = xftf_kws

    pars_kws['k_mult'] = k_mult
    
    # define feff paths, give expressions for path parameters
    pathlist = []
    for i in range(nfpaths):
        path_val = 'feff%i_path'%(i+1)
        pathlist.append(feffpath(data_kws[path_val], _larch=session, **path_pars[i]))
    
    # define dataset to include data, pathlist, transform
    dset  = feffit_dataset(data=data, pathlist=pathlist, transform=trans, _larch=session)
    
    # perform fit
    out   = feffit(pars, dset, _larch=session)
    
    # assigning attributes to out object
    out.data_kws   = data_kws
    out.pars_kws   = pars_kws

    # assigning save methods to out object
    out.lsf_report      = types.MethodType(lsf_report, out)
    out.save_lsf_report = types.MethodType(save_lsf_report, out)
    out.save_lsf_data   = types.MethodType(save_lsf_data, out)

    return (out)
