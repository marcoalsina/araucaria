#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Collection of routines to extract components from a XAS dataset.
"""

def ecf(data_kws, fit_type, fit_window, ncomps=2, method='cg',
                 k_mult=2, pre_edge_kws=None, autobk_kws=None):
    """Extraction of components from a XAS dataset.
    
    This function performs extraction of componentes from a XAS
    dataset based on maximization of signal difference between components.
    

    --------------
    Required input:
    data_kws [dict]    : dictionary containing the filepaths of
                         the databases containing the spectrum and
                         references, along with the file names.
                         It requires at least the following keys:
                         'dat1_path',
                         'dat1_name',
                         'dat2_path',
                         'dat2_name'.
    fit_type [str]     : fit type. Accepted values are 'dxanes',
                         'xanes', or 'exafs'.
    fit_window [list]  : min/max fit window in either energy or
                         wavenumber. Requires 2 values.
    n_comps [int]      : number of components to extract.
                         Default is 2.
    method [str]       : optimization solver. Default is 'cg'.
                         currently it only supports 2 components.
    k_mult [int]       : multiplier for wavenumber k. Only used
                         for 'exafs' fit. Default value is 2.
    pre_edge_kws [dict]: dictionary with pre-edge parameters.
    autobk_kws [dict]  : dictionary with autobk parameters.
    --------------
    Output:
    out [obj]: Fit object containing the results of the
               linear combination fit.
    """
    import os
    import types
    from numpy import where, gradient, linspace, around
    from scipy.interpolate import interp1d
    from lmfit import Parameters, minimize
    import larch
    from larch import Group
    from larch.xafs import pre_edge, autobk
    from pyxas import get_scan_type
    from pyxas.io import read_hdf5
    from pyxas.fit import lcf_report
    
    # verifying fit type
    fit_types = ['dxanes', 'xanes','exafs']
    if fit_type not in fit_types:
        raise ValueError('fit_type %s not recognized.'%fit_type)
    
    # counting the number of spectra (nspectra)
    # and veifying that filepaths exist
    nspectra = 0
    for key in data_kws:
        if 'name' in key:
            nspectra += 1
        elif 'path' in key:
            if os.path.isfile(data_kws[key]):
                True
            else:
                raise IOError('File %s does not exist.' % data_kws[key]) 

    # verifying that the number of components to extract (ncomps) 
    # is smaller than the number of spectra (nspectra)
    # IMPORTANT: Currently ncomps is hardocded to 2
    ncomps = 2
    if nspectra < ncomps:
        raise ValueError('Number of components to extract (%s) exceeds the dataset size (%s).'
                         % (ncomps, nspectra))

    # required datasets
    # at least a dataset with 2 spectra must be provided (ncomps = 2)
    req_keys =['dat1_path', 'dat1_name', 'dat2_path', 'dat2_name']
    for i in range(nspectra-2):
        req_keys.append('dat%i_path' % (i+2))
        req_keys.append('dat%i_name' % (i+2))
    
    for key in req_keys:
        if key not in data_kws:
            raise ValueError("Argument '%s' is missing." % key)

    # storing report parameters
    pars_kws = {'ncomps':ncomps, 'fit_type':fit_type, 'fit_window':fit_window}
    if pre_edge_kws is None:
        pars_kws['pre_edge_kws'] = 'default'
    else:
        pars_kws['pre_edge_kws'] = pre_edge_kws

    # report parameters for exafs lcf
    if fit_type == 'exafs':
        pars_kws['k_mult'] = k_mult
        xvar = 'k'    # storing name of x-variable (exafs)
        if autobk_kws is None:
            pars_kws['autobk_kws'] = 'default'
        else:
            pars_kws['autobk_kws'] = autobk_kws
    
    # report parameters for xanes/dxanes lcf
    else:
        xvar = 'energy'    # storing name of x-variable (xanes/dxanes)

    # reading and processing spectra
    session   = larch.Interpreter(with_plugins=False)
    fullgroup = Group()   # container for full dataset
    datgroup  = Group()   # containerfor interpolated spectra
    
    # first a group containing the entire processed dataset is created
    # the spectra with the lowest number of points in the fit region is then
    # selected for interpolation
    for i in range(nspectra):
        # reading spectra based on required keys
        dname = 'dat'+str(i+1)
        data  = Group(**read_hdf5(data_kws[req_keys[2*i]], name=data_kws[req_keys[2*i+1]]))
        scantype = get_scan_type(data)
        
        # processing xanes spectra
        if pre_edge_kws is None:
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session)
        else:
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session, **pre_edge_kws)
        
        # prceossing exafs spectra
        if fit_type == 'exafs':
            if autobk_kws is None:
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session)
            else:
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session, **autobk_kws)
        
        # setting xvar index for spectra interpolation
        # initially we consider the entire length of xvar from the first data element
        if i == 0:
            xvals = getattr(data,xvar)
        
        # saving the index for the respective data element
        iindex = where((getattr(data, xvar) >= fit_window[0]) &
                          (getattr(data, xvar) <= fit_window[1]))
        ixvals = getattr(data,xvar)[iindex]
        
        # the smallest xvar index is kept for interpolation
        if len(ixvals) < len(xvals):
            xvals = ixvals
        
        # storing the full data element as an attribute of fullgroup
        setattr(fullgroup, dname, data)

    # parameters for fit model
    params   = Parameters()
    initvals = around(linspace(0.2,0.8, num=nspectra), decimals=1)
    
    # data interpolation based on xvals
    for i in range(nspectra):
        dname = 'dat'+str(i+1)
        data  = getattr(fullgroup, dname)
        
        if fit_type == 'exafs':
            s = interp1d(getattr(data, xvar), getattr(data, xvar)**k_mult*data.chi, kind='cubic')
        elif fit_type =='xanes':
            s = interp1d(getattr(data, xvar), data.norm, kind='cubic')
        else:
            s = interp1d(getattr(data, xvar), gradient(data.norm), kind='cubic')
        yvals = s(xvals)
    
        # setting corresponding y-variable as an attribute of datgroup
        setattr(datgroup, dname, yvals)

        # setting parameters for fit model
        parname = 'amp1'+str(i+1)
        params.add(parname, value=initvals[i], min=0, max=1, vary=True)

    # setting x-variable, ncomps, uncertainty and xi as attributes of datgroup
    setattr(datgroup, xvar, xvals)
    datgroup.ncomps = ncomps
    datgroup.eps    = 1.0    # epsilon value for lmfit minimization
    datgroup.xi     = 1e-4   # xi value to prevent division by zero 
    
    # perform fit
    out = minimize(ec_minfunc, params, method='cg', args=(datgroup,),)
    #help(minimize)
    
     # storing data and arguments
    #fit = sum_references(out.params, datgroup)
    #datgroup.fit = fit
    
    out.data_group = datgroup
    out.data_kws   = data_kws
    out.pars_kws   = pars_kws

    # assigning save methods to out object
    out.fit_report      = types.MethodType(lcf_report, out)
    out.save_fit_report = types.MethodType(save_lcf_report, out)
    out.save_fit_data   = types.MethodType(save_lcf_data, out)

    return (out)

def ecf_minfunc(pars, data):
    """
    This function returns the residuals of the substraction
    of a spectrum from its LCF with known references
    standards.
    """
    from numpy import empty, mean, std
    from numpy.linalg import norm
    
    nspectra = len(pars)

    # initializing dict for components (x1,x2,...)
    # each component is stored as an empty array
    # it is more efficient to initialize the entire matrix
    # and then start populating it with data
    # IMPORTANT: Currently ncomps is hardcoded to 2.
    comps = {}
    nrows = len(data.dat1) 
    ncols = int(nspectra*(nspectra-1)/2)
    
    for i in range(data.ncomps):
        comps['x%s'%(i+1)] = empty((nrows,ncols))

    # we use hk notation to compute components
    # hk indexing starts from 1
    j = 0    # auxiliary index to populate each x array
    for h in range(1,nspectra+1):
        for k in range(h+1, nspectra+1):
            if data.ncomps == 2:
                denom = pars['amp1'+str(h)] - pars['amp1'+str(k)] + data.xi
                
                comps['x1'][:,j] = ((1 - pars['amp1'+str(k)]) * getattr(data, 'dat'+str(h)) -
                                    (1 - pars['amp1'+str(h)]) * getattr(data, 'dat'+str(k))) / denom
                
                comps['x2'][:,j] = (  (- pars['amp1'+str(k)]) * getattr(data, 'dat'+str(h)) +
                                      (  pars['amp1'+str(h)]) * getattr(data, 'dat'+str(k))) / denom
            j += 1    # updating counter
    
    # once calculation of components is done we can return the residual value
    if data.ncomps == 2:
        for i in range(data.ncomps):
            comps['x%s_mean'%(i+1)] = mean(comps['x%s'%(i+1)], axis=1)
        
        # norm over difference
        norm_x     = norm(comps['x1_mean']-comps['x2_mean'])
        # norm over standard deviation
        norm_sigma = norm(std(comps['x1'], axis=1)) + norm(std(comps['x1'], axis=1))
        
    return (norm_sigma - norm_x)
