#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of routines to perform linear combination fit (LCF) 
analysis on a XAS spectrum.
"""

def lcf(data_kws, fit_type, fit_window, k_mult=2,
        sum_one=True, pre_edge_kws=None, autobk_kws=None):
    """
    This function performs a linear combination fit 
    on a spectrum given a set of references.
    --------------
    Required input:
    data_kws [dict]    : dictionary containing the filepaths of
                         the databases containing the spectrum and
                         references, along with the file names.
                         It requires at least the following keys:
                         'spectrum_path',
                         'spectrum_name',
                         'ref1_path',
                         'ref1_name'.
    fit_type [string]  : fit type. Accepted values are 'dxanes',
                         'xanes', or 'exafs'.
    fit_window [list]  : min/max fit window in either energy or
                         wavenumber. Requires 2 values.
    k_mult [int]       : multiplier for wavenumber k. Only used
                         for 'exafs' fit. Default value is 2.
    sum_one [bool]     : If 'true' the sum of fractions is forced
                         to be one. Default is 'true'.
    pre_edge_kws [dict]: dictionary with pre-edge parameters.
    autobk_kws [dict]  : dictionary with autobk parameters.
    --------------
    Output:
    out [obj]: Fit object containing the results of the
               linear combination fit.
    """
    import os
    import types
    from numpy import where, gradient, around
    from scipy.interpolate import interp1d
    from lmfit import Parameters, minimize
    import larch
    from larch import Group
    from larch.xafs import pre_edge, autobk
    from pyxas import get_scan_type
    from pyxas.io import read_hdf5
    from pyxas.fit import residuals, sum_references
    from pyxas.fit import lcf_report, save_lcf_report, save_lcf_data
    
    # verifying fit type
    fit_types = ['dxanes', 'xanes','exafs']
    if fit_type not in fit_types:
        raise ValueError('fit_type %s not recognized.'%fit_type)
    
    # counting the number of spectra
    # and checking if filepaths exist
    nspectra = 0
    for key in data_kws:
        if 'name' in key:
            nspectra += 1
        elif 'path' in key:
            if os.path.isfile(data_kws[key]):
                True
            else:
                raise IOError('File %s does not exist.' % data_kws[key]) 
    
    # required datasets
    # at least a spectrum and a single reference must be provided
    req_keys =['spectrum_path', 'spectrum_name', 'ref1_path', 'ref1_name']
    for i in range(nspectra-2):
        req_keys.append('ref%i_path' % (i+2))
        req_keys.append('ref%i_name' % (i+2))
    
    for key in req_keys:
        if key not in data_kws:
            raise ValueError("Argument '%s' is missing." % key)

    # storing report parameters
    pars_kws = {'fit_type':fit_type, 'fit_window':fit_window, 'sum_one':sum_one}
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
    session = larch.Interpreter(with_plugins=False)
    datgroup = Group()   # container for spectra to perform LCF analysis    
    for i in range(nspectra):
        # reading spectra based on required keys
        dname = 'spectrum' if i==0 else 'ref'+str(i)
        data  = Group(**read_hdf5(data_kws[req_keys[2*i]], name=data_kws[req_keys[2*i+1]]))
        scantype = get_scan_type(data)
        
        # processing xanes spectra
        if pre_edge_kws is None:
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session)
        else:
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session, **pre_edge_kws)
        
        if fit_type == 'exafs':
            # prceossing exafs spectra
            if autobk_kws is None:
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session)
            else:
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session, **autobk_kws)
        
        if i == 0:
            # first value is the spectrum, so we extract the 
            # interpolation values for the corresponding x-variable
            # inside the fit window
            index = where((getattr(data, xvar) >= fit_window[0]) &
                          (getattr(data, xvar) <= fit_window[1]))
            xvals = getattr(data,xvar)[index]
            
            # storing the y-variable
            if fit_type == 'exafs':
                yvals = xvals**k_mult*data.chi[index]
            elif fit_type == 'xanes':
                yvals = data.norm[index]
            else:
                yvals = gradient(data.norm[index])

        else:
            # spline interpolation of references
            if fit_type == 'exafs':
                s = interp1d(getattr(data, xvar), getattr(data, xvar)**k_mult*data.chi, kind='cubic')
            elif fit_type =='xanes':
                s = interp1d(getattr(data, xvar), data.norm, kind='cubic')
            else:
                s = interp1d(getattr(data, xvar), gradient(data.norm), kind='cubic')
            yvals = s(xvals)
        
        # setting corresponding y-variable as an attribute of datgroup
        setattr(datgroup, dname, yvals)

    # setting x-variable as an attribute of datgroup
    setattr(datgroup, xvar, xvals)
    
    # setting parameters for fit model
    initval = around(1/(nspectra-1), decimals=1)
    params  = Parameters()
    expr    = str(1)
    
    for i in range(0, nspectra-1):
        parname = 'amp'+str(i+1)
        if (i == nspectra-2) and (sum_one == True):
            params.add(parname, expr=expr)
        else:
            params.add(parname, value=initval, min=0, max=1, vary=True)
            expr += ' - amp'+str(i+1)
    
    # setting uncertainty
    datgroup.eps  = 1.0

    # perform fit
    out = minimize(residuals, params, args=(datgroup,),)
    
    # storing data and arguments
    fit = sum_references(out.params, datgroup)
    datgroup.fit = fit
    
    out.data_group = datgroup
    out.data_kws   = data_kws
    out.pars_kws   = pars_kws

    # assigning save methods to out object
    out.lcf_report      = types.MethodType(lcf_report, out)
    out.save_lcf_report = types.MethodType(save_lcf_report, out)
    out.save_lcf_data   = types.MethodType(save_lcf_data, out)

    return (out)

def sum_references(pars, data):
    '''
    This function returns the linear sum of references based on 
    the amplitude values stored in a dictionary with LCF parameters.
    '''
    from numpy import sum as npsum
    return (npsum([pars['amp'+str(i)]* getattr(data, 'ref'+str(i)) 
                   for i in range(1,len(pars)+1)], axis=0))


def residuals(pars,data):
    """
    This function returns the residuals of the substraction
    of a spectrum from its LCF with known references
    standards.
    """
    from .fit import sum_references
    return ((data.spectrum - sum_references(pars, data))/data.eps)
