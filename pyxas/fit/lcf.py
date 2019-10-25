#!/usr/bin/env python
'''
filename: lcf.py

Routine to perform LCF on a XAS spectrum
'''

def lcf(data_kws, fit_type, fit_window, k_mult=2,
        sum_one=True, pre_edge_kws=None, autobk_kws=None):
    '''
    Include description of function...
    '''
    import os
    from numpy import where, gradient
    from scipy.interpolate import interp1d
    from lmfit import Parameters, minimize
    import larch
    from larch import Group
    from larch.xafs import pre_edge, autobk
    from pyxas import get_scan_type
    from pyxas.io import read_hdf5
    #from pyxas.fit import residuals, sum_standards, fit_report
    
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
        req_keys.append('ref%i_path' % (i+1))
        req_keys.append('ref%i_name' % (i+1))
    
    for key in req_keys:
        if key not in data_kws:
            raise ValueError("Argument '%s' is missing." % key)
    
    # reading and processing spectra
    session = larch.Interpreter(with_plugins=False)
    datgroup = Group()   # container for spectra to perform LCF analysis
    
    for i in range(nspectra):
        # reading spectra 
        dname = 'spectrum' if i==0 else 'ref'+str(i)
        data = Group(**read_hdf5(data_kws[req_keys[2*i]], name=data_kws[req_keys[2*i+1]]))
        scantype = get_scan_type(data)
        
        # standard report parameters
        pars_kws = {'fit_type':fit_type, 'fit_window':fit_window, 'sum_one':sum_one}
        
        # processing xanes spectra
        if pre_edge_kws is None:
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session)
            pars_kws['pre_edge_kws'] = 'default'
        else:
            pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session, **pre_edge_kws)
            pars_kws['pre_edge_kws'] = pre_edge_kws
        
        if fit_type == 'exafs':
            # prceossing exafs spectra
            if autobk_kws is None:
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session)
                pars_kws['autobk_kws'] = 'default'
            else:
                autobk(data.energy, getattr(data, scantype), group=data, _larch=session, **autobk_kws)
                pars_kws['autobk_kws'] = autobk_kws
            
            pars_kws['k_mult'] = k_mult
            # storing name of x-variable (exafs)
            xvar = 'k'
        else:
            # storing name of x-variable (xanes)
            xvar = 'energy'
        
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
                s = interp1d(getattr(data, xvar), getattr(data, xvar)**k_mult*data.chi)
            elif fit_type =='xanes':
                s = interp1d(getattr(data, xvar), data.norm)
            else:
                s = interp1d(getattr(data, xvar), gradient(data.norm))
            yvals = s(xvals)
        
        # setting corresponding y-variable as an attribute of datgroup
        setattr(datgroup, dname, yvals)

    # setting x-variable as an attribute of datgroup
    setattr(datgroup, xvar, xvals)
    
    # setting parameters for fit model
    params = Parameters()
    expr = str(1)
    
    for i in range(0, nspectra-1):
        parname = 'amp'+str(i+1)
        if (i == nspectra-2) and (sum_one == True):
            params.add(parname, expr=expr)
        else:
            params.add(parname, value=0.5, min=0, max=1, vary=True)
            expr += ' - amp'+str(i+1)
    
    # setting uncertainty
    datgroup.eps  = 1.0

    # perform fit
    out = minimize(residuals, params, args=(datgroup,),)
    
    # storing data and argumens
    fit = sum_standards(out.params, datgroup)
    datgroup.fit = fit
    
    out.data_group = datgroup
    out.data_kws = data_kws
    out.pars_kws = pars_kws
    
    return (out)
