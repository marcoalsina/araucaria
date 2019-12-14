#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Functions to merge XAS spectra.
"""

def calibrate_energy(data, e0, session):
    '''
    This function calibrates the threshold energy
    of the "reference channel" based on the arbitrary 
    value assigned for E0 (enot).

    Parameters
    ----------
    data: Larch group containing the spectra to calibrate.
    e0: arbitrary value for calibration of e0.
    session: valid Larch session.

    Returns
    -------
    e_offset : Appended value to the data group with the magnitude
               of the calibration energy.
    '''
    # calculation of E0 based on Ifeffit standard
    import warnings
    import larch    
    from larch.xafs import find_e0
    
    if hasattr(data, 'e_offset'):
        warnings.warn('data group was already aligned or calibrated! Resetting energy to original value.')
        data.energy = data.energy-data.e_offset
        data.e_offset = 0
    
    data.e_offset = e0-find_e0(data.energy, data.mu_ref, _larch=session)
    
    # currently this script modifies the energy array!
    data.energy = data.energy+data.e_offset

    
def align_scans(objdat, refdat, session, e_offset=0, window=[-50,50]):
    '''
    This function aligns the first derivative of
    the reference channel of a data group against 
    the first derivative of the reference channel
    of a reference group.
    --------------
    Required input:
    objdat: objective data group.
    refdat: reference data group.
    session: valid Larch session.
    e_offset: initial energy offset value for alignment (optional).
    window: array with +- window values for alignment w/r to e0 (optional).
    --------------
    Output:
    e_offset : Appended value to the data group with the magnitude
               of the alignment energy.
    '''
    import warnings
    from numpy import where, gradient, sum
    from scipy.interpolate import interp1d
    from scipy.optimize import fmin
    import larch
    from larch.xafs import find_e0

    if hasattr(objdat, 'e_offset'):
        warnings.warn('Data group was already aligned or calibrated! Resetting energy to original value.')
        objdat.energy = objdat.energy-objdat.e_offset
        objdat.e_offset = 0
    
    # calculation of e0 to determine the optimization window
    e0 = find_e0(refdat.energy, refdat.mu_ref, _larch=session)
    min_lim = e0+window[0]
    max_lim = e0+window[1]
    
    # calculation of energy points for interpolation and comparison
    # this energy array is static
    index = where((refdat.energy >= min_lim) & (refdat.energy <= max_lim))
    ref_energy = refdat.energy[index]
    
    # both the reference mu and the objetive mu
    # exist in the same energy grid as the reference
    ref_spline = interp1d(refdat.energy, refdat.mu_ref, kind='cubic')
    ref_dmu = gradient(ref_spline(ref_energy))/gradient(ref_energy)
    
    
    # the objective function is the difference of derivatives
    def objfunc(x):
        obj_spline = interp1d(objdat.energy + x, objdat.mu_ref, kind='cubic')
        obj_dmu = gradient(obj_spline(ref_energy))/gradient(ref_energy)
        return (sum((ref_dmu - obj_dmu)**2))
    
    objdat.e_offset = fmin(objfunc, e_offset, disp=False)[0]
    
    # currently this script modifies the energy array of objdat.
    objdat.energy = objdat.energy + objdat.e_offset


def merge_scans(group, scantype='mu', kind='cubic'):
    '''
    This function rebins the scans provided in the
    list group, and merges in the xmu space.
    --------------
    Required input:
    group: list of Larch groups to be merged in xmu.
    refdat: reference data group.
    scantype [string]:  eiher 'fluo', 'mu', or 'mu_ref'.
    kind [string]: spline kind. Default is 'cubic'. 
    --------------
    Output:
    merge: Larch group containing the merged xmu scans.
    '''
    import warnings
    from numpy import size, resize, append, mean
    from scipy.interpolate import interp1d
    from larch import Group
    
    scanlist = ['fluo','mu', 'mu_ref']
    # Testing that the scan string exists in the current dictionary 
    if scantype not in scanlist:
        warnings.warn("scan type %s not recognized. Merging transmission data ('mu').")
        scantype = 'mu'
    
    # Energy arrays are first compared to create an interpolation array
    # that is fully contained in the original arrays.
    energy_eval=0
    for i in range(len(group)):
        # Searching the energy array with the largest initial value.
        # Alternative algorithm is to search the array with lowest final value.
        if group[i].energy[0] > energy_eval:
            energy_eval = group[i].energy[0]
            gindex = i
    
    energy = group[gindex].energy
    for gr in group:
        # Determining the last point in the selected energy array that
        # is contained in the other energy vectors
        while energy[-1] > gr.energy[-1]:
            energy = energy[:-1]
    
    mu = []      # container for the interpolated data
    if scantype != 'mu_ref':
        mu_ref = []  # container for the interpolated reference channel
    
    for gr in group:
        # interpolation of the initial data
        # the getattr method is employed to recycle the variable scan
        mu_spline = interp1d(gr.energy, getattr(gr,scantype), kind=kind)
        
        # appending the interpolated data to the container
        mu = append(mu, mu_spline(energy), axis=0)
        
        if scantype != 'mu_ref':
            # interpolating also the reference channel
            mu_ref_spline= interp1d(gr.energy, gr.mu_ref, kind=kind)
            mu_ref = append(mu_ref, mu_ref_spline(energy), axis=0)
            
    # resizing the container
    mu = resize(mu,(len(group), len(energy)))
    # calculating the average of the spectra
    mu_avg = mean(mu, axis=0)
    
    if scantype != 'mu_ref':
        mu_ref = resize(mu_ref,(len(group), len(energy)))
        mu_ref_avg = mean(mu_ref, axis=0)    
        data = Group(**{'energy':energy, scantype:mu_avg, 'mu_ref':mu_ref_avg})
    else:
        data = Group(**{'energy':energy, scantype:mu_avg})
    return (data)

def merge_report(group, merge):
    """Report of merge on XAS spectra.
    
    Parameters
    ----------
    
    Returns
    -------

    """
    import larch
    from larch.xafs import pre_edge
    from pyxas import get_scan_type, DataReport

    # larch parameters
    session = larch.Interpreter(with_plugins=False)

    # initializing report
    report_pars = {'cols': [3,24,10,10,10,10], 
                   'names': ['id', 'filename', 'scantype', 'step', 'e_offset', 'e0 [eV]']}
    report      = DataReport()
    report.set_columns(**report_pars)

    # retrieving values from data group
    for i, data in enumerate(group):
        scantype = get_scan_type(data)
        pre_edge(data.energy, getattr(data, scantype), group=data, _larch=session)
        try:
            data.e_offset
        except:
            data.e_offset = 0.0

        report.add_content([i+1, data.name, scantype, data.edge_step, data.e_offset, data.e0])

    # retrieving values from merge
    report.add_midrule()
    pre_edge(merge.energy, getattr(merge, scantype), group=merge, _larch=session)
    report.add_content(['','merge', scantype, merge.edge_step, 0.0, merge.e0])

    return (report)

def merge_spectra(fpaths, scantype='mu', ftype='dnd', align_kws=None, 
                  print_report=True, write_kws=None):
    """Merge XAS scans.

    This function ...
    
    Parameters
    ----------
    fpaths : list
    scantype : str
    ftype : str
    align_kws : dict
    write_kws : dict
    
    Returns
    -------
    group : list of larch groups
    merge : larch group
    
    """
    from os import path
    import numpy as np
    import larch
    from larch import Group
    import pyxas.io
    from .merge import align_scans, merge_scans, merge_report
    
    # supporting reading formats
    format_dict = {'dnd' : 'read_dnd', 
                   'xmu' : 'read_xmu' }
    
    # required align/write keys
    req_keys = ['name', 'dbpath']
    
    # testing that file type is supported    
    if ftype not in format_dict:
        raise ValueError("file type %s currently not supported")

    # testing that files exist in the given path 
    for fpath in fpaths:
        if not path.isfile(fpath):
            raise IOError('file %s does not exists.' % fpath)
    
    # checking alignment keys from input
    req_keys = ['name', 'dbpath']
    if align_kws is not None:
        for key in req_keys:
            if key not in align_kws:
                raise ValueError ("Either 'name' or 'dbpath' key is missing in the align dictionary.")
            else:
                align = True
        
        # setting initial energy offset for alignment routine
        try:
            e_offset = align_kws['e_offset']
        except:
            e_offset = 0.0    
        
        # reading reference scan
        ref = Group(**pyxas.io.read_hdf5(align_kws['dbpath'], align_kws['name']))
    else:
        align = False

    # loading larch session
    session = larch.Interpreter(with_plugins=False)

    # reading files
    group = []
    for fpath in fpaths:
        # retrieving reading function
        read_func = getattr(pyxas.io, format_dict[ftype])
        data      = read_func(fpath, scantype)
        if align:
            align_scans(data, ref, session, e_offset=e_offset)
        
        data.name = path.split(fpath)[1]
        group = np.append(group, data)

    # merging scans
    if len(fpaths) > 1:
        merge = merge_scans(group, scantype)
    else:
        merge = data

    # saving list of merged scans as attribute
    merge.merged_scans = str([data.name for data in group])

    # print merge report
    if print_report:
        report = merge_report(group, merge)
        report.show()

    # write merge group in a hdf5 database
    # checking write keys from input
    if write_kws is not None:
        for key in req_keys:
            if key not in write_kws:
                raise ValueError ("Either 'name' or 'dbpath' key is missing in the write dictionary.")
        # setting replace option in write_hdf5
        try:
            replace = write_kws['replace']
        except:
            replace = False

        pyxas.io.write_hdf5(write_kws['dbpath'], merge, 
                            name = write_kws['name'], replace=replace)

    return(group, merge)

def merge_ref(fpaths, e0, ftype='dnd', e_offset=0.0, 
              print_report=True, write_kws=None):
    """Merge XAS reference scans.

    This function ...
    
    Parameters
    ----------
    fpaths : list
    e0 : float
    ftype : str
    e_offset : float
    print_report : bool
    write_kws : dict
    
    Returns
    -------
    group : list of larch groups
    merge : larch group
    
    """
    from os import path
    import numpy as np
    import larch
    from larch import Group
    import pyxas.io
    from .merge import calibrate_energy, align_scans, merge_scans, merge_report
    
    # supporting reading formats
    format_dict = {'dnd' : 'read_dnd', 
                   'xmu' : 'read_xmu' }
    
    # required write keys
    req_keys = ['name', 'dbpath']
    
    # testing that file type is supported    
    if ftype not in format_dict:
        raise ValueError("file type %s currently not supported")

    # testing that files exist in the given path 
    for fpath in fpaths:
        if not path.isfile(fpath):
            raise IOError('file %s does not exists.' % fpath)

    # loading larch session
    session = larch.Interpreter(with_plugins=False)
    
    # retrieving reading function
    read_func = getattr(pyxas.io, format_dict[ftype])
    
    # reading reference file
    # the scan with the lowest e_offset is selected as the reference
    for fpath in fpaths:
        data = read_func(fpath,scantype)
        calibrate_energy(data, e0, session)
        
        # the first file is the first iteration
        if fpath == fpaths[0]:
            ref   = data
            e_ref = data.e_offset
        
        # ref file is modified if e_offset is smaller
        elif abs(data.e_offset) < abs(e_ref):
            ref   = data
            e_ref = ref.e_offset

    # reading files
    group = []
    for fpath in fpaths:
        data = read_func(fpath, scantype)
        align_scans(data, ref, session, e_offset=e_offset)
        data.name = path.split(fpath)[1]
        group     = np.append(group, data)

    # merging scans
    if len(fpaths) > 1:
        merge = merge_scans(group, scantype)
    else:
        merge = data

    # saving list of merged scans as attribute
    merge.merged_scans = str([data.name for data in group])

    # print merge report
    if print_report:
        report = merge_report(group, merge)
        report.show()

    # write merge group in a hdf5 database
    # checking write keys from input
    if write_kws is not None:
        for key in req_keys:
            if key not in write_kws:
                raise ValueError ("Either 'name' or 'dbpath' key is missing in the write dictionary.")
        # setting replace option in write_hdf5
        try:
            replace = write_kws['replace']
        except:
            replace = False

        pyxas.io.write_hdf5(write_kws['dbpath'], merge, 
                            name = write_kws['name'], replace=replace)

    return(group, merge)