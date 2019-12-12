#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Basic functions to merge XAS spectra.
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
    kind [string]: spline kind. Defaults to 'cubic'. 
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
    if scantype| not in scanlist:
        warnings.warn("scan type %s not recognized. Merging transmission data ('mu').")
        scantype = 'mu'
    
    # Energy arrays are compared to create the interpolation array
    # that is completely contained in all the previous arrays.
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
    if scan != 'mu_ref':
        mu_ref = []  # container for the interpolated reference channel
    
    for gr in group:
        # interpolation of the initial data
        # the getattr method is employed to recycle the variable scan
        mu_spline = interp1d(gpenergy, getattr(gr,scan), kind=kind)
        
        # appending the interpolated data to the container
        mu = append(mu, mu_spline(energy), axis=0)
        
        if scan != 'mu_ref':
            # interpolating also the reference channel
            mu_ref_spline= interp1d(gpenergy, gr.mu_ref, kind=kind)
            mu_ref = append(mu_ref, mu_ref_spline(energy), axis=0)
            
    # resizing the container
    mu = resize(mu,(len(group), len(energy)))
    # calculating the average of the spectra
    mu_avg = mean(mu, axis=0)
    
    if scan != 'mu_ref':
        mu_ref = resize(mu_ref,(len(group), len(energy)))
        mu_ref_avg = mean(mu_ref, axis=0)    
        data = Group(**{'energy':energy, scan:mu_avg, 'mu_ref':mu_ref_avg})
    else:
        data = Group(**{'energy':energy, scan:mu_avg})
    return (data)