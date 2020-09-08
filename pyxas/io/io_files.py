#!/usr/bin/python
# -*- coding: utf-8 -*-
__DOC__="""
Functions to read files from different beamlines.

function        description
--------        -----------
read_p65        Reads a file from P65 beamline (PETRA III)
read_dnd        Reads a file fom DND-CAT beamline (APS).
read_xmu        Reads a file based on given columns.

read_file       Base function to read a file based on given columns.
read_rawfile    Base function to read a file based on given raw columns.
"""

def read_p65(fpath, scan='mu', tol=1e-4):
    """Reads a XAFS file from the P65 beamline.
 
    The P65 beamline is located in the PETRA III storage ring (DESY, Zurich, Germany).
    
    Parameters
    ----------
    fpath : str
        Path to the file.
    scan : str
        Requested measurement, either transmission ('mu'), fluorescence ('fluo'),
        or transmission reference ('mu_ref').
    tol : float
        Tolerance in energy units to remove duplicate values.

    Returns
    -------
    data : ndarray group
        Group containing the following arrays: {'energy', 'mu'/'fluo', 'mu_ref'}
    
    Notes
    -----
        The transmission reference measurement is always returned when ``scan`` is either 'mu' or 'fluo'.
        The option 'mu_ref' will return only the transmission reference measurement.
    
    See also
    --------
    :func:`~.main.index_dups`
    :func:`read_rawfile`
    """
    import warnings
    from .io_files import read_rawfile
    
    scandict = ['fluo', 'mu', 'mu_ref']
    chdict   = {'i0': 10, 'it1': 11, 'it2':12, 'if':13}
    
    # testing that the scan string exits in the current dictionary 
    if scan not in scandict:
        warnings.warn("scan type %s not recognized. Extracting transmission spectrum ('mu')." %scan)
        scan = 'mu'

    usecols  = (0, chdict['i0'], chdict['it1'], chdict['it2'], chdict['if'])
    data     = read_rawfile(fpath, usecols, scan, tol)
    return (data)



def read_dnd(fpath, scan='mu', tol=1e-4):
    """Reads a XAFS file from the DND-CAT beamline.
 
    DND-CAT corresponds to sector 5BM-D of the Advanced Photon Source (APS, Argonne, USA).
    
    Parameters
    ----------
    fpath : str
        Path to the file.
    scan : str
        Requested measurement, either transmission ('mu'), fluorescence ('fluo'),
        or transmission reference ('mu_ref').
    tol : float
        Tolerance in energy units to remove duplicate values.

    Returns
    -------
    data : ndarray group
        Group containing the following arrays: {'energy', 'mu'/'fluo', 'mu_ref'}
    
    Notes
    -----
    The transmission reference measurement is always returned when ``scan`` is either 'mu' or 'fluo'.
    The option 'mu_ref' will return only the transmission reference measurement.
        
    See also
    --------
    :func:`~.main.index_dups`
    :func:`read_file`
    """
    import warnings
    from .io_files import read_file
    
    scandict = {'fluo':16, 'mu':17, 'mu_ref':18}
    # testing that the scan string exits in the current dictionary 
    if scan not in scandict:
        warnings.warn("scan type %s not recognized. Extracting transmission spectrum ('mu')." %scan)
        scan = 'mu'

    usecols  = (0, scandict[scan], scandict['mu_ref'])
    data     = read_file(fpath, usecols, scan, tol)
    return (data)

def read_xmu(fpath, scan='mu', tol=1e-4):
    """Reads a generic XAFS file in plain format.
 
    Parameters
    ----------
    fpath : str
        Path to the file.
    scan : str
        Requested measurement, either transmission ('mu'), fluorescence ('fluo'),
        or transmission reference ('mu_ref').
    tol : float
        Tolerance in energy units to remove duplicate values.

    Returns
    -------
    data : ndarray group
        Group containing the following arrays: {'energy', 'mu'/'fluo', 'mu_ref'}
    
    Notes
    -----
    The transmission reference measurement is always returned when ``scan`` is either 'mu' or 'fluo'.
    The option 'mu_ref' will return only the transmission reference measurement.
    
    :func:`read_xmu` assumes the following column order in the file:
    
    1. energy
    2. transmission/fluorescence measurement
    3. transmission reference measurement
    
    See also
    --------
    :func:`~.main.index_dups`
    :func:`read_file`
    
    """
    import warnings
    from .io_files import read_file


    scandict = {'fluo':1, 'mu':1, 'mu_ref':2}
    # testing that the scan string exits in the current dictionary 
    if scan not in scandict:
        warnings.warn("scan type %s not recognized. Extracting transmission spectrum ('mu')." %scan)
        scan = 'mu'

    usecols  = (0, scandict[scan], scandict['mu_ref'])
    data = read_file(fpath, usecols, scan, tol)
    return (data)

def read_file(fpath, usecols, scan, tol):
    """Utility function to read a XAFS file in plain format.
        
    Parameters
    ----------
    fpath : str
        Path to the file.
    usecols : tuple
        Tuple with column indexes to extract from the file.
    scan : str
        Assigned measurement, either transmission ('mu'), fluorescence ('fluo'),
        or transmission reference ('mu_ref').
    tol : float
        Tolerance in energy units to remove duplicate values.

    Returns
    -------
    data : ndarray group
        Group containing the following arrays: {'energy', 'mu'/'fluo', 'mu_ref'}.
    
    Raises
    ------
    IOError
        If the file does not exist in the specified path.
    
    Notes
    -----
    The transmission reference measurement is always returned when ``scan`` is either 'mu' or 'fluo'.
    The option 'mu_ref' will return only the transmission reference measurement.
    
    ``usecols`` should be a tuple with column indexes in the following order:
    
    1. energy
    2. transmission/fluorescence measurement (mu/fluo)
    3. reference meausrement (mu_ref)
    
    Warning
    -------
    The indexing order of ``usecols`` must be respected, or
    the assigned measurement will likely be incorrect.

    """
    from os import path
    from numpy import loadtxt, delete
    from pyxas import Group, index_dups

    # Testing that the file exits in the current directory 
    if not path.isfile(fpath):
        raise IOError('file %s does not exists.' % fpath)
        
    raw    = loadtxt(fpath, usecols=usecols)

    # deleting duplicate energy points
    index  = index_dups(raw[:,0],tol)
    raw    = delete(raw,index,0)

    if scan == 'mu_ref':
        # print "Extracting only reference data ('mu_ref')."
        data = Group(**{'energy':raw[:,0], 'mu_ref':raw[:,2]})
    else:
        # Extracting the requested data
        data = Group(**{'energy':raw[:,0], scan:raw[:,1], 'mu_ref':raw[:,2]})

    return (data)
    
def read_rawfile(fpath, usecols, scan, tol):
    """Utility function to read a raw XAFS file.
    
    Parameters
    ----------
    fpath : str
        Path to the file.
    usecols : tuple
        Tuple with columns indexes to extract from the file.
    scan : str
        Computed measurement, either transmission ('mu'), fluorescence ('fluo'),
        or transmission reference ('mu_ref').
    tol : float
        Tolerance value to remove duplicate energy values.

    Returns
    -------
    data : ndarray group
        Group containing the following arrays: {'energy', 'mu'/'fluo', 'mu_ref'}.
        
    Raises
    ------
    IOError
        If the file does not exist in the specified path.
        
    Notes
    -----
    The transmission reference measurement is always returned when ``scan`` is either 'mu' or 'fluo'.
    The option 'mu_ref' will return only the transmission reference measurement.
    
    ``usecols`` should be a tuple with column indexes in the following order:
    
    1. energy
    2. monochromator intensity (I0)
    3. transmitted intensity (IT1)
    4. transmitted intensity (IT2)
    5. fluorescence intensity (IF)
    
    Warning
    -------
    The indexing order of ``usecols`` must be respected, or 
    the computed measurement will likely be incorrect.

    """
    from os import path
    from numpy import loadtxt, delete, log
    from pyxas import Group, index_dups

    # Testing that the file exits in the current directory 
    if not path.isfile(fpath):
        raise IOError('file %s does not exists.' % fpath)
        
    raw    = loadtxt(fpath, usecols=usecols)

    # deleting duplicate energy points
    index  = index_dups(raw[:,0],tol)
    raw    = delete(raw,index,0)
    
    # convention cols {'energy', 'i0', 'it1', 'it2', 'if'}
    mu_ref = -log(raw[:,3]/raw[:,2])
    
    if scan == 'mu_ref':
        # print "Computing only reference measurement ('mu_ref')."
        data = Group(**{'energy':raw[:,0], 'mu_ref':mu_ref})
    elif scan == 'mu':
        # transmission measurement
        mu = -log(raw[:,2]/raw[:,1])
        data = Group(**{'energy':raw[:,0], scan:mu, 'mu_ref':mu_ref})
    else:
        # fluorescence measurement
        fluo = raw[:,4]/raw[:,1]
        data = Group(**{'energy':raw[:,0], scan:fluo, 'mu_ref':mu_ref})
        
    return (data)