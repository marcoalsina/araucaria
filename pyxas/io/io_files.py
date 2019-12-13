#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Basic functions to read/write files.
"""

def read_dnd(fpath, scan='mu', tol=1e-4):
    """Reads a spectrum file from DND-CAT.
 
    This function returns a Larch Group with 
    a XAFS spectrum from sector 5BM-D of the
    Advanced Photon Source (APS).

    Parameters
    ----------
    fpath : str
        Filepath to the spectrum to read.
    scan : {'fluo', 'mu', 'mu_ref'}
        Channel to retrieve from the file.
        Default value `mu` (transmission).
    tol : float
        Tolerance value to remove duplicate
        energy values.

    Returns
    -------
    data : ndarray group
        Larch group containing the following arrays:
        energy, (fluo,mu), mu_ref.

   Notes
   ----- 
   `read_dnd` assumes that the file columns contain the following:
    column 0 : energy (eV)
    column 16: IF/IO (corrected for deadtime) <-- 'fluo'
    column 17: -log(IT/IO) (corrected for background) <-- 'mu'
    column 18: -log(IT2/IT) (corrected for background) <-- 'mu_ref'
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
    """Reads a generic spectrum file.
 
    This function returns a Larch Group with 
    a XAFS spectrum from a generic file.

    Parameters
    ----------
    fpath : str
        Filepath to the spectrum to read.
    scan : {'fluo', 'mu', 'mu_ref'}
        Channel to retrieve from the file.
        Default value `mu` (transmission).
    tol : float
        Tolerance value to remove duplicate
        energy values.

    Returns
    -------
    data : ndarray group
        Larch group containing the following arrays:
        energy, mu, mu_ref.

   Notes
   ----- 
   `read_xmu` assumes that the file columns contain the following:
    column 0 : energy (eV)
    column 1 : IF/IO        <-- 'fluo'
               -log(IT/IO)  <-- 'mu'
    column 2 : -log(IT2/IT) <-- 'mu_ref'

    The choice for either 'mu' or 'fluo' is only descriptive, since
    the algorithm will assume that either channel is stored in
    column 1 of the file.
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
    """Utility function to read a spectrum file.
        
    Parameters
    ----------
    fpath : str
        Filepath to the spectrum to read.
    usecols : tuple
        Tuple with columns to extract from file.
    scan : {'fluo', 'mu', 'mu_ref'}
        Channel to retrieve from the file.
    tol : float
        Tolerance value to remove duplicate
        energy values.

    Returns
    -------
    data : ndarray group
        Larch group containing the following arrays:
        energy, mu, mu_ref.

    """
    from os import path
    from numpy import loadtxt, delete
    from larch import Group
    from pyxas import index_dups

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
