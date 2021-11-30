#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.io.io_read` submodule offers functions to perform the following tasks:

1. Read XAFS files in common formats;
2. Read data from report files of Linear Combination Fitting (LCF) analysis.

Read XAFS files
***************

The following functions are currently implemented to read XAFS files:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`read_p65`
     - Reads a XAFS file from the P65 beamline (PETRA III).
   * - :func:`read_dnd`
     - Reads a XAFS file fom the DND-CAT beamline (APS).
   * - :func:`read_xmu`
     - Reads a XAFS file based on specified columns.
   * - :func:`read_file`
     - Utility function to read a XAFS file based on specified columns.
   * - :func:`read_rawfile`
     - Utility function to read a XAFS file based on specified count columns.

By convention these read functions will return a ``group`` class with the following attributes:

- ``group.energy``: the energy array.
- ``group.mu``: the transmission mu(E) array. Returned if ``scan='mu'``.
- ``group.fluo`` : the fluorescence mu(E) array. Returned if ``scan='fluo'``.
- ``group.mu_ref``: the transmission reference array. Returned if ``ref=True``.

The attribute ``mu_ref`` is also returned by default when ``scan`` is either 'mu' or 'fluo'.
 
Choose ``scan = None`` and ``ref=True`` to return only the transmission reference.
        
Tip
---
The ``group`` returned by any read method will contain either a ``mu`` 
or a ``fluo`` ìnstance, but not both. If both instances are required, create an 
additional ``group`` by calling the read method with a different ``scan`` value.


Read LCF files
**************
The following functions can be used to extract batch information from 
linear combination fit (LCF) report files:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`read_lcf_coefs`
     - Returns amplitude coefficients for a given reference.
   * - :func:`read_lcf_chisqr`
     - Returns chi-square statistics.

Important
---------
The previous functions expect valid LCF report files
generated by :func:`~araucaria.io.io_write.write_lcf_report`.
"""
from os.path import isfile, basename
from http.client import HTTPResponse
import warnings
from typing import List, Tuple, Union
from pathlib import Path
from numpy import loadtxt, delete, log, append, float, ndarray
from .. import Group
from ..utils import index_dups

def read_p65(fpath: Path, scan: str='mu', ref: bool=True, tol: float=1e-4) -> Group:
    """Reads a XAFS file from the P65 beamline.
 
    P65 is located in the PETRA III storage ring (DESY, Hamburg, Germany).
    
    Parameters
    ----------
    fpath
        Path to file.
    scan
        Requested mu(E). Accepted values are transmission ('mu'), fluorescence ('fluo'),
        or None. The default is 'mu'.
    ref
        Indicates if the transmission reference ('mu_ref') should also be returned.
        The default is True.
    tol
        Tolerance in energy units to remove duplicate values. The default is 1e-4,

    Returns
    -------
    :
        Group containing the requested arrays.

    See also
    --------
    read_rawfile: Reads a XAFS file based on specified count columns.
    
    Examples
    --------
    >>> from araucaria import Group
    >>> from araucaria.io import read_p65
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.utils import check_objattrs
    >>> fpath = get_testpath('p65_testfile.xdi')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_p65(fpath, scan='mu')
    >>> check_objattrs(group_mu, Group, attrlist=['mu', 'mu_ref'])
    [True, True]
    
    >>> # extracting only fluo scan
    >>> group_fluo = read_p65(fpath, scan='fluo', ref=False)
    >>> check_objattrs(group_fluo, Group, attrlist=['fluo'])
    [True]
    
    >>> # extracting only mu_ref scan
    >>> group_ref = read_p65(fpath, scan=None, ref=True)
    >>> check_objattrs(group_ref, Group, attrlist=['mu_ref'])
    [True]
    """
    # default modes and channels
    scandict = ['mu', 'fluo', None]
    chdict   = {'i0': 10, 'it1': 11, 'it2':12, 'if':13}
    
    # testing that scan exists in the current dictionary 
    if scan not in scandict:
        warnings.warn("scan mode %s not recognized. Retrieving transmission measurement ('mu')." %scan)
        scan = 'mu'

    if scan is None:
        usecols = (0, chdict['it1'], chdict['it2'])
    elif scan == 'mu':
        usecols = (0, chdict['i0'], chdict['it1'], chdict['it2'])
    else:
        usecols = (0, chdict['i0'], chdict['it1'], chdict['if'], chdict['it2'])

    group     = read_rawfile(fpath, usecols, scan, ref, tol)
    return (group)

def read_dnd(fpath: Path, scan: str='mu', ref: bool=True, tol: float=1e-4) -> Group:
    """Reads a XAFS file from the DND-CAT beamline (5-BMD).
 
    DND-CAT is located in the Advanced Photon Source (APS, Argonne, USA).
    
    Parameters
    ----------
    fpath
        Path to file.
    scan
        Requested mu(E). Accepted values are transmission ('mu'), fluorescence ('fluo'),
        or None. The default is 'mu'.
    ref
        Indicates if the transmission reference ('mu_ref') should also be returned.
        The default is True.
    tol
        Tolerance in energy units to remove duplicate values.

    Returns
    -------
    :
        Group containing the requested arrays.
        
    See also
    --------
    read_file: Reads a XAFS file based on specified columns.
    
    Examples
    --------
    >>> from araucaria import Group
    >>> from araucaria.io import read_dnd
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.utils import check_objattrs
    >>> fpath = get_testpath('dnd_testfile1.dat')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_dnd(fpath, scan='mu')
    >>> check_objattrs(group_mu, Group, attrlist=['mu', 'mu_ref'])
    [True, True]
    
    >>> # extracting only fluo scan
    >>> group_fluo = read_dnd(fpath, scan='fluo', ref=False)
    >>> check_objattrs(group_fluo, Group, attrlist=['fluo'])
    [True]
    
    >>> # extracting only mu_ref scan
    >>> group_ref = read_dnd(fpath, scan=None, ref=True)
    >>> check_objattrs(group_ref, Group, attrlist=['mu_ref'])
    [True]
    """
    # default modes and channels    
    scandict = ['mu', 'fluo', None]
    coldict  = {'fluo':16, 'mu':17, 'mu_ref':18}
    
    # testing that scan exits in the current dictionary 
    if scan not in scandict:
        warnings.warn("scan mode %s not recognized. Retrieving transmission measurement ('mu')." %scan)
        scan = 'mu'

    if scan is None:
        usecols = (0, coldict['mu_ref'])
    else:
        usecols = (0, coldict[scan], coldict['mu_ref'])

    group = read_file(fpath, usecols, scan, ref, tol)
    return (group)

def read_xmu(fpath: Path, scan: str='mu', ref: bool=True, tol: float=1e-4) -> Group:
    """Reads a generic XAFS file in plain format.
 
    Parameters
    ----------
    fpath
        Path to file.
    scan
        Requested mu(E). Accepted values are transmission ('mu'), fluorescence ('fluo'),
        or None. The default is 'mu'.
    ref
        Indicates if the transmission reference ('mu_ref') should also be returned.
        The default is True.
    tol
        Tolerance in energy units to remove duplicate values.

    Returns
    -------
    :
        Group containing the requested arrays.
    
    Notes
    -----
    :func:`read_xmu` assumes the following column order in the file:
    
    1. energy.
    2. transmission/fluorescence mu(E).
    3. transmission reference.
    
    See also
    --------
    read_file : Reads a XAFS file based on specified columns.
    
    Examples
    --------
    >>> from araucaria import Group
    >>> from araucaria.io import read_xmu
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.utils import check_objattrs
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> check_objattrs(group_mu, Group, attrlist=['mu', 'mu_ref'])
    [True, True]
    
    >>> # extracting only fluo scan
    >>> group_fluo = read_xmu(fpath, scan='fluo', ref=False)
    >>> check_objattrs(group_fluo, Group, attrlist=['fluo'])
    [True]
    
    >>> # extracting only mu_ref scan
    >>> group_ref = read_xmu(fpath, scan=None, ref=True)
    >>> check_objattrs(group_ref, Group, attrlist=['mu_ref'])
    [True]
    """
    # default modes and channels
    scandict = ['mu', 'fluo', None]
    coldict = {'fluo':1, 'mu':1, 'mu_ref':2}
    
    # testing that scan exists in the current dictionary 
    if scan not in scandict:
        warnings.warn("scan mode %s not recognized. Retrieving transmission measurement ('mu')." %scan)
        scan = 'mu'

    if scan is None:
        usecols = (0, coldict['mu_ref'])
    else:
        usecols = (0, coldict[scan], coldict['mu_ref'])

    group = read_file(fpath, usecols, scan, ref, tol)
    return (group)

def read_file(fpath: Union[Path, HTTPResponse], usecols: tuple, 
              scan: str, ref: bool, tol: float) -> Group:
    """Utility function to read a XAFS file based on specified columns.
        
    Parameters
    ----------
    fpath
        Path to file, or output from url open request.
    usecols
        Tuple with column indexes to extract from the file.
    scan
        Assigned mu(E), either transmission ('mu'), fluorescence ('fluo'),
        or None.
    ref
        Indicates if the transmission reference ('mu_ref') should also be returned.
    tol
        Tolerance in energy units to remove duplicate values.

    Returns
    -------
    :
        Group containing the requested arrays.
    
    Raises
    ------
    IOError
        If the file does not exist in the specified path.
    ValueError
        If no mu(E) or transmission reference are requested.
    TypeError
        If ``ref`` is not a valid boolean.
    
    Notes
    -----
    ``usecols`` should provide column indexes in the following order:
    
    1. energy.
    2. transmission/fluorescence mu(E).
    3. transmission reference, if ``mu_ref=True``.
    
    If only ``mu_ref`` scan is requested , ``usecols`` should provide
    column indexes in the following order:
    
    1. energy.
    2. transmission reference.
    
    Warning
    -------
    The indexing order of ``usecols`` must be respected, 
    or the assigned mu(E) will be incorrect.

    """
    # testing if fpath is http response
    if type(fpath) is HTTPResponse:
        pass
    # testing if file exists
    elif not isfile(fpath):
        raise IOError('file %s does not exists.' % fpath)
    
    # testing if mu_ref is valid boolean
    if not isinstance(ref, (int, float)):
        raise TypeError('ref: %s is not a valid boolean.' % ref)
    
    # Testing if no scan was requested
    if scan is None and ref is False:
        raise ValueError('no scan requested from file.' )
        
    raw    = loadtxt(fpath, usecols=usecols)

    # deleting duplicate energy points
    index  = index_dups(raw[:,0],tol)
    raw    = delete(raw,index,0)

    if ref is True:
        # returning the requested scan and the reference
        if scan in ['mu','fluo']:
            # transmission or fluorescence
            group = Group(**{'energy':raw[:,0], scan:raw[:,1], 'mu_ref':raw[:,2]})
        else:
            group = Group(**{'energy':raw[:,0], 'mu_ref':raw[:,1]})
    else:
        # returning only the requested scan
        group = Group(**{'energy':raw[:,0], scan:raw[:,1]})

    # saving filename in group
    if type(fpath) is HTTPResponse:
        fpath = fpath.url
    group.name = basename(fpath)
    return (group)
    
def read_rawfile(fpath: Union[Path, HTTPResponse], usecols: tuple, 
                 scan: str, ref: bool, tol: float) -> Group:
    """Utility function to read a XAFS file based on specified count columns.
    
    Parameters
    ----------
    fpath
        Path to file, or output from url open request.
    usecols
        Tuple with columns indexes to extract from the file.
    scan
        Computed mu(E), either transmission ('mu'), fluorescence ('fluo'),
        or None.
    ref
        Indicates if the transmission reference ('mu_ref') should also be returned.
    tol
        Tolerance value to remove duplicate energy values.

    Returns
    -------
    :
        Group containing the requested arrays.
    
    Raises
    ------
    IOError
        If the file does not exist in the specified path.
    ValueError
        If no mu(E) or transmission reference are requested.
    TypeError
        If ``ref`` is not a valid boolean.

    Notes
    -----
    ``usecols`` should provide column indexes in the following order:

    1. energy.
    2. monochromator intensity (I0).
    3. transmitted intensity (IT1).
    4. fluorescence intensity(IF), if ``scan='fluo'``.
    5. transmitted intensity (IT2), if ``mu_ref=True``.

    If ``mu_ref`` scan is not requested, ``usecols`` should provide
    column indexes in the following order:

    1. energy
    2. monochromator intensity (I0).
    3. transmitted intensity (IT1)/fluorescence intensity(IF).

    If only ``mu_ref`` scan is requested , ``usecols`` should provide
    column indexes in the following order:
    
    1. energy.
    2. transmitted intensity (IT1).
    3. transmitted intensity (IT2).
    
    Warning
    -------
    The indexing order of ``usecols`` must be respected, 
    or the computed mu(E) will be incorrect.

    Important
    ---------
    If ``scan='fluo'`` and ``mu_ref`` is requested, all column
    indexes must be provided.
    """
    # testing if fpath is http response
    if type(fpath) is HTTPResponse:
        pass
    # testing if file exists
    elif not isfile(fpath):
        raise IOError('file %s does not exists.' % fpath)
        
    # testing if mu_ref is valid boolean
    if not isinstance(ref, (int, float)):
        raise TypeError('ref: %s is not a valid boolean.' % ref)
    
    # Testing if no scan was requested
    if scan is None and ref is False:
        raise ValueError('no scan requested from file.' )

    raw    = loadtxt(fpath, usecols=usecols)

    # deleting duplicate energy points
    index  = index_dups(raw[:,0],tol)
    raw    = delete(raw,index,0)    

    if ref is True:
        # returning the requested scan and the reference
        if scan == 'mu':
            # transmission
            mu     = -log(raw[:,2]/raw[:,1])
            mu_ref = -log(raw[:,3]/raw[:,2])
            group  = Group(**{'energy':raw[:,0], scan:mu, 'mu_ref':mu_ref})
        elif scan == 'fluo':
            # fluorescence
            fluo   = raw[:,3]/raw[:,1]
            mu_ref = -log(raw[:,4]/raw[:,2])
            group  = Group(**{'energy':raw[:,0], scan:fluo, 'mu_ref':mu_ref})
        else:
            # no scan requested
            mu_ref = -log(raw[:,2]/raw[:,1])
            group = Group(**{'energy':raw[:,0], 'mu_ref':mu_ref})

    else:
        # returning only the requested scan
        if scan == 'mu':
            # transmission
            mu     = -log(raw[:,2]/raw[:,1])
            group  = Group(**{'energy':raw[:,0], scan:mu})
        elif scan == 'fluo':
            # fluorescence
            fluo   = raw[:,2]/raw[:,1]
            group  = Group(**{'energy':raw[:,0], scan:fluo})

    # saving filename in group
    if type(fpath) is HTTPResponse:
        fpath = fpath.url
    group.name = basename(fpath)
    return (group)

def read_lcf_coefs(fpaths: List[Path], refgroup: str, 
                   error: bool=True) -> Union[Tuple[List], list]:
    """Returns amplitude coefficients for a given LCF reference.
    
    Amplitude coefficients are read directly from a list of paths 
    to LCF report files generated by :func:`~araucaria.io.io_write.write_lcf_report`.

    Parameters
    ----------
    fpaths
        List of paths to valid LCF report files.
    refgroup
        Name of the reference group.
    error
        If True the error of the fit will also be returned.
        The default is True.
    
    Returns
    -------
    :
        Amplitude coefficients and error for the reference in the LCF.
    
    Raises
    ------
    IOError
        If a file does not exist in the specified path.
    TypeError
        If a file is not a valid LCF report.
    ValueError
        If ``refgroup`` was fitted during the LCF analysis (i.e. not a reference).

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_lcf_coefs
    >>> fpath = get_testpath('test_lcf_report.log')
    >>> read_lcf_coefs([fpath], 'group1')
    ([0.40034377], [0.01195335])
    >>> read_lcf_coefs([fpath], 'group2', error=False)
    [0.59428689]
    """
   # testing that the file exists
    for fpath in fpaths:
        if not isfile(fpath):
            raise IOError('file %s does not exists.' % fpath)
    
    vallist = []    # container for values
    errlist = []    # container for errors
    for fpath in fpaths:
        getref = True   # reference is always searched
        getval = False  # value is retrieved only if reference was used during the lcf
        
        f = open(fpath, 'r')
        fline = f.readline()
        if 'lcf report' not in fline:
            raise TypeError('%s is not a valid LCF report file.' % fpath)
        
        while getref:
            line = f.readline()
            if refgroup in line:
                # reference found in line
                if 'scan' in line:
                    raise ValueError('%s was fitted in %s.' %(refgroup, fpath))
                else:
                    # we extract the standard index
                    index = line.split()[0][-1]
                    stdval = "amp"+index
                    getref = False
                    getval = True

            elif "[[Fit Statistics]]" in line:
                # This line indicates that we already passed the [[Group]] section
                # There is nothing else to search so return zeroes instead
                vallist = append(vallist,0.00)
                errlist = append(errlist,0.00)
                getref = False
                break

        while getval:
            line = f.readline()
            if stdval in line:
                val = float(line.split()[1])
                err = float(line.split()[3])
                vallist.append(val)
                errlist.append(err)
                getval = False
        f.close()
    if error:
        return (vallist, errlist)
    else:
        return (vallist)

def read_lcf_chisqr(fpaths: List[Path], redchi: bool=False) -> list:
    """Returns chi square statistic for LCF reports.
    
    Chi square values are read directly from a list of paths 
    to LCF report files generated by :func:`~araucaria.io.io_write.write_lcf_report`.
    
    Parameters
    ----------
    fpaths
        List of paths to valid LCF report files.
    
    redchi
        Indicates if the reduced chi square statistic should be returned
        instead.
    
    Returns
    -------
    :
        Chi square values. Reduced chi square values are optionally
        returned if ``redchi=True``.
    
    Raises
    ------
    IOError
        If a file does not exist in the specified path.
    TypeError
        If a file is not a valid LCF report.

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_lcf_chisqr
    >>> fpath = get_testpath('test_lcf_report.log')
    >>> read_lcf_chisqr([fpath])
    [1.40551323]
    >>> read_lcf_chisqr([fpath], redchi=True)
    [0.01011161]
    """
    # testing that the file exists
    for fpath in fpaths:
        if not isfile(fpath):
            raise IOError('file %s does not exists.' % fpath)   

    if redchi:
        reference = " reduced chi-square"
    else:
        reference = "    chi-square"

    vallist = []    # container for values
    for fpath in fpaths:
        getval = True
        f = open(fpath, 'r')
        fline = f.readline()
        if 'lcf report' not in fline:
            raise TypeError('%s is not a valid LCF report file.' % fpath)
        
        while getval:
            line = f.readline()
            if reference in line:
                val = float(line.split("=")[1])
                vallist.append(val)
                getval = False
        f.close()

    return (vallist)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
