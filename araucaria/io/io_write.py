#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.io.io_write` submodule offers the following functions to write files:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`write_xmu`
     - Writes a XAFS group in xmu format.
   * - :func:`write_lcf`
     - Writes LCF data to a file.
   * - :func:`write_lcf_report`
     - Writes a LCF report to a file.
   * - :func:`write_file`
     - Utility function to write an array to a file.
   * - :func:`set_header`
     - Utility function to write a file header.
"""
from os.path import isfile, basename
from datetime import datetime
from pathlib import Path
from numpy import ndarray, stack, savetxt
from .. import Group, Dataset
from ..fit import lcf_report
from ..utils import check_objattrs

def write_xmu(fpath: Path, group: Group, fmt: str='%12.8g', 
              replace: bool=False) -> str:
    """Writes a file in xmu format.
    
    Parameters
    ----------
    fpath
        Path to file.
    group
        Group dataset to write in file.
    fmt
        Format for numbers. The default is '%12.8g'.
    replace
        Indicates if a previous file should be replaced. The detault is False.
    
    Returns
    -------
    :
        Confirmation message.
    
    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    
    Notes
    -----
    By default the operation will be cancelled if the file already exists. 
    The previous file can be overwritten with the option ``replace=True``.
    
    See also
    --------
    write_file : Writes an array to a file.

    Example
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_xmu
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # saving a new xmu file
    >>> write_xmu('new_file.xmu', group_mu)
    'xmu_testfile.xmu written to new_file.xmu.'
    """
    # checking class
    check_objattrs(group, Group)
    
    # testing that the requested attributes exist
    scan = group.get_mode()
    if group.has_ref() and scan != 'mu_ref':
        # either mu or fluo + ref scan
        data = stack((group.energy, getattr(group,scan), group.mu_ref), axis=1)
        cols = 'energy\t%s\tmu_ref' % scan
    else:
        # single scan available
        data = stack((group.energy, getattr(group,scan)), axis=1)
        cols = 'energy\t%s' % scan
    
    # header
    header = set_header(fpath, 'xmu')
    header = '\n'.join((header, cols))

    msg = write_file(fpath, data, name=group.name, header=header, 
                     fmt=fmt, replace=replace)
    return (msg)

def write_lcf(fpath: Path, out: Dataset, fmt: str='%12.8g', 
              replace: bool=False) -> str:
    """Writes LCF data to a file.
    
    Parameters
    ----------
    fpath
        Path to file.
    out
        Valid Dataset from :func:`~araucaria.fit.lcf.lcf`.
    fmt
        Format for numbers. The default is '%12.8g'.
    replace
        Replace previous file. The detault is False.
    
    Returns
    -------
    :
        Confirmation message.

    Raises
    ------
    TypeError
        If ``out`` is not a valid Dataset instance.
    AttributeError
        If attribute ``min_pars``, ``scangroup``,
        or ``refgroups`` does not exist in ``out``.
    
    Notes
    -----
    The returned file will contain in the header the output from
    :func:`lcf_report`, in addition to the following columns:
    
        - ``energy``   : array with energy values. 
          Returned only if ``fit_region='xanes'`` or ``fit_region='dxanes'``.
        - ``k``        : array with wavenumber values. 
          Returned only if ``fit_region='exafs'``.
        - ``scan``     : array with values of the fitted spectrum.
        - ``ref``      : array with interpolated values for each reference spectrum.
        - ``fit``      : array with fit result.
        - ``residual`` : fit residuals.

    See also
    --------
    :func:`~araucaria.io.io_write.write_file` : Writes an array to a file.
    
    Example
    -------
    >>> from numpy.random import seed, normal
    >>> from numpy import arange, sin, pi
    >>> from araucaria import Group, Collection
    >>> from araucaria.fit import lcf
    >>> from araucaria.io import write_lcf
    >>> seed(1234)  # seed of random values
    >>> k    = arange(0,  12,   0.05)
    >>> eps  = normal(0, 0.1, len(k))
    >>> f1   = 1.2  # freq 1
    >>> f2   = 2.6  # freq 2
    >>> amp1 = 0.4  # amp 1
    >>> amp2 = 0.6  # amp 2
    >>> group1 = Group(**{'name': 'group1', 'k': k, 'chi': sin(2*pi*f1*k)})
    >>> group2 = Group(**{'name': 'group2', 'k': k, 'chi': sin(2*pi*f2*k)})
    >>> group3 = Group(**{'name': 'group3', 'k': k,
    ...                   'chi' : amp1 * group1.chi + amp2 * group2.chi + eps})
    >>> collection = Collection()
    >>> tags = ['ref', 'ref', 'scan']
    >>> for i,group in enumerate((group1,group2, group3)):
    ...     collection.add_group(group, tag=tags[i])
    >>> # performing lcf
    >>> out = lcf(collection, fit_region='exafs', fit_range=[3,10], 
    ...           kweight=0, sum_one=False)
    >>> # saving lcf to a file
    >>> write_lcf('new_fit.lcf', out)
    'lcf data written to new_fit.lcf.'
    """
    # checking class
    check_objattrs(out, Dataset, attrlist=['min_pars', 
    'scangroup', 'refgroups'], exceptions=True)
    
    # header
    header = set_header(fpath, 'lcf',)
    header = '\n'.join((header, lcf_report(out)))
    
    reflist = ['ref'+str(i+1) for i in range(len(out.refgroups))]
    # storing data according to fit_region
    data_header = '\t'.join(('fit', 'scan', *[name for name in reflist], 'residual'))
    if out.lcf_pars['fit_region'] == 'exafs':
        data_header = 'k\t' + data_header
        data = stack((out.k, out.scan, out.fit,
                       *[getattr(out, name) for name in reflist],
                       out.min_pars.residual), axis = 1)
    else:
        data_header = 'energy\t' + data_header
        data = stack((out.energy, out.scan, out.fit,
                      *[getattr(out, name) for name in reflist],
                      out.min_pars.residual), axis = 1)
    
    header = '\n'.join((header, data_header))
    msg = write_file(fpath, data, name='lcf data', fmt=fmt,
                     header=header, replace=replace)
    return (msg)

def write_lcf_report(fpath: Path, out: Dataset, 
                     replace: bool=False) -> str:
    """Writes a LCF report to a file.
    
    Parameters
    ----------
    fpath
        Path to file.
    out
        Valid Dataset from :func:`lcf`.
    replace
        Replace previous file. The detault is False.

    Returns
    -------
    :
        Confirmation message.

    Raises
    ------
    TypeError
        If ``out`` is not a valid Dataset instance.
    AttributeError
        If attribute ``min_pars``, ``lcf_pars``, ``scangroup``,
        or ``refgroups`` does not exist in ``out``.

    Example
    -------
    >>> from numpy.random import seed, normal
    >>> from numpy import arange, sin, pi
    >>> from araucaria import Group, Collection
    >>> from araucaria.fit import lcf
    >>> from araucaria.io import write_lcf_report
    >>> seed(1234)  # seed of random values
    >>> k    = arange(0,  12,   0.05)
    >>> eps  = normal(0, 0.1, len(k))
    >>> f1   = 1.2  # freq 1
    >>> f2   = 2.6  # freq 2
    >>> amp1 = 0.4  # amp 1
    >>> amp2 = 0.6  # amp 2
    >>> group1 = Group(**{'name': 'group1', 'k': k, 'chi': sin(2*pi*f1*k)})
    >>> group2 = Group(**{'name': 'group2', 'k': k, 'chi': sin(2*pi*f2*k)})
    >>> group3 = Group(**{'name': 'group3', 'k': k,
    ...                   'chi' : amp1 * group1.chi + amp2 * group2.chi + eps})
    >>> collection = Collection()
    >>> tags = ['ref', 'ref', 'scan']
    >>> for i,group in enumerate((group1,group2, group3)):
    ...     collection.add_group(group, tag=tags[i])
    >>> # performing lcf
    >>> out = lcf(collection, fit_region='exafs', fit_range=[3,10], 
    ...           kweight=0, sum_one=False)
    >>> # saving lcf report to a file
    >>> write_lcf_report('lcf_report.log', out)
    'lcf report written to lcf_report.log.'
    """
    # checking class
    check_objattrs(out, Dataset, attrlist=['min_pars', 
    'scangroup', 'refgroups'], exceptions=True)
    
    # header
    header = set_header(fpath, 'lcf report',)
    
    if replace:
        fout = open(fpath, 'w')
    else:
        fout = open(fpath, 'x')
    fout.write('\n'.join((header, lcf_report(out))))
    fout.close()
    msg = 'lcf report written to %s.' % fpath
    return (msg)

def write_file(fpath: Path, data: ndarray, name:str, fmt: str='%12.8g',
               header: str="", replace: bool=False) -> None:
    """Utility function to write a data array to a file.
    
    Parameters
    ----------
    fpath
        Path to file.
    data
        array to write in file.
    name
        Name of object to write.
    fmt
        Format for numbers. The default is '%12.8g'.
    header
        Header for the file.
    replace
        Replace previous file. The detault is False.
    
    Returns
    -------
    :
        Confirmation message.
    
    Raises
    ------
    IOError
        If the file already exists and ``replace=False``.
    """
    if isfile(fpath) and replace is False:
        raise IOError('file %s already exists.' % fpath)
    savetxt(fpath, data, fmt=fmt, header=header)
    
    msg = '%s written to %s.' % (name, fpath)
    return (msg)

def set_header(fpath, file_type: str) -> str:
    """Utility function for write a file header.
    
    Parameters
    ----------
    fpath
        Path to file.
    file_type
        Short descriptor for type of file.

    Returns
    -------
    :
        Header for file.

    Example
    -------
    >>> from araucaria.io import set_header
    >>> from araucaria import Group
    >>> header = set_header('testdata.xmu', 'xmu')
    >>> header.splitlines()[0]
    'xmu file created by araucaria'
    >>> header.splitlines()[2]
    'name: testdata.xmu'
    """
    # header info
    type   = '%s file created by araucaria' % file_type
    date   = 'date: %s' % datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    name   = 'name: %s' % basename(fpath)
    header = '\n'.join((type, date, name))
    return (header)

if __name__ == '__main__':
    import os
    import doctest
    doctest.testmod()

    # removing test files    
    for file in ['new_file.xmu', 'new_fit.lcf', 'lcf_report.log']:
        if os.path.exists(file):
            os.remove(file)