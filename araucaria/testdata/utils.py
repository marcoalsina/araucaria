#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The following utility functions are available to access
the datasets:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`get_testfiles`
     - Returns the available test files.
   * - :func:`get_testpath`
     - Returns path to the requested test file.
"""
from pathlib import Path
from copy import copy
import importlib.resources as pkg_resources
from araucaria import testdata

def get_testfiles() -> list:
    """Returns the available test files.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    :
        List with available test files.
    
    Example
    -------
    >>> from araucaria.testdata import get_testfiles
    >>> testfiles = get_testfiles()
    >>> for file in testfiles:
    ...    if 'dnd' in file:
    ...        print(file)
    dnd_glitchfile.dat
    dnd_testfile1.dat
    dnd_testfile2.dat
    dnd_testfile3.dat
    """
    olist = list(pkg_resources.contents(testdata))
    clist = copy(olist)

    for item in olist:
        if ('__' in item) or ('utils' in item):
            clist.remove(item)
    clist.sort()

    return (clist)

def get_testpath(filename:str) -> Path:
    """Returns the path to the requested test file.
    
    Parameters
    ----------
    filename
        Name of the requested test file.
    
    Returns
    -------
    :
        Path to the requested test file.
    
    Example
    -------
    >>> from pathlib import Path
    >>> from araucaria.testdata import get_testpath
    >>> path = get_testpath('dnd_testfile1.dat')
    >>> isinstance(path, Path)
    True
    """
    with pkg_resources.path(testdata, filename) as path:
        fpath = path
    return (fpath)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
