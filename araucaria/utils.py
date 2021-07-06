#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.utils` module offers the following utility functions:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`get_version`
     - Returns version of araucaria.
   * - :func:`check_objattrs`
     - Check type and attributes of an object.
   * - :func:`check_xrange`
     - Range values for an array.
   * - :func:`index_xrange`
     - Index of values in given range for an array.
   * - :func:`index_dups`
     - Index of duplicate values in an array.
   * - :func:`index_nans`
     - Index of NaN values in an array.
   * - :func:`index_nearest`
     - Index of nearest value in an array.
   * - :func:`interp_yvals`
     - Returns interpolated values for 1-D function.
   * - :func:`read_fdicts`
     - Reads file with multiple dictionaries.
"""
from typing import List, Union, TypeVar
from pathlib import Path
from re import findall
from ast import literal_eval
from numpy import (ndarray, diff, abs, argwhere, where, 
                   ravel, apply_along_axis, isnan, isinf)
from scipy.interpolate import interp1d


def get_version(dependencies:bool=False) -> str:
    """Returns installed version of araucaria.

    Parameters
    ----------
    dependencies
        Condition to additionally get version of
        dependencies. The default is False.

    Returns
    -------
    :
        Printable string with version of araucaria.

    Examples
    --------
    >>> from araucaria.utils import get_version
    >>> print(get_version()) #doctest: +ELLIPSIS
    Araucaria version   : ...
    """
    import os, platform
    import numpy as np
    import scipy as sp
    import lmfit as lm
    import h5py  as h5
    import matplotlib as mpl
    import araucaria as ara

    libr = ('Python', 'Numpy', 'Scipy', 'Lmfit', 'H5py', 'Matplotlib')
    verf = ''    # string container

    if dependencies:
        for i, lib in enumerate((platform, np, sp, lm, h5, mpl)):
            if lib == platform: 
                ver = lib.python_version()
            else:
                ver = lib.__version__
            verf   += '{0:20}: {1}\n'.format(libr[i]+' version',ver)

    ver   = ara.__version__
    verf += '{0:20}: {1}'.format('Araucaria version', ver)

    return verf


def check_objattrs(obj: object, objtype: TypeVar, attrlist: list=None, 
                   exceptions: bool=False) -> List[bool]:
    """Check type and attributes of an object.
    
    Parameters
    ----------
    obj
        Object to check.
    objtype
        Type for the object.
    attrlist
        List with names of attributes to check.
    exceptions
        Condition to raise exceptions if attributes 
        are not in the object. The default is False.
    
    Returns
    -------
    :
        List with booleans for each attribute of the object.

    Raises
    ------
    TypeError
        If ``obj`` is not an instance of ``objtype``.

    Examples
    --------
    >>> from araucaria import Group
    >>> from araucaria.utils import check_objattrs
    >>> group   = Group(**{'energy': [1,2,3,4], 'mu': [2,2,3,1]})
    >>> # checking class type
    >>> check_objattrs(group, Group)
    True
    
    >>> # checking class type and attributes
    >>> alist   = ['energy', 'mu', 'mu_ref']
    >>> check_objattrs(group, Group, attrlist = alist)
    [True, True, False]
    """
    if not isinstance(obj, objtype):
        raise TypeError('object is not a valid %s instance.' % objtype.__name__)
    elif attrlist is None:
        return True

    boolist = []
    for attr in attrlist:
        if hasattr(obj, attr) is False:
            if exceptions:
                raise AttributeError("%s instance has no '%s' attribute." % (objtype.__name__, attr))
            else:
                boolist.append(False)
        else:
            boolist.append(True)

    return boolist

def check_xrange(x_range: list, x: ndarray, refval: float=None) -> list:
    """Returns range values inside an array.
    
    Parameters
    ----------
    x_range
        List with min and max values.
        Supports :data:`~numpy.inf` values.
    x
        Array with values.
    refval
        If given, x_range is assumed to contain values relative to refval.
        If None, x_range is assumed to contain absolute values.

    Returns
    -------
    :
        New range values inside the array.
    
    Examples
    --------
    >>> from numpy import inf, linspace
    >>> from araucaria.utils import check_xrange
    >>> k_range = [-inf,inf]
    >>> k       = linspace(0,15)
    >>> krange = check_xrange(k_range, k)
    >>> print(krange)
    [0.0, 15.0]

    >>> # using a reference value
    >>> e_range = [-inf, -50]
    >>> energy  = linspace(8900, 9100)
    >>> e0      = 8979
    >>> erange  = check_xrange(e_range, energy, refval=e0)
    >>> print(erange)
    [-79.0, -50]
    """
    x_range = list(x_range)
    x_range.sort()
    
    x_min = x_range[0]
    x_max = x_range[1]
    
    if refval:
        # min and max values are computed with respect
        # to refval
        xmin = min(x) - refval
        xmax = max(x) - refval
    else:
        xmin = min(x)
        xmax = max(x)
    
    if (isinf(x_min) or x_min < xmin):
        x_min = xmin
    if (isinf(x_max) or x_max > xmax):
        x_max = xmax

    return [x_min, x_max]

def index_xrange(x_range: list, x: ndarray, refval: float=None) -> ndarray:
    """Returns indexes of range values inside an array.
    
    Parameters
    ----------
    x_range
        List with real min and max values.
    x
        Array with values.
    refval
        If given, x_range is assumed to contain values relative to refval.
        If None, x_range is assumed to contain absolute values.

    Returns
    -------
    :
        Array with indexes of range values inside the array.
    
    Examples
    --------
    >>> from numpy import arange, inf
    >>> from araucaria.utils import index_xrange
    >>> k_range = [4, 8]
    >>> k       = arange(0,16)
    >>> index   = index_xrange(k_range, k)
    >>> k[index]
    array([4, 5, 6, 7, 8])

    >>> # using a reference value
    >>> e_range = [-inf, -50]
    >>> energy  = arange(8900, 9100)
    >>> e0      = 8979
    >>> index   = index_xrange(e_range, energy, refval=e0)
    >>> energy[index][0], energy[index][-1]
    (8900, 8929)
    """
    x_range = list(x_range)
    x_range.sort()

    x_min = x_range[0]
    x_max = x_range[1]

    if refval:
        # min and max values are computed with respect
        # to refval
        x_min = x_min + refval
        x_max = x_max + refval
    
    index = where((x >= x_min) & (x <= x_max))
    return index

def index_dups(data: ndarray, tol: float=1e-4) -> ndarray:
    """Index of duplicate values.


    Parameters
    ----------
    data
        Array to search for duplicates.
        
    tol
        Tolerance value (the detault is 1e-4).

    Returns
    -------
    :
        Index array with the location of duplicates.
        
    Notes
    -----
    A value in ``data`` is considered a duplicate if the
    difference with respect to the previous value is strictly
    lower than the given ``tol`` value.

    If the dimension of ``data`` is larger than 1 the array will be 
    flattened by indexing the elements in row-major (i.e. C-style).
    
    Examples
    --------    
    :func:`index_dups` is useful to remove duplicates and ensure monotonicity of a 1-D array.
    
    >>> from numpy import array, delete
    >>> from araucaria.utils import index_dups
    >>> energy = array([9000, 9000.1, 9005, 9005.1, 9008])
    >>> index  = index_dups(energy, tol=0.5)
    >>> print(index)
    [1 3]
    
    >>> # duplicactes
    >>> print(energy[index])
    [9000.1 9005.1]
    
    >>> # removing duplicates
    >>> from numpy import delete
    >>> new_energy = delete(energy,index,0)
    >>> print(new_energy)
    [9000. 9005. 9008.]
    """
    if len(data.shape) > 1:
        data = ravel(data)

    dif   = diff(data)
    index = argwhere(dif < tol)

    return ravel(index + 1)

def index_nans(data: ndarray, axis: int=0):
    """Index of NaN values in an array.
    
    Parameters
    ----------
    data
        Array to search for NaN values.
    axis
        Axis along which NaN values will be searched.
        The detault is 0.

    Returns
    -------
    :
        Index array with NaN values in the given axis.
        
    Raises
    ------
    IndexError
        If ``axis`` is greater than the dimension of ``data``.

    Examples
    --------
    :func:`index_nans` is useful to remove NaN values from arrays.

    >>> from numpy import arange, nan, delete
    >>> from araucaria.utils import index_nans
    >>> data      = arange(20, dtype=float).reshape(5,4)
    >>> data[1,2] = nan; data[3,1] = nan
    >>> print(data)
    [[ 0.  1.  2.  3.]
     [ 4.  5. nan  7.]
     [ 8.  9. 10. 11.]
     [12. nan 14. 15.]
     [16. 17. 18. 19.]]

    >>> # removing NaN values from rows
    >>> rindex = index_nans(data, 0)
    >>> print(rindex)
    [[1]
     [3]]
    >>> print(delete(data, rindex, 0))
    [[ 0.  1.  2.  3.]
     [ 8.  9. 10. 11.]
     [16. 17. 18. 19.]]

    >>> # removing NaN values from columns
    >>> cindex = index_nans(data, 1)
    >>> print(cindex)
    [[1]
     [2]]
    >>> print(delete(data, cindex, 1))
    [[ 0.  3.]
     [ 4.  7.]
     [ 8. 11.]
     [12. 15.]
     [16. 19.]]
    """
    if axis > len(data.shape):
        raise IndexError('axis is larger than the dimensions of the array.')
    
    # values are inverted since there seems to be an inconsistency in
    # apply_along_axis 
    if axis == 0:
        aval = 1
    elif axis == 1:
        aval = 0
    else:
        aval = axis
    
    index = apply_along_axis(lambda x : any(isnan(x)), aval, data)

    return argwhere(index)

def index_nearest(data: ndarray, val: float, kind: str='nearest') -> float:
    """Index of nearest value in an array.
    
    Parameters
    ----------
    data
        Array to search for nearest value.

    val
        Search value. It supports :data:`~numpy.inf`.
    kind
        Either 'lower', 'nearest' or 'higher'. 
        The default is 'nearest'. See Notes for details. 

    Returns
    -------
    :
        Index of nearest value in the array.
    
    Raises
    ------
    ValueError
        If ``kind`` is not recognized.
    
    Notes
    -----
    The ``kind`` parameter controls the returned index:
    
    - If ``kind='lower'`` the returned index will be of a value
      strictly lower than ``val``, or 0 if ``val`` if lower than the
      minimum value of ``data``.
    - If ``kind='nearest'`` the returned index will be of the nearest
      value with respect to ``val``.
    - If ``kind='higher'`` the returned index will be of a value
      strictly higher than ``val``, or -1 if ``val`` is higher than the
      maximum value of ``data``.
    
    Examples
    -------
    >>> from numpy import linspace
    >>> from araucaria.utils import index_nearest
    >>> energy = linspace(8900, 9000, 6)
    >>> val    = 8965
    >>> # find nearest value
    >>> index  = index_nearest(energy, val)
    >>> print(index, energy[index], val)
    3 8960.0 8965
    
    >>> # find strictly lower nearest value
    >>> index  = index_nearest(energy, val, kind='lower')
    >>> print(index, energy[index], val)
    3 8960.0 8965
    
    >>> # find strictly higher nearest value
    >>> index  = index_nearest(energy, val, kind='higher')
    >>> print(index, energy[index], val)
    4 8980.0 8965
    """
    kinds = ['lower', 'nearest', 'higher']
    
    if kind not in kinds:
        raise ValueError('kind %s not recognized.' % kind)
    
    if val <= data[0]:
        index = 0
    elif val >= data[-1]:
        index = len(data) - 1  # index starts from cero.
    elif kind == 'nearest':
        index = abs(data-val).argmin()
    elif kind == 'lower':
        index = max(where(data<=val)[0])
    else: # kind higher
        index = min(where(data>=val)[0])

    return index

def interp_yvals(x: ndarray, y: ndarray, xnew: ndarray, 
                kind: str='cubic') -> ndarray:
    """Returns interpolated values for a 1-D function.

    Parameters
    -----------
    x
        Array with original domain.
    y
        Array with original values of function f(x)=y.
    xnew
        Array with new domain.
    kind
        Type of interpolation.
        See :class:`~scipy.interpolate.interp1d` class
        for valid types.
        Default is 'cubic'.
    
    Returns
    -------
    :
        Array with interpolated values.
    
    Example
    -------
    >>> from numpy import linspace
    >>> from araucaria.utils import interp_yvals
    >>> x  = linspace(0,10)
    >>> y  = x**2
    >>> xp = x[0:10]
    >>> yp = interp_yvals(x,y,xp)
    >>> print(len(yp))
    10
    """
    s = interp1d(x, y, kind=kind)
    yvals = s(xnew)

    return yvals

def read_fdicts(fpath: Path) -> List[dict]:
    """Reads file with multiple dictionaries

    Parameters
    ----------
    fpath
        File path.

    Returns
    -------
    :
        List with dictionaries.

    Example
    -------
    >>> from os import remove
    >>> from araucaria.utils import read_fdicts
    >>> fpath ='file.txt'
    >>> data  = "{'ener': [1,2,3], 'mu': [1,2,3]}"
    >>> # create file with dictionary data
    >>> with open(fpath, 'w') as f:
    ...     fw = f.write(data)
    >>> # reading file with dictionary
    >>> dicts = read_fdicts(fpath)
    >>> remove(fpath)
    >>> for d in dicts:
    ...     print(type(d), d)
    <class 'dict'> {'ener': [1, 2, 3], 'mu': [1, 2, 3]}
    """
    # regex search for dictionaries
    regex = '''(\{[\w\s:.,+\-'"\[\]\(\)]+\})'''

    # reading file
    with open(fpath) as f:
        rawdata = f.read()

    # searching file and creating dictionaries with list comprehension
    data = findall(regex, rawdata)
    data = [literal_eval(raw) for raw in data]

    return data

if __name__ == '__main__':
    import doctest
    doctest.testmod()
