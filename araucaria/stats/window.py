#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import (ndarray, array, nan, isnan, 
                   count_nonzero, nanmedian)

def roll_med(data: ndarray, window: int, min_samples: int=2, 
             edgemethod: str='nan') -> ndarray:
    """Computes the rolling median of a univariate array.
    
    Parameters
    ----------
    data:
        Array to compute the rolling median.
    window:
        Size of the rolling window for analysis.
    min_samples:
        Minimum sample points to calculate the median in each window.
        The default is 2.
    edgemethod :
        Dictates how medians are calculated at the edges of the array.
        Options are 'nan', 'calc' and 'extend'. See the Notes for further details.
        The default is 'nan'.

    Returns
    -------
    :
        Rolling median of the array.

    Raises
    ------
    ValueError
        If ``window`` is not an odd value.
    ValueError
        If ``window`` is smaller or equal than 3.
    TypeError
        If ``window`` is not an integer.
    ValueError
        If ``edgemethod`` is not recognized.

    Notes
    -----
    This function calculates the median of a moving window. Results are returned in the 
    index corresponding to the center of the window. The function ignores :data:`~numpy.nan` 
    values in the array.

    - ``edgemethod='nan'`` uses :data:`~numpy.nan` values for missing values at the edges. 
    - ``edgemethod='calc'`` uses an abbreviated window at the edges 
      (e.g. the first sample will have (window/2)+1 points in the calculation).
    - ``edgemethod='extend'`` uses the nearest calculated value for missing values at the edges.

    Warning
    -------
    If ``window`` is less than ``min_samples``, :data:`~numpy.nan` is given as the median.

    Example
    -------
    .. plot::
        :context: reset
    
    
        >>> from numpy import pi, sin, linspace
        >>> from araucaria.stats import roll_med
        >>> import matplotlib.pyplot as plt
        >>> # generating a signal and its rolling median
        >>> f1   = 0.2 # frequency
        >>> t    = linspace(0,10)
        >>> y    = sin(2*pi*f1*t)
        >>> line = plt.plot(t,y, label='signal')
        >>> for method in ['calc', 'extend', 'nan']:
        ...    fy   = roll_med(y, window=25, edgemethod=method)
        ...    line = plt.plot(t, fy, marker='o', label=method)
        >>> lab = plt.xlabel('t')
        >>> lab =plt.ylabel('y')
        >>> leg = plt.legend()
        >>> plt.show(block=False)
    """
    if window % 2 == 0:
        raise ValueError('window length must be an odd value.')
    elif window < 3 or type(window)!=int:
        raise ValueError('window length must be larger than 3.')

    validEdgeMethods = ['nan', 'extend', 'calc']     
    if edgemethod not in validEdgeMethods:
        raise ValueError('please choose a valid edgemethod.')

    # calculating points on either side of the point of interest in the window
    movement  = int((window - 1) / 2) 
    med_array = array([nan for point in data])
    
    for i, point in enumerate(data[ : -movement]):
        if i>=movement:
            if count_nonzero(isnan(data[i - movement : i + 1 + movement]) == False) >= min_samples:
                med_array[i] = nanmedian(data[i - movement : i + 1 + movement])

    if edgemethod == 'nan':
        return med_array

    for i, point in enumerate(data[ : movement]):
        if edgemethod == 'calc':
            if count_nonzero(isnan(data[0 : i + 1 + movement]) == False) >= min_samples:
                med_array[i] = nanmedian(data[0 : i + 1 + movement])
        elif edgemethod == 'extend':
            med_array[i] = med_array[movement]

    for i, point in enumerate(data[-movement : ]):
        if edgemethod == 'calc':
            if count_nonzero(isnan(data[(-2 * movement) + i : ]) == False) >= min_samples:
                med_array[-movement + i] = nanmedian(data[(-2 * movement) + i : ])
        elif edgemethod == 'extend':
            med_array[-movement + i] = med_array[-movement - 1]

    return med_array

if __name__ == '__main__':
    import doctest
    doctest.testmod()