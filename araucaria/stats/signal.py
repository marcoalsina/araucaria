#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats.signal` module offers the following 
functions to filter and analyze arrays:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`roll_med`
     - Computes the rolling median of an array.
   * - :func:`spectral_entropy`
     - Computes the spectral entropy of an array.
"""
#from typing import Union, Tuple
from numpy import (ndarray, array, nan, isnan,
                   count_nonzero, nanmedian, sum, log, e)
#from numpy import arange, std, append, concatenate, delete, ediff1d, var, inf, sqrt, argmin, abs
#from scipy.fft import fft, fftfreq
from scipy.signal import periodogram, welch
#from scipy.signal import convolve
#from .. import Group
#from ..xas.xasutils import ktoe
#from ..utils import check_objattrs

def roll_med(x: ndarray, window: int, min_samples: int=2, 
             edgemethod: str='nan') -> ndarray:
    """Computes the rolling median of an array.

    Parameters
    ----------
    x:
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
    NameError
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
        raise NameError('please choose a valid edgemethod.')

    # calculating points on either side of the point of interest in the window
    movement  = int((window - 1) / 2) 
    med_array = array([nan for point in x])
    
    for i, point in enumerate(x[ : -movement]):
        if i>=movement:
            if count_nonzero(isnan(x[i - movement : i + 1 + movement]) == False) >= min_samples:
                med_array[i] = nanmedian(x[i - movement : i + 1 + movement])

    if edgemethod == 'nan':
        return med_array

    for i, point in enumerate(x[ : movement]):
        if edgemethod == 'calc':
            if count_nonzero(isnan(x[0 : i + 1 + movement]) == False) >= min_samples:
                med_array[i] = nanmedian(x[0 : i + 1 + movement])
        elif edgemethod == 'extend':
            med_array[i] = med_array[movement]

    for i, point in enumerate(x[-movement : ]):
        if edgemethod == 'calc':
            if count_nonzero(isnan(x[(-2 * movement) + i : ]) == False) >= min_samples:
                med_array[-movement + i] = nanmedian(x[(-2 * movement) + i : ])
        elif edgemethod == 'extend':
            med_array[-movement + i] = med_array[-movement - 1]

    return med_array

def spectral_entropy(x: ndarray, base: float=None, method: str='period', 
                     normalize: bool=False, axis: int=-1, **psd_pars: dict) -> float:
    """Computes the spectral entropy of an array.

    Parameters
    ----------
    x
        Array with data.
    base
        Base to compute the logaritm for entropy.
        The default is None, which defauls to :con:
    method
        Method to compute the power spectral density:

        - 'period': uses the :func:`~scipy.signal.periodogram` function of ``scipy``.
        - 'welch': uses the :func:`~scipy.signal.welch` function of ``scipy``.

        The default is 'period'.
    normalize
        Whether to return the spectral entropy normalized between 0 and 1.
        Otherwise entropy is returned in the respective information unit (e.g. nat).
        The default is False.
    axis
        Axis along which the spectral entropy is computed.
        The default is -1 (over the last axis).
    psd_pars
        Dictionary with additional parameters for the selected power spectral density method.

    Returns
    -------
    :
        Spectral entropy.

    Raises
    ------
    NameError
        If ``method`` is not recognized.

    Notes
    -----
    The spectral entropy is defined as the Shannon entropy [1]_ of the normalized 
    power spectral density of a signal [2]_:

    .. math::

       H(P) = - \sum_{f=0}^{f_s/2} P_f \ log P_f

    where:

    - :math:`P_f`   : normalized power spectral density at a given frequency :math:`f`.
    - :math:`f_s/2` : Nyquist frequency, i.e. half of the sampling rate :math:`f_s`.

    References
    ----------
    .. [1] C.E. Shannon and W. Weaver (1949) "The Mathematical Theory of Communication".
        University of Illinois Press, Chicago, IL, pp. 48-53.

    .. [2] T. Inouye et al. (1991) "Quantification of EEG irregularity by use of the entropy 
       of the power spectrum" Electroencephalography and Clinical Neurophysiology 79(3), 
       pp. 204-210. https://doi.org/10.1016/0013-4694(91)90138-T.

    Examples
    --------
    >>> from numpy import arange, sin, pi, e
    >>> from numpy.random import seed, uniform
    >>> from araucaria.stats import spectral_entropy
    >>> fs   = 10               # sampling frequency, Hz
    >>> N    = 100              # number of points
    >>> freq = 1                # signal frequency
    >>> t    = arange(N)/fs     # time domain
    >>> x    = sin(2*pi*freq*t) # signal
    >>> # spectral entropy (periodogram)
    >>> h = spectral_entropy(x, method='period')
    >>> print('%1.3f' % h)
    0.000

    >>> # spectral entropy (welch method)
    >>> psd_pars = {'window': 'boxcar', 'nperseg': len(t)}
    >>> h = spectral_entropy(x, method='welch', **psd_pars)
    >>> print('%1.3f' % h)
    0.000

    >>> # random noise
    >>> seed(1221)
    >>> N    = 5000
    >>> x    = uniform(size=N)
    >>> h    = spectral_entropy(x, base=2)  # bits
    >>> print('%1.4g' % h)
    10.68
    >>> h_n  = spectral_entropy(x, base=2, normalize=True)
    >>> print('%1.4g' % h_n)
    0.9459
    """
    valid_methods = ['period', 'welch' ]
    if method not in valid_methods:
        raise NameError('%s method not recognized.' % method)

    if method == valid_methods[0]:
        # periodogram
        f, psd = periodogram(x, axis=axis, **psd_pars)
    else:
        # welch
        f, psd = welch(x, axis=axis, **psd_pars)

    # normalized psd
    psd_norm = psd / psd.sum(keepdims=True, axis=axis)

    # computing entropy
    s = -sum(psd_norm * log(psd_norm), axis=axis)
    if base is not None:
        s /= log(base)

    # normalizing if requested
    if normalize:
        s /= log(psd_norm.shape[axis])
        if base is not None:
            s *= log(base)
    return s

if __name__ == '__main__':
    import doctest
    doctest.testmod()