#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.stats.genesd` module offers the following 
functions to detect outliers in a univariate array using the 
generalized extreme Studentized deviate test:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`genesd`
     - Identifies outliers in a data array.
   * - :func:`find_ri`
     - Computes the Ri statistics for the generalized ESD test.
   * - :func:`find_critvals`
     - Computes the critical values for the generalized ESD test.
"""

from typing import Tuple
from numpy import (ndarray, array, argwhere, copy, delete, 
                   mean, std, absolute, argmax, searchsorted)
from scipy.stats import t
from .. import Report

def genesd(data: ndarray, r: int, alpha: float) -> Tuple[str, list]:
    """Identifes outliers in a data array.
    
    This function uses the generalized extreme Studentized 
    deviate (ESD) test to detect one or more outliers 
    in univariate data [1]_.

    Parameters
    ----------
    data:
        Array to identify outliers.
    r:
        Maximum number of outliers.
    alpha:
        Significance level for the statistical test.

    Returns
    -------
    report :
        Report of the generalized ESD test.
    index :
        Indices of outliers in the data.

    Notes
    -----
    The identification of outliers considers the following hypothesis test:
    
    - :math:`H_0`:  there are no outliers in the data.
    - :math:`H_1`: there are up to :math:`r` outliers in the data.
    
    The algorithm performs the following operations:

    1. The :math:`R_i` test statistics are computed for :math:`r` potential outliers, 
       removing the largest potential outlier from the data at each succesive calculation 
       of the test statistic.
    2. The :math:`\\lambda_i` critical values are computed for :math:`r` potential outliers, 
       considering a significance level of :math:`\\alpha` for the t-distribution.
    3. Both values are compared, and the largest number of outliers where 
       :math:`R_i > \\lambda_i` is accepted as the number of outliers.

    References
    ----------
    .. [1] Rosner, B. (1983) "Percentage Points for a Generalized ESD 
       Many-Outlier Procedure", Technometrics, 25(2), pp. 165-172.

    Example
    -------
    >>> # calculating outliers for Rosner data (1983):
    >>> from numpy import loadtxt, allclose
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.stats import genesd
    >>> path  = get_testpath('rosner.dat')
    >>> data  = loadtxt(path)
    >>> r     = 5
    >>> alpha = 0.05
    >>> report, index = genesd(data, r, alpha)
    >>> print(report)
    Generalized ESD test for outliers
      H0: there are no outliers in the data
      H1: there are up to 5 outliers in the data
      Significance level:  alpha = 0.05
      Critical region:  Reject H0 if R_i > lambda_i
    =====================================
    n outliers  x_i    R_i     lambda_i  
    =====================================
    1           6.01   3.1189  3.1588    
    2           5.42   2.943   3.1514    
    3           5.34   3.1794  3.1439    *
    4           4.64   2.8102  3.1362    
    5           -0.25  2.8156  3.1282    
    =====================================
    >>> print(data[index])
    [6.01 5.42 5.34]
    """
    # calculating statistics and critical values
    ri, xi = find_ri(data, r)
    lambdi = find_critvals(len(data), r, alpha)
    
    # creating report
    header  = 'Generalized ESD test for outliers\n'
    header += '  H0: there are no outliers in the data\n'
    header += '  H1: there are up to %i outliers in the data\n' % r
    header += '  Significance level:  alpha = %g\n' % alpha
    header += '  Critical region:  Reject H0 if R_i > lambda_i\n'
    
    report = Report()
    report.set_columns(['n outliers', 'x_i', 'R_i', 'lambda_i'])
    for i, val in enumerate(ri):
        index = i + 1
        report.add_row([index, xi[i], val, lambdi[i]])
    report = report.show(print_report=False)

    # calculating R_i - lambda_i > 0
    diff  = [val[0] - val[1] for val in zip(ri,lambdi)]
    index = argwhere(array(diff) > 0)
    
    if index.size == 0:
        pass
    else:
        ival = index[-1][0] # index for n outliers
        
        # marking column in report
        tcols        = report.split('\n')
        tcols[ival + 3] += '*'
        report       = '\n'.join(tcols)
        
        # calculating index of outliers in original array
        vals  = xi[:(ival + 1)]
        index = [argwhere(data == val)[0][0] for val in vals]

    report = header + report
    return (report, index)

def find_ri(data: ndarray, r: int) -> Tuple[float, float]:
    """Computes the :math:`R_i` test statistics for the 
    generalized extreme Studentized deviate (ESD) test.

    Parameters
    ----------
    data:
        Array to compute test statistic.
    r:
        Maximum number of outliers.

    Returns
    -------
    :
        Test statistic for the generalized ESD test.

    :
        Value of data points furthest from the mean.

    Notes
    -----
    The :math:`R_i` test statistics are calculated as follows:
    
    .. math::

        R_i = \\frac{\\textrm{max} | x_i - \\bar{x}_{n-i+1}| }{s_{n-i+1}} 
        \quad i \in \{1,2, \dots, r \}

    Where

    - :math:`\\bar{x}_{n-i+1}`: sample mean of reduced array.
    - :math:`s_{n-i+1}`       : sample standard deviation of reduced array.
    - :math:`n-i+1`           : number of points in the reduced array.
    - :math:`r`               : maximum number of outliers.

    After each calculation rhe observation that maximizes :math:`| x_i − \\bar{x} |` 
    is removed, and :math:`R_i` is computed with n - i + 1 observations. 
    This procedure is repeated until r observations have been removed 
    from the array.

    Example
    -------
    >>> # calculating test statistics from Rosner's data (1983):
    >>> from numpy import loadtxt
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.stats import find_ri
    >>> path  = get_testpath('rosner.dat')
    >>> data  = loadtxt(path)
    >>> r     = 5
    >>> ri,xi = find_ri(data,r)
    >>> for val in ri:
    ...     print('%1.3f' % val)
    3.119
    2.943
    3.179
    2.810
    2.816
    """
    ri     = []         # container for statistic
    xi     = []         # container for values
    cpdata = copy(data) # copy of data

    for i in range(r):
        avg    = mean(cpdata)
        sigma  = std(cpdata, ddof=1)

        # obtaining index for residual maximum
        residuals = absolute(cpdata - avg)
        max_index = argmax(residuals)

        # appending in list and removing data point
        ri.append(residuals[max_index]/sigma)
        xi.append(cpdata[max_index])

        # deleting max residual value
        cpdata = delete(cpdata, max_index)
    return (ri, xi)


def find_critvals(n: int, r: int, alpha: float) -> list:
    """Computes critical values :math:`\lambda_i` for the 
    generalized extreme Studentized deviate (ESD) test.

    Parameters
    ----------
    n:
        Number of data points.
    r:
        Maximum number of outliers.
    alpha:
        Significance level for the statistical test.
    
    Returns
    -------
    :
        Critical values.

    Notes
    -----
    The :math:`\lambda_i` values are calculated as follows:

    .. math::

        \lambda_i = \\frac{ (n-i)\ t_{p, n-i-1} }{ \sqrt{(n-i-1-t_{n-i-1}^2)(n-i+1)} } 
        \quad i \in \{1,2, \dots, r \}

    .. math::

        p = 1 - \\frac{\\alpha}{2(n-i+1)}

    Where

    - :math:`n`       : number of points in the array.
    - :math:`\\alpha` : significance level.
    - :math:`t_{p,v}` : percent point function of the t-distribution 
      at :math:`p` value and :math:`v` degrees of freedom.
    - :math:`r`               : maximum number of outliers.

    Example
    -------
    >>> from araucaria.stats import find_critvals
    >>> n     = 54    # number of points
    >>> r     = 5     # max number of outliers
    >>> alpha = 0.05  # significance level
    >>> lambd = find_critvals(n, r, alpha)
    >>> for val in lambd:
    ...     print('%1.3f' % val)
    3.159
    3.151
    3.144
    3.136
    3.128
    """
    critvals = []  # container for critical values
    for i in range(1, r + 1):
        p = 1 - (alpha / (2 * (n - i + 1) ) )
    
        # finds t value corresponding to probability that 
        # sample within data set is itself an outlying point
        tval = t.ppf(p, n - i - 1) 
        val  = ( (n - i) * tval) / ( ( (n - i - 1 + (tval**2)) * (n - i + 1) )**(1/2) )
        critvals.append(val)
    return critvals

if __name__ == '__main__':
    import doctest
    doctest.testmod()