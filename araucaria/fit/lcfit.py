#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Linear combination fitting (LCF) refers to the interpretation of an unknown XAFS signal as a
summation of known XAFS reference signals.

For the case of the XANES region of an unknown spectrum :math:`\mu_s(E)`, LCF translates into 
obtaining the amplitude coefficients :math:`\\alpha_i` that minimize the residuals of the following equation:

.. math::
    
    \mu_s(E) = \sum_{i=1}^n \\alpha_i \mu_i(E) + \epsilon(E), \quad i \in \{1,\dots,n \}

Considering the following set of constraints:

.. math::

        0 \leq &\\alpha_i \leq 1, \quad i \in \{1,\dots,n \}

        \sum_{i=1}^n &\\alpha_i = 1 \quad\\textrm{(optional)}

Where

- :math:`E`          : photoelectron energy.
- :math:`\mu_s(E)`   : normalized absorption of the fitted spectrum.
- :math:`\mu_i(E)`   : normalized absorption of  reference spectrum "i".
- :math:`\\alpha_i`  : amplitude coefficient for reference spectrum "i".
- :math:`\epsilon(E)`: residuals.
- :math:`n`          : number of reference spectra.

Analogously, for the case of the EXAFS region of an unknown spectrum :math:`\chi(k)`, 
LCF translates into minimizing the residuals of the following equation:

.. math::
    
    k^{kw}\chi_s(k) = \sum_{i=1}^n \\alpha_i k^{kw}\chi_i(k) + \epsilon(k), \quad i \in \{1,\dots,n \}

Considering the following set of constraints:

.. math::

        0 \leq &\\alpha_i \leq 1, \quad i \in \{1,\dots,n \}

        \sum_{i=1}^n &\\alpha_i = 1 \quad\\textrm{(optional)}

Where

- :math:`k`          : photoelectron wavenumber.
- :math:`\chi_s(k)`  : EXAFS modulation of the fitted spectrum.
- :math:`\chi_i(E)`  : EXAFS modulation of the reference spectrum "i".
- :math:`\epsilon(k)`: residuals.
- :math:`kw`         : weighting coefficient for the photoelectron wavenumber.

The :mod:`~araucaria.fit.lcf` module offers the following functions to perform linear 
combintation fitting (LCF):

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`lcf`
     - Performs LCF on a XAFS spectrum.
   * - :func:`lcf_report`
     - Returns a formatted LCF report.
   * - :func:`sum_references`
     - Sum of weighted references.
   * - :func:`residuals`
     - Calculates residuals for LCF.
"""
from typing import List
from pathlib import Path
from warnings import warn
from numpy import ndarray, where, gradient, around, inf, sum
from scipy.interpolate import interp1d
from lmfit import Parameter, Parameters, minimize, fit_report
from .. import Group, Dataset, Collection
from ..utils import check_objattrs, index_xrange

def lcf(collection: Collection, fit_region: str='xanes',
        fit_range: list=[-inf,inf], scantag: str='scan',
        reftag: str='ref', kweight: int=2, sum_one: bool=True,
        method: str='leastsq') -> Dataset:
    """Performs linear combination fitting on a XAFS spectrum.

    Parameters
    ----------
    collection
        Collection containing the group for LCF analysis and the groups
        with the reference scans.
    fit_region
        XAFS region to perform the LCF. Accepted values are 'dxanes',
        'xanes', or 'exafs'. The default is 'xanes'.
    fit_range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    scantag
        Key to filter the scan group in the collection based on the ``tags`` 
        attribute. The default is 'scan'.
    reftag
        Key to filter the reference groups in the collection based on the ``tags`` 
        attribute. The default is 'scan'.
    kweight
        Exponent for weighting chi(k) by k^kweight. Only valid for ``fit_region='exafs'``. 
        The default is 2.
    sum_one
        Conditional to force  sum of fractions to be one.
        The default is True.
    method
        Fitting method. Currently only local optimization methods are supported.
        See the :func:`~lmfit.minimizer.minimize` function of ``lmfit`` for a list 
        of valid methods.
        The default is ``leastsq``.
    
    Returns
    -------
    :
        Fit group with the following arguments:

        - ``energy``   : array with energy values. 
          Returned only if ``fit_region='xanes'`` or ``fit_region='dxanes'``.
        - ``k``        : array with wavenumber values. 
          Returned only if ``fit_region='exafs'``.
        - ``scangroup``: name of the group containing the fitted spectrum.
        - ``refgroups``: list with names of groups containing reference spectra. 
        - ``scan``     : array with values of the fitted spectrum.
        - ``ref``      : array with interpolated values for each reference spectrum.
        - ``fit``      : array with fit result.
        - ``min_pars`` : object with the optimized parameters and goodness-of-fit statistics.
        - ``lcf_pars`` : dictionary with lcf parameters.

    Raises
    ------
    TypeError
        If ``collection`` is not a valid Collection instance.
    AttributeError
        If ``collection`` has no ``tags`` attribute.
    AttributeError
        If groups have no ``energy`` or ``norm`` attribute.
        Only verified if ``fit_region='dxanes'`` or ``fit_region='xanes'``.
    AttributeError
        If groups have no ``k`` or ``chi`` attribute.
        Only verified if and ``fit_region='exafs'``.
    KeyError
        If ``scantag`` or ``refttag`` are not keys of the ``tags`` attribute.
    ValueError
        If ``fit_region`` is not recognized.
    ValueError
        If ``fit_range`` is outside the doamin of a reference group.

    Important
    ---------
    If more than one group in ``collection`` is tagged with ``scantag``, 
    a warning will be raised and only the first group will be fitted.

    Notes
    -----
    The ``min_pars`` object is returned by the :func:`minimize` function of 
    ``lmfit``, and contains the following attributes (non-exhaustive list):
    
    - ``params``    : dictionary with the optimized parameters.
    - ``var_names`` : ordered list of parameter names used in optimization.
    - ``covar``     : covariance matrix from minimization.
    - ``init_vals`` : list of initial values for variable parameters using 
      ``var_names``.
    - ``success``   : True if the fit succeeded, otherwise False.
    - ``nvarys``    : number of variables.
    - ``ndata``     : number of data points.
    - ``chisqr``    : chi-square.
    - ``redchi``    : reduced chi-square.
    - ``residual``  : array with fit residuals.

    Example
    -------
    >>> from numpy.random import seed, normal
    >>> from numpy import arange, sin, pi
    >>> from araucaria import Group, Dataset, Collection
    >>> from araucaria.fit import lcf
    >>> from araucaria.utils import check_objattrs
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
    >>> for i, group in enumerate((group1,group2, group3)):
    ...     collection.add_group(group, tag=tags[i])
    >>> # performing lcf
    >>> out = lcf(collection, fit_region='exafs', fit_range=[3,10], 
    ...           kweight=0, sum_one=False)
    >>> check_objattrs(out, Dataset, 
    ... attrlist=['k', 'scangroup', 'refgroups', 
    ... 'scan', 'ref1', 'ref2', 'fit', 'min_pars', 'lcf_pars'])
    [True, True, True, True, True, True, True, True, True]
    >>> for key, val in out.min_pars.params.items():
    ...     print('%1.4f +/- %1.4f' % (val.value, val.stderr))
    0.4003 +/- 0.0120
    0.5943 +/- 0.0120
    """    
    # checking class and attributes
    check_objattrs(collection, Collection, attrlist=['tags'], exceptions=True)
    
    # verifying fit type
    fit_valid = ['dxanes', 'xanes','exafs']
    if fit_region not in fit_valid:
        raise ValueError('fit_region %s not recognized.'%fit_region)

    # required groups
    # at least a spectrum and a single reference must be provided
    for tag in (scantag, reftag):
        if tag not in collection.tags:
            raise KeyError("'%s' is not a valid key for the collection." % tag)

    # scan and ref tags
    scangroup = collection.tags[scantag]
    if len(scangroup) > 1:
        warn("More than one group is tagged as scan. Only the first group will be considered.")
    scangroup = scangroup[0]
    refgroups = collection.tags[reftag]
    refgroups.sort()

    # the first element is the scan group
    groups = [scangroup] + refgroups

    # storing report parameters
    lcf_pars = {'fit_region':fit_region, 'fit_range':fit_range, 'sum_one':sum_one}

    # report parameters for exafs lcf
    if fit_region == 'exafs':
        lcf_pars['kweight'] = kweight
        # storing name of x-variable (exafs)
        xvar = 'k'
    # report parameters for xanes/dxanes lcf
    else:
        # storing name of x-variable (xanes/dxanes)
        xvar = 'energy'

    # content dictionary
    content = {'scangroup': scangroup,
               'refgroups': refgroups}
    
    # reading and processing spectra
    for i, name in enumerate(groups):
        dname = 'scan' if i==0 else 'ref'+str(i)
        group = collection.get_group(name).copy()

        if fit_region == 'exafs':
            check_objattrs(group, Group, attrlist=['k', 'chi'], exceptions=True)
        else:
            # fit_region == 'xanes' or 'dxanes'
            check_objattrs(group, Group, attrlist=['energy', 'norm'], exceptions=True)

        if i == 0:
            # first value is the spectrum, so we extract the 
            # interpolation values from xvar
            xvals  = getattr(group, xvar)
            index  = index_xrange(fit_range, xvals)
            xvals  = xvals[index]

            # storing the y-variable
            if fit_region == 'exafs':
                yvals = xvals**kweight*group.chi[index]
            elif fit_region == 'xanes':
                yvals = group.norm[index]
            else:
                # derivative lcf
                yvals = gradient(group.norm[index]) / gradient(group.energy[index])
        else:
            # spline interpolation of references
            if fit_region == 'exafs':
                s = interp1d(group.k, group.k**kweight*group.chi, kind='cubic')
            elif fit_region =='xanes':
                s = interp1d(group.energy, group.norm, kind='cubic')
            else:
                s = interp1d(group.energy, gradient(group.norm)/gradient(group.energy), kind='cubic')
            
            # interpolating in the fit range
            try:
                yvals = s(xvals)
            except:
                raise ValueError('fit_range is outside the domain of group %s' % name)
        
        # saving yvals in the dictionary
        content[dname] = yvals

    # setting xvar as an attribute of datgroup
    content[xvar] = xvals
    
    # setting initial values and parameters for fit model
    initval = around(1/(len(groups)-1), decimals=1)
    params  = Parameters()
    expr    = str(1)
    
    for i in range(len(groups)-1):
        parname = 'amp'+str(i+1)
        if ( (i == len(groups) - 2) and (sum_one == True) ):
            params.add(parname, expr=expr)
        else:
            params.add(parname, value=initval, min=0, max=1, vary=True)
            expr += ' - amp'+str(i+1)

    # perform fit
    min = minimize(residuals, params, method=method, args=(content,))
    
    # storing fit data, parameters, and results
    content['fit']      = sum_references(min.params, content)
    content['lcf_pars'] = lcf_pars
    content['min_pars'] = min
    
    out = Dataset(**content)
    return out

def lcf_report(out: Dataset) -> str:
    """Returns a formatted LCF Report to ``sys.stdout``.
    
    Parameters
    ----------
    out
        Valid Dataset from :func:`lcf`.
    
    Returns
    -------
    :
        LCF report.
        
    Raises
    ------
    TypeError
        If ``out`` is not a valid Dataset instance.
    AttributeError
        If attribute ``min_pars``, ``lcf_pars``, ``scangroup``,
        or ``refgroups`` does not exist in ``group``.

    Notes
    -----
    :func:`lcf_report` is a wrapper for :func:`fit_report` of ``lmfit``,
    that writes additional information on group names and calling parameters.
    
    Example
    -------
    >>> from numpy.random import seed, normal
    >>> from numpy import arange, sin, pi
    >>> from araucaria import Group, Collection
    >>> from araucaria.fit import lcf, lcf_report
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
    >>> print(lcf_report(out))
    [[Parameters]]
        fit_region         = exafs
        fit_range          = [3, 10]
        sum_one            = False
        kweight            = 0
    [[Groups]]
        scan               = group3
        ref1               = group1
        ref2               = group2
    [[Fit Statistics]]
        # fitting method   = leastsq
        # function evals   = 10
        # data points      = 141
        # variables        = 2
        chi-square         = 1.40551323
        reduced chi-square = 0.01011161
        Akaike info crit   = -645.778389
        Bayesian info crit = -639.880869
    [[Variables]]
        amp1:  0.40034377 +/- 0.01195335 (2.99%) (init = 0.5)
        amp2:  0.59428689 +/- 0.01199230 (2.02%) (init = 0.5)
    """
    check_objattrs(out, Dataset, attrlist=['min_pars', 
    'lcf_pars', 'scangroup', 'refgroups'], exceptions=True)
    
    header = '[[Parameters]]\n'
    for key, val in out.lcf_pars.items():
        header = header + '    {0:19}= {1}\n'\
        .format(key, val)

    header = header+'[[Groups]]\n'
    for i, val in enumerate(([out.scangroup] + out.refgroups)):
        if i == 0:
            name = 'scan'
        else:
            name = 'ref%i' % i
        header = header + '    {0:19}= {1}\n'\
        .format(name, val)
    return (header+fit_report(out.min_pars))

def sum_references(pars: Parameter, data: dict) -> ndarray:
    """Returns the sum of references weighted by amplitude coefficients.
    
    Parameters
    ----------
    pars
        Parameter object from ``lmfit`` containing the amplitude
        coefficients for each reference spectrum. At least attribute
        'amp1' should exist in the object.
    data
        Dictionary with the reference arrays. At leasr key 'ref1'
        should exist in the dictionary.
    
    Returns
    -------
    :
        Sum of references weighted by amplitude coefficients.
        
    Important
    -----
    The number of 'amp' attributes in ``pars`` should match the 
    number of 'ref' keys in ``data``.

    Example
    -------
    >>> from numpy import allclose
    >>> from lmfit import Parameters
    >>> from araucaria.fit import sum_references
    >>> pars = Parameters()
    >>> pars.add('amp1', value=0.4)
    >>> pars.add('amp2', value=0.7)
    >>> data = {'ref1': 1.0, 'ref2': 2.0}
    >>> allclose(sum_references(pars, data), 1.8)
    True
    """
    return (sum([pars['amp'+str(i)]* data['ref'+str(i)] 
                   for i in range(1,len(pars)+1)], axis=0))

def residuals(pars: Parameter , data: dict) -> ndarray:
    """Residuals between a spectrum and a linear combination of references.
    
    Parameters
    ----------
    pars
        Parameter object from ``lmfit`` containing the amplitude
        coefficients for each reference spectrum. At least attribute
        'amp1' should exist in the object.
    data
        Dictionary with the scan and reference arrays.
        At least keys 'scan' and 'ref1' should exist in
        the dictionary.
    
    Returns
    -------
    :
        Array with residuals.

    Important
    -----
    The number of 'amp' attributes in ``pars`` should match the 
    number of 'ref' keys in ``data``.

    Example
    -------
    >>> from numpy import allclose
    >>> from lmfit import Parameters
    >>> from araucaria.fit import residuals
    >>> pars = Parameters()
    >>> pars.add('amp1', value=0.4)
    >>> pars.add('amp2', value=0.7)
    >>> data = {'scan': 2, 'ref1': 1.0, 'ref2': 2.0}
    >>> allclose(residuals(pars, data), 0.2)
    True
    >>> data['scan'] = 1.6
    >>> allclose(residuals(pars, data), -0.2)
    True
    """
    eps = 1.0
    return (data['scan'] - sum_references(pars, data))/eps
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()
