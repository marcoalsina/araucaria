#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xas.normalize` module offers the following functions to normalize a scan:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`find_e0`
     - Calculates the absorption threshold energy of a scan.
   * - :func:`guess_edge`
     - Estimates the nearest absorption edge for a scan.
   * - :func:`pre_edge`
     - Pre-edge substaction and normalization of a scan.
"""
from warnings import warn
from numpy import ndarray, array, inf, gradient, isfinite, isinf, where, ptp
from scipy import polyfit, polyval
from .. import Group
from ..xrdb import nearest_edge
from ..utils import index_nearest, check_objattrs, check_xrange

def find_e0(group: Group, method: str='maxder', tol: float=1e-4,
            pre_edge_kws: dict=None, use_mu_ref: bool=False, 
            update: bool=False) -> float:
    """Calculates the absorption threshold energy of a XAFS scan.

    Parameters
    ----------
    group
        Group containing the spectrum to calculate `e0`.
    method
        Name of the method to find `e0`. Valid names are 'maxder' and 'halfedge'.
        See Notes for details. The default is 'maxder'.
    tol
        Tolerance value for convergence of `e0` calculation.
        Only used if ``method='halfedge'``. The defailt is 1e-4.
    pre_edge_kws
        Dictionary with arguments for :func:`~araucaria.xas.normalize.pre_edge`.
        Only used if ``method='halfedge'``. The defailt is None, which considers
        default values for normalization.
    use_mu_ref
        Indicates if `e0` should be calculated with the 
        reference scan. The default is False.
    update
        Indicates if the group should be updated with the 
        value of `e0`. The default is False.

    Returns
    -------
    :
        Value of `e0`.

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``energy`` does not exist in ``group``.
    AttributeError
        If attribute ``mu_ref`` does not exist in ``group``
        when ``use_mu_ref = True``.
    ValueError
        If ``method`` is not recocgnized.

    Notes
    -----
    If ``method=maxder`` the absorption threshold will be calculated as the 
    maximum derivative in absorption.

    If ``method=halfedge`` the absorption threshold will be calculated iteratively 
    as half the edge step. This method calls :func:`~araucaria.xas.normalize.pre_edge`
    to compute the edge step at each iteration. Parameters for the pre-edge calculation
    can be passed with the ``pre_edge_kws`` parameter. A tolerance for the error between 
    iterations can be set with the ``tol`` parameter.

    If ``use_mu_ref=False`` the absorption threshold will be calculated 
    for the scan attribute of ``group``, as determined by the 
    :func:`~araucaria.main.group.Group.get_mode` method.
    This is the default behavior.
    
    If ``use_mu_ref=True`` the absorption threshold will be calculated for the 
    ``group.mu_ref`` attribute.

    If ``update=True`` the following attribute will be created in ``group``:

    - ``group.e0``: absorption threshold energy :math:`E_0`.

    Important
    ---------
    Computing `e0` with ``method=halfedge`` is sensitive to the parameters used
    to compute the edge step by :func:`~araucaria.xas.normalize.pre_edge`.
    Therefore, different parameters for calculation of the edge step will yield 
    different values of `e0` by this method.

    Currently ``method=halfedge`` considers a maximum of 10 iterations to compute
    `e0`.

    Examples
    --------
    >>> # computing e0 as the maximum derivative
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_dnd
    >>> from araucaria.xas import find_e0
    >>> fpath = get_testpath('dnd_testfile1.dat')
    >>> # extracting mu and mu_ref scans
    >>> group = read_dnd(fpath, scan='mu')
    >>> # find e0 of reference scan
    >>> find_e0(group, use_mu_ref=True)
    29203.249

    >>> # computing e0 as half the edge step
    >>> find_e0(group, method='halfedge', use_mu_ref=True)
    29200.62
    """
    valid_methods = ('maxder', 'halfedge')
    if method not in valid_methods:
        raise ValueError('method %s not recognized.' % method)
    
    # checking class and attributes
    if use_mu_ref:
        check_objattrs(group, Group, attrlist=['energy', 'mu_ref'], exceptions=True)
        mu = group.mu_ref
    else:
        check_objattrs(group, Group, attrlist=['energy'], exceptions=True)
        mu = getattr(group, group.get_mode())

    # storing energy as indepedent array
    energy = group.energy

    # find points of high derivative
    # between 6 and 10% total points at edges are substracted from analysis
    dmu    = gradient(mu)/gradient(energy)
    dmu[where(~isfinite(dmu))] = -1.0
    nmin   = max(3, int(len(dmu)*0.05))  
    maxdmu = max(dmu[nmin:-nmin])

    # make exception if maxdmu equals zero
    high_deriv_pts    = where(dmu >  maxdmu*0.1)[0]
    idmu_max, dmu_max = 0, 0

    for i in high_deriv_pts:
        if (i < nmin) or (i > len(energy) - nmin):
            continue
        if (dmu[i] > dmu_max and
            (i+1 in high_deriv_pts) and 
            (i-1 in high_deriv_pts)):
            idmu_max, dmu_max = i, dmu[i]

    e0 = energy[idmu_max]

    # computing half edge
    if method == valid_methods[1]:
        e0_vals = [e0]  # container for iterative e0 values
        maxcount= 10    # maximum number of iterations
        cond    = True  # conditional variable to exit the iterations
        group   = Group(**{'energy': energy, 'mu': mu})

        while cond:
            if pre_edge_kws is None:
                pre = pre_edge(group, e0 = e0_vals[-1])
            else:
                pre = pre_edge(group, e0 = e0_vals[-1], **pre_edge_kws)

            # calculating half edge step value
            ie0        = index_nearest(energy, e0_vals[-1])
            halfed     = ( pre['post_edge'][ie0] + pre['pre_edge'][ie0] ) / 2
            
            # checking range within energy array
            prerange   = check_xrange(pre['pre_edge_pars']['pre_range'],  energy, refval=e0_vals[-1])
            postrange  = check_xrange(pre['pre_edge_pars']['post_range'], energy, refval=e0_vals[-1])
            
            # finding half edge step energy is constrained near the edge
            pre_index  = index_nearest(energy, prerange[1]  + e0)
            post_index = index_nearest(energy, postrange[0] + e0, kind='lower')
            
            nie0   = index_nearest(mu[pre_index:post_index+1], halfed)
            ne0    = energy[pre_index + nie0]
            e0_vals.append(ne0)

            if (abs(e0_vals[-1] - e0_vals[-2]) < tol) or (len(e0_vals) > maxcount):
                cond = False

        # retrieving the last value in the list
        e0 = e0_vals[-1]

    if update:
        group.e0 = e0
    return e0

def guess_edge(group: Group, e0: float=None, update:bool =False) -> dict:
    """Estimates the nearest absorption edge for a XAFS scan.

    Parameters
    ----------
    group
        Group containing the spectrum for pre-edge substraction and normalization.
    e0
        Absorption threshold energy. If None it will seach for the 
        value stored in ``group.e0``. Otherwise it will be calculated
        using :func:`~araucaria.xas.normalize.find_e0`. with default 
        parameters.
    update
        Indicates if the group should be updated with the normalization attributes.
        The default is False.

    Returns
    -------
    :
        Dictionary with the following arguments:
        
        - ``atsym``   : atomic symbol for the absorption edge.
        - ``edge``    : absorption edge in Siegbanh notation.

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``energy`` does not exist in ``group``.
    IndexError
        If ``e0`` is outside the range of ``group.energy``.

    See also
    --------
    :func:`~araucaria.xrdb.xray.nearest_edge`
        Returns the nearest x-ray edge for a given energy.

    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Group
    >>> from araucaria.io import read_dnd
    >>> from araucaria.xas import find_e0
    >>> from araucaria.utils import check_objattrs
    >>> fpath = get_testpath('dnd_testfile1.dat')
    >>> group = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
    >>> attrs = ['atsym', 'edge']
    >>> e0    = find_e0(group)
    >>> edge  = guess_edge(group, e0, update=True)
    >>> check_objattrs(group, Group, attrs)
    [True, True]
    >>> print(edge)
    {'atsym': 'Sn', 'edge': 'K'}
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['energy'], exceptions=True)

    # storing energy and mu as indepedent arrays
    energy = group.energy

    # assigning e0
    if e0 is not None:
        if e0 < min(energy) or e0 > max(energy):
            raise IndexError('e0 is outside the energy range.')
        else:
            e0 = energy[index_nearest(energy, e0)]
            
    elif hasattr(group, 'e0'):
        if group.e0 < min(energy) or group.e0 > max(energy):
            raise IndexError('group.e0 is outside the energy range.')
        else:
            e0 = energy[index_nearest(energy, group.e0)]
    else:
        e0 = find_e0(group, update=False)

    # estimating edge
    edge = nearest_edge(e0)
    content = {'atsym' : edge[0],
               'edge'  : edge[1],
               }

    if update:
        group.add_content(content)

    return content

def pre_edge(group: Group, e0: float=None, nvict: int=0, nnorm: int=2,
             pre_range: list=[-inf,-50], post_range: list=[100,inf],
             update:bool =False) -> dict:
    """Pre-edge substaction and normalization of a XAFS scan.

    Parameters
    ----------
    group
        Group containing the spectrum for pre-edge substraction and normalization.
    e0
        Absorption threshold energy. If None it will seach for the 
        value stored in ``group.e0``. Otherwise it will be calculated
        using :func:`~araucaria.xas.normalize.find_e0`. with default 
        parameters.
    nvict
        Energy exponent for pre-edge fit with a Victoreen polynomial.
        The default is 0. See Notes for details.
    nnorm
        Degree of polynomial for post-edge fit. The default is 2.
    pre_range
        Energy range with respect to `e0` for the pre-edge fit.
        The default is [-:data:`~numpy.inf`, -50].
    post_range
        Energy range with respect to `e0` for the post-edge fit.
        The default is [100, :data:`~numpy.inf`].
    update
        Indicates if the group should be updated with the normalization attributes.
        The default is False.

    Returns
    -------
    :
        Dictionary with the following arguments:

        - ``e0``           : absorption threshold energy :math:`E_0`.
        - ``edge_step``    : absorption edge step :math:`\Delta \mu(E_0)`.
        - ``norm``         : array with normalized :math:`\mu(E)`.
        - ``flat``         : array with flattened :math:`\mu(E)`.
        - ``pre_edge``     : fitted pre-edge polynomial.
        - ``post_edge``    : fitted post-edge polynomial.
        - ``pre_coefs``    : coefficients for the pre-edge Victoreen polynomial.
        - ``post_coefs``   : coefficients for the post-edge polynomial.
        - ``pre_edge_pars``: dictionary with pre-edge parameters.

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``energy`` does not exist in ``group``.
    IndexError
        If ``e0`` is outside the range of ``group.energy``.
    ValueError
        If ``pre_range`` contains less than two energy points.
    ValueError
        If ``post_range`` contains less than two energy points.

    Warning
    -------
    A warning will be raised if the degree of the post-edge polynomial is larger than 3.

    See also
    --------
    :func:`~araucaria.plot.fig_pre_edge.fig_pre_edge`
        Plot the results of pre-edge substraction and normalization.

    Notes
    -----
    Pre-edge substraction and normalization is performed as follows:
       
       1. The absorption threshold is determined (if ``e0`` or ``group.e0`` is not supplied).
       2. A Victoreen polymonial with energy exponent ``nvict`` is fitted to the region below 
          the edge, as specified by ``pre_range`` (2 coefficients are fitted):
          
          :math:`\mu(E) \cdot E^{nvict} = m \cdot E + b`
          
       3. A polymonial of degree ``nnorm`` is fitted to the region above the edge, as specified
          by  ``post_range`` (``nnorm`` + 1 coefficients are fitted).
       4. The edge step is deterimned by extrapolating both curves to `e0`.
       5. A flattetned spectrum is calculated by removing the polynomial above the edge from the
          normalized spectrum, while maintaining the offset of the polynomial at ``e0``.

    If ``update=True`` the contents of the returned dictionary will be
    included as attributes of ``group``.

    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Group
    >>> from araucaria.io import read_dnd
    >>> from araucaria.xas import pre_edge
    >>> from araucaria.utils import check_objattrs
    >>> fpath = get_testpath('dnd_testfile1.dat')
    >>> group = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
    >>> attrs = ['e0', 'edge_step', 'pre_edge', 'post_edge', 'norm', 'flat']
    >>> pre   = pre_edge(group, update=True)
    >>> check_objattrs(group, Group, attrs)
    [True, True, True, True, True, True]
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['energy'], exceptions=True)

    # storing energy and mu as indepedent arrays
    energy = group.energy
    mu     = getattr(group, group.get_mode())
    
    # assigning e0
    if e0 is not None:
        if e0 < min(energy) or e0 > max(energy):
            raise IndexError('e0 is outside the energy range.')
        else:
            e0 = energy[index_nearest(energy, e0)]
            
    elif hasattr(group, 'e0'):
        if group.e0 < min(energy) or group.e0 > max(energy):
            raise IndexError('group.e0 is outside the energy range.')
        else:
            e0 = energy[index_nearest(energy, group.e0)]
    else:
        e0 = find_e0(group, update=False)

    # storing pre_edge_pars in dict
    pre_edge_pars = {'pre_range': pre_range,
                     'post_range': post_range}
    
    # assiging values inside the energy array
    prerange  = check_xrange(pre_range, energy, refval=e0)
    postrange = check_xrange(post_range, energy, refval=e0)

    # retrieving pre-edge indices
    # 1 is added to pre_index[1] to include it during slicing
    pre_index    = [0,-1]
    pre_index[0] = index_nearest(energy, prerange[0] + e0, kind='lower')
    pre_index[1] = index_nearest(energy, prerange[1] + e0)
    
    # indices must be at least 2 values apart
    if ptp(pre_index) < 2:
        raise ValueError('energy range for pre-edge fit provides less than 2 points. consider increasing it.')
        #pre_index[1] = min(len(energy), pre_index[0] + 2)

    omu       = mu * energy**nvict
    pre_coefs = polyfit(energy[pre_index[0]:pre_index[1]], 
                        omu[pre_index[0]:pre_index[1]], 1)
    pre_edge  = polyval(pre_coefs, energy) * energy**(-nvict)
    
    # retrieving post-edge indices
    # 1 is added to post_index[1] to include it during slicing
    post_index    = [0,-1]
    post_index[0] = index_nearest(energy, postrange[0] + e0, kind='lower')
    post_index[1] = index_nearest(energy, postrange[1] + e0)

    # indices must be at least 2 values apart
    if ptp(post_index) < 2:
        raise ValueError('energy range for post-edge fit provides less than 2 points. consider increasing it')
        #post_index[1] = min(len(energy), post_index[0] + 2)

    if nnorm is None:
        nnorm = 2
    elif nnorm > 3:
        warn('polynomial degree for post-edge curve is %s. please verify your results.' % nnorm)

    # post-edge fit
    post_mu    = mu[post_index[0]:post_index[1]]
    post_coefs = polyfit(energy[post_index[0]:post_index[1]], post_mu, nnorm)
    post_edge  = polyval(post_coefs, energy)

    # edge_step
    ie0       = index_nearest(energy, e0)
    edge_step = post_edge[ie0] - pre_edge[ie0]

    # normalized mu
    norm = (mu - pre_edge) / edge_step

    # flattened mu
    flat       = ( (mu - post_edge) / edge_step + 1.0)
    flat[:ie0] = norm[:ie0]

    # output dictionaries
    pre_edge_pars.update({'nvict': nvict, 'nnorm': nnorm})

    content = {'e0'           : e0,
               'edge_step'    : edge_step,
               'norm'         : norm,
               'flat'         : flat,
               'pre_edge'     : pre_edge,
               'post_edge'    : post_edge,
               'pre_coefs'    : pre_coefs,
               'post_coefs'   : post_coefs,
               'pre_edge_pars': pre_edge_pars,
               }

    if update:
        group.add_content(content)

    return content

if __name__ == '__main__':
    import doctest
    doctest.testmod()