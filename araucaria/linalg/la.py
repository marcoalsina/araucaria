#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import List, Union
from numpy import inf
from numpy.linalg import cond
from scipy.linalg.interpolative import estimate_rank, interp_decomp
from .. import Dataset, Collection
from ..xas.xasutils import get_mapped_data

def cond_num(collection: Collection, taglist: List[str]=['all'],
             region: str='xanes', range: list=[-inf,inf], 
             ord: Union[int, str]=None, kweight: int=2) -> Dataset:
    """Computes the condition number of a collection.

    Parameters
    ----------
    collection
        Collection with the groups to compute the condition number.
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].
    region
        XAFS region to compute the condition number. Accepted values are 'dxanes',
        'xanes', or 'exafs'. The default is 'xanes'.
    range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    ord
        Order of the norm. See :func:`~numpy.linalg.cond` for details.
        The default is None, which equals to 2-norm.
    kweight
        Exponent for weighting chi(k) by k^kweight.
        Only valid for ``region='exafs'``.
        The default is 2.

    Returns
    -------
    :
        Dataset with the following arguments:

        - ``cn``           : condition number.
        - ``groupnames``   : list with names of groups.
        - ``energy``       : array with energy values. Returned only if
          ``region='xanes`` or ``region=dxanes``.
        - ``k``            : array with wavenumber values. Returned only if
          ``region='exafs'``.
        - ``matrix``       : array with observed values for groups in ``range``.
        - ``cond_pars``    : dictionary with parameters of calculation.

    Notes
    -----
    The condition number measures the amplification of the output 
    as a function of a small change in the input parameters. 
    For a linear system :math:`Ax=b`, the condition number :math:`\kappa(A)` 
    is formally defined as:

    .. math::

        \kappa(A) = ||A^{-1}|| \\text{ } ||A||

    The interpretation of the condition number can be established with the 
    following inequality [1]_:

    .. math::
        
        \\frac{|| \delta x ||}{||x||} \le \kappa(A) \\frac{|| \delta b||}{||b||}

    Therefore, if :math:`\kappa(A) = 10^k` then one expects to lose 
    :math:`k` digits of precision in :math:`x` when solving the linear system.
    Linear systems with large :math:`\kappa(A)` are said to be **ill-conditioned**,
    and the obtained results can exhibit large errors.

    References
    ----------
    .. [1] Cheney, W. and Kincaid, D. (2008) "Numerical Mathematics and Computing",
       6th Edition, pp. 321.

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Dataset
    >>> from araucaria.xas import pre_edge, autobk
    >>> from araucaria.linalg import cond_num
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> collection.apply(pre_edge)
    >>> out        = cond_num(collection, region='xanes')
    >>> attrs      = ['groupnames', 'energy', 'matrix', 'cond', 'cond_pars']
    >>> check_objattrs(out, Dataset, attrs)
    [True, True, True, True, True]
    >>> print('%1.3f' % out.cond)
    241.808

    >>> # condition number of exafs
    >>> collection.apply(autobk)
    >>> out   = cond_num(collection, region='exafs', range=[0,10])
    >>> attrs = ['groupnames', 'k', 'matrix', 'cond', 'cond_pars']
    >>> check_objattrs(out, Dataset, attrs)
    [True, True, True, True, True]
    >>> print('%1.3f' % out.cond)
    7.019
    """
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=region, 
                                    range=range, kweight=kweight)

    # condtion number
    cn = cond(matrix, p=ord)

    # storing parameters
    cond_pars = {'region' : region, 
                 'range'  : range,
                 'ord'    : ord,}

    # additional parameters
    if region == 'exafs':
        xvar  = 'k'    # x-variable
        cond_pars['kweight'] = kweight
    else:
        # xanes/dxanes clustering
        xvar  = 'energy'    # x-variable

    # storing cluster results
    content = {'groupnames'   : collection.get_names(taglist=taglist),
               xvar           : xvals,
               'matrix'       : matrix,
               'cond'         : cn,
               'cond_pars'    : cond_pars,}

    out = Dataset(**content)
    return out

def imd(collection: Collection, taglist: List[str]=['all'],
                region: str='xanes', range: list=[-inf,inf], 
                eps: float=1e-4, kweight: int=2) -> Dataset:
    """Computes the interpolative matrix decomposition for a collection.

    Parameters
    ----------
    collection
        Collection with the groups to compute the rank.
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].
    region
        XAFS region to compute the rank. Accepted values are 'dxanes',
        'xanes', or 'exafs'. The default is 'xanes'.
    range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    eps
        Relative error of approximation.
        The default is 1e-4.
    kweight
        Exponent for weighting chi(k) by k^kweight.
        Only valid for ``region='exafs'``.
        The default is 2.

    Returns
    -------
    :
        Dataset with the following arguments:

        - ``rank``         : matrix rank.
        - ``idx``          : column index array.
        - ``proj``         : interpolation coefficients.
        - ``groupnames``   : list with names of groups.
        - ``energy``       : array with energy values. Returned only if
          ``region='xanes`` or ``region=dxanes``.
        - ``k``            : array with wavenumber values. Returned only if
          ``region='exafs'``.
        - ``matrix``       : array with observed values for groups in ``range``.
        - ``id_pars``      : dictionary with parameters of calculation.

    Notes
    -----
    The interpolative decomposition (ID) of a :math:`m` by :math:`n` 
    matrix :math:`A` of rank :math:`k < min\{m,n\}` is a factorization 
    of the following form,

    .. math::

       A\Pi = [A\Pi_1 \quad A\Pi_2 ] = A \Pi_1 [I \quad T]
    
    where :math:`\Pi = [\Pi_1 \quad \Pi_2]` is a permutation matrix, 
    with :math:`\Pi_1` being a :math:`k` by :math:`n` matrix.
    
    The latter can be equivalently written as :math:`A = BP`, where :math:`B`
    is known as the *skeleton matrix*, while :math:`P` is known as the 
    *interpolation matrix*.

    Therefore, the original matrix can be factorized into an skeleton matrix that
    preserves the original matrix rank, while the rest of the original matrix can 
    be reconstructed through the interpolation matrix.

    Computation of the interpolative decompsition method is performed with the
    :func:`~scipy.linalg.interpolative.interp_decomp` function of ``scipy``, which is
    based on the ID software package [2]_. Therefore, the *skeleton matrix* can be 
    computed as follows::
    
        B = A[:,idx[:rank]]

    while the *interpolation matrix* can be computed as follows::
    
        P = numpy.hstack([numpy.eye(rank), proj])[:, np.argsort(idx)]

    References
    ----------
     .. [2] P.G. Martinsson, V. Rokhlin, Y. Shkolnisky, M. Tygert. 
        “ID: a software package for low-rank approximation of matrices via interpolative 
        decompositions, version 0.2.” http://tygert.com/id_doc.4.pdf.

    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Dataset
    >>> from araucaria.xas import pre_edge, autobk
    >>> from araucaria.linalg import imd
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> collection.apply(pre_edge)
    >>> out        = imd(collection, region='xanes')
    >>> attrs      = ['groupnames', 'energy', 'matrix', 'rank', 'idx', 'proj', 'id_pars']
    >>> check_objattrs(out, Dataset, attrs)
    [True, True, True, True, True, True, True]
    >>> print(out.idx)
    [0 1 3 2]
    """
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=region, 
                                    range=range, kweight=kweight)

    # condtion number
    rnk, idx, proj = interp_decomp(matrix, eps_or_k=eps)

    # storing parameters
    id_pars = {'region' : region, 
               'range'  : range,
               'eps'    : eps,}

    # additional parameters
    if region == 'exafs':
        xvar  = 'k'    # x-variable
        rank_pars['kweight'] = kweight
    else:
        # xanes/dxanes clustering
        xvar  = 'energy'    # x-variable

    # storing cluster results
    content = {'groupnames' : collection.get_names(taglist=taglist),
               xvar         : xvals,
               'matrix'     : matrix,
               'rank'       : rnk,
               'idx'        : idx,
               'proj'       : proj,
               'id_pars'    : id_pars,}

    out = Dataset(**content)
    return out

def matrix_rank(collection: Collection, taglist: List[str]=['all'],
                region: str='xanes', range: list=[-inf,inf], 
                eps: float=1e-4, kweight: int=2) -> Dataset:
    """Computes the matrix rank of a collection.

    Parameters
    ----------
    collection
        Collection with the groups to compute the rank.
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].
    region
        XAFS region to compute the rank. Accepted values are 'dxanes',
        'xanes', or 'exafs'. The default is 'xanes'.
    range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    eps
        Relative error for computation of rank.
        The default is 1e-4.
    kweight
        Exponent for weighting chi(k) by k^kweight.
        Only valid for ``region='exafs'``.
        The default is 2.

    Returns
    -------
    :
        Dataset with the following arguments:

        - ``rank``         : matrix rank.
        - ``groupnames``   : list with names of groups.
        - ``energy``       : array with energy values. Returned only if
          ``region='xanes`` or ``region=dxanes``.
        - ``k``            : array with wavenumber values. Returned only if
          ``region='exafs'``.
        - ``matrix``       : array with observed values for groups in ``range``.
        - ``rank_pars``    : dictionary with parameters of calculation.

    Notes
    -----
    Computation of matrix rank is based on the interpolative decompsition method.
    See :func:`~scipy.linalg.interpolative.estimate_rank` for further details.

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Dataset
    >>> from araucaria.xas import pre_edge, autobk
    >>> from araucaria.linalg import matrix_rank
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> collection.apply(pre_edge)
    >>> out        = matrix_rank(collection, region='xanes')
    >>> attrs      = ['groupnames', 'energy', 'matrix', 'rank', 'rank_pars']
    >>> check_objattrs(out, Dataset, attrs)
    [True, True, True, True, True]
    >>> print(out.rank)
    4

    >>> # rank of exafs matrix
    >>> collection.apply(autobk)
    >>> out   = matrix_rank(collection, region='exafs', range=[0,10])
    >>> attrs = ['groupnames', 'k', 'matrix', 'rank', 'rank_pars']
    >>> check_objattrs(out, Dataset, attrs)
    [True, True, True, True, True]
    >>> print(out.rank)
    4
    """
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=region, 
                                    range=range, kweight=kweight)

    # condtion number
    rnk = estimate_rank(matrix, eps=eps)

    # storing parameters
    rank_pars = {'region' : region, 
                 'range'  : range,
                 'eps'    : eps,}

    # additional parameters
    if region == 'exafs':
        xvar  = 'k'    # x-variable
        rank_pars['kweight'] = kweight
    else:
        # xanes/dxanes clustering
        xvar  = 'energy'    # x-variable

    # storing cluster results
    content = {'groupnames'   : collection.get_names(taglist=taglist),
               xvar           : xvals,
               'matrix'       : matrix,
               'rank'         : rnk,
               'rank_pars'    : rank_pars,}

    out = Dataset(**content)
    return out
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()