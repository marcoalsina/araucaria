#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Principal component analysis (PCA) is an exploratory data analysis technique
that allows to reduce the dimensionality of a dataset while preserving most of
its variance.
Dimensionality reduction is achieved by changing the basis of the dataset 
in such way that the new vectors constitute an orthonormal basis.

Lets consider a m by n matrix :math:`X`  (m observations and n variables).
Mathematically, PCA computes the eigenvalues and eigenvectors of the covariance
matrix. If observations in :math:`X` are centered, then the covariance is proportional
to :math:`X^TX`:

.. math::

    eigen(covX) \\propto eigen(X^TX)

Following the eigen-decomposition, we can rewrite :math:`X^TX` as follows:

.. math::

    X^TX = WDW^{-1}

Where

- :math:`W`   : matrix of eigenvectors.
- :math:`D`   : diagonal matrix of eigenvalues.

Multiplying the matrix :math:`X` by the matrix of eigenvectors :math:`W` effectively
projects the data into an orthonormal basis: :math:`T=XW` are known as the
**principal components**.

Although PCA can be computed directly as the eigenvalues of the covariance matrix, it
is often convenient to compute it through singular value decomposition (SVD) of :math:`X`:

.. math::

    X = U \\Sigma V^*

Where:

- :math:`U`: m by n semi-unitary matrix of left-singular vectors.
- :math:`\\Sigma`: n by n diagonal matrix of singular values.
- :math:`V`: n by n semi-unitary matrix of right-singular vectors.

It is straightforward to observe the following relation between the eigenvalue decomposition 
of the covariance matrix and the SVD of :math:`X`:

.. math::

    X^TX &= V \\Sigma^T \\Sigma V^* = V \\hat{\\Sigma}^2 V^*
    
    X^TX &= WDW^{-1} = V \\hat{\\Sigma}^2 V^*

Where

- :math:`W` : matrix of eigenvectors.
- :math:`\\hat{\\Sigma^2} = \\Sigma^T \\Sigma`: diagonal matrix of eigenvalues.

Therefore, the principal components can be computed as :math:`T=XV=U \\Sigma`.

Dimensionality reduction
------------------------

A truncated matrix :math:`T_L` can be computed by retaining the L-largest 
singular values and their corresponding singular vectors:

.. math::

    T_L = U_L \\Sigma_L = X V_L

After reduction a new vector :math:`x` of observations can be projected into 
the principal components:

.. math::

    t = U_L^T x

Target transformation
---------------------

These principal components can be transformed back into the original space:

.. math::

    \hat{x} = U_L t = U_L U_L^T x

The latter is commonly referred to as **target transformation**.

The :mod:`~araucaria.stats.pca` module offers the following 
classes and functions to perform principal component analysis based on the SVD
approach:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Class
     - Description
   * - :class:`PCAModel`
     - Container of results from PCA.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`pca`
     - Performs principal component analysis on a collection.
   * - :func:`target_transform`
     - Performs target transformation on a collection.
"""
from __future__ import annotations
from typing import Tuple, List
from numpy import (ndarray, inf, square, diag, sqrt, 
                   dot, cumsum, sum, divide, searchsorted)
from scipy.linalg import svd
from .. import Collection, Dataset
from ..xas.xasutils import get_mapped_data
from ..utils import check_objattrs

class PCAModel(Dataset):
    """PCA Model class.
    
    "This class stores the results of principal component analysis
    on an array.
    
    Parameters
    ----------
    matrix : :class:`~numpy.ndarray`
        m by n array containing m observations and n variables.
    name : :class:`str`
        Name for the collection. The default is None.
    ncomps : :class:`int`
        Number of components to preserve.
        Default is None, which will preserve all components.
    cumvar : :class:`float`
        Cumulative variance to preserve.
        Default is None, which will preserve all components.

    Raises
    ------
    AssertionError
        If ``ncomps`` is not a positive integer.
    AssertionError
        If ``ncomps`` is larger than the number of variables in ``matrix``.
    AssertionError
        If ``cumvar`` is negative or greater than 1.

    Attributes
    ----------
    matrix : :class:`~numpy.ndarray`
        Array containing m observations in rows and n variables in columns.
    U : :class:`~numpy.ndarray`
        m by n array with the semi-unitary matrix of left-singular vectors.
    s : :class:`~numpy.ndarray`
        1-D array with the n singular values.
    Vh : :class:`~numpy.ndarray`
        n by n array with the transpose of the semi-unitary matrix of right-singular vectors.
    variance : :class:`~numpy.ndarray`
        Array with the explained variance of each component.
    components: :class:`~numpy.ndarray`
        m by n array with principal components.

    Notes
    -----
    The following methods are currently implemented:

    .. list-table::
       :widths: auto
       :header-rows: 1

       * - Method
         - Description
       * - :func:`transform`
         - Projects observations into principal components.
       * - :func:`inverse_transform`
         - Converts principal component values into observations.

    Important
    ---------
    - Observations in ``matrix`` must be centered in order to perform PCA.
    - Either ``ncomps`` or ``cumvar`` can be set to reduce dimensionallity
      of the dataset.
      If ``ncomps`` is provided, it will set precedence over ``cumvar``.

    Example
    -------
    >>> from numpy.random import randn
    >>> from araucaria.stats import PCAModel
    >>> from araucaria.utils import check_objattrs
    >>> matrix = randn(10,10)
    >>> model  = PCAModel(matrix)
    >>> type(model)
    <class 'araucaria.stats.pca.PCAModel'>
    
    >>> # verifying attributes
    >>> attrs = ['matrix', 'components', 'variance']
    >>> check_objattrs(model, PCAModel, attrs)
    [True, True, True]
    """
    def __init__(self, matrix: ndarray, name: str=None,
                 ncomps: int=None, cumvar: float=None):
        if name is None:
            name  = hex(id(self))
        self.name   = name
        self.matrix = matrix

        # number of rows and cols
        m, n = matrix.shape

        # singular value decomposition
        self.U, self.s, self.Vh = svd(matrix, full_matrices=False)

        # variance
        var  = square(self.s)
        var  = var/sum(var)
        cvar = cumsum(var)

        # asserting ncomps and cumvar
        if ncomps is not None:
            assert isinstance(ncomps, int), "ncomps must be an integer value."
            assert 1 < ncomps <= n, "comps cannot be larger than the number of variables or smaller than 1."
            self.ncomps = ncomps
        elif cumvar is not None:
            assert 0 < cumvar <= 1, "cumvar must be larger than 0 and smaller than 1."
            self.ncomps = searchsorted(cvar, cumvar) + 1
        else:
            self.ncomps = n

        # variance and principal components
        self.variance   = var[:ncomps]
        self.components = (self.U * self.s)[:, :ncomps]

    def __repr__(self):
        if self.name is not None:
            return '<PCAModel %s>' % self.name
        else:
            return '<PCAModel>'

    def transform(self, obs: ndarray) -> ndarray:
        """Projects observations into principal components.
        
        Parameters
        ----------
        obs
            Array with observed values.

        Returns
        -------
        :
            Array with scores on principal components.
        """
        n = self.ncomps
        return dot( self.U[:, :n].T, obs )

    def inverse_transform(self, p: ndarray) -> ndarray:
        """Converts principal components into observations.
        
        Parameters
        ----------
        p
            Array with scores on principal components.
        
        Returns
        -------
        :
            Array with observed values.
        """
        n = self.ncomps
        return dot( self.U[:,:n], p)


def pca(collection: Collection, taglist: List[str]=['all'],
        pca_region: str='xanes', pca_range: list=[-inf,inf],
        ncomps: int=None, cumvar: float=None, 
        kweight: int=2) -> PCAModel:
    """Performs principal component analysis (PCA) on a collection.

    Parameters
    ----------
    collection
        Collection with the groups for PCA.
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].
    pca_region
        XAFS region to perform PCA. Accepted values are 'dxanes',
        'xanes', or 'exafs'. The default is 'xanes'.
    pca_range
        Domain range in absolute values. Energy units are expected
        for 'dxanes' or 'xanes', while wavenumber (k) units are expected 
        for 'exafs'.
        The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    ncomps
        Number of components to preserve from the PCA.
        Default is None, which will preserve all components.
    cumvar
        Cumulative variance to preserve from the PCA.
        Defaults is None, which will preserve all components.
    kweight
        Exponent for weighting chi(k) by k^kweight.
        Only valid for ``cluster_region='exafs'``.
        The default is 2.

    Returns
    -------
    :
        PCA model with the following arguments:

        - ``components``   : array with principal components.
        - ``variance``     : array with explained variance of each component.
        - ``groupnames``   : list with names of clustered groups.
        - ``energy``       : array with energy values. Returned only if
          ``pca_region='xanes`` or ``pca_region=dxanes``.
        - ``k``            : array with wavenumber values. Returned only if
          ``pca_region='exafs'``.
        - ``matrix``       : array with centered values for groups in ``pca_range``.
        - ``pca_pars`` : dictionary with PCA parameters.

    See also
    --------
    :class:`PCAModel` : Class to store results from principal component analysis.
    :func:`~araucaria.plot.fig_pca.fig_pca` : Plots the results of principal component analysis.

    Important
    ---------
    - Group datasets in ``collection`` will be centered before performing PCA.
    - Either ``ncomps`` or ``cumvar`` can be set to reduce dimensionallity
      of the dataset.
      If ``ncomps`` is provided, it will set precedence over ``cumvar``.

    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.xas import pre_edge
    >>> from araucaria.stats import PCAModel, pca
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> collection.apply(pre_edge)
    >>> out        = pca(collection, pca_region='xanes')
    >>> attrs      = ['energy', 'matrix', 'components', 'variance', 'groupnames', 'pca_pars']
    >>> check_objattrs(out, PCAModel, attrs)
    [True, True, True, True, True, True]
    """
    # mapped data
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=pca_region, 
                                    range=pca_range, kweight=kweight)
    # centering data
    matrix = matrix - matrix.mean(axis=0)

    # singular value decomposition
    pca    = PCAModel(matrix, ncomps=ncomps, cumvar=cumvar)

    # storing pca parameters
    pca_pars = {'pca_region': pca_region, 
                'pca_range' : pca_range,}

    # additional pca parameters
    if pca_region == 'exafs':
        xvar  = 'k'    # x-variable
        pca_pars['kweight'] = kweight
    # xanes/dxanes pca
    else:
        xvar  = 'energy'    # x-variable

    # storing pca results
    setattr(pca,xvar, xvals)
    pca.groupnames = collection.get_names(taglist=taglist)
    pca.pca_pars   = pca_pars
    return pca

def target_transform(model: PCAModel, collection: Collection,
                     taglist: List[str]=['all']) -> Dataset:
    """Performs target transformation on a collection.
    
    Parameters
    ----------
    model
        PCA model to perform the projection and inverse transformation.
    collection
        Collection with the groups for target transformatino.
    taglist
        List with keys to filter groups based on their ``tags``
        attributes in the Collection.
        The default is ['all'].

    Returns
    -------
    :
        Dataset with the following attributes.

        - ``groupnames``: list with names of transformed groups.
        - ``energy``       : array with energy values. Returned only if
          ``pca_region='xanes`` or ``pca_region=dxanes``.
        - ``k``            : array with wavenumber values. Returned only if
          ``pca_region='exafs'``.
        - ``matrix``    : original array with mapped values.
        - ``tmatrix``   : array with target transformed groups.
        - ``scores``    : array with scores in the principal component basis.
        - ``chi2``      : :math:`\\chi^2` values of the target tranformed groups.

    Raises
    ------
    TypeError
        If ``model`` is not a valid PCAModel instance
    KeyError
        If attributes from :func:`~araucaria.stats.pca.pca`
        do not exist in ``model``.

    See also
    --------
    :func:`~araucaria.stats.pca.pca` : Performs principal component analysis on a collection.
    :func:`~araucaria.plot.fig_pca.fig_target_transform` : Plots the results of target transformation.

    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Dataset
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.xas import pre_edge
    >>> from araucaria.stats import pca, target_transform
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> collection.apply(pre_edge)
    >>> model      = pca(collection, pca_region='xanes', cumvar=0.9)
    >>> data       = target_transform(model, collection)
    >>> attrs      = ['groupnames', 'tmatrix', 'chi2', 'scores', 'energy']
    >>> check_objattrs(data, Dataset, attrs)
    [True, True, True, True, True]
    """
    check_objattrs(model, PCAModel, attrlist=['groupnames', 'matrix', 
                   'variance', 'pca_pars'], exceptions=True)

    # retrieving pca parameters
    pca_region   = model.pca_pars['pca_region']
    pca_range    = model.pca_pars['pca_range']

    # setting panels based on pca region
    region = (model.pca_pars['pca_region'])
    if region == 'exafs':
        xvar    = 'k'
        kweight = model.pca_pars['kweight']
    else:
        xvar    = 'energy'
        kweight = 2

    # mapped data for collection
    domain = getattr(model, xvar)
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=pca_region, 
                                    domain=domain, kweight=kweight)

    # centering data
    matrix = matrix - matrix.mean(axis=0)

    # target transformation
    scores  = model.transform(matrix)
    tmatrix = model.inverse_transform(scores)
    chi2    = sum(divide( (matrix-tmatrix)**2, matrix), axis=0)

    # storing target transformation results
    content = {'groupnames' : collection.get_names(taglist=taglist),
               xvar         : domain,
               'matrix'     : matrix,
               'tmatrix'    : tmatrix,
               'scores'     : scores,
               'chi2'       : chi2}

    # dataset class
    out = Dataset(**content)
    return out

if __name__ == '__main__':
    import doctest
    doctest.testmod()