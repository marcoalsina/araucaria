#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Principal component analysis is an exploratory data analysis technique
aimed at reducing the dimensionality of a dataset while preserving most of
its variance.
Dimensionality reduccion is achieved by changing the basis of the dataset 
in such way that the new vectors constitute an orthonormal basis.

The :mod:`~araucaria.stats.pca` module offers the following 
classes and functions to perform principal component analysis:

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
"""
from __future__ import annotations
from typing import Tuple, List
from numpy import ndarray, inf, square, diag, sqrt, dot, cumsum, searchsorted
from scipy.linalg import svd
from .. import Collection, Dataset
from ..xas.xasutils import get_mapped_data

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
    ncomps : :class:`str`
        Number of components to preserve from the PCA.
        Default is None, which will preserve all components.
    cumvar : :class:`float`
        Cumulative variance to preserve from the PCA.
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
        m by n array from the SVD.
    s : :class:`~numpy.ndarray`
        Array with the n eigenvalues from the SVD.
    Vh : :class:`~numpy.ndarray`
        n by n array from the SVD
    variance : :class:`~numpy.ndarray`
        Array with the explained variance of each component.
    components: :class:`~numpy.ndarray`
        Array with principal components.

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
        var = square(self.s)
        var = var/sum(var)
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

    def inverse_transform(self, p):
        """Converts principal components into observations.
        """
        n = self.ncomps
        return dot( self.U[:,:n], p.T )

    def transform(self, obs):
        """Projects observations into principal components.
        """
        n = self.ncomps
        return dot( self.U[:, :n].T, obs )

    #def vars_to_pc(self, x):
    #    """Projects an array of variables in the principal components.
    #    """
    #    n = self.ncomps
    #    return self.s[:n] * dot( self.Vh[:n], x.T ).T

    #def pc_to_vars(self, p):
    #    """Converts an array of principal components into variables.
    #    """
    #    n = self.ncomps
    #    sinv = array([ 1/s if s > self.s[0] * 1e-6  else 0 for s in self.s ])
    #    return dot(self.Vh[:n].T, (sinv[:n] * p).T ).T
    #def obs(self, x):
    #    return self.inverse_transform(self.vars_to_pc(x))

    #def vars(self, obs):
    #    return self.pc_to_vars(self.transform(obs))

def pca(collection: Collection, taglist: List[str]=['all'],
        pca_region: str='xanes', pca_range: list=[-inf,inf],
        ncomps: int=None, cumvar: float=None, kweight: int=2, 
        pre_edge_kws: dict=None, autobk_kws: dict=None) -> PCAModel:
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
    pre_edge_kws
        Dictionary with parameters for :func:`~araucaria.xas.normalize.pre_edge`.
        The default is None, indicating that this step will be skipped.
    autobk_kws
        Dictionary with parameters :func:`~araucaria.xas.autobk.autobk`.
        Only valid for ``cluster_region='exafs'``.
        The default is None, indicating that this step will be skipped.

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
    :class:`PCAModel` : class to store results from PCA.
    :func:`~araucaria.plot.fig_pca.fig_pca` : Plots the results of PCA.

    Important
    ---------
    - Group datasets in ``collection`` will be centered before performing PCA.
    - Either ``ncomps`` or ``cumvar`` can be set to reduce dimensionallity
      of the dataset.
      If ``ncomps`` is provided, it will set precedence over ``cumvar``.
    - If given, ``pre_edge_kws`` or ``autobk_kws`` will only be used to 
      perform PCA. Results from normalization and background removal 
      will not be written in ``collection``.

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_collection_hdf5
    >>> from araucaria.stats import PCAModel, pca
    >>> from araucaria.utils import check_objattrs
    >>> fpath      = get_testpath('Fe_database.h5')
    >>> collection = read_collection_hdf5(fpath)
    >>> out        = pca(collection, pca_region='xanes', pre_edge_kws={})
    >>> attrs      = ['energy', 'matrix', 'components', 'variance', 'groupnames', 'pca_pars']
    >>> check_objattrs(out, PCAModel, attrs)
    [True, True, True, True, True, True]
    """
    # mapped data
    xvals, matrix = get_mapped_data(collection, taglist=taglist, region=pca_region, 
                                    range=pca_range, kweight=kweight,
                                    pre_edge_kws=pre_edge_kws, autobk_kws=autobk_kws)
    # centering data
    matrix = matrix - matrix.mean(axis=0)

    # singular value decomposition
    pca    = PCAModel(matrix, ncomps, cumvar)

    # storing pca parameters
    pca_pars = {'pca_region': pca_region, 
                'pca_range' : pca_range,}

    # additional pca parameters
    if pca_region == 'exafs':
        xvar  = 'k'    # x-variable
        pca_pars['kweight'] = kweight
        if pre_edge_kws is None:
            pca_pars['pre_edge_kws'] = None
        else:
            pca_pars['pre_edge_kws'] = pre_edge_kws
        if autobk_kws is None:
            pca_pars['autobk_kws'] = None
        else:
            pca_pars['autobk_kws'] = autobk_kws
    # xanes/dxanes pca
    else:
        xvar  = 'energy'    # x-variable
        if pre_edge_kws is None:
            pca_pars['pre_edge_kws'] = None
        else:
            pca_pars['pre_edge_kws'] = pre_edge_kws

    # storing pca results
    setattr(pca,xvar, xvals)
    pca.groupnames = collection.get_names(taglist=taglist)
    pca.pca_pars   = pca_pars

    return pca

if __name__ == '__main__':
    import doctest
    doctest.testmod()