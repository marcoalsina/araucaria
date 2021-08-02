#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Some explanation goes here.

.. math::

    k = \sqrt{ k_{\\textrm{feff}}^2 - {2m_e E_0}/{\hbar^2} }

.. math::

    p = p' + i p'' = \sqrt{ \\big[ p_{\\textrm{real}}(k) - 
    i / \lambda(k) \\big]^2 - i \, 2 m_e E_i /{\hbar^2} }

.. math::

    \\begin{align}
        \chi(k) = &\\textrm{Im} \Big[ \\frac{f(k)NS_0^2}{k(R_{\\textrm{eff}} + \\Delta R)^2} \, 
        \\textrm{exp}(-2p''R_{\\textrm{eff}} - \\frac{2}{3}p^4c_4) \\\\
        &\\textrm{exp}(\,i\,\\{ 2kR_{\\textrm{eff}} + \delta(k) + 2p( \Delta R - 
        2 \sigma^2/R_{\\textrm{eff}}) \\frac{4}{3}p^3 c_3 \\}\,) \Big]
    \\end{align}
    
References
----------

.. [1] J.J. Kas et al. (2020) The FEFF Code. In International Tables of Crystallography, 
   Volume I. X-Ray Absorption Spectroscopy and Related Techniques, https://doi.org/10.1107/S1574870720003274.

.. [2] M. Newville (2004) EXAFS analysis using FEFF and FEFFIT, 
   Journal of Synchrotron Radiation 8(2): 96-100,
   https://doi.org/10.1107/S0909049500016290.

.. [3] H. Funke et al. (2207) A new FEFF-based wavelet for EXAFS data analysis, 
   Journal of Synchrotron Radiation 14(5): 426-432,
   https://doi.org/10.1107/S0909049507031901.
"""
from typing import Tuple
from os.path import isfile, basename
from pathlib import Path
from numpy import ndarray, array, arange, sqrt, where, exp
from lmfit import Parameters
from ..utils import check_objattrs, check_dictkeys, interp_yvals
from ..xas.xasutils import ktoe, etok

class FeffPath(object):
    """Feffpath storage class.

    This class stores and manipulates data from a Feff path 
    file calculation (FEFFNNNN.dat) [4]_.

    Parameters
    ----------
    name : :class:`str`
        Name for the Feffpath. The default is None.
    fpath: :class:`~pathlib.Path`
        Path to Feffdat data file. The default is None.

    Attributes
    ----------
    path_pars : :class:`dict`
        Dictionary with Feff path parameters:
        
        - ``filename``: name of file.
        - ``nlegs``   : number of path legs.
        - ``degen``   : path degeneracy.
        - ``reff``    : nominal path length.
        - ``rnrmav``  : norman radius (bohr).
        - ``edge``    : relative energy threshold (eV).

    feffdat : :class:`dict`
        Dictionary with Feff scattering data (static arrays):

        - ``k``          : photoelectron wavenumber in eV (:class:`ndarray`).
        - ``real_phc``   : central atom phase shift (:class:`ndarray`).
        - ``mag_feff``   : feff magnitude (:class:`ndarray`).
        - ``phase_feff`` : scattering phase shift (:class:`ndarray`).
        - ``red_factor`` : amplitude reduction factor (:class:`ndarray`).
        - ``lambd``      : mean free path (:class:`ndarray`).
        - ``real_p``     : real part of the complex wavenumber

    geom : :class:`list`
        List with Feff path geometry parameters: x, y, z, ipot, atnum, atsym.

    Notes
    -----
    The following methods are currently implemented:

    .. list-table::
       :widths: auto
       :header-rows: 1

       * - Method
         - Description
       * - :func:`read_feffdat`
         - Reads a Feff data file.
       * - :func:`get_chi`
         - Returns chi(k) for the Feff path.

    References
    ----------
    .. [4] S. I. Zabinsky et al. (2002) "Chapter 6: Ouput files", 
       Documentation Feff 6L Version 6.01l, University of Washington. 
       https://github.com/newville/ifeffit/blob/master/src/feff6/DOC/feff6L.doc.

    Example
    -------
    >>> from araucaria.fit import FeffPath
    >>> feffpath = FeffPath()
    >>> type(feffpath)
    <class 'araucaria.fit.feffit.FeffPath'>
    """
    def __init__(self, fpath:Path=None, name: str=None):
        if fpath is not None:
            self.read_feffdat(fpath)
        if name is None:
            name  = hex(id(self))
        self.name = name

    def __repr__(self):
        if self.name is not None:
            return '<Feffpath %s>' % self.name
        else:
            return '<Feffpath>'

    def read_feffdat(self, fpath: Path) -> None:
        """Reads a Feff data file.

        Parameters
        ----------
        fpath
            Path to Feffpath data file.

        Returns
        -------
        :

        Raises
        ------
        IOError
            If the file does not exist in the specified path.

        Example
        -------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.fit import FeffPath
        >>> from araucaria.utils import check_objattrs
        >>> fpath    = get_testpath('feff0001.dat')
        >>> attrlist = ['geom', 'path_pars', 'feffdat']
        >>> feffpath = FeffPath()
        >>> # empty FeffPath instance
        >>> check_objattrs(feffpath, FeffPath, attrlist)
        [False, False, False]
        >>> # reading feffdat file
        >>> feffpath.read_feffdat(fpath)
        >>> check_objattrs(feffpath, FeffPath, attrlist)
        [True, True, True]
        

        """
        # veifying existence of path
        if not isfile(fpath):
            raise IOError("file %s does not exists." % fpath)

        # containers for data
        self.path_pars = {} # path parameters
        self.feffdat   = {} # scattering functions
        self.geom      = [] # [x y z] pot atnum atsym

        # saving filename``
        self.path_pars['filename'] = basename(fpath)

        # reading file
        with open(fpath, 'r') as file:
            # skipping header data
            for line in file:
                if '----' in line:
                    break

            # saving path parameters
            row = next(file).strip().split()
            self.path_pars['nleg']   = int(row[0])   # number of legs
            self.path_pars['degen']  = float(row[1]) # degeneracy
            self.path_pars['reff']   = float(row[2]) # reff
            self.path_pars['rnrmav'] = float(row[3]) # rnrmav (bohr)
            self.path_pars['edge']   = float(row[4]) # edge
            next(file)

            # saving geometry parameters
            for line in file:
                line = line.strip().lstrip('#')
                if line.startswith('k') and line.endswith('real[p]@#'):
                    break
                row = line.split()
                loc = array(row[:3], dtype=float)
                ipot, z, elem = int(row[3]), int(row[4]), row[5]
                self.geom.append(list(loc) + [ipot, z, elem])

            # reading scattering values
            data  = []
            for line in file:
                data.append([float(val) for val in line.strip().split()])

        # storing scattering values
        data                       = array(data).T
        self.feffdat['k']          = data[0]    # photoelectron wavenumber
        self.feffdat['real_phc']   = data[1]    # central atom phase shift
        self.feffdat['mag_feff']   = data[2]    # feff magnitude
        self.feffdat['phase_feff'] = data[3]    # scattering phase shift
        self.feffdat['red_factor'] = data[4]    # amplitude reduction factor
        self.feffdat['lambd']      = data[5]    # mean free path
        self.feffdat['real_p']     = data[6]    # real part of the complex wavenumber
        del data
        # locking arrays from modification
        for key in self.feffdat:
            self.feffdat[key].flags.writeable = False

    def get_chi(self, pars: Parameters=None, kstep: float=None) -> Tuple[ndarray, ndarray]:
        """Returns :math:`\chi(k)` for the Feff path.

        Parameters
        ----------
        pars
            Parameter object from ``lmfit``. It will consider the following keys:
            
            - ``s02``    : amplitude reduction factor for the path.
            - ``sigma2`` : debye-waller factor for the path.
            - ``degen``  : path degeneracy.
            - ``deltaE`` : :math:`\Delta E` for the path (eV).
            - ``deltaR`` : :math:`\Delta R` for the path (Angstrom).
            - ``ei``     : :math:`E_i` for the path.
            - ``c3``     : third cumulant parameter.
            - ``c4``     : fourth cumulant parameter.

            The default is None, which assigns the Feff path value for ``degen``,
            one for ``s02``, and zero for remaining parameters.

        kstep
            Step in k array.
            The default is None, which returns the original k-spacing of the Feff data file.

        Returns
        -------
        :
            Array with :math:`k` values.
        :
            Array with :math:`\chi(k)` values.
        
        Raises
        -------
        AttributeError
            If the self instance has no ``feffdat`` attribute.

        Example
        -------
        .. plot::
            :context: reset

            >>> from araucaria.testdata import get_testpath
            >>> from araucaria.fit import FeffPath
            >>> from araucaria.utils import check_objattrs
            >>> from araucaria.plot import fig_xas_template
            >>> import matplotlib.pyplot as plt
            >>> fpath    = get_testpath('feff0001.dat')
            >>> feffpath = FeffPath(fpath)
            >>> k, chi   = feffpath.get_chi(kstep=0.05)
            >>> kw       = 1
            >>> fig, ax  = fig_xas_template(panels='e', fig_pars={'kweight': kw})
            >>> lin      = ax.plot(k, k**kw*chi)
            >>> plt.show(block=False)
        """
        # checking self attribute
        check_objattrs(self, FeffPath, attrlist=['feffdat'], exceptions=True)

        # check pars attribute
        req_keys   = ['s02', 'sigma2', 'degen', 'deltaR', 'deltaE', 'ei', 'c3', 'c4']
        def_values = [  1.0,      0.0,     1.0,      0.0,      0.0,  0.0,  0.0, 0.0 ]

        params = {}
        if pars is None:
            for i, key in enumerate(req_keys):
                if key == 'degen':
                    params[key] = self.path_pars[key]
                else:
                    params[key] = def_values[i]
        else:
            for i, key in enumerate(req_keys):
                if key in pars:
                    params[key] = pars[key]
                else:
                    if key == 'degen':
                        params[key] = self.path_pars[key]
                    else:
                        params[key] = def_values[i]

        # computing k shifted and approximating k=0
        k_n = etok( ktoe(self.feffdat['k']) - params['deltaE'] )
        k_s = where(k_n == 0, 1e-10, k_n)

        # computing complex wavenumber
        p = sqrt( (self.feffdat['real_p'] + (1j/ self.feffdat['lambd']))**2 - 1j*etok(params['ei']))

        # computing complex chi(k)
        chi_feff  = params['degen']*params['s02']*self.feffdat['red_factor']*self.feffdat['mag_feff']
        chi_feff /= ( k_s * (self.path_pars['reff'] + params['deltaR'] )**2 )
        chi_feff  = chi_feff.astype(complex)
        chi_feff *= exp(-2 * self.path_pars['reff'] * p.imag - 2 * ( p**2 ) * params['sigma2'] \
                    + 2 * (p**4) * params['c4'] / 3)
        chi_feff *= exp(1j * (2 * k_n * self.path_pars['reff'] + self.feffdat['phase_feff'] + \
                    self.feffdat['real_phc'] + 2 * p * ( params['deltaR'] - \
                    (2 * params['sigma2'] / self.path_pars['reff'] )) - 4 * (p**3) * params['c3'] / 3))

        # retaning the imaginary part
        chi_feff = chi_feff.imag

        if kstep is not None:
            kvals = arange(k_n[0], k_n[-1]+kstep/2, kstep)
            chi_feff = interp_yvals(k_n, chi_feff, kvals)
            return kvals, chi_feff
        else:
            return k_s, chi_feff