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

The :mod:`~araucaria.fit.feffit` module offers the following 
classes and functions to perform EXAFS fitting based on Feff calculations: 

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Class
     - Description
   * - :class:`FeffPath`
     - Feffpath storage class.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`fefftochi`
     - Returns chi(k) from a list of Feff paths.

References
----------

.. [1] J.J. Kas et al. (2020) The FEFF Code. In International Tables of Crystallography, 
   Volume I. X-Ray Absorption Spectroscopy and Related Techniques, 
   https://doi.org/10.1107/S1574870720003274.

.. [2] M. Newville (2004) EXAFS analysis using FEFF and FEFFIT, 
   Journal of Synchrotron Radiation 8(2): 96-100,
   https://doi.org/10.1107/S0909049500016290.

.. [3] H. Funke et al. (2207) A new FEFF-based wavelet for EXAFS data analysis, 
   Journal of Synchrotron Radiation 14(5): 426-432,
   https://doi.org/10.1107/S0909049507031901.
"""
from typing import Tuple, List
from os.path import isfile, basename
from pathlib import Path
from numpy import (ndarray, array, sqrt, where, exp, arange,
                   around, modf, argmin, argmax, delete, insert)
from scipy.interpolate import UnivariateSpline
from lmfit import Parameters
from ..utils import (check_objattrs, check_dictkeys, interp_yvals,
                     count_decimals, maxminval, minmaxval)
from ..xas.xasutils import ktoe, etok

class FeffPath(object):
    """Feffpath storage class.

    This class stores and manipulates data from a Feff path 
    file calculation (FEFFNNNN.dat) [4]_.

    Parameters
    ----------
    name : :class:`str`
        Name for the Feffpath. The default is None.
    ffpath: :class:`~pathlib.Path`
        Path to Feffdat data file. The default is None.

    Attributes
    ----------
    path_pars : :class:`dict`
        Dictionary with Feff path parameters:
        
        - ``filename`` (:class:`str`): name of file.
        - ``nlegs`` (:class:`float`) : number of path legs.
        - ``degen`` (:class:`float`) : path degeneracy.
        - ``reff``  (:class:`float`) : nominal path length.
        - ``rnrmav`` (:class:`float`): norman radius (bohr).
        - ``edge`` (:class:`float`)  : relative energy threshold (eV).

    feffdat : :class:`dict`
        Dictionary with Feff scattering data (static arrays):

        - ``k`` (:class:`ndarray`)         : photoelectron wavenumber in eV.
        - ``real_phc`` (:class:`ndarray`)  : central atom phase shift.
        - ``mag_feff`` (:class:`ndarray`)  : feff magnitude.
        - ``phase_feff`` (:class:`ndarray`): scattering phase shift.
        - ``red_factor`` (:class:`ndarray`): amplitude reduction factor.
        - ``lambd`` (:class:`ndarray`)     : mean free path.
        - ``real_p`` (:class:`ndarray`)    : real part of the complex wavenumber
    
    splines : :class:`dict`
        Dictionary with splines based on Feff scattering data:
        
        - ``ph`` (:class:`InterpolatedUnivariateSpline`)     : phase shift (central atom + scattering phase shifts).
        - ``amp`` (:class:`InterpolatedUnivariateSpline`)    : amplitude (feff magnitude * amplitude reduction factor).
        - ``real_p`` (:class:`InterpolatedUnivariateSpline`) : real part of the complex wavenumber.
        - ``lambd`` (:class:`InterpolatedUnivariateSpline`)  : mean free path.

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
    def __init__(self, ffpath:Path=None, name: str=None):
        if ffpath is not None:
            self.read_feffdat(ffpath)
        if name is None:
            name  = hex(id(self))
        self.name = name

    def __repr__(self):
        if self.name is not None:
            return '<Feffpath %s>' % self.name
        else:
            return '<Feffpath>'

    def read_feffdat(self, ffpath: Path) -> None:
        """Reads a Feff data file.

        Parameters
        ----------
        ffpath
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
        if not isfile(ffpath):
            raise IOError("file %s does not exists." % fpath)

        # containers for data
        self.path_pars = {} # path parameters
        self.feffdat   = {} # scattering functions
        self.geom      = [] # [x y z] pot atnum atsym
        self.splines   = {} # splines based on scattering functions
        # saving filename``
        self.path_pars['filename'] = basename(ffpath)

        # reading file
        with open(ffpath, 'r') as file:
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
        data       = array(data).T
        k          = data[0]    # photoelectron wavenumber
        real_phc   = data[1]    # central atom phase shift
        mag_feff   = data[2]    # feff magnitude
        phase_feff = data[3]    # scattering phase shift
        red_factor = data[4]    # amplitude reduction factor
        lambd      = data[5]    # mean free path
        real_p     = data[6]    # real part of the complex wavenumber
        
        self.feffdat['k']          = k
        self.feffdat['real_phc']   = real_phc
        self.feffdat['mag_feff']   = mag_feff
        self.feffdat['phase_feff'] = phase_feff
        self.feffdat['red_factor'] = red_factor
        self.feffdat['lambd']      = lambd
        self.feffdat['real_p']     = real_p
        
        del data
        # locking arrays from modification
        for key in self.feffdat:
            self.feffdat[key].flags.writeable = False
            
        # storing splines based on scattering values
        self.splines['ph']     = UnivariateSpline(k, phase_feff + real_phc, s=0)
        self.splines['amp']    = UnivariateSpline(k, red_factor * mag_feff, s=0)
        self.splines['real_p'] = UnivariateSpline(k, real_p, s=0)
        self.splines['lambd']  = UnivariateSpline(k, lambd, s=0)

    def get_chi(self, parsdict: dict=None, params: Parameters=None,  
                kstep: float=0.05, kmax: float=20) -> Tuple[ndarray, ndarray]:
        """Returns :math:`\chi(k)` for the Feff path.

        Parameters
        ----------
        parsdict
            Dictionary with either parameter values, or parameter names of 
            an ``lmfit`` Parameters object. At least one of the following 
            keys should be available:

            - ``s02``    : amplitude reduction factor for the path.
            - ``sigma2`` : debye-waller factor for the path.
            - ``degen``  : path degeneracy.
            - ``deltaE`` : :math:`\Delta E` for the path (eV).
            - ``deltaR`` : :math:`\Delta R` for the path (Angstrom).
            - ``ei``     : :math:`E_i` for the path.
            - ``c3``     : third cumulant parameter.
            - ``c4``     : fourth cumulant parameter.

            The default is None, which assigns the Feff path value for ``degen``,
            one for ``s02``, and zero for the remaining parameters. These default
            values will also be used if any of the listed keys is absent.
        params
            Parameter object from ``lmfit`` contaning the paramater names established 
            in ``dictpars``.
            The default is None.
        kstep
            Step in k array.
            The default is 0.05 :math:`\\unicode{x212B}^{-1}`.
		kmax
            Maximum possible value in k array. The actual maximum will be the greatest
			value less than or equal to kmax evenly divisible by the kstep.
            The default is 20 :math:`\\unicode{x212B}^{-1}`.

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
        TypeError
            If ``params`` is not a valid Parameters instance from ``lmfit``.
        NameError
            If ``parsdict`` contains strings and no ``params`` object is provided.

        Important
        ---------
        If present, negative values are stripped from the :math:`k` array and zero is included.

        Example
        -------
        .. plot::
            :context: reset

            >>> from araucaria.testdata import get_testpath
            >>> from araucaria.fit import FeffPath
            >>> from araucaria.plot import fig_xas_template
            >>> import matplotlib.pyplot as plt
            >>> fpath    = get_testpath('feff0001.dat')
            >>> feffpath = FeffPath(fpath)
            >>> k, chi   = feffpath.get_chi(kstep=0.05)
            >>> kw       = 1
            >>> fig, ax  = fig_xas_template(panels='e', fig_pars={'kweight': kw})
            >>> lin      = ax.plot(k, k**kw*chi)
            >>> fig.tight_layout()
            >>> plt.show(block=False)
        """
        # check pars attribute
        req_keys   = ['s02', 'sigma2', 'degen', 'deltaR', 'deltaE', 'ei', 'c3', 'c4']
        def_values = [  1.0,      0.0,     1.0,      0.0,      0.0,  0.0,  0.0, 0.0 ]

        # checking self attribute
        check_objattrs(self, FeffPath, attrlist=['feffdat'], exceptions=True)

        values = {}
        if parsdict is None:
        # default values
            for i, key in enumerate(req_keys):
                if key == 'degen':
                    values[key] = self.path_pars[key]
                else:
                    values[key] = def_values[i]
        elif params is None:
        # parsdict are values
            for i, key in enumerate(req_keys):
                if key in parsdict:
                    values[key] = parsdict[key]
                else:
                    if key == 'degen':
                        values[key] = self.path_pars[key]
                    else:
                        values[key] = def_values[i]
        else:
        # using parameters object
            check_objattrs(params, Parameters)
            for i, key in enumerate(req_keys):
                if key in parsdict:
                    pkey = parsdict[key]
                    values[key] = params[pkey].value
                else:
                    if key == 'degen':
                        values[key] = self.path_pars[key]
                    else:
                        values[key] = def_values[i]

        # computing k shifted and approximating k=0
        # first, an array built on specified kstep and kmax
        k_arr = np.linspace(0, int(round(kmax/kstep, 6))*kstep, int(round(kmax/kstep, 6)+1))
        k_s   = etok( ktoe(k_arr) - values['deltaE'] )
        k_den = where(k_s == 0, 1e-10, k_s)
        
        ph     = self.splines['ph'](k_den)
        amp    = self.splines['amp'](k_den)
        real_p = self.splines['real_p'](k_den)
        lambd  = self.splines['lambd'](k_den)
        
        # computing complex wavenumber
        p = np.sqrt((real_p + (1j/lambd))**2 - 1j*etok(values['ei']))
        
        # computing complex chi(k)
        chi_feff  = values['degen'] * values['s02'] * amp
        chi_feff /= ( k_den * (self.path_pars['reff'] + values['deltaR'] )**2 )
        chi_feff  = chi_feff.astype(complex)
        chi_feff *= exp(-2 * self.path_pars['reff'] * p.imag - 2 * ( p**2 ) * values['sigma2'] \
                    + 2 * (p**4) * values['c4'] / 3)
        chi_feff *= exp(1j * (2 * k_den * self.path_pars['reff'] + ph + 2 * p * ( values['deltaR'] - \
                    (2 * values['sigma2'] / self.path_pars['reff'] )) - 4 * (p**3) * values['c3'] / 3))

        # retaining the imaginary part
        chi_feff = chi_feff.imag

        return k_arr, chi_feff

def fftochi(ffpathlist: List[FeffPath], parslist: List[dict]=None, 
              params: Parameters=None, kstep: float=0.05) -> Tuple[ndarray, ndarray]:
    """Returns :math:`\chi(k)` for a list of Feff paths.

    Parameters
    ----------
    ffpathlist
        List with Feff paths to consider for the computation of :math:`\chi(k)`.
    parslist
        List of dictionaries with either parameter values, or parameter names of 
        an ``lmfit`` Parameters object. At least one of the following 
        keys should be available for each dictionary:

            - ``s02``    : amplitude reduction factor for the path.
            - ``sigma2`` : debye-waller factor for the path.
            - ``degen``  : path degeneracy.
            - ``deltaE`` : :math:`\Delta E` for the path (eV).
            - ``deltaR`` : :math:`\Delta R` for the path (Angstrom).
            - ``ei``     : :math:`E_i` for the path.
            - ``c3``     : third cumulant parameter.
            - ``c4``     : fourth cumulant parameter.

        The default is None, which assigns the respective Feff path value for ``degen``,
        one for ``s02``, and zero for the remaining parameters. These default
        values will also be used if any of the listed keys is absent.
    params
        Parameter object from ``lmfit`` contaning the paramater names established 
        in the dictionaries in ``parslist``.
        The default is None.
    kstep
        Step in k array.
        The default is 0.05 :math:`\\unicode{x212B}^{-1}`.

    Returns
    -------
    :
        Array with :math:`k` values.
    :
        Array with :math:`\chi(k)` values.

    Raises
    ------
    TypeError
        If items in ``ffpathlist`` are not FeffPath instances.

    Important
    ---------
    The returned array will be restricted to the highest initial value 
    and lowest final value in :math:`k` for the parameterized Feff paths.

    Example
    -------
        .. plot::
            :context: reset

            >>> from araucaria.testdata import get_testpath
            >>> from araucaria.fit import FeffPath, fftochi
            >>> from araucaria.plot import fig_xas_template
            >>> from lmfit import Parameters
            >>> import matplotlib.pyplot as plt
            >>> # reading feffpath files
            >>> fpath1    = get_testpath('feff0001.dat')
            >>> fpath2    = get_testpath('feff0002.dat')
            >>> feffpath1 = FeffPath(fpath1)
            >>> feffpath2 = FeffPath(fpath2)
            >>> # path parameters
            >>> params   = Parameters()
            >>> params.add('sigma2_1'  , 0.005)
            >>> params.add('sigma2_2' ,  0.007)
            >>> fefflist = [feffpath1, feffpath2]
            >>> parslist = [{'sigma2': 'sigma2_1'},
            ...             {'sigma2': 'sigma2_2'}]
            >>> k, chi   = fftochi(fefflist, parslist, params)
            >>> # plotting parameters
            >>> kw      = 1
            >>> fig, ax = fig_xas_template(panels='e', fig_pars={'kweight': kw})
            >>> lin     = ax.plot(k, k**kw*chi)
            >>> fig.tight_layout()
            >>> plt.show(block=False)
    """
    #checking feffpaths
    for item in ffpathlist:
        if isinstance(item, FeffPath):
            pass
        else:
            raise TypeError('%s is not a valid FeffPath instance.' % item.__class__)

    # containers for k and chi values
    k_list   = []
    chi_list = []

    if parslist is not None:
        if len(parslist) != len(ffpathlist):
            raise IndexError("number of feffpaths doesn't matches the number of path dicts")

        for i, ffpath in enumerate(ffpathlist):
            k, chi = ffpath.get_chi(parslist[i], params)
            k_list.append(k)
            chi_list.append(chi)
    else:
        for ffpath in ffpathlist:
            k, chi = ffpath.get_chi()
            k_list.append(k)
            chi_list.append(chi)

    # computing k-values
    # values are rounded to avoid interpolation errors
    dec = count_decimals(kstep, maxval=4)
    kmin = maxminval(k_list, kstep)
    kmax = minmaxval(k_list, kstep)
    k    = around(arange(kmin, kmax + kstep/2, kstep), dec)

    # computing chi values
    for i, chi in enumerate(chi_list):
        chi_list[i] = interp_yvals(k_list[i], chi, k)

    chi = array(chi_list).sum(axis=0)
    return k, chi