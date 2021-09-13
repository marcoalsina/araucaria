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
from numpy import (ndarray, array, sqrt, where, exp, arange, linspace,
                   around, modf, argmin, argmax, delete, insert, zeros)
from scipy.interpolate import UnivariateSpline
from lmfit import Parameters, Minimizer
from lmfit.minimizer import MinimizerResult
from ..utils import (check_objattrs, check_dictkeys, interp_yvals,
                     count_decimals, maxminval, minmaxval)
from ..xas.xasutils import ktoe, etok
from ..xas import xftf
from .. import Group

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
        self.params={}
        self._params={}
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

    def get_chi(self, degen: float=None, s02:float=1, sigma2: float=0, deltaR: float=0, 
                deltaE: float=0, ei: float=0, c3: float=0, c4: float=0, params: Parameters=None,
                kstep: float=0.05, kmax: float=20, k: ndarray=None) -> Tuple[ndarray, ndarray]:
        """Returns :math:`\chi(k)` for the Feff path.

        Parameters
        ----------
        degen
            Path degeneracy.
            For each path parameter, a float, int, or lmfit parameter may be input for the calculation.
            The default is None, which will use the degen value from the feffpath.
        s02
            Amplitude reduction factor for the path.
            The default is 1.
        sigma2
            Debye-waller factor for the path.
            The default is 0.
        deltaE
            :math:`\Delta E` for the path (eV).
            The default is 0.
        deltaR
            :math:`\Delta R` for the path (Angstrom).
            The default is 0.
        ei
            :math:`E_i` for the path.
            The default is 0.
        c3
            Third cumulant parameter.
            The default is 0.
        c4
            Fourth cumulant parameter.
            The default is 0.
        params
            lmfit Parameters to be used in chi calculations.
            One may call parameters in this groupusing a string corresponding to the
            Parameter name in the Parameters (e.g. s02='s02')
            The default is None.
        kstep
            Step in k array.
            The default is 0.05 :math:`\\unicode{x212B}^{-1}`.
        kmax
            Maximum possible value in k array. The actual maximum will be the greatest
			value less than or equal to kmax evenly divisible by the kstep.
            The default is 20 :math:`\\unicode{x212B}^{-1}`.
        k
            Optional array of k values to use for chi calculations.
            The default is None.

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

        # checking self attribute
        check_objattrs(self, FeffPath, attrlist=['feffdat'], exceptions=True)
        
        path_params = [degen, s02, sigma2, deltaR, deltaE, ei, c3, c4]
        for i, path_param in enumerate(path_params):
            if type(path_param) is str:
                path_params[i] = params[path_param]
        degen, s02, sigma2, deltaR, deltaE, ei, c3, c4 = path_params
        
        # computing k shifted and approximating k=0
        # first, an array built on specified kstep and kmax
        if k is None:
            k_arr = linspace(0, int(round(kmax/kstep, 6))*kstep, int(round(kmax/kstep, 6)+1))
        else:
            k_arr = k
        
        if degen is None:
            degen = self.path_pars['degen']
            
        k_s   = etok( ktoe(k_arr) - deltaE )
        k_den = where(k_s == 0, 1e-10, k_s)
        
        ph     = self.splines['ph'](k_den)
        amp    = self.splines['amp'](k_den)
        real_p = self.splines['real_p'](k_den)
        lambd  = self.splines['lambd'](k_den)
        
        # computing complex wavenumber
        p = sqrt((real_p + (1j/lambd))**2 - 1j*etok(ei))
        
        # computing complex chi(k)
        chi_feff  = degen * s02 * amp
        chi_feff /= ( k_den * (self.path_pars['reff'] + deltaR )**2 )
        chi_feff  = chi_feff.astype(complex)
        chi_feff *= exp(-2 * self.path_pars['reff'] * p.imag - 2 * ( p**2 ) * sigma2 \
                    + 2 * (p**4) * c4 / 3)
        chi_feff *= exp(1j * (2 * k_den * self.path_pars['reff'] + ph + 2 * p * ( deltaR - \
                    (2 * sigma2 / self.path_pars['reff'] )) - 4 * (p**3) * c3 / 3))

        # retaining the imaginary part
        chi_feff = chi_feff.imag
        
        self.k_dat = k_arr
        self.chi   = chi_feff
        
        return k_arr, chi_feff
    
    def assign_params(self, assignment_dictionary: dict=None):
        """Assigns parameters to be used in chi calculations to the feffpath.
        One may assign either a string corresponding to the name of a ``Parameter`` 
        or a float to the name of a ``get_chi`` variable.
        
        Parameters
        ----------
        assignment_dictionary
            Dictionary matching get_chi inputs to parameter names or values.
            Default is None.
        
        
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.fit import FeffPath
        >>> from araucaria.utils import check_objattrs
        >>> from lmfit import Parameters
        >>> feffpath = FeffPath()
        >>> # empty FeffPath instance
        >>> params = Parameters()
        >>> params.add('s02', value=1, min=0)
        >>> assignment_dictionary = {'s02':'s02', 'sigma2':0.007}
        >>> feffpath.assign_params(assignment_dictionary)
        >>> feffpath.params
        {'s02': 's02', 'sigma2': 0.007}
        """        
        self.params = assignment_dictionary
        return    

def fftochi(ffpathlist: List[FeffPath], params: Parameters=None, kstep: float=0.05,
            kmax: float=20, k: ndarray=None) -> Tuple[ndarray, ndarray]:
    """Returns :math:`\chi(k)` for a list of Feff paths.

    Parameters
    ----------
    ffpathlist
        List with Feff paths to consider for the computation of :math:`\chi(k)`.
    params
        Parameter object from ``lmfit`` contaning the paramater names established 
        in the dictionaries in ``params`` attribute within the Feff paths.
        The default is None.
    kstep
        Step in k array.
        The default is 0.05 :math:`\\unicode{x212B}^{-1}`.
    kmax
        Maximum possible value in k array. The actual maximum will be the greatest
                    value less than or equal to kmax evenly divisible by the kstep.
        The default is 20 :math:`\\unicode{x212B}^{-1}`.
    k
        Optional array of k values to use for chi calculations.
        The default is None.

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
            >>> feffpath1.assign_params({'sigma2':'sigma2_1'})
            >>> feffpath2.assign_params({'sigma2':'sigma2_2'})
            >>> fefflist  = [feffpath1, feffpath2]
            >>> sum_paths = fftochi(fefflist, params)
            >>> # plotting parameters
            >>> kw      = 1
            >>> fig, ax = fig_xas_template(panels='e', fig_pars={'kweight': kw})
            >>> lin     = ax.plot(sum_paths.k, sum_paths.k**kw*sum_paths.chi)
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
    
    if k is None:
        k_arr = linspace(0, int(round(kmax/kstep, 6))*kstep, int(round(kmax/kstep, 6)+1))
    else:
        k_arr = k
    
    chi = zeros(len(k_arr))
    
    for i, path in enumerate(ffpathlist):
        if path.params:
            params._asteval.symtable['_rf'+str(i)] = path.path_pars['reff']
            # allows 'reff' (and 'degen') to be used in expressions on more than
            # one path by changing the name for each path
            if 'degen' not in path.params.keys():
                params._asteval.symtable['_dgn'+str(i)] = path.path_pars['degen']
                # if degen is not specified, the value from the Feff calculation is used
            for item in path.params.items(): #tupled parameters
                if type(item[1]) == str:
                    path._params[item[0]] = params[item[1]]
                    
                    if params[item[1]].expr:
                        expr = params[item[1]].expr
                        if 'reff' in expr:
                            expr = expr.replace('reff','_rf'+str(i))
                        if 'degen' in expr and 'degen' not in path.params.keys():
                            expr = expr.replace('degen','_dgn'+str(i))
                        path._params[item[0]].expr = expr
                        
                else:
                    path._params[item[0]] = item[1]
                    
        path.get_chi(k=k_arr, **path._params)
        chi += path.chi
    
    sum_paths = Group()
    sum_paths.k = k_arr
    sum_paths.chi = chi
    return sum_paths

def fit_feff(data: Group, ffpathlist: List[FeffPath], params: Parameters=None, 
             fitspace: str='k', r_range: tuple=(1,10), fitmethod: str='leastsq', 
             **xftf_kws) -> MinimizerResult:
    """Returns fit results from fitting data to calculations from list of Feff paths.
    
    Parameters
    ----------
    data
        Group with EXAFS data to be fit by Feff paths.
    ffpathlist
        List with Feff paths to consider for the computation of :math:`\chi(k)`.
    params
        Parameter object from ``lmfit`` contaning the paramater names established 
        in the dictionaries in ``params`` attribute within the Feff paths.
        The default is None.
    fitspace
        Space upon which the fit will be performed. Options are 'k' or 'r'.
        The default is 'k'.
    r_range
        Minimum and maximum values of r to use when fitting in r-space.
        The default is (1,10).
    fitmethod
        Fitting method passed to ``lmfit``.
        The default is 'leastsq'.
    xftf_kws
        Dictionary of parameters passed to the xftf function.        
    """    
    xftf(data, **xftf_kws, update=True)
    
    def fit_fcn(params, fitspace):
        # function used in lmfit model
        # returns the difference between the data and the Feff model
        chi_model = fftochi(ffpathlist, params=params, k=data.k)
        
        if fitspace == 'k':
            weighted_chi  = data.k**data.xftf_pars['kweight'] * data.chi * data.kwin
            weighted_chim = data.k**data.xftf_pars['kweight'] * chi_model.chi * data.kwin
            difference = weighted_chi - weighted_chim
        elif fitspace == 'r':
            xftf(chi_model, **xftf_kws, update=True)
            condition  = where((data.r >= r_range[0]) & (data.r <= r_range[1]))
            difference = data.chir[condition] - chi_model.chir[condition]
            difference = difference.view(float)
        
        return(difference)
    
    fitter = Minimizer(fit_fcn, params, fcn_args=(fitspace))
    out = fitter.minimize(method=fitmethod)
    return(out)