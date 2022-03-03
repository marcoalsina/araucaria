#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Feffit refers to the intepretation of an experimental EXAFS spectrum :math:`\chi(k)`
as a summation of theoretical EXAFS multiple scattering paths derived from the FEFF code [1]_.
The implementation of Feffit in ``araucaria`` follows as closely as possible the Feffit algorithm
developed by M. Newville [2]_ as implemented in the Larch software package [3]_.

FEFF is an ab-initio self-consistent code to compute the multiple scattering paths that compose
an EXAFS spectrum. Calculations of FEFF are controlled by a single input file (feff.inp), whose details 
are beyond the scope of this documentation. There are however numerous resources available to produce
such input file based on crsytallographic or structural information:

- `FEFF 6L Documentation <https://github.com/newville/ifeffit/blob/master/src/feff6/DOC/feff6L.doc>`_
- `Webatoms <https://bruceravel.github.io/demeter/documents/SinglePage/wa.html>`_

Important
---------
FEFF is not open source, so a license is required to use version 7 or higher. 
Nonetheless the authors of the code offer free lite versions of FEFF6 or FEFF8
(use limited to EXAFS analysis). Such versions are available for download at 
the `FEFF website 
<http://monalisa.phys.washington.edu/feffproject-feff-download.html>`_.


Computing :math:`\chi(k)_{\\textrm{feff}}`
******************************************

The amplitudes and phase shifts for a given scattering path :math:`j` computed by FEFF are stored in 
an individual file (feffNNN.dat) [4]_. The contents of each column in the file are as follows:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Column
     - Description
   * - k
     - photoelectron wavenumber (:math:`\\textrm{Å}^{-1}`), :math:`k_{\\textrm{feff}}`
   * - real[2*phc]
     - central atom phase shift
   * - mag[feff] 
     - magnitude of the effective wave backscattering amplitude
   * - phase[feff]
     - scattering phase shift
   * - red factor
     - amplitude reduction factor
   * - lambda
     - photoelectron mean free path (Å), :math:`\lambda(k)`
   * - real[p]
     - real part of the complex wavenumber, :math:`p'_j(k)`

Note that FEFF computes a phase shift for the central atom and for the scattering 
atom as a function of the wavenumber. The sum of both phase shift components is designated 
:math:`\delta_j(k)`. Analogously, the amplitude terms are multiplied and designated :math:`F_j(k)`.

The wavenumber is first recalculated considering an energy shift of :math:`\Delta E_0`, 
in order to allow comparison against an experimental EXAFS spectrum:

.. math::

    k = \sqrt{ k_{\\textrm{feff}}^2 - {2m_e \Delta E_0}/{\hbar^2} }

where :math:`m_e` corresponds to the electron mass, and :math:`\hbar` the reduced Planck constant.

Next, the complex wavenumber is computed in order to account for self-energy effects derived from
interaction with multiple electrons:

.. math::

    p_j(k) = p_j'(k) + ip_j''(k)
           = \sqrt{ \\big[ p'_j(k) + i/\lambda(k) \\big]^2 - i\, 2 m_e \Delta E_i /{\hbar^2} }

Where :math:`\Delta E_i` corresponds to the complex energy shift for the scattering path, and
:math:`\lambda(k)` corresponds to the photoelectron mean free path.

Finally, the theoretical scattering signal for a given path can be computed through the EXAFS
equation:

.. math::

    \\begin{align}
        \chi_j(k)_{\\textrm{feff}} = 
        &\\textrm{Im} \Big[ \\frac{F_j(k) N_j S_0^2}{k(R_{\\textrm{eff},j} + \\Delta R_j)^2} \, 
        \\textrm{exp}(-2p_j''(k)R_{\\textrm{eff}, j} - 2p_j(k)^2\sigma_j^2 + 
        \\frac{2}{3}p_j(k)^4c_{4,j})
        \\\\
        \\times\,&\\textrm{exp}(\,j\,\\{ 2kR_{\\textrm{eff},j} + \delta_j(k) + 2p_j(k)( \Delta R_j - 
        2 \sigma_j^2/R_{\\textrm{eff},j}) - \\frac{4}{3}p_j(k)^3 c_{3,j} \\}\,) \Big]
    \\end{align}

Where

- :math:`\chi_j(k)_{\\textrm{feff}}`: theoretical scattering signal for path.
- :math:`N_j`: path degeneracy.
- :math:`S_0^2`: amplitude reduction factor for path.
- :math:`R_{\\textrm{eff},j}`: path length.
- :math:`\Delta R_j`: shift in path length.
- :math:`\sigma_j^2`: Debye-Waller factor for path.
- :math:`c_{3,j}`: trhird cumulant for path.
- :math:`c_{4,j}`: trhird cumulant for path.

Note
----
:math:`\Delta R_j, \sigma_j^2, c_{3,j}` and :math:`c_{4,j}` correspond to the first four 
cumulants of the atomic pair distribution of the given scatttering path.


Performing Feffit
*****************

Feffit refers to the interpretation of an experimental spectrum :math:`\chi(k)` in terms 
of a summation of theoretical scattering paths :math:`\chi(k)_{\\textrm{feff}}`:

.. math::

    \chi(k) = \sum_{j=1}^n \chi_j(k, \\vec{\\beta_j})_{\\textrm{feff}} + \epsilon(k)

Where :math:`\epsilon(k)` corresponds to the residuals, :math:`n` corresponds to the number 
of scattering paths considered for the fit, and :math:`\\vec{\\beta_j}` corresponds to the 
set of variables that can be **fitted** for each scattering path.
The latter are detailed in the following table:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Variable name
     - Feff parameter
     - Description
   * - ``degen``
     - :math:`N_j`
     - Path degeneracy
   * - ``s02``
     - :math:`S_0^2`
     - Amplitude reduction factor
   * - ``deltaE0``
     - :math:`\Delta E_0`
     - Energy shift (eV)
   * - ``deltaEi``
     - :math:`\Delta E_i`
     - Complex energy shift (eV)
   * - ``deltaR``
     - :math:`\Delta R_j`
     - Path length shift (Å)
   * - ``sigma2``
     - :math:`\sigma^2_j`
     - Debye-Waller factor for path (:math:`\\textrm{Å}^2`)
   * - ``c3``
     - :math:`c_{3,j}`
     - Third cumulant for path
   * - ``c4``
     - :math:`c_{4,j}`
     - Fourth cumulant for path

Each variable can be represented as a :class:`~lmfit.parameter.Parameter` from ``lmfit``. 
As a consequence, bound constrains and mathematical expressions can be used to define the variable 
and fit it during Feffit.

The number of paths and variables to be considered for Feffit are at the discretion of the analyst.
Nonetheless, such interpretation must consider the information limit available in an EXAFS spectrum, 
particularly the number of independent points (:math:`N_{ind}`) given by the Nyquist-Shannon sampling 
theorem:

.. math::

    N_{ind} = \\frac{2 \, \Delta k \, \Delta R}{\pi} + 1

Where

- :math:`\Delta k`: range in wavenumber domain used for analysis. 
- :math:`\Delta R`: range in FT-space domain used for analysis.

Goodness of fit metrics
***********************

Describe the metrics considered to assess goodness of fit.

References
----------

.. [1] J.J. Kas et al. (2020) The FEFF Code. In International Tables of Crystallography, 
   Volume I. X-Ray Absorption Spectroscopy and Related Techniques, 
   https://doi.org/10.1107/S1574870720003274.

.. [2] M. Newville (2004) EXAFS analysis using FEFF and FEFFIT, 
   Journal of Synchrotron Radiation 8(2): 96-100,
   https://doi.org/10.1107/S0909049500016290.

.. [3] M. Newville (2013) Larch: An Analysis Package For XAFS And Related Spectroscopies,
    Journal of Physics: Conference Series, 430:012007, 
    https://doi.org/10.1088/1742-6596/430/1/012007.

.. [4] S. I. Zabinsky et al. (2002) Chapter 6: Ouput files, 
    Documentation FEFF 6L Version 6.01l, University of Washington,
    https://github.com/newville/ifeffit/blob/master/src/feff6/DOC/feff6L.doc.

Contents of the Feffit module
*****************************

The :mod:`~araucaria.fit.feffit` module offers the following 
classes and functions to perform EXAFS fitting based on FEFF calculations: 

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Class
     - Description
   * - :class:`FeffPars`
     - Dictionary of Paramater objects.
   * - :class:`FeffPath`
     - Feffpath storage class.

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`fftochi`
     - Returns chi(k) from a list of Feff paths.
   * - :func:`feffit`
     - Returns results from Feffit.
"""
from typing import Tuple, List, Union
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

# aux variables
PAR_NAMES = ['degen', 's02', 'sigma2', 'deltaR', 'deltaE0', 'deltaEi', 'c3', 'c4']

class FeffPars(Parameters):
    """Dictionary of Parameter objects.
    
    This class is inherited from the :class:`~lmfit.parameter.Parameters` 
    class of ``lmfit``, and compatible with FeffPath instances.
    """
    def __init__(self, asteval=None, usersyms=None):
        Parameters.__init__(self, asteval, usersyms)
        self._asteval.symtable['reff']  = 0.0
        self._asteval.symtable['degen'] = 0.0

class FeffPath(object):
    """Feffpath storage class.

    This class stores and manipulates data from a Feff path 
    file calculation (feffNNNN.dat).

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
        Dictionary with Feff path scattering data. Data are stored as 
        `read-only <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html>`_ 
        arrays:

        - ``k`` (:class:`~numpy.ndarray`)         : photoelectron wavenumber (eV).
        - ``real_phc`` (:class:`~numpy.ndarray`)  : central atom phase shift.
        - ``mag_feff`` (:class:`~numpy.ndarray`)  : feff magnitude.
        - ``phase_feff`` (:class:`~numpy.ndarray`): scattering phase shift.
        - ``red_factor`` (:class:`~numpy.ndarray`): amplitude reduction factor.
        - ``lambd`` (:class:`~numpy.ndarray`)     : photoelectron mean free path (Å).
        - ``real_p`` (:class:`~numpy.ndarray`)    : real part of the complex wavenumber.
    
    feffspl : :class:`dict`
        Dictionary with cubic splines for Feff path scattering data:
        
        - ``ph`` (:class:`~scipy.interpolate.UnivariateSpline`)    : total phase shift.
        - ``amp`` (:class:`~scipy.interpolate.UnivariateSpline`)   : total amplitude factor.
        - ``real_p`` (:class:`~scipy.interpolate.UnivariateSpline`): real part of the complex 
          wavenumber.
        - ``lambd`` (:class:`~scipy.interpolate.UnivariateSpline`) : photoelectron mean free path.

    geom : :class:`list`
        List with Feff path geometry parameters: 
        
        - coords (:class:`~numpy.ndarray`): cartesian coordinates of atom x, y, z (Å).
        - ipot  (:class:`int`)            : unique feff potential index (absorbing atom has index 0).
        - atnum (:class:`int`)            : atomic number.
        - atsym (:class:`str`)            :atomic symbol.

    params : :class:`dict`
        Dictionary with parameters used to compute :math:`\chi(k)` for the Feff path.
        Parameters are initially set at default values, but can be changed with the 
        :func:`assign_params` method.

        - ``degen``: path degeneracy. The default is None, which will evaluate 
          the degen value stored in the ``path_pars`` attribute.
        - ``s02``: amplitude reduction factor for the path. The default is 1.
        - ``sigma2``: Debye-Waller factor for the path. The default is 0.
        - ``deltaR``: :math:`\Delta R` for the path (Å). The default is 0.
        - ``deltaE0``: :math:`\Delta E_0` for the path (eV). The default is 0.
        - ``deltaEi``: :math:`\Delta E_i` for the path (eV). The default is 0.
        - ``c3``: third cumulant parameter. The default is 0.
        - ``c4``: fourth cumulant parameter. The default is 0.
        - ``lm_pars``: Optional pointer to a :class:`FeffPars` object.

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
       * - :func:`interp_feff`
         - Returns interpolated Feff scattering data.
       * - :func:`assign_params`
         - Assign parameters for computation of chi(k).
       * - :func:`get_chi`
         - Returns chi(k) for a Feff path.
       * - :func:`fftochi`
         - Feff to chi.

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
        >>> attrlist = ['geom', 'path_pars', 'feffdat', 'feffspl', 'params']
        >>> feffpath = FeffPath()
        >>> # empty FeffPath instance
        >>> check_objattrs(feffpath, FeffPath, attrlist)
        [False, False, False, False, False]
        >>> # reading feffdat file
        >>> feffpath.read_feffdat(fpath)
        >>> check_objattrs(feffpath, FeffPath, attrlist)
        [True, True, True, True, True]
        >>> print(feffpath.geom[0])
        [array([0., 0., 0.]), 0, 50, 'Sn']
        """
        # veifying existence of path
        if not isfile(ffpath):
            raise IOError("file %s does not exists." % fpath)

        # containers for data
        self.path_pars = {} # path parameters
        self.feffdat   = {} # scattering values
        self.geom      = [] # [x, y, z], ipot, atnum, atsym
        self.feffspl   = {} # splines based on scattering values
        # saving filename
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
                row           = line.split()
                loc           = array(row[:3], dtype=float)
                ipot, z, elem = int(row[3]), int(row[4]), row[5]
                self.geom.append([loc, ipot, z, elem])

            # reading scattering values
            data  = []
            for line in file:
                data.append([float(val) for val in line.strip().split()])

        # storing scattering values
        data                       = array(data).T
        k                          = data[0] # photoelectron wavenumber
        real_phc                   = data[1] # central atom phase shift
        mag_feff                   = data[2] # feff magnitude
        phase_feff                 = data[3] # scattering phase shift
        red_factor                 = data[4] # amplitude reduction factor
        lambd                      = data[5] # mean free path
        real_p                     = data[6] # real part of the complex wavenumber

        # constructing feffdat dictionary
        self.feffdat['k']          = k
        self.feffdat['real_phc']   = real_phc
        self.feffdat['mag_feff']   = mag_feff
        self.feffdat['phase_feff'] = phase_feff
        self.feffdat['red_factor'] = red_factor
        self.feffdat['lambd']      = lambd
        self.feffdat['real_p']     = real_p

        # delete data array and flag feffdat as read-only
        del data
        for key in self.feffdat:
            self.feffdat[key].flags.writeable = False
            
        # storing splines based on scattering values
        self.feffspl['ph']         = UnivariateSpline(k, phase_feff + real_phc, s=0)
        self.feffspl['amp']        = UnivariateSpline(k, red_factor * mag_feff, s=0)
        self.feffspl['real_p']     = UnivariateSpline(k, real_p, s=0)
        self.feffspl['lambd']      = UnivariateSpline(k, lambd, s=0)

        # dict to store the assigned parameter values
        self.params  = {'degen'     : None,
                        's02'       : float(1),
                        'sigma2'    : float(0),
                        'deltaR'    : float(0),
                        'deltaE0'   : float(0),
                        'deltaEi'   : float(0),
                        'c3'        : float(0),
                        'c4'        : float(0),
                        'lm_pars'   : None,
                        }

        # _params: dictionary to store evaluation of parameters
        self._params  = {}
        return

    def interp_feff(self, par: str=None, k: ndarray=None) -> ndarray:
        """Returns interpolated data for the selected Feff scattering function.

        Parameters
        ----------
        par
            Feff parameter to interpolate. Accepts any valid key from
            the ``feffspl`` dictionary.
        k
            Array of k values to compute the scattering function.

        Returns
        -------
        :
            Array with interpolated values of the scattering function.

        Raises
        ------
        AttributeError
            If the self instance has no ``feffspl`` attribute.
        KeyError
            If ``par`` key is not available in the ``feffspl`` dictionary.

        Example
        -------
        .. plot::
            :context: reset

            >>> from numpy import linspace
            >>> from araucaria.testdata import get_testpath
            >>> from araucaria.fit import FeffPath
            >>> import matplotlib.pyplot as plt
            >>> fpath      = get_testpath('feff0001.dat')
            >>> feffpath   = FeffPath(fpath)
            >>> k_feff     = feffpath.feffdat['k']
            >>> amp_feff   = feffpath.feffdat['mag_feff'] * feffpath.feffdat['red_factor']
            >>> k_interp   = linspace(1, 8)
            >>> amp_interp = feffpath.interp_feff('amp', k_interp)
            >>> # plotting results
            >>> fig, ax    = plt.subplots(1,1)
            >>> lin        = ax.plot(k_feff  , amp_feff  , marker='o', label='feff')
            >>> lin        = ax.plot(k_interp, amp_interp, marker='x', label='interp')
            >>> label      = ax.set_xlabel('$k\,[\AA^{-1}]$')
            >>> label      = ax.set_ylabel('$F_{feff}(k)$')
            >>> legend     = ax.legend()
            >>> fig.tight_layout()
            >>> plt.show(block=False)
        """
        # checking self attribute
        check_objattrs(self, FeffPath, attrlist=['feffspl'], exceptions=True)

        # checking key in dictionary
        check_dictkeys(self.feffspl, keylist=[par], exceptions=True)

        # interpolating requested feff parameter
        yvals = self.feffspl[par](k)

        return yvals

    def assign_params(self, degen: Union[float, str]=None, s02: Union[float, str]=None,
                      sigma2: Union[float, str]=None, deltaR: Union[float, str]=None,
                      deltaE0: Union[float, str]=None, deltaEi: Union[float, str]=None,
                      c3: Union[float, str]=None, c4: Union[float, str]=None,
                      lm_pars: FeffPars=None) -> None:
        """Assigns parameters for computation of :math:`\chi(k)` for the Feff path.

        Arguments can be assigned either with a float, or a string expression refering 
        to a parameter name from a :class:`~lmfit.parameter.Parameters` object from 
        ``lmfit``. Assigned parameters are updated in the ``params`` attribute of the 
        Feffpath instance.

        Parameters
        ----------
        degen
            Path degeneracy. The default is None.
        s02
            Amplitude reduction factor for the path. The default is None.
        sigma2
            Debye-Waller factor for the path. The default is None.
        deltaR
            :math:`\Delta R` for the path (Å). The default is None.
        deltaE0
            :math:`\Delta E_0` for the path (eV). The default is None.
        deltaEi
            :math:`\Delta E_i` for the path (eV). The default is None.
        c3
            Third cumulant parameter. The default is None.
        c4
            Fourth cumulant parameter. The default is None.
        lm_pars
            Optional FeffPars object considered for evaluation of
            parameter expressions. One may call parameters from this object 
            using their corresponding Parameter name (e.g. s02='amp').
            See the Notes for further details. The default is None.

        Returns
        -------
        :

        Raises
        ------
        TypeError
            If ``lm_pars`` is not a valid FeffPars instance.
            Only verified if ``lm_pars`` is not None.
        ValueError
            If any argument contains a string and no ``lm_pars`` object is 
            provided.
        NameError
            If any argument contains a Parameter name not declared in the 
            provided ``lm_pars`` object.

        Notes
        -----
        Any parameter whose corresponding argument is set to None will retain 
        the value previously assigned.

        If an optional ``lm_pars`` argument is provided, the following wildcards 
        may also be used to compose mathematical expressions:

        - "reff": nominal path length.
        - "degen": path degeneracy .

        These wildcards will be replaced in the expression by specific symbol names 
        depending on a intenal indexing of paths inside the FeffPars object
        (e.g. 'reff' -> '_rf0').

        Important
        ---------
        If an optional ``lm_pars`` argument is provided, the ``params`` attribute of the 
        Feffpath class will retain a **pointer** to the respective FeffPars object.
        Therefore, external modification of the original FeffPars object will in turn 
        **modify** the evaluation of Feffpath parameters during computation of :math:`\chi(k)`.

        See also
        --------
        get_chi : Returns chi(k) for the Feff path.

        Example
        -------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.fit import FeffPath, FeffPars
        >>> fpath    = get_testpath('feff0001.dat')
        >>> feffpath = FeffPath(fpath)
        >>> params   = FeffPars()
        >>> params.add('delr', expr='reff-1.0')
        >>> kwargs   = {'deltaR' : 'delr',
        ...             'sigma2' : 0.007,
        ...             'lm_pars': params}
        >>> feffpath.assign_params(**kwargs)
        >>> feffpath.params
        {'degen': None,
         's02': 1.0,
         'sigma2': 0.007,
         'deltaR': 'delr',
         'deltaE0': 0.0,
         'deltaEi': 0.0,
         'c3': 0.0,
         'c4': 0.0,
         'lm_pars': FeffPars([('delr',
                    <Parameter 'delr', value=1.0476, bounds=[-inf:inf], expr='_rf0-1.0'>)])}
        """
        # checking FeffPars object
        if lm_pars is None:
            pass
        else:
            check_objattrs(lm_pars, FeffPars, exceptions=True)
            self.params['lm_pars'] = lm_pars

            # assign counter for lmfit object (n)
            # and index for parameters set (i)
            if hasattr(lm_pars, '_cnt'):
                # counter exist: the lmfit object is referred by at least one Feffpath
                if hasattr(self, '_id'):
                    # index exist: Feffpath has a pointer to a lmfit object
                    if self._id > lm_pars._cnt:
                        # new index given if index is larger than counter
                        i = n = lm_pars._cnt + 1
                    else:
                        # index and counter retained if index is less/equal than counter
                        i = self._id
                        n = lm_pars._cnt
                else:
                    # index doesnt exist: index created and counter updated
                    i = n = lm_pars._cnt + 1
            else:
                # counter doesnt exist: new counter and index created
                i = n = 0
            self._id     = i
            lm_pars._cnt = n

            # storing reff and degen params in lmfit object
            lm_pars._asteval.symtable['_rf'+str(i)]  = self.path_pars['reff']
            lm_pars._asteval.symtable['_dgn'+str(i)] = self.path_pars['degen']

        # assign param names
        for par in PAR_NAMES:
            val = vars()[par]
            if val is None:
                    pass
            elif isinstance(val, str):
                if lm_pars is None:
                    raise ValueError("Name %s has no lm_pars object" %par)
                elif val in lm_pars:
                    self.params[par] = val
                    # updating FeffPars expressions
                    pval = self.params['lm_pars'][val]
                    if pval.expr:
                        # value is lmfit expression
                        pval = pval.expr
                        if 'reff' in pval:
                            pval = pval.replace('reff','_rf'+str(i))
                        if 'degen' in pval:
                            pval = pval.replace('reff','_rf'+str(i))
                        self.params['lm_pars'][val].expr = pval
                else:
                    raise NameError("Name %s is not available in the lm_pars object" % val )
            else:
                self.params[par] = val
        return

    def eval_params(self) -> None:
        """Evaluates parameters for the computation of :math:`\chi(k)` for the Feff path.

        Numerical evaluation is required when parameters are assigned as names of a 
        :class:`FeffPars` object. The resulting evaluation is stored in the ``_parameters`` 
        attribute of the Feffpath instance.

        Example
        -------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.fit import FeffPath, FeffPars
        >>> fpath    = get_testpath('feff0001.dat')
        >>> feffpath = FeffPath(fpath)
        >>> params   = FeffPars()
        >>> params.add('delr', expr='reff-1.0')
        >>> kwargs   = {'deltaR' : 'delr',
        ...             'sigma2' : 0.007,
        ...             'lm_pars': params}
        >>> feffpath.assign_params(**kwargs)
        >>> feffpath.eval_params()
        >>> feffpath._params
        {'degen': 2.0,
         's02': 1.0,
         'sigma2': 0.007,
         'deltaR': 1.0476,
         'deltaE0': 0.0,
         'deltaEi': 0.0,
         'c3': 0.0,
         'c4': 0.0}
        """
        for par in PAR_NAMES:
            val = self.params[par]
            if val is None:
                if par == 'degen':
                    self._params[par] = self.path_pars['degen']
                else:
                    self._params[par] = 0.0
            elif isinstance(val, str):
                val = self.params['lm_pars'][val]
                self._params[par] = val.value
            else:
                self._params[par] = val
        return    

    def get_chi(self, kstep: float=0.05, kmax: float=20, 
                k: ndarray=None) -> Tuple[ndarray, ndarray]:
        """Returns :math:`\chi(k)` for the Feff path.

        Parameters
        ----------
        kstep
            Step in k array.
            The default is 0.05 :math:`\\textrm{Å}^{-1}`.
        kmax
            Maximum possible value in k array. The actual maximum will be the greatest
            value less than or equal to kmax evenly divisible by the kstep.
            The default is 20 :math:`\\textrm{Å}^{-1}`.
        k
            Optional array of k values to use for the calculation.
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
            If the self instance has no ``path_pars`` attribute.
        AttributeError
            If the self instance has no ``feffspl`` attribute.
        AttributeError
            If the self instance has no ``params`` attribute.

        Important
        ---------
        - Each path parameter accepts as input either a float, int, or a 
          :class:`~lmfit.parameter.Parameter` object from ``lmfit``.
        - If present, negative values are stripped from the :math:`k` 
          array and zeroes are padded.

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
            >>> # plotting chi
            >>> kw       = 1
            >>> fig, ax  = fig_xas_template(panels='e', fig_pars={'kweight': kw})
            >>> lin      = ax.plot(k, k**kw*chi)
            >>> fig.tight_layout()
            >>> plt.show(block=False)
        """
        # checking self attribute
        check_objattrs(self, FeffPath, attrlist=['path_pars', 'feffspl', 'params'], 
        exceptions=True)

        # evaluate parameters and assign variables
        self.eval_params()
        degen     = self._params['degen']
        s02       = self._params['s02']
        sigma2    = self._params['sigma2']
        deltaR    = self._params['deltaR']
        deltaE0   = self._params['deltaE0']
        deltaEi   = self._params['deltaEi']
        c3        = self._params['c3']
        c4        = self._params['c4']
        
        # computing k shifted and approximating k=0
        # first, an array built on specified kstep and kmax
        if k is None:
            k_arr = linspace(0, int(round(kmax/kstep, 6))*kstep, int(round(kmax/kstep, 6)+1))
        else:
            k_arr = k

        # computing k and interpolated scattering functions
        k_s       = etok( ktoe(k_arr) - deltaE0 )
        k_den     = where(k_s == 0, 1e-10, k_s)
        ph        = self.feffspl['ph'](k_den)
        amp       = self.feffspl['amp'](k_den)
        real_p    = self.feffspl['real_p'](k_den)
        lambd     = self.feffspl['lambd'](k_den)

        # computing complex wavenumber
        p         = sqrt( (real_p + (1j/lambd))**2 - 1j*etok(deltaEi) )

        # computing complex chi(k)
        chi_feff  = degen * s02 * amp
        chi_feff /= ( k_den * (self.path_pars['reff'] + deltaR )**2 )
        chi_feff  = chi_feff.astype(complex)
        chi_feff *= exp(-2 * self.path_pars['reff'] * p.imag - 2 * ( p**2 ) * sigma2 \
                    + 2 * (p**4) * c4 / 3)
        chi_feff *= exp(1j * (2 * k_den * self.path_pars['reff'] + ph + 2 * p * ( deltaR - \
                    (2 * sigma2 / self.path_pars['reff'] )) - 4 * (p**3) * c3 / 3))

        # retaining the imaginary part
        chi_feff   = chi_feff.imag

        return k_arr, chi_feff

def fftochi(ffpathlist: List[FeffPath], kstep: float=0.05,
            kmax: float=20, k: ndarray=None) -> Tuple[ndarray, ndarray]:
    """Returns :math:`\chi(k)` for a list of Feff paths.

    Parameters
    ----------
    ffpathlist
        List with Feff paths to consider for the computation of :math:`\chi(k)`.
    kstep
        Step in k array.
        The default is 0.05 :math:`\\textrm{Å}^{-1}`.
    kmax
        Maximum possible value in k array. The actual maximum will be the greatest
        value less than or equal to kmax evenly divisible by the kstep.
        The default is 20 :math:`\\textrm{Å}^{-1}`.
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
            >>> from araucaria.fit import FeffPars, FeffPath, fftochi
            >>> from araucaria.plot import fig_xas_template
            >>> import matplotlib.pyplot as plt
            >>> # reading feffpath files
            >>> fpath1    = get_testpath('feff0001.dat')
            >>> fpath2    = get_testpath('feff0002.dat')
            >>> feffpath1 = FeffPath(fpath1)
            >>> feffpath2 = FeffPath(fpath2)
            >>> # path parameters
            >>> params    = FeffPars()
            >>> params.add('sig2_1'  , 0.005)
            >>> params.add('sig2_2' ,  0.007)
            >>> feffpath1.assign_params(sigma2='sig2_1', lm_pars=params)
            >>> feffpath2.assign_params(sigma2='sig2_2', lm_pars=params)
            >>> fefflist   = [feffpath1, feffpath2]
            >>> k, chi     = fftochi(fefflist)
            >>> # plotting chi
            >>> kw      = 1
            >>> fig, ax = fig_xas_template(panels='e', fig_pars={'kweight': kw})
            >>> lin     = ax.plot(k, k**kw*chi)
            >>> fig.tight_layout()
            >>> plt.show(block=False)
    """
    # checking feffpaths
    for item in ffpathlist:
        if isinstance(item, FeffPath):
            pass
        else:
            raise TypeError('%s is not a valid FeffPath instance.' % item.__class__)

    # computing k and chi arrays
    if k is None:
        k_arr = linspace(0, int(round(kmax/kstep, 6))*kstep, int(round(kmax/kstep, 6)+1))
    else:
        k_arr = k
    chi = zeros(len(k_arr))

    for i, path in enumerate(ffpathlist):
        # computing chi for each path and adding results
        _k, _chi = path.get_chi(k=k_arr)
        chi     += _chi

    return k_arr, chi

def feffit(data: Group, ffpathlist: List[FeffPath], params: Parameters=None, 
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