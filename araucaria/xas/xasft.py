#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xas.xasft` module offers the following functions to perform 
discrete fast Fourier transforms (FFT) on a XAFS scan:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`ftwindow`
     - Returns a FT window.
   * - :func:`xftf`
     - Calculates a forward FFT of a XAFS signal.
   * - :func:`xftr`
     - Calculates a reverse FFT of a XAFS signal.
   * - :func:`xftf_kwin`
     - Calculates a forward FFT of a preset XAFS signal.
   * - :func:`xftr_kwin`
     - Calculates a reverse FFT of a preset XAFS signal.
"""
from numpy import (ndarray, where, pi, inf, exp, arange, 
                   linspace, ceil, zeros, sqrt, sin, cos, dtype)
from scipy.special import i0 as bessel_i0
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft
from .. import Group
from ..utils import check_objattrs, check_xrange

def complex_phase(arr):
    #return phase, modulo 2pi jumps
    from numpy import arctan2, diff, round, sign, pi
    
    phase    = arctan2(arr.imag, arr.real)
    d        = diff(phase)/pi
    out      = 1.0*phase[:]
    out[1:] -= pi*(round(abs(d))*sign(d)).cumsum()
    return out

def ftwindow(x: ndarray, x_range: list=[-inf,inf], 
             dx1: float=1, dx2: float=None, win: str='hanning') -> ndarray:
    """Returns a FT window.

    Parameters
    ----------
    x
        Array for the FT window.
    x_range
        Range for the FT window. The default is [-:data:`~numpy.inf`, :data:`~numpy.inf`].
    dx1
        First tapering parameter for FT window. The defaut is 1.
    dx2
        Second tapering parameter for FT window. If None it will use the same value as dx1.
    win
        Name of the window type.  The default is 'hanning'. See Notes for valid names.

    Returns
    -------
    :
        1-d array with the FT window.

    Raises
    ------
    ValueError
        If ``window`` name is not recognized.

    Notes
    -----
    Valid window names:
    
    - 'hanning'  : cosine-squared function window.
    - 'parzen'   : linear function window.
    - 'welch'    : quadratic function window.
    - 'gaussian' : Gaussian (normal) function window.
    - 'sine'     : sine function window.
    - 'kaiser'   : Kaiser-Bessel function-derived window.

    Example
    --------
    .. plot::
        :context: reset

        >>> from numpy import arange
        >>> import matplotlib.pyplot as plt
        >>> from araucaria.xas import ftwindow
        >>> from araucaria.plot import fig_xas_template
        >>> k       = arange(0, 10.1, 0.05)
        >>> k_range = [2,8]
        >>> windows = ['hanning', 'parzen', 'welch', 
        ...            'gaussian', 'sine', 'kaiser']
        >>> dk      = 1.0
        >>> fig_kws = {'sharex' : True}
        >>> fig, axes = fig_xas_template(panels='ee/ee/ee', **fig_kws)
        >>> for i, ax in enumerate(axes.flatten()):
        ...     win  = ftwindow(k, k_range, dk, win= windows[i])
        ...     line = ax.plot(k, win, label=windows[i])
        ...     for val in k_range:
        ...         line = ax.axvline(val - dk/2, color='gray', ls=':')
        ...         line = ax.axvline(val + dk/2, color='gray', ls=':')
        ...     leg  = ax.legend()
        ...     text = ax.set_ylabel('')
        ...     text = ax.set_xlabel('')
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    # available windows
    windows = ['han', 'fha', 'gau', 'kai', 'par', 'wel', 'sin', 'bes']
    name    = win.strip().lower()[:3]
    if name not in windows:
        raise ValueError("window name %s not recognized." % win)

    if dx2 is None:
        dx2 = dx1

    # asigning values inside the x array
    xrange = check_xrange(x_range, x)
    xstep  = (x[-1] - x[0]) / (len(x) - 1 )
    xeps   = 1.e-4 * xstep
    
    x1 = max(min(x), xrange[0] - dx1/2.0)
    x2 = xrange[0] + dx1/2.0  + xeps
    x3 = xrange[1] - dx2/2.0  - xeps
    x4 = min(max(x), xrange[1] + dx2/2.0)

    if name == 'fha':
        if dx1 < 0: dx1 = 0
        if dx2 > 1: dx2 = 1
        x2 = x1 + xeps + dx1*(xrange[1] - xrange[0]) / 2.0
        x3 = x4 - xeps - dx2*(xrange[1] - xrange[0]) / 2.0
    
    elif name == 'gau':
        dx1 = max(dx1, xeps)

    # return values as integer for indices
    def asint(val):
        return int((val+xeps)/xstep)
    
    i1 = max(0, asint(x1))
    i2 = max(0, asint(x2))
    i3 = min(len(x)-1, asint(x3))
    i4 = min(len(x)-1, asint(x4))
    if i2 == i1:
        i1 = max(0, i2 - 1)
    if i4 == i3:
        i3 = max(i2, i4 - 1)
    
    x1, x2, x3, x4 = x[i1], x[i2], x[i3], x[i4]
    if x1 == x2: x2 = x2 + xeps
    if x3 == x4: x4 = x4 + xeps
    
    # initial window
    fwin =  zeros(len(x))
    if i3 > i2:
        fwin[i2:i3] = 1.0

    # final window
    if name in ('han', 'fha'):
        fwin[i1:i2 + 1] = sin((pi/2)*(x[i1:i2 + 1]-x1) / (x2 - x1))**2
        fwin[i3:i4 + 1] = cos((pi/2)*(x[i3:i4 + 1]-x3) / (x4 - x3))**2
    
    elif name == 'par':
        fwin[i1:i2 + 1] =     (x[i1:i2+1]-x1) / (x2 - x1)
        fwin[i3:i4 + 1] = 1 - (x[i3:i4+1]-x3) / (x4 - x3)
    
    elif name == 'wel':
        fwin[i1:i2 + 1] = 1 - ((x[i1:i2 + 1]-x2) / (x2 - x1))**2
        fwin[i3:i4 + 1] = 1 - ((x[i3:i4 + 1]-x3) / (x4 - x3))**2
    
    elif name  in ('kai', 'bes'):
        cen  = (x4 + x1) / 2
        wid  = (x4 - x1) / 2
        arg  = 1 - (x-cen)**2 / (wid**2)
        
        arg[where(arg < 0)] = 0
        if name == 'bes': # 'bes' : ifeffit 1.0 implementation of kaiser-bessel
            fwin = bessel_i0(dx1* sqrt(arg)) / bessel_i0(dx1)
            fwin[where(x <= x1)] = 0
            fwin[where(x >= x4)] = 0
        else: # better version
            scale = max(1.e-10, bessel_i0(dx1)-1)
            fwin = (bessel_i0(dx1 * sqrt(arg)) - 1) / scale
    elif name == 'sin':
        fwin[i1:i4 + 1] = sin(pi*(x4-x[i1:i4 + 1]) / (x4 - x1))

    elif name == 'gau':
        cen  = (x4 + x1) / 2
        fwin =  exp(-(((x - cen)**2)/(2*dx1**2)))

    return fwin

def xftf(group: Group, k_range: list=[0,20], kweight: int=0, 
    dk1: float=1, dk2: float=None, win: str ='hanning', 
    rmax_out: float =10, nfft: int=2048, kstep: float=0.05, 
    with_phase: bool=False, update: bool=False) -> dict:
    """Calculates a forward FFT of a XAFS signal.

    A XAFS forward FFT decomposes :math:`\chi(k)` into :math:`\chi(R)`.

    Parameters
    ----------
    group
        Group containing chi(k) for the forward FT.
    k_range
        Photoelectron wavenumber range for the FT (:math:`Å^{-1}`).
        The default is [0, 20].
    kweight
        Exponent for weighting chi(k) by k**kweight.
        The default is 0.
    dk1
        First tapering parameter for the FT window.
        The detault is 1.
    dk2
        Second tapering parameter for FT window.
        If None it will take the value of dk1.
    win
        Name of the FT window type. The default is 'hanning'.
    rmax_out
        Highest R value for :math:`\chi(R)` (Å). The default is 10.
    nfft
        Array size for the FT.  The default is 2048.
    kstep
        Wavenumber step size for the FT (:math:`Å^{-1}`).  The default is 0.05.
    with_phase
        Return the phase as well as magnitude, real, and imaginary parts.
        The default is False.
    update
        Indicates if the group should be updated with the ftkf attributes.
        The default is False.

    Returns
    -------
    :
        Dictionary with the following arguments:

        - ``kwin``     : array with the FT window.
        - ``r``        : array with R values (Å).
        - ``chir``     : array with :math:`\chi(R)`.
        - ``chir_mag`` : array with magnitude of :math:`\chi(R)`.
        - ``chir_re``  : array with real part of :math:`\chi(R)`.
        - ``chir_im``  : array with imaginary part of :math:`\chi(R)`.
        - ``chir_pha`` : array with phase of :math:`\chi(R)`. Returned if ``with_phase=True``.

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``k`` or ``chi`` does not exist in ``group``.

    Notes
    -----
    If ``update=True`` the contents of the returned dictionary will be
    included as attributes of ``group``.

    Example
    -------
    .. plot::
        :context: reset

        >>> from araucaria.testdata import get_testpath
        >>> from araucaria import Group
        >>> from araucaria.io import read_dnd
        >>> from araucaria.xas import pre_edge, autobk, xftf
        >>> from araucaria.utils import check_objattrs
        >>> kw      = 2
        >>> k_range = [2,10]
        >>> fpath   = get_testpath('dnd_testfile1.dat')
        >>> group   = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
        >>> pre     = pre_edge(group, update=True)
        >>> autbk   = autobk(group, update=True)
        >>> fft     = xftf(group, k_range=k_range, kweight=kw, update=True)
        >>> attrs = ['kwin', 'r', 'chir', 'chir_mag', 'chir_re', 'chir_im']
        >>> check_objattrs(group, Group, attrs)
        [True, True, True, True, True, True]

        >>> # plotting forward FFT signal
        >>> import matplotlib.pyplot as plt
        >>> from araucaria.plot import fig_xas_template
        >>> fig, ax = fig_xas_template(panels='er', fig_pars={'kweight':kw})
        >>> line = ax[0].plot(group.k, group.k**kw*group.chi)
        >>> line = ax[0].plot(group.k, group.kwin, color='firebrick')
        >>> xlim = ax[0].set_xlim(0,12)
        >>> line = ax[1].plot(group.r, group.chir_mag)
        >>> xlim = ax[1].set_xlim(0,6)
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['k', 'chi'], exceptions=True)

    # extracting k and chi
    k   = group.k
    chi = group.chi

    # number of points
    if dk2 is None:
        dk2 = dk1
    k_range.sort()
    kmax = max(max(k), k_range[1] + dk2)
    kpts  = int(ceil(kmax)/kstep)

    # calculating window and weighted chik
    k_    = kstep * arange(kpts)
    chi_  = (interp1d(k, chi, fill_value='extrapolate'))(k_)
    win   = ftwindow(k_, x_range=k_range, dx1=dk1, win=win)[:kpts]
    cchi  = chi_[:kpts] * k_[:kpts]**kweight

    # calculating xftf
    out   = xftf_kwin(win*cchi, kstep=kstep, nfft=nfft)
    rstep = pi/(kstep*nfft)
    rpts  = int(min(nfft/2, ceil(rmax_out/rstep)))
    r     = rstep * arange(rpts)
    mag   = sqrt(out.real**2 + out.imag**2)

    # output dictionary
    content = {'kwin'     : win[:len(chi)],
               'r'        : r[:rpts],
               'chir'     : out[:rpts],
               'chir_mag' : mag[:rpts],
               'chir_re'  : out.real[:rpts],
               'chir_im'  : out.imag[:rpts]
               }
    if with_phase:
        content['chir_pha'] =  complex_phase(out[:rpts])

    if update:
        group.add_content(content)
    return content

def xftr(group: Group , r_range: list=[0,20], rweight: int=0, 
         dr1: float=1, dr2: float=None, win='hanning', 
         nfft: int=2048, kstep: float=0.05, qmax_out: float=20, 
         with_phase: bool=False, update: bool=False) -> dict:
    """Calculates a reverse FFT of a XAFS signal.

    A XAFS reverse FFT recovers :math:`\chi(q)` from :math:`\chi(R)`.

    Parameters
    ----------
    group
        Group containing chi(k) for the forward FT.
    r_range
        R range for the reverse FT (Å).
        The default is [0, 20].
    rweight
        Exponent for weighting chi(R) by R**rweight.
        The default is 0.
    dr1
        First tapering parameter for the reverse FT window.
        The default is 1.
    dr2
        Second tapering parameter for reverse FT window.
        If None it will take the value of dr1.
    win
        Name of the FT window type. The default is 'hanning'.
    qmax_out
        Highest q value for :math:`\chi(R)` ((:math:`Å^{-1}`)). The default is 20.
    nfft
        Array size for the reverse FT.  The default is 2048.
    kstep
        Wavenumber step size for the reverse FT (:math:`Å^{-1}`).  The default is 0.05.
    with_phase
        Return the phase as well as magnitude, real, and imaginary parts.
        The default is False.
    update
        Indicates if the group should be updated with the ftkf attributes.
        The default is False.

    Returns
    -------
    :
        Dictionary with the following arguments:

        - ``rwin``     : array with the reverse FT window.
        - ``q``        : array with q values (Å).
        - ``chiq``     : array with :math:`\chi(R)`.
        - ``chiq_mag`` : array with magnitude of :math:`\chi(q)`.
        - ``chiq_re``  : array with real part of :math:`\chi(q)`.
        - ``chiq_im``  : array with imaginary part of :math:`\chi(q)`.
        - ``chiq_pha`` : array with phase of :math:`\chi(q)`. Returned if ``with_phase=True``.
    
    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``q`` or ``chir`` does not exist in ``group``.

    Notes
    -----
    If ``update=True`` the contents of the returned dictionary will be
    included as attributes of ``group``.

    Example
    -------
    .. plot::
        :context: reset

        >>> from araucaria.testdata import get_testpath
        >>> from araucaria import Group
        >>> from araucaria.io import read_dnd
        >>> from araucaria.xas import pre_edge, autobk, xftf, xftr
        >>> from araucaria.utils import check_objattrs
        >>> kw      = 2
        >>> k_range = [2,10]
        >>> r_range = [0.5, 2]
        >>> fpath   = get_testpath('dnd_testfile1.dat')
        >>> group   = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
        >>> pre     = pre_edge(group, update=True)
        >>> autbk   = autobk(group, update=True)
        >>> fft     = xftf(group, k_range=k_range, kweight=kw, update=True)
        >>> rft     = xftr(group, r_range=r_range, update=True)
        >>> attrs = ['rwin', 'q', 'chiq', 'chiq_mag', 'chiq_re', 'chiq_im']
        >>> check_objattrs(group, Group, attrs)
        [True, True, True, True, True, True]

        >>> # plotting forward FFT signal
        >>> import matplotlib.pyplot as plt
        >>> from araucaria.plot import fig_xas_template
        >>> fig, ax = fig_xas_template(panels='rq', fig_pars={'kweight': kw})
        >>> line = ax[0].plot(group.r, group.chir_mag)
        >>> line = ax[0].plot(group.r, group.rwin, color='firebrick')
        >>> xlim = ax[0].set_xlim(0,6)
        >>> line = ax[1].plot(group.k, group.k**kw*group.chi)
        >>> line = ax[1].plot(group.q, group.chiq_re)
        >>> xlim = ax[1].set_xlim(0,12)
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['r', 'chir'], exceptions=True)

    # extracting r and chir
    r     = group.r
    chir  = group.chir
    
    # calculating rstep and kstep
    rstep = r[1] - r[0]
    r_    = rstep * arange(nfft, dtype='float64')
    kstep = pi/(rstep*nfft)
    
    # setting scale for output
    if chir.dtype == dtype('complex128'):
        scale = 0.5
    else:
        scale = 1.0

    # calculating window and weighted chir
    if dr2 is None:
        dr2 = dr1
    r_range.sort()
    win   = ftwindow(r_, x_range=r_range, dx1=dr1, dx2=dr2, win=win)
    cchir = zeros(nfft, dtype='complex128')
    cchir[:len(chir)] = chir * r_[:len(chir)]**rweight
    
    # calculating xftr
    out   = scale * xftr_kwin( cchir*win, kstep=kstep, nfft=nfft)
    q     = linspace(0, qmax_out, int(ceil(qmax_out/kstep)))
    iqmax = len(q)
    mag   = sqrt(out.real**2 + out.imag**2)

    # output dictionary
    content = {'rwin'     : win[:len(chir)],
               'q'        : q,
               'chiq'     : out[:iqmax],
               'chiq_mag' : mag[:iqmax],
               'chiq_re'  : out.real[:iqmax],
               'chiq_im'  : out.imag[:iqmax],
              }

    if with_phase:
        content['chiq_pha'] =  complex_phase(out[:npts])

    if update:
        group.add_content(content)
    return content

def xftf_kwin(chi: ndarray, nfft: int=2048, kstep: float=0.05) -> ndarray:
    """Calculates a forward FFT of a preset XAFS signal.

    Parameters
    ----------
    chi
        Array of :math:`\chi(k)` on a uniform grid. 
        Window and k-weighting are assumed to have been 
        already applied to the signal.
    nfft
        Array size for the FFT. The default is 2048.
    kstep
        Photoelectron wavenumber step size for the FFT (:math:`Å^{-1}`).
        The default is 0.05.

    Returns
    -------
    :
        Complex array of :math:`\chi(R)`.

    Example
    -------
    .. plot::
        :context: reset

        >>> from numpy import arange, sin, pi
        >>> from scipy.fftpack import fftfreq
        >>> from araucaria.xas import ftwindow, xftf_kwin
        >>> nfft = 2048  # number of points for FFT
        >>> ks   = 0.05  # delta k (angstrom^-1)
        >>> f1   = 0.5   # freq1 (angstrom)
        >>> f2   = 1.2   # freq2 (angstrom)
        >>> k    = arange(0, 10, ks)
        >>> win  = ftwindow(k, x_range=(0,10), dx1=0.5, win='sine')
        >>> chi  = 0.5*sin(2*pi*k*f1) + 0.1*sin(2*pi*k*f2)
        >>> chir = xftf_kwin(win*chi, nfft=nfft, kstep=ks)
        >>> freq = fftfreq(nfft, ks)
        >>> print(chir.dtype)
        complex128

        >>> # plotting forward FFT signal
        >>> import matplotlib.pyplot as plt
        >>> from araucaria.plot import fig_xas_template
        >>> fig, ax = fig_xas_template(panels='er', fig_pars={'kweight':0})
        >>> line = ax[0].plot(k, win*chi)
        >>> line = ax[1].plot(freq[:int(nfft/2)], abs(chir[:int(nfft/2)]))
        >>> xlim = ax[1].set_xlim(0,2)
        >>> xlab = ax[1].set_xlabel('$R/\pi$ [$\AA$]')
        >>> for f in (f1,f2):
        ...     line = ax[1].axvline(f, color='gray', ls=':')
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    cchi = zeros(nfft, dtype='complex128')
    cchi[0:len(chi)] = chi
    chir = (kstep / sqrt(pi)) * fft(cchi)[:int(nfft/2)]
    return chir

def xftr_kwin(chir: ndarray, nfft: int=2048, kstep: float=0.05) -> ndarray:
    """Calculates a reverse FFT of a preset XAFS signal.

    Parameters
    ----------
    chir
        Array of :math:`\chi(R)` on a uniform grid.
        Window and k-weighting are assumed to have been 
        already applied to the signal.
    nfft
        Array size  for the FFT.  The default is 2048.
    kstep
        Photoelectron wavenumber step size for the FFT (:math:`Å^{-1}`).
        The default is 0.05.

    Returns
    -------
    :
        Complex array for :math:`\chi(q)`.

    Example
    -------
    .. plot::
        :context: reset    
    
        >>> from numpy import arange, sin, pi
        >>> from scipy.fftpack import fftfreq
        >>> from araucaria.xas import ftwindow, xftf_kwin, xftr_kwin
        >>> nfft = 2048  # number of points for FFT
        >>> ks   = 0.05  # delta k (angstrom^-1)
        >>> f1   = 0.5   # freq1 (angstrom)
        >>> k    = arange(0, 10, ks)
        >>> wink = ftwindow(k, x_range=(0,10), dx1=0.5, win='sine')
        >>> chi  = 0.5*sin(2*pi*k*f1)
        >>> chir = xftf_kwin(wink*chi, nfft=nfft, kstep=ks)
        >>> freq = fftfreq(nfft, ks)[:nfft//2]
        >>> chiq = xftr_kwin(chir, nfft=nfft, kstep=ks)[:len(k)]
        >>> print(chiq.dtype)
        complex128
        
        >>> # plotting reverse FFT signal
        >>> import matplotlib.pyplot as plt
        >>> from araucaria.plot import fig_xas_template
        >>> fig, ax = fig_xas_template(panels='re', fig_pars={'kweight':0})
        >>> line = ax[0].plot(freq, abs(chir))
        >>> xlim = ax[0].set_xlim(0,2)
        >>> xlab = ax[0].set_xlabel('$R/\pi$ [$\AA$]')
        >>> line = ax[1].plot(k, chiq)
        >>> text = ax[1].set_xlabel(r'$q(\AA^{-1})$')
        >>> text = ax[1].set_ylabel(r'$\chi(q)$')
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    cchi = zeros(nfft, dtype='complex128')
    cchi[0:len(chir)] = chir
    chiq = (4*sqrt(pi)/kstep) * ifft(cchi)[:int(nfft/2)]
    return chiq