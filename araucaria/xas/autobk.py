#!/usr/bin/python
# -*- coding: utf-8 -*-
from warnings import warn
from numpy import (pi, sign, sqrt, ceil, copy,
                   ptp, arange, zeros, inf, 
                   ndarray, array, concatenate)
from scipy.interpolate import interp1d, splrep, splev
from lmfit import Parameters, minimize
from .. import Group
from .normalize import pre_edge
from .xasft import ftwindow, xftf_kwin
from .xasutils import etok, ktoe
from ..utils import index_nearest, check_objattrs, check_xrange

fmt_coef = 'coef_%2.2i'  # formated coefficient

def realimag(arr):
    #return real array of real/imag pairs from complex array
    return array([(i.real, i.imag) for i in arr]).flatten()

def spline_eval(kraw, mu, knots, coefs, order, kout):
    #eval bkg(kraw) and chi(k) for knots, coefs, order
    bkg = splev(kraw, [knots, coefs, order])
    chi = interp1d(kraw, (mu-bkg), kind='cubic')(kout)
    return (bkg, chi)

def residuals(pars, knots=None, order=3, irbkg=1, nfft=2048,
            kraw=None, mu=None, kout=None, ftwin=1, kweight=1, 
            chi_std=None, nclamp=0, clamp_lo=1, clamp_hi=1):
    # residuals function
    coefs    = [pars[fmt_coef % i].value for i in range(len(pars.keys()))]
    bkg, chi = spline_eval(kraw, mu, knots, coefs, order, kout)
    if chi_std is not None:
        chi = chi - chi_std
    out =  realimag(xftf_kwin(chi*ftwin, nfft=nfft)[:irbkg])
    if nclamp == 0:
        return out
    # spline clamps
    scale       = (1.0 + 100*(out**2).sum() )/ (len(out)*nclamp)
    scaled_chik = scale * kout**kweight * chi
    return concatenate((out,
                        abs(clamp_lo)*scaled_chik[:nclamp],
                        abs(clamp_hi)*scaled_chik[-nclamp:]))


def autobk(group: Group, rbkg: float=1.0, k_range: list=[0,inf], 
           kweight: int=2, win: str='hanning',  dk: float=0.1, 
           nfft: int=2048, kstep: float=0.05, k_std: ndarray=None, 
           chi_std: ndarray=None, nclamp: int=2, clamp_lo: int=1, 
           clamp_hi: int=1, update: bool=False) -> dict:
    """Autobk algorithm to remove background of a XAFS scan.

    Parameters
    ----------
    group
        Group containing the spectrum for background removal.
    rbkg
        Distance (Å) for :math:`\chi(R)` above which the signal is ignored.
        The default is 1.0.
    k_range
        Wavenumber range (:math:`Å^{-1}`).The default is [0, :data:`~numpy.inf`].
    kweight
        Exponent for weighting chi(k) by k**kweight.
        The default is 2.
    win
        Name of the the FT window type. The default is 'hanning'.
    dk
        Tapering parameter for the FT window. The default is 0.1.
    nfft
        Array size for the FT.  The default is 2048.
    kstep
        Wavenumber step size for the FT (:math:`Å^{-1}`).  The default is 0.05.
    k_std
        Optional k array for standard :math:`\chi(k)`.
    chi_std
        Optional array for standard :math:`\chi(k)`.
    nclamp
        Number of energy end-points for clamp. The default is 2.
    clamp_lo
        Weight of low-energy clamp. The default is 1.
    clamp_hi
        Weight of high-energy clamp. The default is 1.
    update
        Indicates if the group should be updated with the autobk attributes.
        The default is False.
      
    Returns
    -------
    :
        Dictionary with the following arguments:

        - ``bkg``         : array with background signal :math:`\mu_0(E)`.
        - ``chie``        : array with :math:`\chi(E)`.
        - ``chi``         : array with :math:`\chi(k)`.
        - ``k``           : array with wavenumbers.
        - ``autobk_pars`` : dictionary with autobk parameters.

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``e0`` or ``edge_step`` does not exist in ``group``.

    Warning
    -------
    ``rbkg`` cannot be lower than 2 x :math:`\pi`/(kstep x nfft), which 
    corresponds to the grid resolution of :math:`\chi(R)`.

    See also
    --------
    :func:`~araucaria.plot.fig_autobk.fig_autobk`: Plot the results of background removal.

    Notes
    -----
    The Autobk algorithm [1]_ approximates an EXAFS bakground signal by 
    fitting a cubic B-spline to the :math:`\chi(R)` signal below
    a ``rbkg`` value.

    The background removal is performed as follows:

    1. The B-spline is constructed considering approximately equally 
       distant knots in ``krange``. The number of knots is calculated as
       the integer value of 2 * ``rbkg`` * ``krange`` /:math:`\pi` + 2, with 
       a minimum value of 5 and a maximum value of 64.

    2. The initial coefficients (:math:`c_i`) for the B-spline at each 
       knot are calculated with a weighted average window:
    
    .. math::

            c_i = \\frac{\mu_{i-5} + 2 \cdot \mu_i + \mu_{i+5}}{4}

    3. The coefficients of the B-spline are then optimized in order to
       fit :math:`\chi(R)` below the ``rbkg`` value.
    4. If ``nclamp`` is provided, the given number of points at the 
       extremes of the weighted :math:`\chi(k)` signal are also included 
       in the minimize function, in order to fit such points considering
       the ``clamp_lo`` and ``clamp_hi`` weights.
    5. The fitted B-spline is removed from :math:`\mu(E)` in order to 
       compute :math:`\chi(k)`.

    If ``update=True`` the contents of the returned dictionary 
    will be included as attributes of ``group``.
    
    References
    ----------
    .. [1]  Newville, M., Livins, P., Yacoby, Y., Rehr, J. J.,  & Stern, E. A. (1993) 
       "Near-edge x-ray-absorption fine structure of Pb: A comparison of theory 
       and experiment". Physical Review B, 47(21), pp. 14126–14131.
    
    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria import Group
    >>> from araucaria.io import read_dnd
    >>> from araucaria.xas import pre_edge, autobk
    >>> from araucaria.utils import check_objattrs
    >>> fpath = get_testpath('dnd_testfile1.dat')
    >>> group = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
    >>> pre   = pre_edge(group, update=True)
    >>> attrs = ['bkg', 'chie', 'chi', 'k', 'autobk_pars']
    >>> autbk = autobk(group, update=True)
    >>> check_objattrs(group, Group, attrs)
    [True, True, True, True, True]
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['e0', 'edge_step'], exceptions=True)

    # extracting data and mu as independent arrays
    energy    = group.energy
    mu        = getattr(group, group.get_mode())
    e0        = group.e0
    edge_step = group.edge_step

    # index for e0 (ie0)
    ie0   = index_nearest(energy, e0)
    e0    = energy[ie0]

    # index for rbkg (irbkg)
    rgrid = pi / (kstep * nfft)
    if rbkg < 2*rgrid:
        warn('rbkg is lower than 2 x grid resolution of chi(R). Resetting tbkg to this limit.')
        rbkg = 2*rgrid
    irbkg = int(ceil(rbkg/rgrid))

    # ungridded k (kraw)
    enrel = energy[ie0:] - e0
    kraw  = sign(enrel) * etok(abs(enrel))

    # grided k (kout)
    krange = check_xrange(k_range, kraw)
    kout   = kstep * arange(ceil(krange[1]/kstep) )

    # index for max energy
    iemax  = min(len(energy) - 1, index_nearest(energy, e0 + ktoe(krange[1])) )

    # interpolate provided chi(k) onto the kout grid
    if chi_std is not None and k_std is not None:
        chi_std = interp1d(kout, k_std, chi_std, kind='cubic')(kout)
        
    # FT window (*k**kweight)
    ftwin = kout**kweight * ftwindow(kout, x_range=krange, win=win, dx1=dk)

    # calc k-value and initial guess for y-values of spline params
    # a minimum of 5 knots and a maximum of 64 knots are considered for the spline.
    nspl = max(5, min(64, int(2*rbkg*ptp(krange)/pi) + 2))
    spl_y, spl_k, spl_e  = [], [], []

    for i in range(nspl):
        # looping through the spline points
        # spline window in kraw is [ik - 5, ik + 5], except at the extremes of the array
        # a weighted average for mu is calculated in the extremes and center of this window.
        q   = krange[0] + i*ptp(krange)/(nspl - 1)
        ik  = index_nearest(kraw, q)
        i1  = min(len(kraw)-1, ik + 5)
        i2  = max(0, ik - 5)
        spl_k.append(kraw[ik])
        spl_e.append(energy[ik+ie0])
        spl_y.append( (mu[i1+ie0] + 2*mu[ik+ie0]  + mu[i2+ie0]) / 4.0 )

    # get B-spline represention: knots, coefs, order=3
    # coefs will be fitted
    knots, coefs, order = splrep(spl_k, spl_y , k=3, s=0)

    # set fit parameters from initial coefficients
    params = Parameters()
    tol    = 1.e-5
    for i in range(len(coefs)):
        params.add(name = fmt_coef % i, value=coefs[i], vary=i<len(spl_y))

    initbkg, initchi = spline_eval(kraw[:iemax-ie0+1], mu[ie0:iemax+1],
                                   knots, coefs, order, kout)

    result = minimize(residuals, params, method='leastsq',
                      gtol=tol, ftol=tol, xtol=tol, epsfcn=tol,
                      kws = dict(chi_std =chi_std,
                                 knots=knots, order=order,
                                 kraw=kraw[:iemax-ie0+1],
                                 mu=mu[ie0:iemax+1], irbkg=irbkg, 
                                 kout=kout, ftwin=ftwin, 
                                 kweight=kweight, nfft=nfft, 
                                 nclamp=nclamp, clamp_lo=clamp_lo, 
                                 clamp_hi=clamp_hi))

    # optimized coefficients
    coefs    = [result.params[fmt_coef % i].value for i in range(len(coefs))]
    bkg, chi = spline_eval(kraw[:iemax-ie0+1], mu[ie0:iemax+1],
                           knots, coefs, order, kout)
    obkg = copy(mu)
    obkg[ie0:ie0+len(bkg)] = bkg

    # output dictionaries
    init_bkg = copy(mu)
    init_bkg[ie0:ie0+len(bkg)] = initbkg
    
    autobk_pars = {'init_bkg'     : init_bkg,
                   'init_chi'     : initchi/edge_step,
                   'knots_e'      : spl_e,
                   'knots_y'      : [coefs[i] for i in range(nspl)],
                   'init_knots_y' : spl_y,
                   'nfev'         : result.nfev,
                   'k_range'      : krange,
                   'kweight'      : kweight,
                   'win'          : win,
                   'dk'           : dk,
                  }

    content = {'bkg'         : obkg,
               'chie'        : (mu-obkg)/edge_step,
               'k'           : kout,
               'chi'         : chi/edge_step,
               'edge_step'   : edge_step,
               'e0'          : e0,
               'autobk_pars' : autobk_pars
              }
    if update:
        group.add_content(content)
    return content

if __name__ == '__main__':
    import doctest
    doctest.testmod()