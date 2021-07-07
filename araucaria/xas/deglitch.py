#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Union
from numpy import array, append, where, copy, delete
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from .. import Group
from ..utils import check_objattrs
from ..xas.normalize import find_e0
#from ..xas.xasutils import roll_med
from ..stats.signal import roll_med
from ..stats.genesd import genesd

def deglitch(group: Group, e_window: Union[str,list]='xas', sg_window_length: int=9, 
             sg_polyorder:int =3, alpha: float=.025, 
             max_glitches: Union[int,str]='default', max_glitch_length: int=4,
             update: bool=False) -> dict:
    """Algorithm to deglitch a XAFS spectrum.

    Parameters
    ----------
    group
        Group containing the spectrum to deglitch.
    e_window
        Energy window to seach for outliers.
        Oprions are 'xas', 'xanes' and 'exafs'.
        Alternatively a list with 2 floats can be provided for
        the start and end energy for the search.
        See the Notes for further details. The default is 'xas'.
    sg_window_length
        Windows length for the Savitzky-Golay filter on the normalized spectrum.
        Must be an odd value. The default is 7.
    sg_polyorder
        Polynomial order for the Savitzky-Golay filter on the normalized spectrum.
        The default is 3.
    alpha
        Significance level for generalized ESD test for outliers.
        The default is 0.025.
    max_glitches
         Maximum number of outliers to remove.
         The default is the floor division of the array length by 10.
    max_glitch_length
        Maximum length of glitch in energy points. The default is 4.
    update
        Indicates if the group should be updated with the autobk attributes.
        The default is False.
    
    Returns
    -------
    :
        Dictionary with the following arguments:

        - ``index_glitches`` : indices of glitches in the original energy array.
        - ``energy_glitches``: glitches in the original energy array.
        - ``energy``         : deglitched energy array.
        - ``mu``             : deglitched array. Returned if ``group.get_mode() = 'mu'``.
        - ``fluo``           : deglitched array. Returned if ``group.get_mode() = 'fluo'``.
        - ``mu_ref``         : deglitched array. Returned if ``group.get_mode() = 'mu_ref'``.
        - ``deglitch_pars``  : dictionary with deglitch parameters.
    
    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``energy`` does not exist in ``group``.

    Warning
    -------
    Running :func:`~araucaria.xas.deglitch.deglitch` with ``update=True`` will overwrite
    the ``energy`` and the absorption attribute of ``group``.

    Notes
    -----
    This function deglitches a XAFS spectrum through a 
    two-step fitting with Savitzky-Golay filter and outlier 
    identification with a generalized extreme Studentized deviate (ESD) 
    test [1]_.
    
    - ``e_window='xas'`` considers the full spectrum for deglitching.
    - ``e_window='xanes'`` considers the beginning of the energy array 
      up to 150 eV above :math:`E_0`.
    - ``e_window='exafs'`` considers from 150eV above :math:`E_0` to the 
      end of the energy array
    - ``e_window=[float,float]`` provides start and end energies in eV.
    
    References
    ----------
    .. [1] Wallace, S. M., Alsina, M. A., & Gaillard, J. F. (2021) 
       "An algorithm for the automatic deglitching of x-ray absorption 
       spectroscopy data". J. Synchrotron Rad. 28, https://doi.org/10.1107/S1600577521003611
    
    Example
    -------
    .. plot::
        :context: reset
        
        >>> from numpy import allclose
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria import Group
        >>> from araucaria.io import read_dnd
        >>> from araucaria.xas import deglitch, pre_edge, autobk
        >>> from araucaria.utils import check_objattrs
        >>> fpath  = get_testpath('dnd_glitchfile.dat')
        >>> group  = read_dnd(fpath, scan='fluo')  # extracting fluo and mu_ref scans
        >>> cgroup = group.copy()
        >>> degli  = deglitch(cgroup, update=True)
        >>> attrs  = ['index_glitches', 'energy_glitches', 'deglitch_pars']
        >>> check_objattrs(cgroup, Group, attrs)
        [True, True, True]
        >>> allclose(cgroup.energy_glitches, group.energy[cgroup.index_glitches])
        True
        >>> print(cgroup.energy_glitches)
        [7552.2789 7548.1747 7390.512  7387.2613]

        >>> # plotting original and deglitched spectrum
        >>> from araucaria.plot import fig_xas_template
        >>> import matplotlib.pyplot as plt
        >>> for g in [group, cgroup]:
        ...     pre   = pre_edge(g, update=True)
        ...     autbk = autobk(g, update=True)
        >>> fig, ax = fig_xas_template(panels='xe')
        >>> line = ax[0].plot(group.energy,  group.norm,  label='original', color='tab:red')
        >>> line = ax[0].plot(cgroup.energy, cgroup.norm, label ='degliched', color='k')
        >>> line = ax[1].plot(group.k, group.k**2 * group.chi, color='tab:red')
        >>> line = ax[1].plot(cgroup.k, cgroup.k**2 * cgroup.chi, color='k')
        >>> leg  = ax[0].legend()
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['energy'], exceptions=True)
    
    # extracting data and mu as independent arrays
    energy  = group.energy
    mode    = group.get_mode()
    mu      = getattr(group, mode)

    # computing the energy window to perform the deglitch:
    e_lim     = 150 # energy limit to separates xanes from exafs
    e_windows = ['xas', 'xanes', 'exafs']
    if e_window in e_windows:
        if e_window =='xas':
            e_window = [energy[0], energy[-1]]
        else:
            if 'e0' not in dir(group):
                e0 = find_e0(group)
            else:
                e0 = getattr(group, 'e0')
            if e_window =='xanes':
                e_window  = [energy[0], e0 + e_lim]
            else: # exafs
                e_window  = [e0 + e_lim, energy[-1]]
    
    # energy indexes to perform deglitch
    index = where((energy >= e_window[0]) & (energy <= e_window[1]))[0]
    
    # savitzky-golay filter applied to entire mu array
    sg_init = savgol_filter(mu, sg_window_length, sg_polyorder) 

    # computing difference between normalized spectrum and savitzky-golay
    res1      = mu - sg_init
    
    # computing window size and rolling median
    win_size   = 2 * (sg_window_length + (max_glitch_length - 1)) + 1
    roll_mad1 = roll_med(abs(res1), window = win_size, edgemethod='calc')
    res_norm  = res1 / roll_mad1
    
    # if max_glitches is not set to an int, it will be set to the default
    if type(max_glitches) != int or max_glitches == 'default':
        max_glitches = len(res1)//10

    # finds outliers in residuals between data and savitzky-golay filter
    report, out1 = genesd(res_norm[index], max_glitches, alpha)
    
    # compensating for nonzero starting index in e_window
    if index[0] != 0: 
        out1 = out1 + index[0]
        
    # deglitching ends here if no outliers are found in this first stage
    if len(out1) == 0:
        index_glitches  = None
        energy_glitches = None

    else:
        # creating additional copy of mu
        mu_copy = copy(mu)
        
        # removes points that are poorly fitted by the S-G filter
        e2 = delete(energy, out1) 
        n2 = delete(mu, out1)
        
        #interpolates mu at the removed energy points
        f          = interp1d(e2, n2, kind='cubic') 
        interp_pts = f(energy[out1]) 
    
        # inserts interpolated points into normalized data
        for i, point in enumerate(out1):
            mu_copy[point] = interp_pts[i] 
    
        # fits mu with the interpolated points
        sg_final  = savgol_filter(mu_copy, sg_window_length, sg_polyorder) 
        res2      = mu - sg_final
        win_size  = (2*max_glitch_length) + 1
        roll_mad2 = roll_med(abs(res2), window = win_size, edgemethod='calc')
        res_norm2 = res2 / roll_mad2

        # normalizing the standard deviation to the same window as the savitzky-golay filter
        # allows to tackle the full spectrum, accounting for the data noise.
        report, glitches_init = genesd(res_norm2[index], max_glitches, alpha)

        # compensating for nonzero starting index in e_window
        if index[0] != 0:
            glitches_init = glitches_init + index[0]
        
        glitches = array([])
        for glitch in glitches_init:
            if True in where(abs(glitch-out1) < (sg_window_length//2) + 1, True, False):
                glitches = append(glitches, glitch)
        glitches[::-1].sort()
        index_glitches  = glitches.astype(int)
        energy_glitches = energy[index_glitches]

        if len(glitches) == 0:
            index_glitches = None
            energy_glitches = None
        else:
            # deglitching arrays
            energy = delete(energy, index_glitches)
            mu     = delete(mu, index_glitches)

    deglitch_pars = { 'e_window'          : e_window,
                      'sg_window_length'  : sg_window_length, 
                      'sg_polyorder'      : sg_polyorder,
                      'alpha'             : alpha,
                      'max_glitches'      : max_glitches,
                      'max_glitch_length' : max_glitch_length
                    }
    
    content = { 'index_glitches' : index_glitches,
                'energy_glitches': energy_glitches,
                'energy'         : energy,
                 mode            : mu,
                'deglitch_pars'  : deglitch_pars,
              }
    
    if update:
        group.add_content(content)
    return content

if __name__ == '__main__':
    import doctest
    doctest.testmod()