#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Optional, Tuple, List
from itertools import cycle
from numpy import ravel
from matplotlib.pyplot import Axes, Figure, subplots

class FigPars(dict):
    """Class dictionary with argument for plotting functions.
    
    This utility class stores arguments to preset 
    ``Matplotlib`` axes.

    Parameters
    ----------
    prop_cycle : :class:`list`
        List of property cycle dictionaries for each
        axes. List is iterated if the elements of 
        the list are less than the number of panels.
        See the :meth:`~matplotlib.axes.Axes.set_prop_cycle`
        method for further details.
    e_range : :class:`list`
        XANES energy range.        
    e_ticks : :class:`list`
        XANES energy tick marks.
    mu_range : :class:`list`
        XANES norm. abs. range.
    mu_ticks : :class:`list`
        XANES norm. abs. tick marks.
    dmu_range : :class:`list`
        XANES deriv norm. abs. range.
    dmu_ticks : :class:`list`
        XANES deriv norm. abs. tick marks.
    kweight   : :class:`int`
        EXAFS k-weight.
    k_range   : :class:`list`
        EXAFS wavenumber range.
    k_ticks   : :class:`list`
        EXAFS wavenumber tick marks.
    chi_range : :class:`list`
        EXAFS chi(k) range.
    chi_ticks : :class:`list`
        EXAFS chi(k) tick marks.
    r_range   : :class:`list`
        FT-EXAFS R range.
    r_ticks   : :class:`list`
        FT-EXAFS R tick marks.
    chir_range : :class:`list`
        FT-EXAFS magnitude range.
    chir_ticks : :class:`list`
        FT-EXAFS magnitude tick marks.
    q_range   : :class:`list`
        reverse FT-EXAFS wavenumber range.
    q_ticks   : :class:`list`
        reverse FT-EXAFS wavenumber tick marks.
    """
    prop_cycle : list
    e_ticks    : List[float]
    mu_range   : List[float]
    mu_ticks   : List[float]
    dmu_range  : List[float]
    dmu_ticks  : List[float]
    kweight    : int
    k_range    : List[float]
    k_ticks    : List[float]
    chik_range : List[float]
    chik_ticks : List[float]
    r_range    : List[float]
    r_ticks    : List[float]
    chir_range : List[float]
    q_range    : List[float]
    q_ticks    : List[float]

def fig_xas_template(panels: str='xx', fig_pars: FigPars=None, 
                     **fig_kws: dict) -> Tuple[Figure,Axes]:
    """Returns a preset ``Matplotlib`` figure and axes object 
    to plot XAS spectra.

    Panel elements (axes) are indexed in row-major (C-style order).

    Parameters
    ----------
    panels
        Panels to plot. Valid arguments are as follows:

        - 'd'   : Derivative of XANES spectra.
        - 'x'   : XANES spectra.
        - 'e'   : EXAFS spectra.
        - 'r'   : FT-EXAFS spectra.
        - 'q'   : Reverse FT-EXAFS spectra.
        - 'u'   : Unassigned pannel.
        - '/'   : Character for a new row.

        The characters can be concatenated to produce multiple panels and rows.
        Examples: 'dxe', 'xx', 'xer', 'xx/xx'.
    fig_pars
        Dictionary arguments for the figure.
        See :class:`~araucaria.plot.template.FigPars` for details.    
    fig_kws
        Additional arguments to pass to the :meth:`~matplotlib.figure.Figure.subplots` 
        routine of ``Matplotlib``.

    Returns
    -------
    figure
        ``Matplolib`` figure object.
    axes
        ``Matplotlib`` axes object.

    Raises
    ------
    ValueError
        If number of columns and rows do not match in ``panels``.
    ValueError
        If requested panel type in ``panels`` is not recognized.

    Example
    --------
    .. plot::
        :context: reset
        
        >>> import matplotlib.pyplot as plt
        >>> from araucaria.plot import fig_xas_template
        >>> pars = {'e_range' : (0,100),
        ...         'mu_range': (0,1.5),
        ...         'k_range' : (0,15),
        ...         'r_range' : (0,6)}
        >>> fig, axes = fig_xas_template('dx/er', fig_pars=pars)
        >>> plt.show(block=False)
    """
    # valid axis type
    valid_panel_types = ['d', 'x', 'e', 'r', 'q', 'u']
    
    # axes methods
    met = ['set_xlim', 'set_ylim' , 'set_xticks', 'set_yticks']

    # counting the number of rows
    rows  = panels.split('/')
    nrows = len(rows)

    # veryfing that each row has the same number of elements
    if nrows == 1:
        ncols = len(panels)
    else:
        ncols = len(rows[0])
        for cols in rows[1:]:
            if len(cols) != ncols:
                raise ValueError('number of columns does not match between rows.')

    # converting panel argument to list of strings
    panels = ''.join(rows)    # merging all rows in a single string
    panels = list(panels)     # separating string in characters

    try:    
        for panel in panels:
            panel in valid_panel_types
    except:
        raise ValueError('%s panel type not recognized.' % panel)

    # creating figure and axes
    fig, axes = subplots(nrows, ncols, **fig_kws)

    # empty dict if None was provided
    if fig_pars is None:
        fig_pars = {}

    # formatting axes
    if 'prop_cycle' in fig_pars:
        prop_cycler = cycle(fig_pars['prop_cycle'])
        
    # setting k multiplier for EXAFS plots
    if 'kweight' in fig_pars:
        k = fig_pars['kweight']
    else:
        k = 2  # default value
    
    for i, ax in enumerate(ravel(axes)):
        # prop_cycles
        if 'prop_cycle' in fig_pars:
            prop_elem   = next(prop_cycler)
            ax.set_prop_cycle(**prop_elem)
        else:
            pass
 
        # XANES derivative axis
        if panels[i] == 'd':
            ax.set_xlabel(r'Energy [$eV$]')
            ax.set_ylabel('Deriv. abs. [a.u.]')
            keys = ['e_range', 'dmu_range', 'e_ticks', 'dmu_ticks']

        # XANES axis
        elif panels[i] == 'x':
            ax.set_xlabel(r'Energy [$eV$]')
            ax.set_ylabel('Norm. abs. [a.u.]')
            keys = ['e_range', 'mu_range', 'e_ticks', 'mu_ticks']

        # EXAFS axis
        elif panels[i] == 'e':
            ax.set_xlabel(r'$k$ [$\AA^{-1}$]')
            if k == 0:
                ax.set_ylabel(r'$\chi(k)$')
            else:
                ax.set_ylabel(r'$k^%i\chi(k)$ [$\AA^{%i}$]' % (k, k) )
            keys = ['k_range', 'chi_range', 'k_ticks', 'chi_ticks']

        # FT-EXAFS axis
        elif panels[i] == 'r':
            ax.set_xlabel(r'$R$ [$\AA$]')
            ax.set_ylabel(r'|$\chi(R)$|  [$\AA^{-%i}$]' % (k+1))
            keys = ['r_range', 'chir_range', 'r_ticks', 'chir_ticks']
        
        # reverse FT-EXAFS axis
        elif panels[i] == 'q':
            ax.set_xlabel(r'$q$ [$\AA^{-1}$]')
            if k == 0:
                ax.set_ylabel(r'$\chi(q)$')
            else:
                ax.set_ylabel(r'$q^%i\chi(q)$ [$\AA^{%i}$]' % (k, k) )
            keys = ['q_range', 'chi_range', 'q_ticks', 'chi_ticks']   
        
        # unnasigned axis
        elif panels[i] == 'u':
            keys = []

        # setting attribute values for each pannel (axes)
        for j, key in enumerate(keys):
            if key in fig_pars:
                getattr(ax, met[j])(fig_pars[key])
            else:
                pass

    fig.tight_layout()
    return (fig, axes)