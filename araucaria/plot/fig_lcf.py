#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Tuple
from numpy import ptp
from matplotlib.pyplot import Axes, Figure
from .. import Group, FitGroup
from .template import fig_xas_template, FigPars
from ..utils import check_objattrs

def fig_lcf(group: FitGroup, offset: float=0.5, 
            annotate: bool=True, fontsize: float=8,
            fig_pars: FigPars=None, **fig_kws) -> Tuple[Figure,Axes]:
    """Plots the results of linear combination fitting on a collecton.

    Parameters
    ----------
    group
        Valid FitGroup from :func:`lcf`.
    offset
        Offset step value for plots.
        The default is 0.5.
    annotate
        Indicates if the plot should be annotated.
        The detault is True
    fontsize
        Font size for annotations.
        The default is 8.
    fig_pars
        Dictionary arguments for the figure.
        See :class:`~araucaria.plot.template.FigPars` for details.    
    fig_kws
        Additional arguments to pass to the :meth:`~matplotlib.figure.Figure.subplots` 
        routine of ``Matplotlib``.

    Returns
    -------
    figure
        Matplolib figure object.
    axes
        Matplotlib axes object. 
   
    Raises
    ------
    TypeError
        If ``group`` is not a valid FitGroup instance.
    AttributeError
        If attribute ``min_pars``, ``lcf_pars``, ``scangroup``,
        or ``refgroups`` does not exist in ``group``.
    """
    check_objattrs(group, FitGroup, attrlist=['min_pars', 
    'lcf_pars', 'scangroup', 'refgroups'], exceptions=True)

    # setting the figure type
    fig_type = group.lcf_pars['fit_region']
    if fig_type == 'exafs':
        panels = 'ee'
    elif fig_type == 'xanes':
        panels = 'xx'
    else:
        panels = 'dd'

    # data object names
    groupnames = [group.scangroup] + group.refgroups
    datnames   = ['scan'] + ['ref%s' % (i+1) for i in range(len(group.refgroups))]

    # initializing figure and axes
    fig, axes = fig_xas_template(panels=panels, fig_pars=fig_pars, **fig_kws)

    # panel 1: plotting original spectra
    for i, name in enumerate(groupnames):
        if fig_type == 'exafs':
            xvar    = 'k'             # plot variable in x-axis
            yvar    = [0, 0.25, 1.5]  # vars to locate annotated text in y-axis 
        else:
            xvar = 'energy'
            if fig_type == 'xanes':
                yvar = [1, 0.25, 0.5]
            else:
                yvar = [0, 0.25, 0.5]
        axes[0].plot(getattr(group, xvar), i*offset + getattr(group, datnames[i]),
                     label=name)
        
        # location of text in the x-axis
        xloc = axes[0].set_xlim()[0] + 0.98*ptp(axes[0].set_xlim())
        #axes[0].text(xloc, yvar[0] + (yvar[1] + i)*offset, name, fontsize=fontsize, ha='right')

    # panel 2: plotting data and fitted model
    axes[1].plot(getattr(group, xvar), yvar[2]*offset+group.scan, label=group.scangroup)
    axes[1].plot(getattr(group, xvar), yvar[2]*offset+group.fit , label='fit')
    axes[1].plot(getattr(group, xvar), group.min_pars.residual, label='residual')
    
    
    # increasing y-lim to include legend
    yloc = axes[1].get_ylim()
    axes[1].set_ylim(yloc[0], 1.1*yloc[1])
    
    if annotate:
        # summary results for plot
        summary = r'$\chi^2$ = %1.4f' % group.min_pars.chisqr +'\n'
        for i, var in enumerate(group.min_pars.params):
            val = group.min_pars.params[var].value
            err = group.min_pars.params[var].stderr
            summary += group.refgroups[i]+r': %1.2f$\pm$%1.2f' % (val, err)
            summary += '\n'

        if fig_type == 'dxanes':
            yloc = yloc[0] + 0.6*ptp(yloc)
        elif fig_type == 'xanes':
            yloc = yloc[0] + 0.4*ptp(yloc)
        else:
            yloc = yloc[0] + 0.4*ptp(yloc)

        axes[1].text(xloc, yloc, summary, ha='right', va='center', fontsize=fontsize)

    # axes decorators
    axline_kws = {'color': 'lightgray', 'dashes': [4,2], 'linewidth': 1.0, 'zorder':-2}
    axes[1].axhline(0, **axline_kws)
    for ax in axes:
        ax.set_yticks([])
        ax.legend(edgecolor='k', fontsize=fontsize)
        # fit range decorations
        for val in group.lcf_pars['fit_range']:
            ax.axvline(val, **axline_kws)

    return(fig, axes)

if __name__ == '__main__':
    import doctest
    doctest.testmod()