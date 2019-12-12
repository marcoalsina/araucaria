#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Basic functions to work with XAS spectra.
"""

class DataReport:
    """Data Report class.
    
    This class allows to print user defined information
    of the processed XAS spectra.

    Column types can be specified with the method :func:`~main.ClassReport.set_columns`.
    Content is added *row wise* with the method :func:`~main.ClassReport.add_content`.
    Content is printed to `stdout` with the method :func:`~main.ClassReport.show`.

    Attributes
    ----------
    decimal : int
        Printed decimals for float content.
    marker: str
        Character for row separator.
    """

    def __init__(self):
        self.decimal = 3    # decimals for float types
        self.content = ''   # container for contents of report
        self.marker  = '='  # separator marker
        self.cols    = None

    def set_columns(self, **pars):
        """Sets parameters for each printed column.
        
        This method sets the length and title of each
        column that will be printed.
        Optional parameters include the number of decimals,
        and the row separator marker.

        Parameters
        ----------
        cols : list, `float`
            List with the length for each column field.
        names : list, `str`
            List with the names for each column field.
        decimal : 'int', optional
            Decimal point for floats.
        marker : 'str', optional
            Character for row separation.
        """

        self.cols = pars['cols']
        self.names = pars['names']
        try:    
            self.decimal = pars['decimal']
            self.marker = pars['marker']
        except:
            pass
        
        if len(self.cols) != len(self.names):
            raise ValueError ('Columns length is different than names length!')
        
        self.ncols = len(self.cols)
        self.row_len = sum(self.cols)
    
    def add_content(self, content):
        """Adds a row of content to the report.

        Parameters
        ----------
        content : list, `str`
            List with values for each column in a report row.
            The list of values must match the col length specified
            with the method :func:`~main.ClassReport.set_columns`.
        """

        if self.cols is None:
            raise ValueError("Columns have not been set!")
        elif len(content) != self.ncols:
            raise ValueError('Content length does not match number of columns!')
        
        content_format = ''
        for i, col in enumerate(self.cols):
            if isinstance(content[i], float):
                content_format += '{%i:<%i.%if}' % (i,self.cols[i], self.decimal)
            else:
                content_format += '{%i:<%i}' % (i,self.cols[i])
        
        self.content += content_format.format(*content)
        self.content += '\n'

    def add_midrule(self, marker='-'):
        """Adds a midrule to the report.

        Parameters
        ----------
        marker : `str`
            Character to be used for the midrule.
            Default is "-".

        """
        
        self.content += marker*self.row_len
        self.content += '\n'        

    def show(self, header=True, endrule=True):
        """Prints the report to stdout.

        Parameters
        ----------
        header : bool, optional
            Controls the print of a header rule separator.
            Default value is `True`.
        endrule: bool, optional
            Controls the print of an end rule separator.
            Default value is `True`.
        """

        self.separator = self.marker*self.row_len
        if header:
            header_format = ''
            for i,col in enumerate(self.cols):
                header_format += '{%i:<%i}' % (i,self.cols[i])
            
            self.header = self.separator + '\n'
            self.header += header_format.format(*self.names) + '\n'
            self.header += self.separator
            print (self.header)
            
        if endrule:
            print (self.content+self.separator)
        else:
            print (self.content)


def index_dups(energy, tol=1e-4):
    """Index of duplicates.

    This utility function returns an index array with 
    consecutive energy duplicates.

    Consecutive energy values are considered duplicates 
    if their absolute difference is below a given tolerance.

    The returned index can be used to remove duplicates
    in the energy and other channels of the spectrum.

    Parameters
    ----------
    energy : ndarray
        Energy array to analyze for duplicates.
    tol : float
        Tolerance value to identify duplicate values.

    Returns
    -------
    index : ndarray
        Index array containing the location of duplicates.
    """

    from numpy import argwhere, diff

    dif = diff(energy)
    index = argwhere(dif < tol)

    return (index+1)

def get_scan_type(data):
    '''
    This function return the scan type for a given
    dataset.
    if only 'mu_ref' exists then it is returned.
    otherwise either 'mu' or 'fluo' are returned.
    --------------
    Required input:
    data: Larch group containing the data.
    --------------
    Output:
    scan [string]:  eiher 'fluo', 'mu', or 'mu_ref'.
    '''
    scanlist = ['mu', 'fluo', 'mu_ref']
    scan = None

    for scantype in scanlist:
        if scantype in dir(data):
            scan = scantype
            break

    try:
        scan is not None
    except ValueError:
        print ('scan type not recognized!')

    return (scan)

def xftf_pha(group, path, kmult):
    '''
    This function returns the phase corrected 
    magnitude of the forward XAFS fourier-transform 
    for the data and, if available, the FEFFIT model.
    --------------
    Required input:
    group: Larch group containing the FT XAFS data.
    path: Larch FEFF path to extract the phase shift
    kmult [int]: k-multiplier of the XAFS data.
    --------------
    Output:
    Ouput is appended to the parsed Larch group.
        group.data.chir_pha_mag
        [optional] group.model.chir_pha_mag 
    '''
    from numpy import interp, sqrt, exp
    import larch
    from larch.xafs import xftf_fast

    nrpts = len(group.data.r)
    feff_pha = interp(group.data.k, path._feffdat.k, path._feffdat.pha)
    
    # phase-corrected Fourier Transform
    data_chir_pha = xftf_fast(group.data.chi * exp(0-1j*feff_pha) *
                    group.data.kwin * group.data.k**kmult)[:nrpts] 
    group.data.chir_pha_mag = sqrt(data_chir_pha.real**2 + data_chir_pha.imag**2)

    try:
        model_chir_pha = xftf_fast(group.model.chi * exp(0-1j*feff_pha) *
                         group.model.kwin * group.model.k**kmult)[:nrpts]
        group.model.chir_pha_mag = sqrt(model_chir_pha.real**2 + model_chir_pha.imag**2)
    except:
        pass
    return
