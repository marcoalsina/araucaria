#!/usr/bin/python
# -*- coding: utf-8 -*-
__DOC__="""
List of classes and functions implemented here:

class              description
-----              -----------
DataReport         stores user-refined information for print to stdout.
Group              generic container for data.

function           desciption
--------           ----------
index_dups         index of duplicate values.
get_scan_type      scan type of XAS dataset. 
xftf_pha           phase-corrected FFT for XAS dataset.
"""

class Group(object):
    """Data storage class.
    
    This class stores XAS datasets, variables and subgroups.
    """

    def __init__(self, name=None, **kws):
        if name is None:
            name = hex(id(self))
        self.__name__ = name
        for key, val in kws.items():
            setattr(self, key, val)

    def __repr__(self):
        if self.__name__ is not None:
            return '<Group %s>' % self.__name__
        else:
            return '<Group>'


class DataReport:
    """Data Report class.
    
    This class stores user-defined information of XAS spectra for convenient print to *stdout*.

    Attributes
    ----------
    decimal : int, optional
        Printed decimals for floats (the default is 3).
    marker: str, optional
        Character for row separator (the default is '=').
    
    Notes
    -----
    - Column types can be specified with the method :func:`~main.ClassReport.set_columns`.
    - Content is added *row wise* with the method :func:`~main.ClassReport.add_content`.
    - Content is printed to `stdout` with the method :func:`~main.ClassReport.show`.
    """

    def __init__(self, name=None):
        if name is None:
            name = hex(id(self))
        self.__name__= name
        self.decimal = 3    # decimals for float types
        self.content = ''   # container for contents of report
        self.marker  = '='  # separator marker
        self.cols    = None
    
    def __repr__(self):
        if self.__name__ is not None:
            return '<DataReport %s>' % self.__name__
        else:
            return '<DataReport>'    
    
    def set_columns(self, **pars):
        """Sets parameters for each printed column.
        
        This method sets the character length and title 
        of each column that will be printed.
        
        Optional parameters include the number of decimals,
        and the row separator marker.

        Parameters
        ----------
        cols : list, `float`
            List with the length for each column field.
        names : list, `str`
            List with the names for each column field.
        decimal : 'int', optional
            Decimal point for floats (the default is 3).
        marker : 'str', optional
            Character for row separator (the default is "=").
            
        Returns
        -------
        None
        
        Raises
        -----
        ValueError
            If the length of ´cols´ is different than the length of ´names´.
        """

        self.cols  = pars['cols']
        self.names = pars['names']
        try:    
            self.decimal = pars['decimal']
            self.marker  = pars['marker']
        except:
            pass
        
        if len(self.cols) != len(self.names):
            raise ValueError ('Length of columns is different than length of names.')
        
        self.ncols   = len(self.cols)
        self.row_len = sum(self.cols)
    
    def add_content(self, content):
        """Adds a row of content to the report.

        Parameters
        ----------
        content : list, `str`
            List of values for each column in a report row.
        
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If the length of ´content´ is different than the length of ´cols´ in :func:`~main.ClassReport.set_columns`.
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
            Character for the midrule (the default is "-").

        Returns
        -------
        None
        """
        
        self.content += marker*self.row_len
        self.content += '\n'        

    def show(self, header=True, endrule=True):
        """Prints the report to *stdout*.

        Parameters
        ----------
        header : bool, optional
            Prints a header rule (the default is `True`).
        endrule: bool, optional
            Prints an end rule (the default is `True`).
        
        Returns
        -------
        None
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
    """Index of duplicate values.

    This utility function returns an index array with 
    consecutive energy duplicates.

    Parameters
    ----------
    energy : ndarray
        Energy array to analyze for duplicates.
    tol : float
        Tolerance value to identify duplicate values (the detault is 1e-4).

    Returns
    -------
    index : ndarray
        Index array containing the location of duplicates.
        
    Notes
    -----
    - Consecutive energy values are considered duplicates if their absolute difference is below the given ´tol´ value.
    - The index can be used to remove data from channels of the XAS dataset.
    """

    from numpy import argwhere, diff

    dif = diff(energy)
    index = argwhere(dif < tol)

    return (index+1)

def get_scan_type(data):
    """Returns XAS scan type.
    
    This function returns the scan types for a given XAS dataset:
    
    - "mu_ref": refers to the reference scan. Returned only if no other scans are present.
    - "mu": refers to a transmision mode scan.
    - "fluo": refers to a fluorescence mode scan.
    
    Parameters
    ----------
    data: group
        Larch group containing the XAS dataset.

    Returns
    -------
    scan: str
        "fluo", "mu", or "mu_ref".
    
    Raises
    ------
    ValueError
        If the scan type is not recognized.
    
    Warning
    -------
    The scan type is provided during reading of the XAS dataset, and should follow the convention indicated here.
    """
    
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
    """Phase-corrected magnitude of FFT XAFS spectrum.
    
    This function writes the phase corrected 
    magnitude of the forward XAFS fourier-transform 
    for the data and, if available, the FEFFIT model.
    
    Parameters
    ----------
    group: group
        Larch group containing the FFT XAFS data.
    path: group
        Larch FEFF path to extract the phase shift
    kmult int
        k-multiplier of the XAFS data.

    Returns
    -------
    None
    
    Notes
    -----
    The following data is appended to the parsed Larch group:
    
    - group.data.chir_pha_mag.
    - group.model.chir_pha_mag (optional).
    """
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