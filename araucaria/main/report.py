#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Optional
from numpy import dtype, ndarray, array, vstack

class Report:
    """Report class.
    
    Stores user-defined information for print to ``sys.stdout``.

    Parameters
    ----------
    name
        Name of the instance.
    sep
        Separation between printed columns. The default is 2.
    sdigits
        Significant digits for printed floats. The default is 5.
    marker
        Character for header and endrule. The default is '='.

    Example
    -------    
    >>> from araucaria import Report
    >>> report = Report()
    >>> type(report)
    <class 'araucaria.main.report.Report'>
    >>> report.sep
    2
    >>> report.sdigits
    5
    >>> report.marker
    '='
    """
    
    def __init__(self, name: Optional[str] = None, sep: int=2, 
                 sdigits: int=5, marker: str='='):
        # setting objects attributes
        if name is None:
            name = hex(id(self))
        self.name    = name
        self.sep     = sep     # separation for columns
        self.sdigits = sdigits # significant digits for float types
        self.marker  = marker  # separator marker
        self.content = None    # container for report
        
    
    def __repr__(self):
        if self.name is not None:
            return '<Report %s>' % self.name
        else:
            return '<Report>'    
    
    def set_columns(self, names: list) -> None:
        """Sets parameters for each printed column.

        Parameters
        ----------
        names
            List with the names for each column field.
        
        Returns
        -------
        :

        Example
        -------
        >>> from araucaria import Report
        >>> report = Report()
        >>> names  = ['Name', 'Description']
        >>> report.set_columns(names)
        >>> report.names
        ['Name', 'Description']
        """
        self.names = names
        self.ncols = len(self.names)
        self.nrows = 0
    
    def add_row(self, row: list) -> None:
        """Adds a row of content to the report.

        Content can be accesed through the ``self.content`` attribute.

        Parameters
        ----------
        row
            List with values for each column in a single report row.

        Returns
        -------
        :

        Raises
        ------
        AttributeError
            If ``names`` has not been set with :func:`~Report.set_columns`.
        TypeError
            If ``row`` is not a list.
        IndexError
            If the length of ``row`` is different than the 
            length of ``names`` given in :func:`~Report.set_columns`.

        Example
        -------
        >>> from araucaria import Report
        >>> report = Report()
        >>> names  = ['Name', 'Description']
        >>> report.set_columns(names)
        >>> report.add_row(['filename 1', 'a single scan'])
        >>> report.content
        array(['filename 1', 'a single scan'], dtype=object)
        """
        if self.names is None:
            raise AttributeError("names have not been set.")
        elif type(row) is not list: 
                raise TypeError('content is not a list.')
        elif len(row) != self.ncols:
            raise IndexError('content length does not match number of columns.')

        row_format = []
        for j in range(self.ncols):
            try:
                # check if value is float
                val          = float(row[j])
                float_format = '{:<.%ig}' % (self.sdigits)
                val          = float_format.format(val)
                row_format.append(val)
            except:
                row_format.append(row[j])

        if self.content is None:
            self.content = array(row_format, dtype=object)
        else:
            self.content = vstack((self.content, array(row_format)))
        self.nrows += 1

    def add_midrule(self, marker: str = '-') -> None:
        """Adds a midrule to the report.

        Parameters
        ----------
        marker
            Character for the midrule. The default is '-'.

        Returns
        -------
        :
        
        Example
        -------
        >>> from araucaria import Report
        >>> report = Report()
        >>> names  = ['Name', 'Description']
        >>> report.set_columns(names)
        >>> report.add_midrule()
        >>> report.content
        array(['-', '-'], dtype=object)
        """
        midrule = [marker] * len(self.names)
        if self.content is None:
            self.content = array(midrule, dtype=object)
        else:
            self.content = vstack((self.content, midrule))
        self.nrows += 1

    def get_cols(self, names: list=['all'], astype: dtype=object) -> ndarray:
        """Returns columns of the report by name.

        Parameters
        ----------
        names
            List with names of columns to return.
            The default is ['all'], which returns the 
            entire report content.
        astype
            Data type conversion for the returned array.
            The default is :class:`object`.

        Returns
        -------
        :
            Array with the requested columns.
        
        Raises
        ------
        ValueError
            If ``names`` does not contain valid column names.

        Example
        -------
        >>> import random
        >>> from araucaria import Report
        >>> random.seed(1111)
        >>> report = Report()
        >>> names  = ['Name', 'Value']
        >>> report.set_columns(names)
        >>> for i in range(1,4):
        ...     report.add_row(['filename %s'% i, random.random()])
        >>> col = report.get_cols(names=['Value',])
        >>> print(col)
        ['0.2176' '0.34438' '0.64225']
        
        >>> # returning column as float
        >>> col = report.get_cols(names=['Value',], astype=float)
        >>> print(col)
        [0.2176  0.34438 0.64225]
        >>> print(col.dtype)
        float64
        """
        cindex = []
        if names == ['all']:
            return self.content
        else:
            for name in names:
                if name not in self.names:
                    raise ValueError('%s is not a valid column name.')
                else:
                    cindex.append(self.names.index(name))
            if len(cindex) == 1:
                cindex = cindex[0]
        return self.content[:, cindex].astype(astype)

    def get_rows(self, index: list=['all'], astype: dtype=object) -> ndarray:
        """Returns rows of the report by index.

        Parameters
        ----------
        index
            List with indexes of rows to return.
            The default is ['all'], which returns the 
            entire report content.
        astype
            Data type conversion for the returned array.
            The default is :class:`object`.

        Returns
        -------
        :
            Array with the requested rows.
        Example
        -------
        >>> import random
        >>> from araucaria import Report
        >>> random.seed(1111)
        >>> report = Report()
        >>> names  = ['Name', 'Value']
        >>> report.set_columns(names)
        >>> for i in range(1,4):
        ...     report.add_row(['filename %s'% i, random.random()])
        >>> row = report.get_rows(index=[1,])
        >>> print(row)
        ['filename 2' '0.34438']
        """
        if index == ['all']:
            return self.content
        else:
            if len(index) == 1:
                index = index[0]
            return self.content[index,:].astype(astype)

    def show(self, header: bool=True, endrule: bool=True, 
             print_report: bool=True) -> Optional[str]:
        """Returns the formatted report.

        Parameters
        ----------
        header
            Prints a header rule and column names. The default is True.
        endrule
            Prints an end rule. The default is True.
        print_report
             Prints the report to ``sys.stdout``.
             If False the formatted report is returned as a :class:`str`
             The default is True.
            
        Returns
        -------
        :
            Formatted report. Returned only if ``print_report=False``.
        
        Example
        -------
        >>> import random
        >>> from araucaria import Report
        >>> random.seed(1111)
        >>> report = Report()
        >>> names  = ['Name', 'Value']
        >>> report.set_columns(names)
        >>> for i in range(1,4):
        ...     report.add_row(['filename %s'% i, random.random()])
        >>> report.show()
        =====================
        Name        Value    
        =====================
        filename 1  0.2176   
        filename 2  0.34438  
        filename 3  0.64225  
        =====================
        """
        linesep    = '\n'
        
        # calculating column sizes
        name_sizes     = [len(name) for name in self.names]

        if self.nrows == 0:
            col_sizes = [0 for i in range(self.ncols)]
            content   = []
        elif self.nrows == 1:
            col_sizes = [len(self.content[i]) for i in range(self.ncols)]
            content   = [self.content, ]
        else:
            col_sizes = [len(max(self.content[:,i], key=len)) for i in range(self.ncols)]
            content   = self.content

        self.col_sizes = []
        content_format = ''
        
        for i in range(self.ncols):
            self.col_sizes.append(max(col_sizes[i], name_sizes[i]) + self.sep)
            content_format += '{%i:<%i}' % (i, self.col_sizes[-1])
        
        # formatted content
        # gests reseted with each call to show()
        print_str      = ''

        for i, row in enumerate(content):
            # content format gets reseted for each row
            print_row      = []  # container of printed values
            if all(row == row[0]):
                # midrule condition
                print_row = [row[0]*size for size in self.col_sizes]
            else:
                print_row = row
            
            # conditional to print str with linesep
            if (i + 1) == self.nrows:
                print_str += content_format.format(*print_row)
            else:
                print_str += content_format.format(*print_row) + linesep

        separator = self.marker*sum(self.col_sizes)
        if header:
            header_format = ''
            header_str    = ''
            for i,col in enumerate(self.names):
                header_format += '{%i:<%i}' % (i, self.col_sizes[i])
            
            header_str = separator + linesep
            header_str += header_format.format(*self.names) + linesep
            header_str += separator + linesep
            print_str = header_str + print_str

        if endrule:
            if self.nrows == 0:
                print_str += separator
            else:
                print_str += linesep + separator
    
        if print_report:
            print(print_str)
            return
        else:
            return print_str

if __name__ == '__main__':
    import doctest
    doctest.testmod()