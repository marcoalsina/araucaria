#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations
from copy import deepcopy

class Group(object):
    """Group storage class.

    This class stores a single XAFS dataset.
    
    Parameters
    ----------
    name
        Name for the group. The default is None.
    kwargs
        Dictionary with content for the group.

    Notes
    -----
    The following methods are currently implemented:

    .. list-table::
       :widths: auto
       :header-rows: 1

       * - Method
         - Description
       * - :func:`add_content`
         - Adds content to the group.
       * - :func:`copy`
         - Returns a copy of the group.
       * - :func:`get_mode`
         - Returns the scan mode of the group.
       * - :func:`has_ref`
         - Tests if the group has a reference scan.
       * - :func:`rename`
         - Renames the group.

    Example
    -------
    >>> from araucaria import Group
    >>> group = Group()
    >>> type(group)
    <class 'araucaria.main.group.Group'>
    """
    def __init__(self, name: str=None, **kwargs:dict):
        if name is None:
            name  = hex(id(self))
        self.name = name
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        if self.name is not None:
            return '<Group %s>' % self.name
        else:
            return '<Group>'

    def add_content(self, content: dict) -> None:
        """Adds content to the group.
        
        Parameters
        ----------
        content
            Dictionary with content for the group.
        
        Returns
        -------
        :
        
        Raises
        ------
        TypeError
            If ``content`` is not a dictionary.

        Example
        -------
        >>> from araucaria import Group
        >>> from araucaria.utils import check_objattrs
        >>> content = {'var': 'xas'}
        >>> group   = Group()
        >>> group.add_content(content)
        >>> check_objattrs(group, Group, attrlist=['name', 'var'])
        [True, True]
        """
        if not isinstance(content, dict):
            raise TypeError('content is not a valid dictionary.')
        else:
            for key, val in content.items():
                setattr(self, key, val)

    def copy(self) -> Group:
        """Returns a deep copy of the group.

        Parameters
        ----------
        None

        Returns
        -------
        :
            Copy of the group.


        Example
        -------
        >>> from numpy import allclose
        >>> from araucaria import Group
        >>> content = {'energy': [1,2,3,4,5,6]}
        >>> group1  = Group()
        >>> group1.add_content(content)
        >>> group2 = group1.copy()
        >>> allclose(group1.energy, group2.energy)
        True
        """
        return deepcopy(self)

    def get_mode(self) -> str:
        """Returns scan mode of mu(E) for the group.

        Parameters
        ----------
        None

        Returns
        -------
        :
            Scan mode of mu(E). Either 'fluo', 'mu', or 'mu_ref'.
    
        Raises
        ------
        ValueError
            If the scan mode is unavailable or not recognized.
    
        Important
        ---------
        The scan mode of mu(E) is assigned during reading of a file, 
        and should adhere to the following convention:
        
        - ``mu`` corresponds to a transmision mode scan.
        - ``fluo`` corresponds to a fluorescence mode scan.
        - ``mu_ref`` corresponds to a reference scan.
        
        Examples
        --------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_xmu
        >>> fpath = get_testpath('xmu_testfile.xmu')
        >>> # extracting mu and mu_ref scans
        >>> group_mu = read_xmu(fpath, scan='mu')
        >>> group_mu.get_mode()
        'mu'

        >>> # extracting only fluo scan
        >>> group_fluo = read_xmu(fpath, scan='fluo', ref=False)
        >>> group_fluo.get_mode()
        'fluo'

        >>> # extracting only mu_ref scan
        >>> group_ref = read_xmu(fpath, scan=None, ref=True)
        >>> group_ref.get_mode()
        'mu_ref'
        """
        scanlist = ['mu', 'fluo', 'mu_ref']
        scan = None

        for scantype in scanlist:
            if scantype in dir(self):
                scan = scantype
                break

        if scan is None:
            raise ValueError('scan type unavailable or not recognized.')

        return scan

    def has_ref(self) -> bool:
        """Tests if the group contains a reference scan for mu(E).

        Parameters
        ----------
        None

        Returns
        -------
        :
            True if attribute ``mu_ref`` exists in the group. False otherwise.
        
        Examples
        --------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_xmu
        >>> fpath = get_testpath('xmu_testfile.xmu')
        >>> # extracting mu and mu_ref scans
        >>> group_mu = read_xmu(fpath, scan='mu')
        >>> group_mu.has_ref()
        True
    
        >>> # extracting only fluo scan
        >>> group_fluo = read_xmu(fpath, scan='fluo', ref=False)
        >>> group_fluo.has_ref()
        False
    
        >>> # extracting only mu_ref scan
        >>> group_ref = read_xmu(fpath, scan=None, ref=True)
        >>> group_ref.has_ref()
        True
        """
        if 'mu_ref' in dir(self):
            return True
        else:
            return False

    def rename(self, newname: str) -> None:
        """Renames the group.
        
        Parameters
        ----------
        newname
            New name for the group.
        
        Returns
        -------
        :
        
        Raises
        ------
        TypeError
            If ``newname`` is not a string.

        Example
        -------
        >>> from araucaria import Group
        >>> content = {'name': 'group1'}
        >>> group   = Group(name = 'group1')
        >>> print(group.name)
        group1
        >>> group.rename('group2')
        >>> print(group.name)
        group2
        """
        if not isinstance(newname, str):
            raise TypeError('newname is not a valid string.')
        else:
            self.name = newname

if __name__ == '__main__':
    import doctest
    doctest.testmod()