#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import chain
from re import search
from typing import List, Optional
from numpy import ndarray, linspace, inf
from . import Group, Report

class Collection(object):
    """Collection storage class.
    
    This class stores a collection of :class:`~araucaria.main.group.Group` objects.
    
    Parameters
    ----------
    name
        Name for the collection. The default is None.
    
    Attributes
    ----------
    tags : :class:`dict`
        Dictionary with available groups in the collection based on tag keys.
    
    Notes
    -----
    Each group will be stored as an attribute of the collection.
    The ``tags`` attribute classifies group names based on a 
    ``tag`` key, which is useful for joint manipulation of groups.
    
    Warning
    -------
    Each group can only have a single ``tag`` key. 
    
    Example
    -------
    >>> from araucaria import Collection
    >>> collection = Collection()
    >>> type(collection)
    <class 'araucaria.main.collection.Collection'>
    """
    def __init__(self, name: str=None):
        if name is None:
            name  = hex(id(self))
        self.name = name
        self.tags: dict = {}

    def __repr__(self):
        if self.name is not None:
            return '<Collection %s>' % self.name
        else:
            return '<Collection>'
    
    def add_group(self, group: Group, tag: str='scan') -> None:
        """Adds a group dataset to the Collection.
        
        Parameters
        ----------
        group
            The data group to add to the Collection.
        tag
            Key for the ``tags`` attribute of the Collection.
            The default is 'scan'.
        
        Returns
        -------
        :
        
        Raises
        ------
        TypeError
            If ``group`` is not a valid Group instance.
        ValueError
            If ``group.name`` is already in the Collection.

        Example
        -------
        >>> from araucaria import Collection, Group
        >>> from araucaria.utils import check_objattrs
        >>> collection = Collection()
        >>> g1 = Group(**{'name': 'group1'})
        >>> g2 = Group(**{'name': 'group2'})
        >>> for group in (g1, g2):
        ...     collection.add_group(group)
        >>> check_objattrs(collection, Collection, attrlist=['group1','group2'])
        [True, True]
        
        >>> # using tags
        >>> g3 = Group(**{'name': 'group3'})
        >>> collection.add_group(g3, tag='ref')
        >>> for key, value in collection.tags.items():
        ...     print(key, value, type(value))
        scan ['group1', 'group2'] <class 'list'>
        ref ['group3'] <class 'list'>
        """
        if not isinstance(group, Group):
            raise TypeError('group is not a valid Group instance.')
        name = group.name
        if name in self.get_names():
            raise ValueError('group name already in the Collection.')
        else:
            setattr(self, name, group)
        
        # updating tags
        if tag in self.tags:
            self.tags[tag].append(name)
            self.tags[tag].sort()
        else:
            self.tags[tag] = [name]

    def rename_group(self, name: str, newname: str) -> None:
        """Renames a group in the Collection.
        
        Parameters
        -----------
        name
            Name of group to modify.
        newname
            New name for the group.
        
        Returns
        -------
        :
        
        Raises
        ------
        AttributeError
            If ``name`` is not a group in the Collection.
        TypeError
            If ``newname`` is not a string.
        
        Example
        -------
        >>> from araucaria import Collection, Group
        >>> collection = Collection()
        >>> g1   = Group(**{'name': 'group1'})
        >>> g2   = Group(**{'name': 'group2'})
        >>> for i, group in enumerate([g1, g2]):
        ...     collection.add_group(group)
        >>> collection.rename_group('group1', 'group3')
        >>> print(collection.get_names())
        ['group2', 'group3']
        >>> print(collection.group3.name)
        group3
        """
        if not hasattr(self, name):
            raise AttributeError('collection has no %s group.' % name)
        elif not isinstance(newname, str):
            raise TypeError('newname is not a valid string.')
        else:
            self.__dict__[newname] = self.__dict__.pop(name)
            
            # retrieving original tag key
            for key, val in self.tags.items():
                if name in val:
                    tag = key
                    break
            
            # replacing record name with new name
            self.tags[tag].remove(name)
            self.tags[tag].append(newname)
            self.tags[tag].sort()
            
            # modifying name of group
            self.__dict__[newname].name = newname

    def get_tag(self, name) -> str:
        """Returns tag for a group in the Collection.

        Parameters
        ----------
        name
            Name of group to retrieve tag.

        Returns
        -------
        :
            Tag of the group.

        Raises
        ------
        TypeError
            If ``name`` is not in a group in the Collection.
        
        Example
        -------
        >>> from araucaria import Collection, Group
        >>> collection = Collection()
        >>> g1   = Group(**{'name': 'group1'})
        >>> g2   = Group(**{'name': 'group2'})
        >>> tags = ('scan', 'ref')
        >>> for i, group in enumerate([g1, g2]):
        ...     collection.add_group(group, tag=tags[i])
        >>> print(collection.get_tag('group1'))
        scan
        >>> print(collection.get_tag('group2'))
        ref
        """
        if not hasattr(self, name):
            raise AttributeError('collection has no %s group.' % name)
        
        # retrieving original tag key
        for key, val in self.tags.items():
            if name in val:
                tag = key
                break

        return tag

    def retag(self, name: str, tag: str) -> None:
        """Modifies tag of a group in the Collection.
        
        Parameters
        ----------
        name
            Name of group to modify.
        tag
            New tag for the group.
        
        Returns
        -------
        :
        
        Raises
        ------
        AttributeError
            If ``name`` is not a group in the Collection.
        
        Example
        -------
        >>> from araucaria import Collection, Group
        >>> collection = Collection()
        >>> g1   = Group(**{'name': 'group1'})
        >>> g2   = Group(**{'name': 'group2'})
        >>> tags = ('scan', 'ref')
        >>> for i, group in enumerate([g1, g2]):
        ...     collection.add_group(group, tag=tags[i])
        >>> collection.retag('group1', 'ref')
        >>> for key, value in collection.tags.items():
        ...     print(key, value)
        ref ['group1', 'group2']
        """        
        # retrieving original tag key
        initag = self.get_tag(name)
        
        if initag == tag:
            # nothing needs to be changed
            return
        else:
            # removing groupname from original tag
            self.tags[initag].remove(name)
            
            # removing entire key if group list is empty
            if not self.tags[initag]:
                del self.tags[initag]

            # reassigning groupname to new tag
            if tag in self.tags:
                self.tags[tag].append(name)
                self.tags[tag].sort()
            else:
                self.tags[tag] = name

    def get_group(self, name) -> Group:
        """Returns a group dataset from the Collection.
        
        Parameters
        ----------
        name
            Name of group to retrieve.
        
        Returns
        -------
        :
            Requested group.
        
        Raises
        ------
        TypeError
            If ``name`` is not in a group in the Collection.

        Example
        -------
        >>> from araucaria import Collection, Group
        >>> from araucaria.utils import check_objattrs
        >>> collection = Collection()
        >>> g1    = Group(**{'name': 'group1'})
        >>> collection.add_group(g1)
        >>> gcopy = collection.get_group('group1')
        >>> check_objattrs(gcopy, Group)
        True
        >>> print(gcopy.name)
        group1
        """
        if not hasattr(self, name):
            raise AttributeError('collection has no %s group.' % name)
        
        return getattr(self, name)

    def get_names(self, taglist: List[str]=['all']) -> List[str]:
        """Returns group names in the Collection.
        
        Parameters
        ----------
         taglist
            List with keys to filter groups in the Collection based 
            on the ``tags`` attribute. The default is ['all'].

        Returns
        -------
        :
            List with names in the Collection.

        Raises
        ------
        ValueError
            If any item in ``taglist`` is not a key of the ``tags`` attribute.
        
        Example
        -------
        >>> from araucaria import Collection, Group
        >>> collection = Collection()
        >>> g1   = Group(**{'name': 'group1'})
        >>> g2   = Group(**{'name': 'group2'})
        >>> g3   = Group(**{'name': 'group3'})
        >>> g4   = Group(**{'name': 'group4'})
        >>> tags = ('scan', 'ref', 'ref', 'scan')
        >>> for i, group in enumerate([g1, g2, g3, g4]):
        ...     collection.add_group(group, tag=tags[i])
        >>> collection.get_names()
        ['group1', 'group2', 'group3', 'group4']
        >>> collection.get_names(taglist=['scan'])
        ['group1', 'group4']
        >>> collection.get_names(taglist=['ref'])
        ['group2', 'group3']
        """
        names = []
        iterchain  = False
        for tag in taglist:
            if tag == 'all':
                # retrieving all groups
                names = self.tags.values()
                names = [item for sublist in names for item in sublist]
                break
            elif tag not in self.tags:
                raise ValueError('%s is not a valid key for the collection.')
            else:
                # retrieving selected tag
                names = names + self.tags[tag]

        names.sort()
        return names

    def del_group(self, name) -> None:
        """Removes a group dataset from the Collection.
        
        Parameters
        ----------
        name
            Name of group to remove.
        
        Returns
        -------
        :
        
        Raises
        ------
        TypeError
            If ``name`` is not in a group in the Collection.

        Example
        -------
        >>> from araucaria import Collection, Group
        >>> from araucaria.utils import check_objattrs
        >>> collection = Collection()
        >>> g1 = Group(**{'name': 'group1'})
        >>> g2 = Group(**{'name': 'group2'})
        >>> for group in (g1, g2):
        ...     collection.add_group(group)
        >>> check_objattrs(collection, Collection, attrlist=['group1','group2'])
        [True, True]
        >>> collection.del_group('group2')
        >>> check_objattrs(collection, Collection, attrlist=['group1','group2'])
        [True, False]
        >>> # verifying that the deleted group has no tag
        >>> for key, value in collection.tags.items():
        ...     print(key, value)
        scan ['group1']
        """
        if not hasattr(self, name):
            raise AttributeError('collection has no %s group.' % name)

        # retrieving original tag key
        for key, val in self.tags.items():
            if name in val:
                initag = key
                break

        # removing groupname from original tag
        self.tags[initag].remove(name)

        # removing entire key if group list is empty
        if not self.tags[initag]:
            del self.tags[initag]

        # removing group
        delattr(self, name)

    def summary(self, taglist: List[str]=['all'], regex: str=None,
                optional: Optional[list]=None, **pre_edge_kws:dict) -> Report:
        """Returns a summary report of groups in a Collection.

        Parameters
        ----------
        taglist
            List with keys to filter groups in the Collection based 
            on the ``tags`` attribute. The default is ['all'].
        regex
            Search string to filter results by group name. See Notes for details.
            The default is None.
        optional
            List with optional parameters. See Notes for details.
            The default is None.
        pre_edge_kws
            Dictionary with arguments for :func:`~araucaria.xas.normalize.pre_edge`.

        Returns
        -------
        :
            Report for datasets in the HDF5 file.

        Raises
        ------
        ValueError
            If any item in ``taglist`` is not a key of the ``tags`` attribute.

        Notes
        -----
        Summary data includes the following:

        1. Group index.
        2. Group name.
        3. Group tag.
        4. Measurement mode.
        5. Numbers of scans.
        6. Absorption edge step :math:`\Delta\mu(E_0)`, if ``optional=['edge_step']``.
        7. Absorption threshold energy :math:`E_0`, if ``optional=['e0']``.
        8. Merged scans, if ``optional=['merged_scans']``.
        9. Optional parameters if they exist as attributes in the group.

        A ``regex`` value can be used to filter group names based
        on a regular expression (reges). For valid regex syntax, please 
        check the documentation of the module :mod:`re`.

        The number of scans and names of merged files are retrieved 
        from the ``merged_scans`` attribute of ``collection``.

        The absorption threshold and the edge step are retrieved by 
        calling the function :func:`~araucaria.xas.normalize.pre_edge`.

        Optional parameters will be retrieved from the groups as 
        attributes. Currently only :class:`str`, :class:`float` or
        :class:`int` will be retrieved. Otherswise an empty character
        will be printed in the report.

        See also
        --------
        :class:`~araucaria.main.report.Report`

        Examples
        --------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_all_hdf5
        >>> fpath      = get_testpath('test_database.h5')
        >>> collection = read_all_hdf5(fpath)
        >>> # printing default summary
        >>> report = collection.summary()
        >>> report.show()
        =================================
        id  dataset       tag   mode  n  
        =================================
        1   dnd_testfile  scan  mu    3  
        2   p65_testfile  scan  mu    2  
        3   xmu_testfile  scan  mu    1  
        =================================
    
        >>> # printing summary of dnd file with merged scans
        >>> report = collection.summary(regex='dnd', optional=['e0', 'merged_scans'])
        >>> report.show()
        ==========================================================
        id  dataset       tag   mode  n  e0     merged_scans      
        ==========================================================
        1   dnd_testfile  scan  mu    3  29203  dnd_test_001.dat  
                                                dnd_test_002.dat  
                                                dnd_test_003.dat  
        ==========================================================
    
        >>> # printing custom summary
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria import Collection
        >>> from araucaria.io import read_xmu
        >>> fpath = get_testpath('xmu_testfile.xmu')
        >>> # extracting mu and mu_ref scans
        >>> group_mu = read_xmu(fpath, scan='mu')
        >>> # adding additional attributes
        >>> group_mu.symbol = 'Zn'
        >>> group_mu.temp   = 25.0
        >>> # saving in a collection
        >>> collection = Collection()
        >>> collection.add_group(group_mu)
        >>> report = collection.summary(optional=['symbol','temp'])
        >>> report.show()
        ===================================================
        id  dataset           tag   mode  n  symbol  temp  
        ===================================================
        1   xmu_testfile.xmu  scan  mu    1  Zn      25    
        ===================================================
        """
        from ..xas import pre_edge

        # list with parameter names
        field_names = ['id', 'dataset', 'tag', 'mode', 'n']
        opt_list    = ['merged_scans', 'edge_step', 'e0']

        if pre_edge_kws == {}:
            # default values
            pre_edge_kws={'pre_range':[-150,-50], 'nnorm':3, 'post_range':[150, inf]}

        # verifying optional values
        if optional is not None:
            for opt_val in optional:
                field_names.append(opt_val)

        # instanciating report class
        report   = Report()
        report.set_columns(field_names)

        # number of records
        names = self.get_names(taglist=taglist)
        if regex is None:
            pass
        else:
            index = []
            for i, name in enumerate(names):
                if search(regex, name) is None:
                    pass
                else:
                    index.append(i)
            names = [names[i] for i in index]
        ncols = len(names)

        for i, name in enumerate(names):
            data    = self.get_group(name)
            scanval = data.get_mode()
            tag     = self.get_tag(name)
            extra_content = False  # aux variable for 'merged_scans'
            try:
                # number of merged_scans
                nscans = len(data.merged_scans)
            except:
                nscans = 1

            field_vals = [i+1, name, tag, scanval, nscans]
            if optional is not None:
                for j, opt_val in enumerate(optional):
                    if opt_val == 'merged_scans':
                        if i == 0:
                            # storing the col merge_index
                            merge_index = len(field_vals)
                        try:
                            list_scans = data.merged_scans
                            field_vals.append(data.merged_scans[0])
                            extra_content = True
                        except:
                            field_vals.append('None')

                    elif opt_val in opt_list[1:]:
                        out = pre_edge(data, **pre_edge_kws)
                        field_vals.append(out[opt_val])
                    else:
                        # custom optional field
                        try:
                            val = getattr(data, opt_val)
                            if isinstance(val, (int, float, str)):
                                # if val is int or float print it
                                field_vals.append(val)
                            else:
                                field_vals.append('')
                        except:
                            field_vals.append('')
            report.add_row(field_vals)
        
            if extra_content:
                for item in list_scans[1:]:
                    field_vals = []
                    for j,index in enumerate(field_names):
                        if j !=  merge_index:
                            field_vals.append('')
                        else:
                            field_vals.append(item)
                    report.add_row(field_vals)
                if i < (ncols - 1):
                    report.add_midrule()

        return report

    def get_mcer(self, num: int=None, taglist: List[str]=['all']) -> ndarray:
        """Returns the minimum common energy range for the Collection.

        Parameters
        ----------
        num
            Number of data points for the energy array.
        taglist
            List with keys to filter groups in the Collection based 
            on the ``tags`` attribute. The default is ['all'].

        Returns
        -------
        :
            Array containing the minimum common energy range

        Raises
        ------
        AttributeError
            If ``energy`` is not an attribute of the requested groups.
        ValueError
            If any item in ``taglist`` is not a key of the ``tags`` attribute.

        Notes
        -----
        By default the returned array contains the lowest number of points
        available in the minimum common energy range of the groups.
        
        Providing a value for ``num`` will return the desired number 
        of points for the minimum common energy range.
        
        Examples
        --------
        >>> from numpy import linspace
        >>> from araucaria import Collection, Group
        >>> collection = Collection()
        >>> g1   = Group(**{'name': 'group1', 'energy': linspace(1000, 2000, 6)})
        >>> g2   = Group(**{'name': 'group2', 'energy': linspace(1500, 2500, 11)})
        >>> tags = ('scan', 'ref')
        >>> for i, group in enumerate([g1, g2]):
        ...     collection.add_group(group, tag=tags[i])
        >>> # mcer for tag 'scan'
        >>> print(collection.get_mcer(taglist=['scan']))
        [1000. 1200. 1400. 1600. 1800. 2000.]
        >>> # mcer for tag 'ref'
        >>> print(collection.get_mcer(taglist=['ref']))
        [1500. 1600. 1700. 1800. 1900. 2000. 2100. 2200. 2300. 2400. 2500.]
        
        >>> # mcer for 'all' groups
        >>> print(collection.get_mcer())
        [1600. 1800. 2000.]
        >>> # mcer for 'all' groups explicitly
        >>> print(collection.get_mcer(taglist=['scan', 'ref']))
        [1600. 1800. 2000.]

        >>> # mcer with given number of points
        >>> print(collection.get_mcer(num=11))
        [1500. 1550. 1600. 1650. 1700. 1750. 1800. 1850. 1900. 1950. 2000.]
        """
        # retrieving list with group names
        names = self.get_names(taglist=taglist)
        
        for item in names:
            if not hasattr(getattr(self, item), 'energy'):
                raise AttributeError('%s has no energy attribute.' % item)

        # finding the maximum of minimum energy values
        emc_min = max([getattr(self, item).energy[0] for item in names])
        # finding the minimum of maximum energy values
        emc_max = min([getattr(self, item).energy[-1] for item in names])
        
        if num is not None:
            # returning a formatted array
            earray = linspace(emc_min, emc_max, num)
        else:
            # returning an array with the least ammount of points
            for i, item in enumerate(names):
                energy = getattr(self, item).energy
                energy = energy[(energy >= emc_min ) & (energy <= emc_max)]
                if i == 0:
                    earray = energy
                else:
                    if len(energy) < len(earray):
                        earray = energy
        return earray

if __name__ == '__main__':
    import doctest
    doctest.testmod()
