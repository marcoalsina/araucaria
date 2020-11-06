#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import chain
from typing import List
from . import Group
from numpy import ndarray, linspace

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
    ``tag`` key, which is useful for manipulations of groups.
    
    Example
    -------
    >>> from araucaria import Collection
    >>> collection = Collection()
    >>> type(collection)
    <class 'araucaria.main.collection.Collection'>
    """
    def __init__(self, name: str=None):
        if name is None:
            name = hex(id(self))
        self.__name__   = name
        self.tags: dict = {}

    def __repr__(self):
        if self.__name__ is not None:
            return '<Collection %s>' % self.__name__
        else:
            return '<Collection>'
    
    def add_group(self, group: Group, tag: str='scan') -> None:
        """Adds a data group to the Collection.
        
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

        Example
        -------
        >>> from araucaria import Collection, Group
        >>> from araucaria.utils import check_objattrs
        >>> collection = Collection()
        >>> group1 = Group(**{'name': 'group1'})
        >>> group2 = Group(**{'name': 'group2'})
        >>> for group in (group1, group2):
        ...     collection.add_group(group)
        >>> check_objattrs(collection, Collection, attrlist=['group1','group2'])
        [True, True]
        
        >>> # using tags
        >>> group3 = Group(**{'name': 'group3'})
        >>> collection.add_group(group3, tag='ref')
        >>> for key, value in collection.tags.items():
        ...     print(key, value, type(value))
        scan ['group1', 'group2'] <class 'list'>
        ref ['group3'] <class 'list'>
        """
        if not isinstance(group, Group):
            raise TypeError('group is not a valid Group instance.')
        name = group.__name__
        setattr(self, name, group)
        
        # updating tags
        if tag in self.tags:
            self.tags[tag].append(name)
            self.tags[tag].sort()
        else:
            self.tags[tag] = [name]

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
        >>> group1     = Group(**{'name': 'group1'})
        >>> group2     = Group(**{'name': 'group2'})
        >>> tags       = ('scan', 'ref')
        >>> for i, group in enumerate([group1, group2]):
        ...     collection.add_group(group, tag=tags[i])
        >>> collection.retag('group1', 'ref')
        >>> for key, value in collection.tags.items():
        ...     print(key, value)
        ref ['group1', 'group2']
        """
        if not hasattr(self, name):
            raise AttributeError('collection has no %s group.' % name)
        
        # retrieving original tag key
        for key, val in self.tags.items():
            if name in val:
                initag = key
                break
        
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
        """Returns a data group from the Collection.
        
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
        >>> group1 = Group(**{'name': 'group1'})
        >>> collection.add_group(group1)
        >>> gcopy  = collection.get_group('group1')
        >>> check_objattrs(gcopy, Group)
        True
        >>> print(gcopy.__name__)
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
        >>> group1 = Group(**{'name': 'group1'})
        >>> group2 = Group(**{'name': 'group2'})
        >>> group3 = Group(**{'name': 'group3'})
        >>> group4 = Group(**{'name': 'group4'})
        >>> tags   = ('scan', 'ref', 'ref', 'scan')
        >>> for i, group in enumerate([group1, group2, group3, group4]):
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
        """Removes a data group from the Collection.
        
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
        >>> group1 = Group(**{'name': 'group1'})
        >>> group2 = Group(**{'name': 'group2'})
        >>> for group in (group1, group2):
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
        >>> group1     = Group(**{'name': 'group1', 'energy': linspace(1000, 2000, 6)})
        >>> group2     = Group(**{'name': 'group2', 'energy': linspace(1500, 2500, 11)})
        >>> tags       = ('scan', 'ref')
        >>> for i, group in enumerate([group1, group2]):
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
