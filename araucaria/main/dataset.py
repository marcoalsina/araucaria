#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import annotations

class Dataset(object):
    """Dataset storage class.

    This class stores a dataset analysis of a Group or Collection.

    Parameters
    ----------
    name
        Name for the Dataset. The default is None.
    kwargs
        Dictionary with content for the Dataset.

    Example
    -------
    >>> from araucaria import Dataset
    >>> dataset = Dataset()
    >>> type(dataset)
    <class 'araucaria.main.dataset.Dataset'>
    """
    def __init__(self, name: str=None, **kwargs:dict):
        if name is None:
            name  = hex(id(self))
        self.name = name
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        if self.name is not None:
            return '<Dataset %s>' % self.name
        else:
            return '<Dataset>'

if __name__ == '__main__':
    import doctest
    doctest.testmod()