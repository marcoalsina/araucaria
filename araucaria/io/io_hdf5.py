#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.io.io_hdf5` submodule offers the following functions to read, write and 
manipulate data in the Hierarchical Data Format ``HDF5``:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Function
     - Description
   * - :func:`read_hdf5`
     - Reads a single group dataset from an HDF5 file.
   * - :func:`read_collection_hdf5`
     - Reads multiple group datasets from an HDF5 file.
   * - :func:`write_hdf5`
     - Writes a single group dataset in an HDF5 file.
   * - :func:`write_collection_hdf5`
     - Writes a collection in an HDF5 file.
   * - :func:`rename_dataset_hdf5`
     - Renames a group dataset in an HDF5 file.
   * - :func:`delete_dataset_hdf5`
     - Deletes a group dataset in an HDF5 file.
   * - :func:`summary_hdf5`
     - Returns a summary of datasets in an HDF5 file.
"""
from os.path import isfile
from ast import literal_eval
from re import search
from pathlib import Path
from typing import Optional, Union
from numpy import ndarray, inf
from h5py import File, Dataset
from .. import Group, Report, Collection
from ..xas import pre_edge

def read_hdf5(fpath: Path, name: str)-> Group:
    """Reads a single group dataset from an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    name
        Dataset name to retrieve from the HDF5 file.

    Returns
    -------
    :
        Group containing the requested dataset.

    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If ``name`` does not exist in the HDF5 file.
    
    Example
    -------
    >>> from araucaria import Group
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.utils import check_objattrs
    >>> from araucaria.io import read_hdf5
    >>> fpath = get_testpath('Fe_database.h5')
    >>> # extracting geothite scan
    >>> group_mu = read_hdf5(fpath, name='Goethite_20K')
    >>> check_objattrs(group_mu, Group, attrlist=['mu', 'mu_ref'])
    [True, True]
    """    
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "r")
    else:
        raise IOError("file %s does not exists." % fpath)

    if name in hdf5:
        data = {}
        for key, record in hdf5.get(name).items():
            if isinstance(record, Dataset):
                # verifying strings saved as bytes
                if isinstance(record[()], bytes):
                    # converting bytes record to proper type
                    eval_record = convert_bytes_hdf5(record)
                    data[key] = eval_record
                else:
                    data[key]=record[()]

    else:
        hdf5.close()
        raise ValueError("%s does not exists in %s!" % (name, fpath))

    hdf5.close()

    # writting group and saving name
    group = Group(**data)
    group.name = name
    return (group)

def read_collection_hdf5(fpath: Path, names: list=['all'])-> Collection:
    """Reads multiple group datasets from an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    names
        List with group datasets to read.

    Returns
    -------
    :
        Collection containing the requested datasets.

    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If the requested ``names`` do not exist in the HDF5 file.

    Warning
    -------
    The HDF5 file does not store the ``tags`` attribute of a Collection.
    Therefore the returned collection will automatically assign 
    ``tag='scan'`` to each group dataset.
    
    Example
    -------
    >>> from araucaria import Collection
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.utils import check_objattrs
    >>> from araucaria.io import read_collection_hdf5
    >>> fpath = get_testpath('Fe_database.h5')
    >>> # reading database
    >>> collection = read_collection_hdf5(fpath)
    >>> check_objattrs(collection, Collection)
    True
    >>> collection.get_names()
    ['FeIISO4_20K', 'Fe_Foil', 'Ferrihydrite_20K', 'Goethite_20K']
    
    >>> # read selected group datasets
    >>> collection = read_collection_hdf5(fpath, names=['Fe_Foil'])
    >>> collection.get_names()
    ['Fe_Foil']
    """    
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "r")
    else:
        raise IOError("file %s does not exists." % fpath)
    
    if names == ['all']:
        names = [name for name in hdf5]
    else:
        for name in names:
            if name not in hdf5:
                raise ValueError("group %s does not exists in the HDF5 file." % name)

    collection = Collection()
    for name in names:
        data = {}
        for key, record in hdf5.get(name).items():
            if isinstance(record, Dataset):
                # verifying strings saved as bytes
                if isinstance(record[()], bytes):
                    # converting bytes record to proper type
                    eval_record = convert_bytes_hdf5(record)
                    data[key] = eval_record
                else:
                    data[key]=record[()]
        group = Group(**data)
        group.name = name
        collection.add_group(group)
    hdf5.close()

    return (collection)

def convert_bytes_hdf5(record: Dataset) -> Union[dict, list, str]:
    """Utility function to convert a :class:`bytes` record from an HDF5 file.
    
    Returned value will be either a :class:`dict`, :class:`list`, 
    or :class:`str`.
    
    Parameters
    ----------
    record : 
        HDF5 dataset record.        
    
    Returns
    -------
    :
        Converted record.

    Raises
    ------
    TypeError:
        If  value stored inside ``record`` is not of type 
        :class:`bytes`.

    Notes
    -----
    ``araucaria`` stores :class:`dict` or :class:`list` records 
    as :class:`bytes` in the HDF5 file.
    Such records  need to be converted back to their original 
    types during reading.
    """
    if not isinstance(record[()], bytes):
        raise TypeError ('record %s is not of type bytes.' % record)
    
    conv_types = (dict, list)
    opt_types  = (int, float, str)
    record_str = record.asstr()[()]
    try:
        eval_record = literal_eval( record_str )
    except:
        eval_record = record.asstr()[()]
     
    if isinstance(eval_record, conv_types):
        # return either dict or list
        return eval_record
    elif isinstance(eval_record, opt_types):
        # int, float types were originally stored as str
        return record_str

def write_hdf5(fpath: Path, group: Group, name: str=None, 
               replace: bool=False) -> None:
    """Writes a group dataset in an HDF5 file.

    Parameters
    ----------
    fpath
        Path to HDF5 file.
    group
        Group to write in the HDF5 file.
    name
        Name for the dataset in the HDF5 file.
        The default is None, which preserves the 
        original group name.
    replace
        Replace previous dataset. The default is False.

    Returns
    -------
    :
    
    Raises
    ------
    IOError
        If dataset cannot be written to the HDF5 file.
    TypeError
        If ``group`` is not a valid Group instance.
    ValueError
        If ``name`` dataset already exists in the HDF5 file and ``replace=False``.

    Notes
    -----
    If the file specified by ``fpath`` does not exists, it will be automatically created.
    If the file already exists then the dataset will be appended.

    By default the write operation will be canceled if ``name`` already exists in the HDF5 file.
    The previous dataset can be overwritten with the option ``replace=True``. 

    Example
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_hdf5
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # saving a new hdf5 file
    >>> write_hdf5('database.h5', group_mu, name='xmu_testfile', replace=True)
    xmu_testfile written to database.h5.
    """    
    # testing that the group exists 
    if type(group) is not Group:
        raise TypeError('%s is not a valid Group instance.' % group)

    # verifying existence of path:
    # (a)ppend to existing file
    # (w)rite to new file.
    if isfile(fpath):
        hdf5 = File(fpath, "a")
    else:
        hdf5 = File(fpath, "w")

    # testing name on the dataset
    if name is None:
        name = group.name
    if name in hdf5:
        # dataset present in the file
            if replace:
                hdf5.__delitem__(name)
            else:
                hdf5.close()
                raise ValueError("%s already exists in %s." % (name, fpath))

    dataset = hdf5.create_group(name)
    try:
        write_recursive_hdf5(dataset, group)
        print("%s written to %s." % (name, fpath))
    except:
        hdf5.__delitem__(name)
        hdf5.close()
        raise IOError("%s couldn't be written to %s." % (name, fpath))

    hdf5.close()
    return

def write_collection_hdf5(fpath: Path, collection: Collection, 
                          names: list=['all'], replace: bool=False) -> None:
    """Writes a collection in an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    collection
        Collection to write in the HDF5 file.
    names
        List with group dataset names to write in the HDF5 file.
    replace
        Replace previous dataset. The default is False.

    Returns
    -------
    :
    
    Raises
    ------
    IOError
        If dataset cannot be written to the HDF5 file.
    ValueError
        If ``names`` dataset does not exist in the colleciton.
    ValueError
        If ``names`` dataset already exists in the HDF5 file and ``replace=False``.

    Notes
    -----
    If the file specified by ``fpath`` does not exists, it will be automatically created.
    If the file already exists then the datasets in the collection will be appended.

    By default the write operation will be canceled if any ``names`` dataset in ``collection``
    already exists in the HDF5 file.
    Previous datasets can be overwritten with the option ``replace=True``. 

    Warning
    -------
    The ``tags`` attribute of the ``collection`` will not be stored in the HDF5 file.
    
    Example
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_collection_hdf5, write_collection_hdf5
    >>> fpath = get_testpath('Fe_database.h5')
    >>> # reading database
    >>> collection = read_collection_hdf5(fpath)
    >>> # saving collection in a new hdf5 file
    >>> write_collection_hdf5('database.h5', collection, replace=True)
    FeIISO4_20K written to database.h5.
    Fe_Foil written to database.h5.
    Ferrihydrite_20K written to database.h5.
    Goethite_20K written to database.h5.

    >>> # write selected group dataset
    >>> write_collection_hdf5('database.h5', collection, names=['Fe_Foil'], replace=True)
    Fe_Foil written to database.h5.
    """    
    # testing that the group exists 
    if type(collection) is not Collection:
        raise TypeError('%s is not a valid Collection instance.' % collection)

    # veryfying requested group names to write
    all = collection.get_names()
    if names == ['all']:
        names = all
    else:
        for name in names:
            if name not in all:
                raise ValueError('%s group is not in the Collection.' % name)
    
    # verifying existence of path:
    # (a)ppend to existing file
    # (w)rite to new file.
    if isfile(fpath):
        hdf5 = File(fpath, "a")
    else:
        hdf5 = File(fpath, "w")

    for name in names:    
        # testing name on the HDF5 file
        if name in hdf5:
            # dataset present in the file
                if replace:
                    hdf5.__delitem__(name)
                else:
                    hdf5.close()
                    raise ValueError("%s already exists in %s." % (name, fpath))
    
        dataset = hdf5.create_group(name)
        group   = collection.get_group(name)
        try:
            write_recursive_hdf5(dataset, group)
            print("%s written to %s." % (name, fpath))
        except:
            hdf5.__delitem__(name)
            hdf5.close()
            raise IOError("%s couldn't be written to %s." % (name, fpath))

    hdf5.close()
    return

def write_recursive_hdf5(dataset: Dataset, group: Group) -> None:
    """Utility function to write a Group recursively in an HDF5 file.
    
    Parameters
    ----------
    dataset
        Dataset in the HDF5 file.
    group
        Group to write in the HDF5 file.

    Returns
    -------
    :
    
    Warning
    -------
    Only :class:`str`, :class:`float`, :class:`int` and :class:`~numpy.ndarray` 
    types are currently supported for recursive writting in an HDF5 :class:`~h5py.Dataset`.
    
    :class:`dict` and :class:`list` types will be convertet to :class:`str`, which is in
    turn saved as :class:`bytes` in the HDF5 database.
    If read with :func:`read_hdf5`, such records will be automatically converted to their
    original type in the group.
    
    """
    # accepted type variables for recursive writting
    accepted_types  = (str, float, int, ndarray)
    converted_types = (dict, list)
    
    for key in dir(group):
        if '__' not in key:
            record =getattr(group,key)
            #vtype = type(record).__name__
        
            if isinstance(record, accepted_types):
                dataset.create_dataset(key, data=record)
            
            elif isinstance(record, converted_types):
                dataset.create_dataset(key, data=str(record))
    return

def rename_dataset_hdf5(fpath: Path, name: str, newname: str) -> None:
    """Renames a dataset in an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    name
        Name of Group dataset.
    newname
        New name for Group dataset.

    Returns
    -------
    :
    
    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If ``name`` dataset does not exist in the HDF5 file.
    ValueError
        If ``newname`` dataset already exists in the HDF5 file.
    
    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_hdf5, rename_dataset_hdf5
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # saving a new hdf5 file
    >>> write_hdf5('database.h5', group_mu, name='xmu_testfile', replace=True)
    xmu_testfile written to database.h5.
    >>> # renaming dataset
    >>> rename_dataset_hdf5('database.h5', 'xmu_testfile', 'xmu_renamed')
    xmu_testfile renamed to xmu_renamed in database.h5.
    """
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "a")
    else:
        raise IOError("file %s does not exists." % fpath)
    
    if newname in hdf5:
        hdf5.close()
        raise ValueError('%s already exists in %s' % (newname, fpath))
    
    # verifying existence of datagroup
    if name in hdf5:
        hdf5[newname] = hdf5[name]
    else:
        hdf5.close()
        raise ValueError("%s does not exists in %s." % (name, fpath))
       
    hdf5.__delitem__(name)
    hdf5.close()
    print ("%s renamed to %s in %s." % (name, newname, fpath))
    return
    
def delete_dataset_hdf5(fpath: Path, name: str) -> None:
    """Deletes a dataset from an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    name
        Name of dataset to delete.
    
    Returns
    -------
    :
    
    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If ``name`` dataset does not exist in the HDF5 file.
    
    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_hdf5, rename_dataset_hdf5
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # saving a new hdf5 file
    >>> write_hdf5('database.h5', group_mu, name='xmu_testfile', replace=True)
    xmu_testfile written to database.h5.
    >>> # deleting dataset
    >>> delete_dataset_hdf5('database.h5', 'xmu_testfile')
    xmu_testfile deleted from database.h5.
    """
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "a")
    else:
        hdf5.close()
        raise IOError("File %s does not exists." % fpath)
        
    # verifying existence of datagroup
    if name in hdf5:
        hdf5.__delitem__(name)
        hdf5.close()
        print ("%s deleted from %s." % (name, fpath))
    else:
        hdf5.close()
        raise ValueError ("%s does not exists in %s." % (name, fpath))
    return

def summary_hdf5(fpath: Path, regex: str=None, optional: Optional[list]=None, 
                 **pre_edge_kws:dict) -> Report:
    """Returns a summary report of datasets in an HDF5 file.

    Parameters
    ----------
    fpath
        Path to HDF5 file.
    regex
        Search string to filter results by dataset name. See Notes for details.
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
    IOError
        If the HDF5 file does not exist in the specified path.      

    Notes
    -----
    Summary data includes the following:

    1. Dataset index.
    2. Dataset name.
    3. Measurement mode.
    4. Numbers of scans.
    5. Absorption edge step :math:`\Delta\mu(E_0)`, if ``optional=['edge_step']``.
    6. Absorption threshold energy :math:`E_0`, if ``optional=['e0']``.
    7. Merged scans, if ``optional=['merged_scans']``.
    8. Optional parameters if they exist as attributes in the dataset.

    A ``regex`` value can be used to filter dataset names based
    on a regular expression (reges). For valid regex syntax, please 
    check the documentation of the module :mod:`re`.

    The number of scans and names of merged files are retrieved 
    from the ``merged_scans`` attribute of the HDF5 dataset.

    The absorption threshold and the edge step are retrieved by 
    calling the function :func:`~araucaria.xas.normalize.pre_edge`.

    Optional parameters will be retrieved from the dataset as 
    attributes. Currently only :class:`str`, :class:`float` or
    :class:`int` will be retrieved. Otherswise an empty character
    will be printed in the report.

    See also
    --------
    :func:`read_hdf5`
    :class:`~araucaria.main.report.Report`

    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import summary_hdf5
    >>> fpath = get_testpath('Fe_database.h5')
    >>> # printing default summary
    >>> report = summary_hdf5(fpath)
    >>> report.show()
    =================================
    id  dataset           mode    n  
    =================================
    1   FeIISO4_20K       mu      5  
    2   Fe_Foil           mu_ref  5  
    3   Ferrihydrite_20K  mu      5  
    4   Goethite_20K      mu      5  
    =================================

    >>> # printing summary with merged scans of Goethite groups
    >>> report = summary_hdf5(fpath, regex='Goe', optional=['merged_scans'])
    >>> report.show()
    =======================================================
    id  dataset       mode  n  merged_scans                
    =======================================================
    1   Goethite_20K  mu    5  20K_GOE_Fe_K_240.00000.xdi  
                               20K_GOE_Fe_K_240.00001.xdi  
                               20K_GOE_Fe_K_240.00002.xdi  
                               20K_GOE_Fe_K_240.00003.xdi  
                               20K_GOE_Fe_K_240.00004.xdi  
    =======================================================

    >>> # printing custom parameters
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_hdf5
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # adding additional attributes
    >>> group_mu.symbol = 'Zn'
    >>> group_mu.temp   = 25.0
    >>> # saving a new hdf5 file
    >>> write_hdf5('database2.h5', group_mu, name='xmu_testfile', replace=True)
    xmu_testfile written to database2.h5.
    >>> report = summary_hdf5('database2.h5', optional=['symbol','temp'])
    >>> report.show()
    =========================================
    id  dataset       mode  n  symbol  temp  
    =========================================
    1   xmu_testfile  mu    1  Zn      25    
    =========================================
    """
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "r")
    else:
        raise IOError("file %s does not exists." % fpath)

    # list with parameter names
    field_names = ['id', 'dataset', 'mode', 'n']
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
    keys  = list(hdf5.keys())
    if regex is None:
        pass
    else:
        index = []
        for i, key in enumerate(keys):
            if search(regex, key) is None:
                pass
            else:
                index.append(i)
        keys = [keys[i] for i in index]
    nkeys = len(keys)

    for i, key in enumerate(keys):
        data    = read_hdf5(fpath, str(key))
        scanval = data.get_mode()
        extra_content = False  # aux variable for 'merged_scans'
        try:
            # merged_scans is saved as string, so we count the number of commas
            nscans = hdf5[key]['merged_scans'].asstr()[()].count(',') + 1
        except:
            nscans = 1

        field_vals = [i+1, key, scanval, nscans]
        if optional is not None:
            for j, opt_val in enumerate(optional):
                if opt_val == 'merged_scans':
                    if i == 0:
                        # storing the col merge_index
                        merge_index = len(field_vals)
                    try:
                        list_scans = literal_eval(hdf5[key]['merged_scans'].asstr()[()] )
                        field_vals.append(list_scans[0])
                        extra_content = True
                    except:
                        field_vals.append('None')

                elif opt_val in opt_list[1:]:
                    out = pre_edge(data, **pre_edge_kws)
                    field_vals.append(out[opt_val])
                else:
                    # custom optional field
                    try:
                        val = hdf5[key][opt_val]
                        if isinstance(val[()], (int, float)):
                            # if val is int or float print it
                            field_vals.append(val[()])
                        elif isinstance(val[()], bytes):
                            # if val is bytes we convert it and check
                            val = convert_bytes_hdf5(val)
                            if isinstance(val, str):
                                field_vals.append(val)
                            else:
                                field_vals.append('')
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
            if i < (nkeys - 1):
                report.add_midrule()

    hdf5.close()
    return report

if __name__ == '__main__':
    import os
    import doctest
    doctest.testmod()

    # removing temp files    
    for fpath in ['database.h5', 'database2.h5']:
        if os.path.exists(fpath):
            os.remove(fpath)