#!/usr/bin/python
# -*- coding: utf-8 -*-

def merge_spectra(fpaths, scantype='mu', ftype='dnd', align_kws=None, 
                  print_report=True, write_kws=None):
    """Merge XAS scans.

    This function ...
    
    Parameters
    ----------
    fpaths : list
    scantype : str
    ftype : str
    align_kws : dict
    write_kws : dict
    
    Returns
    -------
    group : list of larch groups
    merge : larch group
    
    """
    from os import path
    import numpy as np
    import larch
    from larch import Group
    import pyxas.io
    from .merge import align_scans, merge_scans, merge_report
    
    # supporting reading formats
    format_dict = {'dnd' : 'read_dnd', 
                   'xmu' : 'read_xmu' }
    
    # required align/write keys
    req_keys = ['name', 'dbpath']
    
    # testing that file type is supported    
    if ftype not in format_dict:
        raise ValueError("file type %s currently not supported")

    # testing that files exist in the given path 
    for fpath in fpaths:
        if not path.isfile(fpath):
            raise IOError('file %s does not exists.' % fpath)
    
    # checking alignment keys from input
    req_keys = ['name', 'dbpath']
    if align_kws is not None:
        for key in req_keys:
            if key not in align_kws:
                raise ValueError ("Either 'name' or 'dbpath' key is missing in the align dictionary.")
            else:
                align = True
        
        # setting initial energy offset for alignment routine
        try:
            e_offset = align_kws['e_offset']
        except:
            e_offset = 0.0    
        
        # reading reference scan
        ref = Group(**pyxas.io.read_hdf5(align_kws['dbpath'], align_kws['name']))
    else:
        align = False

    # loading larch session
    session = larch.Interpreter(with_plugins=False)

    # reading files
    group = []
    for fpath in fpaths:
        # retrieving reading function
        read_func = getattr(pyxas.io, format_dict[ftype])
        data      = read_func(fpath, scantype)
        if align:
            align_scans(data, ref, session, e_offset=e_offset)
        
        data.name = path.split(fpath)[1]
        group = np.append(group, data)

    # merging scans
    if len(fpaths) > 1:
        merge = merge_scans(group, scantype)
    else:
        merge = data

    # saving list of merged scans as attribute
    merge.merged_scans = str([data.name for data in group])

    # print merge report
    if print_report:
        report = merge_report(group, merge)
        report.show()

    # write merge group in a hdf5 database
    # checking write keys from input
    if write_kws is not None:
        for key in req_keys:
            if key not in write_kws:
                raise ValueError ("Either 'name' or 'dbpath' key is missing in the write dictionary.")
        # setting replace option in write_hdf5
        try:
            replace = write_kws['replace']
        except:
            replace = False

        pyxas.io.write_hdf5(write_kws['dbpath'], merge, 
                            name = write_kws['name'], replace=replace)

    return(group, merge)

def merge_ref(fpaths, e0, ftype='dnd', e_offset=0.0, 
              print_report=True, write_kws=None):
    """Merge XAS reference scans.

    This function ...
    
    Parameters
    ----------
    fpaths : list
    e0 : float
    ftype : str
    e_offset : float
    print_report : bool
    write_kws : dict
    
    Returns
    -------
    group : list of larch groups
    merge : larch group
    
    """
    from os import path
    import numpy as np
    import larch
    from larch import Group
    import pyxas.io
    from .merge import calibrate, align, merge, merge_report
    
    # supporting reading formats
    format_dict = {'dnd' : 'read_dnd', 
                   'xmu' : 'read_xmu' }
    
    # required write keys
    req_keys = ['name', 'dbpath']
    
    # testing that file type is supported    
    if ftype not in format_dict:
        raise ValueError("file type %s currently not supported")

    # testing that files exist in the given path 
    for fpath in fpaths:
        if not path.isfile(fpath):
            raise IOError('file %s does not exists.' % fpath)

    # loading larch session
    session = larch.Interpreter(with_plugins=False)
    
    # retrieving reading function
    read_func = getattr(pyxas.io, format_dict[ftype])
    
    # reading reference file
    # the scan with the lowest e_offset is selected as the reference
    for fpath in fpaths:
        data = read_func(fpath,scantype)
        calibrate(data, e0, session)
        
        # the first file is the first iteration
        if fpath == fpaths[0]:
            ref   = data
            e_ref = data.e_offset
        
        # ref file is modified if e_offset is smaller
        elif abs(data.e_offset) < abs(e_ref):
            ref   = data
            e_ref = ref.e_offset

    # reading files
    group = []
    for fpath in fpaths:
        data = read_func(fpath, scantype)
        align_scans(data, ref, session, e_offset=e_offset)
        data.name = path.split(fpath)[1]
        group     = np.append(group, data)

    # merging scans
    if len(fpaths) > 1:
        merge = merge_scans(group, scantype)
    else:
        merge = data

    # saving list of merged scans as attribute
    merge.merged_scans = str([data.name for data in group])

    # print merge report
    if print_report:
        report = merge_report(group, merge)
        report.show()

    # write merge group in a hdf5 database
    # checking write keys from input
    if write_kws is not None:
        for key in req_keys:
            if key not in write_kws:
                raise ValueError ("Either 'name' or 'dbpath' key is missing in the write dictionary.")
        # setting replace option in write_hdf5
        try:
            replace = write_kws['replace']
        except:
            replace = False

        pyxas.io.write_hdf5(write_kws['dbpath'], merge, 
                            name = write_kws['name'], replace=replace)

    return(group, merge)