#!/usr/bin/env python
'''
filename: io_hdf5.py

Initialization of functions to read, write and inspect HDF5 databases.

Implemented functions:
    read_hdf5
    write_hdf5
    write_recursive_hdf5
    rename_dataset_hdf5
    delete_dataset_hdf5
    summary_dataset_hdf5
    summary_scans_dataset_hdf5
'''

def read_hdf5(filename, name=None):
    '''
    This function reads a single dataset from a HDF5
    file and returns a Group with the relevant data.
    --------------
    Required input:
    filename [str]: name of the HDF5 file.
    name [str]    : data name to read.
    -------------
    Output:
    data [group]  : group data.
    '''
    import os
    import h5py
    import numpy as np
    
    # verifying existence of path:
    if os.path.isfile(filename):
        hdf5 = h5py.File(filename, "r")
    else:
        print ("Error: File %s does not exists." % filename)
        return
    
    if name is None:
        print ("Error: must specify the required dataset name.")
        return
    elif name in hdf5:
        data = {}
        #flag='not recovered'
        for key, record in hdf5.get(name).items():
            #vtype = type(record).__name__
            
            if isinstance(record, h5py._hl.dataset.Dataset):
                data[key]=record.value
                #flag='recovered'
            
            #print '%s dataset: %s <%s>' % (flag, key, vtype)
    else:
        print ("Error: requested dataset does not exists in the file!")
        return
    
    hdf5.close()
    return data

def write_hdf5(filename, data, name='dataset1', replace=False):
    '''
    This function writes a single datagroup to a HDF5 file.
    --------------
    Required input:
    filename [array]: name of the HDF5 file.
    data[group]     : group data to write.
    name [str]      : name for the group data.
    replace[bool]   : replace any previous data (optional).
    -------------
    Output:
    data [group]    : group data.
    '''
    import os
    import h5py
    
    # verifying existence of path:
    # (a)ppend to existing file
    # (w)rite to new file.
    if os.path.isfile(filename):
        hdf5 = h5py.File(filename, "a")
    else:
        hdf5 = h5py.File(filename, "w")
    
    # testing name of the dataset
    if name in hdf5:
        # dataset present in the file
            if replace:
                hdf5.__delitem__(name)
            else:
                print ("Operation cancelled: Dataset %s already exists in database %s!" % (name, filename))
                return
    
    group = hdf5.create_group(name)
    try:
        write_recursive_hdf5(group,data)
        print ('%s writen to file %s' % (name, filename))
    except:
        print ('Error: data not written to file.')
    
    hdf5.close()
    return
    
def write_recursive_hdf5(group, data):
    '''
    Helper function to write HDF5 files over nested information.
    Functions to save and recover data different than str or numeric data
    are not yet implemented.
    --------------
    Required input:
    grouo[group]       : group data to write nested information.
    data [var]         : data to write.
    -------------
    Output:
    None.
    '''
    import numpy as np

    accepted_types=(str,float, int, np.ndarray)
    
    for key in dir(data):
        record =getattr(data,key)
        vtype = type(record).__name__
        #flag='rejected'
        
        if isinstance(record, (str,float, int, np.ndarray)):
            group.create_dataset(key, data=record)
            #flag='accepted'
        #print '%s datased: %s <%s>' % (flag, key, vtype)
    return

def rename_dataset_hdf5(filename, oldname, newname):
    '''
    This function renames a dataset from a HDF5 file.
    --------------
    Required input:
    filename [str]: name of the HDF5 file.
    oldname [str] : name of the group data to rename.
    newname[str]  : replace any previous data (optional).
    -------------
    Output:
    None.
    '''
    import os
    import h5py
    
    # verifying existence of path:
    if os.path.isfile(filename):
        hdf5 = h5py.File(filename, "a")
    else:
        print ("Error: File %s does not exists." % filename)
        return
        
    # verifying existence of datagroup
    if oldname in hdf5:
        hdf5[newname] = hdf5[oldname]
    else:
        print ("Error: dataset %s does not exists in %s." % (oldname, filename))
        return
       
    hdf5.__delitem__(oldname)
    print ("Dataset %s renamed %s." % (oldname, newname))
    return
    
def delete_dataset_hdf5(filename, name):
    '''
    This function deletes a dataset from a HDF5 file.
    --------------
    Required input:
    filename [str]: name of the HDF5 file.
    name [str] : name of the group data to delete.
    -------------
    Output:
    None.
    '''
    import os
    import h5py
    
    # verifying existence of path:
    if os.path.isfile(filename):
        hdf5 = h5py.File(filename, "a")
    else:
        print ("Error: File %s does not exists." % filename)
        return
        
    # verifying existence of datagroup
    if name in hdf5:
        hdf5.__delitem__(name)
        print ("Dataset %s deleted." % name)
    else:
        print ("Error: dataset %s does not exists in %s." % (name, filename))
    return

def summary_dataset_hdf5(filename, optional=None, **pre_edge_kws):
    '''
    This function returns a summary of the XAS
    data stored in the respective HDF5 database file.
    An optional list can be given to report 'mu0' and 'E0'.
    --------------
    Required input:
    filename [str]     : name of the HDF5 file.
    optional [list]    : list containing 'mu0 and/or 'E0' (optional).
    pre_edge_kws [dict]: dictionary with pre edge normalization arguments.
    -------------
    Output:
    summary [str]: summary of the XAS data stored in the HDF5 database.
    '''
    import os
    import h5py
    import larch
    from larch import Group
    from larch.xafs import find_e0, pre_edge
    from pyxas.io import read_hdf5

    # verifying existence of path:
    if os.path.isfile(filename):
        hdf5 = h5py.File(filename, "r")
    else:
        print ("Error: File %s does not exists." % filename)
        return

    session = larch.Interpreter(with_plugins=False)
    hdf5 = h5py.File(filename, 'r')
    
    if pre_edge_kws == {}:
        # default values
        pre_edge_kws={'pre1':-150, 'pre2':-50, 'nnorm':3, 'norm1':150}
    
    field_names = ['id', 'name', 'type', 'n']
    opt_list    = ['mu0', 'e0']
    
    # fixed separator spaces
    sp_n      =  '4'
    sp_name   = '30'
    sp_type   = '15'
    sp_opt    = '10'
    float_dec =  '3'
    
    # initial separation spaces
    sp = [sp_n, sp_name, sp_type, sp_n]
    
    if optional is not None:
        for opt_val in optional:
            if opt_val not in opt_list:
                raise ValueError("Optional parameter '%s' not recognized!" % opt_val)
            else:
                field_names.append(opt_val)
                sp.append(sp_opt)
    
    tot_sep = sum([int(item) for item in sp])
    separator = '='*tot_sep
    sp_header = ''
    sp_field  = ''
    for i, sp_val in enumerate(sp):
        sp_header += '{'+str(i)+':'+sp_val+'}'
        if field_names[i] in ['id', 'n']:
            sp_field  += '{'+str(i)+':<'+sp_val+'}'
        elif field_names[i] in opt_list:
            sp_field  += '{'+str(i)+':<'+sp_val+'.'+float_dec+'f}'
        else:
            sp_field  += '{'+str(i)+':'+sp_val+'}'
        
    summary = ''
    summary += os.path.abspath(filename)+'\n'
    summary += separator+'\n'
    summary += sp_header.format(*field_names)+'\n'
    summary += separator+'\n'
    for i, key in enumerate(hdf5.keys()):
        data = read_hdf5(filename, str(key))
        data = Group(**data)
        nscans = str(hdf5[key]['merged_scans'].value).count(',') + 1
        if 'mu' in dir(data):
            scanval  ='mu'
            scantype = 'transmission'
        elif 'fluo' in dir(data):
            scanval  ='fluo'
            scantype = 'fluorescence'
        else:
            scanval  ='mu_ref'
            scantype = 'reference'
         
        field_vals = [i+1, key, scantype, nscans]
        
        if optional is not None:
            pre_edge(data.energy, getattr(data,scanval), group=data, _larch=session, **pre_edge_kws)
            for opt_val in optional:
                if opt_val == 'mu0':
                    field_vals.append(data.edge_step)
                if opt_val == 'e0':
                    field_vals.append(data.e0)
                    
        summary += sp_field.format(*field_vals)+'\n'
    
    summary += separator+'\n'
    hdf5.close()
    return (summary)

def summary_scans_dataset_hdf5(filename, names='all'):
    '''
    This function returns a summary of the merged
    scans for XAS spectrum stored in the respective 
    HDF5 database file.
    An optional list of names can be given to prevent
    delivering the entire dataset by default ('all').
    --------------
    Required input:
    filename [str] : name of the HDF5 file.
    names[list]    : list containing the group names to retrieve.
    -------------
    Output:
    summary [str]: summary of the merged scans per group names.
    '''
    import os
    import h5py
    from pyxas.io import read_hdf5

    # verifying existence of path:
    if os.path.isfile(filename):
        hdf5 = h5py.File(filename, "r")
    else:
        print ("Error: File %s does not exists." % filename)
        return

    # fixed separator spaces
    sp_n      =  '4'
    sp_name   = '30'
    sp_field  = '{0:<'+sp_n+'}{1:<'+sp_name+'}'
    sep       = '='*40
    fsep      = '-'*40
    hdf5 = h5py.File(filename, 'r')

    # initial separation spaces
    sp = [sp_n, sp_name]

    summary = ''
    summary += os.path.abspath(filename)+'\n'
    summary += sep+'\n'
    summary += 'Summary of merged scans'+'\n'
    summary += sep+'\n'
    
    if names == 'all':
        names = hdf5.keys()

    for name in names:
        if name in hdf5:
            data   = read_hdf5(filename, name)
            scans  = data['merged_scans'].split(',')
            nscans = len(scans) + 1
            
            summary += name+'\n'
            summary += fsep+'\n'
            for i, scan in enumerate(scans):
                # we remove unwanted characters ("[,],'") and spaces.
                pname = scan.strip().replace("'",'').replace('[','').replace(']','')
                summary += sp_field.format(i+1,pname)+'\n'
 
        else:
            print ("Error: requested dataset does not exists in the file!")
            return
        summary += fsep+'\n\n'

    summary += sep+'\n'
    hdf5.close()
    return (summary)
