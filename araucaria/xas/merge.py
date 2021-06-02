#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xas.merge` module offers the following functions to pre-process 
and merge scans:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`calibrate`
     - Calibrates the absorption threshold energy of a scan.
   * - :func:`align`
     - Aligns a scan with respect to a reference.
   * - :func:`merge`
     - Merge groups in a collection.
"""
from warnings import warn
from typing import Tuple, List
from numpy import column_stack, mean, where, gradient, sum
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from .. import Group, Collection, Report
from .normalize import find_e0
from ..utils import check_objattrs

def calibrate(group: Group, e0: float, update: bool=True) -> float:
    """Calibrates the absorption threshold energy of the reference scan.

    Parameters
    ----------
    group
        Group containing the spectrum to calibrate.
    e0
        Arbitrary value for the absorption threshold.
    update
        Indicates if the group should be updated following calibration.
        The default is True.

    Returns
    -------
    :
        Energy difference between ``e0`` and the initial energy threshold.

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``energy`` does not exist in ``group``.
    AttributeError
        If attribute ``mu_ref`` does not exist in ``group``.

    Notes
    -----
    Calibration is performed by offsetting the ``group.energy`` attribute in order
    to match the absorption threshold energy of ``group.mu_ref`` with the given ``e0`` 
    value.

    If ``update=True`` the following attributes of ``group`` will be modified or created:

    - ``group.energy``: modified by the ``e_offset`` value.
    - ``group.e_offset``:  difference between ``e0`` and the initial| threshold energy.

    If ``update=False`` the ``e_offset`` value will be returned but not stored
    in ``group``.

    Warning
    -------
    If ``e_offset`` already exists in the provided ``group``, the ``group.energy`` array will be 
    reverted to its original values before performing calibration with the new ``e0`` value.
    
    See also
    --------
    ~araucaria.xas.normalize.find_e0 : Finds the absorption threshold value.

    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_dnd
    >>> from araucaria.xas import calibrate
    >>> fpath = get_testpath('dnd_testfile1.dat')
    >>> group_mu = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
    >>> e0       = 29200  # threshold energy for calibration
    >>> e_offset = calibrate(group_mu, e0, update=False)  # energy offset
    >>> print('%1.3f' % e_offset)
    -3.249
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['energy', 'mu_ref'], exceptions=True)

    # auxiliary energy array
    energy = group.energy
    
    if hasattr(group, 'e_offset'):
        warn('group was already aligned or calibrated: resetting energy to original value.')
        energy   = group.energy - group.e_offset

    # auxiliary group
    calgroup = Group(**{'energy': energy, 'mu_ref': group.mu_ref})
    e_offset = e0-find_e0(calgroup)
    
    if update:
        # updating the group
        group.e_offset = e_offset
        group.energy   = energy + e_offset

    return e_offset

def align(group: Group, refgroup: Group, offset: float=0., window: list=[-50,50], 
          use_mu_ref: bool=True, update: bool=True) -> float:
    """Aligns the scan of a data group with respect to a reference.

    Parameters
    ----------
    group
        Group containing the spectrum to align.
    refgroup
        Reference group for alignment.
    offset
        Initial energy offset for alignment. The default is 0.
    window
        Energy window with respect to `e0` to perform the alignent.
        The detaulf is [-50, 50].
    use_mu_ref
        Indicates if the reference scan of each group should be used 
        for alignment. The default is True.
    update
        Indicates if the group should be updated following alignment.
        The default is True.
    
    Returns
    -------
    :
        Energy offset for the group after alignment.

    Raises
    ------
    TypeError
        If either ``group`` or ``refgroup`` is not a valid Group instance.
    AttributeError
        If attribute ``energy`` does not exist in either ``group`` or ``refgroup``.
    AttributeError
        If attribute ``mu_ref`` does not exist in either ``group`` or ``refgroup``
        when ``use_mu_ref = True``.

    Notes
    -----
    The energy alignment is performed by minimizing the squared residuals between the
    derivative scans of ``group`` and ``refgroup``, considering an energy window computed
    with respect to the absorption threshold `e0` of ``refgroup``.
    
    If ``use_mu_ref=True`` the ``group.mu_ref`` array will be aligned with respect
    to the ``refgroup.mu_ref`` array. This is the default behavior.
    
    If ``use_mu_ref=False`` the scan attributes of ``group`` and ``refgroup`` will be 
    used for alignment, as determined by the 
    :func:`~araucaria.main.group.Group.get_mode` method.

    If ``update=True`` the following attributes of ``group`` will be modified or created:
    
    - ``group.energy``: modified by the ``e_offset`` value.
    - ``group.e_offset``:  energy offset following the alignment.

    If ``update=False`` the ``e_offset`` will be returned but not stored
    in ``group``.

    Important
    ---------
    The energy alignment performs a local optimization that depends on the 
    initial ``offset`` parameter. Therefore a judicious selection of ``offset`` 
    is required in order to obtain meaningful results.

    Warning
    -------
    If ``e_offset`` already exists in the provided ``group``, the ``group.energy`` 
    array will be reverted to its original values before performing alignment.

    See also
    --------
    ~araucaria.xas.normalize.find_e0 : Finds the absorption threshold value.

    Examples
    --------
    >>> from numpy import allclose
    >>> from araucaria import Group
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_dnd
    >>> from araucaria.xas import align
    >>> fpath = get_testpath('dnd_testfile1.dat')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_dnd(fpath, scan='mu')
    >>> # creating a reference group with an offset
    >>> offset = 3.125
    >>> ref = Group(**{'energy': group_mu.energy + offset, 
    ...                'mu': group_mu.mu, 
    ...                'mu_ref': group_mu.mu_ref})
    >>> # aligning with respect to 'mu_ref'
    >>> e_offset = align(group_mu, ref, update=False)
    >>> allclose(offset, e_offset)
    True
    
    >>> # aligning with respect to 'mu'
    >>> offset = 4.233
    >>> ref = Group(**{'energy': group_mu.energy + offset, 
    ...                'mu': group_mu.mu})
    >>> e_offset = align(group_mu, ref, use_mu_ref=False, update=False)
    >>> allclose(offset, e_offset)
    True
    """    
    vals = ['group', 'refgroup']
    for i, obj in enumerate((group, refgroup)):
        if not isinstance(obj, Group):
            raise TypeError('%s is not a valid Group instance.' % vals[i])
        
        if hasattr(obj, 'mu_ref') is False and use_mu_ref:
            raise AttributeError("%s has no 'mu_ref' attribue." % vals[i])

        if hasattr(obj, 'energy') is False:
            raise AttributeError("%s has no 'enery' attribute." % vals[i])

    # auxiliary energy array
    energy = group.energy

    if hasattr(group, 'e_offset'):
        warn('group already aligned or calibrated. Resetting energy to original value.')
        energy   = group.energy - group.e_offset

    # calculation of e0 to determine the optimization window
    # definition of reference spline
    if use_mu_ref:
        e0         = find_e0(refgroup, use_mu_ref=True, update=False)
        ref_spline = interp1d(refgroup.energy, refgroup.mu_ref, kind='cubic')
    else:
        scantype   = refgroup.get_mode()
        e0         = find_e0(refgroup, update=False)
        ref_spline = interp1d(refgroup.energy, getattr(refgroup, scantype), kind='cubic')

    min_lim = e0 + window[0]
    max_lim = e0 + window[1]
    
    # calculation of energy points for interpolation and comparison
    # this energy array is static
    index = where((refgroup.energy >= min_lim) & (refgroup.energy <= max_lim))
    ref_energy = refgroup.energy[index]

    # both the reference mu and the objective mu
    # exist in the same energy grid as the reference
    ref_dmu = gradient(ref_spline(ref_energy))/gradient(ref_energy)

    # the objective function is the difference of derivatives
    if use_mu_ref:
        scantype = 'mu_ref'
    else:
        scantype = group.get_mode()

    def objfunc(x):
        obj_spline = interp1d(energy + x, getattr(group, scantype), kind='cubic')
        obj_dmu = gradient(obj_spline(ref_energy))/gradient(ref_energy)
        return (sum((ref_dmu - obj_dmu)**2))

    e_offset = fmin(objfunc, offset, disp=False)[0]

    if update:
        group.e_offset = e_offset
        group.energy = energy + e_offset

    return e_offset

def merge(collection: Collection, taglist: List[str]=['all'], 
          name: str='merge', only_mu_ref: bool=False) -> Tuple[Report, Group]:
    """Merge groups in a collection.

    Parameters
    ----------
    collection
        Collection with the groups to be merged.
    taglist
        List with keys to filter groups to be merged based 
        on their ``tags`` attributes in the Collection.
        The default is ['all'].
    name
        Name for the merged group.
        The default is 'merge'.
    only_mu_ref
        Indicates if only the reference scans should be merged.
        The default is False.

    Return
    ------
    report
        Report for the merged scan.
    group
        Group containing the merged scan.

    Raises
    ------
    TypeError
        If ``collection`` is not a valid Collection instance.
    AttributeError
        If ``collection`` has no ``tags`` attribute.
    AttributeError
        If attribute ``mu_ref`` does not exist in the selected groups.
    ValueError
        If any item in ``taglist`` is not a key of the ``tags`` attribute.

    Warning
    -------
    If only one group in ``collection`` is selected for merge, a None report 
    and the single group will be returned.
    
    If different scan types are being merged, the scan attribute of the merge 
    group will be labeled ``mu``.

    Notes
    -----
    If ``only_mu_ref=False`` the scan arrays of the selected groups will be 
    merged, as determined by the :func:`~araucaria.main.group.Group.get_mode` 
    method. The ``mu_ref`` arrays of the selected groups will also be merged 
    separately. This is the detault behavior.
    
    If ``only_mu_ref=True`` only the ``mu_ref`` arrays of the selected groups 
    will be merged.
    
    The following attribute will be created for the returned group:
    
    - ``group.merged_scans``: list with the merged scan files.
    
    See also
    --------
    :func:`~araucaria.plot.fig_merge.fig_merge`: Plots merged scans.

    Example
    -------
    >>> from araucaria import Collection
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_dnd
    >>> from araucaria.xas import merge
    >>> collection = Collection()
    >>> files = ['dnd_testfile1.dat' , 'dnd_testfile2.dat', 'dnd_testfile3.dat']
    >>> for file in files:
    ...     fpath = get_testpath(file)
    ...     group_mu = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
    ...     collection.add_group(group_mu)         # adding group to collection
    >>> report, mgroup = merge(collection)
    >>> report.show()
    ===================================================
    id  filename           mode  e_offset[eV]  e0[eV]  
    ===================================================
    1   dnd_testfile1.dat  mu    0             29203   
    2   dnd_testfile2.dat  mu    0             29203   
    3   dnd_testfile3.dat  mu    0             29203   
    ---------------------------------------------------
        merge              mu    0             29203   
    ===================================================
    """
    # checking class and attributes
    check_objattrs(collection, Collection, attrlist=['tags'], exceptions=True)

    listgroups = collection.get_names(taglist=taglist)

    # initializing report
    names  = ['id', 'filename', 'mode', 'e_offset[eV]', 'e0[eV]']
    report = Report()
    report.set_columns(names)
    
    for item in listgroups:
        if not hasattr(collection.get_group(item), 'mu_ref'):
            raise AttributeError("%s has no 'mu_ref' attribute.")

    if len(listgroups) == 1:
        warn('single group selected for merge. Returning that group.')
        return (None, collection.get_group(listgroups[0]))

    energy   = collection.get_mcer(taglist=taglist)
    scanlist = []  # container for scan types

    for i, item in enumerate(listgroups):
        group    = collection.get_group(item)
        # storing energy offset
        try:
            e_offset = group.e_offset
        except:
            e_offset = 0

        # interpolating and storing the reference channel
        mu_ref_spline= interp1d(group.energy, group.mu_ref, kind='cubic')
        if i == 0:
            mu_ref = mu_ref_spline(energy)
        else:
            mu_ref = column_stack((mu_ref, mu_ref_spline(energy)))

        # interpolating and storing the scan
        if not only_mu_ref:
            mode = group.get_mode()
            scanlist.append(mode)
            mu_spline = interp1d(group.energy, getattr(group,mode), kind='cubic')
            if  i == 0:
                mu = mu_spline(energy)
            else:
                mu = column_stack((mu, mu_spline(energy)))

            # storing data on report
            e0 = find_e0(group, update=False)
            report.add_row([i+1, group.name, mode, e_offset, e0])
        else:
            e0 = find_e0(group, use_mu_ref=True, update=False)
            report.add_row([i+1, group.name, 'mu_ref', e_offset, e0])
        
    mu_ref_avg = mean(mu_ref, axis=1)    
    
    # adding midrule
    report.add_midrule()
    
    if not only_mu_ref:
        # calculating the average of the spectra
        mu_avg = mean(mu, axis=1)
        if all(item == mode for item in scanlist):
            merge = Group(**{'name':name, 'energy':energy, mode:mu_avg,
                             'mu_ref':mu_ref_avg})
            mergescan = mode
        else:
            merge = Group(**{'name':name, 'energy':energy, 'mu':mu_avg, 
                             'mu_ref':mu_ref_avg})
            mergescan = 'mu'
        
        # storing data on report
        e0 = find_e0(merge, update=False)
        report.add_row(['',name, mergescan, 0.0, e0])
        
    else:
        merge = Group(**{'name':name, 'energy':energy, 'mu_ref':mu_ref_avg})
        
        # storing data on report
        e0 = find_e0(merge, use_mu_ref=True, update=False)
        report.add_row(['',name, 'mu_ref', 0.0, e0])

    setattr(merge, 'merged_scans', listgroups)
    return (report, merge)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
