#!/usr/bin/env python
'''
filename: main.py

Collection of functions to work with XAS data.

Implemented classes:
    DataReport

Implemented functions:
    index_dups
    read_dnd
    calibrate_energy
    align_scans
    merge_scans
    get_scan_type
    xftf_pha
'''

class DataReport:
    '''
    This class allows to print information
    about the processed XAS spectra.
    '''
    def __init__(self):
        self.decimal = 3    # decimals for float types
        self.content = ''   # container for contents of report
        self.marker = '='   # separator marker
        self.cols = None

    def set_columns(self, **pars):
        '''
        Sets length and title for each column.
        Accepted parameters include the following:
        cols [list] : Lengths for each column field.
        names [list]: Names for each column field.
        
        Optional parameters:
        decimal [int]: decimal point for floats.
        marker [str] : separator string.
        '''
        self.cols = pars['cols']
        self.names = pars['names']
        try:    
            self.decimal = pars['decimal']
            self.marker = pars['marker']
        except:
            pass
        
        if len(self.cols) != len(self.names):
            raise ValueError ('Columns length is different than names length!')
        
        self.ncols = len(self.cols)
        self.row_len = sum(self.cols)
    
    def add_content(self, content):
        '''
        Adds a row of content for the report.
        Content must be provided as a list of strings for each columnd field.
        '''
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
        
    def show(self, header=True, endrule=True):
        '''
        Prints the report to stdout.
        '''
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


def index_dups(energy,tol=1e-4):
    '''
    This function returns an index array
    with consecutive energy duplicates.
    Consecutive energy values are considered 
    duplicates if their absolute difference 
    is below the given tolerance.
    --------------
    Required input:
    energy [array]: energy array.
    tol [float]   : tolerance value.
    -------------
    Output:
    index [array]: index array with duplicates.
    '''
    from numpy import argwhere, diff

    dif = diff(energy)
    index = argwhere(dif < tol)

    return (index+1)


def read_dnd(fname, scan='mu', tol=1e-4):
    '''
    This function returns a Larch Group with 
    XAFS data from sector 5BM-D of the APS.
    --------------
    Required input:
    fname [string]: filename containing the data.
    scan [string] :  eiher 'fluo', 'mu', or 'mu_ref'.
    tol [float]   : tolerance value to remove duplicates.
    
    Extracted columns:
    column 0 : energy (eV).
    column 16: IF/IO (corrected for deadtime) <-- 'fluo'
    column 17: -log(IT/IO) (corrected for background) <-- 'mu'
    column 18: -log(IT2/IT) (corrected for background) <-- 'mu_ref'
    -------------
    Output:
    dat: Larch group containing the following arrays:
    energy, (fluo,mu), mu_ref
    '''
    from os import path
    import warnings
    from numpy import loadtxt, delete
    import larch
    from larch import Group
    from pyxas import index_dups
    
    # Testing that the file exits in the current directory 
    if not path.isfile(fname):
        raise IOError('file %s does not exist in the current path.' % fname)
    
    scandict = {'fluo':16, 'mu':17, 'mu_ref':18}
    # testing that the scan string exits in the current dictionary 
    if scan not in scandict:
        warnings.warn("scan type %s not recognized. Extracting transmission spectrum ('mu')." %scan)
        scan = 'mu'
    
    raw    = loadtxt(fname, usecols=(0,scandict[scan],scandict['mu_ref']))
    # deleting duplicate energy points
    index  = index_dups(raw[:,0],tol)
    raw    = delete(raw,index,0)
    if scan == 'mu_ref':
    #    print "Extracting only reference data ('mu_ref')."
        data = Group(**{'energy':raw[:,0], scan:raw[:,1], 'mu_ref':raw[:,2]})
    else:
        # Extracting the requested data
        data = Group(**{'energy':raw[:,0], scan:raw[:,1], 'mu_ref':raw[:,2]})
    
    return (data)


def calibrate_energy(data, e0, session):
    '''
    This function calibrates the threshold energy
    of the "reference channel" based on the arbitrary 
    value assigned for E0 (enot).
    --------------
    Required input:
    data: Larch group containing the spectra to calibrate.
    e0: arbitrary value for calibration of e0.
    session: valid Larch session.
    --------------
    Output:
    e_offset : Appended value to the data group with the magnitude
               of the calibration energy.
    '''
    # calculation of E0 based on Ifeffit standard
    import warnings
    import larch    
    from larch.xafs import find_e0
    
    if hasattr(data, 'e_offset'):
        warnings.warn('data group was already aligned or calibrated! Resetting energy to original value.')
        data.energy = data.energy-data.e_offset
        data.e_offset = 0
    
    data.e_offset = e0-find_e0(data.energy, data.mu_ref, _larch=session)
    
    # currently this script modifies the energy array!
    data.energy = data.energy+data.e_offset

    
def align_scans(objdat, refdat, session, e_offset=0, window=[-50,50]):
    '''
    This function aligns the first derivative of
    the reference channel of a data group against 
    the first derivative of the reference channel
    of a reference group.
    --------------
    Required input:
    objdat: objective data group.
    refdat: reference data group.
    session: valid Larch session.
    e_offset: initial energy offset value for alignment (optional).
    window: array with +- window values for alignment w/r to e0 (optional).
    --------------
    Output:
    e_offset : Appended value to the data group with the magnitude
               of the alignment energy.
    '''
    import warnings
    from numpy import where, gradient, sum
    from scipy.interpolate import UnivariateSpline
    from scipy.optimize import fmin
    import larch
    from larch.math import remove_dups
    from larch.xafs import find_e0

    #objdat.energy = remove_dups(objdat.energy)
    if hasattr(objdat, 'e_offset'):
        warnings.warn('Data group was already aligned or calibrated! Resetting energy to original value.')
        objdat.energy = objdat.energy-objdat.e_offset
        objdat.e_offset = 0
    
    # calculation of e0 to determine the optimization window
    e0 = find_e0(refdat.energy, refdat.mu_ref, _larch=session)
    min_lim = e0+window[0]
    max_lim = e0+window[1]
    
    # calculation of energy points for interpolation and comparison
    # this energy array is static
    index = where((refdat.energy >= min_lim) & (refdat.energy <= max_lim))
    ref_energy = refdat.energy[index]
    
    # both the reference mu and the objetive mu
    # exist in the same energy grid as the reference
    ref_spline = UnivariateSpline(refdat.energy, refdat.mu_ref, s=0)
    ref_dmu = gradient(ref_spline(ref_energy))/gradient(ref_energy)
    
    
    # the objective function is the difference of derivatives
    def objfunc(x):
        obj_spline = UnivariateSpline(objdat.energy + x, objdat.mu_ref, s=0)
        obj_dmu = gradient(obj_spline(ref_energy))/gradient(ref_energy)
        return sum((ref_dmu - obj_dmu)**2)
    
    objdat.e_offset = fmin(objfunc, e_offset, disp=False)[0]
    
    # currently this script modifies the energy array of objdat.
    objdat.energy = objdat.energy + objdat.e_offset


def merge_scans(group, scan='mu', order='3'):
    '''
    This function rebins the scans provided in the
    list group, and merges in the xmu space.
    --------------
    Required input:
    group: list of Larch groups to be merged in xmu.
    refdat: reference data group.
    scan [string]:  eiher 'fluo', 'mu', or 'mu_ref'.
    order [int]: spline order. Defaults to 3. 
    --------------
    Output:
    merge: Larch group containing the merged xmu scans.
    '''
    import warnings
    from numpy import size, resize, append, mean
    from scipy.interpolate import UnivariateSpline
    from larch.math import remove_dups
    from larch import Group
    
    scanlist = ['fluo','mu', 'mu_ref']
    # Testing that the scan string exists in the current dictionary 
    if scan not in scanlist:
        warnings.warn("scan type %s not recognized. Merging transmission data ('mu').")
        scan = 'mu'
    
    # Energy arrays are compared to create the interpolation array
    # that is completely contained in all the previous arrays.
    energy_eval=0
    for i in range(len(group)):
        # Searching the energy array with the largest initial value.
        # Alternative algorithm is to search the array with lowest final value.
        if group[i].energy[0] > energy_eval:
            energy_eval = group[i].energy[0]
            gindex = i
    
    energy = group[gindex].energy
    for gr in group:
        # Determining the last point in the selected energy array that
        # is contained in the other energy vectors
        while energy[-1] > gr.energy[-1]:
            energy = energy[:-1]
    
    energy = remove_dups(energy)
    mu = []      # container for the interpolated data
    if scan != 'mu_ref':
        mu_ref = []  # container for the interpolated reference channel
    
    for gr in group:
        # interpolation of the initial data
        # the getattr method is employed to recycle the variable scan
        gpenergy  = remove_dups(gr.energy)
        mu_spline = UnivariateSpline(gpenergy, getattr(gr,scan), s=0, k=order)
        
        # appending the interpolated data to the container
        mu = append(mu, mu_spline(energy), axis=0)
        
        if scan != 'mu_ref':
            # interpolating also the reference channel
            mu_ref_spline= UnivariateSpline(gpenergy, gr.mu_ref, s=0)
            mu_ref = append(mu_ref, mu_ref_spline(energy), axis=0)
            
    # resizing the container
    mu = resize(mu,(len(group), len(energy)))
    # calculating the average of the spectra
    mu_avg = mean(mu, axis=0)
    
    if scan != 'mu_ref':
        mu_ref = resize(mu_ref,(len(group), len(energy)))
        mu_ref_avg = mean(mu_ref, axis=0)    
        data = Group(**{'energy':energy, scan:mu_avg, 'mu_ref':mu_ref_avg})
    else:
        data = Group(**{'energy':energy, scan:mu_avg})
    return (data)

def get_scan_type(data):
    '''
    This function return the scan type for a given
    dataset.
    if only 'mu_ref' exists then it is returned.
    otherwise either 'mu' or 'fluo' are returned.
    --------------
    Required input:
    data: Larch group containing the data.
    --------------
    Output:
    scan [string]:  eiher 'fluo', 'mu', or 'mu_ref'.
    '''
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
    '''
    This function returns the phase corrected 
    magnitude of the forward XAFS fourier-transform 
    for the data and, if available, the FEFFIT model.
    --------------
    Required input:
    group: Larch group containing the FT XAFS data.
    path: Larch FEFF path to extract the phase shift
    kmult [int]: k-multiplier of the XAFS data.
    --------------
    Output:
    Ouput is appended to the parsed Larch group.
        group.data.chir_pha_mag
        [optional] group.model.chir_pha_mag 
    '''
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
