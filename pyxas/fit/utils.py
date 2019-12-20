#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colletion of routines to work with LCF output and log files.

Implemented methods (class lmfit out):
* fit_report
* save_fit_report
* save_ecf_data
* save_lcf_data
    
Implemented methods (class feffit out):
* lsf_report
* save_lsf_report
* save_lsf_data


Implemented functions:
* get_lcf_data
* get_chi2
"""

def fit_report(self):
    """Fit report.
    
    This function recieves an lmfit object and
    returns a fit report.
    """
    import os
    from lmfit import fit_report
    
    # assigning title
    if 'ncomps' in self.pars_kws:
        title = '='*11+' EXTRACT COMPONENTS FIT RESULTS '+'='*11+'\n'
    else:
        title = '='*11+' LINEAR COMBINATION FIT RESULTS '+'='*11+'\n'

    header = title + '[[Parameters]]\n'
    for key in self.pars_kws:
        val    = ' '.join(key.split('_'))
        header = header + '    {0:19}= {1}\n'\
        .format(val, self.pars_kws[key])

    header = header+'[[Data]]\n'
    for key in self.data_kws:
        val    = ' '.join(key.split('_'))
        if 'path' in key:
            keyval = os.path.abspath(self.data_kws[key])
        else:
            keyval = self.data_kws[key]
        header = header + '    {0:19}= {1}\n'\
        .format(val, keyval)

    return (header+fit_report(self))

def save_fit_report(self, filepath):
    """Saves fit report to file.
    
    This function saves an fit report 
    in a file specificed by filepath.
    """
    from .utils import fit_report

    fout = open(filepath, 'w')
    fout.write(fit_report(self))
    fout.close()
    return    


def save_fit_data(self, filepath):
    """Saves LCF data to file.
    
    This function saves LCF data in a file
    specificed by filepath.
    """
    from numpy import column_stack, savetxt
    from .utils import fit_report

    # fit report
    rep_header = fit_report(self)

    # saving ECF spectra
    if 'ncomps' in self.pars_kws:
        if self.pars_kws['fit_type'] == 'exafs':
            data_header = 'k [A-1]\t' + 'k^%s chi_1(k)\t'%self.pars_kws['k_mult'] + 'k^%s chi_2(k)'%self.pars_kws['k_mult']
            data = column_stack((self.data_group.k, self.data_group.comps['x1_mean'], self.data_group.comps['x2_mean']))
        else:
            data = column_stack((self.data_group.energy, self.data_group.comps['x1_mean'], self.data_group.comps['x2_mean']))
            if self.pars_kws['fit_type'] == 'xanes':
                data_header = 'Energy [eV]\t' + 'Norm. abs. x1 [adim]\t' + 'Norm. abs. x2 [adim]'
            else:
                data_header = 'Energy [eV]\t' + 'Deriv. norm. abs. x1 [adim]\t' +  'Deriv. norm. abs. x2 [adim]'

    # saving LCF spectra
    else:
        if self.pars_kws['fit_type'] == 'exafs':
            data_header = 'k [A-1]\t' + 'k^%s chi(k)'%self.pars_kws['k_mult']+ '\tFit\tResidual'
            data = column_stack((self.data_group.k, self.data_group.spectrum, self.data_group.fit, self.residual))
        else:
            data = column_stack((self.data_group.energy, self.data_group.spectrum, self.data_group.fit, self.residual))
            if self.pars_kws['fit_type'] == 'xanes':
                data_header = 'Energy [eV]\t' + 'Norm. abs. [adim]'+ '\tFit\tResidual'
            else:
                data_header = 'Energy [eV]\t' + 'Deriv. norm. abs. [adim]'+ '\tFit\tResidual'

    savetxt(filepath, data, fmt='%.6f',  header=rep_header + '\n' + data_header)
    return


def lsf_report(self):
    """FEEFIT least squares fit (LSF) report.
    
    This function recieves a feffit object and
    returns an updated least squares fit report.
    """
    import os
    from larch.xafs.feffit import feffit_report

    insert = '[[Parameters]]\n'
    for key in self.pars_kws:
        val    = ' '.join(key.split('_'))
        insert = insert + '   {0:19}= {1}\n'\
        .format(val, self.pars_kws[key])

    report = feffit_report(self)

    # searching for data section
    search_string = '[[Data]]'
    index         = report.index(search_string)

    pre_report  = report[:index]
    post_report = report[index+len(search_string)+1:]

    insert = insert+'\n'+search_string+'\n'
    for key in self.data_kws:
        if 'spectrum' in key:
            val    = ' '.join(key.split('_'))
            if 'path':
                keyval = os.path.abspath(self.data_kws[key])
            elif 'name' in key:
                keyval = self.data_kws[key]

            insert = insert + '   {0:19}= {1}\n'.format(val, keyval)

    report = pre_report + insert + post_report
    return (report)


def save_lsf_report(self, filepath):
    '''
    This function saves a feffit least squares
    fit report in a file specificed by filepath.
    '''
    from .utils import lsf_report

    fout = open(filepath, 'w')
    fout.write(lsf_report(self))
    fout.close()
    return


def save_lsf_data(self, filepath, save='exafs'):
    '''
    This function saves the results from a feffit
    least squares in a file specificed by filepath.
    Either 'exafs' or 'xftf' spectra can be saved.
    Default save is 'exafs'.
    '''
    from numpy import column_stack, savetxt
    from .utils import lsf_report

    # verifying save type
    save_types = ['exafs', 'xftf']
    if save not in save_types:
        raise ValueError("save option '%s' not recognized."%save)

    # saving feffit lsf spectra
    rep_header = lsf_report(self)

    # extracting data and model
    dset  = self.datasets[0].data
    model = self.datasets[0].model

    if save == 'exafs':
        #datsav     = np.column_stack((dset.data.k, dset.data.chi, dset.model.chi))
        data_header = 'k [A-1]\t' + 'k^%i chi(k)\t'%self.pars_kws['k_mult']+ 'Fit'
        data = column_stack((dset.k, dset.chi, model.chi))
    else:
        data_header = 'R [A]\t' + '|chi(R)| [A^-%i]\t'%(self.pars_kws['k_mult']+1) + 'Fit'
        data = column_stack((dset.r, dset.chir_mag, model.chir_mag))

    savetxt(filepath, data, fmt='%.6f',  header=rep_header + '\n' + data_header)
    return

def get_lcf_data_legacy(files, reference, error=True):
    """
    This function reads a list of LCF log files and returns 
    a numpy array with the values associated with the specified reference
    The calculated standard deviation can be retrieved optionally.
    IMPORTANT: This is a legacy function from a previous LCF log file format.
    """
    import os
    from numpy import append, float

    reference = reference
    vallist   = []    # container for values
    errlist   = []    # container for errors

    for file in files:
        getref = True
        getval = False
        f = open(file, 'r')
        while getref:
            line = f.readline()
            if reference in line:
                # Reference found in line, we extract the standard index
                index = line.split(',')[0][-2:-1]
                stdval = "amp"+index
                getref = False
                getval = True

            elif "[[Correlations" in line:
                # This line indicates that we already passed the reference values
                # There is nothing else to search so return zeroes instead
                vallist = append(vallist,0.00)
                errlist = append(errlist,0.00)
                getref = False
                break

        while getval:
            line = f.readline()
            if stdval in line:
                val = float(line.split()[1]) * 100
                err = float(line.split()[3]) * 100
                vallist = append(vallist,val)
                errlist = append(errlist,err)
                getval = False

    if error:
        return (vallist, errlist)
    else:
        return (vallist)


def get_lcf_data(files, reference, error=True):
    '''
    This function reads a list of LCF log files and returns 
    a numpy array with the values associated with the specified reference
    The calculated standard deviation can be retrieved optionally.
    '''
    import os
    from numpy import append, float

    reference = reference
    vallist   = []    # container for values
    errlist   = []    # container for errors

    for file in files:
        getref = True
        getval = False
        f = open(file, 'r')
        while getref:
            line = f.readline()
            if reference in line:
                # Reference found in line, we extract the standard index
                index = line.split()[0][-1]
                stdval = "amp"+index
                getref = False
                getval = True

            elif "[[Fit Statistics]]" in line:
                # This line indicates that we already passed the [[Data]] section
                # There is nothing else to search so return zeroes instead
                vallist = append(vallist,0.00)
                errlist = append(errlist,0.00)
                getref = False
                break

        while getval:
            line = f.readline()
            if stdval in line:
                val = float(line.split()[1]) * 100
                err = float(line.split()[3]) * 100
                vallist = append(vallist,val)
                errlist = append(errlist,err)
                getval = False

    if error:
        return (vallist, errlist)
    else:
        return (vallist)


def get_chi2(files, reduced=False):
    """
    This function reads a list of lcf log files and returns 
    a numpy array with the chi-squared values associated with the fit.
    The reduced chi-square can be retrieved optionally.
    """
    import os
    from numpy import append, float

    if reduced:
        reference = " reduced chi-square"
    else:
        reference = "    chi-square"

    vallist = []    # container for values

    for file in files:
        getval = True
        f = open(file, 'r')
        while getval:
            line = f.readline()
            if reference in line:
                val = float(line.split("=")[1])
                vallist = append(vallist,val)
                getval = False

    return (vallist)
