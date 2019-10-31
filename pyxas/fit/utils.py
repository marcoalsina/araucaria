#!/usr/bin/env python
'''
filename: utils.py

Colletion of routines to work with LCF output and log files.

Implemented methods (class lmfit out):
    lcf_report
    save_lcf_report
    save_lcf_data
    
Implemented methods (class feffit out):
    lsf_report
    save_lsf_report
    save_lsf_data


Implemented functions:
    sum_references
    residuals
    get_lcf_data
    get_chi2
'''

def lcf_report(self):
    '''
    This function recieves an lmfit object and
    returns an LCF report.
    '''
    import os
    from lmfit import fit_report

    header = '[[Parameters]]\n'
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


def save_lcf_report(self, filepath):
    '''
    This function saves an LCF report 
    in a file specificed by filepath.
    '''
    from .utils import lcf_report

    fout = open(filepath, 'w')
    fout.write(lcf_report(self))
    fout.close()
    return


def save_lcf_data(self, filepath):
    '''
    This function saves LCF data in a file
    specificed by filepath.
    '''
    from numpy import column_stack, savetxt
    from .utils import lcf_report

    # saving LCF spectra
    rep_header = lcf_report(self)

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
    '''
    This function recieves a feffit object and
    returns an updated least squares fit report.
    '''
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
    #from .utils import lsf_report

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
    #from .utils import lsf_report

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



def sum_references(pars, data):
    '''
    This function returns the linear sum of references based on 
    the amplitude values stored in a dictionary with LCF parameters.
    '''
    from numpy import sum as npsum
    return (npsum([pars['amp'+str(i)]* getattr(data, 'ref'+str(i)) 
                   for i in range(1,len(pars)+1)], axis=0))


def residuals(pars,data):
    '''
    This function returns the residuals of the substraction
    of a spectrum from its LCF with known references
    standards.
    '''
    return (data.spectrum - sum_references(pars, data))/data.eps


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


def get_chi2(files, reduced=False):
    '''
    This function reads a list of lcf log files and returns 
    a numpy array with the chi-squared values associated with the fit.
    The reduced chi-square can be retrieved optionally.
    '''
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
