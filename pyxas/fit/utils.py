#!/usr/bin/env python
'''
filename: utils.py

Colletion of routines to work with LCF output and log files.

Implemented functions:
    sum_standards
    residuals
    get_lcf_data
    get_chi2
'''

def sum_standards(pars, data):
    '''
    This function returns the linear sum of standards based on 
    the amplitude values stored in a dictionary with lcf parameters.
    '''
    from numpy import sum as npsum
    return (npsum([pars['amp'+str(i)]* getattr(data, 'ref'+str(i)) 
                   for i in range(1,len(pars)+1)], axis=0))


def residuals(pars,data):
    '''
    This function returns the residuals of the substraction
    of a spectrum from its linear combination fit with known
    standards
    '''
    return (data.spectrum - sum_standards(pars, data))/data.eps


def lcf_report(out):
    '''
    This function returns an updated LCF report.
    '''
    import os
    from lmfit import fit_report
    
    header = '[[Parameters]]\n'
    for key in out.pars_kws:
        val    = ' '.join(key.split('_'))
        header = header + '    {0:19}= {1}\n'\
        .format(val, out.pars_kws[key])
    
    header = header+'[[Data]]\n'
    for key in out.data_kws:
        val    = ' '.join(key.split('_'))
        if 'path' in key:
            keyval = os.path.abspath(out.data_kws[key])
        else:
            keyval = out.data_kws[key]
        header = header + '    {0:19}= {1}\n'\
        .format(val, keyval)

    return (header+fit_report(out))


def get_lcf_data(files, reference, error=True):
    '''
    This function reads a list of lcf log files and returns 
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
