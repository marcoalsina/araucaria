#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xrdb.chem` module offers the following 
routines to access chemical data:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~araucaria.xrdb.chem.ztosym`
     - Returns atomic symbol.
   * - :func:`~araucaria.xrdb.chem.symtoz`
     - Returns atomic number.
   * - :func:`~araucaria.xrdb.chem.at_weight`
     - Returns atomic weight.
   * - :func:`~araucaria.xrdb.chem.formula_parser`
     - Returns a parsed chemical formula.
   * - :func:`~araucaria.xrdb.chem.formula_weight`
     - Returns formula weight.

References
---------
- W.T. Elam, B.D. Ravel, J.R. Sieber (2002)
  "A new atomic database for X-ray spectroscopic calculations",
  Radiation Physics and Chemistry 63(2): pp 121-128, 
  `DOI:10.1016/S0969-806X(01)00227-4 <https://doi.org/10.1016/S0969-806X(01)00227-4>`_.
"""
from typing import List, Tuple
from os import pardir
from os.path import join, dirname
from re import search, findall
from sqlite3 import connect

# FILE PATH FOR ELAM DATABASE
# ---------------------------
dbpath = join(dirname(__file__), 'data', 'elam.db') 

def ztosym(z:int) -> str:
    """Returns atomic symbol.

    Parameters
    ----------
    z
        Atomic number.

    Returns
    -------
    :
        Atomic symbol.

    Raises
    ------
    NameError
        If ``z`` is not available.

    Example
    -------
    >>> from araucaria.xrdb import ztosym
    >>> z    = [26, 29, 30, 33]
    >>> syms = []
    >>> for val in z:
    ...     syms.append(ztosym(val))
    >>> print(syms)
    ['Fe', 'Cu', 'Zn', 'As']
    """
    # connecting to database
    conn   = connect(dbpath)
    cur    = conn.cursor()

    # executing search
    exe_str= "SELECT symbol FROM chem WHERE atnumber==?"
    cur.execute(exe_str, [z])

    # saving search result
    result = cur.fetchone()
    if result is None:
        raise NameError('z: %s not available.' % z)
    else:
        result = result[0]

    # closing database and returning value
    conn.close()
    return result

def symtoz(sym:str) -> int:
    """Returns atomic number.

    Parameters
    ----------
    sym
        Atomic symbol.

    Returns
    -------
    :
        Atomic number.

    Raises
    ------
    NameError
        If ``sym`` is not available.    

    Example
    -------
    >>> from araucaria.xrdb import symtoz
    >>> syms = ['Fe', 'Cu', 'Zn', 'As']
    >>> z    = []
    >>> for item in syms:
    ...     z.append(symtoz(item))
    >>> print(z)
    [26, 29, 30, 33]
    """
    # connecting to database
    conn   = connect(dbpath)
    cur    = conn.cursor()

    # executing search
    exe_str= "SELECT atnumber FROM chem WHERE symbol==?"
    cur.execute(exe_str, [sym])

    # saving search result
    result = cur.fetchone()
    if result is None:
        raise NameError('symbol: %s not available.' % sym)
    else:
        result = result[0]    

    # closing database and returning value
    conn.close()
    return result

def at_weight(sym: str) -> float:
    """Returns atomic weight.

    Parameters
    ----------
    sym
        Atomic symbol.

    Returns
    -------
    :
        Atomic weight (grams/mole).

    Raises
    ------
    NameError
        If ``sym`` is not available.  

    Example
    -------
    >>> from araucaria.xrdb import at_weight
    >>> syms = ['Fe', 'Cu', 'Zn', 'As']
    >>> atw  = []
    >>> for item in syms:
    ...     atw.append(at_weight(item))
    >>> print(atw)
    [55.847, 63.546, 65.38, 74.9216]
    """
    # connecting to database
    conn   = connect(dbpath)
    cur    = conn.cursor()
    
    # executing search
    exe_str= "SELECT atweight FROM chem WHERE symbol ==?"
    cur.execute(exe_str, [sym])

    # saving search result
    result = cur.fetchone()
    if result is None:
        raise NameError('symbol: %s not available.' % sym)
    else:
        result = result[0]    

    # closing database and returning value
    conn.close()
    return result

def _parser(formula: str, mult: float=1) -> list:
    """Utility parser for chemical formulas.

    Parameters
    ----------
    formula
        Chemical formula to parse.
    mult
        Multiplier for chemical formula.

    Returns
    -------
    :
        List with parsed formula

    Raises
    ------
    NameError
        If parentheses in ``formula`` are not balanced.

    Notes
    -----
    The returned list requires formatting.
    """
    isp = -1                     # index start parenthesis
    iep = -1                     # index end parenthesis
    icp =  0                     # current index parenthesis
    lstr = len(formula)

    # finding the first parenthesis
    for i in range(lstr):
        if formula[i] == '(':
            icp += 1   # update counter
            isp =  i
            break

    # finding the last parenthesis (if first parenthesis exists)
    if icp > 0:
        for j in range(i + 1, lstr):
            if formula[j] == '(':
                icp += 1   # update counter
            elif formula[j] == ')':
                icp -= 1   # update counter
                if icp == 0:
                    iep = j
                    break
    if icp != 0:
        raise NameError('parenthesis are not balanced.')

    # no parenthesis found, provide exit
    if isp == -1:
        cregex = '[A-Z][a-z]?\d*[.]?\d*'
        elreg  = '[A-Z][a-z]?'
        nureg  = '\d*\.?\d+|\d+'
        cseq   = findall(cregex, formula)
        vals   = []
        for item in cseq:
            reg = search(elreg, item).group()
            try:
                num = search(nureg, item).group()
            except:
                num = 1
            vals.append((reg, float(num)*mult))
        return vals

    # parenthesis found, recursive call
    else:
        f_regex = '^\d*[.]?\d+' # float regex
        seq     = []
        if isp != 0:
            # string before parenthesis
            seq.append( (formula[:isp], mult) )

        # major parenthesis
        num = search(f_regex, formula[iep + 1:])
        if not num:
            num = 1
            end = 0
        else:
            end = num.end()
            num = num.group()
        seq.append( (formula[isp + 1: iep], float(num) * mult) )

        # string after parenthesis
        if (iep + end + 1) < lstr:
            seq.append( (formula[iep + end + 1:], mult) )

        return [_parser(val[0], val[1]) for val in seq]


def _format_parser(x) -> list:
    """Utility parser formatter.

    Parameters
    ----------
    x
        Unformated parsed list.

    Returns
    -------
    :
        Formatted parsed list.
    """
    if isinstance(x, tuple):
        return [x]
    else:
        return [a for item in x for a in _format_parser(item)]

def formula_parser(formula: str, mult: float=1) -> dict:
    """Returns a dictionary with parsed formula.

    Parameters
    ---------
    formula
        Chemical formula to parse.
    mult
        Multiplier for chemical formula.

    Returns
    -------
    :
        Dictionary with parsed formula.

    Raises
    ------
    NameError
        If parentheses in ``formula`` are not balanced.

    Example
    -------
    >>> from araucaria.xrdb import formula_parser
    >>> formulas = ['As2O3', 'Fe(OH)3', 'CuCO3Cu(OH)2']
    >>> for f in formulas:
    ...     print(formula_parser(f))
    {'As': 2.0, 'O': 3.0}
    {'Fe': 1.0, 'H': 3.0, 'O': 3.0}
    {'C': 1.0, 'Cu': 2.0, 'H': 2.0, 'O': 5.0}
    """
    out = _parser(formula, mult=mult)
    out = _format_parser(out)
    
    # unique elements
    unique = set([val[0] for val in out])

    # dict container
    fdict = {}
    for item in sorted(unique):
        val = 0
        for pars in out:
            if pars[0] == item:
                val += pars[1]
        fdict[item] = val
    return fdict

def formula_weight(formula: str, mult: float=1) -> float:
    """Returns formula weight.

    Parameters
    ---------
    formula
        Chemical formula to parse.
    mult
        Multiplier for chemical formula.

    Returns
    -------
    :
        Formula weight (gr/mole).

    See also
    --------
    :func:`formula_parser`
        Returns a parsed chemical formula.
    :func:`at_weight`
        Returns atomic weight.

    Example
    -------
    >>> from araucaria.xrdb import formula_weight
    >>> formulas = ['As2O3', 'Fe(OH)3', 'CuCO3Cu(OH)2']
    >>> for f in formulas:
    ...     fweight = formula_weight(f)
    ...     print('{0:12} : {1:1.3f} gr/mole'.format(f, fweight))
    As2O3        : 197.841 gr/mole
    Fe(OH)3      : 106.869 gr/mole
    CuCO3Cu(OH)2 : 221.116 gr/mole
    """
    fdict = formula_parser(formula, mult=mult)

    # formula weight
    fweight = 0.0    
    for key in fdict:
        fweight += at_weight(key) * fdict[key]
    return fweight

if __name__ == '__main__':
    import doctest
    doctest.testmod()