#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from .main import index_dups, get_scan_type, DataReport
from .main import xftf_pha
from .merge import calibrate_energy, align_scans, merge_scans, merge_spectra, merge_report

# reading file version
print(__file__)
f   = open(os.path.join('..','version'), 'r')
ver = f.readline()
f.close()

__version__ = ver
