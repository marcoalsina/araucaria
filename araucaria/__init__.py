#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from .main import Group, Collection, Dataset, Report

# reading file version
f     = open(os.path.join(os.path.dirname(__file__), 'version'), 'r')
ver   = f.readline()
f.close()

__version__ = ver.replace('\n','')
