#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.io` module offers several functions to read and write files in plain text format.
The module also offers functions to read, write and manipulate datasets in the 
Hierarchical Data Format ``HDF5``, for efficient data storage of large datasets.

The following submodules are currently implemented:

- The :mod:`~araucaria.io.io_read` module contains functions to read files in plain text format.
- The :mod:`~araucaria.io.io_write` module contains functions to write files in plain text format.
- The :mod:`~araucaria.io.io_hdf5` module contains functions to read, write and manipulate files in ``HDF5``.
"""
from .io_hdf5 import read_hdf5, read_collection_hdf5, convert_bytes_hdf5
from .io_hdf5 import write_hdf5, write_collection_hdf5, write_recursive_hdf5
from .io_hdf5 import rename_dataset_hdf5, delete_dataset_hdf5, summary_hdf5
from .io_read import read_p65, read_dnd, read_xmu, read_file, read_rawfile
from .io_read import read_lcf_coefs, read_lcf_chisqr
from .io_write import write_xmu, write_lcf, write_lcf_report, write_file, set_header
