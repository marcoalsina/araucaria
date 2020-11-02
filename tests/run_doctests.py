#!/usr/bin/python
# -*- coding: utf-8 -*-
#!/usr/bin/env python

import unittest
import doctest
import os

temp_files = ['database.h5',
              'new_file.xmu',
              'new_fit.lcf', 'lcf_report.log',
              ]

if __name__ == "__main__":
    files = []
    root_dir = os.path.join(os.pardir, 'araucaria')
    for root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == '__init__.py' or filename[-3:] != '.py':
                continue
            f = os.path.join(root, filename)
            f = f.replace(os.pardir+os.sep, '').replace(os.sep, '.')[:-3]
            files.append(f)

    suite = unittest.TestSuite()
    for module in files:
        suite.addTest(doctest.DocTestSuite(module))
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    # removing temp files    
    for fpath in temp_files:
        if os.path.exists(fpath):
            os.remove(fpath)