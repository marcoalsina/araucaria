import shutil, tempfile
from os import path
import importlib.resources as pkg_resources
import unittest
from numpy import allclose
from araucaria import testdata
from araucaria.io import read_xmu, write_xmu

class TestWriteFunctions(unittest.TestCase):
    def setUp(self):
        # create a temp directory
        self.temp_dir   = tempfile.mkdtemp()
        self.temp_fname = 'demo.xmu'
        self.temp_fpath = path.join(self.temp_dir, self.temp_fname)

    def tearDown(self):
        # remove the directory after the test
        shutil.rmtree(self.temp_dir)

    def test_write_xmu(self):
        # test ValueError exception
        invalid_group = ''
        self.assertRaises(TypeError, write_xmu, *(self.temp_fpath, invalid_group))
        
        # testing written file in temp folder
        with pkg_resources.path(testdata, 'xmu_testfile.xmu') as path:
            file_path  = path
        
        # testing multiple scans
        for scan in ['mu', 'fluo']:
            group_original = read_xmu(file_path, scan=scan)
            
            # write file in temp folder 
            write_xmu(self.temp_fpath, group_original, replace=True)
            
            if scan == 'fluo':
                self.assertRaises(IOError, write_xmu, *(self.temp_fpath, group_original))

            # reading written file
            group_read  = read_xmu(self.temp_fpath, scan=scan)

            # asserting scans
            self.assertTrue(allclose(getattr(group_original, scan), 
                                     getattr(group_read, scan)))
            self.assertTrue(allclose(getattr(group_original, 'mu_ref'), 
                                     getattr(group_read, 'mu_ref')))

if __name__ == '__main__':
    unittest.main