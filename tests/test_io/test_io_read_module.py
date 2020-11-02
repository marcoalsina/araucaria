import unittest
import os
import importlib.resources as pkg_resources
from numpy import ndarray, shape, loadtxt, log, allclose, delete, diff
from araucaria import Group
from araucaria import testdata
from araucaria.io import read_p65, read_dnd, read_xmu, read_file, read_rawfile
from araucaria.utils import index_dups

class TestReadFunctions(unittest.TestCase):

    def setUp(self):
        self.path_dict = {
            'fname_p65' : 'p65_testfile.xdi',
            'fname_dnd' : 'dnd_testfile.dat',
            'fname_xmu' : 'xmu_testfile.xmu'
            }

    def test_read_p65(self):
        """Test function for read_p65.
        """
        with pkg_resources.path(testdata, self.path_dict['fname_p65']) as path:
            fpath = path

        # extracting scans from original file
        raw   = loadtxt(fpath, usecols=(0,10,11,12,13))
        index = index_dups(raw[:,0], 1e-4)
        raw   = delete(raw,index,0)
        
        muref  = -log(raw[:,3]/raw[:,2])
        mu     = -log(raw[:,2]/raw[:,1])
        fluo   = raw[:,4]/raw[:,1] 
        
        for scan in ['mu', 'fluo', None]:
            # testing with request for mu_ref
            group = read_p65(fpath, scan=scan)
            
            self.assertIsInstance(group, Group)              # group class is returned
            self.assertIsInstance(group.energy, ndarray)     # ndarray type is returned
            self.assertIsInstance(group.mu_ref, ndarray)     # ndarray type is returned
            self.assertTrue(allclose(group.mu_ref, muref))   # mu_ref is properly retrieved
            if scan == 'mu':
                self.assertIsInstance(getattr(group,scan), ndarray)
                self.assertFalse(hasattr(group, 'fluo'))     # instance fluo is not returned
                self.assertTrue(allclose(group.mu, mu))      # mu_ref is properly retrieved
            elif scan == 'fluo':
                self.assertIsInstance(getattr(group,scan), ndarray)
                self.assertFalse(hasattr(group, 'mu'))       # instance mu is not returned
                self.assertTrue(allclose(group.fluo, fluo))  # fluo is properly retrieved
        
        for scan in ['mu', 'fluo']:
            # testing with no request for mu_ref
            group = read_p65(fpath, scan=scan, ref=False)
            
            self.assertIsInstance(group, Group)              # group class is returned
            self.assertIsInstance(group.energy, ndarray)     # ndarray type is returned
            self.assertFalse(hasattr(group, 'mu_ref'))       # instance mu_ref is not returned
            if scan == 'mu':
                self.assertFalse(hasattr(group, 'fluo'))     # instance fluo is not returned
                self.assertTrue(allclose(group.mu, mu))      # mu_ref is properly retrieved
            elif scan == 'fluo':
                self.assertFalse(hasattr(group, 'mu'))       # instance mu is not returned
                self.assertTrue(allclose(group.fluo, fluo))  # fluo is properly retrieved

    def test_read_dnd(self):
        """Test function for read_dnd.
        """
        with pkg_resources.path(testdata, self.path_dict['fname_dnd']) as path:
            fpath = path
        
        # extracting scans from original file
        raw   = loadtxt(fpath, usecols=(0,16,17,18))
        index = index_dups(raw[:,0], 1e-4)
        raw   = delete(raw,index,0)
        
        muref  = raw[:,3]
        mu     = raw[:,2]
        fluo   = raw[:,1]
        
        for scan in ['mu', 'fluo', None]:
            # testing with request for mu_ref
            group = read_dnd(fpath, scan=scan)
            
            self.assertIsInstance(group, Group)              # group class is returned
            self.assertIsInstance(group.energy, ndarray)     # ndarray type is returned
            self.assertIsInstance(group.mu_ref, ndarray)     # ndarray type is returned
            self.assertTrue(allclose(group.mu_ref, muref))   # mu_ref is properly retrieved
            if scan == 'mu':
                self.assertIsInstance(getattr(group,scan), ndarray)
                self.assertFalse(hasattr(group, 'fluo'))     # instance fluo is not returned
                self.assertTrue(allclose(group.mu, mu))      # mu_ref is properly retrieved
            elif scan == 'fluo':
                self.assertIsInstance(getattr(group,scan), ndarray)
                self.assertFalse(hasattr(group, 'mu'))       # instance mu is not returned
                self.assertTrue(allclose(group.fluo, fluo))  # fluo is properly retrieved
        
        for scan in ['mu', 'fluo']:
            # testing with no request for mu_ref
            group = read_dnd(fpath, scan=scan, ref=False)
            
            self.assertIsInstance(group, Group)              # group class is returned
            self.assertIsInstance(group.energy, ndarray)     # ndarray type is returned
            self.assertFalse(hasattr(group, 'mu_ref'))       # instance mu_ref is not returned
            if scan == 'mu':
                self.assertFalse(hasattr(group, 'fluo'))     # instance fluo is not returned
                self.assertTrue(allclose(group.mu, mu))      # mu_ref is properly retrieved
            elif scan == 'fluo':
                self.assertFalse(hasattr(group, 'mu'))       # instance mu is not returned
                self.assertTrue(allclose(group.fluo, fluo))  # fluo is properly retrieved

    def test_read_xmu(self):
        """Test function for read_xmu.
        """
        with pkg_resources.path(testdata, self.path_dict['fname_xmu']) as path:
            fpath = path
        
        # extracting scans from original file
        raw   = loadtxt(fpath, usecols=(0,1,2))
        index = index_dups(raw[:,0], 1e-4)
        raw   = delete(raw,index,0)
        
        muref  = raw[:,2]
        mu     = raw[:,1]
        fluo   = raw[:,1]
        
        for scan in ['mu', 'fluo', None]:
            # testing with request for mu_ref
            group = read_xmu(fpath, scan=scan)
            
            self.assertIsInstance(group, Group)              # group class is returned
            self.assertIsInstance(group.energy, ndarray)     # ndarray type is returned
            self.assertIsInstance(group.mu_ref, ndarray)     # ndarray type is returned
            self.assertTrue(allclose(group.mu_ref, muref))   # mu_ref is properly retrieved
            if scan == 'mu':
                self.assertIsInstance(getattr(group,scan), ndarray)
                self.assertFalse(hasattr(group, 'fluo'))     # instance fluo is not returned
                self.assertTrue(allclose(group.mu, mu))      # mu_ref is properly retrieved
            elif scan == 'fluo':
                self.assertIsInstance(getattr(group,scan), ndarray)
                self.assertFalse(hasattr(group, 'mu'))       # instance mu is not returned
                self.assertTrue(allclose(group.fluo, fluo))  # fluo is properly retrieved
        
        for scan in ['mu', 'fluo']:
            # testing with no request for mu_ref
            group = read_xmu(fpath, scan=scan, ref=False)
            
            self.assertIsInstance(group, Group)              # group class is returned
            self.assertIsInstance(group.energy, ndarray)     # ndarray type is returned
            self.assertFalse(hasattr(group, 'mu_ref'))       # instance mu_ref is not returned
            if scan == 'mu':
                self.assertFalse(hasattr(group, 'fluo'))     # instance fluo is not returned
                self.assertTrue(allclose(group.mu, mu))      # mu_ref is properly retrieved
            elif scan == 'fluo':
                self.assertFalse(hasattr(group, 'mu'))       # instance mu is not returned
                self.assertTrue(allclose(group.fluo, fluo))  # fluo is properly retrieved
    
    def test_read_file(self):
        """Test function for read_file
        """
        with pkg_resources.path(testdata, self.path_dict['fname_xmu']) as path:
            fpath = path

        usecols = (0,1,2)
        scan    = 'mu'
        ref     = True
        tol     = 1e-4
        incorrect_path = 'some_file.xmu'

        # testing IOError
        self.assertRaises(IOError, read_file, *(incorrect_path, usecols, scan, ref, tol))

        # testing ValueError
        self.assertRaises(ValueError, read_file, *(fpath, usecols, None, False, tol))

        # testing TypeError
        self.assertRaises(TypeError, read_file, *(fpath, usecols, scan, scan, tol))

    def test_read_rawfile(self):
        """Test function for read_rawfile
        """
        with pkg_resources.path(testdata, self.path_dict['fname_p65']) as path:
            fpath = path
        
        usecols = (0,10,11,12,13)
        scan    = 'mu'
        ref     = True
        tol     = 1e-4
        incorrect_path = 'some_file.xmu'

        # testing IOError
        self.assertRaises(IOError, read_rawfile, *(incorrect_path, usecols, scan, ref, tol))
        
        # testing ValueError
        self.assertRaises(ValueError, read_rawfile, *(fpath, usecols, None, False, tol))
        
        # testing TypeError
        self.assertRaises(TypeError, read_rawfile, *(fpath, usecols, scan, scan, tol))        
        
        
if __name__ == '__main__':
    unittest.main()