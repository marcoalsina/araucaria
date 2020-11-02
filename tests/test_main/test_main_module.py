import unittest
import random
from numpy import arange, allclose
from araucaria import Group, Collection, Report


class TestMainClasses(unittest.TestCase):

    def setUp(self):
        self.group_methods   = ['get_scan_type', 'has_ref']
        self.collect_methods = []
        self.report_methods  = ['set_columns', 'add_content', 'add_midrule', 'show', 
                                'sdigits', 'marker']

    def test_group(self):
        """Test function for Group class.
        """
        group = Group()
        self.assertIsInstance(group, Group)   # proper instance of Group
        for method in self.group_methods:
            self.assertIn(method, dir(group)) # asserting group methods
        
        # testing group methods
        # testing exception for get_scan_type
        self.assertRaises(ValueError, group.get_scan_type)
        
        for scan in ('mu', 'fluo'):
            pars  = {'energy': 0, scan: 1, 'mu_ref':2}
            group = Group(**pars)
            self.assertTrue(group.get_scan_type() == scan) # testing with mu_ref
            self.assertTrue(group.has_ref())
            
            pars  = {'energy': 0, scan: 1}
            group = Group(**pars)
            self.assertTrue(group.get_scan_type() == scan) # testing with no mu_ref
            self.assertFalse(group.has_ref())
            
            pars  = {'energy': 0, 'mu_ref': 2}
            group = Group(**pars)
            self.assertTrue(group.get_scan_type() == 'mu_ref') # testing only mu_ref
            self.assertTrue(group.has_ref())
            
    def test_collection(self):
        """Test function for Collection class.
        """
        collection = Collection()
        self.assertIsInstance(collection, Collection)   # proper instance of Group

    def test_report(self):
        """Test function for Report class.
        """
        report = Report()
        self.assertIsInstance(report, Report)   # proper instance of Report
        for method in self.report_methods:
            self.assertIn(method, dir(report))  # asserting report methods

        # testing method set_columns
        # setting test values
        random.seed = 12345
        ncols   = 3
        width   = 10
        cols    = [width for i in range(ncols)]
        names   = ['val%s' % (i+1) for i in range(ncols)]
        tcontent= ['name%s' % (i+1) for i in range(ncols)]
        ncontent= [10*random.random() for i in range(ncols)]
        dec     = 3
        mark    = '*'
        
        # printed values
        nformat = ''
        tformat = ''
        for i, col in enumerate(cols):
            nformat += '{%i:<%i.%if}' % (i,cols[i],dec)
            tformat += '{%i:<%i}' % (i, cols[i])

        printed_title = tformat.format(*names)
        printed_text  = tformat.format(*tcontent)
        printed_vals  = nformat.format(*ncontent)

        # testing IndexError exception for method set_columns
        pars = {'cols': cols, 'names': names[:2]}
        self.assertRaises(IndexError, report.set_columns, **pars)

        # testing AttributeError exception for method add_content
        self.assertRaises(AttributeError, report.add_content, tcontent)

        # testing method set_columns
        pars   = {'cols': cols, 'names': names, 'decimal': dec, 'marker': mark}
        report.set_columns(**pars)

        # testing IdexError exception for method add_content
        self.assertRaises(IndexError, report.add_content, tcontent[:2])

        # testing method set_columns
        self.assertTrue(report.cols == cols)
        self.assertTrue(report.names == names)
        self.assertTrue(report.decimal == dec)
        self.assertTrue(report.marker == mark)

        # testing method add_content
        report.add_content(tcontent)
        report.add_content(ncontent)
        self.assertTrue(report.content == printed_text + '\n' + printed_vals)
        
        # testing method add_midrule
        for item in ['*', '=', '.']:
            midrule = width*ncols*item
            report  = Report()
            report.set_columns(**pars)
            report.add_midrule(item)
            self.assertTrue(report.content == midrule)

        # testing method show
        headfoot= width*ncols*mark
        report  = Report()
        report.set_columns(**pars)
        report.add_content(tcontent)
        report.add_content(ncontent)

        # testing with no header or footer
        print_exp = printed_text + '\n' + printed_vals
        print_obs = report.show(header=False, endrule=False, print_report=False)
        self.assertTrue(print_exp == print_obs)

        # testing with header
        print_exp = headfoot + '\n' + printed_title + '\n' + headfoot
        print_exp +=  '\n' + printed_text + '\n' + printed_vals
        print_obs = report.show(endrule=False, print_report=False)
        self.assertTrue(print_exp == print_obs)

        # testing with footer
        print_exp = printed_text + '\n' + printed_vals + '\n' + headfoot
        print_obs = report.show(header=False, print_report=False)
        self.assertTrue(print_exp == print_obs)

        # testing with header and footer
        print_exp = headfoot + '\n' + printed_title + '\n' + headfoot
        print_exp +=  '\n' + printed_text + '\n' + printed_vals + '\n' + headfoot
        print_obs = report.show(print_report=False)
        self.assertTrue(print_exp == print_obs)

if __name__ == '__main__':
    unittest.main()