import os
from setuptools import setup, find_packages

# retrieving version
f   = open(os.path.join('araucaria', 'version'), 'r')
ver = f.readline()
f.close()

setup(
    name        = 'araucaria',
    version     = ver,
    description = 'Python library to manipulate XAFS spectra',
    author      = 'Marco A. Alsina',
    author_email= 'marco.alsina@utalca.cl',
    url         = 'https://github.com/marcoalsina/araucaria',
    license     = 'BSD',
    test_suite  = 'tests',
    packages    = find_packages(),
    long_description = open('README.md').read(),
    include_package_data = True,
    install_requires = [
        'numpy>=1.16.4',
        'scipy>=1.3.1',
        'matplotlib>=3.1.0',
        'lmfit>=1.0.0',
        'h5py>=3.0.0',
    ],
)
