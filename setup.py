import os
from setuptools import setup

# retrieving version
f   = open(os.path.join('pyxas', 'version'), 'r')
ver = f.readline()
f.close()

setup(
    name        = 'pyxas',
    version     = ver,
    description = 'Python routines to process XAS spectra',
    author      = 'Marco A. Alsina',
    author_email= 'marco.alsina@utalca.cl',
    url         = 'https://github.com/marcoalsina/pyxas',
    license     = 'BSD',
    packages    = ['pyxas', 'pyxas.io', 'pyxas.plot', 'pyxas.fit'],
    long_description = open('README.md').read(),
    include_package_data = True,
    install_requires = [
        'numpy>=1.16.4',
        'scipy>=1.3.1',
        'matplotlib>=3.1.0',
        'xraylarch>=0.9.46',
        'lmfit>=1.0.0',
        'h5py>=2.10.0',
        'sqlalchemy>=1.3.13',
        'pyshortcuts>=1.7',
    ],
)
