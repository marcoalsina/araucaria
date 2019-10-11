from setuptools import setup

setup(
    name='pyxas',
    version='0.1.1',
    description='Python routines to process XAS spectra',
    author='Marco A. Alsina',
    author_email='marco.alsina@utalca.cl',
    license='BSD',
    packages=['pyxas'],
    long_description=open('README.md').read(),
    install_requires=[
        'numpy>=1.16.4',
        'scipy>=1.3.1',
        'matplotlib>=3.1.0',
        'xraylarch>=0.9.46'],
)
