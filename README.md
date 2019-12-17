# pyxas

[![License](https://img.shields.io/badge/License-BSD%202--Clause-green.svg)](https://github.com/marcoalsina/pyxas/blob/master/LICENSE)


A collection of python routines to process and analyze X-ray absorption spectra.

## How to install
Pyxas is still in development for an official release.
If you want to test the development version, the following install options are available:

### Install with Git
If you have Git in your machine you can execute the following command in the console:

```console
name@machine:~$ pip install git+https://github.com/marcoalsina/pyxas.git
```
Pip should be able to download the required dependencies.
If you have conda installed in your machine (Anaconda or Miniconda), be sure to activate your envrionment before running pip.
```console
name@machine:~$ conda activate <yourenvironment>
```

### Install with http
If you don't have git installed you can download the source and install directly. Open up a terminal and execute the following:

```console
name@machine:~$ wget https://github.com/marcoalsina/pyxas/archive/master.zip
name@machine:~$ unzip master.zip
name@machine:~$ cd pyxas-master
name@machine:~$ pip install .
```
