# araucaria

[![License](https://img.shields.io/badge/License-BSD%202--Clause-green.svg)](https://github.com/marcoalsina/araucaria/blob/master/LICENSE)
![GitHub branch checks state](https://img.shields.io/github/checks-status/marcoalsina/araucaria/master)


`araucaria` is a Python library to read, process and analyze X-ray absorption fine structure 
(XAFS) spectra. The library is designed to be modular, transparent, and light-weight, allowing 
the development of routines that are reproducible, exchangeable, and readily extensible.

The library is under active develeopment, but in its current state allows to perform both
basic and advanced tasks on XAFS spectra such as linear combination fitting (LCF). 
Additional functionality will be added on a regular basis.

## How to install
The following install options are curently available for the alpha version of `araucaria`:

### Install with Git

If you have [`Git`](https://git-scm.com/) in your machine, you can execute the following command in the console:

```console
name@machine:~$ pip install git+https://github.com/marcoalsina/araucaria.git
```

``pip`` should be able to download the required dependencies.
If you have [`Conda`](https://docs.conda.io/en/latest/) installed (Anaconda or Miniconda), be sure to activate your environment:

```console
name@machine:~$ conda activate <yourenvironment>
```

### Install with http

Alternatively, you can download the source code and install ``araucaria`` directly.
Open up a `terminal` and execute the following commands:

```console
name@machine:~$ wget https://github.com/marcoalsina/araucaria/archive/master.zip
name@machine:~$ unzip master.zip
name@machine:~$ cd araucaria-master
name@machine:~$ pip install .
```

## Documentation

The official documentation of `araucaria` is available at 
[https://marcoalsina.github.io/araucaria](https://marcoalsina.github.io/araucaria).
