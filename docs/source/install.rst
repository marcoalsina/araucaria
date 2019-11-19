How to install
==============

Pyxas is still in development for an official release.
If you still want to test the development version, the following install options are available:

Install with Git
****************

If you have Git in your machine you can execute the following command in the console:

.. code-block:: bash

   name@machine:~$ pip install git+https://github.com/marcoalsina/pyxas.git

Pip should be able to download the required dependencies.
If you have conda installed in your machine (Anaconda or Miniconda), be sure to activate your environment before running pip.

.. code-block:: bash

   name@machine:~$ conda activate <yourenvironment>

Install with http
*****************

If you don't have git installed you can download the source and install directly. Open up a terminal and execute the following:

.. code-block:: bash

    name@machine:~$ wget https://github.com/marcoalsina/pyxas/archive/master.zip
    name@machine:~$ unzip master.zip
    name@machine:~$ cd pyxas-master
    name@machine:~$ pip install .
