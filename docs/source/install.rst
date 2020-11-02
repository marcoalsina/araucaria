How to install
==============

The following install options are curently available for the development version of ``araucaria``:

Install with Git
----------------

If you have `Git <https://git-scm.com/>`_ in your machine, you can execute the following command in the console:

.. code-block:: bash

   name@machine:~$ pip install git+https://github.com/marcoalsina/araucaria.git

``pip`` should be able to download the required dependencies.
If you have `Conda <https://docs.conda.io/en/latest/>`_ installed (Anaconda or Miniconda), be sure to activate your environment:

.. code-block:: bash

   name@machine:~$ conda activate <yourenvironment>

Install with http
-----------------

Alternatively, you can download the source code and install ``araucaria`` directly.
Open up a `terminal` and execute the following commands:

.. code-block:: bash

    name@machine:~$ wget https://github.com/marcoalsina/araucaria/archive/master.zip
    name@machine:~$ unzip master.zip
    name@machine:~$ cd araucaria-master
    name@machine:~$ pip install .
