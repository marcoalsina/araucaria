Installation
============

The alpha release of ``araucaria`` can be installed with ``pip``, which automatically verifies and downloads the required dependencies.

If you prefer to handle the installation and execution of ``araucaria`` inside a `Conda <https://docs.conda.io/en/latest/>`_ environment, please check the following section. Otherwise you can proceed to the install section.

Managing a conda environment
----------------------------

If you have Anaconda or Miniconda installed in your machine, make sure to activate your environment before installing ``araucaria``:

.. code-block:: bash

   name@machine:~$ conda activate <yourenvironment>

In most cases ``araucaria`` should install correctly in a conda environment. However, you may prefer to create a `new enviromnent <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_ for the library, particularly if you want to work on new features or compile a local copy of the documentation.

In such cases we offer the following ``yaml`` environment file: :download:`araucaria.yml <_downloads/araucaria.yml>`.
Once downloaded you can execute the following commands in a terminal to create and activate a development environment for araucaria:

.. code-block:: bash

	name@machine:~$ conda env create -f araucaria.yml
	name@machine:~$ conda activate araucaria

Once your conda environment has been configured, you have the following install options:

- :ref:`Install with Git`
- :ref:`Install with HTTP`

Install with Git
----------------

If you have `Git <https://git-scm.com/>`_ installed in your machine, you can install ``araucaria`` directly by executing the following command in a terminal:

.. code-block:: bash

   name@machine:~$ pip install git+https://github.com/marcoalsina/araucaria.git


Install with HTTP
-----------------

Alternatively, you can download the source code and install ``araucaria`` directly.
Open up a terminal and execute the following commands:

.. code-block:: bash

    name@machine:~$ wget https://github.com/marcoalsina/araucaria/archive/master.zip
    name@machine:~$ unzip master.zip
    name@machine:~$ cd araucaria-master
    name@machine:~$ pip install .
