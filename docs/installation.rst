Installation
============

Osprey is written in Python, and can be installed with standard Python
machinery; we highly recommend using an
`Anaconda Python distribution <https://www.continuum.io/downloads>`_.


Release Version
---------------


With Anaconda, installation is as easy as:

.. code-block:: bash

  $ conda install -c omnia osprey

You can also install Osprey with `pip`:

.. code-block:: bash

  $ pip install osprey

Alternatively, you can install directly our
`GitHub repository <https://github.com/msmbuilder/osprey>`_.:

.. code-block:: bash

  $ git clone https://github.com/msmbuilder/osprey.git
  $ cd osprey && git checkout 1.1.0
  $ python setup.py install


Development Version
-------------------

To grab the latest version from github, run:

.. code-block:: bash

  $ pip install git+git://github.com/pandegroup/osprey.git

Or clone the repo yourself and run `setup.py`:

.. code-block:: bash

  $ git clone https://github.com/pandegroup/osprey.git
  $ cd osprey && python setup.py install


Dependencies
------------
- ``python>=2.7.11``
- ``six>=1.10.0``
- ``pyyaml>=3.11``
- ``numpy>=1.10.4``
- ``scipy>=0.17.0``
- ``scikit-learn>=0.17.0``
- ``sqlalchemy>=1.0.10``
- ``bokeh>=0.12.0``
- ``matplotlib>=1.5.0``
- ``GPy`` (optional, required for ``gp`` strategy)
- ``hyperopt`` (optional, required for ``hyperopt_tpe`` strategy)
- ``nose`` (optional, for testing)

You can grab most of them with conda. ::

  $ conda install six pyyaml numpy scikit-learn sqlalchemy nose bokeh matplotlib

Hyperopt can be installed with pip. ::

  $ pip install hyperopt


Getting GPy
-----------

To run the gaussian process (``gp``) search strategy, ``osprey`` uses
`GPy <https://github.com/SheffieldML/GPy>`_


To use ``gp`` search, you must install GPy on the machines you use to run
osprey. For easy installation, use the conda binary packages that
we've compiled. ::

  conda install -c omnia gpy
