Installation
============

Osprey is written in Python, and can be installed with standard python
machinery


Development Version
-------------------

.. code-block:: bash

  # grab the latest version from github
  $ pip install git+git://github.com/pandegroup/osprey.git

.. code-block:: bash

  # or clone the repo yourself and run `setup.py`
  $ git clone https://github.com/pandegroup/osprey.git
  $ cd osprey && python setup.py install

Release Version
---------------

Currently, **we recommend that you use the development version**, since things are
moving fast. However, release versions from PyPI can be installed using ``pip``.

.. code-block:: bash

  # grab the release version from PyPI
  $ pip install osprey


Dependencies
------------
- ``six``
- ``pyyaml``
- ``numpy``
- ``scikit-learn``
- ``sqlalchemy``
- ``hyperopt`` (recommended, required for ``engine=hyperopt_tpe``)
- ``GPy`` (recommended, required for ``engine=gp``)
- ``scipy`` (optional, for testing)
- ``nose`` (optional, for testing)

You can grab most of them with conda. ::

  $ conda install six pyyaml numpy scikit-learn sqlalchemy nose

Hyperopt can be installed with pip. ::

  $ pip install hyperopt


Getting GPy
-----------

To use the Gaussian Process (``gp``) search strategy, ``osprey`` uses
GPy <https://github.com/SheffieldML/GPy>


To use the GPy python API, you must install GPy on the machines you use to run
osprey. For easy installation, use the conda binary packages that
we've compiled. ::

  conda install -c omnia gp
