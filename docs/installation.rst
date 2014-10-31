Installation
============

Osprey is written in Python, and can be installed with standard python
machinery


Development Version
-------------------

.. code-block:: bash

  # grab the latest version from github
  $ pip install git+git://github.com/rmcgibbo/osprey.git

.. code-block:: bash

  # or clone the repo yourself and run `setup.py`
  $ git clone https://github.com/rmcgibbo/osprey.git
  $ cd osprey && python setup.py install


Dependencies
------------
- ``six``
- ``pyyaml``
- ``numpy``
- ``scikit-learn``
- ``sqlalchemy``
- ``hyperopt`` (recommended, required for ``engine=hyperopt_tpe``)
- ``MOE`` (recommended, required for ``engine=moe``)
- ``scipy`` (optional, for testing)
- ``nose`` (optional, for testing)

Most of them with conda ::

  $ conda install six pyyaml numpy scikit-learn sqlalchemy nose

Getting Moe
-----------

To use the MOE search strategy, ``osprey`` can call MOE via two interfaces

 - MOE's REST API, over HTTP
 - MOE's python API

Using the first requires that you set up a MOE server somewhere. The recommended
way to do this is via the MOE docker image. See the `MOE documentation <https://github.com/Yelp/MOE#install-in-docker>`_ for more information.

To use the MOE python API, you must install MOE on the machines you use to run
osprey. The MOE documentation has some information on how to do this, but it
can be tricky. An easier alternative, on linux, is to use the conda binary packages that we compiled. ::

  conda install -c https://conda.binstar.org/rmcgibbo moe

See `the github repo <https://github.com/rmcgibbo/conda-moe>`_ for more info
on the compilation of these binaries.
