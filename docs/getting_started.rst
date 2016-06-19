
Getting Started
===============

Getting started with Osprey is as easy as setting up a single ``YAML``
configuration file. This configuration file will contain your model
estimators (``estimator``), hyperparameter search strategy
(``strategy``), hyperparameter search space (``search_space``), dataset
information (``dataset_loader``), cross-validation strategy (``cv``),
and a path to a ``SQL``-like database (``trials``). This page will go
over how to set up a basic Osprey toy project and then a more realistic
example for a `molecular
dynamics <https://en.wikipedia.org/wiki/Molecular_dynamics>`__ dataset.

First, we'll begin with a simple C-Support Vector Classification example
using ``sklearn`` to introduce the basic ``YAML`` fields for Osprey. To
tell Osprey that we want to use ``sklearn``'s ``SVC`` as our estimator,
we can type:

.. code:: yaml

    estimator:
      entry_point: sklearn.svm.SVC

If we want to use `gaussian process
prediction <https://en.wikipedia.org/wiki/Gaussian_process#Gaussian_process_prediction.2C_or_kriging>`__
to decide where to search in hyperparameter space, we can add:

.. code:: yaml

    strategy:
      name: gp
      params:
        seeds: 5

The search space can be defined for any hyperparameter available in the
``estimator`` class. Here we can adjust the value range of the ``C`` and
``gamma`` hyperparamters. We'll search over a range of 0.1 to 10 for
``C`` and over 1E-5 to 1 in log-space (note: ``warp: log``) for
``gamma``.

.. code:: yaml

    search_space:
      C:
        min: 0.1
        max: 10
        type: float

      gamma:
        min: 1e-5
        max: 1
        warp: log
        type: float

.. code:: yaml

    cv: 5

.. code:: yaml

    dataset_loader:
      name: sklearn_dataset
      params:
        method: load_digits

.. code:: yaml

    trials:
        uri: sqlite:///osprey-trials.db
