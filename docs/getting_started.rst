
Getting Started
===============

Introduction
------------

Getting started with Osprey is as simple as setting up a single ``YAML``
configuration file. This configuration file will contain your model
estimators (``estimator``), hyperparameter search strategy
(``strategy``), hyperparameter search space (``search_space``), dataset
information (``dataset_loader``), cross-validation strategy (``cv``),
and a path to a ``SQL``-like database (``trials``). This page will go
over how to set up a basic Osprey toy project and then a more realistic
example for a `molecular
dynamics <https://en.wikipedia.org/wiki/Molecular_dynamics>`__ dataset.

``scikit-learn`` Example
------------------------

First, we'll begin with a basic C-Support Vector Classification example
using ``scikit-learn`` to introduce the basic ``YAML`` fields for Osprey. To
tell Osprey that we want to use ``sklearn``'s ``SVC`` as our estimator,
we can type:

.. code:: yaml

    estimator:
      entry_point: sklearn.svm.SVC

If we want to use random search to decide where to search next in
hyperparameter space, we can add:

.. code:: yaml

    strategy:
      name: random

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

To perform 5-fold cross validation, we add:

.. code:: yaml

    cv: 5

To load the digits classification example dataset from ``scikit-learn``,
we write:

.. code:: yaml

    dataset_loader:
      name: sklearn_dataset
      params:
        method: load_digits

And finally we need to list the SQL database where our cross-validation
results will be saved:

.. code:: yaml

    trials:
        uri: sqlite:///osprey-trials.db

Once this all has been written to a ``YAML`` file (e.g. ``config.yaml``),
we can start an osprey job in the command-line by invoking:

.. code:: bash

    $ osprey worker config.yaml


``msmbuilder`` Example
----------------------

Now that we understand the basics, we can move on to a more practical example.
This section will go over how to set up a Osprey configuration for
cross-validating Markov state models from protein simulations. Our model will
be constructed by first calculating torsion angles, performing dimensionality
reduction using tICA, clustering using mini-batch k-means, and, finally, an
maximum-likelihood estimated Markov state model.

We begin by defining a ``Pipeline`` which will construct our desired model:

.. code:: yaml

    estimator:
        eval: |
            Pipeline([
                    ('featurizer', DihedralFeaturizer()),
                    ('tica', tICA()),
                    ('cluster', MiniBatchKMeans()),
                    ('msm', MarkovStateModel(n_timescales=5, verbose=False)),
            ])
        eval_scope: msmbuilder

Notice that we can easily set default parameters (e.g. ``msm.n_timescales``)
in our ``Pipeline`` even if we don't plan on optimizing them.

If we wish to use `gaussian process
prediction <https://en.wikipedia.org/wiki/Gaussian_process#Gaussian_process_prediction.2C_or_kriging>`__
to decide where to search in hyperparameter space, we can add:

.. code:: yaml

    strategy:
        name: gp
        params:
          seeds: 50

In this example, we'll be optimizing the type of featurization,
the number of cluster centers and the number of independent components:

.. code:: yaml

    search_space:

    featurizer__types:
      choices:
        - ['phi', 'psi']
        - ['phi', 'psi', 'chi1']
      type: enum

    tica__n_components:
      min: 2
      max: 5
      type: int

    cluster__n_clusters:
      min: 10
      max: 100
      type: int

As seen in the previous example, we'll set ``tica__n_components`` and
``cluster__n_clusters`` as integers with a set range. Notice that we can
change which torsion angles to use in our featurization by creating an ``enum``
which contains a list of different dihedral angle types.


In this example, we'll be using 50-50 ``shufflesplit`` cross-validation.
This method is optimal for Markov state model cross-validation, as it maximizes
the amount of unique data available in your training and test sets:

.. code:: yaml

    cv:
      name: shufflesplit
    params:
      n_iter: 5
      test_size: 0.5

We'll be using MDTraj to load our trajectories. Osprey already includes an
``mdtraj`` dataset loader to make it easy to list your trajectory and topology
files as a glob-string:

.. code:: yaml

    dataset_loader:
      name: mdtraj
      params:
        trajectories: ~/local/msmbuilder/Tutorial/XTC/*/*.xtc
        topology: ~/local/msmbuilder/Tutorial/native.pdb
        stride: 1

And finally we need to list the SQL database where our cross-validation
results will be saved:

.. code:: yaml

    trials:
      uri: sqlite:///osprey-trials.db


Just as before, once this all has been written to a ``YAML`` file
we can start an osprey job in the command-line by invoking:

.. code:: bash

    $ osprey worker config.yaml
