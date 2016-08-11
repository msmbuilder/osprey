
Getting Started
===============

Introduction
------------

Getting started with Osprey is as simple as setting up a single ``YAML``
configuration file. This configuration file will contain your model
estimators, hyperparameter search strategy, hyperparameter search space,
dataset information, cross-validation strategy, and a path to a
``SQL``-like database. You can use the command ``osprey skeleton`` to
generate an example configuration file.

First, we will describe how to prepare your dataset for Osprey. Then, we will
show how to use Osprey for a simple scikit-learn classification task. And
finally, we will show how one might use Osprey to model a
`molecular dynamics (MD) <https://en.wikipedia.org/wiki/Molecular_dynamics>`_
dataset.


Formatting Your Dataset
-----------------------

Osprey supports a wide variety of file formats (see :ref:`here
<config_file#dataset-loader>` for a full list); however, some of these offer
more flexibility than others. In general, your data should be formatted as a
two-dimensional array, where columns represent different features or variables
and rows are individual observations. This is a fairly natural format for
delimiter-separated value files (e.g ``.csv``, ``.tsv``), which Osprey handles
natively using ``DSVDatasetLoader``. If you choose to save your dataset as a
``.pkl``, ``.npz``, or ``.npy`` file, it's as simple as saving your datasets as
2d NumPy arrays. Note that each file should only contain a single NumPy array.
If you'd like to store multiple arrays to a single file for Osprey to read, we
recommend storing your data in an HDF5 file.

When working with datasets with labels or a response variable, there are slight
differences in how your data should be stored. With delimiter-separated value,
NumPY files, and HDF5 files, you can simply append these as an additional
column and then select its index as the ``y_col`` parameter in the corresponding
dataset loader. With Pickle and JobLib files, you should instead save each as a
separate value in a ``dict`` object and declare the corresponding keys
(``x_name`` and ``y_name``) in the ``JoblibDatasetLoader``. Please note that if
you wish to use multiple response variables the ``JoblibDatasetLoader`` is the
only dataset loader currently equipped to do so.


SVM Classification with ``scikit-learn``
----------------------------------------

Let's train a basic C-Support Vector Classification example using
``scikit-learn`` and introduce the basic ``YAML`` fields for Osprey. To
tell Osprey that we want to use ``sklearn``'s ``SVC`` as our estimator, we
can type:

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

Once this all has been written to a ``YAML`` file (in this example
``config.yaml``), we can start an osprey job in the command-line by invoking:

.. code:: bash

    $ osprey worker config.yaml


Molecular Dynamics with ``msmbuilder``
--------------------------------------

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
