.. _config_file:

Configuration File
==================

``osprey`` jobs are configured via a small configuration file, which is written
in a hand-editable `YAML <http://www.yaml.org/start.html>`_ markup.

The command ``osprey skeleton`` will create an example ``config.yaml`` file
for you to get started with. The sections of the file are described below.

.. _estimator:

Estimator
---------

The estimator section describes the model that ``osprey`` is tasked
with optimizing. It can be specified either as a python entry point,
a pickle file, or as a raw string which is passed to python's ``eval()``.
However specified, the estimator should be an instance or subclass of
sklearn's ``BaseEstimator``

Examples:

::

  estimator:
    entry_point: sklearn.linear_model.LinearRegression

::

  estimator:
    eval: Pipeline([('vectorizer', TfidfVectorizer), ('logistic', LogisticRegression())])
    eval_scope: sklearn

::

  estimator:
    pickle: my-model.pkl   # path to pickle file on disk


.. _search_space:

Search Space
------------

The search space describes the space of hyperparameters to search over
to find the best model. It is specified as the product space of
bounded intervals for different variables, which can either be of type
``int``, ``float``, or ``enum``. Variables of type ``float`` can also
be warped into log-space, which means that the optimization will be
performed on the log of the parameter instead of the parameter itself.

Example: ::

  search_space:
    logistic__C:
      min: 1e-3
      max: 1e3
      type: float
      warp: log

   logistic__penalty:
      choices:
        - l1
        - l2
     type: enum


You can also transform ``float`` and ``int`` variables into enumerables by
declaring a ``jump`` variable:

Example: ::

    search_space:
      logistic__C:
        min: 1e-3
        max: 1e3
        num: 10
        type: jump
        var_type: float
        warp: log

In the example above, we have declared a ``jump`` variable ``C`` for the
``logistic`` estimator. This variable is essentially an ``enum`` with
10 possible ``float`` values that are evenly spaced apart in log-space within
the given ``min`` and ``max`` range.


.. _strategy:

Strategy
--------

Three probablistic search strategies and grid search are supported. First,
random search (``strategy: {name: random}``) can be used, which samples
hyperparameters randomly from the search space at each model-building iteration.
Random search has `been shown to be <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`_ significantly more effiicent than pure grid search. Example: ::

  strategy:
    name: random

``strategy: {name: hyperopt_tpe}`` is an alternative strategy which uses a Tree of Parzen
estimators, described in `this paper <http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization>`_. This algorithim requires that the external
package `hyperopt <https://github.com/hyperopt/hyperopt>`_ be installed. Example: ::

  strategy:
    name: hyperopt_tpe

``osprey`` supports a Gaussian process expected improvement search
strategy, using the package `GPy <https://github.com/SheffieldML/GPy>`_, with
``strategy: {name: gp}``.
``url`` param. Example: ::

  strategy:
    name: gp

Finally, and perhaps simplest of all, is the
`grid search strategy <https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search>`_
(``strategy: {name: grid}``). Example: ::

    strategy:
      name: grid

Please note, that grid search only supports ``enum`` and ``jump`` variables.

.. _dataset_loader:

Dataset Loader
--------------

Osprey supports a wide variety of file formats. These include `pickle` files,
``numpy`` files, delimiter-separated values files (e.g. ``.csv``, ``.tsv`),
``hdf5`` files, and most molecular trajectory file formats (see `mdtraj.org <http://mdtraj.org/1.7.2/load_functions.html#format-specific-loading-functions>`_ for reference).
For more information about formatting your dataset for use with Osprey, please
refer to our :ref:`"Getting Started" <getting_started>` page.

Below is an example of using the ``dsv`` loader to load multiple ``.csv`` files
into Osprey:

Example: ::

  dataset_loader:
    name: dsv
    params:
      filenames: /path/to/files/*.csv, /another/path/to/myfile.csv
      delimiter: ','
      skip_header: 2
      skip_footer: 1
      y_col: 42
      usecols: 0, 1, 2, 3, 4, 5
      concat: True

Notice that we can pass a glob string and/or a comma-separated list of paths to
``filenames`` to tell Osprey where our data is located. ``delimiter`` defines
the separator pattern used to parse the data files (default: ``','``).
``skip_header`` and ``skip_footer`` tell Osprey how many lines to ignore at the
beginning and end of the files, respectively (default: ``0``). ``y_col`` is used
to specify which column to select as a response variable (default: ``None``).
``usecols`` can be used to specify which columns to use as explanatory variables
(default: uses all columns). And finally, ``concat`` specifies whether or not to
treat all loaded files as a single dataset (defaut: ``False``).

Here's a complete list of supported file formats, along with their loader
``name`` mappings:

* ``numpy``: `NumPy <http://docs.scipy.org/doc/numpy/neps/npy-format.html>`_ format
* ``msmbuilder``: `MSMBuilder dataset <http://msmbuilder.org/development/persistence.html>`_ format
* ``hdf5``: `HDF5 <https://www.hdfgroup.org/HDF5/whatishdf5.html>`_ format
* ``dsv``: `Delimiter-separated value (DSV) <https://en.wikipedia.org/wiki/Delimiter-separated_values>`_ format
* ``joblib``: Pickle and `Joblib <https://pythonhosted.org/joblib/persistence.html>`_ formats

In addition, we provide two additional loaders:

* ``sklearn_dataset``: Allows users to load any ``scikit-learn`` `dataset <http://scikit-learn.org/stable/datasets/#toy-datasets>`_
* ``filename``: Allows users to pass a set of filenames to the Osprey estimator. Useful for custom dataset loading.

.. _cross_validation:

Cross Validation
----------------

Many types of cross-validation iterators are supported. The simplest
option is to simply pass an ``int``, which sets up k-fold cross validation.
Example: ::

  cv: 5

To access the other iterators, use the ``name`` and ``params`` keywords: ::

  cv:
    name: shufflesplit
    params:
      n_iter: 5
      test_size: 0.5
      random_state: 42

Here's a complete list of supported iterators, along with their ``name`` mappings:

* ``kfold``: `KFold <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold>`_
* ``shufflesplit``: `ShuffleSplit <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html#sklearn.cross_validation.ShuffleSplit>`_
* ``loo``: `LeaveOneOut <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeaveOneOut.html#sklearn.cross_validation.LeaveOneOut>`_
* ``stratifiedkfold``: `StratifiedKFold <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html#sklearn.cross_validation.StratifiedKFold>`_
* ``stratifiedshufflesplit``: `StratifiedShuffleSplit <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html#sklearn.cross_validation.StratifiedShuffleSplit>`_

.. _trials:


Random Seed
----------------
In case you need reproducible Osprey trials, you can also include an
optional random seed as seen below:

Example: ::

  random_seed: 42

Please note that this makes parallel trials redundant and, thus, not
recommended when scaling across multiple jobs.

Trials Storage
--------------

Example: ::

  trials:
    # path to a databse in which the results of each hyperparameter fit
    # are stored any SQL database is suppoted, but we recommend using
    # SQLite, which is simple and stores the results in a file on disk.
    # the string format for connecting to other database is described here:
    # http://docs.sqlalchemy.org/en/rel_0_9/core/engines.html#database-urls
    uri: sqlite:///osprey-trials.db
