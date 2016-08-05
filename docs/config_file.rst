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

Please note that grid search only supports ``enum`` and ``jump`` variables.

.. _dataset_loader:

Dataset Loader
--------------

Example: ::

  dataset_loader:
    name: joblib
    params:
      filenames: ~/path/to/file.pkl

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

Here's a complete list of supported iterators, along with their ``name`` mappings:

* ``kfold``: `KFold <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html#sklearn.cross_validation.KFold>`_
* ``shufflesplit``: `ShuffleSplit <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.ShuffleSplit.html#sklearn.cross_validation.ShuffleSplit>`_
* ``loo``: `LeaveOneOut <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.LeaveOneOut.html#sklearn.cross_validation.LeaveOneOut>`_
* ``stratifiedkfold``: `StratifiedKFold <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedKFold.html#sklearn.cross_validation.StratifiedKFold>`_
* ``stratifiedshufflesplit``: `StratifiedShuffleSplit <http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html#sklearn.cross_validation.StratifiedShuffleSplit>`_

.. _trials:

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
