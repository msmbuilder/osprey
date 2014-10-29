Osprey
======

Osprey is a tool for practical hyperparameter optimization of machine learning
algorithms. It's designed to provide a practical, **easy to use** way for
application scientists to find parameters that maximize the cross-validation
score of a model. Osprey is being developed by researchers at Stanford
University with primary application areas in computations protein dynamics and
drug design.

Overview
--------
Osprey is a command line tool. It runs using a simple configuration file which
sets up the problem by describing the :ref:`estimator <estimator>`,
:ref:`search space <search_space>`, :ref:`strategy <strategies>`,
:ref:`dataset <dataset_loader>`, :ref:`cross validation <cross_validation>`,
and storage for the :ref:`results <trials>`.


Related tools include and Spearmint, Hyperopt, MOE



::

    $ cat config.yaml
    estimator:
      entry_point: sklearn.linear_model.LogisticRegression

    search_space:
      penalty:
        choices: ['l1', 'l2']
        type: enum
      C:
        min: 1e-10
        max: 10
        warp: log
        type: float

    cv: 5

    dataset_loader:
      name: sklearn_dataset
      params:
        method: load_digits

    trials:
        uri: sqlite:///osprey-trials.db



.. toctree::
   :maxdepth: 2

   installation
   estimator
   search_space
   batch_submission


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

