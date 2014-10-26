Osprey [![Build Status](https://travis-ci.org/rmcgibbo/osprey.svg?branch=master)](https://travis-ci.org/rmcgibbo/osprey)
======

osprey is an easy-to-use tool for hyperparameter optimization for machine
learning algorithms in python using scikit-learn (or using scikit-learn
compatible APIs).

Each osprey experiment combines an dataset, an estimator, a search space
(and engine), cross validation and asynchronous serialization for distributed
parallel optimization of model hyperparameters.

Example (with [mixtape](https://github.com/rmcgibbo/mixtape) models/datasets)
-------------------------------------------------------------
```
$ cat config.yaml
estimator:
    eval: |
        Pipeline([
                ('featurizer', DihedralFeaturizer(types=['phi', 'psi'])),
                ('cluster', MiniBatchKMeans()),
                ('msm', MarkovStateModel(n_timescales=5, verbose=False)),
        ])
search:
    engine: hyperopt_tpe
    space:
        cluster__n_clusters:
            min: 10
            max: 100
            type: int
        featurizer__types:
            choices:
                - ['phi', 'psi']
                - ['phi', 'psi', 'chi1']
            type: enum
cv: 5
dataset:
    trajectories: ~/local/msmbuilder/Tutorial/XTC/*/*.xtc
    topology: ~/local/msmbuilder/Tutorial/native.pdb
    stride: 1
trials:
    uri: sqlite:///osprey-trials.db
```

Then run `osprey worker`. You can run multiple parallel instances
of `osprey worker` simultaniously on a cluster too.

```
$ osprey worker config.yaml
======================================================================
= osprey is a tool for machine learning hyperparameter optimization. =
======================================================================

Loading .ospreyrc from /Users/rmcgibbo/.ospreyrc...
Loading config file from config.yaml...
Loading trials database from sqlite:///osprey-trials.db (table = "trials")...

Loading dataset...
  100 elements without labels
Instantiated estimator:
  Pipeline(steps=[('featurizer', DihedralFeaturizer(sincos=True, types=['phi', 'psi'])), ('cluster', MiniBatchKMeans(batch_size=100, compute_labels=True, init='k-means++',
        init_size=None, max_iter=100, max_no_improvement=10, n_clusters=8,
        n_init=3, random_state=None, reassignment_ratio=0.01, tol=0.0,
        verbose=0)), ('msm', MarkovStateModel(ergodic_cutoff=1, lag_time=1, n_timescales=5, prior_counts=0,
         reversible_type='mle', verbose=False))])
Hyperparameter search space:
  featurizer__types	    (enum)    choices = (['phi', 'psi'], ['phi', 'psi', 'chi1'])
  cluster__n_clusters	(int)         10 <= x <= 100

----------------------------------------------------------------------
Beginning iteration                                              1 / 1
----------------------------------------------------------------------
History contains: 0 trials
Choosing next hyperparameters with hyperopt_tpe...
  {'cluster__n_clusters': 80, 'featurizer__types': ('phi', 'psi', 'chi1')}

Fitting 5 folds for each of 1 candidates, totalling 5 fits
[Parallel(n_jobs=1)]: Done   1 jobs       | elapsed:    1.0s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    3.8s finished
---------------------------------
Success! Model score = 4.370210
(best score so far   = 4.370210)
---------------------------------

1/1 models fit successfully.
osprey-worker exiting.
```
You can dump the database to JSON or CSV with `osprey dump`.


Installation
------------
```
# grab the latest version from github
$ pip install git+git://github.com/rmcgibbo/osprey.git
```

```
# or clone the repo yourself and run `setup.py`
$ git clone https://github.com/rmcgibbo/osprey.git
$ cd osprey && python setup.py install
```

Dependencies
------------
- `six`
- `pyyaml`
- `numpy`
- `scikit-learn`
- `sqlalchemy`
- `hyperopt` (recommended, required for `engine=hyperopt_tpe`)
- `scipy` (optional, for testing)
- `nose` (optional, for testing)
