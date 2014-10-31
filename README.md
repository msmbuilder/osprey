Osprey [![Build Status](https://travis-ci.org/rmcgibbo/osprey.svg?branch=master)](https://travis-ci.org/rmcgibbo/osprey) [![PyPi version](https://pypip.in/v/osprey/badge.png)](https://pypi.python.org/pypi/osprey/) [![Supported Python versions](https://pypip.in/py_versions/osprey/badge.svg)](https://pypi.python.org/pypi/osprey/) [![License](https://pypip.in/license/osprey/badge.svg)](https://pypi.python.org/pypi/osprey/)
======

osprey is an easy-to-use tool for hyperparameter optimization for machine
learning algorithms in python using scikit-learn (or using scikit-learn
compatible APIs).

Each osprey experiment combines an dataset, an estimator, a search space
(and engine), cross validation and asynchronous serialization for distributed
parallel optimization of model hyperparameters.

[Full documentation](http://osprey.rtfd.org)

Example (with [mixtape](https://github.com/rmcgibbo/mixtape) models/datasets)
-------------------------------------------------------------
```
$ cat config.yaml
estimator:
  eval_scope: mixtape
  eval: |
    Pipeline([
        ('featurizer', DihedralFeaturizer(types=['phi', 'psi'])),
        ('cluster', MiniBatchKMeans()),
        ('msm', MarkovStateModel(n_timescales=5, verbose=False)),
    ])

search_space:
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

dataset_loader:
  name: mdtraj
  params:
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

osprey version:  0.2_10_g18392d9_dirty-py2.7.egg
time:            October 27, 2014 10:44 PM
hostname:        dn0a230538.sunet
cwd:             /private/var/folders/yb/vpt17lxs67vf02qpvgvjrc5m0000gn/T/tmpDgBwlU
pid:             99407

Loading config file:     config.yaml...
Loading trials database: sqlite:///osprey-trials.db (table = "trials")...

Loading dataset...
  100 elements without labels
Instantiated estimator:
  Pipeline(steps=[('featurizer', DihedralFeaturizer(sincos=True, types=['phi', 'psi'])), ('tica', tICA(gamma=0.05, lag_time=1, n_components=4, weighted_transform=False)), ('cluster', MiniBatchKMeans(batch_size=100, compute_labels=True, init='k-means++',
        init_size=None, max_iter=100, max_no_improvement=...toff=1, lag_time=1, n_timescales=5, prior_counts=0,
         reversible_type='mle', verbose=False))])
Hyperparameter search space:
  featurizer__types        	(enum)    choices = (['phi', 'psi'], ['phi', 'psi', 'chi1'])
  cluster__n_clusters      	(int)         10 <= x <= 100

----------------------------------------------------------------------
Beginning iteration                                              1 / 1
----------------------------------------------------------------------
History contains: 0 trials
Choosing next hyperparameters with random...
  {'cluster__n_clusters': 20, 'featurizer__types': ['phi', 'psi']}

Fitting 5 folds for each of 1 candidates, totalling 5 fits
[Parallel(n_jobs=1)]: Done   1 jobs       | elapsed:    0.3s
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    1.8s finished
---------------------------------
Success! Model score = 4.080646
(best score so far   = 4.080646)
---------------------------------

1/1 models fit successfully.
time:         October 27, 2014 10:44 PM
elapsed:      4 seconds.
osprey worker exiting.
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

On python2.6, the `argparse` and `importlib` backports are also required
