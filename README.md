Osprey
======
[![Build Status](https://travis-ci.org/pandegroup/osprey.svg?branch=master)](https://travis-ci.org/pandegroup/osprey)
[![Documentation Status](https://readthedocs.org/projects/osprey/badge/?version=latest)](http://osprey.rtfd.org)

osprey is an easy-to-use tool for hyperparameter optimization for machine
learning algorithms in python using scikit-learn (or using scikit-learn
compatible APIs).

Each osprey experiment combines an dataset, an estimator, a search space
(and engine), cross validation and asynchronous serialization for distributed
parallel optimization of model hyperparameters.

Documentation
------------
For full documentation, please visit the [Osprey homepage](http://msmbuilder.org/osprey).


Installation
------------

If you have an Anaconda Python distribution, installation is as easy as:
```
$ conda install -c omnia osprey
```

You can also install with `pip`:
```
$ pip install git+git://github.com/pandegroup/osprey.git
```

Alternatively, you can install directly from this GitHub repo:
```
$ git clone https://github.com/msmbuilder/osprey.git
$ cd osprey && python setup.py install
```


Example using [MSMBuilder](https://github.com/msmbuilder/msmbuilder)
-------------------------------------------------------------
Below is an example of an osprey `config` file to cross validate Markov state
models based on varying the number of clusters and dihedral angles used in a
model:
```yaml
estimator:
  eval_scope: msmbuilder
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
of `osprey worker` simultaneously on a cluster too.

```
$ osprey worker config.yaml

...

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


Dependencies
------------
- `six`
- `pyyaml`
- `numpy`
- `scikit-learn`
- `sqlalchemy`
- `GPy` (optional, required for `gp` strategy)
- `scipy` (optional, required for `gp` strategy)
- `hyperopt` (optional, required for `hyperopt_tpe` strategy)
- `nose` (optional, for testing)
