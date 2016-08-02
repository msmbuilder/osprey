from __future__ import print_function, absolute_import, division
import numpy as np
import os
import tempfile

from six.moves import cPickle
from sklearn.cluster import KMeans

from osprey.config import Config
from osprey.search_space import IntVariable, FloatVariable, EnumVariable
from osprey.strategies import RandomSearch, HyperoptTPE, GP


os.environ['OSPREYRC'] = ' '


def test_estimator_pickle():
    with tempfile.NamedTemporaryFile('w+b', 0) as f:

        cPickle.dump(KMeans(), f)

        config = Config.fromdict({
            'estimator': {'pickle': f.name}
        }, check_fields=False)
        assert isinstance(config.estimator(), KMeans)


def test_estimator_eval_1():
    config = Config.fromdict({
        'estimator': {
            'eval': 'KMeans()',
            'eval_scope': 'sklearn',
        }
    }, check_fields=False)
    assert isinstance(config.estimator(), KMeans)


def test_estimator_eval_2():
    config = Config.fromdict({
        'estimator': {
            'eval': 'KMeans()',
            'eval_scope': ['sklearn'],
        }
    }, check_fields=False)
    assert isinstance(config.estimator(), KMeans)


def test_estimator_entry_point():
    config = Config.fromdict({
        'estimator': {
            'entry_point': 'sklearn.cluster.KMeans',
        }
    }, check_fields=False)
    assert isinstance(config.estimator(), KMeans)


def test_estimator_entry_point_params():
    config = Config.fromdict({
        'estimator': {
            'entry_point': 'sklearn.cluster.KMeans',
            'params': {
                'n_clusters': 15
            }
        }
    }, check_fields=False)
    assert isinstance(config.estimator(), KMeans)
    assert config.estimator().n_clusters == 15


def test_search_space():
    config = Config.fromdict({
        'search_space': {
            'intvar': {'type': 'int', 'min': 1, 'max': 2},
            'logivar': {'type': 'int', 'min': 1, 'max': 2, 'warp': 'log'},
            'fvar': {'type': 'float', 'min': 1, 'max': 3.5},
            'logfvar': {'type': 'float', 'min': 1, 'max': 2.5, 'warp': 'log'},
            'enumvar': {'type': 'enum', 'choices': [1, False]},
            'jumpivar': {'type': 'jump',  'min': 1, 'max': 3, 'num': 3, 'var_type': int},
            'jumpfvar': {'type': 'jump',  'min': 1, 'max': 3, 'num': 3, 'var_type': float},
            'logjumpivar': {'type': 'jump',  'min': 10, 'max': 1000, 'num': 3, 'warp': 'log', 'var_type': int},
            'logjumpfvar': {'type': 'jump',  'min': 10, 'max': 1000, 'num': 3, 'warp': 'log', 'var_type': float}
        }}, check_fields=False)

    searchspace = config.search_space()
    assert searchspace['intvar'] == IntVariable('intvar', 1, 2, warp=None)
    assert searchspace['logivar'] == IntVariable('logivar', 1, 2, warp='log')
    assert searchspace['fvar'] == FloatVariable('fvar', 1, 3.5, warp=None)
    assert searchspace['logfvar'] == FloatVariable('logfvar', 1, 2.5,
                                                   warp='log')
    assert searchspace['enumvar'] == EnumVariable('enumvar', [1, False])
    assert searchspace['jumpivar'] == EnumVariable('jumpivar', [1, 2, 3])
    assert searchspace['jumpfvar'] == EnumVariable('jumpfvar', [1.0, 2.0, 3.0])
    assert searchspace['logjumpivar'] == EnumVariable('logjumpivar', [10, 100, 1000])
    assert searchspace['logjumpfvar'] == EnumVariable('logjumpfvar', [10.0, 100.0, 1000.0])


def test_strategy_random():
    config = Config.fromdict({
        'strategy': {'name': 'random'}
    }, check_fields=False)
    assert isinstance(config.strategy(), RandomSearch)


def test_search_engine_hyperopt_tpe():
    config = Config.fromdict({
        'strategy': {'name': 'hyperopt_tpe'}
    }, check_fields=False)
    assert isinstance(config.strategy(), HyperoptTPE)


def test_search_engine_gp():
    config = Config.fromdict({
        'strategy': {'name': 'gp'}
    }, check_fields=False)
    assert isinstance(config.strategy(), GP)


def test_scoring():
    config = Config.fromdict({
        'scoring': 'sdfsfsdf'
    }, check_fields=False)
    assert config.scoring() is 'sdfsfsdf'


def test_random_seed():
    config = Config.fromdict({
        'random_seed': 42
    }, check_fields=False)
    assert config.random_seed() == 42


def test_cv_1():
    from sklearn.cross_validation import ShuffleSplit
    for name in ['shufflesplit', 'ShuffleSplit']:
        config = Config.fromdict({
            'cv': {'name': name, 'params': {'n_iter': 10}}
        }, check_fields=False)
        cv = config.cv(range(100))
        assert isinstance(cv, ShuffleSplit)
        assert cv.n_iter == 10


def test_stratified_cv():
    from sklearn.cross_validation import StratifiedShuffleSplit
    config = Config.fromdict({
        'cv': {'name': 'stratifiedshufflesplit', 'params': {'n_iter': 10}}
    }, check_fields=False)
    cv = config.cv(range(100), np.random.randint(2, size=100))
    assert isinstance(cv, StratifiedShuffleSplit)
    assert cv.n_iter == 10
