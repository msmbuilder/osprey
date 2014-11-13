from __future__ import print_function, absolute_import, division
import numpy as np
import os
import tempfile

from six.moves import cPickle
from sklearn.cluster import KMeans

from osprey.config import Config
from osprey.search_space import IntVariable, FloatVariable, EnumVariable
from osprey.strategies import RandomSearch, HyperoptTPE, MOE


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
            'fvar': {'type': 'float', 'min': 1, 'max': 3.5},
            'logvar': {'type': 'float', 'min': 1, 'max': 2.5, 'warp': 'log'},
            'enumvar': {'type': 'enum', 'choices': [1, False]},
        }}, check_fields=False)
    searchspace = config.search_space()
    assert searchspace['intvar'] == IntVariable('intvar', 1, 2)
    assert searchspace['fvar'] == FloatVariable('fvar', 1, 3.5, warp=None)
    assert searchspace['logvar'] == FloatVariable('logvar', 1, 2.5, warp='log')
    assert searchspace['enumvar'] == EnumVariable('enumvar', [1, False])


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


def test_search_engine_moe_1():
    config = Config.fromdict({
        'strategy': {'name': 'moe', 'params': {'url': 'sdfsdf'}}
    }, check_fields=False)
    assert isinstance(config.strategy(), MOE)


def test_search_engine_moe_2():
    config = Config.fromdict({
        'strategy': {'name': 'moe', 'params': {'url': 'abc'}}
    }, check_fields=False)
    strat = config.strategy()
    assert isinstance(strat, MOE)
    assert strat.url == 'abc'


def test_scoring():
    config = Config.fromdict({
        'scoring': 'sdfsfsdf'
    }, check_fields=False)
    assert config.scoring() is 'sdfsfsdf'


def test_cv():
    from sklearn.cross_validation import ShuffleSplit
    config = Config.fromdict({
        'cv': {'name': 'shufflesplit', 'params': {'n_iter': 10}}
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
