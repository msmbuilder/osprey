import os
import tempfile

from six.moves import cPickle
from sklearn.cluster import KMeans

from osprey.config import Config
from osprey.search_space import IntVariable, FloatVariable, EnumVariable
from osprey import search_engines


os.environ['OSPREYRC'] = ' '


def mock_eval_globals():
    return {'KMeans': KMeans}


def test_estimator_pickle():
    with tempfile.NamedTemporaryFile('w+b', 0) as f:
        cPickle.dump(KMeans(), f)

        config = Config.fromdict({
            'estimator': {'pickle': f.name}
        }, check_fields=False)
        assert isinstance(config.estimator(), KMeans)


def test_estimator_eval():
    config = Config.fromdict({
        'estimator': {
            'eval': 'KMeans()',
            '__eval_globals__': 'osprey.tests.test_config.mock_eval_globals'
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


def test_search_space():
    config = Config.fromdict({
        'search': {'space': {
            'intvar': {'type': 'int', 'min': 1, 'max': 2},
            'fvar': {'type': 'float', 'min': 1, 'max': 3.5},
            'logvar': {'type': 'float', 'min': 1, 'max': 2.5, 'warp': 'log'},
            'enumvar': {'type': 'enum', 'choices': [1, False]},
        }}}, check_fields=False)
    searchspace = config.search_space()
    assert searchspace['intvar'] == IntVariable('intvar', 1, 2)
    assert searchspace['fvar'] == FloatVariable('fvar', 1, 3.5, warp=None)
    assert searchspace['logvar'] == FloatVariable('logvar', 1, 2.5, warp='log')
    assert searchspace['enumvar'] == EnumVariable('enumvar', [1, False])


def test_search_engine_random():
    config = Config.fromdict({
        'search': {'engine': 'random'}
    }, check_fields=False)
    assert config.search_engine() is search_engines.random


def test_search_engine_hyperopt_tpe():
    config = Config.fromdict({
        'search': {'engine': 'hyperopt_tpe'}
    }, check_fields=False)
    assert config.search_engine() is search_engines.hyperopt_tpe


def test_scoring():
    config = Config.fromdict({
        'scoring': 'sdfsfsdf'
    }, check_fields=False)
    assert config.scoring() is 'sdfsfsdf'


def test_cv():
    config = Config.fromdict({
        'cv': 2
    }, check_fields=False)
    assert config.cv() == 2
