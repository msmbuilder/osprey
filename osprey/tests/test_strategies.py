from __future__ import print_function, absolute_import, division

import sys
from six import iteritems
import numpy as np
from numpy.testing.decorators import skipif

from osprey.search_space import SearchSpace
from osprey.search_space import IntVariable, EnumVariable, FloatVariable
from osprey.strategies import RandomSearch, HyperoptTPE, GP, GridSearch

try:
    from hyperopt import hp, fmin, tpe, Trials
except:
    pass


def test_random():
    searchspace = SearchSpace()
    searchspace.add_float('x', -10, 10)
    random = np.random.RandomState(0)
    RandomSearch(seed=random).suggest([], searchspace)


def test_grid():
    searchspace = SearchSpace()
    searchspace.add_enum('x', [1, 2])
    searchspace.add_jump('y', min=3, max=4, num=2)
    grid_search = GridSearch()
    suggestions = [grid_search.suggest([], searchspace) for _ in range(4)]
    suggestions = [(s['x'], s['y']) for s in suggestions]
    assert suggestions == [(1, 3), (1, 4), (2, 3), (2, 4)], "Didn't examine whole space correctly"


def test_check_repeated_params():
    searchspace = SearchSpace()
    searchspace.add_enum('x', [1, 2])
    searchspace.add_jump('y', min=3, max=4, num=2)

    history = []
    grid_search1 = GridSearch()
    for _ in range(4):
        params = grid_search1.suggest(history, searchspace)
        history.append((params, 0.0, 'SUCCEEDED'))

    grid_search2 = GridSearch()
    for _ in range(4):
        params = grid_search2.suggest(history, searchspace)
        assert grid_search2.is_repeated_suggestion(params, history)

    history = []
    grid_search3 = GridSearch()
    for _ in range(4):
        params = grid_search3.suggest(history, searchspace)
        history.append((params, 0.0, 'FAILED'))

    grid_search4 = GridSearch()
    for _ in range(4):
        params = grid_search4.suggest(history, searchspace)
        assert not grid_search4.is_repeated_suggestion(params, history)


def hyperopt_x2_iterates(n_iters=100):
    iterates = []
    trials = Trials()
    random = np.random.RandomState(0)

    def fn(params):
        iterates.append(params['x'])
        return params['x']**2

    for i in range(n_iters):
        fmin(fn=fn, algo=tpe.suggest, max_evals=i+1, trials=trials,
             space={'x': hp.uniform('x', -10, 10)},
             **HyperoptTPE._hyperopt_fmin_random_kwarg(random))

    return np.array(iterates)


def our_x2_iterates(n_iters=100):
    history = []
    searchspace = SearchSpace()
    searchspace.add_float('x', -10, 10)
    random = np.random.RandomState(0)

    # note the switch of sign, because _our_ function hyperopt_tpe is
    # a maximizer, not a minimizer
    def fn(params):
        return -params['x']**2

    for i in range(n_iters):
        params = HyperoptTPE(seed=random).suggest(history, searchspace)
        history.append((params, fn(params), 'SUCCEEDED'))

    return np.array([h[0]['x'] for h in history])


@skipif('hyperopt.fmin' not in sys.modules, 'this test requires hyperopt')
def test_1():
    ours = our_x2_iterates(25)
    ref = hyperopt_x2_iterates(25)

    np.testing.assert_array_equal(ref, ours)

# TODO this error message needs changing.
@skipif('GPy' not in sys.modules, 'this test requires hyperopt')
def test_gp():
    searchspace = SearchSpace()
    searchspace.add_float('x', -10, 10)
    searchspace.add_float('y', 1, 10, warp='log')
    searchspace.add_int('z', -10, 10)
    searchspace.add_enum('w', ['opt1', 'opt2'])

    history = [(searchspace.rvs(), np.random.random(), 'SUCCEEDED')
               for _ in range(4)]

    params = GP().suggest(history, searchspace)
    for k, v in iteritems(params):
        assert k in searchspace.variables
        if isinstance(searchspace[k], EnumVariable):
            assert v in searchspace[k].choices
        elif isinstance(searchspace[k], FloatVariable):
            assert searchspace[k].min <= v <= searchspace[k].max
        elif isinstance(searchspace[k], IntVariable):
            assert searchspace[k].min <= v <= searchspace[k].max
        else:
            assert False
