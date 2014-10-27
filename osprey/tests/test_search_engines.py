from __future__ import print_function, absolute_import, division
import sys
from six import iteritems
import numpy as np
from numpy.testing.decorators import skipif
try:
    from hyperopt import hp, fmin, tpe, Trials
except:
    pass

from osprey.search_space import (SearchSpace, EnumVariable, FloatVariable,
                                 IntVariable)
from osprey.search_engines import hyperopt_tpe, _hyperopt_fmin_random_kwarg
from osprey.search_engines import moe_rest


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
             **_hyperopt_fmin_random_kwarg(random))

    return np.array(iterates)


def our_x2_iterates(n_iters=100):
    history = []
    searchspace = SearchSpace()
    searchspace.add_float('x', -10, 10)
    random = np.random.RandomState(0)

    # note the switch of sign, because _our_ function hyperopt_tpe is
    # a maximizer, not a minimizer
    fn = lambda params: -params['x']**2

    for i in range(n_iters):
        params = hyperopt_tpe(history, searchspace, random)
        history.append((params, fn(params), 'SUCCEEDED'))

    return np.array([h[0]['x'] for h in history])


@skipif('hyperopt.fmin' not in sys.modules, 'this test requires hyperopt')
def test_1():
    ours = our_x2_iterates(25)
    ref = hyperopt_x2_iterates(25)

    np.testing.assert_array_equal(ref, ours)

def test_moe_rest_1():
    moe_url = 'http://vspm9.stanford.edu/'
    searchspace = SearchSpace()
    searchspace.add_float('x', -10, 10)
    searchspace.add_float('y', 1, 10, warp='log')
    searchspace.add_int('z', -10, 10)
    searchspace.add_enum('w', ['opt1', 'opt2'])

    for i in range(10):
        history = [(searchspace.rvs(), np.random.random(), 'SUCCEEDED')
                   for _ in range(4)]
        params = moe_rest(history, searchspace, moe_url=moe_url)
        for k, v in iteritems(params):
            assert k in searchspace.variables
            if isinstance(searchspace[k], EnumVariable):
                assert v in searchspace[k].choices
            elif isinstance(searchspace[k], FloatVariable):
                assert searchspace[k].min <= v < searchspace[k].max
            elif isinstance(searchspace[k], IntVariable):
                assert searchspace[k].min <= v <= searchspace[k].max
            else:
                assert False


