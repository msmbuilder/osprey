import numpy as np
from hyperopt import hp, fmin, tpe

from osprey.search_space import SearchSpace
from osprey.search_engines import hyperopt_tpe


def hyperopt_x2_iterates(n_iters=100):
    iterates = []

    def fn(params):
        iterates.append(params['x'])
        return params['x']**2

    fmin(fn=fn, algo=tpe.suggest, max_evals=n_iters,
         space={'x': hp.uniform('x', -10, 10)},
         rstate=np.random.RandomState(0))
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


def test_1():
    ref = hyperopt_x2_iterates(50)
    ours = our_x2_iterates(50)

    np.testing.assert_array_equal(ref, ours)
