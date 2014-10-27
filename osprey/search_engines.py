from __future__ import print_function, absolute_import, division
import sys
import json
import inspect

from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urljoin

from sklearn.utils import check_random_state
try:
    from hyperopt import (Trials, tpe, fmin, STATUS_OK, STATUS_RUNNING,
                          STATUS_FAIL)
except ImportError:
    # hyperopt is optional, but required for hyperopt_tpe()
    pass

from .utils import dict_is_subset
from .search_space import EnumVariable

__all__ = ['random', 'hyperopt_tpe']


def random(history, searchspace, random_state=None):
    """Randomly suggest params from searchspace.

    Parameters
    ----------
    history : list of 3-tuples
        History of past function evaluations. Each element in history should
        be a tuple `(params, score, status)`, where `params` is a dict mapping
        parameter names to values
    searchspace : SearchSpace
        Instance of search_space.SearchSpace
    random_state :i nteger or numpy.RandomState, optional
        The random seed for sampling. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Notes
    -----
    `history` is not used by this function, but present so that we have a
    common interface between all of the search engines.

    Returns
    -------
    new_params : dict
    """
    return searchspace.rvs(random_state)


def hyperopt_tpe(history, searchspace, random_state=None):
    """
    Suggest params to maximize an objective function based on the
    function evaluation history using a tree of Parzen estimators (TPE),
    as implemented in the hyperopt package.

    Use of this function requires that hyperopt be installed.

    Parameters
    ----------
    history : list of 3-tuples
        History of past function evaluations. Each element in history should
        be a tuple `(params, score, status)`, where `params` is a dict mapping
        parameter names to values
    searchspace : SearchSpace
        Instance of search_space.SearchSpace
    random_state :i nteger or numpy.RandomState, optional
        The random seed for sampling. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Returns
    -------
    new_params : dict
    """
    # This function is very odd, because as far as I can tell there's
    # no real documented API for any of the internals of hyperopt. Its
    # execution model is that hyperopt calls your objective function (instead
    # of merely providing you with suggested points, and then you calling
    # the function yourself), and its very tricky (for me) to use the internal
    # hyperopt data structures to get these predictions out directly.

    # so they path we take in this function is to construct a synthetic
    # hyperopt.Trials database which from the `history`, and then call
    # hyoperopt.fmin with a dummy objective function that logs the value used,
    # and then return that value to our client.

    # The form of the hyperopt.Trials database isn't really documented in
    # the code -- most of this comes from reverse engineering it, by running
    # fmin() on a simple function and then inspecting the form of the
    # resulting trials object.
    if 'hyperopt' not in sys.modules:
        raise ImportError('No module named hyperopt')

    random = check_random_state(random_state)
    hp_searchspace = searchspace.to_hyperopt()

    trials = Trials()
    for i, (params, score, status) in enumerate(history):
        if status == 'SUCCEEDED':
            # we're doing maximization, hyperopt.fmin() does minimization,
            # so we need to swap the sign
            result = {'loss': -score, 'status': STATUS_OK}
        elif status == 'PENDING':
            result = {'status': STATUS_RUNNING}
        elif status == 'FAILED':
            result = {'status': STATUS_FAIL}
        else:
            raise RuntimeError('unrecognized status: %s' % status)

        # the vals key in the trials dict is basically just the params
        # dict, but enum variables (hyperopt hp.choice() nodes) are
        # different, because the index of the parameter is specified
        # in vals, not the parameter itself.

        vals = {}
        for var in searchspace:
            if isinstance(var, EnumVariable):
                # get the index in the choices of the parameter, and use
                # that.
                matches = [i for i, c in enumerate(var.choices)
                           if c == params[var.name]]
                assert len(matches) == 1
                vals[var.name] = matches
            else:
                # the other big difference is that all of the param values
                # are wrapped in length-1 lists.
                vals[var.name] = [params[var.name]]

        trials.insert_trial_doc({
            'misc': {
                'cmd': ('domain_attachment', 'FMinIter_Domain'),
                'idxs': dict((k, [i]) for k in hp_searchspace.keys()),
                'tid': i,
                'vals': vals,
                'workdir': None},
            'result': result,
            'tid': i,
            # bunch of fixed fields that hyperopt seems to require
            'owner': None, 'spec': None, 'state': 2, 'book_time': None,
            'exp_key': None, 'refresh_time': None, 'version': 0
            })

    trials.refresh()
    chosen_params_container = []

    def mock_fn(x):
        # http://stackoverflow.com/a/3190783/1079728
        # to get around no nonlocal keywork in python2
        chosen_params_container.append(x)
        return 0

    fmin(fn=mock_fn, algo=tpe.suggest, space=hp_searchspace, trials=trials,
         max_evals=len(trials.trials)+1, **_hyperopt_fmin_random_kwarg(random))
    chosen_params = chosen_params_container[0]

    return chosen_params


def _hyperopt_fmin_random_kwarg(random):
    if 'rstate' in inspect.getargspec(fmin).args:
        # 0.0.3-dev version uses this argument
        kwargs = {'rstate': random}
    elif 'rseed' in inspect.getargspec(fmin).args:
        # 0.0.2 version uses different argument
        kwargs = {'rseed': random.randint(2**32-1)}
    return kwargs


def moe_rest(history, searchspace, random_state=None, moe_url=None):
    """Suggest params to maximize an objective function based on the
    function evaluation history using a Gaussian Process Expected Improvement
    (GPEI) algorithm, as implemented by MOE [1].

    This function connects to a MOE REST API over the internet -- the
    base URL for the MOE service must be passed in `moe_url`.

    .. [1] http://yelp.github.io/MOE/index.html
    """
    # configurable
    noise_variance = 0.1

    endpoint = urljoin(moe_url, '/gp/next_points/epi')

    points_sampled = []
    points_being_sampled = []
    for param_dict, score, status in history:
        # transform points into the MOE domain. This invloves bringing
        # int and enum variables to floating point, etc.
        point = [var.point_to_moe(param_dict[var.name]) for var in searchspace]
        if status == 'SUCCEEDED':
            points_sampled.append({
                'point': point,
                'value': -score,
                'value_var': noise_variance,
            })
        elif status == 'PENDING':
            points_being_sampled.append(point)
        elif status == 'FAILED':
            pass
            # not sure how to deal with these yet
        else:
            raise RuntimeError('unrecognized status: %s' % status)

    data = {
        'num_to_sample': 1,
        'domain_info': {
            'dim': searchspace.n_dims,
            'domain_bounds': [var.domain_to_moe() for var in searchspace],
        }, 'gp_historical_info': {
            'points_sampled': points_sampled
        },
        'points_being_sampled': points_being_sampled
    }

    # call MOE
    resp = urlopen(endpoint, json.dumps(data))
    result = json.loads(resp.read())

    # check erro field
    expected = {u'optimizer_success': {
        u'gradient_descent_tensor_product_domain_found_update': True}}
    if not dict_is_subset(expected, result['status']):
        raise ValueError('failure from MOE. received %s' % result)

    # Note that MOE only deals with float-valued variables, so we have
    # a transform step on either side, where int and enum valued variables
    # are transformed before calling moe, and then the result suggested by
    # MOE needs to be reverse-transformed.
    out = {}
    for moevalue, var in zip(result['points_to_sample'][0], searchspace):
        out[var.name] = var.point_from_moe(float(moevalue))

    return out
