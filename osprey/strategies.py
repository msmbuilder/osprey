from __future__ import print_function, absolute_import, division
import sys
import json
import inspect
import socket
from argparse import Namespace

import numpy as np
from six.moves.urllib.error import HTTPError, URLError
from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlparse
DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT


from sklearn.utils import check_random_state
try:
    from hyperopt import (Trials, tpe, fmin, STATUS_OK, STATUS_RUNNING,
                          STATUS_FAIL)
except ImportError:
    # hyperopt is optional, but required for hyperopt_tpe()
    pass

from .search_space import EnumVariable


class BaseStrategy(object):
    short_name = None

    def suggest(self, history, searchspace):
        """
        Parameters
        ----------
        history : list of 3-tuples
            History of past function evaluations. Each element in history
            should be a tuple `(params, score, status)`, where `params` is a
            dict mapping parameter names to values
        searchspace : SearchSpace
            Instance of search_space.SearchSpace
        random_state :i nteger or numpy.RandomState, optional
            The random seed for sampling. If an integer is given, it fixes the
            seed. Defaults to the global numpy random number generator.

        Returns
        -------
        new_params : dict
        """
        raise NotImplementedError()


class RandomSearch(BaseStrategy):
    short_name = 'random'

    def __init__(self, seed=None):
        self.seed = seed

    def suggest(self, history, searchspace):
        """Randomly suggest params from searchspace.
        """
        return searchspace.rvs(self.seed)


class HyperoptTPE(BaseStrategy):
    short_name = 'hyperopt_tpe'

    def __init__(self, seed=None):
        self.seed = seed

    def suggest(self, history, searchspace):
        """
        Suggest params to maximize an objective function based on the
        function evaluation history using a tree of Parzen estimators (TPE),
        as implemented in the hyperopt package.

        Use of this function requires that hyperopt be installed.
        """
        # This function is very odd, because as far as I can tell there's
        # no real documented API for any of the internals of hyperopt. Its
        # execution model is that hyperopt calls your objective function
        # (instead of merely providing you with suggested points, and then
        # you calling the function yourself), and its very tricky (for me)
        # to use the internal hyperopt data structures to get these predictions
        # out directly.

        # so they path we take in this function is to construct a synthetic
        # hyperopt.Trials database which from the `history`, and then call
        # hyoperopt.fmin with a dummy objective function that logs the value
        # used, and then return that value to our client.

        # The form of the hyperopt.Trials database isn't really documented in
        # the code -- most of this comes from reverse engineering it, by
        # running fmin() on a simple function and then inspecting the form of
        # the resulting trials object.
        if 'hyperopt' not in sys.modules:
            raise ImportError('No module named hyperopt')

        random = check_random_state(self.seed)
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
             max_evals=len(trials.trials)+1,
             **self._hyperopt_fmin_random_kwarg(random))
        chosen_params = chosen_params_container[0]

        return chosen_params

    @staticmethod
    def _hyperopt_fmin_random_kwarg(random):
        if 'rstate' in inspect.getargspec(fmin).args:
            # 0.0.3-dev version uses this argument
            kwargs = {'rstate': random}
        elif 'rseed' in inspect.getargspec(fmin).args:
            # 0.0.2 version uses different argument
            kwargs = {'rseed': random.randint(2**32-1)}
        return kwargs


class MOE(BaseStrategy):
    short_name = 'moe'

    def __init__(self, url=None, noise_variance=0.1, method='constant_liar',
                 lie_method='constant_liar_min'):
        self.url = url
        self.noise_variance = noise_variance
        self.method = method
        if method not in ('epi', 'kriging', 'constant_liar'):
            raise ValueError("method must be one of 'epi', 'kriging', "
                             "'constant_liar'")

        if lie_method not in ('constant_liar_min', 'constant_liar_max',
                              'constant_liar_mean'):
            raise ValueError("method lie_method be one of 'constant_liar_min',"
                             " 'constant_liar_max', 'constant_liar_mean'")

        if self.url:
            self._use_local_moe = False
        else:
            try:
                # check if it's importable
                from moe.views.rest.gp_next_points_epi import GpNextPointsEpi
                # force flake8 not to complain about unused import
                bool(GpNextPointsEpi)
                self._use_local_moe = True
            except ImportError:
                msg = ('with strategy = "moe", either "url" parameter must be '
                       'set to point to an external MOE REST API, or you must '
                       'have a local copy of MOE installed and importable. '
                       'See the MOE documentation at '
                       'http://yelp.github.io/MOE/ for details')
                raise RuntimeError(msg)

    def suggest(self, history, searchspace):
        """Suggest params to maximize an objective function based on the
        function evaluation history using a Gaussian Process Expected
        Parallel Improvement (GPEI) algorithm, as implemented by MOE [1].

        .. [1] http://yelp.github.io/MOE/index.html
        """
        request = self._build_request(history, searchspace)

        if self._use_local_moe:
            results = self._call_moe_locally(request)
        else:
            results = self._call_moe_rest_api(request)

        return self._build_response(results, searchspace)

    def _build_request(self, history, searchspace):
        points_sampled = []
        points_being_sampled = []
        for param_dict, score, status in history:
            # transform points into the MOE domain. This invloves bringing
            # int and enum variables to floating point, etc.
            point = searchspace.point_to_moe(param_dict)
            if status == 'SUCCEEDED':
                points_sampled.append({
                    'point': point,
                    'value': -score,
                    'value_var': self.noise_variance,
                })
            elif status == 'PENDING':
                points_being_sampled.append(point)
            elif status == 'FAILED':
                pass
                # not sure how to deal with these yet
            else:
                raise RuntimeError('unrecognized status: %s' % status)

        # shift the 'score' to be zero mean, which is suggested
        # in the MOE docs
        # http://yelp.github.io/MOE/moe.views.schemas.html#moe.views.schemas.base_schemas.GpHistoricalInfo
        mean = np.mean([p['value'] for p in points_sampled])
        for p in points_sampled:
            p['value'] = p['value'] - mean

        request = {
            'num_to_sample': 1,
            'domain_info': {
                'dim': searchspace.n_dims,
                'domain_bounds': [var.domain_to_moe() for var in searchspace],
            }, 'gp_historical_info': {
                'points_sampled': points_sampled
            },
            'points_being_sampled': points_being_sampled
        }

        if self.method == 'constant_liar':
            request['lie_method'] = self.lie_method
            request['lie_noise_variance'] = 1e-12
        if self.method == 'kriging':
            request['kriging_noise_variance'] = 1e-12
        return request

    def _build_response(self, results, searchspace):
        if 'optimizer_success' not in results['status']:
            raise ValueError('failure from MOE. received %s' % results)

        # Note that MOE only deals with float-valued variables, so we have
        # a transform step on either side, where int and enum valued variables
        # are transformed before calling moe, and then the result suggested by
        # MOE needs to be reverse-transformed.
        out = {}
        for moevalue, var in zip(results['points_to_sample'][0], searchspace):
            out[var.name] = var.point_from_moe(float(moevalue))

        return out

    def _call_moe_rest_api(self, request):
        base = self.url
        if base.endswith('/'):
            base = base[:-1]
        endpoint = base + '/gp/next_points/' + self.method
        parsed = urlparse(endpoint)
        if parsed.netloc == '':
            endpoint = 'http://' + endpoint

        if not bool(urlparse(endpoint)):
            raise RuntimeError('moe url "%s" is not a valid endpoint' %
                               endpoint)

        jdata = json.dumps(request).encode('utf-8')
        resp = urlopen_with_retries(endpoint, jdata)
        result = json.loads(resp.read().decode('utf-8'))
        return result

    def _call_moe_locally(self, request):
        from moe.views.rest.gp_next_points_epi import GpNextPointsEpi
        from moe.views.rest.gp_next_points_kriging import GpNextPointsKriging
        from moe.views.rest.gp_next_points_constant_liar import \
            GpNextPointsConstantLiar

        mock_object = Namespace(json_body=request)
        if self.method == 'epi':
            gp_next_points_epi = GpNextPointsEpi(mock_object)
            result = gp_next_points_epi.gp_next_points_epi_view()
        elif self.method == 'kriging':
            gp_next_points_kriging = GpNextPointsKriging(mock_object)
            result = gp_next_points_kriging.gp_next_points_kriging_view()
        elif self.method == 'constant_liar':
            gp_next_points_cl = GpNextPointsConstantLiar(mock_object)
            result = gp_next_points_cl.gp_next_points_constant_liar_view()
        else:
            raise ValueError('unrecognized method: %s' % self.method)

        return result


def urlopen_with_retries(url, data=None, timeout=DEFAULT_TIMEOUT, n_retries=3):
    error = None
    for i in range(n_retries):
        try:
            return urlopen(url=url, data=data, timeout=timeout)
        except (URLError, HTTPError) as e:
            error = e
            continue
    print("Error hitting url=%s" % url, file=sys.stderr)
    raise error
