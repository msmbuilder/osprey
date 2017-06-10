from __future__ import print_function, absolute_import, division
import sys
import inspect
import socket

import numpy as np
from sklearn.utils import check_random_state
from sklearn.grid_search import ParameterGrid
try:
    from hyperopt import (Trials, tpe, fmin, STATUS_OK, STATUS_RUNNING,
                          STATUS_FAIL)
except ImportError:
    # hyperopt is optional, but required for hyperopt_tpe()
    pass

try:
    from GPy import kern
    from GPy.kern import RBF, Fixed, Bias
    from GPy.util.linalg import tdot
    from GPy.models import GPRegression
    from scipy.optimize import minimize
    # If the GPy modules fail we won't do this unnecessarily.
    from .entry_point import load_entry_point
    KERNEL_BASE_CLASS = kern.src.kern.Kern
except:
    # GPy is optional, but required for gp
    GPRegression = kern = minimize = None
    pass
from .search_space import EnumVariable

DEFAULT_TIMEOUT = socket._GLOBAL_DEFAULT_TIMEOUT


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

    @staticmethod
    def is_repeated_suggestion(params, history):
        """
        Parameters
        ----------
        params : dict
            Trial param set
        history : list of 3-tuples
            History of past function evaluations. Each element in history
            should be a tuple `(params, score, status)`, where `params` is a
            dict mapping parameter names to values

        Returns
        -------
        is_repeated_suggestion : bool
        """
        if any(params == hparams and hstatus == 'SUCCEEDED' for hparams, hscore, hstatus in history):
            return True
        else:
            return False


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
        for i, (params, scores, status) in enumerate(history):
            if status == 'SUCCEEDED':
                # we're doing maximization, hyperopt.fmin() does minimization,
                # so we need to swap the sign
                result = {'loss': -np.mean(scores), 'status': STATUS_OK}
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
            kwargs = {'rstate': random, 'allow_trials_fmin': False}
        elif 'rseed' in inspect.getargspec(fmin).args:
            # 0.0.2 version uses different argument
            kwargs = {'rseed': random.randint(2**32-1)}
        return kwargs


class GP(BaseStrategy):
    short_name = 'gp'

    def __init__(self, kernels, seed=None, seeds=1, max_feval=5E4, max_iter=1E5):
        self.seed = seed
        self.seeds = seeds
        self.max_feval = max_feval
        self.max_iter = max_iter
        self.model = None
        self.n_dims = None
        self.kernel = None
        self._kerns = kernels

    def _create_kernel(self):
        # Turn into entry points.
        # TODO use eval to allow user to specify internal variables for kernels (e.g. V) in config file.
        kernels = []
        for kern in self._kerns:
            try:
                kwargs = kern['kwargs']
                options = kern['options']
                name = kern['name']
            except KeyError as e:
                raise RuntimeError(e.message)

            kernel_ep = load_entry_point(name, 'strategy/params/kernels')
            if issubclass(kernel_ep, KERNEL_BASE_CLASS):
                if options['independent']:
                    kernel = np.sum([kernel_ep(1, **kwargs, active_dims=[i]) for i in range(self.n_dims)])
                else:
                    kernel = kernel_ep(self.n_dims, **kwargs)
            if not isinstance(kernel, KERNEL_BASE_CLASS):
                raise RuntimeError('strategy/params/kernel must load a'
                                   'GPy derived Kernel')
            kernels.append(kernel)
        self.kernel = np.sum(kernels)

    def _fit_model(self, X, Y):
        model = GPRegression(X, Y, self.kernel)
        model.optimize(messages=False, max_f_eval=self.max_feval)
        self.model = model

    def _get_random_point(self):
        return np.array([np.random.uniform(low=0., high=1.)
                         for i in range(self.n_dims)])

    def _optimize(self, init=None):
        if not init:
            init = self._get_random_point()

        def z(x):
            # TODO Could use options dict to specify what type of kernel to create when
            y = x.copy().reshape(-1, self.n_dims)
            s, v = self.model.predict(y)
            return -(s+v).flatten()

        return minimize(z, init, bounds=self.n_dims*[(0., 1.)],
                        options={'maxiter': self.max_iter, 'disp': 0}).x

    def _get_data(self, history, searchspace):
        X = []
        Y = []
        V = []
        ignore = []
        for param_dict, scores, status in history:
            # transform points into the GP domain. This invloves bringing
            # int and enum variables to floating point, etc.
            if status == 'FAILED':
                # not sure how to deal with these yet
                continue

            point = searchspace.point_to_gp(param_dict)
            if status == 'SUCCEEDED':
                X.append(point)
                Y.append(np.mean(scores))
                V.append(np.var(scores))

            elif status == 'PENDING':
                ignore.append(point)
            else:
                raise RuntimeError('unrecognized status: %s' % status)

        return (np.array(X).reshape(-1, self.n_dims),
                np.array(Y).reshape(-1, 1),
                np.array(V).reshape(-1, 1),
                np.array(ignore).reshape(-1, self.n_dims))

    def _from_gp(self, result, searchspace):

        # Note that GP only deals with float-valued variables, so we have
        # a transform step on either side, where int and enum valued variables
        # are transformed before calling gp, and then the result suggested by
        # GP needs to be reverse-transformed.
        out = {}
        for gpvalue, var in zip(result, searchspace):
            out[var.name] = var.point_from_gp(float(gpvalue))

        return out

    def _is_within(self, point, X, tol=1E-2):
        if True in (np.sqrt(((point - X)**2).sum(axis=0)) <= tol):
            return True
        return False

    def suggest(self, history, searchspace, max_tries=5):
        if not GPRegression:
            raise ImportError('No module named GPy')
        if not minimize:
            raise ImportError('No module named SciPy')

        if len(history) < self.seeds:
            return RandomSearch().suggest(history, searchspace)

        self.n_dims = searchspace.n_dims

        X, Y, V, ignore = self._get_data(history, searchspace)
        # TODO make _create_kernel accept optional args.
        self._create_kernel()
        self._fit_model(X, Y)

        suggestion = self._optimize()

        if suggestion in ignore or self._is_within(suggestion, X):
            return RandomSearch().suggest(history, searchspace)

        return self._from_gp(suggestion, searchspace)


class GridSearch(BaseStrategy):
    short_name = 'grid'

    def __init__(self):
        self.param_grid = None
        self.current = -1

    def suggest(self, history, searchspace):
        # Convert searchspace to param_grid
        if self.param_grid is None:
            if not all(isinstance(v, EnumVariable) for v in searchspace):
                raise RuntimeError("GridSearchStrategy is defined only for all-enum search space")

            self.param_grid = ParameterGrid(dict((v.name, v.choices) for v in searchspace))

        # NOTE: there is no way of signaling end of parameters to be searched against
        # so user should pick correctly number of evaluations
        self.current += 1
        return self.param_grid[self.current % len(self.param_grid)]
