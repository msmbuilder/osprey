from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np


class AcquisitionFunction(object):

    def __init__(self, surrogate, acquisition_params, n_dims, n_iter=50, max_iter=1E5):
        """
        Parameters
        ----------
        surrogate : a surrogate function which can has defined on it:
              - surrogate.predict(X) for predicting the mean and variance of the model at X
              # TODO make this the mean response, not the max of the data
              - surrogate.Y.max() for calculating the incumbent
        # TODO make this come from the surrogate, along with what type of parameter each dimension represents
        n_dims : the number of dimensions
        # TODO Decide where to put defaults - at the moment it's in the Bayesian strategy class
        acquisition_params : dictionary of acquisition function options
        n_iter : the optimizer is run this many times with different random seeds - the optimum
            is chosen from this many candidates.
        max_iter : the maximum number of iterations used the optimizer (for a single run of the optimizer)
        """
        self.surrogate = surrogate
        self.n_dims = n_dims
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.acquisition_params = acquisition_params
        self._acquisition_function = None
        self._set_acquisition()
        self.incumbent = None

    def _get_random_point(self):
        # returns a random point in the normalized hyperparameter space
        # TODO make this come from Sobol sequences class
        return np.array([np.random.uniform(low=0., high=1.)
                         for _ in range(self.n_dims)])

    def _ei(self, y_mean, y_var):
        y_std = np.sqrt(y_var)
        z = (y_mean - self.incumbent)/y_std
        result = y_std*(z*norm.cdf(z) + norm.pdf(z))
        return result

    @staticmethod
    def _ucb(y_mean, y_var, kappa=1.0):
        result = y_mean + kappa*np.sqrt(y_var)
        return result

    @staticmethod
    def _osprey(y_mean, y_var):
        return (y_mean+y_var).flatten()

    def _set_acquisition(self):
        if isinstance(self.acquisition_params, list):
            raise RuntimeError('Must specify only one acq_name function')
        if sorted(self.acquisition_params.keys()) != ['name', 'params']:
            raise RuntimeError('strategy/params/acq_name must contain keys '
                               '"name" and "params" ONLY')
        if self.acquisition_params['name'] not in ['ei', 'ucb', 'osprey']:
            raise RuntimeError('strategy/params/acq_name name must be one of '
                               '"ei", "ucb", "osprey"')

        f = eval('self._' + self.acquisition_params['name'])

        def g(y_mean, y_var):
            return f(y_mean, y_var, **self.acquisition_params['params'])

        self._acquisition_function = g

    def get_best_candidate(self):
        """
        Returns
        ----------
        best_candidate : the best candidate hyper-parameters as defined by
        """
        # TODO make this best mean response
        self.incumbent = self.surrogate.Y.max()

        # Objective function
        def z(x):
            # TODO make spread of points around x and take mean value.
            x = x.copy().reshape(-1, self.n_dims)
            y_mean, y_var = self.surrogate.predict(x)
            af = self._acquisition_function(y_mean=y_mean, y_var=y_var)
            # TODO make -1 dependent on flag in inputs for either max or minimization
            return (-1) * af

        # Optimization loop
        af_values = []
        af_args = []

        for i in range(self.n_iter):
            init = self._get_random_point()
            res = minimize(z, init, bounds=self.n_dims * [(0., 1.)],
                           options={'maxiter': int(self.max_iter), 'disp': 0})
            af_args.append(res.x)
            af_values.append(res.fun)

        # Choose the best
        af_values = np.array(af_values).flatten()
        af_args = np.array(af_args)
        best_index = int(np.argmin(af_values))
        best_candidate = af_args[best_index]
        return best_candidate


