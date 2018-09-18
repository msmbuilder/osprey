from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np


class AcquisitionFunction(object):

    def __init__(self, surrogate, acq_name, acq_kwargs, n_dims, n_iter=50, max_iter=1E5):
        """
        Parameters
        ----------
        surrogate : a surrogate function which can has defined on it:
              - surrogate.predict(X) for predicting the mean and variance of the model at X
              # TODO make this the mean response, not the max of the data
              - surrogate.Y.max() for calculating the incumbent
        # TODO make this come from the surrogate, along with what type of parameter each dimension represents
        n_dims : the number of dimensions
        acq_name : the type of acq_name functions, currently implemented:
             ei - expected improvement
             ucb - upper confidence bound
        acq_kwargs : any keyword arguments for the acq_name functions
        n_iter : the optimizer is run this many times with different random seeds - the optimum
            is chosen from this many candidates.
        max_iter : the maximum number of iterations used the optimizer (for a single run of the optimizer)
        """
        self.surrogate = surrogate
        self.acq_name = acq_name
        self.acq_kwargs = acq_kwargs
        self.n_dims = n_dims
        self.n_iter = n_iter
        self.max_iter = max_iter

    def get_random_point(self):
        # returns a random point in the normalized hyperparameter space
        # TODO make this come from Sobol sequences class
        return np.array([np.random.uniform(low=0., high=1.)
                         for _ in range(self.n_dims)])

    @staticmethod
    def ei(y_mean, y_var, y_best):
        y_std = np.sqrt(y_var)
        z = (y_mean - y_best)/y_std
        result = y_std*(z*norm.cdf(z) + norm.pdf(z))
        return result

    @staticmethod
    def ucb(y_mean, y_var, kappa=1.0):
        result = y_mean + kappa*np.sqrt(y_var)
        return result

    def evaluate(self, x):
        pass

    def optimize(self):
        """
        Returns
        ----------
        candidate : the best candidate hyper-parameters
        """
        pass


