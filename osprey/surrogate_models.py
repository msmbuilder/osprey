"""
Surrogate model classes for Bayesian strategy.  These are separate from the strategy classes which just operate the
models.

"""

from __future__ import print_function, absolute_import, division
import numpy as np

try:
    from GPy import kern
    from GPy.kern import RBF, Fixed, Bias
    from GPy.util.linalg import tdot
    from GPy.models import GPRegression
    from scipy.optimize import minimize
    from scipy.stats import norm

    # If the GPy modules fail we won't do this unnecessarily.
    from .entry_point import load_entry_point
    KERNEL_BASE_CLASS = kern.src.kern.Kern
except ImportError:
    # GPy is optional, but required for gp
    GPRegression = kern = minimize = None
    pass


# TODO Make all of these sklearn estimators
class MaximumLikelihoodGaussianProcess(object):
    """
    Gaussian Process model which has its own hyperparameters chosen by a maximum likelihood process
    """

    # Can't have instantiation of model without supplying data
    def __init__(self, X, Y, kernel, max_feval):
        if not GPRegression:
            raise ImportError('No module named GPy')
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.model = GPRegression(X=self.X, Y=self.Y, kernel=self.kernel)
        self.max_feval = max_feval
        # TODO make this a variable.
        self.num_restarts = 20

    def fit(self):
        """
        Fits the model with random restarts.
        :return:
        """
        self.model.optimize_restarts(num_restarts=self.num_restarts, verbose=False)
        # TODO check whether this is actually needed.
        self.model.optimize(messages=False, max_f_eval=self.max_feval)

    def predict(self, x):
        return self.model.predict(Xnew=x)


class GaussianProcessKernel(object):
    def __init__(self, kernel_params, n_dims):
        """
        Kernels for the Gaussian Process surrogates
        :param kernel_params: the param list from yaml.
        """
        self.kernel_params = kernel_params
        self.kernel = None  # The final kernel
        self.n_dims = n_dims
        self._create_kernel()

    def _create_kernel(self):
        """
        creates an additive kernel
        """
        # Check kernels
        kernels = self.kernel_params
        if not isinstance(kernels, list):
            raise RuntimeError('Must provide enumeration of kernels')
        for kernel in kernels:
            if sorted(list(kernel.keys())) != ['name', 'options', 'params']:
                raise RuntimeError(
                    'strategy/params/kernels must contain keys: "name", "options", "params"')

        # Turn into entry points.
        # TODO use eval to allow user to specify internal variables for kernels (e.g. V) in config file.
        kernels = []
        for kern in self.kernel_params:
            params = kern['params']
            options = kern['options']
            name = kern['name']
            kernel_ep = load_entry_point(name, 'strategy/params/kernels')
            if issubclass(kernel_ep, KERNEL_BASE_CLASS):
                if options['independent']:
                    # TODO Catch errors here?  Estimator entry points don't catch instantiation errors
                    kernel = np.sum([kernel_ep(1, active_dims=[i], **params) for i in range(self.n_dims)])
                else:
                    kernel = kernel_ep(self.n_dims, **params)
            if not isinstance(kernel, KERNEL_BASE_CLASS):
                raise RuntimeError('strategy/params/kernel must load a'
                                   'GPy derived Kernel')
            kernels.append(kernel)

        self.kernel = np.sum(kernels)



