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
    # Gaussian Process model which has its own hyperparameters chosen by a maximum likelihood process

    # Can't have instantiation of model without supplying data
    def __init__(self, X, Y, kernel, max_feval):
        if not GPRegression:
            raise ImportError('No module named GPy')
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.model = GPRegression(X=self.X, Y=self.Y, kernel=self.kernel)
        self.max_feval = max_feval

    def fit(self):
        self.model.optimize_restarts(num_restarts=20, verbose=False)
        self.model.optimize(messages=False, max_f_eval=self.max_feval)

    def predict(self, x):
        return self.model.predict(Xnew=x)




