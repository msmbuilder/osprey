"""
Surrogate model classes for Bayesian type models.  These are separate from the strategy classes which just operate the
models.

"""

from __future__ import print_function, absolute_import, division
import numpy as np


class BaseSurrogate(object):
    def get_maximum(self, func):
        """
        Parameters
        ----------
        func : a function to optimize over the model.  Could be identity, -1 (for minimization) or acquisition function
                func must take parameters of the model e.g. y or noise term (for af).
        Returns
        -------
        y = func(X*) : The maximum mean value of the function
        X* : The corresponding parameters
        """
        raise NotImplementedError()

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : parameter history
        y : response history

        Returns
        -------
        None
        """
        raise NotImplementedError()


class MarginalGP(object):
    def __init__(self, kernels, mean, ):
        pass
