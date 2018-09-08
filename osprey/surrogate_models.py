"""
Surrogate model classes for Bayesian type models.  These are separate from the strategy classes which just operate the
models.

"""

from __future__ import print_function, absolute_import, division
import numpy as np


class BaseSurrogate(object):
    def get_maximum(self):
        """
        Parameters
        ----------
        Returns
        -------
        y : The maximum mean value of the model
        X : The corresponding parameters
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


class GaussianProcess(object):
    pass
