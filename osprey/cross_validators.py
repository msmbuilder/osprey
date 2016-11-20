from __future__ import print_function, absolute_import, division

from .utils import num_samples

import numpy as np
from sklearn import model_selection


class BaseCVFactory(object):
    short_name = None

    def load(self):
        raise NotImplementedError('should be implemented in subclass')

    def create(self, X, y):
        raise NotImplementedError('should be implemented in subclass')


class ShuffleSplitFactory(BaseCVFactory):
    __doc__ = model_selection.ShuffleSplit.__doc__
    short_name = ['shufflesplit', 'ShuffleSplit']

    def __init__(self, n_splits=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def create(self, X, y=None):
        return model_selection.ShuffleSplit(n_splits=self.n_splits,
                                            test_size=self.test_size,
                                            train_size=self.train_size,
                                            random_state=self.random_state
                                            )


class KFoldFactory(BaseCVFactory):
    __doc__ = model_selection.KFold.__doc__
    short_name = ['kfold', 'KFold']

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def create(self, X, y=None):
        return model_selection.KFold(n_splits=self.n_splits,
                                     shuffle=self.shuffle,
                                     random_state=self.random_state
                                     )


class LeaveOneOutFactory(BaseCVFactory):
    __doc__ = model_selection.LeaveOneOut.__doc__
    short_name = ['loo', 'LeaveOneOut']

    def __init__(self):
        pass

    def create(self, X, y=None):
        return model_selection.LeaveOneOut()


class LeavePOutFactory(BaseCVFactory):
    __doc__ = model_selection.LeavePOut.__doc__
    short_name = ['lpo', 'LeavePOut']

    def __init__(self, p=3):
        self.p = p

    def create(self, X, y=None):
        return model_selection.LeavePOut(p=self.p)


class StratifiedShuffleSplitFactory(BaseCVFactory):
    __doc__ = model_selection.StratifiedShuffleSplit.__doc__
    short_name = ['stratifiedshufflesplit', 'StratifiedShuffleSplit']

    def __init__(self, n_splits=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def create(self, X, y):
        return model_selection.StratifiedShuffleSplit(n_splits=self.n_splits,
                                                      test_size=self.test_size,
                                                      train_size=self.train_size,
                                                      random_state=self.random_state
                                                      )


class StratifiedKFoldFactory(BaseCVFactory):
    __doc__ = model_selection.StratifiedKFold.__doc__
    short_name = ['stratifiedkfold', 'StratifiedKFold']

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def create(self, X, y):
        return model_selection.StratifiedKFold(n_splits=self.n_splits,
                                               shuffle=self.shuffle,
                                               random_state=self.random_state
                                               )


class TimeSeriesSplitFactory(BaseCVFactory):
    __doc__ = model_selection.TimeSeriesSplit.__doc__
    short_name = ['timeseriessplit', 'TimeSeriesSplit']

    def __init__(self, n_splits=3):
        self.n_splits = n_splits

    def create(self, X, y=None):
        return model_selection.TimeSeriesSplit(
            n_splits=self.n_splits)


class FixedCVFactory(BaseCVFactory):
    """
    Cross-validator to use with a fixed, held-out validation set.

    Parameters
    ----------
    start : int
        Start index of validation set.
    stop : int, optional
        Stop index of validation set.
    """
    short_name = ['fixed', 'Fixed']

    def __init__(self, start, stop=None):
        self.valid = slice(start, stop)

    def create(self, X, y):
        indices = np.arange(num_samples(X))
        valid = indices[self.valid]
        train = np.setdiff1d(indices, valid)

        test_fold = np.ones(num_samples(X))
        test_fold[train] = -1

        return model_selection.PredefinedSplit(test_fold)
