from __future__ import print_function, absolute_import, division

import numpy as np


class BaseCVFactory(object):
    short_name = None

    def load(self):
        raise NotImplementedError('should be implemented in subclass')

    def create(self, X, y):
        raise NotImplementedError('should be implemented in subclass')


class ShuffleSplitFactory(BaseCVFactory):
    short_name = 'shufflesplit'

    def __init__(self, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n_iter = n_iter
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def create(self, X, y=None):
        from sklearn.cross_validation import ShuffleSplit

        return ShuffleSplit(len(X), n_iter=self.n_iter,
                            test_size=self.test_size,
                            train_size=self.train_size,
                            random_state=self.random_state)


class KFoldFactory(BaseCVFactory):
    short_name = 'kfold'

    def __init__(self, n_folds=3, shuffle=False, random_state=None):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def create(self, X, y=None):
        from sklearn.cross_validation import KFold

        return KFold(len(X), n_folds=self.n_folds, shuffle=self.shuffle,
                     random_state=self.random_state)


class LeaveOneOutFactory(BaseCVFactory):
    short_name = 'loo'

    def __init__(self):
        pass

    def create(self, X, y=None):
        from sklearn.cross_validation import LeaveOneOut

        return LeaveOneOut(len(X))


class StratifiedShuffleSplitFactory(BaseCVFactory):
    short_name = 'stratifiedshufflesplit'

    def __init__(self, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n_iter = n_iter
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def create(self, X, y):
        from sklearn.cross_validation import StratifiedShuffleSplit

        return StratifiedShuffleSplit(y, n_iter=self.n_iter,
                                      test_size=self.test_size,
                                      train_size=self.train_size,
                                      random_state=self.random_state)


class StratifiedKFoldFactory(BaseCVFactory):
    short_name = 'stratifiedkfold'

    def __init__(self, n_folds=3, shuffle=False, random_state=None):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def create(self, X, y):
        from sklearn.cross_validation import StratifiedKFold

        return StratifiedKFold(y, n_folds=self.n_folds, shuffle=self.shuffle,
                               random_state=self.random_state)


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
    short_name = 'fixed'

    def __init__(self, start, stop=None):
        self.valid = slice(start, stop)

    def create(self, X, y):
        indices = np.arange(len(X))
        valid = indices[self.valid]
        train = np.setdiff1d(indices, valid)
        return (train, valid),  # return a nested tuple
