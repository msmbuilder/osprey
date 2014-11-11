from __future__ import print_function, absolute_import, division


class BaseCrossValidator(object):
    short_name = None

    def load(self):
        raise NotImplementedError('should be implemented in subclass')


class ShuffleSplitValidator(BaseCrossValidator):
    short_name = 'shufflesplit'

    def __init__(self, n_iter=5, test_size=0.5, train_size=None,
                 random_state=None):

        self.n_iter = n_iter
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def create(self):
        from sklearn.cross_validation import ShuffleSplit

        return lambda n: ShuffleSplit(n, n_iter=self.n_iter,
                                      test_size=self.test_size,
                                      train_size=self.train_size,
                                      random_state=self.random_state)


class KFoldValidator(BaseCrossValidator):
    short_name = 'kfold'

    def __init__(self, n_folds=5, shuffle=False, random_state=None):

        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def create(self):
        from sklearn.cross_validation import KFold

        return lambda n: KFold(n, n_folds=self.n_folds,
                               shuffle=self.shuffle,
                               random_state=self.random_state)


class LeaveOneOutValidator(BaseCrossValidator):
    short_name = 'loo'

    def __init__(self):
        pass

    def create(self):
        from sklearn.cross_validation import LeaveOneOut

        return lambda n: LeaveOneOut(n)
