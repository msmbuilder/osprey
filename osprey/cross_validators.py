from __future__ import print_function, absolute_import, division


class BaseCrossValidator(object):
    short_name = None
    stratified = False

    def load(self):
        raise NotImplementedError('should be implemented in subclass')

    def __call__(self, *args, **kwargs):
        return self.create(*args, **kwargs)

    def create(self, *args, **kwargs):
        raise NotImplementedError('should be implemented in subclass')


class ShuffleSplitValidator(BaseCrossValidator):
    short_name = 'shufflesplit'

    def __init__(self, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n_iter = n_iter
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def create(self, n):
        from sklearn.cross_validation import ShuffleSplit

        return ShuffleSplit(n, n_iter=self.n_iter, test_size=self.test_size,
                            train_size=self.train_size,
                            random_state=self.random_state)


class KFoldValidator(BaseCrossValidator):
    short_name = 'kfold'

    def __init__(self, n_folds=3, shuffle=False, random_state=None):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def create(self, n):
        from sklearn.cross_validation import KFold

        return KFold(n, n_folds=self.n_folds, shuffle=self.shuffle,
                     random_state=self.random_state)


class LeaveOneOutValidator(BaseCrossValidator):
    short_name = 'loo'

    def __init__(self):
        pass

    def create(self, n):
        from sklearn.cross_validation import LeaveOneOut

        return LeaveOneOut(n)


class StratifiedShuffleSplitValidator(BaseCrossValidator):
    short_name = 'stratifiedshufflesplit'
    stratified = True

    def __init__(self, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):
        self.n_iter = n_iter
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state

    def create(self, y):
        from sklearn.cross_validation import StratifiedShuffleSplit

        return StratifiedShuffleSplit(y, n_iter=self.n_iter,
                                      test_size=self.test_size,
                                      train_size=self.train_size,
                                      random_state=self.random_state)


class StratifiedKFoldValidator(BaseCrossValidator):
    short_name = 'stratifiedkfold'
    stratified = True

    def __init__(self, n_folds=3, shuffle=False, random_state=None):
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

    def create(self, y):
        from sklearn.cross_validation import StratifiedKFold

        return StratifiedKFold(y, n_folds=self.n_folds, shuffle=self.shuffle,
                               random_state=self.random_state)
