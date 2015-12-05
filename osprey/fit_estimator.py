from __future__ import print_function, absolute_import, division

import time
from distutils.version import LooseVersion

import numpy as np
import sklearn
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import check_cv, _safe_split, _score

from .utils import check_arrays
from .utils import short_format_time, is_msmbuilder_estimator


if LooseVersion(sklearn.__version__) < LooseVersion('0.16.1'):
    raise ImportError('Please upgrade to the latest version of scikit-learn')


def fit_and_score_estimator(estimator, parameters, cv, X, y=None, scoring=None,
                            iid=True, n_jobs=1, verbose=1,
                            pre_dispatch='2*n_jobs'):
    """Fit and score an estimator with cross-validation

    This function is basically a copy of sklearn's
    grid_search._BaseSearchCV._fit(), which is the core of the GridSearchCV
    fit() method. Unfortunately, that class does _not_ return the training
    set scores, which we want to save in the database, and because of the
    way it's written, you can't change it by subclassing or monkeypatching.

    This function uses some undocumented internal sklearn APIs (non-public).
    It was written against sklearn version 0.16.1. Prior Versions are likely
    to fail due to changes in the design of cross_validation module.

    Returns
    -------
    out : dict, with keys 'mean_test_score' 'test_scores', 'train_scores'
        The scores on the training and test sets, as well as the mean test set
        score.
    """
    scorer = check_scoring(estimator, scoring=scoring)
    n_samples = _num_samples(X)
    X, y = check_arrays(X, y, allow_lists=True, sparse_format='csr',
                        allow_nans=True)
    if y is not None:
        if len(y) != n_samples:
            raise ValueError('Target variable (y) has a different number '
                             'of samples (%i) than data (X: %i samples)'
                             % (len(y), n_samples))
    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))

    out = Parallel(
        n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch
    )(
        delayed(_fit_and_score)(clone(estimator), X, y, scorer,
                                train, test, verbose, parameters,
                                fit_params=None)
        for train, test in cv)

    assert len(out) == len(cv)

    train_scores, test_scores = [], []
    n_train_samples, n_test_samples = [], []
    for test_score, n_test, train_score, n_train, _ in out:
        train_scores.append(train_score)
        test_scores.append(test_score)
        n_test_samples.append(n_test)
        n_train_samples.append(n_train)

    if iid:
        if verbose > 0 and is_msmbuilder_estimator(estimator):
            print('[CV] Using MSMBuilder API n_samples averaging')
            print('[CV]   n_train_samples: %s' % str(n_train_samples))
            print('[CV]   n_test_samples: %s' % str(n_test_samples))
        mean_test_score = np.average(test_scores, weights=n_test_samples)
        mean_train_score = np.average(train_scores, weights=n_train_samples)
    else:
        mean_test_score = np.average(test_scores)
        mean_train_score = np.average(train_scores)

    grid_scores = {
        'mean_test_score': mean_test_score, 'test_scores': test_scores,
        'mean_train_score': mean_train_score, 'train_scores': train_scores,
        'n_test_samples': n_test_samples, 'n_train_samples': n_train_samples}
    return grid_scores


def _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters,
                   fit_params=None):
    if verbose > 1:
        if parameters is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    if len(train) == 0 or len(test) == 0:
        raise RuntimeError(
            'Cross validation error in fit_estimator. The total data set '
            'contains %d elements, which were split into a training set '
            'of %d elements and a test set of %d elements. Unfortunately, '
            'you can\'t have a %s set with 0 elements.' % (
                len(X), len(train), len(test),
                'training' if len(train) == 0 else 'test'))

    # adjust length of sample weights
    n_samples = _num_samples(X)
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, np.asarray(v)[train]
                       if hasattr(v, '__len__') and len(v) == n_samples else v)
                       for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    # fit and score
    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    test_score = _score(estimator, X_test, y_test, scorer)
    train_score = _score(estimator, X_train, y_train, scorer)

    scoring_time = time.time() - start_time

    msmbuilder_api = is_msmbuilder_estimator(estimator)
    n_samples_test = _num_samples(X_test, msmbuilder_api=msmbuilder_api)
    n_samples_train = _num_samples(X_train, msmbuilder_api=msmbuilder_api)
    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg, short_format_time(scoring_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    return (test_score, n_samples_test, train_score, n_samples_train,
            scoring_time)


def _num_samples(x, msmbuilder_api=False):
    """Return number of samples in array-like x."""
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %r" % x)

    if msmbuilder_api:
        assert isinstance(x, list)
        return sum(len(xx) for xx in x)

    return x.shape[0] if hasattr(x, 'shape') else len(x)
