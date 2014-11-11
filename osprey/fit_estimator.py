from __future__ import print_function, absolute_import, division

from distutils.version import LooseVersion
import sklearn
from sklearn.base import is_classifier, clone
from sklearn.cross_validation import _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.validation import _num_samples, check_arrays
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import _check_cv as check_cv

if LooseVersion(sklearn.__version__) < LooseVersion('0.15.0'):
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

    This function uses some undocumented internal APIs (non-public). It was
    written against sklearn version 0.15.2, and tested against version
    0.15.0b1, version 0.14 and before does not work.

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
                                fit_params=None, return_train_score=True,
                                return_parameters=True)
        for train, test in cv)

    assert len(out) == len(cv)

    grid_scores = list()
    n_test_samples = 0
    score = 0
    all_train_scores = []
    all_test_scores = []
    for train_score, test_score, this_n_test_samples, _, _ in out:
        all_train_scores.append(train_score)
        all_test_scores.append(test_score)
        if iid:
            test_score *= this_n_test_samples
            n_test_samples += this_n_test_samples
        score += test_score
    if iid:
        score /= float(n_test_samples)
    else:
        score /= len(cv)

    grid_scores = {'mean_test_score': score, 'train_scores': all_train_scores,
                   'test_scores': all_test_scores}
    return grid_scores
