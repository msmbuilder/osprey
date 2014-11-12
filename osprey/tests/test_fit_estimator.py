from __future__ import print_function, absolute_import, division

import numpy as np
from six import iteritems
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV

from osprey.fit_estimator import fit_and_score_estimator


def test_1():
    X, y = make_regression(n_features=10)

    lasso = Lasso()
    params = {'alpha': 2}
    cv = 5
    out = fit_and_score_estimator(lasso, params, cv=cv, X=X, y=y, verbose=0)

    param_grid = dict((k, [v]) for k, v in iteritems(params))
    g = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=cv)
    g.fit(X, y)

    assert out['mean_test_score'] == g.grid_scores_[0].mean_validation_score
    assert np.all(out['test_scores'] == g.grid_scores_[0].cv_validation_scores)
