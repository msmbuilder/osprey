from __future__ import print_function
import sys
from argparse import ArgumentDefaultsHelpFormatter
import traceback
from socket import gethostname
from getpass import getuser
from datetime import datetime

from six import iteritems
from six.moves import cStringIO

from .config import Config
from .trials import Trial


def configure_parser(sub_parsers):
    help = 'Run worker processes'
    p = sub_parsers.add_parser('worker', description=help, help=help,
                               formatter_class=ArgumentDefaultsHelpFormatter)
    p.add_argument('config', help='Path to worker config file (yaml)')
    p.add_argument('--n-iters', default=1, type=int, help='Number of '
                   'trials to run sequentially.')

    p.set_defaults(func=execute)


def execute(args, parser):
    # Load the config file and extract the fields
    config = Config(args.config)
    estimator = config.estimator()
    session = config.trials()
    cv = config.cv()
    bounds = config.search_space()
    engine = config.search_engine()
    seed = config.search_seed()
    config_sha1 = config.sha1()
    scoring = config.scoring()

    print('Loading dataset...')
    dataset = config.dataset()
    print('  %d sequences' % len(dataset))

    print('Loaded estimator:')
    print('  %r' % estimator)

    for i in range(args.n_iters):
        print('='*78)

        # requery the history ever iteration, because another worker
        # process may have written to it in the mean time
        history = [[t.parameters, t.mean_cv_score]
                   for t in session.query(Trial).all()]
        print('History contains:\n  %d trials' % len(history))

        params = engine(history, bounds, seed)

        run_single_trial(
            estimator=estimator, scoring=scoring, dataset=dataset,
            params=params, cv=cv, config_sha1=config_sha1, session=session)


def run_single_trial(estimator, scoring, dataset, params, cv, config_sha1,
                     session):
    from sklearn.base import clone, BaseEstimator
    from sklearn.grid_search import GridSearchCV

    # make sure we get _all_ the parameters, including defaults on the
    # estimator class
    params = clone(estimator).set_params(**params).get_params()
    params = dict((k, v) for k, v in iteritems(params)
                  if not isinstance(v, BaseEstimator))
    print('Running parameters:\n  %r\n' % params)

    t = Trial(status='PENDING', parameters=params, host=gethostname(),
              user=getuser(), started=datetime.now(),
              config_sha1=config_sha1)
    session.add(t)
    session.commit()

    try:
        grid = GridSearchCV(
            estimator, param_grid={k: [v] for k, v in iteritems(params)},
            scoring=scoring, cv=cv, verbose=1, refit=False)
        grid.fit(dataset)
        score = grid.grid_scores_[0]

        t.mean_cv_score = score.mean_validation_score
        t.cv_scores = score.cv_validation_scores.tolist()
        t.status = 'SUCCEEDED'
        print('Success! Score=%f' % t.mean_cv_score)
    except Exception:
        buf = cStringIO()
        traceback.print_exc(file=buf)

        t.traceback = buf.getvalue()
        t.status = 'FAILED'
        print('-'*78, file=sys.stderr)
        print('Exception encountered while fitting model')
        print('-'*78, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print('-'*78, file=sys.stderr)
    finally:
        t.completed = datetime.now()
        t.elapsed = t.completed - t.started
        session.commit()

    return t.status
