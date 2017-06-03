from __future__ import print_function, absolute_import, division

import os
import sys
import time
import signal
import traceback
from socket import gethostname
from getpass import getuser
from datetime import datetime

from six import iteritems
from six.moves import cStringIO
from sqlalchemy import func
from sklearn.base import clone, BaseEstimator
import numpy as np

from . import __version__
from .config import Config
from .trials import Trial
from .fit_estimator import fit_and_score_estimator
from .utils import Unbuffered, format_timedelta, current_pretty_time
from .utils import is_msmbuilder_estimator, num_samples
from .utils import is_json_serializable


class MaxParamSuggestionRetriesExceeded(Exception):
    pass


def execute(args, parser):
    start_time = datetime.now()
    sys.stdout = Unbuffered(sys.stdout)
    # Load the config file and extract the fields
    print_header()

    config = Config(args.config)
    random_seed = args.seed if args.seed is not None else config.random_seed()
    max_param_suggestion_retries = config.max_param_suggestion_retries()
    estimator = config.estimator()
    if 'random_state' in estimator.get_params().keys():
        estimator.set_params(random_state=random_seed)
    np.random.seed(random_seed)
    searchspace = config.search_space()
    strategy = config.strategy()
    config_sha1 = config.sha1()
    scoring = config.scoring()

    project_name = config.project_name()

    if is_msmbuilder_estimator(estimator):
        print_msmbuilder_version()

    print('\nLoading dataset...\n')
    X, y = config.dataset()
    print('Dataset contains %d element(s) with %s labels'
          % (num_samples(X), 'out' if y is None else ''))
    print('The elements have shape: [%s' %
          ', '.join([str(X[i].shape)
                     if isinstance(X[i], (np.ndarray, np.generic))
                     else '(%s,)' % num_samples(X[i])
                     for i in range(min(num_samples(X), 20))]), end='')
    print(', ...]' if (num_samples(X) > 20) else ']')
    print('Instantiated estimator:')
    print('  %r' % estimator)
    print(searchspace)

    # set up cross-validation
    cv = config.cv(X, y)

    statuses = [None for _ in range(args.n_iters)]

    # install a signal handler to print the footer before exiting
    # from sigterm (e.g. PBS job kill)
    def signal_hander(signum, frame):
        print_footer(statuses, start_time, signum)
        sys.exit(1)
    signal.signal(signal.SIGTERM, signal_hander)

    for i in range(args.n_iters):
        print('\n' + '-'*70)
        print('Beginning iteration %50s' % ('%d / %d' % (i+1, args.n_iters)))
        print('-'*70)

        try:
            trial_id, params = initialize_trial(
                strategy, searchspace, estimator, config_sha1=config_sha1,
                project_name=project_name, sessionbuilder=config.trialscontext,
                max_param_suggestion_retries=max_param_suggestion_retries)
        except MaxParamSuggestionRetriesExceeded:
            print('The search strategy failed to suggest a new set of params not already present in the database after {} attempts'.format(max_param_suggestion_retries))
            break

        s = run_single_trial(
            estimator=estimator, params=params, trial_id=trial_id,
            scoring=scoring, X=X, y=y, cv=cv,
            sessionbuilder=config.trialscontext)

        statuses[i] = s

    print_footer(statuses, start_time)


def initialize_trial(strategy, searchspace, estimator, config_sha1,
                     project_name, sessionbuilder, max_param_suggestion_retries):

    def build_full_params(xparams):
        # make sure we get _all_ the parameters, including defaults on the
        # estimator class, to save in the database
        params = clone(estimator).set_params(**xparams).get_params()
        params = dict((k, v) for k, v in iteritems(params)
                      if is_json_serializable(v) and
                      (k != 'steps'))

        return params

    with sessionbuilder() as session:
        # requery the history ever iteration, because another worker
        # process may have written to it in the mean time
        history = [[t.parameters, t.test_scores, t.status]
                   for t in session.query(Trial).all()
                   if t.project_name == project_name]

        print('History contains: %d trials' % len(history))
        print('Choosing next hyperparameters with %s...' % strategy.short_name)
        start = time.time()

        if max_param_suggestion_retries is None:
            params = strategy.suggest(history, searchspace)
            full_params = build_full_params(params)
        else:
            for num_retries in range(max_param_suggestion_retries):
                params = strategy.suggest(history, searchspace)
                full_params = build_full_params(params)
                if not strategy.is_repeated_suggestion(full_params, history):
                    break
            else:
                raise MaxParamSuggestionRetriesExceeded

        print('  %r' % params)
        print('(%s took %.3f s)\n' % (strategy.short_name,
                                      time.time() - start))
        assert len(params) == searchspace.n_dims

        t = Trial(status='PENDING', parameters=full_params, host=gethostname(),
                  user=getuser(), started=datetime.now(),
                  config_sha1=config_sha1)
        session.add(t)
        session.commit()
        trial_id = t.id

    return trial_id, params


def run_single_trial(estimator, params, trial_id, scoring, X, y, cv,
                     sessionbuilder):

    status = None

    try:
        score = fit_and_score_estimator(
            estimator, params, cv=cv, scoring=scoring, X=X, y=y, verbose=1)
        with sessionbuilder() as session:
            trial = session.query(Trial).get(trial_id)
            trial.mean_test_score = score['mean_test_score']
            trial.mean_train_score = score['mean_train_score']
            trial.test_scores = score['test_scores']
            trial.train_scores = score['train_scores']
            trial.n_test_samples = score['n_test_samples']
            trial.n_train_samples = score['n_train_samples']

            trial.status = 'SUCCEEDED'
            best_so_far = session.query(
                func.max(Trial.mean_test_score)).first()
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Success! Model score = %f' % trial.mean_test_score)
            print('(best score so far   = %f)' %
                  max(trial.mean_test_score, best_so_far[0]))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            trial.completed = datetime.now()
            trial.elapsed = trial.completed - trial.started
            session.commit()
            status = trial.status

    except Exception:
        buf = cStringIO()
        traceback.print_exc(file=buf)

        with sessionbuilder() as session:
            trial = session.query(Trial).get(trial_id)
            trial.traceback = buf.getvalue()
            trial.status = 'FAILED'
            print('-'*78, file=sys.stderr)
            print('Exception encountered while fitting model')
            print('-'*78, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print('-'*78, file=sys.stderr)
            session.commit()
            status = trial.status

    except (KeyboardInterrupt, SystemExit):
        with sessionbuilder() as session:
            trial = session.query(Trial).get(trial_id)
            trial.status = 'FAILED'
            session.commit()
            sys.exit(1)

    return status


def print_header():
    print('='*70)
    print('= osprey is a tool for machine learning '
          'hyperparameter optimization. =')
    print('='*70)
    print()
    print('osprey version:      %s' % __version__)
    print('time:                %s' % current_pretty_time())
    print('hostname:            %s' % gethostname())
    print('cwd:                 %s' % os.path.abspath(os.curdir))
    print('pid:                 %s' % os.getpid())
    print()


def print_msmbuilder_version():
    from msmbuilder.version import full_version as msmb_version
    from mdtraj.version import full_version as mdtraj_version
    print()
    print('msmbuilder version:  %s' % msmb_version)
    print('mdtraj version:      %s' % mdtraj_version)
    print()


def print_footer(statuses, start_time, signum=None):
    n_successes = sum(s == 'SUCCEEDED' for s in statuses)
    elapsed = format_timedelta(datetime.now() - start_time)
    print()

    if signum is not None:
        sigmap = dict((k, v) for v, k in iteritems(signal.__dict__)
                      if v.startswith('SIG'))
        signame = sigmap.get(signum, 'Unknown')
        print('== osprey worker received signal %s!' % signame,
              file=sys.stderr)
        print('== exiting immediately.', file=sys.stderr)

    print('%d/%d models fit successfully.' % (n_successes, len(statuses)))
    print('time:         %s' % current_pretty_time())
    print('elapsed:      %s.' % elapsed)
    print('osprey worker exiting.')
