from __future__ import print_function, absolute_import, division
"""config.py

This module contains the parser and in-memory representation of the config
osprey job file file. The config file has four major sections:

 - estimator:      the specification for the estimator/model to be fit, an
                   instance of sklearn.base.BaseEstimator.
 - search_space:   the specification of the hyperparameter search space
 - strategy:       strategy for adaptive exploration of hyperparameters.
 - dataset_lodaer: the specification of the dataset to fit the models with.
 - trials:         as each hyperparameter setting is explored, the results are
                   serialized to a database specified in this section.
 - cv:             specification for cross-validation.
 - scoring:        the score function used in cross-validation. (optional)
"""

import sys
import six
import hashlib
import traceback
import importlib
import contextlib
from os.path import join, isfile, dirname, abspath

import yaml
import sklearn.base
from six.moves import cPickle
from six import iteritems
from six.moves import reduce
from pkg_resources import resource_filename

from .entry_point import load_entry_point
from .utils import dict_merge, in_directory, prepend_syspath
from .search_space import SearchSpace
from .strategies import BaseStrategy
from .dataset_loaders import BaseDatasetLoader
from .cross_validators import BaseCVFactory
from .trials import make_session
from .subclass_factory import init_subclass_by_name
from . import eval_scopes


FIELDS = {
    'estimator':       ['pickle', 'eval', 'eval_scope', 'entry_point',
                        'params'],
    'dataset_loader':  ['name', 'params'],
    'trials':          ['uri', 'project_name'],
    'search_space':    dict,
    'strategy':        ['name', 'params'],
    'cv':              (int, dict),
    'scoring':         (str, type(None)),
}


class Config(object):

    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose
        if not isfile(self.path):
            raise RuntimeError('%s does not exist' % self.path)
        with open(self.path, 'rb') as f:
            config = parse(f)
        self.config = self._merge_defaults(config)
        self._check_fields()

        if self.verbose:
            print('Loading config file:     %s...' % path)

    def _merge_defaults(self, config):
        """The config object loads its values from two sources, with the
        following precedence:

            1. data/default_config.yaml
            2. The config file itself, passed in to this object in the
               constructor as `path`.

        in case of conflict, the config file dominates.
        """
        fn = resource_filename('osprey', join('data', 'default_config.yaml'))
        with open(fn) as f:
            default = parse(f)
        return reduce(dict_merge, [default, config])

    def _check_fields(self):
        for section, submeta in iteritems(self.config):
            if section not in FIELDS:
                raise RuntimeError("unknown section: %s" % section)
            if isinstance(FIELDS[section], type):
                if not isinstance(submeta, FIELDS[section]):
                    raise RuntimeError("%s should be a %s, but is a %s." % (
                        section, FIELDS[section].__name__,
                        type(submeta).__name__))
            elif isinstance(FIELDS[section], tuple):
                if not any(isinstance(submeta, t) for t in FIELDS[section]):
                    raise RuntimeError(
                        "The %s field should be one of %s, not %s" % (
                            section, [e.__name__ for e in FIELDS[section]],
                            type(submeta).__name__))
            else:
                for key in submeta:
                    if key not in FIELDS[section]:
                        raise RuntimeError("in section %r: unknown key %r" % (
                            section, key))

        missing_fields = set(FIELDS.keys()).difference(self.config.keys())
        if len(missing_fields) > 0:
            raise RuntimeError('The following required fields are missing from'
                               'the config file (%s): %s' % (
                                   ', '.join(missing_fields), self.path))

    @classmethod
    def fromdict(cls, config, check_fields=True):
        """Create a Config object from config dict directly."""
        m = super(Config, cls).__new__(cls)
        m.path = '.'
        m.verbose = False
        m.config = m._merge_defaults(config)
        if check_fields:
            m._check_fields()
        return m

    def get_section(self, section):
        """Sections are top-level entries in the config tree"""
        return self.config.get(section, {})

    def get_value(self, field, default=None):
        """Get an entry from within a section, using a '/' delimiter"""
        section, key = field.split('/')
        return self.get_section(section).get(key, default)

    # ----------------------------------------------------------------------- #

    def estimator(self):
        """Get the estimator, an instance of a (subclass of)
        sklearn.base.BaseEstimator

        It can be loaded either from a pickle, from a string using eval(),
        or from an entry point.

        e.g.

        estimator:
            # only one of the following can actually be active in a given
            # config file.
            pickle: path-to-pickle-file.pkl
            eval: "Pipeline([('cluster': KMeans())])"
            entry_point: sklearn.linear_model.LogisticRegression
        """
        evalstring = self.get_value('estimator/eval')
        if evalstring is not None:
            got = self.get_value('estimator/eval_scope')
            if isinstance(got, six.string_types):
                got = [got]
            elif isinstance(got, list):
                pass
            else:
                raise RuntimeError('unexpected type for estimator/eval_scope')

            scope = {}
            for pkg_name in got:
                if pkg_name in eval_scopes.__all__:
                    scope.update(getattr(eval_scopes, pkg_name)())
                else:
                    try:
                        pkg = importlib.import_module(pkg_name)
                    except ImportError as e:
                        raise RuntimeError(str(e))
                    scope.update(eval_scopes.import_all_estimators(pkg))

            try:
                estimator = eval(evalstring, {}, scope)
                if not isinstance(estimator, sklearn.base.BaseEstimator):
                    raise RuntimeError('estimator/pickle must load a '
                                       'sklearn-derived Estimator')
                return estimator
            except:
                print('-'*78, file=sys.stderr)
                print('Error parsing estimator/eval', file=sys.stderr)
                print('-'*78, file=sys.stderr)

                traceback.print_exc(file=sys.stderr)
                print('-'*78, file=sys.stderr)
                sys.exit(1)

        entry_point = self.get_value('estimator/entry_point')
        if entry_point is not None:
            estimator = load_entry_point(entry_point, 'estimator/entry_point')
            if issubclass(estimator, sklearn.base.BaseEstimator):
                estimator = estimator(
                    **self.get_value('estimator/params', default={}))
            if not isinstance(estimator, sklearn.base.BaseEstimator):
                raise RuntimeError('estimator/pickle must load a '
                                   'sklearn-derived Estimator')
            return estimator

        # load estimator from pickle field
        pkl = self.get_value('estimator/pickle')
        if pkl is not None:
            pickl_dir = dirname(abspath(self.path))
            path = join(pickl_dir, pkl)
            if not isfile(path):
                raise RuntimeError('estimator/pickle %s is not a file' % pkl)
            with open(path, 'rb') as f:
                with prepend_syspath(pickl_dir):
                    estimator = cPickle.load(f)
                if not isinstance(estimator, sklearn.base.BaseEstimator):
                    raise RuntimeError('estimator/pickle must load a '
                                       'sklearn-derived Estimator')
                return estimator

        raise RuntimeError('no estimator field')

    def search_space(self):
        ss = self.get_section('search_space')

        searchspace = SearchSpace()
        for param_name, info in iteritems(ss):
            if 'type' not in info:
                raise RuntimeError('search/space/%s does not contain '
                                   'required field "type"' % (param_name))
            type = info.pop('type')
            if type not in ('int', 'float', 'enum'):
                raise RuntimeError('search/space/%s type="%s" is not valid. '
                                   'valid types are int, float and enum' %
                                   param_name)
            try:
                if type == 'int':
                    if sorted(list(info.keys())) != ['max', 'min']:
                        raise RuntimeError(
                            'search/space/%s type="int" must contain keys '
                            '"min", "max"' % param_name)
                    searchspace.add_int(param_name, **info)
                elif type == 'enum':
                    if sorted(list(info.keys())) != ['choices']:
                        raise RuntimeError(
                            'search/space/%s type="enum" must contain key '
                            '"choices", "type"' % param_name)
                    searchspace.add_enum(param_name, **info)
                elif type == 'float':
                    if sorted(list(info.keys())) not in (
                            ['max', 'min'], ['max', 'min', 'warp']):
                        raise RuntimeError(
                            'search/space/%s type="float" must contain keys '
                            '"min", "max", and optionally "warp"' % param_name)
                    searchspace.add_float(param_name, **info)

            except ValueError as e:
                # searchspace.add_XXX can throw a ValueError on malformed
                # input (e.g. max < min). re-raising as a runtimerror lets the
                # CLI layer catch it appropriately without an "unexpected
                # error" traceback
                raise RuntimeError(e.message)

        return searchspace

    def strategy(self):
        strategy_name = self.get_value('strategy/name')
        strategy_params = self.get_value('strategy/params', default={})

        strat = init_subclass_by_name(BaseStrategy, strategy_name,
                                      strategy_params)
        return strat

    def dataset(self):
        loader_name = self.get_value('dataset_loader/name')
        loader_params = self.get_value('dataset_loader/params', default={})

        loader = init_subclass_by_name(
            BaseDatasetLoader, loader_name, loader_params)

        with in_directory(dirname(abspath(self.path))):
            X, y = loader.load()

        return X, y

    def trials(self):
        uri = self.get_value('trials/uri')
        project_name = self.get_value('trials/project_name')
        if self.verbose:
            print('Loading trials database: %s...' % uri)

        with in_directory(dirname(abspath(self.path))):
            value = make_session(uri, project_name=project_name)
        return value

    @contextlib.contextmanager
    def trialscontext(self):
        session = self.trials()
        yield session
        session.close()

    def scoring(self):
        scoring = self.get_section('scoring')
        assert isinstance(scoring, (str, type(None)))
        return scoring

    def cv(self, X, y=None):
        cv = self.get_section('cv')
        if isinstance(cv, int):
            cv_name = 'kfold'
            cv_params = {'n_folds': cv}
        else:
            cv_name = self.get_value('cv/name')
            cv_params = self.get_value('cv/params', default={})
        return init_subclass_by_name(
            BaseCVFactory, cv_name, cv_params).create(X, y)

    def sha1(self):
        """SHA1 hash of the config file itself."""
        with open(self.path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()


def parse(f):
    res = yaml.load(f)
    if res is None:
        res = {}

    for field, spec in iteritems(FIELDS):
        if field in res:
            if isinstance(spec, type):
                try:
                    res[field] = spec(res[field])
                except ValueError:
                    raise RuntimeError("key %r couldn't be converted to %s" % (
                        field, spec.__name__))
            elif isinstance(spec, tuple):
                if not any(isinstance(res[field], item) for item in spec):
                    raise RuntimeError(
                        "The %s field should be one of %s, not %s" % (
                            field, [e.__name__ for e in spec],
                            res[field].__class__.__name__))

            elif not isinstance(res[field], dict):
                raise RuntimeError("The %s field should be a dict, not %s" % (
                    field, res[field].__class__.__name__))

    return res
