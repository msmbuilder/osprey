"""config.py

This module contains the parser and in-memory representation of the config
osprey job file file. The config file has four major sections:

 - estimator: the specification for the estimator/model to be fit, an instance
              of sklearn.base.BaseEstimator.
 - search:    the specification of the hyperparameter search space and the
              strategy for adaptive exploration of this space.
 - dataset:   the specification of the dataset to fit the models with.
 - trials:    as each hyperparameter setting is explored, the results are
              serialized to a database specified in this section.
 - cv:        specification for cross-validation.
 - scoring:   the score function used in cross-validation. (optional)

Notes
-----
If the eval option is used to load the estimator, a special variable in the
section __eval_globals__ is read. This variable specifies an entry point for a
function that returns a dict of pre-imported variables which can be referenced
inside the eval() context.

Dataset __loader__
"""

from __future__ import print_function, absolute_import, division
import sys
import yaml
import types
import hashlib
import traceback
from os.path import join, isfile, dirname

from six.moves import cPickle
from six import iteritems
from six.moves import reduce
from pkg_resources import resource_filename

from .trials import make_session
from .entry_point import load_entry_point
from .rcfile import load_rcfile
from .utils import dict_merge, in_directory
from .searchspace import SearchSpace
from . import search


FIELDS = {
    'estimator':   ['pickle', 'eval', '__eval_globals__', 'entry_point'],
    'dataset':     dict,

    'trials':      ['uri', 'table_name'],
    'search':      ['engine', 'space', 'seed'],
    'param_grid':  dict,
    'cv':          int,
    'scoring':     str,
}


class Config(object):

    def __init__(self, path, verbose=True):
        self.path = path
        self.verbose = verbose
        if not isfile(self.path):
            raise RuntimeError('%s does not exist' % self.path)
        with open(self.path) as f:
            config = parse(f)
        self.config = self._merge_defaults_and_rc(config)
        self._check_fields()

        if self.verbose:
            print('Loading config file from %s...' % path)

    def _merge_defaults_and_rc(self, config):
        """The config object loads its values from three sources, with the
        following precedence:

            1. data/default_config.yaml
            2. the .ospreyrc file, which can be located in the user's home
               directory, the current directory, or specified with the OSPREYRC
               variable (see rcfile.py)
            3. The config file itself, passed in to this object in the
               constructor as `path`.

        Values specified in files with higher precedence (later in the above
        list) always supersede values from lower in the list, in the case
        of a conflict.
        """
        fn = resource_filename('osprey', join('data', 'default_config.yaml'))
        with open(fn) as f:
            default = parse(f)
        rc = load_rcfile(self.verbose)
        return reduce(dict_merge, [default, rc, config])

    def _check_fields(self):
        for section, submeta in iteritems(self.config):
            if section not in FIELDS:
                raise RuntimeError("unknown section: %s" % section)
            if isinstance(FIELDS[section], type):
                if not isinstance(submeta, FIELDS[section]):
                    raise RuntimeError("in section %d" % section)
            else:
                for key in submeta:
                    if key not in FIELDS[section]:
                        raise RuntimeError("in section %r: unknown key %r" % (
                            section, key))

    @classmethod
    def fromdict(cls, config):
        """Create a Config object from config dict directly."""
        m = super(Config, cls).__new__(cls)
        m.path = '.'
        m.verbose = False
        m.config = m._merge_defaults_and_rc(config)
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
        import sklearn.base

        # load estimator from pickle field
        pkl = self.get_value('estimator/pickle')
        if pkl is not None:
            path = join(dirname(self.path), pkl)
            if not isfile(path):
                raise RuntimeError('estimator/pickle %s is not a file' % pkl)
            with open(path) as f:
                estimator = cPickle.load(f)
                if not isinstance(estimator, sklearn.base.BaseEstimator):
                    raise RuntimeError('estimator/pickle must load a '
                                       'sklearn-derived Estimator')
                return estimator

        evalstring = self.get_value('estimator/eval')
        if evalstring is not None:
            eval_globals_str = self.get_value('estimator/__eval_globals__')
            try:
                eval_globals = load_entry_point(eval_globals_str)()
            except:
                print('ERROR! Failed to load estimator/__eval_globals__',
                      file=sys.stderr)
                raise
            if not isinstance(eval_globals, dict):
                raise RuntimeError('__eval__globals__ must resolve to a '
                                   'callable that returns a dict')

            try:
                estimator = eval(evalstring, {}, eval_globals)
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
                estimator = estimator()
            if not isinstance(estimator, sklearn.base.BaseEstimator):
                raise RuntimeError('estimator/pickle must load a '
                                   'sklearn-derived Estimator')
            return estimator

        raise RuntimeError('no estimator field')

    def search_space(self):
        ss = self.get_value('search/space')

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

    def search_engine(self):
        engine = self.get_value('search/engine')
        if engine not in search.__all__:
            raise RuntimeError('search/engine "%s" not supported. available'
                               'engines are: %r' % (engine, search.__all__))
        return getattr(search, engine)

    def dataset(self):
        loader = load_entry_point(self.get_value('dataset/__loader__'))

        with in_directory(dirname(self.path)):
            X, y = loader(**self.get_section('dataset'))

        return X, y

    def trials(self):
        uri = self.get_value('trials/uri')
        table_name = self.get_value('trials/table_name')
        if self.verbose:
            print('Loading trials database from %s (table = "%s")...' % (
                  uri, table_name))

        with in_directory(dirname(self.path)):
            value = make_session(uri, table_name)
        return value

    def scoring(self):
        scoring = self.get_section('scoring')
        if scoring == {}:
            scoring = None
        assert isinstance(scoring, (str, types.NoneType))
        return scoring

    def search_seed(self):
        return self.get_value('search/seed')

    def cv(self):
        return int(self.get_section('cv'))

    def sha1(self):
        """SHA1 hash of the config file itself."""
        with open(self.path) as f:
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
            elif not isinstance(res[field], dict):
                raise RuntimeError("The %s field should be a dict, not %s" % (
                    field, res[field].__class__.__name__))

    return res
