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

    def __init__(self, path):
        self.path = path
        if not isfile(self.path):
            raise RuntimeError('%s does not exist' % self.path)
        with open(self.path) as f:
            config = parse(f)
        self.config = self._merge_defaults_and_rc(config)
        self._check_fields()

    def _merge_defaults_and_rc(self, config):
        fn = resource_filename('osprey', join('data', 'default_config.yaml'))
        with open(fn) as f:
            default = parse(f)
        rc = load_rcfile()
        return reduce(dict_merge, [default, rc, config])

    @classmethod
    def fromdict(cls, config):
        """Create a Config object from config dict directly."""
        m = super(Config, cls).__new__(cls)
        m.path = '.'
        m.config = m._merge_defaults_and_rc(config)
        m._check_fields()
        return m

    def get_section(self, section):
        return self.config.get(section, {})

    def get_value(self, field, default=None):
        section, key = field.split('/')
        return self.get_section(section).get(key, default)

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

    def estimator(self):
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
            eval_globals = self.get_value('estimator/__eval_globals__')

            try:
                estimator = eval(evalstring, {}, eval_globals())
                if not isinstance(estimator, sklearn.base.BaseEstimator):
                    raise RuntimeError('estimator/pickle must load a '
                                       'sklearn-derived Estimator')
                return estimator
            except:
                print('-'*78)
                print('Error parsing estimator/eval', file=sys.stderr)
                print('-'*78)

                traceback.print_exc(file=sys.stderr)
                print('-'*78)
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
        grid = self.get_value('search/space')
        required_fields = ['min', 'max']

        for param_name, info in iteritems(grid):
            for f in required_fields:
                if f not in info:
                    raise RuntimeError('param_grid/%s does not contain '
                                       'required field "%s"' % (param_name, f))

            if 'type' not in info:
                info['type'] = 'int'
            elif info['type'] not in ['int', 'float']:
                raise RuntimeError('param_grid/%s must be "int" '
                                   'or "float"' % param_name)
        return grid

    def search_engine(self):
        engine = self.get_value('search/engine')
        if engine not in search.__all__:
            raise RuntimeError('search/engine "%s" not supported. available'
                               'engines are: %r' % (engine, search.__all__))
        return getattr(search, engine)

    def search_seed(self):
        return self.get_value('search/seed')

    def cv(self):
        return int(self.get_section('cv'))

    def dataset(self):
        loader = load_entry_point(self.get_value('dataset/__loader__'))

        with in_directory(dirname(self.path)):
            X, y = loader(**self.get_section('dataset'))

        return X, y

    def trials(self):
        uri = self.get_value('trials/uri')
        table_name = self.get_value('trials/table_name')

        with in_directory(dirname(self.path)):
            value = make_session(uri, table_name)
        return value

    def sha1(self):
        with open(self.path) as f:
            return hashlib.sha1(f.read()).hexdigest()

    def scoring(self):
        scoring = self.get_section('scoring')
        if len(scoring) == 0:
            scoring = None
        assert isinstance(scoring, (str, types.NoneType))
        return scoring


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
