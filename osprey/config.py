from __future__ import print_function, absolute_import, division
import os
import sys
import yaml
import types
import hashlib
import traceback

from six.moves import cPickle
from six import iteritems

from .trials import make_session
from .entry_point import load_entry_point
from .rcfile import RC
from . import search
from . import mixtape_evalsupport


FIELDS = {
    'estimator':   ['pickle', 'eval', 'entry_point'],
    'dataset':     dict,

    'trials': ['uri', 'table_name'],
    'search':  ['engine', 'space', 'seed'],
    'param_grid':  dict,
    'cv':          int,
    'scoring':      str,
}

DEFAULT_DATASET_LOADER = RC.get('default_dataset_loader',
                                'osprey.MDTrajDataset')
DEFAULT_ESTIMATOR_EVAL_GLOBALS = mixtape_evalsupport.globals


class Config(object):

    def __init__(self, path):
        self.path = path
        if not os.path.isfile(self.path):
            raise RuntimeError('%s does not exist' % self.path)
        with open(self.path) as f:
            self.config = parse(f)

    @classmethod
    def fromdict(cls, config):
        """
        Create a Config object from config dict directly.
        """
        m = super(Config, cls).__new__(cls)
        m.path = '.'
        m.config = config
        return m

    def get_section(self, section):
        return self.config.get(section, {})

    def get_value(self, field, default=None):
        section, key = field.split('/')
        return self.get_section(section).get(key, default)

    def check_fields(self):
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
            path = os.path.join(os.path.dirname(self.path), pkl)
            if not os.path.isfile(path):
                raise RuntimeError('estimator/pickle %s is not a file' % pkl)
            with open(path) as f:
                estimator = cPickle.load(f)
                if not isinstance(estimator, sklearn.base.BaseEstimator):
                    raise RuntimeError('estimator/pickle must load a '
                                       'sklearn-derived Estimator')
                return estimator

        evalstring = self.get_value('estimator/eval')
        if evalstring is not None:
            try:
                estimator = eval(evalstring, {},
                                 DEFAULT_ESTIMATOR_EVAL_GLOBALS())
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
        loader_str = self.get_value('dataset/__loader__',
                                    DEFAULT_DATASET_LOADER)
        loader_factory = load_entry_point(loader_str, 'dataset/__loader__')
        loader = loader_factory(**self.get_section('dataset'))

        curdir = os.path.abspath(os.curdir)
        try:
            os.chdir(os.path.dirname(self.path))
            X, y = loader()
        finally:
            os.chdir(curdir)

        return X, y

    def trials(self):
        uri = self.get_value('trials/uri')
        table_name = self.get_value('trials/table_name', default='trials')

        curdir = os.path.abspath(os.curdir)
        try:
            # move into that directory when creating the DB, since if it's
            # an sqlite3 DB, we want relative paths on the filesystem we want
            # those to be interpreted relative to the config file's directory,
            # not the curdir of the client instantiating this python script.
            os.chdir(os.path.dirname(self.path))
            value = make_session(uri, table_name)
        finally:
            os.chdir(curdir)

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


def parse(data):
    res = yaml.load(data)
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
