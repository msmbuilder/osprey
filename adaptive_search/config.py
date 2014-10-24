from __future__ import print_function, absolute_import, division
import os
import yaml
import glob
from six.moves import cPickle
from six import iteritems

from .trials import make_session
from . import search


FIELDS = {
    'estimator':   ['pickle'],
    'dataset':     ['trajectories', 'topology', 'stride'],

    'trials': ['uri'],
    'search':  ['engine', 'space', 'seed'],
    'param_grid':  dict,
    'cv':          int,
}


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
        # load estimator from pickle field
        pkl = self.get_value('estimator/pickle')
        if pkl is not None:
            path = os.path.join(os.path.dirname(self.path), pkl)
            if not os.path.isfile(path):
                raise RuntimeError('estimator/pickle %s is not a file' % pkl)
            with open(path) as f:
                return cPickle.load(f)

        # load from any other methods we implement...
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
        traj_glob = self.get_value('dataset/trajectories')
        topology = self.get_value('dataset/topology')
        stride = int(self.get_value('dataset/stride', 1))

        if traj_glob is not None:
            import mdtraj as md

            traj_glob = expand_path(traj_glob, os.path.dirname(self.path))
            if topology is not None:
                topology = expand_path(topology, os.path.dirname(self.path))

            filenames = glob.glob(traj_glob)
            return [md.load(f, top=topology, stride=stride) for f in filenames]

        # we should also support loading datasets from other forms
        # (e.g. hdf5 or pickle files containing numpy arrays)
        raise NotImplementedError()

    def trials(self):
        uri = self.get_value('trials/uri')
        curdir = os.path.abspath(os.curdir)
        try:
            # move into that directory when creating the DB, since if it's
            # an sqlite3 DB, we want relative paths on the filesystem we want
            # those to be interpreted relative to the config file's directory,
            # not the curdir of the client instantiating this python script.
            os.chdir(os.path.dirname(self.path))
            value = make_session(uri)
        finally:
            os.chdir(curdir)

        return value


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


def expand_path(path, base):
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)
    return path
