from __future__ import print_function, absolute_import, division
import os
import yaml
import glob
from six.moves import cPickle
from six import iteritems

from .dataset import MDTrajDataset


FIELDS = {
    'estimator':   ['pickle'],
    'dataset':     ['trajectories', 'topology', 'stride'],

    'experiments': str,
    'param_grid':  dict,
    'cv':          int,
}


class Config(object):

    def __init__(self, path):
        self.path = path
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
                        raise RuntimeError("in section %r: unknown key %r" %
                                 (section, key))
                                 
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
    
    def param_grid(self):
        grid = self.get_section('param_grid')
        required_fields = ['min', 'max']
        
        for param_name, info in iteritems(grid):
            for f in required_fields:
                if f not in info:
                    raise RuntimeError('param_grid/%s does not contain '
                                       'required field "%s"' % (param_name, f))

            # default the "type" field to float
            type = info.get('type', float)
            if type in [float, 'float']:
                info['type'] = float
            elif type in [int, 'int']:
                info['type'] = int
            else:
                raise RuntimeError('param_grid/%s must be "int" or "float"' % param_name)
        return grid

    def cv(self):
        return int(self.get_section('cv'))

    def dataset(self):
        traj_glob = self.get_value('dataset/trajectories')
        topology = self.get_value('dataset/topology')
        stride = int(self.get_value('dataset/stride', 1))

        if traj_glob is not None:
            import mdtraj as md

            traj_glob = os.path.expanduser(traj_glob)
            if not os.path.isabs(traj_glob):
                traj_glob = os.path.join(os.path.dirname(self.path), traj_glob)
            
            filenames = glob.glob(traj_glob)
            return MDTrajDataset(filenames, topology=topology, stride=stride)
            
        # we should also support loading datasets from other forms
        # (e.g. hdf5 or pickle files containing numpy arrays)
        raise NotImplementedError()

    def experiments(self):
        pass

def parse(data):
    res = yaml.load(data)
    if res is None:
        res = {}
            
    for field, spec in iteritems(FIELDS):
        if field in res:
            if isinstance(spec, type):
                try:
                    res[field] = spec(res[field])
                except ValueError as e:
                    raise RuntimeError("key %r could not be converted to %s" % (
                        field, spec.__name__))
            elif not isinstance(res[field], dict):
                raise RuntimeError("The %s field should be a dict, not %s" % (
                    field, res[field].__class__.__name__))
            
    return res