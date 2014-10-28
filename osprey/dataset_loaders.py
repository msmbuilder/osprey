from __future__ import print_function, absolute_import, division

import glob
from .utils import expand_path


class BaseDatasetLoader(object):
    short_name = None


class MDTrajDatasetLoader(BaseDatasetLoader):
    short_name = 'mdtraj'

    def __init__(self, trajectories, topology=None, stride=1):
        self.trajectories = trajectories
        self.topology = topology
        self.stride = stride

    def load(self):
        import mdtraj

        filenames = glob.glob(expand_path(self.trajectories))

        top = self.topology
        if top is not None:
            top = expand_path(self.topology)

        X = [mdtraj.load(f, top=top, stride=self.stride) for f in filenames]
        y = None

        return X, y
