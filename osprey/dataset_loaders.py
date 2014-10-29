from __future__ import print_function, absolute_import, division

import glob
import os
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


class FilenameDatasetLoader(BaseDatasetLoader):
    """Just pass a bunch of filenames to the first step of the pipeline

    The pipeline will do the loading.
    """
    short_name = 'filename'

    def __init__(self, trajectories, abs_path=True):
        self.traj_glob = trajectories
        self.abs_path = abs_path

    def load(self):
        filenames = glob.glob(expand_path(self.traj_glob))
        if self.abs_path:
            filenames = [os.path.abspath(fn) for fn in filenames]
        return filenames, None
