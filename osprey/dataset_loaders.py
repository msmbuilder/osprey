from __future__ import print_function, absolute_import, division

import glob
import os
import numpy as np
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


class JoblibDatasetLoader(BaseDatasetLoader):
    short_name = 'joblib'

    def __init__(self, filenames, x_name=None, y_name=None,
                 system_joblib=False):
        self.filenames = filenames
        self.x_name = x_name
        self.y_name = y_name
        self.system_joblib = system_joblib

    def load(self):
        if self.system_joblib:
            import joblib
        else:
            from sklearn.externals import joblib

        X, y = [], []

        filenames = glob.iglob(expand_path(self.filenames))
        for fn in filenames:
            obj = joblib.load(fn)
            if isinstance(obj, (list, np.ndarray)):
                X.append(obj)
            else:
                X.append(obj[self.x_name])
                y.append(obj[self.y_name])

        if len(X) == 1:
            X = X[0]
        if len(y) == 1:
            y = y[0]
        elif len(y) == 0:
            y = None

        return X, y
