from __future__ import print_function, absolute_import, division

import glob
import os
import numpy as np
from .utils import expand_path


class BaseDatasetLoader(object):
    short_name = None

    def load(self):
        raise NotImplementedError('should be implemented in subclass')

class MSMBuilderDatasetLoader(BaseDatasetLoader):
    short_name = 'msmbuilder'
    def __init__(self, path, fmt=None, verbose=False):
        self.path = path
        self.fmt = fmt
        self.verbose = verbose

    def load(self):
        from msmbuilder.dataset import dataset
        ds = dataset(self.path, mode='r', fmt=self.fmt, verbose=self.verbose)
        print(ds.provenance)
        return ds, None

class MDTrajDatasetLoader(BaseDatasetLoader):
    short_name = 'mdtraj'

    def __init__(self, trajectories, topology=None, stride=1, verbose=False):
        self.trajectories = trajectories
        self.topology = topology
        self.stride = stride
        self.verbose = verbose

    def load(self):
        import mdtraj

        filenames = sorted(glob.glob(expand_path(self.trajectories)))

        top = self.topology
        kwargs = {}
        if top is not None:
            top = expand_path(self.topology)
            kwargs = {'top': top}

        X = []
        y = None

        for fn in filenames:
            if self.verbose:
                print('[mdtraj] loading %s' % fn)
            X.append(mdtraj.load(fn, stride=self.stride, **kwargs))

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
        filenames = sorted(glob.glob(expand_path(self.traj_glob)))
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

        filenames = sorted(glob.glob(expand_path(self.filenames)))
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


class SklearnDatasetLoader(BaseDatasetLoader):
    short_name = 'sklearn_dataset'

    def __init__(self, method, x_name='data', y_name='target', **kwargs):
        self.method = method
        self.x_name = x_name
        self.y_name = y_name
        self.kwargs = kwargs

    def load(self):
        import sklearn.datasets

        try:
            loader = getattr(sklearn.datasets, self.method)
        except AttributeError:
            raise RuntimeError('no %s in sklearn.datasets' % self.method)

        bunch = loader(**self.kwargs)

        X = bunch[self.x_name]
        y = bunch[self.y_name]

        return X, y
