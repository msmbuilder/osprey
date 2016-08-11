from __future__ import print_function, absolute_import, division

import glob
import os
import numpy as np

from .utils import expand_path, num_samples


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
        print('Dataset provenance:\n')
        print(ds.provenance)
        return ds, None


class NumpyDatasetLoader(BaseDatasetLoader):
    short_name = 'numpy'

    def __init__(self, filenames):
        self.filenames = filenames

    def load(self):
        filenames = sorted(glob.glob(expand_path(self.filenames)))
        if len(filenames) == 0:
            raise RuntimeError('no filenames matched by pattern: %s' %
                               self.filenames)
        ds = [np.load(f) for f in filenames]
        return ds, None


class HDF5DatasetLoader(BaseDatasetLoader):
    short_name = 'hdf5'

    def __init__(self, filenames, y_col=None, stride=1, concat=False):
        self.filenames = filenames
        self.y_col = y_col
        self.stride = stride
        self.concat = concat

    def transform(self, X):
        n_rows = X.shape[0]
        X = np.atleast_2d(X)
        if X.shape[0] != n_rows:
            X = X.T
        if self.y_col is not None:
            cols = range(X.shape[1])
            x_idx = [i for i, val in enumerate(cols) if val != self.y_col]
            y_idx = [i for i, val in enumerate(cols) if val == self.y_col]
            return X[::self.stride, x_idx], X[::self.stride, y_idx].ravel()
        return X[::self.stride, :], None

    def loader(self, fn):
        from mdtraj import io
        dataset = io.loadh(fn)
        for key in dataset.iterkeys():
            yield dataset[key]

    def load(self):
        X = []
        y = []
        filenames = sorted(glob.glob(expand_path(self.filenames)))
        for fn in filenames:
            for data in self.loader(fn):
                data = self.transform(data)
                X.append(data[0])
                y.append(data[1])

        if self.concat:
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)
        if num_samples(X) == 1:
            X = X[0]
            y = y[0]
        if self.y_col is not None:
            return X, y

        return X, None


class DSVDatasetLoader(BaseDatasetLoader):
    short_name = 'dsv'

    def __init__(self, filenames, y_col=None, delimiter=',', skip_header=0,
                 skip_footer=0, filling_values=np.nan, usecols=None, stride=1,
                 concat=False):
        self.filenames = filenames
        self.y_col = y_col
        self.delimiter = delimiter
        self.skip_header = skip_header
        self.skip_footer = skip_footer
        self.filling_values = filling_values
        if usecols and isinstance(usecols, str):
            usecols = list(map(int, usecols.strip().split(',')))
        elif usecols and isinstance(usecols, (tuple, set)):
            usecols = sorted(list(usecols))
        if usecols and y_col:
            if y_col not in usecols:
                usecols.append(y_col)
        self.usecols = usecols
        self.stride = stride
        self.concat = concat

    def transform(self, X):
        n_rows = X.shape[0]
        X = np.atleast_2d(X)
        if X.shape[0] != n_rows:
            X = X.T
        if self.y_col is not None:
            cols = list(range(X.shape[1]))
            if self.usecols:
                cols = self.usecols
            x_idx = [i for i, val in enumerate(cols) if val != self.y_col]
            y_idx = [i for i, val in enumerate(cols) if val == self.y_col]
            return X[::self.stride, x_idx], X[::self.stride, y_idx].ravel()
        return X[::self.stride, :], None

    def loader(self, fn):
        return np.genfromtxt(fn, delimiter=self.delimiter,
                             skip_header=self.skip_header,
                             skip_footer=self.skip_footer,
                             filling_values=self.filling_values,
                             usecols=self.usecols)

    def load(self):
        X = []
        y = []
        filenames = sorted(glob.glob(expand_path(self.filenames)))
        for fn in filenames:
            data = self.transform(self.loader(fn))
            X.append(data[0])
            y.append(data[1])

        if self.concat:
            X = np.concatenate(X, axis=0)
            y = np.concatenate(y, axis=0)
        if num_samples(X) == 1:
            X = X[0]
            y = y[0]
        if self.y_col is not None:
            return X, y

        return X, None


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
        if len(filenames) == 0:
            raise RuntimeError('no filenames matched by pattern: %s' %
                               self.trajectories)

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
        if len(filenames) == 0:
            raise RuntimeError('no filenames matched by pattern: %s' %
                               self.traj_glob)

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
        if len(filenames) == 0:
            raise RuntimeError('no filenames matched by pattern: %s' %
                               self.filenames)

        for fn in filenames:
            obj = joblib.load(fn)
            if isinstance(obj, (list, np.ndarray)):
                X.append(obj)
            else:
                X.append(obj[self.x_name])
                y.append(obj[self.y_name])

        if num_samples(X) == 1:
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
