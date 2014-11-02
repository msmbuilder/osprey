from __future__ import print_function, absolute_import, division
import os
import shutil
import tempfile

import numpy as np
import sklearn.datasets
from sklearn.externals.joblib import dump

from osprey.dataset_loaders import FilenameDatasetLoader
from osprey.dataset_loaders import JoblibDatasetLoader
from osprey.dataset_loaders import SklearnDatasetLoader


def test_FilenameDatasetLoader_1():
    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    try:
        os.chdir(dirname)

        open('filename-1', 'w').close()
        open('filename-2', 'w').close()

        assert FilenameDatasetLoader.short_name == 'filename'
        loader = FilenameDatasetLoader('filename-*')
        X, y = loader.load()

        X_ref = list(map(os.path.abspath, ['filename-1', 'filename-2']))
        assert sorted(X) == X_ref, X
        assert y is None, y

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def test_JoblibDatasetLoader_1():
    assert JoblibDatasetLoader.short_name == 'joblib'

    cwd = os.path.abspath(os.curdir)
    dirname = tempfile.mkdtemp()
    try:
        os.chdir(dirname)

        # one file
        dump(np.zeros((10, 2)), 'f1.pkl')
        loader = JoblibDatasetLoader('f1.pkl')
        X, y = loader.load()
        assert np.all(X == np.zeros((10, 2)))
        assert y is None

        # two files
        dump(np.ones((10, 2)), 'f2.pkl')
        loader = JoblibDatasetLoader('f*.pkl')
        X, y = loader.load()
        assert isinstance(X, list)
        assert np.all(X[0] == np.zeros((10, 2)))
        assert np.all(X[1] == np.ones((10, 2)))
        assert y is None

        # one file, with x and y
        dump({'foo': 'baz', 'bar': 'qux'}, 'foobar.pkl')
        loader = JoblibDatasetLoader('foobar.pkl', x_name='foo', y_name='bar')
        X, y = loader.load()
        assert X == 'baz', X
        assert y == 'qux', y

    finally:
        os.chdir(cwd)
        shutil.rmtree(dirname)


def test_SklearnDatasetLoader_1():
    assert SklearnDatasetLoader.short_name == 'sklearn_dataset'
    X, y = SklearnDatasetLoader('load_iris').load()
    iris = sklearn.datasets.load_iris()
    assert np.all(X == iris['data'])
    assert np.all(y == iris['target'])
