from argparse import Namespace
import json
import numpy as np
from numpy.testing.decorators import skipif
import shutil
import sys
import tempfile
import unittest

try:
    import pylearn2
    del pylearn2  # for flake8, since not used in tests
except ImportError:
    pass

from osprey import execute_worker, execute_dump
from osprey.plugins.plugin_pylearn2 import Pylearn2DatasetLoader


class TestPluginPylearn2(unittest.TestCase):
    """
    Test plugin_pylearn2.
    """
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def run_osprey(self, config):
        """
        Run osprey-worker.

        Parameters
        ----------
        config : str
            Configuration string.
        """
        fh, filename = tempfile.mkstemp(dir=self.temp_dir)
        with open(filename, 'wb') as f:
            f.write(config)
        args = Namespace(config=filename, n_iters=1, output='json')
        execute_worker.execute(args, None)
        dump = json.loads(execute_dump.execute(args, None))
        assert len(dump) == 1
        assert dump[0]['status'] == 'SUCCEEDED', dump[0]['status']

    @skipif('pylearn2' not in sys.modules, 'this test requires pylearn2')
    def test_pylearn2_classifier(self):
        config = """
estimator:
  entry_point: osprey.plugins.plugin_pylearn2.Pylearn2Classifier
  params:
    yaml_string: "!obj:pylearn2.train.Train {
  dataset: null,
  model: !obj:pylearn2.models.mlp.MLP {
    nvis: 4,
    layers: [
      !obj:pylearn2.models.mlp.Sigmoid {
        layer_name: h0,
        dim: $dim,
        irange: 0.05,
      },
      !obj:pylearn2.models.mlp.Softmax {
        layer_name: y,
        n_classes: 3,
        irange: 0.,
      },
    ],
  },
  algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
    batch_size: 10,
    termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
      max_epochs: 1,
    },
  },
}"

dataset_loader:
  name: sklearn_dataset
  params:
    method: load_iris

strategy:
  name: random

search_space:
  dim:
    min: 1
    max: 10
    type: int

cv:
  name: stratifiedkfold

trials:
  uri: sqlite:///test.db
"""
        self.run_osprey(config)

    @skipif('pylearn2' not in sys.modules, 'this test requires pylearn2')
    def test_pylearn2_regressor(self):
        config = """
estimator:
  entry_point: osprey.plugins.plugin_pylearn2.Pylearn2Regressor
  params:
    yaml_string: "!obj:pylearn2.train.Train {
  dataset: null,
  model: !obj:pylearn2.models.mlp.MLP {
    nvis: 13,
    layers: [
      !obj:pylearn2.models.mlp.Sigmoid {
        layer_name: h0,
        dim: $dim,
        irange: 0.05,
      },
      !obj:pylearn2.models.mlp.Linear {
        layer_name: y,
        dim: 1,
        irange: 0.,
      },
    ],
  },
  algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
    batch_size: 10,
    termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
      max_epochs: 1,
    },
  },
}"

dataset_loader:
  name: sklearn_dataset
  params:
    method: load_boston

strategy:
  name: random

search_space:
  dim:
    min: 1
    max: 10
    type: int

cv: 3

trials:
  uri: sqlite:///test.db
"""
        self.run_osprey(config)

    @skipif('pylearn2' not in sys.modules, 'this test requires pylearn2')
    def test_pylearn2_autoencoder(self):
        config = """
estimator:
  entry_point: osprey.plugins.plugin_pylearn2.Pylearn2Autoencoder
  params:
    yaml_string: "!obj:pylearn2.train.Train {
  dataset: null,
  model: !obj:pylearn2.models.autoencoder.Autoencoder {
    act_dec: null,
    act_enc: sigmoid,
    irange: 0.05,
    nvis: 13,
    nhid: $dim,
  },
  algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
    batch_size: 10,
    cost: !obj:pylearn2.costs.autoencoder.MeanSquaredReconstructionError {},
    termination_criterion: !obj:pylearn2.termination_criteria.EpochCounter {
      max_epochs: 1,
    },
  },
}"

dataset_loader:
  name: sklearn_dataset
  params:
    method: load_boston

strategy:
  name: random

search_space:
  dim:
    min: 1
    max: 10
    type: int

cv: 3

trials:
  uri: sqlite:///test.db
"""
        self.run_osprey(config)

    @skipif('pylearn2' not in sys.modules, 'this test requires pylearn2')
    def test_pylearn2_dataset_loader(self):
        yaml_string = """
!obj:pylearn2.testing.datasets.random_dense_design_matrix {
  rng: !obj:numpy.random.RandomState { seed: 1 },
  num_examples: 10,
  dim: 5,
  num_classes: 3,
}
"""
        dataset_loader = Pylearn2DatasetLoader(yaml_string, one_hot=False)
        X, y = dataset_loader.load()
        assert X.shape == (10, 5)
        assert y.shape == (10,)
        assert np.unique(y).size == 3

    @skipif('pylearn2' not in sys.modules, 'this test requires pylearn2')
    def test_pylearn2_dataset_loader_one_hot(self):
        yaml_string = """
!obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
  rng: !obj:numpy.random.RandomState { seed: 1 },
  num_examples: 10,
  dim: 5,
  num_classes: 3,
}
"""
        dataset_loader = Pylearn2DatasetLoader(yaml_string, one_hot=True)
        X, y = dataset_loader.load()
        assert X.shape == (10, 5)
        assert y.shape == (10,)
        assert np.unique(y).size == 3
