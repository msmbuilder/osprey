"""
pylearn2 plugin.
"""
import numpy as np
import re
from string import Template

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error

from osprey.dataset_loaders import BaseDatasetLoader


class Pylearn2Estimator(BaseEstimator):
    """
    Wrapper for pylearn2 models that conforms to the sklearn BaseEstimator API.

    This class does not handle Train objects directly, because it is much
    simpler to handle set_params with the YAML and simple string formatting.

    YAML strings should be formatted for compatibility with string.Template.

    Parameters
    ----------
    yaml_string : str
        YAML string defining the model.
    """
    def __init__(self, yaml_string, **kwargs):
        self.trainer = None
        self.yaml_string = yaml_string
        self.set_params(**kwargs)

    def _get_param_names(self):
        """
        Get mappable parameters from YAML.
        """
        template = Template(self.yaml_string)
        names = ['yaml_string']  # always include the template
        for match in re.finditer(template.pattern, template.template):
            name = match.group('named') or match.group('braced')
            assert name is not None
            names.append(name)
        return names

    def _get_dataset(self, X, y=None):
        """
        Construct a pylearn2 dataset.

        Parameters
        ----------
        X : array_like
            Training examples.
        y : array_like, optional
            Labels.
        """
        from pylearn2.datasets import DenseDesignMatrix

        X = np.asarray(X)
        assert X.ndim > 1
        if y is not None:
            y = self._get_labels(y)
        if X.ndim == 2:
            return DenseDesignMatrix(X=X, y=y)
        return DenseDesignMatrix(topo_view=X, y=y)

    def _get_labels(self, y):
        """
        Construct pylearn2 dataset labels.

        Parameters
        ----------
        y : array_like, optional
            Labels.
        """
        y = np.asarray(y)
        if y.ndim == 1:
            return y.reshape((y.size, 1))
        assert y.ndim == 2
        return y

    def fit(self, X, y=None):
        """
        Build a trainer and run main_loop.

        Parameters
        ----------
        X : array_like
            Training examples.
        y : array_like, optional
            Labels.
        """
        from pylearn2.config import yaml_parse
        from pylearn2.train import Train

        # build trainer
        params = self.get_params()
        yaml_string = Template(self.yaml_string).substitute(params)
        self.trainer = yaml_parse.load(yaml_string)
        assert isinstance(self.trainer, Train)
        if self.trainer.dataset is not None:
            raise ValueError('Train YAML database must evaluate to None.')
        self.trainer.dataset = self._get_dataset(X, y)

        # update monitoring dataset(s)
        if (hasattr(self.trainer.algorithm, 'monitoring_dataset') and
                self.trainer.algorithm.monitoring_dataset is not None):
            monitoring_dataset = self.trainer.algorithm.monitoring_dataset
            if len(monitoring_dataset) == 1 and '' in monitoring_dataset:
                monitoring_dataset[''] = self.trainer.dataset
            else:
                monitoring_dataset['train'] = self.trainer.dataset
            self.trainer.algorithm._set_monitoring_dataset(monitoring_dataset)
        else:
            self.trainer.algorithm._set_monitoring_dataset(
                self.trainer.dataset)

        # run main loop
        self.trainer.main_loop()

    def predict(self, X):
        """
        Get model predictions.

        Parameters
        ----------
        X : array_like
            Test dataset.
        """
        return self._predict(X)

    def _predict(self, X, method='fprop'):
        """
        Get model predictions.

        See pylearn2.scripts.mlp.predict_csv and
        http://fastml.com/how-to-get-predictions-from-pylearn2/.

        Parameters
        ----------
        X : array_like
            Test dataset.
        method : str
            Model method to call for prediction.
        """
        import theano

        X_sym = self.trainer.model.get_input_space().make_theano_batch()
        y_sym = getattr(self.trainer.model, method)(X_sym)
        f = theano.function([X_sym], y_sym, allow_input_downcast=True)
        return f(X)

    def score(self, X, y):
        """
        Score predictions.

        Parameters
        ----------
        X : array_like
            Test examples.
        y : array_like, optional
            Labels.
        """
        raise NotImplementedError('should be implemented in subclasses')


class Pylearn2Classifier(Pylearn2Estimator):
    """
    pylearn2 classifier.
    """
    def _get_labels(self, y):
        """
        Construct pylearn2 dataset labels.

        Parameters
        ----------
        y : array_like, optional
            Labels.
        """
        y = np.asarray(y)
        assert y.ndim == 1
        # convert to one-hot
        labels = np.unique(y).tolist()
        oh = np.zeros((y.size, len(labels)), dtype=float)
        for i, label in enumerate(y):
            oh[i, labels.index(label)] = 1.
        return oh

    def predict_proba(self, X):
        """
        Get model predictions.

        Parameters
        ----------
        X : array_like
            Test dataset.
        """
        return self._predict(X)

    def predict(self, X):
        """
        Get model predictions.

        Parameters
        ----------
        X : array_like
            Test dataset.
        """
        return np.argmax(self._predict(X), axis=1)

    def score(self, X, y):
        """
        Score predictions.

        Parameters
        ----------
        X : array_like
            Test examples.
        y : array_like, optional
            Labels.
        """
        return accuracy_score(y, self.predict(X))


class Pylearn2Regressor(Pylearn2Estimator):
    """
    pylearn2 regressor.
    """
    def score(self, X, y):
        """
        Score predictions.

        Parameters
        ----------
        X : array_like
            Test examples.
        y : array_like, optional
            Labels.
        """
        return mean_squared_error(y, self.predict(X))


class Pylearn2Autoencoder(Pylearn2Estimator):
    """
    pylearn2 autoencoder.
    """
    def _predict(self, X, method='reconstruct'):
        return super(Pylearn2Autoencoder, self)._predict(X, method)

    def score(self, X, y):
        """
        Score predictions.

        Parameters
        ----------
        X : array_like
            Test examples.
        y : array_like, optional
            Labels (not used).
        """
        return mean_squared_error(X, self.predict(X))


class Pylearn2DatasetLoader(BaseDatasetLoader):
    """
    pylearn2 dataset loader.

    Parameters
    ----------
    yaml_string : str
        YAML specification of a pylearn2 dataset.
    one_hot : bool, optional
        Take argmax of one-hot labels to get classes.
    """
    short_name = 'pylearn2_dataset'

    def __init__(self, yaml_string, one_hot=False):
        self.yaml_string = yaml_string
        self.one_hot = one_hot

    def load(self):
        """
        Load the dataset using pylearn2.config.yaml_parse.
        """
        from pylearn2.config import yaml_parse
        from pylearn2.datasets import Dataset

        dataset = yaml_parse.load(self.yaml_string)
        assert isinstance(dataset, Dataset)
        data = dataset.iterator(mode='sequential', num_batches=1,
                                data_specs=dataset.data_specs,
                                return_tuple=True).next()
        if len(data) == 2:
            X, y = data
            y = np.squeeze(y)
            if self.one_hot:
                y = np.argmax(y, axis=1)
        else:
            X = data
            y = None
        return X, y
