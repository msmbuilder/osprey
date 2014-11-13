"""
pylearn2 plugin.
"""
import numpy as np
import re
from string import Template

from pylearn2.config import yaml_parse
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.train import Train

from sklearn.base import BaseEstimator

import theano


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
    def __init__(self, yaml_string):
        self.trainer = None
        self.yaml_string = self.check_yaml_string(yaml_string)

    def check_yaml_string(self, string):
        """
        Inspect a YAML string.

        Parameters
        ----------
        string : str
            YAML string.
        """
        return string

    def _get_param_names(self):
        """
        Get mappable parameters from YAML.
        """
        template = Template(self.yaml_string)
        names = []
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
        raise NotImplementedError('should be implemented in subclasses')

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
        self.trainer = yaml_parse.load(self.yaml_string % self.get_params())
        assert isinstance(self.trainer, Train)
        if self.trainer.database is not None:
            raise ValueError('Train YAML database must evaluate to None.')
        self.trainer.database = self._get_dataset(X, y)
        self.trainer.main_loop()

    def _predict(self, X):
        """
        Get model predictions.

        See pylearn2.scripts.mlp.predict_csv and
        http://fastml.com/how-to-get-predictions-from-pylearn2/.

        Parameters
        ----------
        X : pylearn2.datasets.Dataset
            Test dataset.
        """
        X_sym = self.trainer.model.get_input_space().make_theano_batch()
        y_sym = self.trainer.model.fprop(X_sym)
        f = theano.function([X_sym], y_sym)
        return f(X)


class Pylearn2Classifier(Pylearn2Estimator):
    """
    Pylearn2 classifier.
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
        return y

    def predict_proba(self, X):
        """
        Get model predictions.

        Parameters
        ----------
        X : pylearn2.datasets.Dataset
            Test dataset.
        """
        return self._predict(X)

    def predict(self, X):
        """
        Get model predictions.

        Parameters
        ----------
        X : pylearn2.datasets.Dataset
            Test dataset.
        """
        return np.argmax(self._predict(X), axis=1)


class Pylearn2Regressor(Pylearn2Estimator):
    """
    Pylearn2 regressor.
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
        if y.ndim == 1:
            return y.reshape((y.size, 1))
        assert y.ndim == 2
        return y

    def predict(self, X):
        """
        Get model predictions.

        Parameters
        ----------
        X : pylearn2.datasets.Dataset
            Test dataset.
        """
        return self._predict(X)
