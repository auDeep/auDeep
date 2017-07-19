# Copyright (C) 2017  Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
#
# This file is part of auDeep.
#
# auDeep is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# auDeep is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep.  If not, see <http://www.gnu.org/licenses/>.

"""Machine learning algorithms for classification"""
import datetime
from abc import ABCMeta, abstractmethod
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC

from audeep.backend.models.mlp import MLPModel


class LearnerBase:
    """
    Defines a common interface for all classification algorithms.
    """
    __metaclass__ = ABCMeta

    _num_features = None

    @property
    def num_features(self):
        """
        Returns the number of features for which the learner was trained.
        
        This property is updated each time the `fit` method is called on some data. Any subsequent calls to the
        `predict` method then expect the same number of features.
        
        Returns
        -------
        int
            The number of features for which the learner was trained
        """
        return self._num_features

    @abstractmethod
    def fit(self,
            features: np.ndarray,
            labels: np.ndarray):
        """
        Fit the learner to the specified data.
        
        Parameters
        ----------
        features: numpy.ndarray
            Feature matrix of shape [num_samples, num_features]
        labels: numpy.ndarray
            Label matrix of shape [num_samples]

        Raises
        ------
        ValueError
            If the feature or label matrices have invalid shape
        """
        if len(features.shape) != 2:
            raise ValueError("invalid feature matrix dimensionality: {}".format(len(features.shape)))
        if len(labels.shape) != 1:
            raise ValueError("invalid label matrix dimensionality: {}".format(len(labels.shape)))
        if features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels have different numbers of instances")

        self._num_features = features.shape[1]

    @abstractmethod
    def predict(self,
                features: np.ndarray):
        """
        Predict the labels of the specified data.
        
        Parameters
        ----------
        features: numpy.ndarray
            Feature matrix of shape [num_samples, num_features]

        Returns
        -------
        numpy.ndarray
            The predicted labels of the specified data
            
        Raises
        ------
        ValueError
            If the specified feature matrix has invalid shape
        """
        if len(features.shape) != 2:
            raise ValueError("invalid feature matrix dimensionality: {}".format(len(features.shape)))
        if features.shape[1] != self.num_features:
            raise ValueError("learner was trained with different number of features")


class LibLINEARLearner(LearnerBase):
    """
    A learner using LibLINEAR for classification. 
    """

    def __init__(self,
                 complexity: float):
        """
        Create and initialize a new LibLINEARLearner with the specified SVM complexity parameter.
        
        Parameters
        ----------
        complexity: float
            The SVM complexity parameter
        """
        self._model = LinearSVC(C=complexity)

    def fit(self,
            features: np.ndarray,
            labels: np.ndarray,
            quiet=False):
        # generic parameter checks
        super().fit(features, labels)

        self._model.fit(X=features, y=labels)

    def predict(self,
                features: np.ndarray):
        # generic parameter checks
        super().predict(features)

        return self._model.predict(features)


class TensorflowMLPLearner(LearnerBase):
    """
    A learner using Tensorflow to build an MLP classifier.
    """

    def __init__(self,
                 checkpoint_dir: Path,
                 num_layers: int,
                 num_units: int,
                 learning_rate=0.001,
                 num_epochs=400,
                 keep_prob=0.6,
                 shuffle_training=False):
        """
        Create and initialize a new TensorflowMLPLearner with the specified parameters.
        
        Due to limitations of Tensorflow (or my inability to find a better solution), this learner has to serialize
        trained model to the hard disk between invocations of the `fit` and `predict` methods.
        
        Furthermore, the entire training data is copied to GPU memory to increase training performance. Thus, only 
        moderately-sized models can be trained using this learner.
        
        Parameters
        ----------
        checkpoint_dir: pathlib.Path
            A directory in which temporary models can be stored
        num_layers: int
            The number of layers in the MLP
        num_units: int
            The number of units in each layer
        learning_rate: float, default 0.001
            The learning rate for training
        num_epochs: int, default 400
            The number of training epochs
        keep_prob: float, default 0.6
            The probability to keep hidden activations
        shuffle_training: bool, default False
            Shuffle training data between epochs
        """
        self._checkpoint_dir = checkpoint_dir
        self._latest_checkpoint = None
        self._num_labels = None
        self._model = MLPModel(num_layers, num_units)
        self._learning_rate = learning_rate
        self._num_epochs = num_epochs
        self._keep_prob = keep_prob
        self._shuffle_training = shuffle_training

    def fit(self,
            features: np.ndarray,
            labels: np.ndarray,
            quiet=False):
        # generic parameter checks
        super().fit(features, labels)

        self._num_labels = len(np.unique(labels))

        graph = tf.Graph()

        with graph.as_default():
            tf_inputs = tf.Variable(initial_value=features, trainable=False, dtype=tf.float32)
            tf_labels = tf.Variable(initial_value=labels, trainable=False, dtype=tf.int32)

            if self._shuffle_training:
                tf_inputs = tf.random_shuffle(tf_inputs, seed=42)
                tf_labels = tf.random_shuffle(tf_labels, seed=42)

            with tf.variable_scope("mlp"):
                tf_logits = self._model.inference(tf_inputs, self._keep_prob, self._num_labels)
                tf_loss = self._model.loss(tf_logits, tf_labels)
                tf_train_op = self._model.optimize(tf_loss, self._learning_rate)

            tf_init_op = tf.global_variables_initializer()
            tf_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mlp"))

        session = tf.Session(graph=graph)
        session.run(tf_init_op)

        for epoch in range(self._num_epochs):
            session.run(tf_train_op)

        # timestamped model file
        self._latest_checkpoint = self._checkpoint_dir / "model_{:%Y%m%d%H%M%S%f}".format(datetime.datetime.now())
        tf_saver.save(session, str(self._latest_checkpoint), write_meta_graph=False)

        session.close()

    def predict(self,
                features: np.ndarray):
        # generic parameter checks
        super().predict(features)

        if self._latest_checkpoint is None:
            raise RuntimeError("no model has been built yet. Invoke fit before predict")

        graph = tf.Graph()

        with graph.as_default():
            tf_inputs = tf.placeholder(shape=[None, self.num_features], dtype=tf.float32, name="inputs")

            with tf.variable_scope("mlp"):
                tf_logits = self._model.inference(tf_inputs, 1.0, self._num_labels)
                tf_prediction = self._model.prediction(tf_logits)

            tf_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mlp"))

        with tf.Session(graph=graph) as session:
            tf_saver.restore(session, str(self._latest_checkpoint))

            predictions = session.run(tf_prediction, feed_dict={tf_inputs: features})

        return predictions
