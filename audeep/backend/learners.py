# Copyright (C) 2017-2018 Michael Freitag, Shahin Amiriparian, Sergey Pugachevskiy, Nicholas Cummins, Björn Schuller
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with auDeep. If not, see <http://www.gnu.org/licenses/>.

"""Machine learning algorithms for classification"""
import datetime
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Mapping

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from audeep.backend.data.data_set import DataSet
from audeep.backend.data.upsample import upsample
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
            data_set: DataSet):
        """
        Fit the learner to the specified data.
        
        Parameters
        ----------
        data_set: DataSet
            The data set on which to train the learner. Must be fully labelled.

        Raises
        ------
        ValueError
            If the feature matrix has invalid shape
        """
        if len(data_set.feature_shape) != 1:
            raise ValueError("invalid number of feature dimensions: {}".format(len(data_set.feature_shape)))

        self._num_features = data_set.feature_shape[0]

    @abstractmethod
    def predict(self,
                data_set: DataSet):
        """
        Predict the labels of the specified data set.
        
        Parameters
        ----------
        data_set: DataSet
            Data set containing instances for which to predict labels.

        Returns
        -------
        numpy.ndarray
            The predicted labels of the specified data
            
        Raises
        ------
        ValueError
            If the specified feature matrix has invalid shape
        """
        if len(data_set.feature_shape) != 1:
            raise ValueError("invalid number of feature dimensions: {}".format(len(data_set.feature_shape)))
        if data_set.feature_shape[0] != self.num_features:
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
            data_set: DataSet):
        # generic parameter checks
        super().fit(data_set)

        self._model.fit(X=data_set.features, y=data_set.labels_numeric)

    def predict(self,
                data_set: DataSet):
        # generic parameter checks
        super().predict(data_set)

        return self._model.predict(data_set.features)


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
            data_set: DataSet):
        # generic parameter checks
        super().fit(data_set)

        self._num_labels = len(data_set.label_map)

        graph = tf.Graph()

        with graph.as_default():
            tf_inputs = tf.Variable(initial_value=data_set.features, trainable=False, dtype=tf.float32)
            tf_labels = tf.Variable(initial_value=data_set.labels_numeric, trainable=False, dtype=tf.int32)

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
                data_set: DataSet):
        # generic parameter checks
        super().predict(data_set)

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

            predictions = session.run(tf_prediction, feed_dict={tf_inputs: data_set.features})

        return predictions


def _majority_vote(chunked_data_set: DataSet,
                   chunked_predictions: np.ndarray) -> Mapping[str, int]:
    """
    Compute predictions on a chunked data set using majority voting.

    Predictions for an audio file are computed by looking at the predictions for the individual chunks, and selecting 
    the most-chosen prediction.

    Parameters
    ----------
    chunked_data_set: DataSet
        A data set containing filename and chunk number metadata. No label information is required.
    chunked_predictions: numpy.ndarray
        A one-dimensional NumPy array containing predictions for the chunks. Must have the same number of entries as 
        the specified data set has instances.

    Returns
    -------
    map of str to int
        The predictions for the audio files computed using majority voting over the predictions on the individual chunks
    """
    predictions = {}

    for index in chunked_data_set:
        instance = chunked_data_set[index]

        if instance.filename not in predictions:
            predictions[instance.filename] = []

        predictions[instance.filename].append(chunked_predictions[index])

    predictions = {item[0]: np.argmax(np.bincount(item[1])) for item in predictions.items()}

    return predictions


class PreProcessingWrapper(LearnerBase):
    """
    Wraps another learner, and applies the "standard" pre-processing used in the auDeep system.
    
    Standardization coefficients are computed from training data, and used to standardize features during prediction. 
    Training data is always shuffled, and can optionally be upsampled to balance classes. Furthermore, majority voting
    can be used to obtain predictions in the case of chunked instances.
    """

    def __init__(self,
                 learner: LearnerBase,
                 upsample: bool,
                 majority_vote: bool):
        """
        Create and initialize a new PreProcessingWrapper with the specified parameters
        
        Parameters
        ----------
        learner: LearnerBase
            The learned which is wrapped by this class. Actual training and prediction is delegated to this learner.
        upsample: bool
            Whether to balance classes in the training data through upsampling
        majority_vote: bool
            Whether to use majority voting to obtain predictions on chunked instances. This parameter can be set even if
            the data on which predictions are computed does not contain chunked instances.
        """
        self._learner = learner
        self._upsample = upsample
        self._majority_vote = majority_vote
        self._scaler = None

    def fit(self,
            data_set: DataSet):
        # generic parameter checks
        super().fit(data_set)

        if self._upsample:
            data_set = upsample(data_set)

        # shuffle data set after upsampling
        data_set = data_set.shuffled()

        # standardize features and remember coefficients for prediction
        self._scaler = StandardScaler()
        self._scaler.fit(data_set.features)

        data_set = data_set.scaled(self._scaler)

        # train model
        self._learner.fit(data_set)

    def predict(self,
                data_set: DataSet) -> Mapping[str, int]:
        super().predict(data_set)

        if self._scaler is None:
            raise RuntimeError("no model has been built yet. Invoke fit before predict")

        # no upsampling during prediction - we may not even have labels at this point
        # standardize data using coefficients computed during training
        data_set = data_set.scaled(self._scaler)

        # get predictions
        chunked_predictions = self._learner.predict(data_set)

        if self._majority_vote:
            return _majority_vote(data_set, chunked_predictions)
        else:
            return dict(zip(data_set.filenames, chunked_predictions))
