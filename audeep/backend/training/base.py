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

"""An abstract interface for feature learning algorithms"""
import abc
import time
from pathlib import Path
from typing import Sequence, Mapping, Iterable, Dict, Optional

import numpy as np
import tensorflow as tf

from audeep.backend.data.data_set import DataSet
from audeep.backend.log import LoggingMixin
from audeep.backend.models.spectrogram_queue import SpectrogramQueue


class GraphWrapper(LoggingMixin):
    """
    Wraps a Tensorflow graph containing a feature learning model, and provides access to several commonly used tensors.
    
    Since graphs are often loaded from metagraph files, there is usually no way of obtaining an instance of the model
    class that was used to create a model. Thus, we add key operations and tensors to singleton collections in the
    Tensorflow graph, so that they can be retrieved through the respective collection name. The following collections
    are currently used:
    
    - `representation`: Contains the hidden representation of input sequences, of shape [batch_size, ...]
    - `train_op`: Contains the operation for performing a single training step
    - GraphKeys.LOSSES: Contains one or more scalar tensors which are added to obtain the global loss
    
    In addition to providing access to tensors, this class provides the functionality to restore the variables in a 
    graph from a checkpoint file, or to initialize them using their respective initializers.
    """

    def __init__(self,
                 graph: tf.Graph,
                 saver: tf.train.Saver):
        """
        Creates and initializes a new GraphWrapper for the specified graph.
        
        The specified `Saver` instance is used internally when restoring variables from a checkpoint file.
        
        Parameters
        ----------
        graph: tf.Graph
            The graph which should be wrapped
        saver: tf.train.Saver
            Saver for restoring variables from checkpoint files
        """
        super().__init__()

        self.graph = graph
        self.saver = saver

    def restore_or_initialize(self,
                              session: tf.Session,
                              model_filename: Path,
                              global_step: Optional[int]) -> int:
        """
        Tries to find a checkpoint file from which to restore variables in the graph.
        
        If no checkpoint file is found, the variables are initialized using their respective initializers. 
        
        This method searches for checkpoint files in the directory containing `model_filename`. It is assumed that 
        model files have the same base filename as `model_filename`, followed by a dash, followed by the global step 
        number. If, for example, `model_filename` were to be "./log/model", valid checkpoint filenames would be, for
        instance, "./log/model-10", "./log/model-100".
        
        If the parameter `global_step` is not given, this method scans all checkpoint files matching the pattern 
        described above, and selects the latest checkpoint. If `global_step` is given, this method tries to restore
        variables from a checkpoint file for that specific global step, and fails if such a file does not exist.
        
        Parameters
        ----------
        session: tf.Session
            The Tensorflow session in which to restore variables
        model_filename: pathlib.Path
            The name of model files, without extension. Tensorflow saves models in several files per checkpoint, and
            appends, for example, the global step number to filenames. This parameter should indicate the common prefix
            for these filenames, analogous to the `save_path` parameter of the `tf.train.Saver.save` method.
        global_step: int, optional
            If given, restore variables values at the specified global step. Otherwise, restore variables from the
            latest checkpoint.

        Returns
        -------
        int
            The global step number from which variables were restored, or None if no checkpoint files were found and
            variables were initialized using their initializers
        """
        if global_step is None:
            self.log.debug("no global step specified - searching for checkpoints")

            global_step = -1

            for file in model_filename.parent.glob("%s-*.index" % model_filename.name):
                name = str(file.name)
                step = int(name[len(model_filename.name) + 1:name.find(".")])

                self.log.debug("found checkpoint for global step %d at %s", step, file)

                global_step = max(global_step, step)

        if global_step is -1:
            self.log.info("initializing variables")
            session.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "model")))
        else:
            filename = model_filename.with_name("%s-%d" % (model_filename.name, global_step))

            self.log.info("restoring variables from %s", filename)
            self.saver.restore(session, str(filename))

        return global_step if global_step != -1 else None

    @property
    def representation(self) -> tf.Tensor:
        """
        Returns a tensor containing the hidden representation of input sequences.
        
        Returns
        -------
        tf.Tensor
            A tensor containing the hidden representation of input sequences
        """
        return self.graph.get_collection("representation")[0]

    @property
    def loss(self) -> tf.Tensor:
        """
        Returns a scalar tensor containing the loss of a batch.
        
        Returns
        -------
        tf.Tensor
            A scalar tensor containing the loss of a batch
        """
        return tf.add_n(self.graph.get_collection(tf.GraphKeys.LOSSES))

    @property
    def train_op(self) -> tf.Operation:
        """
        Returns an operation for performing a single training step.
        
        Returns
        -------
        tf.Operation
            An operation for performing a single training step
        """
        return self.graph.get_collection("train_op")[0]


class BaseFeatureLearningWrapper(LoggingMixin):
    """
    Wrapper for a feature learning model.
    
    This class defines a common interface for all DNN-based feature learning algorithms. Core operations of this
    interface are listed below.
    
     - Initializing a model: Given the network architecture and other architectural hyperparameters, build a Tensorflow 
       graph for the model and serialize it to a meta-graph file. This operation initializes the model with dummy inputs
       of the correct shape and data type, which are stripped from the graph before serialization. That way, models can
       be serialized independently from the input pipeline. This enables, among others, continuation of training with
       different input data, or transparently switching input queues for feed dictionaries.
     - Training a model: Deserialize a model and train it on some data. This operation can be invoked multiple times 
       on the same model, with different training data and training hyperparameters such as the learning rate. After
       each training epoch, the model is serialized.
     - Using a trained model to generate features: Deserialize a trained model, and use it to generate features from 
       some data. Features are generated by feeding data to the model, and extracting the hidden representation for each
       input instance as a feature vector or matrix for that instance.
    
    This class can be used with a wide variety of feature learning models, which have to meet only a minimal set of
    requirements, which are listed below.
    
     - Training of models must occur fully unsupervised.
     - Models must expose the interface defined by the `GraphWrapper` class, i.e. they must expose a representation
       tensor, one or more loss tensors, and a training operation.
     
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _create_model(self,
                      tf_inputs: tf.Tensor,
                      **kwargs):
        """
        Build the computation graph for the model using dummy inputs.
        
        The `tf_inputs` parameter contains a Tensorflow placeholder for the training data which will be passed to the
        model. The only purpose of this placeholder is to provide input data shape information for graph construction,
        since it will be stripped from the graph before serialization.
        
        Subclasses must initialize the actual model under a variable scope with name "model". Everything below this
        variable scope will be serialized, while everything not below this variable scope will be stripped from the 
        graph before serialization. Subclasses may define an arbitrary number of additional input placeholders, as long
        as they are placed outside the "model" scope. These additional input placeholders must be named, and a mapping
        for the respective name must be present in the maps returned by the `_training_parameters` and 
        `_generation_parameters` functions.
        
        Parameters
        ----------
        tf_inputs: tf.Tensor
            Dummy input sequence tensor. 
        kwargs: keyword arguments
            Additional keyword arguments. Any additional keyword arguments passed to the `initialize_model` method will
            be forwarded to this method. They can be used, for example, to specify the model architecture.
        """
        pass

    @abc.abstractmethod
    def _training_parameters(self,
                             **kwargs) -> Dict[str, tf.Tensor]:
        """
        Provide bindings during training for any additional input placeholders created by the `_create_model` function.
        
        This function returns a mapping of input placeholder names to actual tensors, which will be bound to the input
        placeholders with the respective names. Consider, for example, the following simple case. A subclass implements
        a wrapper around a model which requires a learning rate value as input. In order to achieve this, a scalar
        placeholder with name "learning_rate" is added to the graph outside the "model" variable scope in the 
        `_create_model` method:
        
        >>> def _create_model(self,
        >>>                   tf_inputs: tf.Tensor,
        >>>                   **kwargs):
        >>>     tf_learning_rate = tf.placeholder(name="learning_rate",
        >>>                                       shape=[],
        >>>                                       dtype=tf.float32)
        >>>     
        >>>     with tf.variable_scope("model"):
        >>>         Model(inputs=tf_inputs,
        >>>               learning_rate=tf_learning_rate,
        >>>               ...)
        
        Since the `tf_learning_rate` placeholder is created outside the "model" vairable scope, it will be stripped 
        from the graph before serialization, which will therefore contain an unbound input with name "learning_rate".
        
        Given the situation outlined above, the `_training_parameters` mapping method must contain a mapping for the 
        "learning_rate" input placeholder. One option would be, for example, to use a constant learning rate value which
        is set by a keyword argument. In this case, the `_training_parameters` could be implemented as follows.
        
        >>> def _training_parameters(self,
        >>>                         **kwargs):
        >>>     return {
        >>>         "learning_rate": tf.constant(name="learning_rate", value=kwargs["learning_rate"])
        >>>     }
        
        Parameters
        ----------
        kwargs: keyword arguments
            Additional keyword arguments. Any additional keyword arguments passed to the `train_model` method will be
            forwarded to this method. They can be used, for example, to specify values for various training 
            hyperparameters
            
        Returns
        -------
        map of str to tf.Tensor
            A mapping of input placeholder names to tensors
        """
        pass

    @abc.abstractmethod
    def _generation_parameters(self,
                               **kwargs) -> Dict[str, tf.Tensor]:
        """
        Provide bindings during generation for any additional input placeholders created by the `_create_model` 
        function.
        
        This function returns a mapping of input placeholder names to actual tensors, which will be bound to the input
        placeholders with the respective names. This method must return a mapping of the same structure as the 
        `_training_parameters` method. However, since many training hyperparameters can or must be set to default values
        when using a trained model for feature generation, a separate method is used to create the mapping during
        feature generation.
        
        Parameters
        ----------
        kwargs: keyword arguments
            Additional keyword arguments. Any additional keyword arguments passed to the `generate` method will be
            forwarded to this method. They can be used, for example, to specify values for various training 
            hyperparameters

        Returns
        -------
        map of str to tf.Tensor
            A mapping of input placeholder names to tensors
        """
        pass

    def _load_graph(self,
                    model_filename: Path,
                    input_map: Mapping[str, tf.Tensor]) -> GraphWrapper:
        """
        Deserialize a Tensorflow graph.
                
        Parameters
        ----------
        model_filename: pathlib.Path
            The name of model files, without extension. Tensorflow saves models in several files per checkpoint, and
            appends, for example, the global step number to filenames. This parameter should indicate the common prefix
            for these filenames, analogous to the `save_path` parameter of the `tf.train.Saver.save` method.
        input_map: map of str to tf.Tensor
            Mapping containing bindings for any unbound inputs in the serialized graph. The prefix "$unbound_inputs_"
            will automatically be added to every mapping key, in order to ensure compatibility with Tensorflows
            `import_meta_graph` function.

        Returns
        -------
        GraphWrapper
            The Tensorflow graph deserialized from the specified file, with any unbound inputs bound to the respective
            tensors defined in the `input_map` parameter
        """
        input_map = {k: input_map[k] for k in input_map}

        saver = tf.train.import_meta_graph(str(model_filename.with_suffix(".meta")),
                                           clear_devices=True, import_scope=None,
                                           input_map=input_map)

        return GraphWrapper(graph=tf.get_default_graph(),
                            saver=saver)

    def initialize_model(self,
                         model_filename: Path,
                         feature_shape: Sequence[int],
                         **kwargs):
        """
        Initialize a feature learning model and serialize it to the specified file.
        
        The shape of the feature matrices that will be used as training data for the model must be known at 
        graph-construction time.
        
        Parameters
        ----------
        model_filename: pathlib.Path
            The name of the model file, without extension. Tensorflow saves models in several files per checkpoint, and
            appends, for example, the global step number to filenames. This parameter should indicate the common prefix
            for these filenames, analogous to the `save_path` parameter of the `tf.train.Saver.save` method.
        feature_shape: list of int
            The shape of the feature matrices that will be used as training data for the model
        kwargs: keyword arguments
            Additional keyword arguments specifying the model architecture
        """
        with tf.Graph().as_default() as g:


            tf_inputs = tf.placeholder(name="inputs",
                                       shape=[feature_shape[0], None, feature_shape[1]],
                                       dtype=tf.float32)

            self._create_model(tf_inputs, **kwargs)


            tf.train.export_meta_graph(filename=str(model_filename.with_suffix(".meta")),
                                       clear_devices=True, export_scope=None)

    def train_model(self,
                    model_filename: Path,
                    record_files: Iterable[Path],
                    feature_shape: Sequence[int],
                    num_instances: int,
                    num_epochs: int,
                    batch_size: int,
                    global_step: int = None,
                    checkpoints_to_keep: int = None,
                    **kwargs):
        """
        Deserialize a model and train it on some data, serializing the updated model after each training epoch.
                
        Parameters
        ----------
        model_filename: pathlib.Path
            The name of the model file, without extension. Tensorflow saves models in several files per checkpoint, and
            appends, for example, the global step number to filenames. This parameter should indicate the common prefix
            for these filenames, analogous to the `save_path` parameter of the `tf.train.Saver.save` method.
        record_files: list of pathlib.Path
            One or more TFRecords files containing training data. 
        feature_shape: list of int
            Shape of the feature matrices in the training data. The shape must match the feature shape that was used
            to initialize the model. 
        num_instances: int
            The total number of training instances in the specified TFRecords files
        num_epochs: int
            The number of training epochs
        batch_size: int
            The training batch size
        global_step: int, optional
            If set, restore the model variables to their state at the specified global step. If not set, the latest
            checkpoint is used to restore variables.
        checkpoints_to_keep: int, optional
            The number of checkpoints to keep on disk. Defaults to None, which will keep all checkpoints. If a number is
            set, only the most recent checkpoints will be kept.
        kwargs: keyword arguments
            Additional keyword arguments. They can be used, for example, to specify additional training hyperparameters
            such as the learning rate.
            
        See Also
        --------
        GraphWrapper.restore_or_initialize
        """
        if num_instances % batch_size == 0:
            num_batches = num_instances // batch_size
        else:
            num_batches = num_instances // batch_size + 1

        self.log.info("building computation graph")
        tf.reset_default_graph()

        queue = SpectrogramQueue(record_files=record_files,
                                 feature_shape=feature_shape,
                                 batch_size=batch_size)

        input_map = self._training_parameters(**kwargs)
        input_map["inputs"] = queue.input_queue

        graph_wrapper = self._load_graph(model_filename=model_filename,
                                         input_map=input_map)

        with tf.Session(graph=graph_wrapper.graph) as session:
            step = graph_wrapper.restore_or_initialize(session=session,
                                                       model_filename=model_filename,
                                                       global_step=global_step) or 0

            tf_merged_summaries = tf.summary.merge_all()

            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "model"),
                                   max_to_keep=checkpoints_to_keep)
            summary_writer = tf.summary.FileWriter(str(model_filename.parent), graph_wrapper.graph)

            self.log.info("preparing input queues")

            session.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=session, coord=coord)

            for epoch in range(num_epochs):
                for batch in range(num_batches):
                    start_time = time.time()

                    _, l, summary = session.run([graph_wrapper.train_op, graph_wrapper.loss, tf_merged_summaries])

                    duration = time.time() - start_time

                    self.log.info("epoch %d/%d, batch %d/%d, loss: %.4f (%.3f seconds)",
                                  epoch + 1, num_epochs, batch + 1, num_batches, l, duration)

                    summary_writer.add_summary(summary, global_step=step)
                    step += 1

                saver.save(session, str(model_filename),
                           global_step=step,
                           write_meta_graph=False)

            coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            summary_writer.close()

    def generate(self,
                 model_filename: Path,
                 data_set: DataSet,
                 batch_size: int,
                 global_step: int = None,
                 **kwargs) -> DataSet:
        """
        Use a trained model to generate features for the specified data.
        
        Parameters
        ----------
        model_filename: pathlib.Path
            The name of the model file, without extension. Tensorflow saves models in several files per checkpoint, and
            appends, for example, the global step number to filenames. This parameter should indicate the common prefix
            for these filenames, analogous to the `save_path` parameter of the `tf.train.Saver.save` method.
        data_set: DataSet
            Data set containing instances for which features should be generated. The shape of the feature matrices in
            this data set must match the shape for which the model was created.
        batch_size: int
            Generate features in batches of the specified size.
        global_step: int
            If set, restore the model variables to their state at the specified global step. If not set, the latest
            checkpoint is used to restore variables.
        kwargs: keyword arguments
            Additional keyword arguments

        Returns
        -------
        DataSet
            A data set containing the generated features for each instance in the specified data set, with the same
            metadata
        """
        self.log.info("building computation graph")
        tf.reset_default_graph()

        input_placeholder = tf.placeholder(name="inputs",
                                           shape=[data_set.feature_shape[0], None, data_set.feature_shape[1]],
                                           dtype=tf.float32)

        input_map = self._generation_parameters(**kwargs)
        input_map["inputs"] = input_placeholder

        graph_wrapper = self._load_graph(model_filename=model_filename,
                                         input_map=input_map)

        with tf.Session(graph=graph_wrapper.graph) as session:
            graph_wrapper.restore_or_initialize(session=session,
                                                model_filename=model_filename,
                                                global_step=global_step)

            # [batch, time, frequency]
            features = data_set.features
            # [time, batch, frequency]
            features = np.transpose(features, axes=[1, 0, 2])

            indices = np.arange(batch_size, features.shape[1], batch_size)
            feature_batches = np.split(features, indices, axis=1)

            new_features = []

            for index, feature_batch in enumerate(feature_batches):
                # shape: [batch, features]
                representation = session.run(graph_wrapper.representation, feed_dict={
                    input_placeholder: feature_batch
                })

                new_features.append(representation)

                self.log.info("processed batch %d/%d", index + 1, len(feature_batches))

            new_features = np.concatenate(new_features, axis=0)

            if len(new_features.shape) == 2:
                result = data_set.with_feature_dimensions([("generated", new_features.shape[1])])
            else:
                feature_dimensions = list(zip(["generated_%d" % i for i in range(len(features.shape) - 1)],
                                              features.shape[1:]))
                result = data_set.with_feature_dimensions(feature_dimensions)

            result.features = new_features
            result.freeze()

        return result
