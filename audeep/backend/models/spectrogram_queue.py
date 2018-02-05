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

"""A Tensorflow input queue for two-dimensional input data"""
from pathlib import Path
from typing import Iterable, Sequence

import tensorflow as tf

from audeep.backend.data.records import to_tensor
from audeep.backend.decorators import scoped_subgraph_initializers, scoped_subgraph


@scoped_subgraph_initializers
class SpectrogramQueue:
    """
    An input queue for spectrogram data.
    
    The input queue reads data from TFRecord files, which contain two-dimensional feature matrices for each instance.
    The first dimension of these feature matrices is assumed to correspond to the time-axis of the data. The last 
    dimension is assumed to correspond to the feature dimension, for example, to the frequency dimension of 
    spectrograms.
    """

    def __init__(self,
                 record_files: Iterable[Path],
                 feature_shape: Sequence[int],
                 batch_size: int):
        """
        Create and initialize a SpectrogramQueue with the specified parameters.
        
        Parameters
        ----------
        record_files: iterable of Path
            A collection of TFRecords files from which instances should be read
        feature_shape: list of int
            Shape of the feature matrices
        batch_size: int
            The batch size in which to read instances
        """
        if len(feature_shape) != 2:
            raise ValueError("feature shape must be two-dimensional")

        self._batch_size = batch_size
        self._record_files = record_files
        self._feature_shape = feature_shape

        # initialize computation graph
        self.init_input_queue()

    @scoped_subgraph
    def input_queue(self) -> tf.Tensor:
        """
        Creates the input queue.
        
        Returns
        -------
        tf.Tensor
            A tensor containing batched instances read from the TFRecords file passed to this class
        """
        filename_tensor = tf.constant(value=[str(file) for file in self._record_files], dtype=tf.string,
                                      name="filenames")
        filename_queue = tf.train.string_input_producer(filename_tensor, num_epochs=None)

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        data = to_tensor(serialized_example, self._feature_shape)

        data_batches = tf.train.shuffle_batch([data],
                                              enqueue_many=False,
                                              batch_size=self._batch_size,
                                              num_threads=2,
                                              capacity=100 * self._batch_size,
                                              min_after_dequeue=50 * self._batch_size,
                                              allow_smaller_final_batch=True)

        # batches are batch major; also we assume that features are stored as [time, frequency]
        data_batches = tf.transpose(data_batches, perm=[1, 0, 2])

        # [time, batch, frequency]
        return data_batches
