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

"""Helpers for converting instances to Example protobufs and vice-versa"""
from typing import Iterable

import numpy as np
import tensorflow as tf

_DATA_RAW_KEY = "data_raw"


def _bytes_feature(value: bytes) -> tf.train.Feature:
    """
    Create a Feature protobuf containing the specified bytes.
    
    Parameters
    ----------
    value: bytes
        The value of the protobuf

    Returns
    -------
    tf.train.Feature
        A Feature protobuf containing the specified bytes
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def to_example(data: np.ndarray) -> tf.train.Example:
    """
    Create an Example protobuf from the specified feature matrix.
    
    Parameters
    ----------
    data: numpy.ndarray
        The feature matrix which should be converted to an Example protobuf

    Returns
    -------
    tf.train.Example
        An Example protobuf containing the specified feature matrix
    """
    return tf.train.Example(features=tf.train.Features(feature={
        _DATA_RAW_KEY: _bytes_feature(data.astype(np.float32).flatten().tobytes()),
    }))


def to_tensor(serialized_example: tf.Tensor,
              shape: Iterable[int]) -> tf.Tensor:
    """
    Creates a deserialization operation for Example protobufs.
    
    Parameters
    ----------
    serialized_example: tf.Tensor
        Tensor containing serialized Example protobufs
    shape: list of int
        The shape of the feature matrices in the Example protobufs.

    Returns
    -------
    tf.Tensor
        The deserialized feature matrix
    """
    features = tf.parse_single_example(
        serialized_example,
        features={
            _DATA_RAW_KEY: tf.FixedLenFeature([], tf.string),
        })

    data = tf.decode_raw(features[_DATA_RAW_KEY], tf.float32)
    data = tf.reshape(data, shape)

    return data
