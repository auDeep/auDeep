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

"""Tensorflow summary operations"""
import tensorflow as tf


def variable_summaries(tensor: tf.Tensor):
    """
    Creates several summary operations for a multidimensional tensor.
    
    This operation adds summary operations for the mean and standard deviation of the values in the input tensor, as
    well as a histogram of the values.
    
    Parameters
    ----------
    tensor: tf.Tensor
        The tensor for which summary operations should be created
    """
    mean = tf.reduce_mean(tensor)
    tf.summary.scalar("mean", mean)
    with tf.name_scope("stddev"):
        # noinspection PyUnresolvedReferences
        stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
    tf.summary.scalar("stddev", stddev)
    tf.summary.histogram("histogram", tensor)


def scalar_summaries(tensor: tf.Tensor,
                     name: str = "value"):
    """
    Creates a summary operation for the value of a scalar tensor.
    
    Parameters
    ----------
    tensor: tf.Tensor
        The (scalar) tensor for which a summary operation should be created
    name: str, optional
        A name for the summary operation (default "value")
    """
    tf.summary.scalar(name, tensor)


def reconstruction_summaries(reconstruction: tf.Tensor,
                             targets: tf.Tensor,
                             name: str = "targets_vs_reconstruction"):
    """
    Creates an image summary for the reconstruction of a target sequence.
    
    The `reconstruction` and `targets` parameters must both be tensors with shape [max_time, batch_size, num_features].
    The sequences are concatenated along the feature dimension, and several instances from the batch are plotted as
    images.
    
    Parameters
    ----------
    reconstruction: tf.Tensor
        The reconstructed sequences with shape [max_time, batch_size, num_features]
    targets: tf.Tensor
        The reconstruction target sequences with shape [max_time, batch_size, num_features]
    name: str, optional
        A name for the summary operation (default "targets_vs_reconstruction")
    """
    # concat targets and reconstruction along feature dimension
    images = tf.concat([targets, reconstruction], axis=2)
    images = tf.transpose(images, perm=[1, 0, 2])
    images = tf.expand_dims(images, axis=3)
    tf.summary.image(name, images)


def image_summaries(images: tf.Tensor,
                    max_outputs: int = 3,
                    name: str = "image"):
    """
    Creates an image summary operation for a tensor containing image data.
    
    Parameters
    ----------
    images: tf.Tensor
        A tensor containing image data, with shape [batch_size, height, width, channels]
    max_outputs: int, optional
        The number of images to show (default 3)
    name: str, optional
        A name for the summary operation (default "image")
    """
    tf.summary.image(name, images, max_outputs)
