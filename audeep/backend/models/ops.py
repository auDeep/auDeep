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

"""Library of commonly used Tensorflow operations"""
import logging
from typing import Union, Sequence, Optional

import tensorflow as tf
from tensorflow.python.ops.init_ops import Initializer


def lrelu(input: tf.Tensor,
          leak: float = 0.2,
          name: str = "lrelu") -> tf.Tensor:
    """
    Applies leaky rectified linear activation to a tensor.
    
    Parameters
    ----------
    input: tf.Tensor
        The input tensor of arbitrary shape
    leak: float, optional
        The slope of the activation function for negative values (default 0.2)
    name: str, optional
        A name for the operation (default "lrelu")

    Returns
    -------
    tf.Tensor
        Leaky rectified linear activation of the input tensor
    """
    with tf.variable_scope(scope=name):
        # noinspection PyTypeChecker
        return tf.maximum(input, input * leak,
                          name=name)


def batch_norm(input: tf.Tensor,
               is_training: Union[bool, tf.Tensor] = True,
               epsilon: float = 1e-5,
               momentum: float = 0.9,
               name: str = "batch_norm") -> tf.Tensor:
    """
    Applies batch normalization to a tensor.
    
    Parameters
    ----------
    input: tf.Tensor
        The tensor to which batch normalization should be applied, with shape [batch_size, height, width, channels]
    is_training: bool, optional
        Whether the batch norm layer is in training mode (default True). In training mode, mean and variance are 
        accumulated using an exponential moving average, in evaluation mode these coefficients are used for batch 
        normalization.
    epsilon: float, optional
        Epsilon constant added to the variance to avoid dividing by zero (default 1e-5)
    momentum: float, optional
        Decay for the moving average (default 0.9)
    name: str, optional
        A name for the operation (default "batch_norm")

    Returns
    -------
    tf.Tensor
        The batch-normalized input tensor
    """
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_training,
                                        scope=name)


def linear(input: tf.Tensor,
           output_size: int,
           weight_initializer: Optional[Initializer] = None,
           bias_initializer: Optional[Initializer] = None,
           name: str = "linear") -> tf.Tensor:
    """
    Apply a linear transformation to a tensor.
    
    Parameters
    ----------
    input: tf.Tensor
        The tensor which should be linearly transformed
    output_size: int
        The desired output size of the linear transformation
    weight_initializer: tf.Initializer, optional
        A custom initializer for the weight matrix of the linear transformation
    bias_initializer: tf.Initializer, optional
        A custom initializer for the bias vector of the linear transformation
    name: str, optional
        A name for the operation (default "linear")

    Returns
    -------
    tf.Tensor
        The linearly transformed input tensor
    """
    shape = input.get_shape().as_list()

    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights",
                                  shape=[shape[-1], output_size],
                                  dtype=tf.float32,
                                  initializer=weight_initializer)

        bias = tf.get_variable(name="bias",
                               shape=[output_size],
                               initializer=bias_initializer)

        return tf.matmul(input, weights) + bias


def conv2d(input: tf.Tensor,
           output_dim: int,
           kernel_width: int = 5,
           kernel_height: int = 5,
           horizontal_stride: int = 2,
           vertical_stride: int = 2,
           weight_initializer: Optional[Initializer] = None,
           bias_initializer: Optional[Initializer] = None,
           name: str = "conv2d"):
    """
    Apply a 2D-convolution to a tensor.
    
    Parameters
    ----------
    input: tf.Tensor
        The tensor to which the convolution should be applied. Must be of shape [batch_size, height, width, channels]
    output_dim: int
        The number of convolutional filters
    kernel_width: int, optional
        The width of the convolutional filters (default 5)
    kernel_height: int, optional
        The height of the convolutional filters (default 5)
    horizontal_stride: int, optional
        The horizontal stride of the convolutional filters (default 2)
    vertical_stride: int, optional
        The vertical stride of the convolutional filters (default 2)
    weight_initializer: tf.Initializer, optional
        A custom initializer for the weight matrices of the filters
    bias_initializer: tf.Initializer, optional
        A custom initializer for the bias vectors of the filters
    name: str, optional
        A name for the operation (default "conv2d")

    Returns
    -------
    tf.Tensor
        The result of applying a 2D-convolution to the input tensor.
    """
    shape = input.get_shape().as_list()

    with tf.variable_scope(name):
        weights = tf.get_variable(name="weights",
                                  shape=[kernel_height, kernel_width, shape[-1], output_dim],
                                  initializer=weight_initializer)

        bias = tf.get_variable(name="bias",
                               shape=[output_dim],
                               initializer=bias_initializer)

        conv = tf.nn.conv2d(input,
                            filter=weights,
                            strides=[1, vertical_stride, horizontal_stride, 1],
                            padding='SAME')

        conv = tf.nn.bias_add(conv, bias)

        return conv


def deconv2d(input: tf.Tensor,
             output_shape: Sequence[Union[int, tf.Tensor]],
             kernel_width: int = 5,
             kernel_height: int = 5,
             horizontal_stride: int = 2,
             vertical_stride: int = 2,
             weight_initializer: Optional[Initializer] = None,
             bias_initializer: Optional[Initializer] = None,
             name: str = "deconv2d"):
    """
    Applies a 2D-deconvolution to a tensor.
    
    Parameters
    ----------
    input: tf.Tensor
        The tensor to which a 2D-deconvolution should be applied. Must be of shape [batch_size, height, width, channels]
    output_shape: list of int or tf.Tensor
        The desired output shape.
    kernel_width: int, optional
        The width of the convolutional filters (default 5)
    kernel_height: int, optional
        The height of the convolutional filters (default 5)
    horizontal_stride: int, optional
        The horizontal stride of the convolutional filters (default 2)
    vertical_stride: int, optional
        The vertical stride of the convolutional filters (default 2)
    weight_initializer: tf.Initializer, optional
        A custom initializer for the weight matrices of the filters
    bias_initializer: tf.Initializer, optional
        A custom initializer for the bias vectors of the filters
    name: str, optional
        A name for the operation (default "deconv2d")

    Returns
    -------
    tf.Tensor
        The result of applying a 2D-deconvolution to the input tensor
    """
    shape = input.get_shape().as_list()

    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        weights = tf.get_variable(name="weights",
                                  shape=[kernel_height, kernel_width, output_shape[-1], shape[-1]],
                                  initializer=weight_initializer)

        biases = tf.get_variable(name="bias",
                                 shape=[output_shape[-1]],
                                 initializer=bias_initializer)

        deconv = tf.nn.conv2d_transpose(input,
                                        filter=weights,
                                        output_shape=output_shape,
                                        strides=[1, vertical_stride, horizontal_stride, 1])

        deconv = tf.nn.bias_add(deconv, biases)
        deconv.set_shape([None] + output_shape[1:])

        return deconv


def flatten_time(inputs: tf.Tensor):
    """
    Flattens the time dimension of a tensor representing time-series data.
    
    This operation reshapes the input tensor in such a way that time steps in the original tensor are converted to
    individual batches, so that the same operation can be applied to all time steps.
    
    Parameters
    ----------
    inputs: tf.Tensor
        The tensor which should be flattened, of shape [max_time, batch_size, num_features]

    Returns
    -------
    tf.Tensor
        The input tensor with flattened time dimension, of shape [max_time * batch_size, num_features]
    """
    max_time, batch_size, _ = tf.unstack(tf.shape(inputs))
    num_features = inputs.shape[2].value

    return tf.reshape(inputs, shape=[max_time * batch_size, num_features], name="flatten_time")


def restore_time(inputs: tf.Tensor,
                 max_time: tf.Tensor,
                 batch_size: tf.Tensor,
                 num_features: int) -> tf.Tensor:
    """
    Restores the time dimension of tensor representing time-series data in which the time dimension has been flattened.
    
    Parameters
    ----------
    inputs: tf.Tensor
        The tensor of which the time dimension should be restored, of shape [max_time * batch_size, num_features]
    max_time: tf.Tensor
        Tensor containing the desired number of time steps
    batch_size: tf.Tensor
        Tensor containing the batch size
    num_features: int
        The number of features in the input tensor. Required since the reshaping operation will otherwise lose the
        static shape information

    Returns
    -------
    tf.Tensor
        The input tensor with restored time dimension, of shape [max_time, batch_size, num_features]
    """
    return tf.reshape(inputs, shape=[max_time, batch_size, num_features], name="restore_time")


def time_distributed_linear(inputs: tf.Tensor,
                            output_size: int,
                            weight_initializer: Optional[Initializer] = None,
                            bias_initializer: Optional[Initializer] = None,
                            name: str = "time_dist_linear") -> tf.Tensor:
    """
    Applies the same linear transformation to all time steps of a sequence.
    
    Parameters
    ----------
    inputs: tf.Tensor
        The input sequences, of shape [max_time, batch_size, num_features]
    output_size: int
        The desired number of features in the output sequences
    weight_initializer: tf.Initializer, optional
        A custom initializer for the weight matrix of the linear transformation
    bias_initializer: tf.Initializer, optional
        A custom initializer for the bias vector of the linear transformation
    name: str, optional
        A name for the operation (default "time_dist_linear")

    Returns
    -------
    tf.Tensor
        The linearly transformed input sequences, of shape [max_time, batch_size, output_size]
    """
    max_time, batch_size, _ = tf.unstack(tf.shape(inputs))
    static_shape = inputs.shape.as_list()

    with tf.variable_scope(name):
        result = flatten_time(inputs)
        result = linear(result,
                        output_size=output_size,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer)
        result = restore_time(result, max_time, batch_size, output_size)
        result.set_shape([static_shape[0], static_shape[1], output_size])

        return result


def rank(tensor: tf.Tensor) -> int:
    """
    Returns the rank of a tensor.
    
    Parameters
    ----------
    tensor: tf.Tensor
        The tensor of which the rank should be computed
        
    Returns
    -------
    int 
        The rank of the input tensor
    """
    return tensor.shape.ndims


def window_features(inputs: tf.Tensor,
                    window_width: int,
                    window_overlap: int,
                    name: str = "window_features") -> tf.Tensor:
    """
    Performs windowing of the features in a tensor.
    
    Given an input tensor of shape [batch_size, num_features], the features of the tensor are split into windows with 
    width `window_width` and overlap `window_overlap`. The resulting tensors are then stacked along a new dimension,
    resulting in an output tensor of shape [num_windows, batch_size, window_width].
    
    Parameters
    ----------
    inputs: tf.Tensor
        The tensor of which features should be windowed, with shape [batch_size, num_features]
    window_width: int
        The number of features per window
    window_overlap: int
        The overlap between windows
    name: str, optional
        A name for the operation (default "window_features")

    Returns
    -------
    tf.Tensor
        A tensor containing the windowed features of the input tensor, with shape [num_windows, batch_size, window_width]
    """
    if rank(inputs) != 2:
        raise ValueError("rank of inputs must be 2 ([batch_size, num_features]), is: {}".format(rank(inputs)))

    log = logging.getLogger(__name__)
    num_features = inputs.shape.as_list()[1]  # must be known at graph-construction time

    with tf.variable_scope(scope=name):
        window_step = window_width - window_overlap
        num_windows = (num_features - window_width) // window_step + 1
        covered_features = window_width + (num_windows - 1) * window_step

        if covered_features != num_features:
            log.warning(
                "the last %d of %d features will be dropped for windows with width %d and overlap %d",
                num_features - covered_features, num_features, window_width, window_overlap)

        windows = [(window * window_step, window * window_step + window_width)
                   for window in range(num_windows)]
        window_tensors = [inputs[:, window_start:window_end] for window_start, window_end in windows]

        return tf.stack(window_tensors, axis=0)  # shape: [num_windows, batch_size, window_width]
