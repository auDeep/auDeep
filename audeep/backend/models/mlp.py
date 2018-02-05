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

"""Tensorflow model for a simple multilayer perceptron for classification"""
import tensorflow as tf


class MLPModel:
    def __init__(self,
                 num_layers: int,
                 num_hidden: int):
        self.num_layers = num_layers
        self.num_hidden = num_hidden

    def inference(self,
                  inputs: tf.Tensor,
                  keep_prob: float,
                  num_classes: int) -> tf.Tensor:
        num_features = inputs.shape[1]

        layer_input = inputs

        for layer in range(self.num_layers):
            with tf.variable_scope("layer_{}".format(layer + 1)):
                in_dim = num_features if layer == 0 else self.num_hidden
                out_dim = self.num_hidden

                weights = tf.get_variable("weights", shape=[in_dim, out_dim], dtype=tf.float32)
                bias = tf.get_variable("bias", shape=[out_dim])

                layer_input = tf.nn.relu(tf.matmul(layer_input, weights) + bias)
                layer_input = tf.nn.dropout(layer_input, keep_prob)

        with tf.variable_scope("output"):
            weights = tf.get_variable("weights",
                                      shape=[self.num_hidden if self.num_layers > 0 else num_features, num_classes],
                                      dtype=tf.float32)
            bias = tf.get_variable("bias", shape=[num_classes])

            output = tf.nn.relu(tf.matmul(layer_input, weights) + bias)

        return output

    def prediction(self,
                   logits: tf.Tensor) -> tf.Tensor:
        return tf.argmax(logits, axis=1)

    def loss(self,
             logits: tf.Tensor,
             targets: tf.Tensor) -> tf.Tensor:
        with tf.variable_scope("loss"):
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))

    def optimize(self,
                 loss: tf.Tensor,
                 learning_rate: float) -> tf.Operation:
        with tf.variable_scope("optimize"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            return optimizer.minimize(loss, name="train_op")
