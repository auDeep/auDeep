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

"""Implements a feature learning wrapper for frequency-autoencoders"""
from typing import Dict

import tensorflow as tf

from audeep.backend.models.frequency_autoencoder import FrequencyAutoencoder
from audeep.backend.training.base import BaseFeatureLearningWrapper


class FrequencyAutoencoderWrapper(BaseFeatureLearningWrapper):
    """
    Implementation of the `BaseFeatureLearningWrapper` interface for frequency-autoencoders.
    """

    def _create_model(self,
                      tf_inputs: tf.Tensor,
                      **kwargs):
        tf_learning_rate = tf.placeholder(name="learning_rate",
                                          shape=[],
                                          dtype=tf.float32)
        tf_keep_prob = tf.placeholder(name="keep_prob",
                                      shape=[],
                                      dtype=tf.float32)
        tf_encoder_noise = tf.placeholder(name="encoder_noise",
                                          shape=[],
                                          dtype=tf.float32)
        tf_decoder_feed_previous_prob = tf.placeholder(name="decoder_feed_previous_prob",
                                                       shape=[],
                                                       dtype=tf.float32)

        with tf.variable_scope("model"):
            FrequencyAutoencoder(encoder_architecture=kwargs["encoder_architecture"],
                                 decoder_architecture=kwargs["decoder_architecture"],
                                 frequency_window_width=kwargs["frequency_window_width"],
                                 frequency_window_overlap=kwargs["frequency_window_overlap"],
                                 inputs=tf_inputs,
                                 learning_rate=tf_learning_rate,
                                 keep_prob=tf_keep_prob,
                                 encoder_noise=tf_encoder_noise,
                                 decoder_feed_previous_prob=tf_decoder_feed_previous_prob)

    def _training_parameters(self, **kwargs) -> Dict[str, tf.Tensor]:
        return {
            "learning_rate": tf.constant(value=kwargs["learning_rate"],
                                         name="learning_rate"),
            "keep_prob": tf.constant(value=kwargs["keep_prob"],
                                     name="keep_prob"),
            "encoder_noise": tf.constant(value=kwargs["encoder_noise"],
                                         name="encoder_noise"),
            "decoder_feed_previous_prob": tf.constant(value=kwargs["decoder_feed_previous_prob"],
                                                      name="decoder_feed_previous_prob"),
        }

    def _generation_parameters(self, **kwargs) -> Dict[str, tf.Tensor]:
        return {
            "learning_rate": tf.constant(value=0.0,
                                         name="learning_rate"),
            "keep_prob": tf.constant(value=1.0,
                                     name="keep_prob"),
            "encoder_noise": tf.constant(value=0.0,
                                         name="encoder_noise"),
            "decoder_feed_previous_prob": tf.constant(value=0.0,
                                                      name="decoder_feed_previous_prob"),
        }
