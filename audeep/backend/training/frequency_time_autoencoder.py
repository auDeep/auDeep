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

"""Implements a feature learning wrapper for frequency-time-autoencoders"""
from typing import Dict

import tensorflow as tf

from audeep.backend.models.frequency_time_autoencoder import FrequencyTimeAutoencoder
from audeep.backend.training.base import BaseFeatureLearningWrapper


class FrequencyTimeAutoencoderWrapper(BaseFeatureLearningWrapper):
    """
    Implementation of the `BaseFeatureLearningWrapper` interface for frequency-time-autoencoders.
    """

    def _create_model(self,
                      tf_inputs: tf.Tensor,
                      **kwargs):
        tf_learning_rate = tf.placeholder(name="learning_rate", shape=[], dtype=tf.float32)
        tf_keep_prob = tf.placeholder(name="keep_prob", shape=[], dtype=tf.float32)
        tf_f_encoder_noise = tf.placeholder(name="f_encoder_noise", shape=[], dtype=tf.float32)
        tf_t_encoder_noise = tf.placeholder(name="t_encoder_noise", shape=[], dtype=tf.float32)
        tf_f_decoder_feed_previous_prob = tf.placeholder(name="f_decoder_feed_previous_prob", shape=[],
                                                         dtype=tf.float32)
        tf_t_decoder_feed_previous_prob = tf.placeholder(name="t_decoder_feed_previous_prob", shape=[],
                                                         dtype=tf.float32)

        with tf.variable_scope("model"):
            FrequencyTimeAutoencoder(f_encoder_architecture=kwargs["f_encoder_architecture"],
                                     t_encoder_architecture=kwargs["t_encoder_architecture"],
                                     f_decoder_architecture=kwargs["f_decoder_architecture"],
                                     t_decoder_architecture=kwargs["t_decoder_architecture"],
                                     frequency_window_width=kwargs["frequency_window_width"],
                                     frequency_window_overlap=kwargs["frequency_window_overlap"],
                                     inputs=tf_inputs,
                                     learning_rate=tf_learning_rate,
                                     keep_prob=tf_keep_prob,
                                     f_encoder_noise=tf_f_encoder_noise,
                                     t_encoder_noise=tf_t_encoder_noise,
                                     f_decoder_feed_previous_prob=tf_f_decoder_feed_previous_prob,
                                     t_decoder_feed_previous_prob=tf_t_decoder_feed_previous_prob)

    def _training_parameters(self, **kwargs) -> Dict[str, tf.Tensor]:
        return {
            "learning_rate": tf.constant(value=kwargs["learning_rate"],
                                         name="learning_rate"),
            "keep_prob": tf.constant(value=kwargs["keep_prob"],
                                     name="keep_prob"),
            "f_encoder_noise": tf.constant(value=kwargs["f_encoder_noise"],
                                           name="f_encoder_noise"),
            "t_encoder_noise": tf.constant(value=kwargs["t_encoder_noise"],
                                           name="t_encoder_noise"),
            "f_decoder_feed_previous_prob": tf.constant(value=kwargs["f_decoder_feed_previous_prob"],
                                                        name="f_decoder_feed_previous_prob"),
            "t_decoder_feed_previous_prob": tf.constant(value=kwargs["t_decoder_feed_previous_prob"],
                                                        name="t_decoder_feed_previous_prob"),
        }

    def _generation_parameters(self, **kwargs) -> Dict[str, tf.Tensor]:
        return {
            "learning_rate": tf.constant(value=0.0,
                                         name="learning_rate"),
            "keep_prob": tf.constant(value=1.0,
                                     name="keep_prob"),
            "f_encoder_noise": tf.constant(value=0.0,
                                           name="f_encoder_noise"),
            "t_encoder_noise": tf.constant(value=0.0,
                                           name="t_encoder_noise"),
            "f_decoder_feed_previous_prob": tf.constant(value=0.0,
                                                        name="f_decoder_feed_previous_prob"),
            "t_decoder_feed_previous_prob": tf.constant(value=0.0,
                                                        name="t_decoder_feed_previous_prob"),
        }
