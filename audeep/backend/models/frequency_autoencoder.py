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

"""A frequency autoencoder derived from the frequency-RNN of F-T-RNNs"""
from typing import Optional

import tensorflow as tf

from audeep.backend.decorators import scoped_subgraph_initializers, scoped_subgraph
from audeep.backend.models import summaries
from audeep.backend.models.ops import flatten_time, window_features, linear, restore_time
from audeep.backend.models.rnn_base import RNNArchitecture, StandardRNN, FeedPreviousRNN


@scoped_subgraph_initializers
class FrequencyAutoencoder:
    """
    A recurrent autoencoder which processes spectrogram data sequentially along the frequency axis.
    """

    def __init__(self,
                 encoder_architecture: RNNArchitecture,
                 decoder_architecture: RNNArchitecture,
                 mask_silence: bool,
                 frequency_window_width: int,
                 frequency_window_overlap: int,
                 inputs: tf.Tensor,
                 learning_rate: tf.Tensor,
                 keep_prob: Optional[tf.Tensor],
                 encoder_noise: Optional[tf.Tensor],
                 decoder_feed_previous_prob: Optional[tf.Tensor]):
        """
        Create and initialize a new FrequencyAutoencoder with the specified parameters.

        Parameters
        ----------
        encoder_architecture: RNNArchitecture
            The architecture of the encoder RNN
        decoder_architecture: RNNArchitecture
            The architecture of the decoder RNN
        mask_silence: bool
            Mask silence in the loss function (experimental)
        frequency_window_width: int
            Feed frequency vectors to the RNNs in windows of the specified width
        frequency_window_overlap: int
            Overlap between frequency windows
        inputs: tf.Tensor
            Tensor containing the input sequences, of shape [max_time, batch_size, num_features]
        learning_rate: tf.Tensor
            Scalar tensor containing the learning rate
        keep_prob: tf.Tensor, optional
            Scalar tensor containing the probability to keep activations in the RNNs
        encoder_noise: tf.Tensor, optional
            Scalar tensor containing the probability to corrupt the encoder inputs
        decoder_feed_previous_prob: tf.Tensor, optional
            Scalar tensor containing the probability at each time step to feed the previous output of the decoder as 
            input 
        """

        # inputs
        self.inputs = inputs
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.encoder_noise = encoder_noise
        self.decoder_feed_previous_prob = decoder_feed_previous_prob

        # network topology
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture
        self.mask_silence = mask_silence
        self.frequency_window_width = frequency_window_width
        self.frequency_window_overlap = frequency_window_overlap

        # initialize subgraphs
        self.init_targets()
        self.init_encoder_inputs()
        self.init_encoder()
        self.init_representation()
        self.init_decoder_inputs()
        self.init_decoder()
        self.init_reconstruction()
        self.init_loss()
        self.init_optimizer()

    @property
    def max_time(self) -> tf.Tensor:
        """
        Returns the number of time steps in the input sequences.

        This property may be unknown at graph construction time, thus a tensor containing the value is returned.

        Returns
        -------
        tf.Tensor
            The number of time steps in the input sequences
        """
        return tf.shape(self.inputs)[0]

    @property
    def batch_size(self) -> tf.Tensor:
        """
        Returns the number of input sequences per batch.

        This property may be unknown at graph construction time, thus a tensor containing the value is returned.

        Returns
        -------
        tf.Tensor
            The number of input sequences per batch
        """
        return tf.shape(self.inputs)[1]

    @property
    def num_features(self) -> int:
        """
        Returns the number of features in each time step of the input sequences.

        This property is known at graph-construction time.

        Returns
        -------
        int
            The number of features in each time step of the input sequence
        """
        return self.inputs.shape.as_list()[2]

    @scoped_subgraph
    def targets(self) -> tf.Tensor:
        """
        Returns the reconstruction target sequences.
        
        The target sequences are simply the input sequences reversed along the frequency axis.
        
        Returns
        -------
        tf.Tensor
            The target sequences, of shape [max_time, batch_size, num_features]
        """
        return self.inputs[:, :, ::-1]

    @scoped_subgraph
    def encoder_inputs(self) -> tf.Tensor:
        """
        Returns the input sequences for the encoder.
        
        The encoder input sequences are built by splitting the input spectrograms into windows of width 
        `frequency_window_width` and overlap `frequency_window_overlap` along the frequency axis. These windows are then
        fed in order to the encoder RNN.
        
        Returns
        -------
        tf.Tensor
            The input sequences for the encoder
        """
        # shape: [max_time * batch_size, num_features]
        inputs_flat = flatten_time(self.inputs)

        # shape: [num_windows, max_time * batch_size, window_width]
        return window_features(inputs=inputs_flat,
                               window_width=self.frequency_window_width,
                               window_overlap=self.frequency_window_overlap)

    @scoped_subgraph
    def encoder(self) -> StandardRNN:
        """
        Creates the encoder RNN.

        The encoder RNN receives the windowed frequency vectors as input, and is initialized with zero initial states.

        Returns
        -------
        StandardRNN
            The encoder RNN
        """
        return StandardRNN(architecture=self.encoder_architecture,
                           inputs=self.encoder_inputs,
                           initial_state=None,
                           keep_prob=self.keep_prob,
                           input_noise=self.encoder_noise)

    @scoped_subgraph
    def representation(self) -> tf.Tensor:
        """
        Computes the hidden representation of the input sequences.

        The hidden representation of an input sequence is computed by applying a linear transformation with hyperbolic
        tangent activation to the final state of the encoder. The output size of the linear transformation matches the
        state vector size of the decoder.

        Returns
        -------
        tf.Tensor
            The hidden representation of the input sequences, of shape [batch_size, decoder_state_size]
        """

        # shape: [max_time * batch_size, encoder.state_size]
        internal_rep = tf.tanh(linear(self.encoder.final_state, self.decoder_architecture.state_size))

        # shape: [max_time, batch_size, encoder.state_size]
        rep = tf.reshape(internal_rep,
                         [self.max_time, self.batch_size, self.encoder_architecture.state_size])
        # shape: [batch_size, max_time, encoder.state_size]
        rep = tf.transpose(rep, perm=[1, 0, 2])

        tf.add_to_collection("representation", rep)
        summaries.variable_summaries(rep)

        return internal_rep

    @scoped_subgraph
    def decoder_inputs(self) -> tf.Tensor:
        """
        Computes the decoder RNN input sequences.

        At each time step, the input to the decoder RNN is the expected output at the previous frequency step. Thus, the
        decoder input sequences are the encoder input sequences shifted by one step along the frequency axis.

        Returns
        -------
        tf.Tensor
            The decoder RNN input sequences, of shape [num_windows, batch_size*max_time, window_width]
        """
        num_windows = self.encoder_inputs.shape.as_list()[0]
        decoder_inputs = self.encoder_inputs[::-1]
        decoder_inputs = decoder_inputs[:num_windows - 1]
        decoder_inputs = tf.pad(decoder_inputs, paddings=[[1, 0], [0, 0], [0, 0]], mode="constant")

        return decoder_inputs

    @scoped_subgraph
    def decoder(self) -> FeedPreviousRNN:
        """
        Creates the decoder RNN.
        
        The decoder RNN receives the decoder input sequence as input, and is initialized with the hidden representation
        as the initial state vector.
        
        Returns
        -------
        FeedPreviousRNN
            The decoder RNN
        """
        return FeedPreviousRNN(architecture=self.decoder_architecture,
                               inputs=self.decoder_inputs,
                               initial_state=self.representation,
                               keep_prob=self.keep_prob,
                               feed_previous_prob=self.decoder_feed_previous_prob)

    @scoped_subgraph
    def reconstruction(self) -> tf.Tensor:
        """
        Computes the reconstruction of the input sequence.
        
        Although an output projection is implicitly added by the decoder RNN, another linear transformation of the 
        decoder output is performed. This is done to preserve consistency of models between the unidirectional and
        bidirectional cases, since in the latter case the decoder output sequence has twice as many features as the
        input sequence. Thus, another linear projection is required in the bidirectional case.
        
        Returns
        -------
        tf.Tensor
            The reconstruction of the input sequence, of shape [max_time, batch_size, num_features]
        """

        # shape: [num_windows, max_time * batch_size, decoder_frequency.output_size]
        decoder_output = self.decoder.output
        num_windows = decoder_output.shape.as_list()[0]  # must be known at graph-construction time

        # shape: [max_time * batch_size, num_windows, decoder_frequency.output_size]
        decoder_output = tf.transpose(decoder_output, [1, 0, 2])
        # shape: [max_time * batch_size, num_windows * decoder_frequency.output_size]
        decoder_output = tf.reshape(decoder_output,
                                    [self.max_time * self.batch_size,
                                     num_windows * self.decoder.output_size])

        # shape: [max_time * batch_size, num_features]
        reconstruction = tf.tanh(linear(decoder_output,
                                        output_size=self.num_features))

        reconstruction = restore_time(inputs=reconstruction,
                                      max_time=self.max_time,
                                      batch_size=self.batch_size,
                                      num_features=self.num_features)

        tf.add_to_collection("reconstruction", reconstruction)
        summaries.reconstruction_summaries(reconstruction, self.targets)

        return reconstruction

    @scoped_subgraph
    def loss(self) -> tf.Tensor:
        """
        Computes the reconstruction loss of the autoencoder.

        The reconstruction loss is computed as the root mean square error between the target sequence and the 
        reconstructed sequence.

        Returns
        -------
        tf.Tensor
            Scalar tensor containing the reconstruction loss averaged over the entire input batch
        """
        reconstruction = self.reconstruction

        if self.mask_silence:
            reconstruction = tf.where(self.targets == -1., -tf.ones_like(reconstruction), reconstruction)

        loss = tf.sqrt(tf.reduce_mean(tf.square(self.targets - reconstruction)))
        summaries.scalar_summaries(loss)

        tf.add_to_collection(tf.GraphKeys.LOSSES, loss)

        return loss

    @scoped_subgraph
    def optimizer(self) -> tf.Operation:
        """
        Creates the optimization operation used for training the autoencoder.

        Gradient clipping of values outside [-2;2] is automatically applied to prevent exploding gradients.

        Returns
        -------
        tf.Operation
            The optimization operation used for training the autoencoder
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(self.loss)

        with tf.variable_scope("clip_gradients"):
            capped_gvs = [(grad, var) if grad is None else (tf.clip_by_value(grad, -2., 2.), var) for grad, var in
                          gvs]

        train_op = optimizer.apply_gradients(capped_gvs)

        tf.add_to_collection("train_op", train_op)

        return train_op
