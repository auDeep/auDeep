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

"""A frequency-time autoencoder, using a time RNN stacked on top of a frequency RNN"""
from typing import Optional

import tensorflow as tf

from audeep.backend.decorators import scoped_subgraph_initializers, scoped_subgraph
from audeep.backend.models import summaries
from audeep.backend.models.ops import linear, flatten_time, restore_time, window_features
from audeep.backend.models.rnn_base import RNNArchitecture, StandardRNN, FeedPreviousRNN


@scoped_subgraph_initializers
class FrequencyTimeAutoencoder:
    """
    A recurrent autoencoder which first processes spectrogram data sequentially along the frequency axis, and then
    processes the frequency representations sequentially along the time axis.
    
    This is achieved by stacking a time-recurrent autoencoder on top of a frequency-recurrent autoencoder.
    """

    def __init__(self,
                 f_encoder_architecture: RNNArchitecture,
                 t_encoder_architecture: RNNArchitecture,
                 f_decoder_architecture: RNNArchitecture,
                 t_decoder_architecture: RNNArchitecture,
                 mask_silence: bool,
                 frequency_window_width: int,
                 frequency_window_overlap: int,
                 inputs: tf.Tensor,
                 learning_rate: tf.Tensor,
                 keep_prob: Optional[tf.Tensor],
                 f_encoder_noise: Optional[tf.Tensor],
                 t_encoder_noise: Optional[tf.Tensor],
                 f_decoder_feed_previous_prob: Optional[tf.Tensor],
                 t_decoder_feed_previous_prob: Optional[tf.Tensor]):
        """
        Create and initialize a new FrequencyTimeAutoencoder with the specified parameters.
        
        Parameters
        ----------
        f_encoder_architecture: RNNArchitecture
            Architecture of the frequency encoder RNN
        t_encoder_architecture: RNNArchitecture
            Architecture of the time encoder RNN
        f_decoder_architecture: RNNArchitecture
            Architecture of the frequency decoder RNN
        t_decoder_architecture: RNNArchitecture
            Architecture of the time decoder RNN
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
        f_encoder_noise
        t_encoder_noise
        f_decoder_feed_previous_prob
        t_decoder_feed_previous_prob
        """

        # inputs
        self.inputs = inputs
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.f_encoder_noise = f_encoder_noise
        self.t_encoder_noise = t_encoder_noise
        self.f_decoder_feed_previous_prob = f_decoder_feed_previous_prob
        self.t_decoder_feed_previous_prob = t_decoder_feed_previous_prob

        # architecture
        self.f_encoder_architecture = f_encoder_architecture
        self.t_encoder_architecture = t_encoder_architecture
        self.f_decoder_architecture = f_decoder_architecture
        self.t_decoder_architecture = t_decoder_architecture
        self.mask_silence = mask_silence
        self.frequency_window_width = frequency_window_width
        self.frequency_window_overlap = frequency_window_overlap

        # initialize subgraphs
        self.init_targets()
        self.init_encoder_frequency_inputs()
        self.init_encoder_frequency()
        self.init_encoder_time_inputs()
        self.init_encoder_time()
        self.init_representation()
        self.init_decoder_time_targets()
        self.init_decoder_time_inputs()
        self.init_decoder_time()
        self.init_decoder_frequency_inputs()
        self.init_decoder_frequency_initial_state()
        self.init_decoder_frequency()
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
        
        The target sequences are simply the input sequences reversed along the time and frequency axes.
        
        Returns
        -------
        tf.Tensor
            The target sequences, of shape [max_time, batch_size, num_features]
        """
        return self.inputs[::-1, :, ::-1]

    @scoped_subgraph
    def encoder_frequency_inputs(self) -> tf.Tensor:
        """
        Returns the input sequences for the frequency encoder.
        
        The frequency encoder input sequences are built by splitting the input spectrograms into windows of width 
        `frequency_window_width` and overlap `frequency_window_overlap` along the frequency axis. These windows are then
        fed in order to the encoder RNN.
        
        Returns
        -------
        tf.Tensor
            The input sequences for the frequency encoder
        """

        # shape: [max_time * batch_size, num_features]
        inputs_flat = flatten_time(self.inputs)

        # shape: [num_windows, max_time * batch_size, window_width]
        return window_features(inputs=inputs_flat,
                               window_width=self.frequency_window_width,
                               window_overlap=self.frequency_window_overlap)

    @scoped_subgraph
    def encoder_frequency(self) -> StandardRNN:
        """
        Creates the frequency encoder RNN.
        
        The frequency encoder RNN receives the windowed frequency vectors as input, and is initialized with zero 
        initial states.
        
        Returns
        -------
        StandardRNN
            The frequency encoder RNN
        """
        return StandardRNN(architecture=self.f_encoder_architecture,
                           inputs=self.encoder_frequency_inputs,
                           initial_state=None,
                           keep_prob=self.keep_prob,
                           input_noise=self.f_encoder_noise)

    @scoped_subgraph
    def encoder_time_inputs(self) -> tf.Tensor:
        """
        Returns the input sequences for the time encoder.
        
        The time encoder input sequences are built by stacking the final hidden states of the frequency encoder at each
        time step of a sequence along the time dimension.
        
        Returns
        -------
        tf.Tensor
            The input sequences for the time encoder
        """

        # shape: [max_time * batch_size, encoder_frequency.state_size]
        encoder_frequency_final_state = self.encoder_frequency.final_state

        return restore_time(inputs=encoder_frequency_final_state,
                            max_time=self.max_time,
                            batch_size=self.batch_size,
                            num_features=self.f_encoder_architecture.state_size)

    @scoped_subgraph
    def encoder_time(self) -> StandardRNN:
        """
        Creates the time encoder RNN.
        
        The time encoder RNN receives the final hidden states of the frequency encoder at each time step of a sequence 
        as input, and is initialized with zero initial states.
        
        Returns
        -------
        StandardRNN
            The time encoder RNN
        """
        return StandardRNN(architecture=self.t_encoder_architecture,
                           inputs=self.encoder_time_inputs,
                           initial_state=None,
                           keep_prob=self.keep_prob,
                           input_noise=self.t_encoder_noise)

    @scoped_subgraph
    def representation(self) -> tf.Tensor:
        """
        Computes the hidden representation of the input sequences.
        
        The hidden representation of an input sequence is computed by applying a linear transformation with hyperbolic
        tangent activation to the final state of the time encoder. The output size of the linear transformation matches 
        the state vector size of the time decoder.
        
        Returns
        -------
        tf.Tensor
            The hidden representation of the input sequences, of shape [batch_size, time_decoder_state_size]
        """

        representation = tf.tanh(linear(input=self.encoder_time.final_state,
                                        output_size=self.t_decoder_architecture.state_size))

        tf.add_to_collection("representation", representation)
        summaries.variable_summaries(representation)

        return representation

    @scoped_subgraph
    def decoder_time_targets(self) -> tf.Tensor:
        """
        Target sequence for the time decoder.
        
        The target sequence for the time decoder is the output of the frequency encoder RNN, reversed along the time
        axis.
        
        Returns
        -------
        tf.Tensor
            The target sequence for the time decoder
        """
        return self.encoder_time_inputs[::-1]

    @scoped_subgraph
    def decoder_time_inputs(self) -> tf.Tensor:
        """
        Input sequence for the time decoder.
        
        At each time step, the input to the time decoder RNN is the expected output at the previous frequency step. 
        Thus, the time decoder input sequences are the time decoder target sequences shifted by one step along the 
        time axis.

        Returns
        -------
        tf.Tensor
            The input sequences for the time decoder
        """
        # shape: [max_time, batch_size, encoder_frequency.state_size]
        decoder_time_inputs = self.decoder_time_targets
        decoder_time_inputs = decoder_time_inputs[:self.max_time - 1]
        decoder_time_inputs = tf.pad(decoder_time_inputs, paddings=[[1, 0], [0, 0], [0, 0]], mode="constant")

        return decoder_time_inputs

    @scoped_subgraph
    def decoder_time(self) -> FeedPreviousRNN:
        """
        Creates the time decoder RNN.
        
        The time decoder RNN receives the time decoder input sequence as input, and is initialized with the hidden 
        representation as the initial state vector.
        
        Returns
        -------
        FeedPreviousRNN
            The time decoder RNN
        """
        return FeedPreviousRNN(architecture=self.t_decoder_architecture,
                               inputs=self.decoder_time_inputs,
                               initial_state=self.representation,
                               keep_prob=self.keep_prob,
                               feed_previous_prob=self.t_decoder_feed_previous_prob)

    @scoped_subgraph
    def decoder_frequency_initial_state(self) -> tf.Tensor:
        """
        The initial states of the frequency decoder RNN.
        
        The outputs of the time decoder RNN at each time step are passed through a linear transformation layer with
        hyperbolic tangent activation, and used as the initial states of the frequency decoder RNN.
        
        Returns
        -------
        tf.Tensor
            The initial states of the frequency decoder RNN
        """
        # shape: [max_time, batch_size, decoder_time.output_size]
        decoder_frequency_initial_state = self.decoder_time.output
        # shape: [max_time * batch_size, decoder_time.output_size]
        decoder_frequency_initial_state = flatten_time(decoder_frequency_initial_state)

        decoder_frequency_initial_state = tf.tanh(linear(decoder_frequency_initial_state,
                                                         output_size=self.f_decoder_architecture.state_size))

        return decoder_frequency_initial_state

    @scoped_subgraph
    def decoder_frequency_inputs(self) -> tf.Tensor:
        """
        Computes the frequency decoder RNN input sequences.
        
        At each time step, the input to the frequency decoder RNN is the expected output at the previous frequency step. 
        Thus, the decoder input sequences are the frequency encoder input sequences shifted by one step along the 
        frequency axis.
        
        Returns
        -------
        tf.Tensor
           The frequency decoder RNN input sequences, of shape [num_windows, batch_size*max_time, window_width]
        """
        # shape: [max_time * batch_size, num_features]
        decoder_frequency_inputs = flatten_time(self.targets)
        # shape: [num_windows, max_time * batch_size, window_width]
        decoder_frequency_inputs = window_features(inputs=decoder_frequency_inputs,
                                                   window_width=self.frequency_window_width,
                                                   window_overlap=self.frequency_window_overlap)

        num_windows = decoder_frequency_inputs.shape.as_list()[0]

        decoder_frequency_inputs = decoder_frequency_inputs[:num_windows - 1, :, :]
        decoder_frequency_inputs = tf.pad(decoder_frequency_inputs, paddings=[[1, 0], [0, 0], [0, 0]], mode="constant")

        return decoder_frequency_inputs

    @scoped_subgraph
    def decoder_frequency(self) -> FeedPreviousRNN:
        """
        Creates the frequency decoder RNN.
        
        The frequency decoder RNN receives the decoder input sequence as input, and is initialized with the time decoder 
        output at each time step as the initial state vector.
        
        Returns
        -------
        FeedPreviousRNN
            The frequency decoder RNN
        """
        return FeedPreviousRNN(architecture=self.f_decoder_architecture,
                               inputs=self.decoder_frequency_inputs,
                               initial_state=self.decoder_frequency_initial_state,
                               keep_prob=self.keep_prob,
                               feed_previous_prob=self.f_decoder_feed_previous_prob)

    @scoped_subgraph
    def reconstruction(self):
        """
        Computes the reconstruction of the input sequence.
        
        Although an output projection is implicitly added by the decoder RNN, another linear transformation of the 
        frequency decoder output is performed. This is done to preserve consistency of models between the unidirectional 
        and bidirectional cases, since in the latter case the frequency decoder output sequence has twice as many 
        features as the input sequence. Thus, another linear projection is required in the bidirectional case.
        
        Returns
        -------
        tf.Tensor
            The reconstruction of the input sequence, of shape [max_time, batch_size, num_features]
        """

        # shape: [num_windows, max_time * batch_size, decoder_frequency.output_size]
        decoder_frequency_output = self.decoder_frequency.output
        num_windows = decoder_frequency_output.shape.as_list()[0]  # must be known at graph-construction time

        # shape: [max_time * batch_size, num_windows, decoder_frequency.output_size]
        decoder_frequency_output = tf.transpose(decoder_frequency_output, [1, 0, 2])
        # shape: [max_time * batch_size, num_windows * decoder_frequency.output_size]
        decoder_frequency_output = tf.reshape(decoder_frequency_output,
                                              [self.max_time * self.batch_size,
                                               num_windows * self.decoder_frequency.output_size])

        # shape: [max_time * batch_size, num_features]
        reconstruction = tf.tanh(linear(decoder_frequency_output,
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
