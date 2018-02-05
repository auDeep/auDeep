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

"""A time-recurrent autoencoder"""
from typing import Optional

import tensorflow as tf

from audeep.backend.decorators import scoped_subgraph_initializers, scoped_subgraph
from audeep.backend.log import LoggingMixin
from audeep.backend.models import summaries
from audeep.backend.models.ops import linear, time_distributed_linear
from audeep.backend.models.rnn_base import RNNArchitecture, StandardRNN, FeedPreviousRNN, _RNNBase


@scoped_subgraph_initializers
class TimeAutoencoder(LoggingMixin):
    """
    A recurrent autoencoder which processes time-series data sequentially along the time axis.
    """

    def __init__(self,
                 encoder_architecture: RNNArchitecture,
                 decoder_architecture: RNNArchitecture,
                 mask_silence: bool,
                 inputs: tf.Tensor,
                 learning_rate: tf.Tensor,
                 keep_prob: Optional[tf.Tensor] = None,
                 encoder_noise: Optional[tf.Tensor] = None,
                 decoder_feed_previous_prob: Optional[tf.Tensor] = None):
        """
        Create and initialize a new TimeAutoencoder with the specified parameters.
        
        Parameters
        ----------
        encoder_architecture: RNNArchitecture
            The architecture of the encoder RNN
        decoder_architecture: RNNArchitecture
            The architecture of the decoder RNN
        mask_silence: bool
            Mask silence in the loss function (experimental)
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
        super().__init__()

        # inputs
        self.inputs = inputs
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.encoder_noise = encoder_noise
        self.decoder_feed_previous_prob = decoder_feed_previous_prob

        # network topology
        self.mask_silence = mask_silence
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture

        # initialize subgraphs
        self.init_targets()
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
        
        The target sequences are simply the input sequences reversed along the time axis.
        
        Returns
        -------
        tf.Tensor
            The target sequences, of shape [max_time, batch_size, num_features]
        """
        return self.inputs[::-1]

    @scoped_subgraph
    def encoder(self) -> StandardRNN:
        """
        Creates the encoder RNN.
        
        The encoder RNN receives the input sequences passed to the autoencoder as input, and is initialized with zero
        initial states.
        
        Returns
        -------
        StandardRNN
            The encoder RNN
        """
        return StandardRNN(architecture=self.encoder_architecture,
                           inputs=self.inputs,
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
        rep = tf.tanh(linear(input=self.encoder.final_state,
                             output_size=self.decoder_architecture.state_size))

        tf.add_to_collection("representation", rep)
        summaries.variable_summaries(rep)

        return rep

    @scoped_subgraph
    def decoder_inputs(self) -> tf.Tensor:
        """
        Computes the decoder RNN input sequences.
        
        At each time step, the input to the decoder RNN is the expected output at the previous time step. Thus, the
        decoder input sequences are the target sequences shifted forward in time by one time step.
        
        Returns
        -------
        tf.Tensor
            The decoder RNN input sequences, of shape [max_time, batch_size, num_features]
        """
        # noinspection PyTypeChecker
        decoder_inputs = self.targets[:self.max_time - 1]
        decoder_inputs = tf.pad(decoder_inputs, paddings=[[1, 0], [0, 0], [0, 0]], mode="constant")

        return decoder_inputs

    @scoped_subgraph
    def decoder(self) -> _RNNBase:
        """
        Creates the decoder RNN.
        
        The decoder RNN receives the decoder input sequence as input, and is initialized with the hidden representation
        as the initial state vector. 
        
        If the decoder architecture is bidirectional, we have currently disabled using a FeedPreviousRNN due to possibly
        lower model performance. A suitable warning will be emitted to notify users of this behavior.
        
        Returns
        -------
        _RNNBase
            The decoder RNN
        """
        if self.decoder_architecture.bidirectional:
            self.log.warn("'decoder_feed_previous_prob' set on bidirectional decoder will be ignored. If you have set "
                          "--decoder-feed-prob 0, or omitted the option, the network will behave as expected and you "
                          "can safely ignore this warning.")

            # Make sure that decoder_feed_previous_prob stays in the computation graph, otherwise TensorFlow will
            # complain that we pass decoder_feed_previous_prob in input map although it is not in the graph. This is
            # somewhat questionable behavior on TensorFlow's side, which we just have to live with.
            add = tf.add(self.decoder_feed_previous_prob, 1)

            with tf.control_dependencies([add]):
                return StandardRNN(architecture=self.decoder_architecture,
                                   inputs=self.decoder_inputs,
                                   initial_state=self.representation,
                                   keep_prob=self.keep_prob,
                                   input_noise=None)
        else:
            return FeedPreviousRNN(architecture=self.decoder_architecture,
                                   inputs=self.decoder_inputs,
                                   initial_state=self.representation,
                                   keep_prob=self.keep_prob,
                                   feed_previous_prob=self.decoder_feed_previous_prob)

    @scoped_subgraph
    def reconstruction(self) -> tf.Tensor:
        """
        Computes the reconstruction of the input sequence.
        
        If the `decoder_feed_previous_prob` parameter is set, the decoder is a FeedPreviousRNN. In this case,
        although an output projection is implicitly added by the decoder RNN, another linear transformation of the 
        decoder output is performed. This is done to preserve consistency of models between the unidirectional and
        bidirectional cases, since in the latter case the decoder output sequence has twice as many features as the
        input sequence. Thus, another linear projection is required in the bidirectional case.
        
        Returns
        -------
        tf.Tensor
            The reconstruction of the input sequence, of shape [max_time, batch_size, num_features]
        """
        output = tf.tanh(time_distributed_linear(inputs=self.decoder.output,
                                                 output_size=self.num_features))

        tf.add_to_collection("reconstruction", output)
        summaries.reconstruction_summaries(output, self.targets)

        return output

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
