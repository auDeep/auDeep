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

"""Common RNN functionality"""
import abc
from enum import Enum
from typing import Optional, List, Any

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple, DropoutWrapper, GRUCell, LSTMCell, MultiRNNCell

from audeep.backend.decorators import scoped_subgraph_initializers, scoped_subgraph
from audeep.backend.models.ops import flatten_time, restore_time


class CellType(Enum):
    """
    RNN cell type.
    """
    GRU = 1
    LSTM = 2


class RNNArchitecture:
    """
    Architecture of a RNN.
    """

    def __init__(self,
                 num_layers: int,
                 num_units: int,
                 bidirectional: bool,
                 cell_type: CellType):
        """
        Creates and initializes a new RNNArchitecture with the specified parameters.
        
        Parameters
        ----------
        num_layers: int
            The number of layers in the RNN
        num_units: int
            The number of RNN cells per layer
        bidirectional: bool
            Indicates whether the RNN is bidirectional
        cell_type: CellType
            The type of RNN cell to use
        """
        self.num_layers = num_layers
        self.num_units = num_units
        self.bidirectional = bidirectional
        self.cell_type = cell_type

    @property
    def state_size(self) -> int:
        """
        Returns the state size of an RNN with this architecture.
        
        Returns
        -------
        int
            The state size of an RNN with this architecture
        """
        size = self.num_units * self.num_layers

        if self.cell_type == CellType.LSTM:
            size *= 2

        if self.bidirectional:
            size *= 2

        return size


@scoped_subgraph_initializers
class _RNNBase:
    """
    Base class for all RNN implementations.
    
    RNNs used in this application are constrained to a common interface, which defines the network inputs and outputs,
    as well as some parameters for controlling the RNN behavior.
    
    Currently, we make the simplifying assumption that RNNs only have to deal with 1-D input sequences of equal length. 
    This assumption might be dropped in future versions of this application. Optionally, an initial state can be passed 
    to an RNN. This initial state must be a single vector of correct size according to the `state_size` property of the 
    RNN architecture. Internally, this state vector is split and passed to the individual RNN cells. Furthermore, an
    optional dropout probability can be passed to a RNN.
    
    Our RNNs expose both the output sequence and the final hidden state. If the RNN is bidirectional, the output vector
    contains the outputs of the forward RNN and the backward RNN concatenated along the feature dimension. The final 
    state contains the concatenated hidden states of the RNN cells. This concatenation is performed in such a way that
    it represents the inverse of the splitting operation used on the initial states passed to the RNNs. That is, if the
    output final state vector of an RNN is passed as the initial state to an RNN with the same architecture, cells in
    the latter RNN will receive the same initial hidden state vector as the final hidden state vector of the
    corresponding cells in the former RNN.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 architecture: RNNArchitecture,
                 inputs: tf.Tensor,
                 initial_state: Optional[tf.Tensor],
                 keep_prob: Optional[tf.Tensor]):
        """
        Create and initialize a _RNNBase with the specified parameters.
        
        Parameters
        ----------
        architecture: RNNArchitecture
            The RNN architecture.
        inputs: tf.Tensor
            Input sequences to this RNN of shape [max_step, batch_size, num_features]
        initial_state: tf.Tensor, optional
            Initial hidden states of the RNN cells of shape [batch_size, state_size]
        keep_prob: tf.Tensor, optional
            Probability to keep hidden activations
        """
        # inputs
        self.inputs = inputs
        self.initial_state = initial_state
        self.keep_prob = keep_prob

        # network topology
        self.num_layers = architecture.num_layers
        self.num_units = architecture.num_units
        self.cell_type = architecture.cell_type
        self.bidirectional = architecture.bidirectional
        self.state_size = architecture.state_size

        # No initialization of subgraphs here. We do not know if subclasses have to initialize subgraphs in any
        # specific order.

    @property
    @abc.abstractmethod
    def output_size(self) -> int:
        """
        The number of features in the output sequence of this RNN.
        
        The `output` tensor of this RNN will have shape [max_step, batch_size, output_size]. This property can not be
        computed based on the RNN architecture, since RNNs might add an output projection layer or similar. This 
        property is known at graph-construction time.
        
        Returns
        -------
        int
            The number of features in the output sequence of this RNN
        """
        pass

    @property
    def max_step(self) -> tf.Tensor:
        """
        The number of steps in the input and output sequences.

        This property might be unknown at graph-construction time, and thus is returned as a tensor representing the
        value.
        
        Returns
        -------
        tf.Tensor
            The number of steps in the input and output sequences
        """
        return tf.shape(self.inputs)[0]

    @property
    def batch_size(self) -> tf.Tensor:
        """
        The batch size in the input and output sequences.
        
        This property might be unknown at graph-construction time, and thus is returned as a tensor representing the
        value.
        
        Returns
        -------
        tf.Tensor
            The batch size in the input and output sequences
        """
        return tf.shape(self.inputs)[1]

    @property
    def num_features(self) -> int:
        """
        The number of features in the input sequence.
        
        This property is known at graph-construction time.
        
        Returns
        -------
        int
            The number of features in the input sequence
        """
        return self.inputs.shape.as_list()[2]

    def _create_rnn_cell(self):
        """
        Creates a single RNN cell according to the architecture of this RNN.
        
        Returns
        -------
        rnn cell
            A single RNN cell according to the architecture of this RNN
        """
        keep_prob = 1.0 if self.keep_prob is None else self.keep_prob

        if self.cell_type == CellType.GRU:
            return DropoutWrapper(GRUCell(self.num_units), keep_prob, keep_prob)
        elif self.cell_type == CellType.LSTM:
            return DropoutWrapper(LSTMCell(self.num_units), keep_prob, keep_prob)
        else:
            raise ValueError("unknown cell type: {}".format(self.cell_type))

    def _create_cells(self) -> List[MultiRNNCell]:
        """
        Creates the multilayer-RNN cells required by the architecture of this RNN.
        
        Returns
        -------
        list of MultiRNNCell
            A list of MultiRNNCells containing one entry if the RNN is unidirectional, and two identical entries if the
            RNN is bidirectional
        """
        cells = [[self._create_rnn_cell()
                  for _ in range(self.num_layers)]
                 for _ in range(2 if self.bidirectional else 1)]

        return [MultiRNNCell(x) for x in cells]

    @scoped_subgraph
    def sequence_length(self) -> tf.Tensor:
        """
        Return a tensor representing the length of each sequence in the input batch.
        
        Currently, we assume equal length of all input sequences.
        
        Returns
        -------
        tf.Tensor
            A tensor of shape [batch_size] representing the length of each sequence in the input batch
        """
        # for now, assume equal-length sequences
        return tf.fill(dims=[self.batch_size], value=self.max_step)

    @scoped_subgraph
    def initial_states_tuple(self):
        """
        Create the initial state tensors for the individual RNN cells.
        
        If no initial state vector was passed to this RNN, all initial states are set to be zero. Otherwise, the initial
        state vector is split into a possibly nested tuple of tensors according to the RNN architecture. The return
        value of this function is structured in such a way that it can be passed to the `initial_state` parameter of the
        RNN functions in `tf.contrib.rnn`.
        
        Returns
        -------
        tuple of tf.Tensor
            A possibly nested tuple of initial state tensors for the RNN cells
        """
        if self.initial_state is None:
            initial_states = tf.zeros(shape=[self.batch_size, self.state_size], dtype=tf.float32)
        else:
            initial_states = self.initial_state

        initial_states = tuple(tf.split(initial_states, self.num_layers, axis=1))

        if self.bidirectional:
            initial_states = tuple([tf.split(x, 2, axis=1) for x in initial_states])
            initial_states_fw, initial_states_bw = zip(*initial_states)

            if self.cell_type == CellType.LSTM:
                initial_states_fw = tuple([LSTMStateTuple(*tf.split(lstm_state, 2, axis=1))
                                           for lstm_state in initial_states_fw])
                initial_states_bw = tuple([LSTMStateTuple(*tf.split(lstm_state, 2, axis=1))
                                           for lstm_state in initial_states_bw])

            initial_states = (initial_states_fw, initial_states_bw)
        else:
            if self.cell_type == CellType.LSTM:
                initial_states = tuple([LSTMStateTuple(*tf.split(lstm_state, 2, axis=1))
                                        for lstm_state in initial_states])

        return initial_states

    @abc.abstractmethod
    def rnn_output_and_state(self) -> (tf.Tensor, tf.Tensor):
        """
        Returns tensors representing the RNN output sequence and final state.
        
        If the RNN is bidirectional, the output sequence tensor contains the outputs of the forward and backward RNNs
        concatenated along the feature dimension.
        
        Returns
        -------
        output: tf.Tensor
            A tensor containing the RNN output sequence, of shape [max_step, batch_size, output_size]
        final_state: tf.Tensor
            A single tensor containing the final states of the RNN cells, of shape [batch_size, state_size]
        """
        pass

    @property
    def output(self) -> tf.Tensor:
        """
        Returns a tensor representing the output sequence of the RNN.
        
        If the RNN is bidirectional, the output sequence tensor contains the outputs of the forward and backward RNNs
        concatenated along the feature dimension.
        
        Returns
        -------
        tf.Tensor
            A tensor containing the RNN output sequence, of shape [max_step, batch_size, output_size]
        """
        return self.rnn_output_and_state[0]

    @property
    def final_state(self) -> tf.Tensor:
        """
        Returns a single tensor containing the final hidden states of the RNN cells.
        
        Returns
        -------
        tf.Tensor
            A single tensor containing the final states of the RNN cells, of shape [batch_size, state_size]
        """
        return self.rnn_output_and_state[1]


@scoped_subgraph_initializers
class StandardRNN(_RNNBase):
    """
    Default implementation of the `_RNNBase` interface.
    
    This class uses a straightforward, possibly bidirectional, RNN to process the input sequence. If desired, an input
    noise probability can be specified. If this probability is greater than zero, the features of each step in the
    input sequence will be set to zero with that probability.
    """

    def __init__(self,
                 architecture: RNNArchitecture,
                 inputs: tf.Tensor,
                 initial_state: Optional[tf.Tensor],
                 keep_prob: Optional[tf.Tensor],
                 input_noise: Optional[tf.Tensor]):
        """
        Create and initialize a StandardRNN with the specified parameters.
        
        Parameters
        ----------
        architecture: RNNArchitecture
            The RNN architecture.
        inputs: tf.Tensor
            Input sequences to this RNN of shape [max_step, batch_size, num_features]
        initial_state: tf.Tensor, optional
            Initial hidden states of the RNN cells of shape [batch_size, state_size]
        keep_prob: tf.Tensor, optional
            Probability to keep hidden activations
        input_noise: tf.Tensor, optional
            Probability to replace steps in the input sequence with zeros
        """
        super().__init__(architecture, inputs, initial_state, keep_prob)

        self.input_noise = input_noise

        # initialize subgraphs
        self.init_sequence_length()
        self.init_initial_states_tuple()
        self.init_noisy_inputs()
        self.init_rnn_output_and_state()

    @property
    def output_size(self) -> int:
        """
        The number of features in the output sequence of this RNN.
        
        If the RNN is unidirectional, the output sequence will contain one feature for each unit in the last RNN layer.
        If it is bidirectional, the output sequence will contain two features for each unit in the last RNN layer, one
        for the forward RNN, and one for the backward RNN.
        
        Returns
        -------
        int
            The number of features in the output sequence of this RNN.
            
        See Also
        --------
        _RNNBase.output_size
        """
        return self.num_units * (2 if self.bidirectional else 1)

    @scoped_subgraph
    def noisy_inputs(self) -> tf.Tensor:
        """
        Return the input sequence, with noise added according to the `input_noise` parameter.
        
        If the `input_noise` parameter is not set, this method simply returns the input sequence. Otherwise, return a 
        tensor in which each time step of the input sequence is randomly set to zeros with probability given by the
        `input_noise` parameter.
        
        Returns
        -------
        tf.Tensor
            The input sequence, with noise added according to the `input_noise` parameter
        """
        if self.input_noise is None:
            return self.inputs

        # drop entire time steps with probability self.noise
        randoms = tf.random_uniform([self.max_step, self.batch_size], minval=0, maxval=1)
        randoms = tf.stack([randoms] * self.num_features, axis=2)

        result = tf.where(randoms > self.input_noise, self.inputs, tf.zeros_like(self.inputs))

        return result

    @scoped_subgraph
    def rnn_output_and_state(self):
        cells = self._create_cells()

        if self.bidirectional:
            initial_states_fw, initial_states_bw = self.initial_states_tuple

            output, (final_state_fw,
                     final_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cells[0],
                                                                       cell_bw=cells[1],
                                                                       inputs=self.noisy_inputs,
                                                                       sequence_length=self.sequence_length,
                                                                       initial_state_fw=initial_states_fw,
                                                                       initial_state_bw=initial_states_bw,
                                                                       dtype=tf.float32,
                                                                       swap_memory=True,
                                                                       time_major=True)

            output = tf.concat(output, axis=2)

            # LSTM states are tuples -> need to be concatenated
            # axis is one because states have shape [batch_size, units]
            if self.cell_type == CellType.LSTM:
                final_state_fw = [tf.concat(lstm_tuple, axis=1) for lstm_tuple in final_state_fw]
                final_state_bw = [tf.concat(lstm_tuple, axis=1) for lstm_tuple in final_state_bw]

            final_state = [tf.concat(x, axis=1) for x in zip(final_state_fw, final_state_bw)]
        else:
            initial_states = self.initial_states_tuple

            output, final_state = tf.nn.dynamic_rnn(cell=cells[0],
                                                    inputs=self.noisy_inputs,
                                                    sequence_length=self.sequence_length,
                                                    initial_state=initial_states,
                                                    dtype=tf.float32,
                                                    swap_memory=True,
                                                    time_major=True)

            # LSTM states are tuples -> need to be concatenated
            # axis is one because states have shape [batch_size, units]
            if self.cell_type == CellType.LSTM:
                final_state = [tf.concat(lstm_tuple, axis=1) for lstm_tuple in final_state]

        final_state = tf.concat(final_state, axis=1)

        return output, final_state


@scoped_subgraph_initializers
class FeedPreviousRNN(_RNNBase):
    """
    An RNN in which, at each time step, the previous output of the RNN is used as input with a given probability.
    
    Since the output and the input of such an RNN must have equal shape, this class implicitly adds a linear output 
    projection layer with hyperbolic tangent activation and the correct size on top of the RNN outputs.
    """

    def __init__(self,
                 architecture: RNNArchitecture,
                 inputs: tf.Tensor,
                 initial_state: Optional[tf.Tensor],
                 keep_prob: Optional[tf.Tensor],
                 feed_previous_prob: Optional[tf.Tensor]):
        """
        Create and initialize a FeedPreviousRNN with the specified parameters.
        
        Parameters
        ----------
        architecture: RNNArchitecture
            The RNN architecture.
        inputs: tf.Tensor
            Input sequences to this RNN of shape [max_step, batch_size, num_features]
        initial_state: tf.Tensor, optional
            Initial hidden states of the RNN cells of shape [batch_size, state_size]
        keep_prob: tf.Tensor, optional
            Probability to keep hidden activations
        feed_previous_prob: tf.Tensor, optional
            Probability at each time step to feed the output of the previous time step as input to the RNN
        """
        super().__init__(architecture, inputs, initial_state, keep_prob)

        self.feed_previous_prob = feed_previous_prob

        # initialize subgraphs
        self.init_sequence_length()
        self.init_initial_states_tuple()
        self.init_rnn_output_and_state()

    @property
    def output_size(self):
        """
        The number of features in the output sequence of this RNN.
        
        Since an output projection layer is implicitly added on top of the RNN output, the number of output features is
        equal to the number of input features if the network is unidirectional, and equal to twice the number of input
        features if the network is bidirectional.
        
        Returns
        -------
        int
            The number of features in the output sequence of this RNN
        """
        return self.num_features * (2 if self.bidirectional else 1)

    def _feed_previous_rnn(self,
                           cell,
                           initial_state,
                           reverse: bool) -> (tf.Tensor, Any):
        """
        Implements a unidirectional RNN with stochastic feeding of previous RNN outputs.
        
        Since the required functionality is not directly supported by the `tf.contrib.rnn` module, this method
        implements a custom loop function used with the `raw_rnn` function.
        
        Parameters
        ----------
        cell
            The RNN cell to use
        initial_state
            A possibly nested tuple of initial states for the RNN cells
        reverse: bool
            Whether to reverse the input sequence

        Returns
        -------
        output: tf.Tensor
            The output sequence of the RNN after applying the output projection
        final_state: Any
            A possible nested tuple of final states of the RNN cells, with the same structure as the `initial_state` 
            tuple
        """
        # input sequence
        if reverse:
            input_sequence = tf.reverse(self.inputs, axis=[0])
        else:
            input_sequence = self.inputs

        # output projection
        weights = tf.get_variable(name="weights",
                                  shape=[self.num_units, self.num_features],
                                  dtype=tf.float32)
        bias = tf.get_variable(name="bias",
                               shape=[self.num_features])

        # feed mask
        if self.feed_previous_prob is None:
            feed_mask = tf.fill(dims=[self.max_step, self.batch_size], value=False)
        else:
            feed_mask = tf.random_uniform([self.max_step, self.batch_size], minval=0, maxval=1)
            feed_mask = feed_mask < self.feed_previous_prob

        def loop_fn_initial():
            initial_elements_finished = (0 >= self.sequence_length)

            initial_input = tf.zeros([self.batch_size, self.num_features], dtype=tf.float32)
            initial_cell_state = initial_state
            initial_cell_output = None
            initial_loop_state = None  # we don't need to pass any additional information

            return (initial_elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, cell_output, cell_state, previous_loop_state):
            elements_finished = (time >= self.sequence_length)

            next_input = tf.where(feed_mask[time - 1],
                                  x=tf.tanh(tf.matmul(cell_output, weights) + bias),
                                  y=input_sequence[time - 1])
            loop_state = None

            return (elements_finished,
                    next_input,
                    cell_state,
                    cell_output,
                    loop_state)

        def loop_fn(time, previous_output, previous_state, previous_loop_state):
            if previous_state is None:  # time == 0
                assert previous_output is None and previous_state is None
                return loop_fn_initial()
            else:
                return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

        outputs_ta, final_state, _ = tf.nn.raw_rnn(cell=cell,
                                                   loop_fn=loop_fn,
                                                   swap_memory=True)
        outputs = outputs_ta.stack()
        outputs = flatten_time(outputs)
        outputs = tf.tanh(tf.matmul(outputs, weights) + bias)
        outputs = restore_time(outputs,
                               max_time=self.max_step,
                               batch_size=self.batch_size,
                               num_features=self.num_features)
        outputs.set_shape([self.inputs.shape[0], self.inputs.shape[1], self.num_features])

        return outputs, final_state

    @scoped_subgraph
    def rnn_output_and_state(self):
        cells = self._create_cells()

        if self.bidirectional:
            initial_states_fw, initial_states_bw = self.initial_states_tuple

            with tf.variable_scope("fw"):
                output_fw, final_state_fw = self._feed_previous_rnn(cell=cells[0],
                                                                    initial_state=initial_states_fw,
                                                                    reverse=False)

            with tf.variable_scope("bw"):
                output_bw, final_state_bw = self._feed_previous_rnn(cell=cells[1],
                                                                    initial_state=initial_states_bw,
                                                                    reverse=True)

            output = tf.concat([output_fw, output_bw], axis=2)

            # LSTM states are tuples -> need to be concatenated
            # axis is one because states have shape [batch_size, units]
            if self.cell_type == CellType.LSTM:
                final_state_fw = [tf.concat(lstm_tuple, axis=1) for lstm_tuple in final_state_fw]
                final_state_bw = [tf.concat(lstm_tuple, axis=1) for lstm_tuple in final_state_bw]

            final_state = [tf.concat(x, axis=1) for x in zip(final_state_fw, final_state_bw)]
        else:
            initial_states = self.initial_states_tuple

            output, final_state = self._feed_previous_rnn(cell=cells[0],
                                                          initial_state=initial_states,
                                                          reverse=False)

            # LSTM states are tuples -> need to be concatenated
            # axis is one because states have shape [batch_size, units]
            if self.cell_type == CellType.LSTM:
                final_state = [tf.concat(lstm_tuple, axis=1) for lstm_tuple in final_state]

        final_state = tf.concat(final_state, axis=1)

        return output, final_state
