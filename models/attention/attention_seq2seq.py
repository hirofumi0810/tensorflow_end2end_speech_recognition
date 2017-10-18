#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.model_base import ModelBase
from models.encoders.load_encoder import load as load_encoder
from models.attention.decoders.attention_layer import AttentionLayer
from models.attention.decoders.attention_decoder import AttentionDecoder, AttentionDecoderOutput
from models.attention.decoders.dynamic_decoder import _transpose_batch_time as time2batch
from models.attention.bridge import InitialStateBridge

# from models.attention.decoders.beam_search.util import choose_top_k
# from models.attention.decoders.beam_search.beam_search_decoder import BeamSearchDecoder

from collections import namedtuple


HELPERS = {
    "training": tf.contrib.seq2seq.TrainingHelper,
    "greedyembedding": tf.contrib.seq2seq.GreedyEmbeddingHelper
}


class EncoderOutput(
    namedtuple("EncoderOutput",
               ["outputs", "final_state", "seq_len"])):
    pass


class AttentionSeq2Seq(ModelBase):
    """Attention-based model.
    Args:
        input_size (int): the dimension of input vectors
        encoder_type (string): blstm or lstm
        encoder_num_units (int): the number of units in each layer of the
            encoder
        encoder_num_layers (int): the number of layers of the encoder
        encoder_num_proj (int): the number of nodes in the projection layer of
            the encoder. This is not used for GRU encoders.
        attention_type (string): the type of attention
        attention_dim: (int) the dimension of the attention layer
        decoder_type (string): lstm or gru
        decoder_num_units (int): the number of units in each layer of the decoder
        # decoder_num_proj (int): the number of nodes in the projection layer of
            the decoder. This is not used for GRU decoders.
        decoder_num_layers (int): the number of layers of the decoder
        embedding_dim (int): the dimension of the embedding in target spaces
        num_classes (int): the number of nodes in softmax layer
        sos_index (int): index of the start of sentence tag (<SOS>)
        eos_index (int): index of the end of sentence tag (<EOS>)
        max_decode_length (int): the length of output sequences to stop
            prediction when EOS token have not been emitted
        lstm_impl (string): a base implementation of LSTM. This is
            not used for GRU models.
                - BasicLSTMCell: tf.contrib.rnn.BasicLSTMCell (no peephole)
                - LSTMCell: tf.contrib.rnn.LSTMCell
                - LSTMBlockCell: tf.contrib.rnn.LSTMBlockCell
                - LSTMBlockFusedCell: under implementation
                - CudnnLSTM: under implementation
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell.
        use_peephole (bool, optional): if True, use peephole connection. This
            is not used for GRU models.
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        parameter_init (float, optional): the ange of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad_norm (float, optional): the range of clipping of gradient
            norm (> 0)
        clip_activation_encoder (float, optional): the range of clipping of
            cell activation of the encoder (> 0). This is not used for GRU
            encoders.
        clip_activation_decoder (float, optional): the range of clipping of
            cell activation of the decoder (> 0). This is not used for GRU
            decoders.
        weight_decay (float, optional): a parameter for weight decay
        time_major (bool, optional): if True, time-major computation will be
            performed
        sharpening_factor (float, optional): a sharpening factor in the
            softmax layer for computing attention weights
        logits_temperature (float, optional): a parameter for smoothing the
            softmax layer in outputing probabilities
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_num_units,
                 encoder_num_layers,
                 encoder_num_proj,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 #  decoder_num_proj,
                 decoder_num_layers,
                 embedding_dim,
                 num_classes,
                 sos_index,
                 eos_index,
                 max_decode_length,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad_norm=5.0,
                 clip_activation_encoder=50,
                 clip_activation_decoder=50,
                 weight_decay=0.0,
                 time_major=True,
                 sharpening_factor=1.0,
                 logits_temperature=1.0,
                 name='attention'):

        super(AttentionSeq2Seq, self).__init__()

        assert input_size % 3 == 0, 'input_size must be divisible by 3 (+ delta, double delta features).'
        # NOTE: input features are expected to including Δ and ΔΔ features
        assert splice % 2 == 1, 'splice must be the odd number'
        assert clip_grad_norm > 0, 'clip_grad_norm must be larger than 0.'
        assert weight_decay >= 0, 'weight_decay must not be a negative value.'

        # Setting for the encoder
        self.input_size = input_size
        self.splice = splice
        self.encoder_type = encoder_type
        self.encoder_num_units = encoder_num_units
        self.encoder_num_proj = encoder_num_proj
        self.encoder_num_layers = encoder_num_layers
        # self.downsample_list = downsample_list
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole

        # Setting for the attention layer
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.sharpening_factor = sharpening_factor
        # NOTE: sharpening_factor is good for narrow focus.
        # 2 is recommended.

        # Setting for the decoder
        self.decoder_type = decoder_type
        self.decoder_num_units = decoder_num_units
        # self.decoder_num_proj = decoder_num_proj
        self.decdoder_num_layers = decoder_num_layers
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes + 2
        # NOTE: add <SOS> and <EOS>
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_decode_length = max_decode_length
        self.logits_temperature = logits_temperature
        # self.beam_width = beam_width
        self.use_beam_search = False

        # Common setting
        self.parameter_init = parameter_init
        self.clip_grad_norm = clip_grad_norm
        self.clip_activation_encoder = clip_activation_encoder
        self.clip_activation_decoder = clip_activation_decoder
        self.weight_decay = weight_decay
        self.time_major = time_major
        self.name = name

        # Summaries for TensorBoard
        self.summaries_train = []
        self.summaries_dev = []

        # Placeholders
        self.inputs_pl_list = []
        self.labels_pl_list = []
        self.inputs_seq_len_pl_list = []
        self.labels_seq_len_pl_list = []
        self.keep_prob_encoder_pl_list = []
        self.keep_prob_decoder_pl_list = []
        self.keep_prob_embedding_pl_list = []

    def _build(self, inputs, labels, inputs_seq_len, labels_seq_len,
               keep_prob_encoder, keep_prob_decoder, keep_prob_embedding):
        """Define model graph.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            labels (placeholder): A tensor of size `[B, T]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            labels_seq_len (placeholder): A tensor of size `[B]`
            keep_prob_encoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the encoder
            keep_prob_decoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the decoder
            keep_prob_embedding (placeholder, float): A probability to keep
                nodes in the embedding layer
        Returns:
            logits: A tensor of size `[B, T_out, num_classes]`
            decoder_outputs_train (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
            decoder_outputs_infer (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
            encoder_outputs.outputs: A tensor of size `[B, T_in, encoder_num_units]`
        """
        with tf.variable_scope('encoder'):
            # Encode input features
            encoder_outputs = self._encode(
                inputs, inputs_seq_len, keep_prob_encoder)

        # Define decoder
        decoder_train = self._create_decoder(
            encoder_outputs=encoder_outputs,
            labels=labels,
            keep_prob_decoder=keep_prob_decoder,
            mode=tf.contrib.learn.ModeKeys.TRAIN)
        decoder_infer = self._create_decoder(
            encoder_outputs=encoder_outputs,
            labels=labels,
            keep_prob_decoder=keep_prob_decoder,
            mode=tf.contrib.learn.ModeKeys.INFER)

        # Wrap decoder only when inference
        # decoder_infer = self._beam_search_decoder_wrapper(
        #     decoder_infer, beam_width=self.beam_width)

        # Connect between encoder and decoder
        bridge = InitialStateBridge(
            encoder_outputs=encoder_outputs,
            decoder_state_size=decoder_train.rnn_cell.state_size,
            parameter_init=self.parameter_init)

        # Call decoder (sharing parameters)
        # Training stage
        decoder_outputs_train, _ = self._decode_train(
            decoder=decoder_train,
            bridge=bridge,
            encoder_outputs=encoder_outputs,
            labels=labels,
            labels_seq_len=labels_seq_len,
            keep_prob_embedding=keep_prob_embedding)

        # Inference stage
        decoder_outputs_infer, _ = self._decode_infer(
            decoder=decoder_infer,
            bridge=bridge,
            encoder_outputs=encoder_outputs)
        # NOTE: decoder_outputs are time-major by default

        # Convert from time-major to batch-major
        if self.time_major:
            decoder_outputs_train = self._convert_to_batch_major(
                decoder_outputs_train)
            decoder_outputs_infer = self._convert_to_batch_major(
                decoder_outputs_infer)

        # Calculate loss per example
        logits = decoder_outputs_train.logits / self.logits_temperature
        # NOTE: This is for better decoding.
        # See details in
        #   https://arxiv.org/abs/1612.02695.
        #   Chorowski, Jan, and Navdeep Jaitly.
        #   "Towards better decoding and language model integration in sequence
        #    to sequence models." arXiv preprint arXiv:1612.02695 (2016).

        return logits, decoder_outputs_train, decoder_outputs_infer, encoder_outputs.outputs

    def _encode(self, inputs, inputs_seq_len, keep_prob_encoder):
        """Encode input features.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob_encoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            encoder_outputs (namedtuple): A namedtuple of
                `(outputs, final_state, seq_len)`
                outputs: Encoder states, a tensor of size
                    `[B, T, num_units (num_proj)]` (always batch-major)
                final_state: A final hidden state of the encoder
                seq_len: equivalent to inputs_seq_len
        """
        # Define encoder
        if self.encoder_type in ['blstm', 'lstm']:
            self.encoder = load_encoder(self.encoder_type)(
                num_units=self.encoder_num_units,
                num_proj=None,  # TODO: add the projection layer
                num_layers=self.encoder_num_layers,
                lstm_impl=self.lstm_impl,
                use_peephole=self.use_peephole,
                parameter_init=self.parameter_init,
                clip_activation=self.clip_activation_encoder,
                time_major=self.time_major)
        else:
            # TODO: add other encoders
            raise NotImplementedError

        outputs, final_state = self.encoder(
            inputs=inputs,
            inputs_seq_len=inputs_seq_len,
            keep_prob=keep_prob_encoder)

        if self.time_major:
            # Convert from time-major to batch-major
            outputs = tf.transpose(outputs, [1, 0, 2])

        return EncoderOutput(outputs=outputs,
                             final_state=final_state,
                             seq_len=inputs_seq_len)

    def _create_decoder(self, encoder_outputs, labels, keep_prob_decoder,
                        mode):
        """Create attention decoder.
        Args:
            encoder_outputs (namedtuple): A namedtuple of
                `(outputs, final_state, seq_len)`
            labels (placeholder): Target labels of size `[B, T_out]`
            keep_prob_decoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the decoder
            mode: tf.contrib.learn.ModeKeys
        Returns:
            decoder (callable): A callable function of `AttentionDecoder` class
        """
        # Define the attention layer (compute attention weights)
        self.attention_layer = AttentionLayer(
            attention_type=self.attention_type,
            num_units=self.attention_dim,
            parameter_init=self.parameter_init,
            sharpening_factor=self.sharpening_factor,
            mode=mode)

        # Define RNN decoder
        cell_initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init, maxval=self.parameter_init)
        with tf.variable_scope('decoder_rnn_cell',
                               initializer=cell_initializer,
                               reuse=True if mode == tf.contrib.learn.ModeKeys.TRAIN else False):
            if self.decoder_type == 'lstm':
                if tf.__version__ == '1.3.0':
                    rnn_cell = tf.contrib.rnn.LSTMBlockCell(
                        self.decoder_num_units,
                        forget_bias=1.0,
                        clip_cell=self.clip_activation_decoder,
                        use_peephole=self.use_peephole)
                else:
                    rnn_cell = tf.contrib.rnn.LSTMBlockCell(
                        self.decoder_num_units,
                        forget_bias=1.0,
                        use_peephole=self.use_peephole)
            elif self.decoder_type == 'gru':
                rnn_cell = tf.contrib.rnn.GRUCell(self.num_units)
            else:
                raise TypeError('decoder_type is "lstm" or "gru".')

            # Dropout for the hidden-hidden connections
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_cell, output_keep_prob=keep_prob_decoder)

        # Define attention decoder
        self.decoder = AttentionDecoder(
            rnn_cell=rnn_cell,
            parameter_init=self.parameter_init,
            max_decode_length=self.max_decode_length,
            num_classes=self.num_classes,
            encoder_outputs=encoder_outputs.outputs,
            encoder_outputs_seq_len=encoder_outputs.seq_len,
            attention_layer=self.attention_layer,
            time_major=self.time_major,
            mode=mode)

        return self.decoder

    def _convert_to_batch_major(self, decoder_outputs):
        """
        Args:
            decoder_outputs (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
        Returns:
            decoder_outputs (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
        """
        logits = time2batch(decoder_outputs.logits)
        predicted_ids = time2batch(decoder_outputs.predicted_ids)
        decoder_output = time2batch(decoder_outputs.decoder_output)
        attention_weights = time2batch(
            decoder_outputs.attention_weights)
        context_vector = time2batch(
            decoder_outputs.context_vector)
        decoder_outputs = AttentionDecoderOutput(
            logits=logits,
            predicted_ids=predicted_ids,
            decoder_output=decoder_output,
            attention_weights=attention_weights,
            context_vector=context_vector)
        return decoder_outputs

    def _decode_train(self, decoder, bridge, encoder_outputs, labels,
                      labels_seq_len, keep_prob_embedding):
        """Runs decoding in training mode.
        Args:
            decoder (callable): A callable function of `AttentionDecoder` class
            bridge (callable): A callable function to connect between the
                encoder and decoder
            encoder_outputs (namedtuple): A namedtuple of
                `(outputs, final_state, seq_len)`
            labels: A tensor of size `[B, T_out]`
            labels_seq_len: A tensor of size `[B]`
            keep_prob_embedding (placeholder, float): A probability to keep
                nodes in the embedding layer
        Returns:
            decoder_outputs (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
            decoder_final_state:
        """
        # Generate embedding of target labels
        with tf.variable_scope("output_embedding"):
            output_embedding = tf.get_variable(
                name="W_embedding",
                shape=[self.num_classes, self.embedding_dim],
                initializer=tf.random_uniform_initializer(
                    -self.parameter_init, self.parameter_init))
            labels_embedded = tf.nn.embedding_lookup(output_embedding,
                                                     labels)
            labels_embedded = tf.nn.dropout(labels_embedded,
                                            keep_prob=keep_prob_embedding)

        helper_train = tf.contrib.seq2seq.TrainingHelper(
            inputs=labels_embedded[:, :-1, :],  # exclude <EOS>
            sequence_length=labels_seq_len - 1,  # exclude <EOS>
            time_major=False)  # TODO: self.time_major??
        # labels_embedded: `[B, T_out, embedding_dim]`

        # The initial decoder state is the final encoder state
        with tf.variable_scope("bridge"):
            decoder_initial_state = bridge()

        # Call decoder class
        decoder_outputs, decoder_final_state = decoder(
            initial_state=decoder_initial_state,
            helper=helper_train)
        # NOTE: These are time-major if self.time_major is True

        return decoder_outputs, decoder_final_state

    def _decode_infer(self, decoder, bridge, encoder_outputs):
        """Runs decoding in inference mode.
        Args:
            decoder (callable): A callable function of `AttentionDecoder` class
            bridge (callable): A callable function to connect between the
                encoder and decoder
            encoder_outputs (namedtuple): A namedtuple of
                `(outputs, final_state, seq_len)`
        Returns:
            decoder_outputs (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
            decoder_final_state:
        """
        batch_size = tf.shape(encoder_outputs.outputs)[0]

        # if self.use_beam_search:
        #     batch_size = self.beam_width
        # TODO: make this batch version

        with tf.variable_scope("output_embedding", reuse=True):
            output_embedding = tf.get_variable(
                name="W_embedding",
                shape=[self.num_classes, self.embedding_dim],
                initializer=tf.random_uniform_initializer(
                    -self.parameter_init, self.parameter_init))
            # NOTE: dropout will not be performed when inference

        helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=output_embedding,
            start_tokens=tf.fill([batch_size], self.sos_index),
            # start_tokens=tf.tile([self.sos_index], [batch_size]),
            end_token=self.eos_index)
        # ex.) Output tensor has shape [2, 3].
        # tf.fill([2, 3], 9) ==> [[9, 9, 9]
        #                         [9, 9, 9]]

        # The initial decoder state is the final encoder state
        with tf.variable_scope("bridge", reuse=True):
            decoder_initial_state = bridge()

        # Call decoder class
        decoder_outputs, decoder_final_state = decoder(
            initial_state=decoder_initial_state,
            helper=helper_infer)
        # NOTE: These are time-major if self.time_major is True

        return decoder_outputs, decoder_final_state

    def create_placeholders(self):
        """Create placeholders and append them to list."""
        self.inputs_pl_list.append(
            tf.placeholder(tf.float32, shape=[None, None, self.input_size],
                           name='input'))
        self.labels_pl_list.append(
            tf.placeholder(tf.int32, shape=[None, None], name='labels'))
        self.inputs_seq_len_pl_list.append(
            tf.placeholder(tf.int32, shape=[None], name='inputs_seq_len'))
        self.labels_seq_len_pl_list.append(
            tf.placeholder(tf.int32, shape=[None], name='labels_seq_len'))

        self.keep_prob_encoder_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_encoder'))
        self.keep_prob_decoder_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_decoder'))
        self.keep_prob_embedding_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_embedding'))

        # These are prepared for computing LER
        self.labels_st_true_pl = tf.SparseTensor(
            tf.placeholder(tf.int64, name='indices_true'),
            tf.placeholder(tf.int32, name='values_true'),
            tf.placeholder(tf.int64, name='shape_true'))
        self.labels_st_pred_pl = tf.SparseTensor(
            tf.placeholder(tf.int64, name='indices_pred'),
            tf.placeholder(tf.int32, name='values_pred'),
            tf.placeholder(tf.int64, name='shape_pred'))

    def _beam_search_decoder_wrapper(self, decoder, beam_width=1,
                                     length_penalty_weight=0.6):
        """Wraps a decoder into a Beam Search decoder.
        Args:
            decoder: An instance of `RNNDecoder` class
            beam_width (int): the number of beams to use
            length_penalty_weight (float): weight for the length penalty factor.
                0 disables the penalty.
        Returns:
            A callable BeamSearchDecoder with the same interfaces as the
                attention decoder
        """
        assert isinstance(beam_width, int)
        assert beam_width >= 1

        if beam_width == 1:
            # Greedy decoding
            self.use_beam_search = False
            return decoder
        else:
            self.use_beam_search = True
            return BeamSearchDecoder(
                decoder=decoder,
                beam_width=beam_width,
                vocab_size=self.num_classes,
                eos_index=self.eos_index,
                length_penalty_weight=length_penalty_weight,
                choose_successors_fn=choose_top_k)

    def compute_loss(self, inputs, labels, inputs_seq_len, labels_seq_len,
                     keep_prob_encoder, keep_prob_decoder, keep_prob_embedding,
                     scope=None):
        """Operation for computing cross entropy sequence loss.
        Args:
            inputs: A tensor of `[B, T_in, input_size]`
            labels: A tensor of `[B, T_out]`
            inputs_seq_len: A tensor of `[B]`
            labels_seq_len: A tensor of `[B]`
            keep_prob_encoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the encoder
            keep_prob_decoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the decoder
            keep_prob_embedding (placeholder, float): A probability to keep
                nodes in the embedding layer
            scope (optional): A scope in the model tower
        Returns:
            total_loss: operation for computing total loss (cross entropy
                sequence loss + L2).
                This is a single scalar tensor to minimize.
            logits: A tensor of size `[B, T_out, num_classes]`
            decoder_outputs_train (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
            decoder_outputs_infer (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
        """
        # Build model graph
        logits, decoder_outputs_train, decoder_outputs_infer, _ = self._build(
            inputs, labels, inputs_seq_len, labels_seq_len,
            keep_prob_encoder, keep_prob_decoder, keep_prob_embedding)

        # For prevent 0 * log(0) in crossentropy loss
        epsilon = tf.constant(value=1e-10)
        logits = logits + epsilon

        # Weight decay
        if self.weight_decay > 0:
            with tf.name_scope("weight_decay_loss"):
                weight_sum = 0
                for var in tf.trainable_variables():
                    if 'bias' not in var.name.lower():
                        weight_sum += tf.nn.l2_loss(var)
                tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("sequence_loss"):
            # batch_size = tf.cast(tf.shape(inputs)[0], tf.float32)
            labels_max_seq_len = tf.shape(labels[:, 1:])[1]
            loss_mask = tf.sequence_mask(tf.to_int32(labels_seq_len - 1),
                                         maxlen=labels_max_seq_len,
                                         dtype=tf.float32)
            sequence_loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=labels[:, 1:],  # exclude <SOS>
                weights=loss_mask,
                average_across_timesteps=True,
                average_across_batch=True,
                softmax_loss_function=None)
            # sequence_loss /= batch_size

            tf.add_to_collection('losses', sequence_loss)

        # Compute total loss
        total_loss = tf.add_n(tf.get_collection('losses', scope),
                              name='total_loss')

        # Add a scalar summary for the snapshot of loss
        if self.weight_decay > 0:
            self.summaries_train.append(
                tf.summary.scalar('weight_loss_train',
                                  weight_sum * self.weight_decay))
            self.summaries_dev.append(
                tf.summary.scalar('weight_loss_dev',
                                  weight_sum * self.weight_decay))
            self.summaries_train.append(
                tf.summary.scalar('total_loss_train', total_loss))
            self.summaries_dev.append(
                tf.summary.scalar('total_loss_dev', total_loss))

        self.summaries_train.append(
            tf.summary.scalar('sequence_loss_train', sequence_loss))
        self.summaries_dev.append(
            tf.summary.scalar('sequence_loss_dev', sequence_loss))

        return total_loss, logits, decoder_outputs_train, decoder_outputs_infer

    def decode(self, decoder_outputs_train, decoder_outputs_infer):
        """Operation for decoding.
        Args:
            decoder_outputs_train (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
            decoder_outputs_infer (namedtuple): A namedtuple of
                `(logits, predicted_ids, decoder_output, attention_weights,
                    context_vector)`
        Return:
            decoded_train: operation for decoding in training.
                A tensor of size `[B, T_out]`
            decoded_infer: operation for decoding in inference.
                A tensor of size `[B, max_decode_length]`
        """
        decoded_train = decoder_outputs_train.predicted_ids

        if self.use_beam_search:
            # Beam search decoding
            decoded_infer = decoder_outputs_infer.predicted_ids[0]

            # predicted_ids = decoder_outputs_infer.beam_search_output.predicted_ids
            # scores = decoder_outputs_infer.beam_search_output.scores[:, :, -1]
            # argmax_score = tf.argmax(scores, axis=0)[0]
            # NOTE: predicted_ids: `[time, 1, beam_width]`

            # Convert to `[beam_width, 1, time]`
            # predicted_ids = tf.transpose(predicted_ids, (2, 1, 0))

            # decoded_infer = predicted_ids[argmax_score]
            # decoded_infer = decoder_outputs_infer.predicted_ids[-1]
        else:
            # Greedy decoding
            decoded_infer = decoder_outputs_infer.predicted_ids

        return decoded_train, decoded_infer

    def compute_ler(self, labels_true, labels_pred):
        """Operation for computing LER (Label Error Rate).
        Args:
            labels_true: A SparseTensor of target labels
            labels_pred: A SparseTensor of predicted labels
        Returns:
            ler_op: operation for computing LER
        """
        # Compute LER (normalize by label length)
        ler_op = tf.reduce_mean(tf.edit_distance(
            labels_pred, labels_true, normalize=True))
        # TODO: consider variable lengths

        # Add a scalar summary for the snapshot of LER
        # with tf.name_scope("ler"):
        #     self.summaries_train.append(tf.summary.scalar(
        #         'ler_train', ler_op))
        #     self.summaries_dev.append(tf.summary.scalar(
        #         'ler_dev', ler_op))
        # TODO: feed_dictのタイミング違うからエラーになる

        return ler_op
