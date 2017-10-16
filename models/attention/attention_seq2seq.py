#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.attention.decoders.beam_search.util import choose_top_k
from models.attention.decoders.beam_search.beam_search_decoder import BeamSearchDecoder

from models.encoders.load_encoder import load as load_encoder
from models.attention.base import AttentionBase
from models.attention.decoders.attention_layer import AttentionLayer
from models.attention.decoders.attention_decoder import AttentionDecoder
from models.attention.decoders.attention_decoder import AttentionDecoderOutput
from models.attention.decoders.dynamic_decoder import _transpose_batch_time as time2batch
from models.attention.bridge import InitialStateBridge

from collections import namedtuple

OPTIMIZER_CLS_NAMES = {
    "adagrad": tf.train.AdagradOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "adam": tf.train.AdamOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "nestrov": tf.train.MomentumOptimizer
}


HELPERS = {
    "training": tf.contrib.seq2seq.TrainingHelper,
    "greedyembedding": tf.contrib.seq2seq.GreedyEmbeddingHelper
}


class EncoderOutput(
    namedtuple("EncoderOutput",
               ["outputs", "final_state", "seq_len"])):
    pass


class AttentionSeq2Seq(AttentionBase):
    """Attention-based model.
    Args:
        input_size (int): the dimension of input vectors
        encoder_type (string): blstm only now
        encoder_num_units (int): the number of units in each layer of the
            encoder
        encoder_num_layers (int): the number of layers of the encoder
        attention_dim: (int) the dimension of the attention layer
        attention_type (string): content or location or hybrid or layer_dot
        decoder_type (string): lstm only now
        decoder_num_units (int): the number of units in each layer of the decoder
        decoder_num_layers (int): the number of layers of the decoder
        embedding_dim (int): the dimension of the embedding in target spaces
        num_classes (int): the number of nodes in softmax layer
        sos_index (int): index of the start of sentence tag (<SOS>)
        eos_index (int): index of the end of sentence tag (<EOS>)
        max_decode_length (int): the length of output sequences to stop
            prediction when EOS token have not been emitted
        attention_smoothing (bool, optional): if True, replace exp to sigmoid
            function in the softmax layer of computing attention weights
        sharpening_factor (float, optional): a parameter for
            smoothing the softmax layer in computating attention weights
        logits_temperature (float, optional):  a parameter for smoothing the
            softmax layer in outputing probabilities
        parameter_init (float, optional): the ange of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad (float, optional): the range of gradient clipping (> 0)
        clip_activation_encoder (float, optional): the range of activation
            clipping in the encoder (> 0)
        clip_activation_decoder (float, optional): the range of activation
            clipping in the decoder (> 0)
        beam_width (int, optional): the number of beam widths. beam width 1
            means greedy decoding.
        weight_decay (float, optional): a parameter for weight decay
        time_major (bool, optional): if True, time-major computation will be
            performed
    """

    def __init__(self,
                 input_size,
                 encoder_type,
                 encoder_num_units,
                 encoder_num_layers,
                 encoder_num_proj,
                 encoder_dropout,
                 attention_type,
                 attention_dim,
                 decoder_type,
                 decoder_num_units,
                 #  decoder_num_proj,
                 decoder_num_layers,
                 decoder_dropout,
                 embedding_dim,
                 embedding_dropout,
                 num_classes,
                 sos_index,
                 eos_index,
                 max_decode_length,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=5.0,
                 clip_activation_encoder=50,
                 clip_activation_decoder=50,
                 weight_decay=0.0,
                 time_major=False,
                 sharpening_factor=1.0,
                 logits_temperature=1.0,
                 name='attention_seq2seq'):

        assert input_size % 3 == 0, 'input_size must be divisible by 3 (+ delta, double delta features).'
        # NOTE: input features are expected to including Δ and ΔΔ features
        # assert splice % 2 == 1, 'splice must be the odd number'
        assert clip_grad > 0, 'clip_grad must be larger than 0.'
        assert weight_decay >= 0, 'weight_decay must not be a negative value.'

        super(AttentionSeq2Seq, self).__init__(
            input_size, attention_dim, embedding_dim, num_classes, sos_index,
            eos_index, clip_grad, weight_decay)

        # Setting for the encoder
        self.input_size = input_size
        self.splice = splice
        self.encoder_type = encoder_type
        self.encoder_num_units = encoder_num_units
        self.encoder_num_proj = encoder_num_proj
        self.encoder_num_layers = encoder_num_layers
        # self.downsample_list = downsample_list
        self.encoder_dropout = encoder_dropout

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
        self.decoder_dropout = decoder_dropout
        self.embedding_dim = embedding_dim
        self.embedding_dropout = embedding_dropout
        self.num_classes = num_classes + 2
        # NOTE: add <SOS> and <EOS>
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.max_decode_length = max_decode_length
        self.logits_temperature = logits_temperature
        # self.beam_width = beam_width
        # self.use_beam_search = True if beam_width >= 2 else False

        # Common setting
        self.parameter_init = parameter_init
        self.clip_grad = clip_grad
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
        self.keep_prob_input_pl_list = []
        self.keep_prob_hidden_pl_list = []
        self.keep_prob_output_pl_list = []

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
                nodes in the hidden-hidden connection of the embedding
        Returns:
            logits ():
            decoder_outputs_train ():
            decoder_outputs_infer ():
        """
        # Encode input features
        encoder_outputs = self._encode(
            inputs, inputs_seq_len, keep_prob_encoder)

        # Define decoder
        decoder_train = self._create_decoder(
            encoder_outputs=encoder_outputs,
            labels=labels,
            keep_prob_decoder=keep_prob_decoder
            reuse=False)
        decoder_infer = self._create_decoder(
            encoder_outputs=encoder_outputs,
            labels=labels,
            keep_prob_decoder=keep_prob_decoder,
            reuse=True)
        # NOTE: initial_state and helper will be substituted in
        # self._decode_train() or self._decode_infer()

        # Wrap decoder only when inference
        # decoder_infer = self._beam_search_decoder_wrapper(
        #     decoder_infer, beam_width=self.beam_width)

        # Connect between encoder and decoder
        bridge = InitialStateBridge(
            encoder_outputs=encoder_outputs,
            decoder_state_size=decoder_train.cell.state_size)

        # Call decoder (sharing parameters)
        # Training state
        decoder_outputs_train, _ = self._decode_train(
            decoder=decoder_train,
            bridge=bridge,
            encoder_outputs=encoder_outputs,
            labels=labels,
            labels_seq_len=labels_seq_len)

        # Inference stage
        decoder_outputs_infer, _ = self._decode_infer(
            decoder=decoder_infer,
            bridge=bridge,
            encoder_outputs=encoder_outputs)
        # NOTE: decoder_outputs are time-major

        # Transpose from time-major to batch-major
        if self.time_major:
            logits = time2batch(decoder_outputs_train.logits)
            predicted_ids = time2batch(decoder_outputs_train.predicted_ids)
            cell_output = time2batch(decoder_outputs_train.cell_output)
            attention_scores = time2batch(
                decoder_outputs_train.attention_scores)
            attention_context = time2batch(
                decoder_outputs_train.attention_context)
            decoder_outputs_train = AttentionDecoderOutput(
                logits=logits,
                predicted_ids=predicted_ids,
                cell_output=cell_output,
                attention_scores=attention_scores,
                attention_context=attention_context)

            logits = time2batch(decoder_outputs_infer.logits)
            predicted_ids = time2batch(decoder_outputs_infer.predicted_ids)
            cell_output = time2batch(decoder_outputs_infer.cell_output)
            attention_scores = time2batch(
                decoder_outputs_infer.attention_scores)
            attention_context = time2batch(
                decoder_outputs_infer.attention_context)
            decoder_outputs_infer = AttentionDecoderOutput(
                logits=logits,
                predicted_ids=predicted_ids,
                cell_output=cell_output,
                attention_scores=attention_scores,
                attention_context=attention_context)

        # Calculate loss per example
        logits = decoder_outputs_train.logits / self.logits_temperature
        # NOTE: This is for better decoding.
        # See details in
        #   https://arxiv.org/abs/1612.02695.
        #   Chorowski, Jan, and Navdeep Jaitly.
        #   "Towards better decoding and language model integration in sequence
        #    to sequence models." arXiv preprint arXiv:1612.02695 (2016).

        return logits, decoder_outputs_train, decoder_outputs_infer

    def _encode(self, inputs, inputs_seq_len, keep_prob):
        """Encode input features.
        Args:
            inputs (placeholder): A tensor of size`[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            encoder_outputs (namedtuple): A namedtuple of
                `(outputs, final_state, seq_len)`
        """
        # Define encoder
        if self.encoder_type in ['blstm']:
            self.encoder = load_encoder(self.encoder_type)(
                num_units=self.encoder_num_units,
                num_proj=None,  # TODO: add the projection layer
                num_layers=self.encoder_num_layers,
                lstm_impl='LSTMBlockCell',
                use_peephole=True,
                parameter_init=self.parameter_init,
                clip_activation=self.clip_activation_encoder)
        else:
            # TODO: add other encoders
            raise NotImplementedError

        outputs, final_state = self.encoder(
            inputs=inputs,
            inputs_seq_len=inputs_seq_len,
            keep_prob=keep_prob)

        return EncoderOutput(outputs=outputs,
                             final_state=final_state,
                             seq_len=inputs_seq_len)

    def _create_decoder(self, encoder_outputs, labels, keep_prob_decoder, reuse):
        """Create attention decoder.
        Args:
            encoder_outputs (namedtuple): A namedtuple of
                `(outputs, final_state, seq_len)`
            labels (placeholder): Target labels of size `[B, T_out]`
            keep_prob_decoder (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection of the decoder
            reuse (bool): if True, reuse parameters
        Returns:
            decoder: An instance of the decoder class
        """
        # Define the attention layer (compute attention weights)
        with tf.variable('attention_layer', reuse=reuse):
            self.attention_layer = AttentionLayer(
                attention_type=self.attention_type,
                num_units=self.attention_dim,
                sharpening_factor=self.sharpening_factor)

        # Define RNN decoder
        decoder_initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init,
            maxval=self.parameter_init)
        with tf.variable_scope('rnn_decoder',
                               initializer=decoder_initializer,
                               reuse=reuse):
            if self.decoder_type == 'lstm':
                rnn_cell = tf.contrib.rnn.LSTMBlockCell(
                    self.decoder_num_units,
                    forget_bias=1.0,
                    # clip_cell=True,
                    use_peephole=self.use_peephole)
                # TODO: cell clipping (update for rc1.3)
            elif self.decoder_type == 'gru':
                rnn_cell = tf.contrib.rnn.GRUCell(self.num_units)

            else:
                raise TypeError('decoder_type is "lstm" or "gru".')

            # Dropout for the hidden-hidden connections
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                rnn_cell, output_keep_prob=keep_prob_decoder)

        # Define attention decoder
        att_decoder_initializer = tf.random_uniform_initializer(
            minval=-self.parameter_init,
            maxval=self.parameter_init)
        with tf.variable_scope('attention_decoder',
                               initializer=att_decoder_initializer,
                               reuse=reuse):
            self.decoder = AttentionDecoder(
                rnn_cell=rnn_cell,
                parameter_init=self.parameter_init,
                max_decode_length=self.max_decode_length,
                num_classes=self.num_classes,
                encoder_outputs=encoder_outputs.outputs,
                encoder_outputs_seq_len=encoder_outputs.seq_len,
                attention_layer=self.attention_layer,
                time_major=self.time_major)

        return self.decoder

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
        self.keep_prob_input_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_input'))
        self.keep_prob_hidden_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_hidden'))
        self.keep_prob_output_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob_output'))

        # These are prepared for computing LER
        self.labels_st_true_pl = tf.SparseTensor(
            tf.placeholder(tf.int64, name='indices_true'),
            tf.placeholder(tf.int32, name='values_true'),
            tf.placeholder(tf.int64, name='shape_true'))
        self.labels_st_pred_pl = tf.SparseTensor(
            tf.placeholder(tf.int64, name='indices_pred'),
            tf.placeholder(tf.int32, name='values_pred'),
            tf.placeholder(tf.int64, name='shape_pred'))

    def _generate_target_embedding(self, reuse):
        """Returns the embedding used for the target sequence."""
        with tf.variable_scope("target_embedding", reuse=reuse):
            return tf.get_variable(
                name="W_embedding",
                shape=[self.num_classes, self.embedding_dim],
                initializer=tf.random_uniform_initializer(
                    -self.parameter_init, self.parameter_init))
        # TODO: Consider shape of target_embedding

    def _beam_search_decoder_wrapper(self, decoder, beam_width=None,
                                     length_penalty_weight=0.6):
        """Wraps a decoder into a Beam Search decoder.
        Args:
            decoder: An instance of `RNNDecoder` class
            beam_width: int, the number of beams to use
            length_penalty_weight: A float value, weight for the length penalty
                factor. 0.0 disables the penalty.
        Returns:
            A callable BeamSearchDecoder with the same interfaces as the
                attention decoder
        """
        if beam_width is None or beam_width <= 1:
            # Greedy decoding
            self.use_beam_search = False
            return decoder

        self.use_beam_search = True
        return BeamSearchDecoder(
            decoder=decoder,
            beam_width=beam_width,
            vocab_size=self.num_classes,
            eos_index=self.eos_index,
            length_penalty_weight=length_penalty_weight,
            choose_successors_fn=choose_top_k)

    def _decode_train(self, decoder, bridge, encoder_outputs, labels,
                      labels_seq_len):
        """Runs decoding in training mode.
        Args:
            decoder: An instance of the decoder class
            bridge:
            encoder_outputs:
            labels: Target labels of size `[B, T_out, num_classes]`
            labels_seq_len: The length of target labels
        Returns:
            decoder_outputs: A tuple of `(AttentionDecoderOutput, final_state)`
        """
        # Generate embedding of target labels
        target_embedding = self._generate_target_embedding(reuse=False)
        target_embedded = tf.nn.embedding_lookup(target_embedding,
                                                 labels)
        # TODO: add Dropout

        helper_train = tf.contrib.seq2seq.TrainingHelper(
            inputs=target_embedded[:, :-1, :],  # embedding of target labels
            # inputs=labels[:, :-1, :],
            sequence_length=labels_seq_len - 1,  # include <SOS>, exclude <EOS>
            time_major=False)  # self.time_major??
        # target_embedded: `[batch_size, time, embedding_dim]`

        decoder_initial_state = bridge(reuse=False)

        # Call decoder class
        (decoder_outputs, final_state) = decoder(
            initial_state=decoder_initial_state,
            helper=helper_train,
            mode=tf.contrib.learn.ModeKeys.TRAIN)
        # NOTE: They are time-major if self.time_major is True

        return (decoder_outputs, final_state)

    def _decode_infer(self, decoder, bridge, encoder_outputs):
        """Runs decoding in inference mode.
        Args:
            decoder: An instance of the decoder class
            bridge:
            encoder_outputs: A namedtuple of
                outputs
                final_state
                attention_values
                attention_values_length
        Returns:
            decoder_outputs: A tuple of `(AttentionDecoderOutput, final_state)`
        """
        batch_size = tf.shape(encoder_outputs.outputs)[0]

        if self.use_beam_search:
            batch_size = self.beam_width
        # TODO: make this batch version

        target_embedding = self._generate_target_embedding(reuse=True)

        helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            # embedding=self.decoder_outputs_train.logits,
            embedding=target_embedding,  # embedding of predicted labels
            start_tokens=tf.fill([batch_size], self.sos_index),
            # start_tokens=tf.tile([self.sos_index], [batch_size]),
            end_token=self.eos_index)
        # ex.)
        # Output tensor has shape [2, 3].
        # tf.fill([2, 3], 9) ==> [[9, 9, 9]
        #                         [9, 9, 9]]

        decoder_initial_state = bridge(reuse=True)

        # Call decoder class
        (decoder_outputs, final_state) = decoder(
            initial_state=decoder_initial_state,
            helper=helper_infer,
            mode=tf.contrib.learn.ModeKeys.INFER)
        # NOTE: They are time-major if self.time_major is True

        return (decoder_outputs, final_state)

    def compute_loss(self, inputs, labels, inputs_seq_len, labels_seq_len,
                     keep_prob_input, keep_prob_hidden, keep_prob_output,
                     scope=None):
        """Operation for computing cross entropy sequence loss.
        Args:
            inputs: A tensor of `[B, T, input_size]`
            labels: A tensor of `[B, T]`
            inputs_seq_len: A tensor of `[B]`
            labels_seq_len: A tensor of `[B]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden layers
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output layer
        Returns:
            loss: operation for computing total loss (cross entropy sequence
                loss + L2). This is a single scalar tensor to minimize.
            logits:
            decoder_outputs_train:
            decoder_outputs_infer:
        """
        # Build model graph
        logits, decoder_outputs_train, decoder_outputs_infer = self._build(
            inputs, labels, inputs_seq_len, labels_seq_len,
            keep_prob_input, keep_prob_hidden, keep_prob_output)

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
            max_time = tf.shape(labels[:, 1:])[1]
            loss_mask = tf.sequence_mask(tf.to_int32(labels_seq_len - 1),
                                         maxlen=max_time,
                                         dtype=tf.float32)
            sequence_losses = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=labels[:, 1:],
                weights=loss_mask,
                average_across_timesteps=True,
                average_across_batch=True,
                softmax_loss_function=None)

            sequence_loss = tf.reduce_sum(sequence_losses,
                                          name='sequence_loss_mean')
            tf.add_to_collection('losses', sequence_loss)

        # Compute total loss
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

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

    def set_optimizer(self, optimizer_name, learning_rate):
        """Set optimizer.
        Args:
            optimizer: string, name of the optimizer in OPTIMIZER_CLS_NAMES
            learning_rate: A float value, a learning rate
        Returns:
            optimizer:
        """
        optimizer_name = optimizer_name.lower()
        if optimizer_name not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer_name))

        # Select optimizer
        if optimizer_name == 'momentum':
            return OPTIMIZER_CLS_NAMES[optimizer_name](
                learning_rate=learning_rate,
                momentum=0.9)
        else:
            return OPTIMIZER_CLS_NAMES[optimizer_name](
                learning_rate=learning_rate)

    def train(self, loss, optimizer, learning_rate=None, clip_norm=False):
        """Operation for training. Only the sigle GPU training is supported.
        Args:
            loss: An operation for computing loss
            optimizer: string, name of the optimizer in OPTIMIZER_CLS_NAMES
            learning_rate: A float value, a learning rate
            clip_norm: if True, clip gradients norm by self.clip_grad
        Returns:
            train_op: operation for training
        """
        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set optimizer
        self.optimizer = self.set_optimizer(optimizer, learning_rate)

        # TODO: Optionally wrap with SyncReplicasOptimizer

        if self.clip_grad is not None:
            # Compute gradients
            grads_and_vars = self.optimizer.compute_gradients(loss)

            # Clip gradients
            clipped_grads_and_vars = self._clip_gradients(grads_and_vars,
                                                          clip_norm)

            # Create gradient updates
            train_op = self.optimizer.apply_gradients(
                clipped_grads_and_vars,
                global_step=global_step)

        else:
            # Use the optimizer to apply the gradients that minimize the loss
            # and also increment the global step counter as a single training
            # step
            train_op = self.optimizer.minimize(loss, global_step=global_step)

        return train_op

    def _clip_gradients(self, grads_and_vars, _clip_norm):
        """Clip gradients.
        Args:
            grads_and_vars: list of (grads, vars) tuples
            _clip_norm: if True, clip gradients norm by self.clip_grad
        Returns:
            clipped_grads_and_vars: list of (clipped grads, vars)
        """
        # TODO: Optionally add gradient noise

        clipped_grads_and_vars = []

        if _clip_norm:
            # Clip gradient norm
            for grad, var in grads_and_vars:
                if grad is not None:
                    clipped_grads_and_vars.append(
                        (tf.clip_by_norm(grad, clip_norm=self.clip_grad), var))
        else:
            # Clip gradient
            for grad, var in grads_and_vars:
                if grad is not None:
                    clipped_grads_and_vars.append(
                        (tf.clip_by_value(grad,
                                          clip_value_min=-self.clip_grad,
                                          clip_value_max=self.clip_grad), var))

        # TODO: Add histograms for variables, gradients (norms)
        # self._tensorboard(trainable_vars)

        return clipped_grads_and_vars

    def decoder(self, decoder_outputs_train, decoder_outputs_infer):
        """Operation for decoding.
        Args:
            decoder_outputs_train: An instance of ``
            decoder_outputs_infer: An instance of ``
        Return:
            decoded_train: operation for decoding in training. A tensor of
                size `[B, ]`
            decoded_infer: operation for decoding in inference. A tensor of
                size `[, max_decode_length]`
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

            argmax_score = None

        return decoded_train, decoded_infer

    def compute_ler(self, labels_true, labels_pred):
        """Operation for computing LER (Label Error Rate).
        Args:
            labels_true: A SparseTensor
            labels_pred: A SparseTensor
        Returns:
            ler_op: operation for computing LER
        """
        # Compute LER (normalize by label length)
        ler_op = tf.reduce_mean(tf.edit_distance(
            labels_pred, labels_true, normalize=True))
        # TODO: パディングを考慮して計算する

        # Add a scalar summary for the snapshot of LER
        # with tf.name_scope("ler"):
        #     self.summaries_train.append(tf.summary.scalar(
        #         'ler_train', ler_op))
        #     self.summaries_dev.append(tf.summary.scalar(
        #         'ler_dev', ler_op))
        # TODO: feed_dictのタイミング違うからエラーになる

        return ler_op
