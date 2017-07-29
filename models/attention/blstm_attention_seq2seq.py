#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention model class based on BLSTM encoder and LSTM decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.attention.attention_seq2seq_base import AttentionBase
from models.attention.encoders.load_encoder import load as load_encoder
from models.attention.decoders.load_decoder import load as load_decoder
from models.attention.decoders.attention_layer import AttentionLayer
from models.attention.decoders.attention_decoder import AttentionDecoder
from models.attention.decoders.attention_decoder import AttentionDecoderOutput
from models.attention.decoders.dynamic_decoder import _transpose_batch_time as time2batch
from models.attention.bridge import InitialStateBridge


class BLSTMAttetion(AttentionBase):
    """Bidirectional LSTM based Attention model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimension of input vectors
        encoder_num_unit: int, the number of units in each layer of the
            encoder
        encoder_num_layer: int, the number of layers of the encoder
        attention_dim: int, the dimension of the attention layer
        attention_type: string, content or location or hybrid or layer_dot
        decoder_num_unit: int, the number of units in each layer of the
            decoder
        decoder_num_layer: int, the number of layers of the decoder
        embedding_dim: int, the dimension of the embedding in target spaces
        num_classes: int, the number of nodes in softmax layer
        sos_index: index of the start of sentence tag (<SOS>)
        eos_index: index of the end of sentence tag (<EOS>)
        max_decode_length: int, the length of output sequences to stop
            prediction when EOS token have not been emitted
        attention_smoothing: bool, if True, replace exp to sigmoid function in
            the softmax layer of computing attention weights
        attention_weights_tempareture: A float value,
        logits_tempareture: A float value,
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (non-negative)
        clip_activation_encoder: A float value. Range of activation clipping in
            the encoder (> 0)
        clip_activation_decoder:
        dropout_ratio_input: A float value. Dropout ratio in the input-hidden
            layer
        dropout_ratio_hidden: A float value. Dropout ratio in the hidden-hidden
            layers
        dropout_ratio_output: A float value. Dropout ratio in the hidden-output
            layer
        beam_width: int, the number of beams to use. 1 diables the beam search
            decoding (greedy decoding).
        weight_decay: A float value. Regularization parameter for weight decay
        time_major: bool,
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 encoder_num_unit,
                 encoder_num_layer,
                 attention_dim,
                 attention_type,
                 decoder_num_unit,
                 decoder_num_layer,
                 embedding_dim,
                 num_classes,
                 sos_index,
                 eos_index,
                 max_decode_length,
                 attention_smoothing=False,
                 attention_weights_tempareture=1.0,
                 logits_tempareture=1.0,
                 parameter_init=0.1,
                 clip_grad=5.0,
                 clip_activation_encoder=50,
                 clip_activation_decoder=50,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 dropout_ratio_output=1.0,
                 weight_decay=0.0,
                 beam_width=1,
                 time_major=False,
                 name='blstm_attention_seq2seq'):

        # AttentionBase.__init__(self)

        self.batch_size = int(batch_size)
        self.input_size = int(input_size)
        self.encoder_num_unit = int(encoder_num_unit)
        self.encoder_num_layer = int(encoder_num_layer)
        self.attention_dim = int(attention_dim)
        self.attention_type = attention_type
        self.decoder_num_unit = int(decoder_num_unit)
        self.decoder_num_layer = int(decoder_num_layer)
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.sos_index = int(sos_index)
        self.eos_index = int(eos_index)
        self.max_decode_length = int(max_decode_length)
        self.attention_smoothing = bool(attention_smoothing)
        self.attention_weights_tempareture = float(
            attention_weights_tempareture)
        self.logits_tempareture = float(logits_tempareture)
        self.parameter_init = float(parameter_init)
        self.clip_grad = float(clip_grad)
        self.clip_activation_encoder = float(clip_activation_encoder)
        self.clip_activation_decoder = float(clip_activation_decoder)
        self.dropout_ratio_input = float(dropout_ratio_input)
        self.dropout_ratio_hidden = float(dropout_ratio_hidden)
        self.dropout_ratio_output = float(dropout_ratio_output)
        self.weight_decay = float(weight_decay)
        self.beam_width = int(beam_width)
        self.time_major = time_major
        self.name = name

        # NOTE: attention_weights_tempareture is good for narrow focus.
        # Assume that β = 1 / attention_weights_tempareture,
        # and β=2 is recommended

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
        self.learning_rate_pl_list = []

    def _encode(self, inputs, inputs_seq_len,
                keep_prob_input, keep_prob_hidden):
        """Encode input features.
        Args:
            inputs: A tensor of `[batch_size, time, input_size]`
            inputs_seq_len: A tensor of `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in the
                input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in the
                hidden-hidden layers
        Returns:
            encoder_outputs: A namedtuple of
            `(outputs final_state attention_values attention_values_length)`
        """
        # Define encoder
        encoder = load_encoder(model_type='blstm_encoder')(
            num_unit=self.encoder_num_unit,
            num_layer=self.encoder_num_layer,
            parameter_init=self.parameter_init,
            clip_activation=self.clip_activation_encoder,
            num_proj=None)

        encoder_outputs = encoder(inputs=inputs,
                                  inputs_seq_len=inputs_seq_len,
                                  keep_prob_input=keep_prob_input,
                                  keep_prob_hidden=keep_prob_hidden)

        return encoder_outputs

    def _create_decoder(self, encoder_outputs, labels):
        """Create attention decoder.
        Args:
            encoder_outputs: A namedtuple of `(outputs, final_state,
                attention_values, attention_values_length)`
            labels: Target labels of size `[batch_size, time]`
        Returns:
            decoder: An instance of the decoder class
        """
        # Define attention layer (calculate attention weights)
        self.attention_layer = AttentionLayer(
            num_unit=self.attention_dim,
            attention_smoothing=self.attention_smoothing,
            attention_weights_tempareture=self.attention_weights_tempareture,
            attention_type=self.attention_type)

        # Define RNN decoder
        rnn_decoder = load_decoder(model_type='lstm_decoder')
        lstm = rnn_decoder(parameter_init=self.parameter_init,
                           num_unit=self.decoder_num_unit,
                           clip_activation=self.clip_activation_decoder)

        # Define attention decoder
        decoder = AttentionDecoder(
            cell=lstm,
            parameter_init=self.parameter_init,
            max_decode_length=self.max_decode_length,
            num_classes=self.num_classes,
            attention_encoder_states=encoder_outputs.outputs,
            attention_values=encoder_outputs.attention_values,
            attention_values_length=encoder_outputs.attention_values_length,
            attention_layer=self.attention_layer,
            time_major=self.time_major)

        return decoder

    def _build(self, inputs, labels, inputs_seq_len, labels_seq_len,
               keep_prob_input, keep_prob_hidden, keep_prob_output):
        """Define model graph.
        Args:
            inputs: A tensor of `[batch_size, time, input_size]`
            labels: A tensor of `[batch_size, time]`
            inputs_seq_len: A tensor of `[batch_size]`
            labels_seq_len: A tensor of `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in
                the input-hidden layer
            keep_prob_hidden: A float value. A probability to keep nodes in
                the hidden-hidden layers
            keep_prob_output: A float value. A probability to keep nodes in
                the hidden-output layer
        Returns:
            logits:
            decoder_outputs_train:
            decoder_outputs_infer:
        """
        # Encode input features
        encoder_outputs = self._encode(
            inputs, inputs_seq_len, keep_prob_input, keep_prob_hidden)

        # Define decoder (initialization)
        decoder_train = self._create_decoder(encoder_outputs, labels)
        decoder_infer = self._create_decoder(encoder_outputs, labels)
        # NOTE: initial_state and helper will be substituted in
        # self._decode_train() or self._decode_infer()

        # Wrap decoder only when inference
        decoder_infer = self._beam_search_decoder_wrapper(
            decoder_infer, beam_width=self.beam_width)

        # Connect between encoder and decoder
        bridge = InitialStateBridge(
            encoder_outputs=encoder_outputs,
            decoder_state_size=decoder_train.cell.state_size)

        # Call decoder (divide into training and inference)
        # Training
        decoder_outputs_train, _ = self._decode_train(
            decoder=decoder_train,
            bridge=bridge,
            encoder_outputs=encoder_outputs,
            labels=labels,
            labels_seq_len=labels_seq_len)

        # Inference
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
        logits = decoder_outputs_train.logits / self.logits_tempareture
        # NOTE: This is for better decoding.
        # See details in
        # https://arxiv.org/abs/1612.02695.
        # Chorowski, Jan, and Navdeep Jaitly.
        # "Towards better decoding and language model integration in sequence
        # to sequence models." arXiv preprint arXiv:1612.02695 (2016).

        return logits, decoder_outputs_train, decoder_outputs_infer
