#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention model class based on BLSTM encoder and LSTM decoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .attention_seq2seq_base import AttentionBase
from .encoders.load_encoder import load as load_encoder
from .decoders.load_decoder import load as load_decoder
from .decoders.attention_layer import AttentionLayer
from .decoders.attention_decoder import AttentionDecoder
from .bridge import InitialStateBridge


class BLSTMAttetion(AttentionBase):
    """Bidirectional LSTM based Attention model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimension of input vectors
        encoder_num_unit: int, the number of units in each layer of the
            encoder
        encoder_num_layer: int, the number of layers of the encoder
        attention_dim: int,
        decoder_num_unit: int, the number of units in each layer of the
            decoder
        decoder_num_layer: int, the number of layers of the decoder
        embedding_dim: int, the dimension of the embedding in target spaces
        output_size: int, the number of nodes in softmax layer
        sos_index: index of the start of sentence tag (<SOS>)
        eos_index: index of the end of sentence tag (<EOS>)
        max_decode_length:
        attention_weights_tempareture:
        logits_tempareture:
        parameter_init: A float value. Range of uniform distribution to
            initialize weight parameters
        clip_grad: A float value. Range of gradient clipping (non-negative)
        clip_activation_encoder: A float value. Range of activation clipping in
            the encoder (> 0)
        clip_activation_decoder:
        dropout_ratio_input: A float value. Dropout ratio in input-hidden
            layers
        dropout_ratio_hidden: A float value. Dropout ratio in hidden-hidden
            layers
        weight_decay:
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 encoder_num_unit,
                 encoder_num_layer,
                 attention_dim,
                 decoder_num_unit,
                 decoder_num_layer,
                 embedding_dim,
                 output_size,
                 sos_index,
                 eos_index,
                 max_decode_length,
                 attention_weights_tempareture=1,
                 logits_tempareture=1,
                 parameter_init=0.1,
                 clip_grad=5.0,
                 clip_activation_encoder=50,
                 clip_activation_decoder=50,
                 dropout_ratio_input=1.0,
                 dropout_ratio_hidden=1.0,
                 weight_decay=0.0,
                 beam_width=0,
                 name='blstm_attention_seq2seq'):

        AttentionBase.__init__(self, batch_size, input_size,
                               attention_dim, embedding_dim,
                               output_size, sos_index, eos_index,
                               logits_tempareture,
                               clip_grad, weight_decay, beam_width, name)

        # Network size
        self.encoder_num_unit = encoder_num_unit
        self.encoder_num_layer = encoder_num_layer
        self.decoder_num_unit = decoder_num_unit
        self.decoder_num_layer = decoder_num_layer

        # Regularization
        self.parameter_init = parameter_init
        self.clip_activation_encoder = clip_activation_encoder
        self.clip_activation_decoder = clip_activation_decoder
        if dropout_ratio_input == 1.0 and dropout_ratio_hidden == 1.0:
            self.dropout = False
        else:
            self.dropout = True
        self.dropout_ratio_input = dropout_ratio_input
        self.dropout_ratio_hidden = dropout_ratio_hidden

        # Setting for se2seq
        self.max_decode_length = max_decode_length
        self.attention_weights_tempareture = attention_weights_tempareture
        # NOTE: attention_weights_tempareture is good for narrow focus.
        # Assume that β = 1 / attention_weights_tempareture, β=2 is recommended.

    def _encode(self, inputs, inputs_seq_len):
        """Encode input features.
        Args:
            inputs:
            inputs_seq_len:
        Returns:
            encoder_outputs: A namedtaple of
            `(outputs final_state attention_values attention_values_length)`
        """
        # Define encoder
        encoder = load_encoder(model_type='blstm_encoder')(
            keep_prob_input=self.keep_prob_input,
            keep_prob_hidden=self.keep_prob_hidden,
            num_unit=self.encoder_num_unit,
            num_layer=self.encoder_num_layer,
            parameter_init=self.parameter_init,
            clip_activation=self.clip_activation_encoder,
            num_proj=None)

        encoder_outputs = encoder(inputs=inputs,
                                  inputs_seq_len=inputs_seq_len)

        return encoder_outputs

    def _create_decoder(self, encoder_outputs, labels):
        """Create attention decoder.
        Args:
            encoder_outputs: A tuple of `()`
            labels: Target labels of sise `[]`
        Returns:
            decoder: The decoder class instance
        """
        # Define attention layer (calculate attention weights)
        self.attention_layer = AttentionLayer(
            num_unit=self.attention_dim,
            attention_weights_tempareture=self.attention_weights_tempareture,
            attention_type='bahdanau')

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
            attention_layer=self.attention_layer)

        return decoder

    def define(self):
        """Define model graph."""

        # Generate placeholders
        self._generate_placeholer()

        # Encode input features
        encoder_outputs = self._encode(self.inputs, self.inputs_seq_len)

        # Define decoder (initialization)
        decoder_train = self._create_decoder(encoder_outputs, self.labels)
        decoder_infer = self._create_decoder(encoder_outputs, self.labels)
        # NOTE: initial_state and helper will be substituted in
        # self._decode_train() or self._decode_infer()

        # Wrap decoder
        # decoder_train = self._beam_search_decoder_wrapper(decoder_train,
        #                                                   beam_width=20)
        # decoder_infer = self._beam_search_decoder_wrapper(decoder_infer,
        #                                                   beam_width=20)

        # Connect between encoder and decoder
        bridge = InitialStateBridge(
            encoder_outputs=encoder_outputs,
            decoder_state_size=decoder_train.cell.state_size)

        # Call decoder (divide into training and inference)
        # Training
        self.decoder_outputs_train, _ = self._decode_train(
            decoder=decoder_train,
            bridge=bridge,
            encoder_outputs=encoder_outputs,
            labels=self.labels,
            labels_seq_len=self.labels_seq_len)

        # Inference
        self.decoder_outputs_infer, _ = self._decode_infer(
            decoder=decoder_infer,
            bridge=bridge,
            encoder_outputs=encoder_outputs)
