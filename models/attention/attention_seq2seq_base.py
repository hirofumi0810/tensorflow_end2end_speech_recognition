#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class of attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict
import tensorflow as tf
# from .decoders.decoder_util import transpose_batch_time, flatten_dict
# from .decoders.beam_search_decoder_from_seq2seq import BeamSearchDecoder


OPTIMIZER_CLS_NAMES = {
    "adagrad": tf.train.AdagradOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "adam": tf.train.AdamOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
}

HELPERS = {
    "training": tf.contrib.seq2seq.TrainingHelper,
    "greedyembedding": tf.contrib.seq2seq.GreedyEmbeddingHelper
}


class AttentionBase(object):
    """Attention Mechanism based seq2seq model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimension of input vectors
        attention_dim:
        output_size: int, the number of nodes in output layer
        embedding_dim:
        sos_index: index of the start of sentence tag (<SOS>)
        eos_index: index of the end of sentence tag (<EOS>)
        clip_grad: A float value. Range of gradient clipping (> 0)
        weight_decay: A float value. Regularization parameter for weight decay
        beam_width: if 0, use greedy decoding
    """

    def __init__(self,
                 batch_size,
                 input_size,
                 attention_dim,
                 embedding_dim,
                 output_size,
                 sos_index,
                 eos_index,
                 clip_grad,
                 weight_decay,
                 beam_width,
                 name=None):

        # Network size
        self.batch_size = batch_size
        self.input_size = input_size
        self.attention_dim = attention_dim
        self.embedding_dim = embedding_dim
        self.num_classes = output_size

        # Regularization
        self.clip_grad = clip_grad
        self.weight_decay = weight_decay

        # Setting for seq2seq
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.beam_width = beam_width

        # Summaries for TensorBoard
        self.summaries_train = []
        self.summaries_dev = []

        self.name = name

    def __call__(self, inputs, labels, labels_seq_len):
        """Creates the model graph. See the model_fn documentation in
           tf.contrib.learn.Estimator class for a more detailed explanation."""
        with tf.variable_scope("model"):
            with tf.variable_scope(self.name):
                return self._build(inputs, labels, labels_seq_len)

    def _generate_placeholer(self):
        """Generate placeholders."""
        # `[batch_size, max_time, input_size]`
        self.inputs = tf.placeholder(tf.float32,
                                     shape=[None, None, self.input_size],
                                     name='input')

        # `[batch_size, max_time]`
        self.labels = tf.placeholder(tf.int32,
                                     shape=[None, None],
                                     name='label')

        # These are prepared for computing LER
        self.label_indices_true = tf.placeholder(tf.int64, name='indices')
        self.label_values_true = tf.placeholder(tf.int32, name='values')
        self.label_shape_true = tf.placeholder(tf.int64, name='shape')
        self.labels_sparse_true = tf.SparseTensor(self.label_indices_true,
                                                  self.label_values_true,
                                                  self.label_shape_true)
        self.label_indices_pred = tf.placeholder(tf.int64, name='indices')
        self.label_values_pred = tf.placeholder(tf.int32, name='values')
        self.label_shape_pred = tf.placeholder(tf.int64, name='shape')
        self.labels_sparse_pred = tf.SparseTensor(self.label_indices_pred,
                                                  self.label_values_pred,
                                                  self.label_shape_pred)

        # The length of input features
        self.inputs_seq_len = tf.placeholder(tf.int32,
                                             shape=[None],  # `[batch_size]`
                                             name='inputs_seq_len')
        # NOTE: change to tf.int64 if you use the bidirectional model

        # The length of target labels
        self.labels_seq_len = tf.placeholder(tf.int32,
                                             shape=[None],  # `[batch_size]`
                                             name='labels_seq_len')

        # For dropout
        self.keep_prob_input = tf.placeholder(tf.float32,
                                              name='keep_prob_input')
        self.keep_prob_hidden = tf.placeholder(tf.float32,
                                               name='keep_prob_hidden')

        # Learning rate
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    def _generate_target_embedding(self, reuse):
        """Returns the embedding used for the target sequence."""
        with tf.variable_scope("target_embedding", reuse=reuse):
            return tf.get_variable(
                name="W_embedding",
                shape=[self.num_classes, self.embedding_dim],
                initializer=tf.random_uniform_initializer(
                    -self.parameter_init,
                    self.parameter_init))
        # TODO: Consider shape of target_embedding

    def _encode(self):
        """Encode input features."""
        NotImplementedError

    def _create_decoder(self):
        """Create attention decoder."""
        NotImplementedError

    def define(self):
        """Define model graph."""
        NotImplementedError

    def choose_top_k(scores_flat, config):
        """Chooses the top-k beams as successors.
        Args:
            scores:
            config:
        Returns:
            next_beam_scores:
            word_indices:
        """
        next_beam_scores, word_indices = tf.nn.top_k(
            scores_flat, k=config.beam_width)
        return next_beam_scores, word_indices

    # def _beam_search_decoder_wrapper(self, decoder, beam_width=None,
    #                                  length_penalty_weight=0.0):
    #     """Wraps a decoder into a Beam Search decoder.
    #     Args:
    #         decoder: The decoder class instance
    #         beam_width: Number of beams to use, an integer
    #         length_penalty_weight: Weight for the length penalty factor. 0.0
    #             disables the penalty.
    #         choose_successors_fn: A function used to choose beam successors
    #             based on their scores.
    #             Maps from (scores, config) => (chosen scores, chosen_ids)
    #     Returns:
    #         A BeamSearchDecoder with the same interfaces as the decoder.
    #     """
    #     if beam_width is None:
    #         # Greedy decoding
    #         return decoder
    #
    #     config = BeamSearchConfig(
    #         beam_width=beam_width,
    #         vocab_size=self.num_classes,
    #         eos_token=self.eos_index,
    #         length_penalty_weight=length_penalty_weight,
    #         choose_successors_fn=self.choose_top_k)
    #
    #     return BeamSearchDecoder(decoder=decoder,
    #                              config=config)

    @property
    def decode(self):
        """Return operation for decoding."""
        NotImplementedError

    def _decode_train(self, decoder, bridge, encoder_outputs, labels,
                      labels_seq_len):
        """Runs decoding in training mode.
        Args:
            decoder: An instance of the decoder class
            bridge:
            encoder_outputs:
            labels: Target labels of size `[batch_size, max_time, num_classes]`
            labels_seq_len: The length of target labels
        Returns:
            decoder_outputs: A tuple of `(AttentionDecoderOutput, final_state)`
        """
        # Convert target labels to one-hot vectors of size
        # `[batch_size, max_time, num_classes]`
        # labels = tf.one_hot(labels,
        #                     depth=self.num_classes,
        #                     on_value=1.0,
        #                     off_value=0.0,
        #                     axis=-1)

        # Generate embedding of target labels
        target_embedding = self._generate_target_embedding(reuse=False)
        target_embedded = tf.nn.embedding_lookup(target_embedding,
                                                 labels)

        helper_train = tf.contrib.seq2seq.TrainingHelper(
            inputs=target_embedded[:, :-1, :],  # embedding of target labels
            # inputs=labels[:, :-1, :],
            sequence_length=labels_seq_len - 1,
            time_major=False)

        decoder_initial_state = bridge(reuse=False)

        # Call decoder class
        (decoder_outputs, final_state) = decoder(
            initial_state=decoder_initial_state,
            helper=helper_train,
            mode=tf.contrib.learn.ModeKeys.TRAIN)

        return (decoder_outputs, final_state)

    def _decode_infer(self, decoder, bridge, encoder_outputs):
        """Runs decoding in inference mode.
        Args:
            decoder: An instance of the decoder class
            bridge:
        Returns:
            decoder_outputs: A tuple of `(AttentionDecoderOutput, final_state)`
        """
        batch_size = tf.shape(self.inputs)[0]
        # if self.use_beam_search:
        #     batch_size = self.beam_width
        # TODO: why?

        target_embedding = self._generate_target_embedding(reuse=True)

        helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            # embedding=self.decoder_outputs_train.logits,
            embedding=target_embedding,  # embedding of predicted labels
            # start_tokens=tf.fill([batch_size], self.sos_index),
            start_tokens=tf.tile([self.sos_index], [batch_size]),
            end_token=self.eos_index)
        # ex.)
        # Output tensor has shape [2, 3].
        # tf.fill([2, 3], 9) ==> [[9, 9, 9]
        #                         [9, 9, 9]]
        # TODO: beam_search_decoder

        decoder_initial_state = bridge(reuse=True)

        # Call decoder class
        (decoder_outputs, final_state) = decoder(
            initial_state=decoder_initial_state,
            helper=helper_infer,
            mode=tf.contrib.learn.ModeKeys.INFER)

        return (decoder_outputs, final_state)

    def compute_loss(self):
        """Operation for computing cross entropy sequence loss.
        Returns:
            loss: operation for computing cross entropy sequence loss.
                  This is a single scalar tensor to minimize.
        """
        # Calculate loss per example
        logits = self.decoder_outputs_train.logits
        max_time = tf.shape(self.labels[:, 1:])[1]
        loss_mask = tf.sequence_mask(tf.to_int32(self.labels_seq_len - 1),
                                     maxlen=max_time,
                                     dtype=tf.float32)
        losses = tf.contrib.seq2seq.sequence_loss(
            logits=logits,
            targets=self.labels[:, 1:],
            weights=loss_mask,
            average_across_timesteps=True,
            average_across_batch=False,
            softmax_loss_function=None)

        # Calculate the average log perplexity
        # self.loss = tf.reduce_sum(losses) / tf.to_float(
        #     tf.reduce_sum(self.labels_seq_len - 1))
        self.loss = tf.reduce_sum(losses)

        # Add a scalar summary for the snapshot of loss
        self.summaries_train.append(
            tf.summary.scalar('loss_train', self.loss))
        self.summaries_dev.append(
            tf.summary.scalar('loss_dev', self.loss))

        return self.loss

    def train(self, optimizer, learning_rate_init=None,
              clip_gradients_by_norm=None, is_scheduled=False):
        """Operation for training.
        Args:
            optimizer: string, name of the optimizer in OPTIMIZER_CLS_NAMES
            learning_rate_init: initial learning rate
            clip_gradients_by_norm: if True, clip gradients by norm of the
                value of self.clip_grad
            is_scheduled: if True, schedule learning rate at each epoch
        Returns:
            train_op: operation for training
        """
        optimizer = optimizer.lower()
        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer's name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))
        if learning_rate_init < 0.0:
            raise ValueError("Invalid learning_rate %s.", learning_rate_init)

        # Select parameter update method
        if is_scheduled:
            learning_rate = self.learning_rate
        else:
            learning_rate = learning_rate_init

        if optimizer == 'momentum':
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate,
                momentum=0.9)
        else:
            self.optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate)

        # TODO: Optionally wrap with SyncReplicasOptimizer
        # TODO: create_learning_rate_decay_fn

        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Gradient clipping
        if self.clip_grad is not None:
            # Compute gradients
            trainable_vars = tf.trainable_variables()
            grads = tf.gradients(self.loss, trainable_vars)
            # TODO: Optionally add gradient noise

            if clip_gradients_by_norm:
                # Clip by norm
                self.clipped_grads = [tf.clip_by_norm(
                    g,
                    clip_norm=self.clip_grad) for g in grads if g is not None]
            else:
                # Clip by absolute values
                self.clipped_grads = [tf.clip_by_value(
                    g,
                    clip_value_min=-self.clip_grad,
                    clip_value_max=self.clip_grad) for g in grads if g is not None]

            # TODO: Add histograms for variables, gradients (norms)
            # TODO: なんでNoneが発生した？

            # Create gradient updates
            train_op = self.optimizer.apply_gradients(
                zip(self.clipped_grads, trainable_vars),
                global_step=global_step,
                name='train')

        # Use the optimizer to apply the gradients that minimize the loss
        # and also increment the global step counter as a single training step
        train_op = self.optimizer.minimize(self.loss, global_step=global_step)

        return train_op

    def _add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
        """Adds scaled noise from a 0-mean normal distribution to gradients."""
        raise NotImplementedError

    def decoder(self, decode_type, beam_width=None):
        """Operation for decoding.
        Args:
            decode_type: greedy or beam_search
            beam_width: beam width for beam search
        Return:
            decoded_train: operation for decoding in training
            decoded_infer: operation for decoding in inference
        """
        if decode_type not in ['greedy', 'beam_search']:
            raise ValueError('decode_type is "greedy" or "beam_search".')

        if decode_type == 'greedy':
            decoded_train = self.decoder_outputs_train.predicted_ids
            decoded_infer = self.decoder_outputs_infer.predicted_ids

        elif decode_type == 'beam_search':
            if beam_width is None:
                raise ValueError('Set beam_width.')
            NotImplementedError

        return decoded_train, decoded_infer

    def compute_ler(self):
        """Operation for computing LER (Label Error Rate).
        Return:
            ler_op: operation for computing LER
        """
        # Compute LER (normalize by label length)
        ler_op = tf.reduce_mean(tf.edit_distance(
            self.labels_sparse_pred, self.labels_sparse_true, normalize=True))
        # TODO: ここでの編集距離はラベルだから，文字に変換しないと正しいCERは得られない
        # TODO: パディングを考慮して計算する

        # Add a scalar summary for the snapshot of LER
        with tf.name_scope("ler"):
            self.summaries_train.append(tf.summary.scalar(
                'ler_train', ler_op))
            self.summaries_dev.append(tf.summary.scalar(
                'ler_dev', ler_op))

        return ler_op

    def attention_weights(self):
        return self.decoder_outputs_infer.attention_scores
