#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class of attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple, OrderedDict
import tensorflow as tf
from .decoders.decoder_util import transpose_batch_time, flatten_dict
from .decoders.beam_search_decoder import BeamSearchDecoder


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


class BeamSearchConfig(namedtuple(
        "BeamSearchConfig",
        [
            "beam_width",
            "vocab_size",
            "eos_token",
            "length_penalty_weight",
            "choose_successors_fn"
        ])):
    pass


def templatemethod(name_):
    """This decorator wraps a method with `tf.make_template`. For example,

    @templatemethod
    def my_method():
      # Create variables
    """

    def template_decorator(func):
        """Inner decorator function"""

        def func_wrapper(*args, **kwargs):
            """Inner wrapper function"""
            templated_func = tf.make_template(name_, func)
            return templated_func(*args, **kwargs)

        return func_wrapper

    return template_decorator


class AttentionBase(object):
    """Attention Mechanism based seq2seq model.
    Args:
        batch_size: int, batch size of mini batch
        input_size: int, the dimension of input vectors
        attention_dim:
        output_size: int, the number of nodes in output layer
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
        labels = tf.one_hot(labels,
                            depth=self.num_classes,
                            on_value=1.0,
                            off_value=0.0,
                            axis=-1)

        helper_train = tf.contrib.seq2seq.TrainingHelper(
            # inputs=target_embedded[:, :-1],  # 正解ラベルの埋め込みベクトル
            inputs=labels[:, :-1],
            sequence_length=labels_seq_len - 1,
            time_major=False)

        decoder_initial_state = bridge()

        # Call decoder class
        decoder_outputs, final_state = decoder(
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

        helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            # embedding=decoder.logits,  # 出力ラベルの埋め込みベクトル
            embedding=self.target_embedding,
            start_tokens=tf.fill([batch_size], self.sos_index),
            # or tf.tile([self.sos_index], [batch_size]),
            end_token=self.eos_index)
        # ex.)
        # Output tensor has shape [2, 3].
        # tf.fill([2, 3], 9) ==> [[9, 9, 9]
        #                         [9, 9, 9]]
        # TODO: beam_search_decoder

        decoder_initial_state = bridge()

        # Call decoder class
        decoder_outputs, final_state = decoder(
            initial_state=decoder_initial_state,
            helper=helper_infer,
            mode=tf.contrib.learn.ModeKeys.INFER)

        return (decoder_outputs, final_state)

    @property
    @templatemethod("target_embedding")
    def target_embedding(self):
        """Returns the embedding used for the target sequence."""
        return tf.get_variable(
            name="W_embedding",
            shape=[self.num_classes, self.num_classes],
            initializer=tf.random_uniform_initializer(
                -self.parameter_init,
                self.parameter_init))
        # TODO: Consider shape of target_embedding

    # def _create_predictions(self, decoder_output, features, labels,
    #                         losses=None):
    #     """Creates the dictionary of predictions that is returned by the
    #     model.
    #     """
    #     predictions = {}
    #
    #     # Add features and, if available, labels to predictions
    #     predictions.update(flatten_dict({"features": features}))
    #     if labels is not None:
    #         predictions.update(flatten_dict({"labels": labels}))
    #
    #     if losses is not None:
    #         predictions["losses"] = transpose_batch_time(losses)
    #
    #     # Decoders returns output in time-major form
    #     # `[max_time, batch_size, ...]`
    #     # Here we transpose everything back to batch-major for the user
    #     output_dict = OrderedDict(
    #         zip(decoder_output._fields, decoder_output))
    #     decoder_output_flat = flatten_dict(output_dict)
    #     decoder_output_flat = {
    #         k: transpose_batch_time(v)
    #         for k, v in decoder_output_flat.items()
    #     }
    #     predictions.update(decoder_output_flat)
    #
    #     return predictions

    def _cross_entropy_sequence_loss(self, logits, labels, labels_seq_len):
        """Calculates the per-example cross-entropy loss for a sequence of
           logits and masks out all losses passed the sequence length.
        Args:
            logits: Logits of shape `[max_time, batch_size, num_classes]`
            labels: Target labels of shape `[max_time, batch_size]`
            labels_seq_len: An int32 tensor of shape `[batch_size]`
                corresponding to the length of each input to the decoder.
                Note that this is different from seq_len, which is the length
                of each input to the encoder.
        Returns:
            A tensor of shape [max_time, batch_size] that contains the loss per
                example, per time step.
        """
        with tf.name_scope("cross_entropy_sequence_loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)

            # Mask out the losses we don't care about
            loss_mask = tf.sequence_mask(
                tf.to_int32(labels_seq_len), tf.to_int32(tf.shape(labels)[0]))
            losses = losses * tf.transpose(tf.to_float(loss_mask), [1, 0])

            return losses

    def _compute_loss(self, decoder_outputs, labels, labels_seq_len):
        """Operation for computing cross entropy sequence loss.
        Args:
            decoder_outputs: An instance of AttentionDecoderOutput
            labels: Target lables of shape `[max_time, batch_size]`
            labels_seq_len: The length of target labels of shape `[batch_size]`
        Returns:
            loss: operation for computing cross entropy sequence loss.
                  This is a single scalar tensor to minimize
            losses: the per-batch losses
        """
        # Calculate loss per example-timestep of shape `[batch_size, max_time]`
        self.losses = self._cross_entropy_sequence_loss(
            logits=decoder_outputs.logits[:, :, :],
            labels=tf.transpose(labels[:, 1:], [1, 0]),
            labels_seq_len=labels_seq_len - 1)

        # Calculate the average log perplexity
        self.loss = tf.reduce_sum(self.losses) / tf.to_float(
            tf.reduce_sum(labels_seq_len - 1))

        # Add a scalar summary for the snapshot of loss
        self.summaries_train.append(
            tf.summary.scalar('loss_train', self.loss))
        self.summaries_dev.append(
            tf.summary.scalar('loss_dev', self.loss))

        return self.losses, self.loss

    def train(self, optimizer, learning_rate_init=None,
              clip_gradients_by_norm=None, is_scheduled=False):
        """Operation for training.
        Args:
            optimizer: adam or adadelta or rmsprop or sgd or momentum
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

    # def ler(self, decode_op):
    #     """Operation for computing LER.
    #     Args:
    #         decode_op: operation for decoding
    #     Return:
    #         ler_op: operation for computing label error rate
    #     """
    # compute phone error rate (normalize by label length)
    # ler_op = tf.reduce_mean(tf.edit_distance(decode_op, self.labels,
    #                                          normalize=True))
    #     # TODO: ここでの編集距離は数字だから，文字に変換しないと正しい結果は得られない
    #
    #     # add a scalar summary for the snapshot ler
    # self.summaries_train.append(tf.summary.scalar(
    #     'Label Error Rate (train)', ler_op))
    # self.summaries_dev.append(tf.summary.scalar(
    #     'Label Error Rate (dev)', ler_op))
    #     return ler_op
