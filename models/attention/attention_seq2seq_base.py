#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class of attention model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.attention.decoders.beam_search.util import choose_top_k
from models.attention.decoders.beam_search.namedtuple import BeamSearchConfig
from models.attention.decoders.beam_search.beam_search_decoder import BeamSearchDecoder


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
        attention_dim: int, the dimension of attention vecors
        num_classes: int, the number of nodes in output layer
        embedding_dim: int, the dimension of target embedding
        sos_index: index of the start of sentence tag (<SOS>)
        eos_index: index of the end of sentence tag (<EOS>)
        clip_grad: A float value. Range of gradient clipping (> 0)
        weight_decay: A float value. Regularization parameter for weight decay
        beam_width: if equal to 1, use greedy decoding
    """

    def __init__(self):
        pass

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

    def _beam_search_decoder_wrapper(self, decoder, beam_width=None,
                                     length_penalty_weight=0.0):
        """Wraps a decoder into a Beam Search decoder.
        Args:
            decoder: An instance of RNNDecoder class
            beam_width: int, the number of beams to use
            length_penalty_weight: Weight for the length penalty factor. 0.0
                disables the penalty.
            choose_successors_fn: A function used to choose beam successors
                based on their scores.
                Maps from (scores, config) => (chosen scores, chosen_ids)
        Returns:
            A BeamSearchDecoder with the same interfaces as the decoder.
        """
        if beam_width is None or beam_width == 1:
            # Greedy decoding
            return decoder

        config = BeamSearchConfig(
            beam_width=beam_width,
            vocab_size=self.num_classes,
            eos_token=self.eos_index,
            length_penalty_weight=length_penalty_weight,
            choose_successors_fn=choose_top_k)

        return BeamSearchDecoder(decoder=decoder, config=config)

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

        # if self.use_beam_search:
        #     batch_size = self.beam_width
        # TODO: why?

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
        # TODO: beam_search_decoder

        decoder_initial_state = bridge(reuse=True)

        # Call decoder class
        (decoder_outputs, final_state) = decoder(
            initial_state=decoder_initial_state,
            helper=helper_infer,
            mode=tf.contrib.learn.ModeKeys.INFER)
        # NOTE: They are time-major if self.time_major is True

        return (decoder_outputs, final_state)

    def compute_loss(self, inputs, labels, inputs_seq_len, labels_seq_len,
                     keep_prob_input, keep_prob_hidden, num_gpu=1, scope=None):
        """Operation for computing cross entropy sequence loss.
        Args:
            inputs: A tensor of `[batch_size, time, input_size]`
            labels: A tensor of `[batch_size, time]`
            inputs_seq_len: A tensor of `[batch_size]`
            labels_seq_len: A tensor of `[batch_size]`
            keep_prob_input: A float value. A probability to keep nodes in
                input-hidden layers
            keep_prob_hidden: A float value. A probability to keep nodes in
                hidden-hidden layers
            num_gpu: int, the number of GPUs
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
            keep_prob_input, keep_prob_hidden)

        # For prevent 0 * log(0) in crossentropy loss
        epsilon = tf.constant(value=1e-10)  # shope??
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
                average_across_batch=False,
                softmax_loss_function=None)

            # Calculate the average log perplexity
            # self.loss = tf.reduce_sum(losses) / tf.to_float(
            #     tf.reduce_sum(self.labels_seq_len - 1))
            sequence_loss = tf.reduce_sum(sequence_losses,
                                          name='sequence_loss_mean')
            tf.add_to_collection('losses', sequence_loss)

        # Compute total loss
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        if num_gpu == 1:
            # Add a scalar summary for the snapshot of loss
            self.summaries_train.append(
                tf.summary.scalar('loss_train', loss))
            self.summaries_dev.append(
                tf.summary.scalar('loss_dev', loss))

        return loss, logits, decoder_outputs_train, decoder_outputs_infer

    def train(self, loss, optimizer, learning_rate_init=None,
              clip_grad_by_norm=False, is_scheduled=False):
        """Operation for training.
        Args:
            loss: An operation for computing loss
            optimizer: string, name of the optimizer in OPTIMIZER_CLS_NAMES
            learning_rate_init: initial learning rate
            clip_grad_by_norm: if True, clip gradients by norm of the
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

        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # Select optimizer
        if is_scheduled:
            learning_rate_init = self.lr

        if optimizer == 'momentum':
            optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate_init,
                momentum=0.9)
        else:
            optimizer = OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate_init)

        # TODO: Optionally wrap with SyncReplicasOptimizer
        # TODO: create_learning_rate_decay_fn

        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if self.clip_grad is not None:
            # Gradient clipping
            train_op = self._gradient_clipping(loss,
                                               optimizer,
                                               clip_grad_by_norm,
                                               global_step)
        else:
            # Use the optimizer to apply the gradients that minimize the loss
            # and also increment the global step counter as a single training
            # step
            train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def _gradient_clipping(self, loss, optimizer, clip_grad_by_norm,
                           global_step):
        # Compute gradients
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(loss, trainable_vars)

        # TODO: Optionally add gradient noise

        if clip_grad_by_norm:
            # Clip by norm
            clipped_grads = []
            for grad, var in zip(grads, trainable_vars):
                if grad is not None:
                    grad = tf.clip_by_norm(grad,
                                           clip_norm=self.clip_grad)
                    clipped_grads.append(grad)

                    # Add histograms for gradients.
                    # self.summaries_train.append(
                    # tf.summary.histogram(var.op.name + '/gradients', grad))

        else:
            # Clip by absolute values
            clipped_grads = []
            for grad, var in zip(grads, trainable_vars):
                if grad is not None:
                    grad = tf.clip_by_value(
                        grad,
                        clip_value_min=-self.clip_grad,
                        clip_value_max=self.clip_grad)
                    clipped_grads.append(grad)

                    # Add histograms for gradients.
                    # self.summaries_train.append(
                    #     tf.summary.histogram(var.op.name + '/gradients', grad))
                    # TODO: Why None occured?

                    # self._tensorboard_statistics(trainable_vars)

        # Create gradient updates
        train_op = optimizer.apply_gradients(
            zip(clipped_grads, trainable_vars),
            global_step=global_step,
            name='train')

        return train_op

    def _add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
        """Adds scaled noise from a 0-mean normal distribution to gradients."""
        raise NotImplementedError

    def decoder(self, decoder_outputs_train, decoder_outputs_infer,
                decode_type, beam_width=None):
        """Operation for decoding.
        Args:
            decoder_outputs_train:
            decoder_outputs_infer:
            decode_type: greedy or beam_search
            beam_width: beam width for beam search
        Return:
            decoded_train: operation for decoding in training
            decoded_infer: operation for decoding in inference
        """
        if decode_type not in ['greedy', 'beam_search']:
            raise ValueError('decode_type is "greedy" or "beam_search".')

        if decode_type == 'greedy':
            decoded_train = decoder_outputs_train.predicted_ids
            decoded_infer = decoder_outputs_infer.predicted_ids

        elif decode_type == 'beam_search':
            if beam_width is None:
                raise ValueError('Set beam_width.')
            raise NotImplementedError

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
        # TODO: ここでの編集距離はラベルだから，文字に変換しないと正しいCERは得られない
        # TODO: パディングを考慮して計算する

        # Add a scalar summary for the snapshot of LER
        # with tf.name_scope("ler"):
        #     self.summaries_train.append(tf.summary.scalar(
        #         'ler_train', ler_op))
        #     self.summaries_dev.append(tf.summary.scalar(
        #         'ler_dev', ler_op))
        # TODO: feed_dictのタイミング違うからエラーになる

        return ler_op
