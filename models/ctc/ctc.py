#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""CTC model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.encoders.load_encoder import load

OPTIMIZER_CLS_NAMES = {
    "adagrad": tf.train.AdagradOptimizer,
    "adadelta": tf.train.AdadeltaOptimizer,
    "adam": tf.train.AdamOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "nestrov": tf.train.MomentumOptimizer
}


class CTC(object):
    """Connectionist Temporal Classification (CTC) network.
    Args:
        encoder_type (string): The type of an encoder
            blstm: Bidirectional LSTM
            lstm: Unidirectional LSTM
            bgru: Bidirectional GRU
            gru: Unidirectional GRU
            vgg_blstm: VGG + Bidirectional LSTM
            vgg_lstm: VGG + Unidirectional LSTM
        input_size (int): the dimensions of input vectors
        num_units (int): the number of units in each layer
        num_layers (int): the number of layers
        num_classes (int): the number of classes of target labels
            (except for a blank label)
        lstm_impl (string, optional): a base implementation of LSTM. This is
            not used for GRU models.
                BasicLSTMCell: tf.contrib.rnn.BasicLSTMCell (no peephole)
                LSTMCell: tf.contrib.rnn.LSTMCell
                LSTMBlockCell: tf.contrib.rnn.LSTMBlockCell
                LSTMBlockFusedCell: under implementation
                CudnnLSTM: under implementation
            Choose the background implementation of tensorflow.
            Default is LSTMBlockCell.
        use_peephole (bool, optional): if True, use peephole connection. This
            is not used for GRU models.
        splice (int, optional): the number of frames to splice. This is used
            when using CNN-like encoder. Default is 1 frame.
        parameter_init (float, optional): the range of uniform distribution to
            initialize weight parameters (>= 0)
        clip_grad (float, optional): the range of gradient clipping (> 0)
        clip_activation (float, optional): the range of activation clipping
            (> 0). This is not used for GRU models.
        num_proj (int, optional): the number of nodes in the projection layer.
            This is not used for GRU models.
        weight_decay (float, optional): a parameter for weight decay
        bottleneck_dim (int, optional): the dimensions of the bottleneck layer
    """

    def __init__(self,
                 encoder_type,
                 input_size,
                 num_units,
                 num_layers,
                 num_classes,
                 lstm_impl='LSTMBlockCell',
                 use_peephole=True,
                 splice=1,
                 parameter_init=0.1,
                 clip_grad=None,
                 clip_activation=None,
                 num_proj=None,
                 weight_decay=0.0,
                 bottleneck_dim=None):

        super(CTC, self).__init__()

        assert input_size % 3 == 0, 'input_size must be divisible by 3 (+ delta, double delta features).'
        assert splice % 2 == 1, 'splice must be the odd number'
        if clip_grad is not None:
            assert float(clip_grad) > 0, 'clip_grad must be larger than 0.'
        assert float(
            weight_decay) >= 0, 'weight_decay must not be a negative value.'

        # Encoder setting
        self.encoder_type = encoder_type
        self.input_size = input_size
        self.splice = splice
        self.num_units = num_units
        if int(num_proj) == 0:
            self.num_proj = None
        elif num_proj is not None:
            self.num_proj = int(num_proj)
        else:
            self.num_proj = None
        self.num_layers = num_layers
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes + 1  # + blank
        self.lstm_impl = lstm_impl
        self.use_peephole = use_peephole

        # Regularization
        self.parameter_init = parameter_init
        self.clip_grad = clip_grad
        self.clip_activation = clip_activation
        self.weight_decay = weight_decay

        # Summaries for TensorBoard
        self.summaries_train = []
        self.summaries_dev = []

        # Placeholders
        self.inputs_pl_list = []
        self.labels_pl_list = []
        self.inputs_seq_len_pl_list = []
        self.keep_prob_pl_list = []

        self.name = encoder_type + '_ctc'

        if encoder_type in ['blstm', 'lstm']:
            self.encoder = load(encoder_type)(
                num_units=num_units,
                num_proj=self.num_proj,
                num_layers=num_layers,
                lstm_impl=lstm_impl,
                use_peephole=use_peephole,
                parameter_init=parameter_init,
                clip_activation=clip_activation)

        elif encoder_type in ['vgg_blstm', 'vgg_lstm']:
            self.encoder = load(encoder_type)(
                input_size=input_size,
                splice=splice,
                num_units=num_units,
                num_proj=self.num_proj,
                num_layers=num_layers,
                lstm_impl=lstm_impl,
                use_peephole=use_peephole,
                parameter_init=parameter_init,
                clip_activation=clip_activation)

        elif encoder_type in ['bgru', 'gru']:
            self.encoder = load(encoder_type)(
                num_units=num_units,
                num_layers=num_layers,
                parameter_init=parameter_init)

        elif encoder_type in ['vgg_wang', 'resnet_wang', 'cnn_zhang']:
            self.encoder = load(encoder_type)(
                input_size=input_size,
                splice=splice,
                parameter_init=parameter_init)

        else:
            self.encoder = None

    def _build(self, inputs, inputs_seq_len, keep_prob):
        """Construct model graph.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            inputs_seq_len (placeholder): A tensor of size` [B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
        Returns:
            logits: A tensor of size `[T, B, num_classes]`
        """
        # inputs: `[B, T, input_size]`
        batch_size = tf.shape(inputs)[0]

        encoder_outputs, final_state = self.encoder(
            inputs, inputs_seq_len, keep_prob)

        # Reshape to apply the same weights over the timesteps
        if final_state is None:
            # CNN-like topology such as VGG and ResNet
            output_dim = tf.shape(
                encoder_outputs).get_shape().as_list()[-1]
            outputs_2d = tf.reshape(
                encoder_outputs, shape=[-1, output_dim])
        elif 'lstm' not in self.encoder_type or self.num_proj is None:
            if 'b' in self.encoder_type:
                # bidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_units * 2])
            else:
                # unidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_units])
        else:
            if 'b' in self.encoder_type:
                # bidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_proj * 2])
            else:
                # unidirectional
                outputs_2d = tf.reshape(
                    encoder_outputs, shape=[-1, self.num_proj])

        if self.bottleneck_dim is not None and self.bottleneck_dim != 0:
            with tf.variable_scope('bottleneck') as scope:
                outputs_2d = tf.contrib.layers.fully_connected(
                    outputs_2d,
                    num_outputs=self.bottleneck_dim,
                    activation_fn=tf.nn.relu,
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=self.parameter_init),
                    biases_initializer=tf.zeros_initializer(),
                    scope=scope)

            # Dropout for the hidden-output connections
            outputs_2d = tf.nn.dropout(
                outputs_2d, keep_prob, name='dropout_bottleneck')

        with tf.variable_scope('output') as scope:
            logits_2d = tf.contrib.layers.fully_connected(
                outputs_2d,
                num_outputs=self.num_classes,
                activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                    stddev=self.parameter_init),
                biases_initializer=tf.zeros_initializer(),
                scope=scope)

            # Reshape back to the original shape
            logits = tf.reshape(
                logits_2d, shape=[batch_size, -1, self.num_classes])

            # Convert to time-major: `[T, B, num_classes]'
            logits = tf.transpose(logits, (1, 0, 2))

        return logits

    def create_placeholders(self):
        """Create placeholders and append them to list."""
        self.inputs_pl_list.append(
            tf.placeholder(tf.float32,
                           shape=[None, None, self.input_size * self.splice],
                           name='input'))
        self.labels_pl_list.append(
            tf.SparseTensor(tf.placeholder(tf.int64, name='indices'),
                            tf.placeholder(tf.int32, name='values'),
                            tf.placeholder(tf.int64, name='shape')))
        self.inputs_seq_len_pl_list.append(
            tf.placeholder(tf.int32, shape=[None], name='inputs_seq_len'))
        self.keep_prob_pl_list.append(
            tf.placeholder(tf.float32, name='keep_prob'))

    def _add_noise_to_inputs(self, inputs, stddev=0.075):
        """Add gaussian noise to the inputs.
        Args:
            inputs: the noise free input-features.
            stddev (float, optional): The standart deviation of the noise.
                Default is 0.075.
        Returns:
            inputs: Input features plus noise.
        """
        # if stddev != 0:
        #     with tf.variable_scope("input_noise"):
        #         # Add input noise with a standart deviation of stddev.
        #         inputs = tf.random_normal(
        #             tf.shape(inputs), 0.0, stddev) + inputs
        # return inputs
        raise NotImplementedError

    def _add_noise_to_gradients(grads_and_vars, gradient_noise_scale,
                                stddev=0.075):
        """Adds scaled noise from a 0-mean normal distribution to gradients.
        Args:
            grads_and_vars:
            gradient_noise_scale:
            stddev (float):
        Returns:
        """
        raise NotImplementedError

    def compute_loss(self, inputs, labels, inputs_seq_len,
                     keep_prob, scope=None):
        """Operation for computing ctc loss.
        Args:
            inputs: A tensor of size `[B, T, input_size]`
            labels: A SparseTensor of target labels
            inputs_seq_len: A tensor of size `[B]`
            keep_prob (placeholder, float): A probability to keep nodes
                in the hidden-hidden connection
            scope (optional): A scope in the model tower
        Returns:
            total_loss: operation for computing total ctc loss
            logits: A tensor of size `[T, B, input_size]`
        """
        # Build model graph
        logits = self._build(inputs, inputs_seq_len, keep_prob)

        # Weight decay
        if self.weight_decay > 0:
            with tf.name_scope("weight_decay_loss"):
                weight_sum = 0
                for var in tf.trainable_variables():
                    if 'bias' not in var.name.lower():
                        weight_sum += tf.nn.l2_loss(var)
                tf.add_to_collection('losses', weight_sum * self.weight_decay)

        with tf.name_scope("ctc_loss"):
            ctc_losses = tf.nn.ctc_loss(
                labels,
                logits,
                # tf.cast(inputs_seq_len, tf.int32),
                inputs_seq_len,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                ignore_longer_outputs_than_inputs=False,
                time_major=True)
            ctc_loss = tf.reduce_mean(ctc_losses, name='ctc_loss_mean')
            tf.add_to_collection('losses', ctc_loss)

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
            tf.summary.scalar('ctc_loss_train', ctc_loss))
        self.summaries_dev.append(
            tf.summary.scalar('ctc_loss_dev', ctc_loss))

        return total_loss, logits

    def _set_optimizer(self, optimizer, learning_rate):
        """Set optimizer.
        Args:
            optimizer (string): the name of the optimizer in
                OPTIMIZER_CLS_NAMES
            learning_rate (float): A learning rate
        Returns:
            optimizer:
        """
        optimizer = optimizer.lower()
        if optimizer not in OPTIMIZER_CLS_NAMES:
            raise ValueError(
                "Optimizer name should be one of [%s], you provided %s." %
                (", ".join(OPTIMIZER_CLS_NAMES), optimizer))

        # Select optimizer
        if optimizer == 'momentum':
            return OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate,
                momentum=0.9)
        elif optimizer == 'nestrov':
            return OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate,
                momentum=0.9,
                use_nesterov=True)
        else:
            return OPTIMIZER_CLS_NAMES[optimizer](
                learning_rate=learning_rate)

    def train(self, loss, optimizer, learning_rate, clip_norm=False):
        """Operation for training. Only the sigle GPU training is supported.
        Args:
            loss: An operation for computing loss
            optimizer (string): name of the optimizer in OPTIMIZER_CLS_NAMES
            learning_rate (placeholder): A learning rate
            clip_norm (bool, optional): if True, clip gradients norm by
                self.clip_grad
        Returns:
            train_op: operation for training
        """
        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set optimizer
        self.optimizer = self._set_optimizer(optimizer, learning_rate)

        if self.clip_grad is not None:
            # Compute gradients
            grads_and_vars = self.optimizer.compute_gradients(loss)

            # Clip gradients
            clipped_grads_and_vars = self._clip_gradients(grads_and_vars,
                                                          clip_norm)

            # Create operation for gradient update
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = self.optimizer.apply_gradients(
                    clipped_grads_and_vars,
                    global_step=global_step)

        else:
            # Use the optimizer to apply the gradients that minimize the loss
            # and also increment the global step counter as a single training
            # step
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = self.optimizer.minimize(
                    loss, global_step=global_step)

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

    def decoder(self, logits, inputs_seq_len, beam_width=1):
        """Operation for decoding.
        Args:
            logits: A tensor of size `[T, B, num_classes]`
            inputs_seq_len: A tensor of size `[B]`
            beam_width (int, optional): beam width for beam search.
                1 disables beam search, which mean greedy decoding.
        Return:
            decode_op: A SparseTensor
        """
        assert isinstance(beam_width, int), "beam_width must be integer."
        assert beam_width >= 1, "beam_width must be >= 1"

        # inputs_seq_len = tf.cast(inputs_seq_len, tf.int32)

        if beam_width == 1:
            decoded, _ = tf.nn.ctc_greedy_decoder(
                logits, inputs_seq_len)
        else:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                logits, inputs_seq_len,
                beam_width=beam_width)

        decode_op = tf.to_int32(decoded[0])

        return decode_op

    def posteriors(self, logits, blank_prior=1):
        """Operation for computing posteriors of each time steps.
        Args:
            logits: A tensor of size `[T, B, num_classes]`
            blank_prior (float): A prior for blank classes. posteriors are
                divided by this prior.
        Return:
            posteriors_op: operation for computing posteriors for each class
        """
        # Convert to batch-major: `[B, T, num_classes]'
        logits = tf.transpose(logits, (1, 0, 2))

        logits_2d = tf.reshape(logits, [-1, self.num_classes])

        # TODO: Divide by blank prior
        # mask = tf.one_hot(
        #     indices=tf.shape(logits_2d),
        #     depth=self.num_classes + 1,
        #     on_value=1,
        #     off_value=0,
        #     axis=-1)
        # mask /= blank_prior
        # logits_2d = tf.multiply(logits_2d, mask)

        posteriors_op = tf.nn.softmax(logits_2d)

        return posteriors_op

    def compute_ler(self, decode_op, labels):
        """Operation for computing LER (Label Error Rate).
        Args:
            decode_op: operation for decoding
            labels: A SparseTensor of target labels
        Return:
            ler_op: operation for computing LER
        """
        # Compute LER (normalize by label length)
        ler_op = tf.reduce_mean(tf.edit_distance(
            decode_op, labels, normalize=True))

        # Add a scalar summary for the snapshot of LER
        self.summaries_train.append(tf.summary.scalar('ler_train', ler_op))
        self.summaries_dev.append(tf.summary.scalar('ler_dev', ler_op))

        return ler_op

    def _tensorboard(self, trainable_vars, dev=True):
        """Compute statistics for TensorBoard plot.
        Args:
            trainable_vars:
            dev (bool, optional): if True, show values in the dev set
        """
        # Histogram
        with tf.name_scope("train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.histogram(var.name, var))

        # Mean
        with tf.name_scope("mean_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.reduce_mean(var)))

        # Standard deviation
        with tf.name_scope("stddev_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.sqrt(
                        tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))))

        # Max
        with tf.name_scope("max_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.reduce_max(var)))

        # Min
        with tf.name_scope("min_train"):
            for var in trainable_vars:
                self.summaries_train.append(
                    tf.summary.scalar(var.name, tf.reduce_min(var)))

        if dev:
            # Histogram
            with tf.name_scope("dev"):
                for var in trainable_vars:
                    self.summaries_dev.append(
                        tf.summary.histogram(var.name, var))

            # Mean
            with tf.name_scope("mean_dev"):
                for var in trainable_vars:
                    self.summaries_dev.append(
                        tf.summary.scalar(var.name, tf.reduce_mean(var)))

            # Standard deviation
            with tf.name_scope("stddev_dev"):
                for var in trainable_vars:
                    self.summaries_dev.append(
                        tf.summary.scalar(var.name, tf.sqrt(
                            tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))))

            # Max
            with tf.name_scope("max_dev"):
                for var in trainable_vars:
                    self.summaries_dev.append(
                        tf.summary.scalar(var.name, tf.reduce_max(var)))

            # Min
            with tf.name_scope("min_dev"):
                for var in trainable_vars:
                    self.summaries_dev.append(
                        tf.summary.scalar(var.name, tf.reduce_min(var)))
