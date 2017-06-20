#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train CTC network with multiple GPUs (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile
import sys
import time
import tensorflow as tf
from setproctitle import setproctitle
import yaml
import shutil

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from data.read_dataset_ctc import DataSet
from models.ctc.load_model import load
from metric.ctc import do_eval_per, do_eval_cer
from utils.sparsetensor import list2sparsetensor
from utils.directory import mkdir, mkdir_join
from utils.parameter import count_total_parameters
from utils.csv import save_loss, save_ler
from utils.multi_gpu import average_gradients


def tower_loss(scope, images, labels):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
      images: Images. 4D tensor of shape [batch_size, height, width, 3].
      labels: Labels. 1D tensor of shape [batch_size].
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    import re
    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)

    return total_loss


def do_train(network, optimizer, learning_rate, batch_size, epoch_num,
             label_type, num_stack, num_skip, gpu_indices):
    """Run training. If target labels are phone, the model is evaluated by PER
    with 39 phones.
    Args:
        network: network to train
        optimizer: string, the name of optimizer.
            ex.) adam, rmsprop
        learning_rate: A flaot value, the initial learning rate
        batch_size: int, teh the size of mini-batch
        epoch_num: int, the number of epochs to train
        label_type: string, phone39 or phone48 or phone61 or character
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        gpu_indices: list of integer
    """
    # Load dataset
    train_data = DataSet(data_type='train', label_type=label_type,
                         batch_size=batch_size,
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=True, num_gpu=len(gpu_indices))
    dev_data = DataSet(data_type='dev', label_type=label_type,
                       batch_size=batch_size,
                       num_stack=num_stack, num_skip=num_skip,
                       is_sorted=False, num_gpu=len(gpu_indices))
    if label_type == 'character':
        # TODO: evaluationのときはどうする？
        test_data = DataSet(data_type='test', label_type='character',
                            batch_size=batch_size,
                            num_stack=num_stack, num_skip=num_skip,
                            is_sorted=False, num_gpu=1)
    else:

        test_data = DataSet(data_type='test', label_type='phone39',
                            batch_size=batch_size,
                            num_stack=num_stack, num_skip=num_skip,
                            is_sorted=False, num_gpu=1)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate)

        # Calculate the gradients for each model tower
        tower_grads = []
        network.inputs = []
        network.labels = []
        network.inputs_seq_len = []
        network.keep_prob_input = []
        network.keep_prob_hidden = []
        # TODO: cpu 用のタワーも用意する

        all_devices = ['/gpu:%d' %
                       i_gpu for i_gpu in range(len(gpu_indices))]
        # NOTE: /cpu:0 is prepared for evaluation

        loss_dict = {}
        total_loss = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i_device in range(len(all_devices)):
                with tf.device(all_devices[i_device]):
                    with tf.name_scope('%s_%d' % ('tower', i_device)) as scope:

                        # Define placeholders in each tower
                        network.inputs.append(tf.placeholder(
                            tf.float32,
                            shape=[None, None, network.input_size],
                            name='input' + str(i_device)))
                        indices_pl = tf.placeholder(
                            tf.int64, name='indices%d' % i_device)
                        values_pl = tf.placeholder(
                            tf.int32,  name='values%d' % i_device)
                        shape_pl = tf.placeholder(
                            tf.int64, name='shape%d' % i_device)
                        network.labels.append(tf.SparseTensor(
                            indices_pl, values_pl, shape_pl))
                        network.inputs_seq_len.append(tf.placeholder(
                            tf.int64, shape=[None],
                            name='inputs_seq_len%d' % i_device))
                        network.keep_prob_input.append(
                            tf.placeholder(tf.float32,
                                           name='keep_prob_input%d'
                                           % i_device))
                        network.keep_prob_hidden.append(
                            tf.placeholder(tf.float32,
                                           name='keep_prob_hidden%d'
                                           % i_device))

                        # Calculate the loss for one tower of the model. This
                        # function constructs the entire model but shares the
                        # variables across　all towers
                        loss, logits = network.compute_loss(
                            network.inputs[i_device],
                            network.labels[i_device],
                            network.inputs_seq_len[i_device],
                            network.keep_prob_input[i_device],
                            network.keep_prob_hidden[i_device])

                        # Assemble all of the losses for the current tower
                        # only
                        losses = tf.get_collection('losses', scope)

                        # Calculate the total loss for the current tower
                        tower_loss = tf.add_n(losses, name='tower_loss')
                        total_loss.append(tower_loss)

                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        # summaries = tf.get_collection(
                        #     tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this
                        # tower.
                        grads = optimizer.compute_gradients(tower_loss)

                        # TODO: gradient clipping

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # Aggregate losses, then calculate average loss.
        loss_op = tf.add_n(total_loss) / len(gpu_indices)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers
        # for i in range(len(gpu_indices)):
        grads = average_gradients(tower_grads)

        # Add a summary to track the learning rate.
        # summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        # for grad, var in grads:
        #   if grad is not None:
        # summaries.append(tf.summary.histogram(var.op.name + '/gradients',
        # grad))

        # Apply the gradients to adjust the shared variables.
        train_op = optimizer.apply_gradients(
            grads, global_step=global_step)

        # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #   summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     0.9999, global_step)
        # variables_averages_op =
        # variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        # train_op = tf.group(apply_gradient_op, variables_averages_op)

        ####################################

        # Add to the graph each operation (Use last placeholders)
        # train_op = network.train(loss_op,
        #                          optimizer='adam',
        #                          learning_rate_init=learning_rate,
        #                          is_scheduled=False)
        decode_op = network.decoder(logits,
                                    network.inputs_seq_len[-1],
                                    decode_type='beam_search',
                                    beam_width=20)
        ler_op = network.compute_ler(decode_op, network.labels[-1])

        # Build the summary tensor based on the TensorFlow collection of
        # summaries
        summary_train = tf.summary.merge(network.summaries_train)
        summary_dev = tf.summary.merge(network.summaries_dev)

        # Add the variable initializer operation
        init_op = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        # Count total parameters
        parameters_dict, total_parameters = count_total_parameters(
            tf.trainable_variables())
        for parameter_name in sorted(parameters_dict.keys()):
            print("%s %d" % (parameter_name, parameters_dict[parameter_name]))
        print("Total %d variables, %s M parameters" %
              (len(parameters_dict.keys()),
               "{:,}".format(total_parameters / 1000000)))

        csv_steps, csv_loss_train, csv_loss_dev = [], [], []
        csv_ler_train, csv_ler_dev = [], []
        # Create a session for running operation on the graph
        # NOTE: Start running operations on the Graph. allow_soft_placement
        # must be set to True to build towers on GPU, as some of the ops do not
        # have GPU implementations.
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)) as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                network.model_dir, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Make generator
            mini_batch_train = train_data.next_batch(session=sess)
            mini_batch_dev = dev_data.next_batch(session=sess)

            # Train model
            iter_per_epoch = int(train_data.data_num /
                                 (batch_size * len(gpu_indices)))
            train_step = train_data.data_num / batch_size
            if train_step != int(train_step):
                iter_per_epoch += 1
            max_steps = iter_per_epoch * epoch_num
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            error_best = 1
            for step in range(max_steps):

                # Create feed dictionary for next mini batch (train)
                inputs, labels_st, inputs_seq_len, _ = mini_batch_train.__next__()
                feed_dict_train, feed_dict_dev = {}, {}
                for i_gpu in range(len(gpu_indices)):
                    feed_dict_train[network.inputs[i_gpu]] = inputs[i_gpu]
                    feed_dict_train[network.labels[i_gpu]] = labels_st[i_gpu]
                    feed_dict_train[network.inputs_seq_len[i_gpu]
                                    ] = inputs_seq_len[i_gpu]
                    feed_dict_train[network.keep_prob_input[i_gpu]
                                    ] = network.dropout_ratio_input
                    feed_dict_train[network.keep_prob_hidden[i_gpu]
                                    ] = network.dropout_ratio_hidden

                # Create feed dictionary for next mini batch (dev)
                inputs, labels_st, inputs_seq_len, _ = mini_batch_dev.__next__()
                for i_gpu in range(len(gpu_indices)):
                    feed_dict_dev[network.inputs[i_gpu]] = inputs[i_gpu]
                    feed_dict_dev[network.labels[i_gpu]] = labels_st[i_gpu]
                    feed_dict_dev[network.inputs_seq_len[i_gpu]
                                  ] = inputs_seq_len[i_gpu]
                    feed_dict_dev[network.keep_prob_input[i_gpu]
                                  ] = network.dropout_ratio_input
                    feed_dict_dev[network.keep_prob_hidden[i_gpu]
                                  ] = network.dropout_ratio_hidden

                # Update parameters
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % int(10 / len(gpu_indices)) == 0:

                    # Compute loss
                    print(loss_op)
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    # loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_loss_train.append(loss_train)
                    # csv_loss_dev.append(loss_dev)

                    # Change to evaluation mode
                    for i_gpu in range(len(gpu_indices)):
                        feed_dict_train[network.keep_prob_input[i_gpu]] = 1.0
                        feed_dict_train[network.keep_prob_hidden[i_gpu]] = 1.0
                        feed_dict_dev[network.keep_prob_input[i_gpu]] = 1.0
                        feed_dict_dev[network.keep_prob_hidden[i_gpu]] = 1.0

                    # Compute accuracy & update event file
                    ler_train, summary_str_train = sess.run(
                        [ler_op, summary_train], feed_dict=feed_dict_train)
                    # ler_dev, summary_str_dev = sess.run(
                    #     [ler_op, summary_dev], feed_dict=feed_dict_dev)
                    csv_ler_train.append(ler_train)
                    # csv_ler_dev.append(ler_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    # summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print("Step %d: loss = %.3f (%.3f) / ler = %.4f (%.4f) (%.3f min)" %
                          (step + 1, loss_train, 1, ler_train,
                           1, duration_step / 60))
                    sys.stdout.flush()
                    start_time_step = time.time()

                # Save checkpoint and evaluate model per epoch
                if (step + 1) % iter_per_epoch == 0 or (step + 1) == max_steps:
                    duration_epoch = time.time() - start_time_epoch
                    epoch = (step + 1) // iter_per_epoch
                    print('-----EPOCH:%d (%.3f min)-----' %
                          (epoch, duration_epoch / 60))

                    # Save model (check point)
                    checkpoint_file = join(network.model_dir, 'model.ckpt')
                    # save_path = saver.save(
                    #     sess, checkpoint_file, global_step=epoch)
                    # print("Model saved in file: %s" % save_path)

                    if epoch >= 10:
                        start_time_eval = time.time()
                        # if label_type == 'character':
                        #     print('=== Dev Data Evaluation ===')
                        #     cer_dev_epoch = do_eval_cer(
                        #         session=sess,
                        #         decode_op=decode_op,
                        #         network=network,
                        #         dataset=dev_data)
                        #     print('  CER: %f %%' % (cer_dev_epoch * 100))
                        #
                        #     if cer_dev_epoch < error_best:
                        #         error_best = cer_dev_epoch
                        #         print('■■■ ↑Best Score (CER)↑ ■■■')
                        #
                        #         print('=== Test Data Evaluation ===')
                        #         cer_test = do_eval_cer(
                        #             session=sess,
                        #             decode_op=decode_op,
                        #             network=network,
                        #             dataset=test_data,
                        #             eval_batch_size=1)
                        #         print('  CER: %f %%' % (cer_test * 100))
                        #
                        # else:
                        #     print('=== Dev Data Evaluation ===')
                        #     per_dev_epoch = do_eval_per(
                        #         session=sess,
                        #         decode_op=decode_op,
                        #         per_op=ler_op,
                        #         network=network,
                        #         dataset=dev_data,
                        #         label_type=label_type)
                        #     print('  PER: %f %%' % (per_dev_epoch * 100))
                        #
                        #     if per_dev_epoch < error_best:
                        #         error_best = per_dev_epoch
                        #         print('■■■ ↑Best Score (PER)↑ ■■■')
                        #
                        #         print('=== Test Data Evaluation ===')
                        #         per_test = do_eval_per(
                        #             session=sess,
                        #             decode_op=decode_op,
                        #             per_op=ler_op,
                        #             network=network,
                        #             dataset=test_data,
                        #             label_type=label_type,
                        #             eval_batch_size=1)
                        #         print('  PER: %f %%' % (per_test * 100))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                start_time_epoch = time.time()
                start_time_step = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Save train & dev loss
            save_loss(csv_steps, csv_loss_train, csv_loss_dev,
                      save_path=network.model_dir)
            save_ler(csv_steps, csv_ler_train, csv_ler_dev,
                     save_path=network.model_dir)

            # Training was finished correctly
            with open(join(network.model_dir, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path, gpu_indices):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type'] == 'phone61':
        output_size = 61
    elif corpus['label_type'] == 'phone48':
        output_size = 48
    elif corpus['label_type'] == 'phone39':
        output_size = 39
    elif corpus['label_type'] == 'character':
        output_size = 30

    # Model setting
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(batch_size=param['batch_size'],
                       input_size=feature['input_size'] * feature['num_stack'],
                       num_unit=param['num_unit'],
                       num_layer=param['num_layer'],
                       output_size=output_size,
                       parameter_init=param['weight_init'],
                       clip_grad=param['clip_grad'],
                       clip_activation=param['clip_activation'],
                       dropout_ratio_input=param['dropout_input'],
                       dropout_ratio_hidden=param['dropout_hidden'],
                       num_proj=param['num_proj'],
                       weight_decay=param['weight_decay'])

    network.model_name = config['model_name'].upper()
    network.model_name += '_' + str(param['num_unit'])
    network.model_name += '_' + str(param['num_layer'])
    network.model_name += '_' + param['optimizer']
    network.model_name += '_lr' + str(param['learning_rate'])
    if param['num_proj'] != 0:
        network.model_name += '_proj' + str(param['num_proj'])
    if feature['num_stack'] != 1:
        network.model_name += '_stack' + str(feature['num_stack'])
    if param['weight_decay'] != 0:
        network.model_name += '_weightdecay' + str(param['weight_decay'])
    network.model_name += '_' + str(len(gpu_indices)) + 'gpu'

    # Set save path
    network.model_dir = mkdir('/n/sd8/inaguma/result/timit/ctc/')
    network.model_dir = mkdir_join(network.model_dir, corpus['label_type'])
    network.model_dir = mkdir_join(network.model_dir, network.model_name)

    # Reset model directory
    if not isfile(join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('multigpu_ctc_timit_' + corpus['label_type'])

    # Save config file
    shutil.copyfile(config_path, join(network.model_dir, 'config.yml'))

    sys.stdout = open(join(network.model_dir, 'train.log'), 'w')
    print(network.model_name)
    do_train(network=network,
             optimizer=param['optimizer'],
             learning_rate=param['learning_rate'],
             batch_size=param['batch_size'],
             epoch_num=param['num_epoch'],
             label_type=corpus['label_type'],
             num_stack=feature['num_stack'],
             num_skip=feature['num_skip'],
             gpu_indices=gpu_indices)
    sys.stdout = sys.__stdout__


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError
    main(config_path=args[1], gpu_indices=list(map(int, args[2].split(','))))
