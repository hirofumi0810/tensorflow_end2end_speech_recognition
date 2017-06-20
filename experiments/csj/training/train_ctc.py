#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train CTC network (CSJ corpus)."""

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
from utils.directory import mkdir, mkdir_join
from utils.parameter import count_total_parameters
from utils.csv import save_loss, save_ler


def do_train(network, optimizer, learning_rate, batch_size, epoch_num,
             label_type, num_stack, num_skip, train_data_size):
    """Run training.
    Args:
        network: network to train
        optimizer: string, the name of optimizer.
            ex.) adam, rmsprop
        learning_rate: A float value, the initial learning rate
        batch_size: int, the size of mini-batch
        epoch_num: int, the number of epochs to train
        label_type: string, phone or character or kanji
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        train_data_size: string, default or large
    """
    # Load dataset
    train_data = DataSet(data_type='train', label_type=label_type,
                         train_data_size=train_data_size,
                         batch_size=batch_size,
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=True)
    dev_data = DataSet(data_type='dev', label_type=label_type,
                       train_data_size=train_data_size,
                       batch_size=batch_size,
                       num_stack=num_stack, num_skip=num_skip,
                       is_sorted=False)
    # eval1_data = DataSet(data_type='eval1', label_type=label_type,
    #                      train_data_size=train_data_size,
    #                      batch_size=batch_size,
    #                      num_stack=num_stack, num_skip=num_skip,
    #                      is_sorted=False)
    # eval2_data = DataSet(data_type='eval2', label_type=label_type,
    #                      train_data_size=train_data_size,
    #                      batch_size=batch_size,
    #                      num_stack=num_stack, num_skip=num_skip,
    #                      is_sorted=False)
    # eval3_data = DataSet(data_type='eval3', label_type=label_type,
    #                      train_data_size=train_data_size,
    #                      batch_size=batch_size,
    #                      num_stack=num_stack, num_skip=num_skip,
    #                      is_sorted=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        network.inputs = tf.placeholder(
            tf.float32,
            shape=[None, None, network.input_size],
            name='input')
        indices_pl = tf.placeholder(tf.int64, name='indices')
        values_pl = tf.placeholder(tf.int32, name='values')
        shape_pl = tf.placeholder(tf.int64, name='shape')
        network.labels = tf.SparseTensor(indices_pl, values_pl, shape_pl)
        network.inputs_seq_len = tf.placeholder(tf.int64,
                                                shape=[None],
                                                name='inputs_seq_len')
        network.keep_prob_input = tf.placeholder(tf.float32,
                                                 name='keep_prob_input')
        network.keep_prob_hidden = tf.placeholder(tf.float32,
                                                  name='keep_prob_hidden')

        # Add to the graph each operation (including model definition)
        loss_op, logits = network.compute_loss(network.inputs,
                                               network.labels,
                                               network.inputs_seq_len,
                                               network.keep_prob_input,
                                               network.keep_prob_hidden)
        train_op = network.train(loss_op,
                                 optimizer=optimizer,
                                 learning_rate_init=float(learning_rate),
                                 is_scheduled=False)
        decode_op = network.decoder(logits,
                                    network.inputs_seq_len,
                                    decode_type='beam_search',
                                    beam_width=20)
        ler_op = network.compute_ler(decode_op, network.labels)

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

        csv_steps, csv_train_loss, csv_dev_loss = [], [], []
        csv_ler_train, csv_ler_dev = [], []
        # Create a session for running operation on the graph
        with tf.Session() as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                network.model_dir, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Make mini-batch generator
            mini_batch_train = train_data.next_batch()
            mini_batch_dev = dev_data.next_batch()

            # Train model
            iter_per_epoch = int(train_data.data_num / batch_size)
            train_step = train_data.data_num / batch_size
            if (train_step) != int(train_step):
                iter_per_epoch += 1
            max_steps = iter_per_epoch * epoch_num
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            error_best = 1
            for step in range(max_steps):

                # Create feed dictionary for next mini batch (train)
                inputs, labels_st, inputs_seq_len, _ = mini_batch_train.__next__()
                feed_dict_train = {
                    network.inputs: inputs,
                    network.labels: labels_st,
                    network.inputs_seq_len: inputs_seq_len,
                    network.keep_prob_input: network.dropout_ratio_input,
                    network.keep_prob_hidden: network.dropout_ratio_hidden,
                    network.lr: learning_rate
                }

                # Create feed dictionary for next mini batch (dev)
                inputs, labels_st, inputs_seq_len, _ = mini_batch_dev.__next__()
                feed_dict_dev = {
                    network.inputs: inputs,
                    network.labels: labels_st,
                    network.inputs_seq_len: inputs_seq_len,
                    network.keep_prob_input: network.dropout_ratio_input,
                    network.keep_prob_hidden: network.dropout_ratio_hidden
                }

                # Update parameters
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % 200 == 0:

                    # Compute loss_
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_train_loss.append(loss_train)
                    csv_dev_loss.append(loss_dev)

                    # Change to evaluation mode
                    feed_dict_train[network.keep_prob_input] = 1.0
                    feed_dict_train[network.keep_prob_hidden] = 1.0
                    feed_dict_dev[network.keep_prob_input] = 1.0
                    feed_dict_dev[network.keep_prob_hidden] = 1.0

                    # Compute accuracy & update event file
                    ler_train, summary_str_train = sess.run(
                        [ler_op, summary_train], feed_dict=feed_dict_train)
                    ler_dev, summary_str_dev = sess.run(
                        [ler_op, summary_dev], feed_dict=feed_dict_dev)
                    csv_ler_train.append(ler_train)
                    csv_ler_dev.append(ler_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print('Step %d: loss = %.3f (%.3f) / ler = %.4f (%.4f) (%.3f min)' %
                          (step + 1, loss_train, loss_dev, ler_train,
                           ler_dev, duration_step / 60))
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
                    save_path = saver.save(
                        sess, checkpoint_file, global_step=epoch)
                    print("Model saved in file: %s" % save_path)

                    if epoch >= 5:
                        start_time_eval = time.time()
                        if label_type in ['character', 'kanji']:
                            print('=== Dev Evaluation ===')
                            cer_dev_epoch = do_eval_cer(
                                session=sess,
                                decode_op=decode_op,
                                network=network,
                                dataset=dev_data,
                                label_type=label_type,
                                eval_batch_size=batch_size)
                            print('  CER: %f %%' % (cer_dev_epoch * 100))

                            if cer_dev_epoch < error_best:
                                error_best = cer_dev_epoch
                                print('■■■ ↑Best Score (CER)↑ ■■■')

                                # print('=== eval1 Evaluation ===')
                                # cer_eval1 = do_eval_cer(
                                #     session=sess,
                                #     decode_op=decode_op,
                                #     network=network,
                                #     dataset=eval1_data,
                                #     label_type=label_type,
                                #     is_test=True,
                                #     eval_batch_size=batch_size)
                                # print('  CER: %f %%' % (cer_eval1 * 100))
                                #
                                # print('=== eval2 Evaluation ===')
                                # cer_eval2 = do_eval_cer(
                                #     session=sess,
                                #     decode_op=decode_op,
                                #     network=network,
                                #     dataset=eval2_data,
                                #     label_type=label_type,
                                #     is_test=-True,
                                #     eval_batch_size=batch_size)
                                # print('  CER: %f %%' % (cer_eval2 * 100))
                                #
                                # print('=== eval3 Evaluation ===')
                                # cer_eval3 = do_eval_cer(
                                #     session=sess,
                                #     decode_op=decode_op,
                                #     network=network,
                                #     dataset=eval3_data,
                                #     label_type=label_type,
                                #     is_test=True,
                                #     eval_batch_size=batch_size)
                                # print('  CER: %f %%' % (cer_eval3 * 100))
                                #
                                # print('=== Mean ===')
                                # cer_mean = (
                                #     cer_eval1 + cer_eval2 + cer_eval3) / 3.
                                # print('  CER: %f %%' % (cer_mean * 100))

                        else:
                            print('=== Dev Evaluation ===')
                            per_dev_epoch = do_eval_per(
                                session=sess,
                                per_op=ler_op,
                                network=network,
                                dataset=dev_data,
                                eval_batch_size=batch_size)
                            print('  PER: %f %%' % (per_dev_epoch * 100))

                            if per_dev_epoch < error_best:
                                error_best = per_dev_epoch
                                print('■■■ ↑Best Score (PER)↑ ■■■')

                                # print('=== eval1 Evaluation ===')
                                # per_eval1 = do_eval_per(
                                #     session=sess,
                                #     per_op=ler_op,
                                #     network=network,
                                #     dataset=eval1_data,
                                #     eval_batch_size=batch_size)
                                # print('  PER: %f %%' % (per_eval1 * 100))
                                #
                                # print('=== eval2 Evaluation ===')
                                # per_eval2 = do_eval_per(
                                #     session=sess,
                                #     per_op=ler_op,
                                #     network=network,
                                #     dataset=eval2_data,
                                #     eval_batch_size=batch_size)
                                # print('  PER: %f %%' % (per_eval2 * 100))
                                #
                                # print('=== eval3 Evaluation ===')
                                # per_eval3 = do_eval_per(
                                #     session=sess,
                                #     per_op=ler_op,
                                #     network=network,
                                #     dataset=eval3_data,
                                #     eval_batch_size=batch_size)
                                # print('  PER: %f %%' % (per_eval3 * 100))
                                #
                                # print('=== Mean ===')
                                # per_mean = (
                                #     per_eval1 + per_eval2 + per_eval3) / 3.
                                # print('  PER: %f %%' % 1 (per_mean * 100))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                        start_time_epoch = time.time()
                        start_time_step = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Save train & dev loss, ler
            save_loss(csv_steps, csv_train_loss, csv_dev_loss,
                      save_path=network.model_dir)
            save_ler(csv_steps, csv_ler_train, csv_ler_dev,
                     save_path=network.model_dir)

            # Training was finished correctly
            with open(join(network.model_dir, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type'] == 'phone':
        output_size = 38
    elif corpus['label_type'] == 'character':
        output_size = 147
    elif corpus['label_type'] == 'kanji':
        output_size = 3386

    # Model setting
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(batch_size=param['batch_size'],
                       input_size=feature['input_size'] * feature['num_stack'],
                       num_unit=param['num_unit'],
                       num_layer=param['num_layer'],
                       bottleneck_dim=param['bottleneck_dim'],
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
    if param['bottleneck_dim'] != 0:
        network.model_name += '_bottoleneck' + str(param['bottleneck_dim'])
    if param['num_proj'] != 0:
        network.model_name += '_proj' + str(param['num_proj'])
    if feature['num_stack'] != 1:
        network.model_name += '_stack' + str(feature['num_stack'])
    if param['weight_decay'] != 0:
        network.model_name += '_weightdecay' + str(param['weight_decay'])
    if corpus['train_data_size'] == 'large':
        network.model_name += '_large'

    # Set save path
    network.model_dir = mkdir('/n/sd8/inaguma/result/csj/monolog/')
    network.model_dir = mkdir_join(network.model_dir, 'ctc')
    network.model_dir = mkdir_join(network.model_dir, corpus['label_type'])
    network.model_dir = mkdir_join(network.model_dir, network.model_name)

    # Reset model directory
    if not isfile(join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('ctc_csj_' + corpus['label_type'] +
                 '_' + corpus['train_data_size'])

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
             train_data_size=corpus['train_data_size'])
    sys.stdout = sys.__stdout__


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError
    main(config_path=args[1])
