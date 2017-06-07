#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train Multitask CTC network (TIMIT corpus)."""

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
from data.read_dataset_multitask_ctc import DataSet
from models.ctc.load_model_multitask import load
from evaluation.eval_ctc import do_eval_per, do_eval_cer
from utils.data.sparsetensor import list2sparsetensor, sparsetensor2list
from utils.util import mkdir, mkdir_join
from utils.parameter import count_total_parameters
from utils.loss import save_loss


def do_train(network, optimizer, learning_rate, batch_size, epoch_num, label_type, num_stack, num_skip):
    """Run training.
    Args:
        network: network to train
        optimizer: string, the name of optimizer. ex.) adam, rmsprop
        learning_rate: initial learning rate
        batch_size: size of mini batch
        epoch_num: epoch num to train
        label_type: phone39 or phone48 or phone61 (+ character)
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
    """
    # Load dataset
    train_data = DataSet(data_type='train', label_type=label_type,
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=True)
    dev_data61 = DataSet(data_type='dev', label_type='phone61',
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=False)
    dev_data39 = DataSet(data_type='dev', label_type='phone39',
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=False)
    test_data = DataSet(data_type='test', label_type='phone39',
                        num_stack=num_stack, num_skip=num_skip,
                        is_sorted=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define model
        network.define()
        # NOTE: define model under tf.Graph()

        # Add to the graph each operation
        loss_op = network.loss()
        train_op = network.train(optimizer=optimizer,
                                 learning_rate_init=learning_rate,
                                 is_scheduled=False)
        decode_op1, decode_op2 = network.decoder(decode_type='beam_search',
                                                 beam_width=20)
        per_op1, per_op2 = network.ler(decode_op1, decode_op2)

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
              (len(parameters_dict.keys()), "{:,}".format(total_parameters / 1000000)))

        csv_steps = []
        csv_train_loss = []
        csv_dev_loss = []
        # Create a session for running operation on the graph
        with tf.Session() as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                network.model_dir, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Train model
            iter_per_epoch = int(train_data.data_num / batch_size)
            if (train_data.data_num / batch_size) != int(train_data.data_num / batch_size):
                iter_per_epoch += 1
            max_steps = iter_per_epoch * epoch_num
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            error_best = 1
            for step in range(max_steps):

                # Create feed dictionary for next mini batch (train)
                inputs, labels_char, labels_phone, seq_len, _ = train_data.next_batch(
                    batch_size=batch_size)
                indices_char, values_char, dense_shape_char = list2sparsetensor(
                    labels_char)
                indices_phone, values_phone, dense_shape_phone = list2sparsetensor(
                    labels_phone)
                feed_dict_train = {
                    network.inputs_pl: inputs,
                    network.label_indices_pl: indices_char,
                    network.label_values_pl: values_char,
                    network.label_shape_pl: dense_shape_char,
                    network.label_indices_pl2: indices_phone,
                    network.label_values_pl2: values_phone,
                    network.label_shape_pl2: dense_shape_phone,
                    network.seq_len_pl: seq_len,
                    network.keep_prob_input_pl: network.dropout_ratio_input,
                    network.keep_prob_hidden_pl: network.dropout_ratio_hidden,
                    network.lr_pl: learning_rate
                }

                # Create feed dictionary for next mini batch (dev)
                inputs, labels_char, labels_phone, seq_len, _ = dev_data61.next_batch(
                    batch_size=batch_size)
                indices_char, values_char, dense_shape_char = list2sparsetensor(
                    labels_char)
                indices_phone, values_phone, dense_shape_phone = list2sparsetensor(
                    labels_phone)
                feed_dict_dev = {
                    network.inputs_pl: inputs,
                    network.label_indices_pl: indices_char,
                    network.label_values_pl: values_char,
                    network.label_shape_pl: dense_shape_char,
                    network.label_indices_pl2: indices_phone,
                    network.label_values_pl2: values_phone,
                    network.label_shape_pl2: dense_shape_phone,
                    network.seq_len_pl: seq_len,
                    network.keep_prob_input_pl: network.dropout_ratio_input,
                    network.keep_prob_hidden_pl: network.dropout_ratio_hidden
                }

                # Update parameters & compute loss
                _, loss_train = sess.run(
                    [train_op, loss_op], feed_dict=feed_dict_train)
                loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                csv_steps.append(step)
                csv_train_loss.append(loss_train)
                csv_dev_loss.append(loss_dev)

                if (step + 1) % 10 == 0:

                    # Change feed dict for evaluation
                    feed_dict_train[network.keep_prob_input_pl] = 1.0
                    feed_dict_train[network.keep_prob_hidden_pl] = 1.0
                    feed_dict_dev[network.keep_prob_input_pl] = 1.0
                    feed_dict_dev[network.keep_prob_hidden_pl] = 1.0

                    # Compute accuracy & update event file
                    cer_train, per_train, summary_str_train = sess.run([per_op1, per_op2, summary_train],
                                                                       feed_dict=feed_dict_train)
                    cer_dev, per_dev, summary_str_dev = sess.run([per_op1, per_op2,  summary_dev],
                                                                 feed_dict=feed_dict_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print('Step %d: loss = %.3f (%.3f) / cer = %.4f (%.4f) / per = %.4f (%.4f) (%.3f min)' %
                          (step + 1, loss_train, loss_dev, cer_train, cer_dev, per_train, per_dev, duration_step / 60))
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

                    if epoch >= 10:
                        start_time_eval = time.time()

                        print('■Dev Data Evaluation:■')
                        error_epoch = do_eval_cer(session=sess,
                                                  decode_op=decode_op1,
                                                  network=network,
                                                  dataset=dev_data39,
                                                  eval_batch_size=1,
                                                  is_multitask=True)
                        do_eval_per(session=sess,
                                    decode_op=decode_op2,
                                    per_op=per_op2,
                                    network=network,
                                    dataset=dev_data39,
                                    label_type=label_type,
                                    eval_batch_size=1,
                                    is_multitask=True)

                        if error_epoch < error_best:
                            error_best = error_epoch
                            print('■■■ ↑Best Score (CER)↑ ■■■')

                            print('■Test Data Evaluation:■')
                            do_eval_cer(session=sess,
                                        decode_op=decode_op1,
                                        network=network,
                                        dataset=test_data,
                                        eval_batch_size=1,
                                        is_multitask=True)

                            do_eval_per(session=sess,
                                        decode_op=decode_op2,
                                        per_op=per_op2,
                                        network=network,
                                        dataset=test_data,
                                        label_type=label_type,
                                        eval_batch_size=1,
                                        is_multitask=True)

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                    start_time_epoch = time.time()
                    start_time_step = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Save train & dev loss
            save_loss(csv_steps, csv_train_loss, csv_dev_loss,
                      save_path=network.model_dir)

            # Training was finished correctly
            with open(join(network.model_dir, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path):

    # Read a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type'] == 'phone61':
        output_size2 = 61
    elif corpus['label_type'] == 'phone48':
        output_size2 = 48
    elif corpus['label_type'] == 'phone39':
        output_size2 = 39

    # Model setting
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(batch_size=param['batch_size'],
                       input_size=feature['input_size'] * feature['num_stack'],
                       num_cell=param['num_cell'],
                       num_layer=param['num_layer'],
                       num_layer2=param['num_layer2'],
                       output_size=30,
                       output_size2=output_size2,
                       main_task_weight=param['main_task_weight'],
                       clip_gradients=param['clip_grad'],
                       clip_activation=param['clip_activation'],
                       dropout_ratio_input=param['dropout_input'],
                       dropout_ratio_hidden=param['dropout_hidden'],
                       num_proj=param['num_proj'],
                       weight_decay=param['weight_decay'])

    network.model_name = config['model_name'].upper()
    network.model_name += '_' + str(param['num_cell'])
    network.model_name += '_' + str(param['num_layer'])
    network.model_name += '_' + str(param['num_layer2'])
    network.model_name += '_' + param['optimizer']
    network.model_name += '_lr' + str(param['learning_rate'])
    if param['num_proj'] != 0:
        network.model_name += '_proj' + str(param['num_proj'])
    if feature['num_stack'] != 1:
        network.model_name += '_stack' + str(feature['num_stack'])
    if param['weight_decay'] != 0:
        network.model_name += '_weightdecay' + str(param['weight_decay'])
    network.model_name += '_taskweight' + str(param['main_task_weight'])

    # Set save path
    network.model_dir = mkdir('/n/sd8/inaguma/result/timit/multitask_ctc/')
    network.model_dir = mkdir_join(network.model_dir, corpus['label_type'])
    network.model_dir = mkdir_join(network.model_dir, network.model_name)

    # Reset model directory
    if not isfile(join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('multitaskctc_timit_' +
                 corpus['label_type'] + '_' + param['optimizer'])

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
             num_skip=feature['num_skip'])
    sys.stdout = sys.__stdout__


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError

    main(config_path=args[1])
