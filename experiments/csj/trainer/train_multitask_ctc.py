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
from data.read_dataset_multitask_ctc import DataSet
from models.ctc.load_model_multitask import load
from evaluation.eval_ctc import do_eval_per, do_eval_cer
from utils.data.sparsetensor import list2sparsetensor, sparsetensor2list
from utils.util import mkdir, mkdir_join
from utils.parameter import count_total_parameters
from utils.loss import save_loss


def do_train(network, optimizer, learning_rate, batch_size, epoch_num,
             label_type_main, label_type_second, num_stack, num_skip,
             train_data_size):
    """Run training.
    Args:
        network: network to train
        optimizer: string, the name of optimizer. ex.) adam, rmsprop
        learning_rate: initial learning rate
        batch_size: size of mini batch
        epoch_num: epoch num to train
        label_type_main: kanji or character
        label_type_second: character or phone
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        train_data_size: default or large
    """
    # Load dataset
    train_data = DataSet(data_type='train', label_type_main=label_type_main,
                         label_type_second=label_type_second,
                         train_data_size=train_data_size,
                         num_stack=num_stack, num_skip=num_skip, is_sorted=True)
    dev_data = DataSet(data_type='dev', label_type_main=label_type_main,
                       label_type_second=label_type_second,
                       train_data_size=train_data_size,
                       num_stack=num_stack, num_skip=num_skip, is_sorted=False)
    eval1_data = DataSet(data_type='eval1', label_type_main=label_type_main,
                         label_type_second=label_type_second,
                         train_data_size=train_data_size,
                         num_stack=num_stack, num_skip=num_skip, is_sorted=False)
    eval2_data = DataSet(data_type='eval2', label_type_main=label_type_main,
                         label_type_second=label_type_second,
                         train_data_size=train_data_size,
                         num_stack=num_stack, num_skip=num_skip, is_sorted=False)
    eval3_data = DataSet(data_type='eval3', label_type_main=label_type_main,
                         label_type_second=label_type_second,
                         train_data_size=train_data_size,
                         num_stack=num_stack, num_skip=num_skip, is_sorted=False)

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
            cer_dev_best = 1
            for step in range(max_steps):
                # Create feed dictionary for next mini batch (train)
                inputs, labels_main, labels_second, seq_len, _ = train_data.next_batch(
                    batch_size=batch_size)
                indices_main, values_main, dense_shape_main = list2sparsetensor(
                    labels_main)
                indices_second, values_second, dense_shape_second = list2sparsetensor(
                    labels_second)
                feed_dict_train = {
                    network.inputs_pl: inputs,
                    network.label_indices_pl: indices_main,
                    network.label_values_pl: values_main,
                    network.label_shape_pl: dense_shape_main,
                    network.label_indices_pl2: indices_second,
                    network.label_values_pl2: values_second,
                    network.label_shape_pl2: dense_shape_second,
                    network.seq_len_pl: seq_len,
                    network.keep_prob_input_pl: network.dropout_ratio_input,
                    network.keep_prob_hidden_pl: network.dropout_ratio_hidden,
                    network.lr_pl: learning_rate
                }

                # Create feed dictionary for next mini batch (dev)
                inputs, labels_main, labels_second, seq_len, _ = dev_data.next_batch(
                    batch_size=batch_size)
                indices_main, values_main, dense_shape_main = list2sparsetensor(
                    labels_main)
                indices_second, values_second, dense_shape_second = list2sparsetensor(
                    labels_second)
                feed_dict_dev = {
                    network.inputs_pl: inputs,
                    network.label_indices_pl: indices_main,
                    network.label_values_pl: values_main,
                    network.label_shape_pl: dense_shape_main,
                    network.label_indices_pl2: indices_second,
                    network.label_values_pl2: values_second,
                    network.label_shape_pl2: dense_shape_second,
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

                if (step + 1) % 100 == 0:
                    # Change feed dict for evaluation
                    feed_dict_train[network.keep_prob_input_pl] = 1.0
                    feed_dict_train[network.keep_prob_hidden_pl] = 1.0
                    feed_dict_dev[network.keep_prob_input_pl] = 1.0
                    feed_dict_dev[network.keep_prob_hidden_pl] = 1.0

                    # Compute accuracy & \update event file
                    ler_main_train, ler_second_train, summary_str_train = sess.run([per_op1, per_op2, summary_train],
                                                                                   feed_dict=feed_dict_train)
                    ler_main_dev, lera_second_dev, summary_str_dev = sess.run([per_op1, per_op2, summary_dev],
                                                                              feed_dict=feed_dict_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print('Step %d: loss = %.3f (%.3f) / ler_main = %.4f (%.4f) / ler_second = %.4f (%.4f) (%.3f min)' %
                          (step + 1, loss_train, loss_dev, ler_main_train, ler_main_dev,
                           ler_second_train, lera_second_dev, duration_step / 60))
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

                    start_time_eval = time.time()
                    print('■Dev Evaluation:■')
                    cer_dev_epoch = do_eval_cer(session=sess,
                                                decode_op=decode_op1,
                                                network=network,
                                                dataset=dev_data,
                                                label_type=label_type_main,
                                                eval_batch_size=batch_size,
                                                is_multitask=True,
                                                is_main=True)
                    if label_type_second == 'character':
                        do_eval_cer(session=sess,
                                    decode_op=decode_op2,
                                    network=network,
                                    dataset=dev_data,
                                    label_type=label_type_second,
                                    eval_batch_size=batch_size,
                                    is_multitask=True,
                                    is_main=False)
                    elif label_type_second == 'phone':
                        do_eval_per(session=sess,
                                    per_op=per_op2,
                                    network=network,
                                    dataset=dev_data,
                                    eval_batch_size=batch_size,
                                    is_multitask=True)

                    if cer_dev_epoch < cer_dev_best:
                        cer_dev_best = cer_dev_epoch
                        print('■■■ ↑Best Score (CER)↑ ■■■')

                        print('■eval1 Evaluation:■')
                        cer_eval1 = do_eval_cer(session=sess,
                                                decode_op=decode_op1,
                                                network=network,
                                                dataset=eval1_data,
                                                label_type=label_type_main,
                                                is_test=True,
                                                eval_batch_size=batch_size,
                                                is_multitask=True,
                                                is_main=True)
                        if label_type_second == 'character':
                            do_eval_cer(session=sess,
                                        decode_op=decode_op2,
                                        network=network,
                                        dataset=eval1_data,
                                        label_type=label_type_second,
                                        eval_batch_size=batch_size,
                                        is_multitask=True,
                                        is_main=False)
                        elif label_type_second == 'phone':
                            do_eval_per(session=sess,
                                        per_op=per_op2,
                                        network=network,
                                        dataset=eval1_data,
                                        eval_batch_size=batch_size,
                                        is_multitask=True)

                        print('■eval2 Evaluation:■')
                        cer_eval2 = do_eval_cer(session=sess,
                                                decode_op=decode_op1,
                                                network=network,
                                                dataset=eval2_data,
                                                label_type=label_type_main,
                                                is_test=-True,
                                                eval_batch_size=batch_size,
                                                is_multitask=True,
                                                is_main=True)
                        if label_type_second == 'character':
                            do_eval_cer(session=sess,
                                        decode_op=decode_op2,
                                        network=network,
                                        dataset=eval2_data,
                                        label_type=label_type_second,
                                        eval_batch_size=batch_size,
                                        is_multitask=True,
                                        is_main=False)
                        elif label_type_second == 'phone':
                            do_eval_per(session=sess,
                                        per_op=per_op2,
                                        network=network,
                                        dataset=eval2_data,
                                        eval_batch_size=batch_size,
                                        is_multitask=True)

                        print('■eval3 Evaluation:■')
                        cer_eval3 = do_eval_cer(session=sess,
                                                decode_op=decode_op1,
                                                network=network,
                                                dataset=eval3_data,
                                                label_type=label_type_main,
                                                is_test=True,
                                                eval_batch_size=batch_size,
                                                is_multitask=True,
                                                is_main=True)
                        if label_type_second == 'character':
                            do_eval_cer(session=sess,
                                        decode_op=decode_op2,
                                        network=network,
                                        dataset=eval3_data,
                                        label_type=label_type_second,
                                        eval_batch_size=batch_size,
                                        is_multitask=True,
                                        is_main=False)
                        elif label_type_second == 'phone':
                            do_eval_per(session=sess,
                                        per_op=per_op2,
                                        network=network,
                                        dataset=eval3_data,
                                        eval_batch_size=batch_size,
                                        is_multitask=True)

                        cer_mean = (cer_eval1 + cer_eval2 + cer_eval3) / 3.
                        print('■Mean:■')
                        print('  CER: %f %%' %
                              (cer_mean * 100))

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

    if corpus['label_type_main'] == 'character':
        output_size_main = 147
    elif corpus['label_type_main'] == 'kanji':
        output_size_main = 3386

    if corpus['label_type_second'] == 'phone':
        output_size_second = 38
    elif corpus['label_type_second'] == 'character':
        output_size_second = 147

    # Model setting
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(batch_size=param['batch_size'],
                       input_size=feature['input_size'] * feature['num_stack'],
                       num_cell=param['num_cell'],
                       num_layer=param['num_layer'],
                       num_layer2=param['num_layer2'],
                       #    bottleneck_dim=param['bottleneck_dim'],
                       output_size=output_size_main,
                       output_size2=output_size_second,
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
    network.model_dir = mkdir('/n/sd8/inaguma/result/csj/monolog/')
    network.model_dir = mkdir_join(network.model_dir, 'ctc')
    network.model_dir = mkdir_join(
        network.model_dir, corpus['label_type_main'] + '_' + corpus['label_type_second'])
    network.model_dir = mkdir_join(
        network.model_dir, corpus['train_data_size'])
    network.model_dir = mkdir_join(network.model_dir, network.model_name)

    # Reset model directory
    if not isfile(join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('multitaskctc_csj_' + corpus['label_type_main'] + '_' +
                 corpus['label_type_second'] + '_' + corpus['train_data_size'])

    # Save config file
    shutil.copyfile(config_path, join(network.model_dir, 'config.yml'))

    sys.stdout = open(join(network.model_dir, 'train.log'), 'w')
    print(network.model_name)
    do_train(network=network,
             optimizer=param['optimizer'],
             learning_rate=param['learning_rate'],
             batch_size=param['batch_size'],
             epoch_num=param['num_epoch'],
             label_type_main=corpus['label_type_main'],
             label_type_second=corpus['label_type_second'],
             num_stack=feature['num_stack'],
             num_skip=feature['num_skip'],
             train_data_size=corpus['train_data_size'])
    sys.stdout = sys.__stdout__


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        sys.exit(0)

    main(config_path=args[1])
