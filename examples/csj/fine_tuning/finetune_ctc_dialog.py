#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Fine tune CTC network (CSJ corpus, for dialog)."""

import os
import sys
import time
import tensorflow as tf
from setproctitle import setproctitle
import yaml
import shutil
import tensorflow.contrib.slim as slim

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from data.read_dataset_ctc import DataSet
from data.read_dataset_ctc_dialog import DataSet as DataSetDialog
from models.ctc.load_model import load
from evaluation.eval_ctc import do_eval_per, do_eval_cer
from evaluation.eval_ctc_dialog import do_eval_fmeasure
from utils.data.sparsetensor import list2sparsetensor, sparsetensor2list
from utils.util import mkdir, join
from utils.parameter import count_total_parameters
from utils.loss import save_loss
from utils.labels.phone import num2phone
from utils.labels.character import num2char


def do_fine_tune(network, optimizer, learning_rate, batch_size, epoch_num,
                 label_type, num_stack, num_skip, social_signal_type,
                 trained_model_path, restore_epoch=None):
    """Run training.
    Args:
        network: network to train
        optimizer: adam or adadelta or rmsprop
        learning_rate: initial learning rate
        batch_size: size of mini batch
        epoch_num: epoch num to train
        label_type: phone or character
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
        social_signal_type: insert or insert2 or insert3 or remove
        trained_model_path: path to the pre-trained model
        restore_epoch: epoch of the model to restore
    """
    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():
        # Read dataset
        train_data = DataSetDialog(data_type='train', label_type=label_type,
                                   social_signal_type=social_signal_type,
                                   num_stack=num_stack, num_skip=num_skip,
                                   is_sorted=True)
        dev_data = DataSetDialog(data_type='dev', label_type=label_type,
                                 social_signal_type=social_signal_type,
                                 num_stack=num_stack, num_skip=num_skip,
                                 is_sorted=False)
        test_data = DataSetDialog(data_type='test', label_type=label_type,
                                  social_signal_type=social_signal_type,
                                  num_stack=num_stack, num_skip=num_skip,
                                  is_sorted=False)
        # TODO：作る
        # eval1_data = DataSet(data_type='eval1', label_type=label_type,
        #                      social_signal_type=social_signal_type,
        #                      num_stack=num_stack, num_skip=num_skip,
        #                      is_sorted=False)
        # eval2_data = DataSet(data_type='eval2', label_type=label_type,
        #                      social_signal_type=social_signal_type,
        #                      num_stack=num_stack, num_skip=num_skip,
        #                      is_sorted=False)
        # eval3_data = DataSet(data_type='eval3', label_type=label_type,
        #                      social_signal_type=social_signal_type,
        #                      num_stack=num_stack, num_skip=num_skip,
        #                      is_sorted=False)

        # Add to the graph each operation
        loss_op = network.loss()
        train_op = network.train(optimizer=optimizer,
                                 learning_rate_init=learning_rate,
                                 is_scheduled=False)
        decode_op = network.decoder(decode_type='beam_search',
                                    beam_width=20)
        per_op = network.ler(decode_op)

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

            # Restore pre-trained model's parameters
            ckpt = tf.train.get_checkpoint_state(trained_model_path)
            if ckpt:
                # Use last saved model
                model_path = ckpt.model_checkpoint_path
                if restore_epoch is not None:
                    model_path = model_path.split('/')[:-1]
                    model_path = '/'.join(model_path) + \
                        '/model.ckpt-' + str(restore_epoch)
            else:
                raise ValueError('There are not any checkpoints.')
            exclude = ['output/Variable', 'output/Variable_1']
            variables_to_restore = slim.get_variables_to_restore(
                exclude=exclude)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, model_path)
            print("Model restored: " + model_path)

            # Train model
            iter_per_epoch = int(train_data.data_num / batch_size)
            if (train_data.data_num / batch_size) != int(train_data.data_num / batch_size):
                iter_per_epoch += 1
            max_steps = iter_per_epoch * epoch_num
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            fmean_best = 0
            for step in range(max_steps):
                # Create feed dictionary for next mini batch (train)
                inputs, labels, seq_len, _ = train_data.next_batch(
                    batch_size=batch_size)
                indices, values, dense_shape = list2sparsetensor(labels)
                feed_dict_train = {
                    network.inputs_pl: inputs,
                    network.label_indices_pl: indices,
                    network.label_values_pl: values,
                    network.label_shape_pl: dense_shape,
                    network.seq_len_pl: seq_len,
                    network.keep_prob_input_pl: network.dropout_ratio_input,
                    network.keep_prob_hidden_pl: network.dropout_ratio_hidden,
                    network.lr_pl: learning_rate
                }

                # Create feed dictionary for next mini batch (dev)
                inputs, labels, seq_len, _ = dev_data.next_batch(
                    batch_size=batch_size)
                indices, values, dense_shape = list2sparsetensor(labels)
                feed_dict_dev = {
                    network.inputs_pl: inputs,
                    network.label_indices_pl: indices,
                    network.label_values_pl: values,
                    network.label_shape_pl: dense_shape,
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

                    # Compute accuracy & \update event file
                    ler_train, summary_str_train = sess.run([per_op, summary_train],
                                                            feed_dict=feed_dict_train)
                    ler_dev, summary_str_dev, labels_st = sess.run([per_op, summary_dev, decode_op],
                                                                   feed_dict=feed_dict_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    # Decode
                    # try:
                    #     labels_pred = sparsetensor2list(labels_st, batch_size)
                    # except:
                    #     labels_pred = [[0] * batch_size]

                    duration_step = time.time() - start_time_step
                    print('Step %d: loss = %.3f (%.3f) / ler = %.4f (%.4f) (%.3f min)' %
                          (step + 1, loss_train, loss_dev, ler_train, ler_dev, duration_step / 60))

                    # if label_type == 'character':
                    #     if social_signal_type == 'remove':
                    #         map_file_path = '../evaluation/mapping_files/ctc/char2num_remove.txt'
                    #     else:
                    #         map_file_path = '../evaluation/mapping_files/ctc/char2num_' + \
                    #             social_signal_type + '.txt'
                    #     print('True: %s' % num2char(labels[-1], map_file_path))
                    #     print('Pred: %s' % num2char(
                    #         labels_pred[-1], map_file_path))
                    # elif label_type == 'phone':
                    #     if social_signal_type == 'remove':
                    #         map_file_path = '../evaluation/mapping_files/ctc/phone2num_remove.txt'
                    #     else:
                    #         map_file_path = '../evaluation/mapping_files/ctc/phone2num_' + \
                    #             social_signal_type + '.txt'
                    #     print('True: %s' % num2phone(
                    #         labels[-1], map_file_path))
                    #     print('Pred: %s' % num2phone(
                    #         labels_pred[-1], map_file_path))

                    sys.stdout.flush()
                    start_time_step = time.time()

                # Save checkpoint and evaluate model per epoch
                if (step + 1) % iter_per_epoch == 0 or (step + 1) == max_steps:
                    duration_epoch = time.time() - start_time_epoch
                    epoch = (step + 1) // iter_per_epoch
                    print('-----EPOCH:%d (%.3f min)-----' %
                          (epoch, duration_epoch / 60))

                    # Save model (check point)
                    checkpoint_file = os.path.join(
                        network.model_dir, 'model.ckpt')
                    save_path = saver.save(
                        sess, checkpoint_file, global_step=epoch)
                    print("Model saved in file: %s" % save_path)

                    start_time_eval = time.time()
                    if label_type == 'character':
                        print('■Dev Evaluation:■')
                        fmean_epoch = do_eval_fmeasure(session=sess, decode_op=decode_op,
                                                       network=network, dataset=dev_data,
                                                       label_type=label_type,
                                                       social_signal_type=social_signal_type)
                        # error_epoch = do_eval_cer(session=sess,
                        #                           decode_op=decode_op,
                        #                           network=network,
                        #                           dataset=dev_data,
                        #                           eval_batch_size=batch_size)

                        if fmean_epoch > fmean_best:
                            fmean_best = fmean_epoch
                            print('■■■ ↑Best Score (F-measure)↑ ■■■')

                            do_eval_fmeasure(session=sess, decode_op=decode_op,
                                             network=network, dataset=test_data,
                                             label_type=label_type,
                                             social_signal_type=social_signal_type)
                            # print('■eval1 Evaluation:■')
                            # do_eval_cer(session=sess, decode_op=decode_op,
                            #             network=network, dataset=eval1_data,
                            #             eval_batch_size=batch_size)
                            # print('■eval2 Evaluation:■')
                            # do_eval_cer(session=sess, decode_op=decode_op,
                            #             network=network, dataset=eval2_data,
                            #             eval_batch_size=batch_size)
                            # print('■eval3 Evaluation:■')
                            # do_eval_cer(session=sess, decode_op=decode_op,
                            #             network=network, dataset=eval3_data,
                            #             eval_batch_size=batch_size)

                    else:
                        print('■Dev Evaluation:■')
                        fmean_epoch = do_eval_fmeasure(session=sess, decode_op=decode_op,
                                                       network=network, dataset=dev_data,
                                                       label_type=label_type,
                                                       social_signal_type=social_signal_type)
                        # error_epoch = do_eval_per(session=sess,
                        #                           per_op=per_op,
                        #                           network=network,
                        #                           dataset=dev_data,
                        #                           eval_batch_size=batch_size)

                        if fmean_epoch < fmean_best:
                            fmean_best = fmean_epoch
                            print('■■■ ↑Best Score (F-measure)↑ ■■■')

                            do_eval_fmeasure(session=sess, decode_op=decode_op,
                                             network=network, dataset=test_data,
                                             label_type=label_type,
                                             social_signal_type=social_signal_type)
                            # print('■eval1 Evaluation:■')
                            # do_eval_per(session=sess, per_op=per_op,
                            #             network=network, dataset=eval1_data,
                            #             eval_batch_size=batch_size)
                            # print('■eval2 Evaluation:■')
                            # do_eval_per(session=sess, per_op=per_op,
                            #             network=network, dataset=eval2_data,
                            #             eval_batch_size=batch_size)
                            # print('■eval3 Evaluation:■')
                            # do_eval_per(session=sess, per_op=per_op,
                            #             network=network, dataset=eval3_data,
                            #             eval_batch_size=batch_size)

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
            with open(os.path.join(network.model_dir, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path, trained_model_path):

    restore_epoch = None  # if None, restore the final epoch

    # Read a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type'] == 'phone':
        if corpus['social_signal_type'] in ['insert', 'insert3']:
            output_size = 41
        elif corpus['social_signal_type'] == 'insert2':
            output_size = 44
        elif corpus['social_signal_type'] == 'remove':
            output_size = 38
    elif corpus['label_type'] == 'character':
        if corpus['social_signal_type'] in ['insert', 'insert3']:
            output_size = 150
        elif corpus['social_signal_type'] == 'insert2':
            output_size = 153
        elif corpus['social_signal_type'] == 'remove':
            output_size = 147

    # Load model
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(batch_size=param['batch_size'],
                       input_size=feature['input_size'] * feature['num_stack'],
                       num_cell=param['num_cell'],
                       num_layer=param['num_layer'],
                       output_size=output_size,
                       clip_gradients=param['clip_grad'],
                       clip_activation=param['clip_activation'],
                       dropout_ratio_input=param['dropout_input'],
                       dropout_ratio_hidden=param['dropout_hidden'],
                       num_proj=param['num_proj'],
                       weight_decay=param['weight_decay'])

    network.model_name = config['model_name'].upper()
    network.model_name += '_' + str(param['num_cell'])
    network.model_name += '_' + str(param['num_layer'])
    network.model_name += '_' + param['optimizer']
    network.model_name += '_lr' + str(param['learning_rate'])
    if feature['num_stack'] != 1:
        network.model_name += '_stack' + str(feature['num_stack'])
    network.model_name += '_transfer_' + corpus['transfer_data_size']

    # Set save path
    network.model_dir = mkdir('/n/sd8/inaguma/result/csj/dialog/')
    network.model_dir = join(network.model_dir, 'ctc')
    network.model_dir = join(network.model_dir, corpus['label_type'])
    network.model_dir = join(network.model_dir, corpus['social_signal_type'])
    network.model_dir = join(network.model_dir, network.model_name)

    # Reset model directory
    if not os.path.isfile(os.path.join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('ctc_csj_dialog_' + corpus['label_type'] + '_' +
                 param['optimizer'] + '_' + corpus['social_signal_type'] +
                 '_transfer_' + corpus['transfer_data_size'])

    # Save config file
    shutil.copyfile(config_path, os.path.join(network.model_dir, 'config.yml'))

    sys.stdout = open(os.path.join(network.model_dir, 'train.log'), 'w')
    print(network.model_name)
    do_fine_tune(network=network,
                 optimizer=param['optimizer'],
                 learning_rate=param['learning_rate'],
                 batch_size=param['batch_size'],
                 epoch_num=param['num_epoch'],
                 label_type=corpus['label_type'],
                 num_stack=feature['num_stack'],
                 num_skip=feature['num_skip'],
                 social_signal_type=corpus['social_signal_type'],
                 trained_model_path=trained_model_path,
                 restore_epoch=restore_epoch)
    sys.stdout = sys.__stdout__


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        ValueError(
            'Usage: python fine_tune_ctc.py path_to_config path_to_trained_model')

    main(config_path=args[1], trained_model_path=args[2])
