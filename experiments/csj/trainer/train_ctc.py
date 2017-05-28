#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train CTC network (CSJ corpus)."""

import os
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
from evaluation.eval_ctc import do_eval_per, do_eval_cer
from utils.data.sparsetensor import list2sparsetensor
from utils.util import mkdir, join


def do_train(network, optimizer, learning_rate, batch_size, epoch_num,
             label_type, num_stack, num_skip, train_data_size):
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
        train_data_size: default or large
    """
    # tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # read dataset
        train_data = DataSet(data_type='train', label_type=label_type,
                             train_data_size=train_data_size,
                             num_stack=num_stack, num_skip=num_skip,
                             is_sorted=True)
        dev_data = DataSet(data_type='dev', label_type='character',
                           train_data_size=train_data_size,
                           num_stack=num_stack, num_skip=num_skip,
                           is_sorted=False)
        eval1_data = DataSet(data_type='eval1', label_type='character',
                             train_data_size=train_data_size,
                             num_stack=num_stack, num_skip=num_skip,
                             is_sorted=False)
        eval2_data = DataSet(data_type='eval2', label_type='character',
                             train_data_size=train_data_size,
                             num_stack=num_stack, num_skip=num_skip,
                             is_sorted=False)
        eval3_data = DataSet(data_type='eval3', label_type='character',
                             train_data_size=train_data_size,
                             num_stack=num_stack, num_skip=num_skip,
                             is_sorted=False)

        # add to the graph each operation
        network.define()
        loss_op = network.loss()
        train_op = network.train(optimizer=optimizer,
                                 learning_rate_init=learning_rate,
                                 is_scheduled=False)
        # decode_op = network.greedy_decoder()
        decode_op = network.beam_search_decoder(beam_width=20)
        per_op = network.ler(decode_op)

        # build the summary tensor based on the TensorFlow collection of
        # summaries
        summary_train = tf.summary.merge(network.summaries_train)
        # summary_dev = tf.summary.merge(network.summaries_dev)

        # add the variable initializer operation
        init_op = tf.global_variables_initializer()

        # create a saver for writing training checkpoints
        saver = tf.train.Saver(max_to_keep=epoch_num)

        # create a session for running operation on the graph
        with tf.Session() as sess:
            # instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                network.model_dir, sess.graph)

            # initialize parameters
            sess.run(init_op)

            # train model
            iter_per_epoch = int(train_data.data_num / batch_size)
            if (train_data.data_num / batch_size) != int(train_data.data_num / batch_size):
                iter_per_epoch += 1
            max_steps = iter_per_epoch * epoch_num
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            error_best = 1
            for step in range(max_steps):
                # create feed dictionary for next mini batch (train)
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

                # create feed dictionary for next mini batch (dev)
                # inputs, labels, seq_len, _ = dev_data.next_batch(
                #     batch_size=batch_size)
                # indices, values, dense_shape = list2sparsetensor(labels)
                # feed_dict_dev = {
                #     network.inputs_pl: inputs,
                #     network.label_indices_pl: indices,
                #     network.label_values_pl: values,
                #     network.label_shape_pl: dense_shape,
                #     network.seq_len_pl: seq_len,
                #     network.keep_prob_input_pl: network.dropout_ratio_input,
                #     network.keep_prob_hidden_pl: network.dropout_ratio_hidden
                # }

                # compute loss
                _, loss_train = sess.run(
                    [train_op, loss_op], feed_dict=feed_dict_train)
                # loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)

                if (step + 1) % 100 == 0:
                    # change feed dict for evaluation
                    feed_dict_train[network.keep_prob_input_pl] = 1.0
                    feed_dict_train[network.keep_prob_hidden_pl] = 1.0
                    # feed_dict_dev[network.keep_prob_input_pl] = 1.0
                    # feed_dict_dev[network.keep_prob_hidden_pl] = 1.0

                    # compute accuracy
                    ler_train = sess.run(per_op, feed_dict=feed_dict_train)
                    # ler_dev = sess.run(per_op, feed_dict=feed_dict_dev)

                    duration_step = time.time() - start_time_step
                    # print('Step %d: loss = %.3f (%.3f) / ler = %.4f (%.4f) (%.3f min)' %
                    #       (step + 1, loss_train, loss_dev, ler_train, ler_dev, duration_step / 60))
                    print('Step %d: loss = %.3f / ler = %.4f (%.3f min)' %
                          (step + 1, loss_train, ler_train, duration_step / 60))

                    # sys.stdout.write
                    sys.stdout.flush()
                    start_time_step = time.time()

                    # update event file
                    summary_str = sess.run(
                        summary_train, feed_dict=feed_dict_train)
                    summary_writer.add_summary(summary_str, step + 1)
                    # summary_str = sess.run(
                    #     summary_dev, feed_dict=feed_dict_dev)
                    # summary_writer.add_summary(summary_str, step + 1)
                    summary_writer.flush()

                # save checkpoint and evaluate model per epoch
                if (step + 1) % iter_per_epoch == 0 or (step + 1) == max_steps:
                    duration_epoch = time.time() - start_time_epoch
                    epoch = (step + 1) // iter_per_epoch
                    print('-----EPOCH:%d (%.3f min)-----' %
                          (epoch, duration_epoch / 60))

                    # save model (check point)
                    checkpoint_file = os.path.join(
                        network.model_dir, 'model.ckpt')
                    save_path = saver.save(
                        sess, checkpoint_file, global_step=epoch)
                    print("Model saved in file: %s" % save_path)

                    start_time_eval = time.time()
                    if label_type == 'character':
                        print('■Dev Evaluation:■')
                        error_epoch = do_eval_cer(session=sess,
                                                  decode_op=decode_op,
                                                  network=network,
                                                  dataset=dev_data,
                                                  eval_batch_size=batch_size)

                        if error_epoch < error_best:
                            error_best = error_epoch
                            print('■■■ ↑Best Score (CER)↑ ■■■')

                            # evaluate only when the score over dev set is best
                            print('■eval1 Evaluation:■')
                            do_eval_cer(session=sess, decode_op=decode_op,
                                        network=network, dataset=eval1_data,
                                        eval_batch_size=batch_size)
                            print('■eval2 Evaluation:■')
                            do_eval_cer(session=sess, decode_op=decode_op,
                                        network=network, dataset=eval2_data,
                                        eval_batch_size=batch_size)
                            print('■eval3 Evaluation:■')
                            do_eval_cer(session=sess, decode_op=decode_op,
                                        network=network, dataset=eval3_data,
                                        eval_batch_size=batch_size)

                    else:
                        print('■Dev Evaluation:■')
                        error_epoch = do_eval_per(session=sess,
                                                  per_op=per_op,
                                                  network=network,
                                                  dataset=dev_data,
                                                  eval_batch_size=batch_size)

                        if error_epoch < error_best:
                            error_best = error_epoch
                            print('■■■ ↑Best Score (PER)↑ ■■■')

                            # evaluate only when the score over dev set is best
                            print('■eval1 Evaluation:■')
                            do_eval_per(session=sess, per_op=per_op,
                                        network=network, dataset=eval1_data,
                                        eval_batch_size=batch_size)
                            print('■eval2 Evaluation:■')
                            do_eval_per(session=sess, per_op=per_op,
                                        network=network, dataset=eval2_data,
                                        eval_batch_size=batch_size)
                            print('■eval3 Evaluation:■')
                            do_eval_per(session=sess, per_op=per_op,
                                        network=network, dataset=eval3_data,
                                        eval_batch_size=batch_size)

                    duration_eval = time.time() - start_time_eval
                    print('Evaluation time: %.3f min' %
                          (duration_eval / 60))

                    start_time_epoch = time.time()
                    start_time_step = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # make a confirmation file to prove that training was finished
            # correctly
            with open(os.path.join(network.model_dir, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path):

    # read a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        corpus = config['corpus']
        feature = config['feature']
        param = config['param']

    if corpus['label_type'] == 'phone':
        output_size = 38
    elif corpus['label_type'] == 'character':
        output_size = 147

    # load model
    CTCModel = load(model_type=config['model_name'])
    network = CTCModel(batch_size=param['batch_size'],
                       input_size=feature['input_size'] * feature['num_stack'],
                       num_cell=param['num_cell'],
                       num_layers=param['num_layer'],
                       output_size=output_size,
                       clip_grad=param['clip_grad'],
                       clip_activation=param['clip_activation'],
                       dropout_ratio_input=param['dropout_input'],
                       dropout_ratio_hidden=param['dropout_hidden'],
                       num_proj=param['num_proj'])

    network.model_name = config['model_name'].upper()
    network.model_name += '_' + str(param['num_cell'])
    network.model_name += '_' + str(param['num_layer'])
    network.model_name += '_' + param['optimizer']
    network.model_name += '_lr' + str(param['learning_rate'])
    if feature['num_stack'] != 1:
        network.model_name += '_stack' + str(feature['num_stack'])

    # set save path
    network.model_dir = mkdir('/n/sd8/inaguma/result/csj/monolog/')
    network.model_dir = join(network.model_dir, 'ctc')
    network.model_dir = join(network.model_dir, corpus['label_type'])
    network.model_dir = join(network.model_dir, corpus['train_data_size'])
    network.model_dir = join(network.model_dir, network.model_name)

    # reset model directory
    if not os.path.isfile(os.path.join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # set process name
    setproctitle('ctc_csj_' + corpus['label_type'] + '_' +
                 param['optimizer'] + '_' + corpus['train_data_size'])

    # save config file
    shutil.copyfile(config_path, os.path.join(network.model_dir, 'config.yml'))

    sys.stdout = open(os.path.join(network.model_dir, 'train.log'), 'w')
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
        sys.exit(0)

    main(config_path=args[1])
