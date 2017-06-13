#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train Attention-based model (TIMIT corpus)."""

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
from data.read_dataset_attention import DataSet
# from models.attention.load_model import load
from models.attention import blstm_attention_seq2seq
from metric.attention import do_eval_per, do_eval_cer
from utils.sparsetensor import list2sparsetensor
from utils.directory import mkdir, mkdir_join
from utils.parameter import count_total_parameters
from utils.loss import save_loss


def do_train(network, optimizer, learning_rate, batch_size, epoch_num,
             label_type, num_stack, num_skip):
    """Run training. If target labels are phone, the model is evaluated by PER
    with 39 phones.
    Args:
        network: network to train
        optimizer: string, the name of optimizer.
            ex.) adam, rmsprop
        learning_rate: initial learning rate
        batch_size: size of mini batch
        epoch_num: epoch num to train
        label_type: phone39 or phone48 or phone61 or character
        num_stack: int, the number of frames to stack
        num_skip: int, the number of frames to skip
    """
    # Load dataset
    train_data = DataSet(data_type='train', label_type=label_type,
                         num_stack=num_stack, num_skip=num_skip,
                         is_sorted=True)
    if label_type == 'character':
        dev_data = DataSet(data_type='dev', label_type='character',
                           num_stack=num_stack, num_skip=num_skip,
                           is_sorted=False)
        test_data = DataSet(data_type='test', label_type='character',
                            num_stack=num_stack, num_skip=num_skip,
                            is_sorted=False)
    else:
        dev_data = DataSet(data_type='dev', label_type='phone39',
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
        loss_op = network.compute_loss()
        train_op = network.train(optimizer=optimizer,
                                 learning_rate_init=learning_rate,
                                 is_scheduled=False)
        decode_op_train, decode_op_infer = network.decoder(
            decode_type='beam_search',
            beam_width=20)
        ler_op = network.compute_ler()

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
        # Create a session for running operation on the graph
        with tf.Session() as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                network.model_dir, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Train model
            iter_per_epoch = int(train_data.data_num / batch_size)
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
                inputs, labels, inputs_seq_len, labels_seq_len = train_data.next_batch(
                    batch_size=batch_size)
                feed_dict_train = {
                    network.inputs: inputs,
                    network.labels: labels,
                    network.inputs_seq_len: inputs_seq_len,
                    network.labels_seq_len: labels_seq_len,
                    network.keep_prob_input: network.dropout_ratio_input,
                    network.keep_prob_hidden: network.dropout_ratio_hidden,
                    network.learning_rate: learning_rate
                }

                # Create feed dictionary for next mini batch (dev)
                inputs, labels, inputs_seq_len, labels_seq_len = dev_data.next_batch(
                    batch_size=batch_size)
                feed_dict_dev = {
                    network.inputs: inputs,
                    network.labels: labels,
                    network.inputs_seq_len: inputs_seq_len,
                    network.labels_seq_len: labels_seq_len,
                    network.keep_prob_input: network.dropout_ratio_input,
                    network.keep_prob_hidden: network.dropout_ratio_hidden,
                    network.learning_rate: learning_rate
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
                    feed_dict_train[network.keep_prob_input] = 1.0
                    feed_dict_train[network.keep_prob_hidden] = 1.0
                    feed_dict_dev[network.keep_prob_input] = 1.0
                    feed_dict_dev[network.keep_prob_hidden] = 1.0

                    # Predict ids
                    predicted_ids_train, predicted_ids_infer = sess.run(
                        [decode_op_train, decode_op_infer],
                        feed_dict=feed_dict_train)

                    # Convert to sparsetensor for computing LER
                    indices, values, shape = list2sparsetensor(labels)
                    indices_train, values_train, shape_train = list2sparsetensor(
                        predicted_ids_train)
                    indices_infer, values_infer, shape_infer = list2sparsetensor(
                        predicted_ids_infer)

                    feed_dict_ler = {
                        network.label_indices_true: indices,
                        network.label_values_true:  values,
                        network.label_shape_true: shape,
                        network.label_indices_pred: indices_infer,
                        network.label_values_pred:  values_infer,
                        network.label_shape_pred: shape_infer
                    }
                    # TODO: Add dev version

                    # Compute accuracy & update event file
                    ler_train = sess.run(
                        [ler_op], feed_dict=feed_dict_ler)
                    # ler_train, summary_str_train = sess.run(
                    #     [ler_op, summary_train], feed_dict=feed_dict_train)
                    # ler_dev, summary_str_dev = sess.run(
                    #     [ler_op, summary_dev], feed_dict=feed_dict_dev)
                    # summary_writer.add_summary(summary_str_train, step + 1)
                    # summary_writer.add_summary(summary_str_dev, step + 1)
                    # summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print("Step %d: loss = %.3f (%.3f) / ler = %.4f (%.4f) (%.3f min)" %
                          (step + 1, loss_train, loss_dev, ler_train,
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
                    save_path = saver.save(
                        sess, checkpoint_file, global_step=epoch)
                    print("Model saved in file: %s" % save_path)

                #     if epoch >= 10:
                #         start_time_eval = time.time()
                #         if label_type == 'character':
                #             print('=== Dev Data Evaluation ===')
                #             error_dev_epoch = do_eval_cer(
                #                 session=sess,
                #                 decode_op=decode_op,
                #                 network=network,
                #                 dataset=dev_data,
                #                 eval_batch_size=1)
                #             print('  CER: %f %%' % (error_dev_epoch * 100))
                #
                #             if error_dev_epoch < error_best:
                #                 error_best = error_dev_epoch
                #                 print('■■■ ↑Best Score (CER)↑ ■■■')
                #
                #                 print('=== Test Data Evaluation ===')
                #                 error_test_epoch = do_eval_cer(
                #                     session=sess,
                #                     decode_op=decode_op,
                #                     network=network,
                #                     dataset=test_data,
                #                     eval_batch_size=1)
                #                 print('  CER: %f %%' %
                #                       (error_test_epoch * 100))
                #
                #         else:
                #             print('=== Dev Data Evaluation ===')
                #             error_dev_epoch = do_eval_per(
                #                 session=sess,
                #                 decode_op=decode_op,
                #                 per_op=ler_op,
                #                 network=network,
                #                 dataset=dev_data,
                #                 label_type=label_type,
                #                 eval_batch_size=1)
                #             print('  PER: %f %%' % (error_dev_epoch * 100))
                #
                #             if error_dev_epoch < error_best:
                #                 error_best = error_dev_epoch
                #                 print('■■■ ↑Best Score (PER)↑ ■■■')
                #
                #                 print('=== Test Data Evaluation ===')
                #                 error_test_epoch = do_eval_per(
                #                     session=sess,
                #                     decode_op=decode_op,
                #                     per_op=ler_op,
                #                     network=network,
                #                     dataset=test_data,
                #                     label_type=label_type,
                #                     eval_batch_size=1)
                #                 print('  PER: %f %%' % (error_dev_epoch * 100))
                #
                #         duration_eval = time.time() - start_time_eval
                #         print('Evaluation time: %.3f min' %
                #               (duration_eval / 60))
                #
                # start_time_epoch = time.time()
                # start_time_step = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Save train & dev loss
            save_loss(csv_steps, csv_train_loss, csv_dev_loss,
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

    if corpus['label_type'] == 'phone61':
        output_size = 63
    elif corpus['label_type'] == 'phone48':
        output_size = 50
    elif corpus['label_type'] == 'phone39':
        output_size = 41
    elif corpus['label_type'] == 'character':
        output_size = 32

    # Model setting
    # CTCModel = load(model_type=config['model_name'])
    network = (
        batch_size=batch_size,
        input_size=inputs[0].shape[1],
        encoder_num_units=256,
        encoder_num_layer=2,
        attention_dim=128,
        decoder_num_units=256,
        decoder_num_layer=1,
        embedding_dim=50,
        output_size=output_size,
        sos_index=output_size - 2,
        eos_index=output_size - 1,
        max_decode_length=50,
        parameter_init=0.1,
        clip_grad=5.0,
        clip_activation_encoder=50,
        clip_activation_decoder=50,
        dropout_ratio_input=1.0,
        dropout_ratio_hidden=1.0,
        weight_decay=1e-6,
        beam_width=0)

    network = blstm_attention_seq2seq.BLSTMAttetion(
        batch_size=param['batch_size'],
        input_size=feature['input_size'] * feature['num_stack'],
        encoder_num_unit=param['encoder_num_units'],
        encoder_num_layer=num_cell=param['num_cell'],
        num_layer=param['num_layer'],
        output_size=output_size,
        clip_grad=param['clip_grad'],
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
    if param['num_proj'] != 0:
        network.model_name += '_proj' + str(param['num_proj'])
    if feature['num_stack'] != 1:
        network.model_name += '_stack' + str(feature['num_stack'])
    if param['weight_decay'] != 0:
        network.model_name += '_weightdecay' + str(param['weight_decay'])

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
    setproctitle('ctc_timit_' + corpus['label_type'])

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
