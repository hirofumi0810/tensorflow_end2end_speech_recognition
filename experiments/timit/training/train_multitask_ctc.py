#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the multi-task CTC model (TIMIT corpus)."""

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

sys.path.append('../../../')
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.timit.metrics.ctc import do_eval_per, do_eval_cer
from experiments.utils.data.sparsetensor import list2sparsetensor
from experiments.utils.training.learning_rate_controller import Controller

from experiments.utils.directory import mkdir, mkdir_join
from experiments.utils.parameter import count_total_parameters
from experiments.utils.csv import save_loss, save_ler
from models.ctc.load_model_multitask import load


def do_train(network, params):
    """Run multi-task CTC training. The target labels in the main task is
    characters and those in the sub task is 61 phones. The model is
    evaluated by CER and PER with 39 phones.
    Args:
        network: network to train
        params: A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=params['batch_size'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True)
    dev_data = Dataset(
        data_type='dev', label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=params['batch_size'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    test_data = Dataset(
        data_type='test', label_type_main=params['label_type_main'],
        label_type_sub='phone39', batch_size=1,
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        network.create_placeholders()
        learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

        # Add to the graph each operation
        loss_op, logits_main, logits_sub = network.compute_loss(
            network.inputs_pl_list[0],
            network.labels_pl_list[0],
            network.labels_sub_pl_list[0],
            network.inputs_seq_len_pl_list[0],
            network.keep_prob_input_pl_list[0],
            network.keep_prob_hidden_pl_list[0],
            network.keep_prob_output_pl_list[0])
        train_op = network.train(
            loss_op,
            optimizer=params['optimizer'],
            learning_rate=learning_rate_pl)
        decode_op_character, decode_op_phone = network.decoder(
            logits_main,
            logits_sub,
            network.inputs_seq_len_pl_list[0],
            decode_type='beam_search',
            beam_width=20)
        cer_op, per_op = network.compute_ler(
            decode_op_character, decode_op_phone,
            network.labels_pl_list[0], network.labels_sub_pl_list[0])

        # Define learning rate controller
        lr_controller = Controller(
            learning_rate_init=params['learning_rate'],
            decay_start_epoch=params['decay_start_epoch'],
            decay_rate=params['decay_rate'],
            decay_patient_epoch=params['decay_patient_epoch'],
            lower_better=True)

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
        csv_cer_train, csv_cer_dev = [], []
        csv_per_train, csv_per_dev = [], []
        # Create a session for running operation on the graph
        with tf.Session() as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                network.model_dir, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Train model
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            cer_dev_best = 1
            learning_rate = float(params['learning_rate'])
            epoch = 1
            for step, (data, next_epoch_flag) in enumerate(train_data()):

                # Create feed dictionary for next mini batch (train)
                inputs, labels_char, labels_phone, inputs_seq_len, _ = data
                feed_dict_train = {
                    network.inputs_pl_list[0]: inputs,
                    network.labels_pl_list[0]: list2sparsetensor(labels_char, padded_value=-1),
                    network.labels_sub_pl_list[0]: list2sparsetensor(labels_phone, padded_value=-1),
                    network.inputs_seq_len_pl_list[0]: inputs_seq_len,
                    network.keep_prob_input_pl_list[0]: network.dropout_ratio_input,
                    network.keep_prob_hidden_pl_list[0]: network.dropout_ratio_hidden,
                    network.keep_prob_output_pl_list[0]: network.dropout_ratio_output,
                    learning_rate_pl: learning_rate
                }

                # Update parameters
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % 10 == 0:

                    # Create feed dictionary for next mini batch (dev)
                    (inputs, labels_char, labels_phone,
                     inputs_seq_len, _), _ = dev_data().__next__()
                    feed_dict_dev = {
                        network.inputs_pl_list[0]: inputs,
                        network.labels_pl_list[0]: list2sparsetensor(labels_char, padded_value=-1),
                        network.labels_sub_pl_list[0]: list2sparsetensor(labels_phone, padded_value=-1),
                        network.inputs_seq_len_pl_list[0]: inputs_seq_len,
                        network.keep_prob_input_pl_list[0]: 1.0,
                        network.keep_prob_hidden_pl_list[0]: 1.0,
                        network.keep_prob_output_pl_list[0]: 1.0
                    }

                    # Compute loss
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_loss_train.append(loss_train)
                    csv_loss_dev.append(loss_dev)

                    # Change to evaluation mode
                    feed_dict_train[network.keep_prob_input_pl_list[0]] = 1.0
                    feed_dict_train[network.keep_prob_hidden_pl_list[0]] = 1.0
                    feed_dict_train[network.keep_prob_output_pl_list[0]] = 1.0

                    # Compute accuracy & update event files
                    cer_train, per_train, summary_str_train = sess.run(
                        [cer_op, per_op, summary_train],
                        feed_dict=feed_dict_train)
                    cer_dev, per_dev, summary_str_dev = sess.run(
                        [cer_op, per_op,  summary_dev],
                        feed_dict=feed_dict_dev)
                    csv_cer_train.append(cer_train)
                    csv_cer_dev.append(cer_dev)
                    csv_per_train.append(per_train)
                    csv_per_dev.append(per_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print("Step % d: loss = %.3f (%.3f) / cer = %.4f (%.4f) / per = % .4f (%.4f) / lr = %.5f (%.3f min)" %
                          (step + 1, loss_train, loss_dev, cer_train, cer_dev,
                           per_train, per_dev, learning_rate, duration_step / 60))
                    sys.stdout.flush()
                    start_time_step = time.time()

                # Save checkpoint and evaluate model per epoch
                if next_epoch_flag:
                    duration_epoch = time.time() - start_time_epoch
                    print('-----EPOCH:%d (%.3f min)-----' %
                          (epoch, duration_epoch / 60))

                    # Save model (check point)
                    checkpoint_file = join(network.model_dir, 'model.ckpt')
                    save_path = saver.save(
                        sess, checkpoint_file, global_step=epoch)
                    print("Model saved in file: %s" % save_path)

                    if epoch >= 20:
                        start_time_eval = time.time()
                        print('=== Dev Data Evaluation ===')
                        cer_dev_epoch = do_eval_cer(
                            session=sess,
                            decode_op=decode_op_character,
                            network=network,
                            dataset=dev_data,
                            label_type=params['label_type_main'],
                            eval_batch_size=1,
                            is_multitask=True)
                        print('  CER: %f %%' % (cer_dev_epoch * 100))
                        per_dev_epoch = do_eval_per(
                            session=sess,
                            decode_op=decode_op_phone,
                            per_op=per_op,
                            network=network,
                            dataset=dev_data,
                            label_type=params['label_type_sub'],
                            eval_batch_size=1,
                            is_multitask=True)
                        print('  PER: %f %%' % (per_dev_epoch * 100))

                        if cer_dev_epoch < cer_dev_best:
                            cer_dev_best = cer_dev_epoch
                            print('■■■ ↑Best Score (CER)↑ ■■■')

                            print('=== Test Data Evaluation ===')
                            cer_test = do_eval_cer(
                                session=sess,
                                decode_op=decode_op_character,
                                network=network,
                                dataset=test_data,
                                label_type=params['label_type_main'],
                                eval_batch_size=1,
                                is_multitask=True)
                            print('  CER: %f %%' % (cer_test * 100))
                            per_test = do_eval_per(
                                session=sess,
                                decode_op=decode_op_phone,
                                per_op=per_op,
                                network=network,
                                dataset=test_data,
                                label_type=params['label_type_sub'],
                                eval_batch_size=1,
                                is_multitask=True)
                            print('  PER: %f %%' % (per_test * 100))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=epoch,
                            value=cer_dev_epoch)

                        if epoch == params['num_epoch']:
                            break

                    epoch += 1
                    start_time_epoch = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Save train & dev loss, ler
            save_loss(csv_steps, csv_loss_train, csv_loss_dev,
                      save_path=network.model_dir)
            save_ler(csv_steps, csv_cer_train, csv_cer_dev,
                     save_path=network.model_dir)
            save_ler(csv_steps, csv_per_train, csv_per_dev,
                     save_path=network.model_dir)

            # Training was finished correctly
            with open(join(network.model_dir, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path, model_save_path):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type_main'] == 'character':
        params['num_classes_main'] = 28
    elif params['label_type_main'] == 'character_capital_divide':
        params['num_classes_main'] = 72

    if params['label_type_sub'] == 'phone61':
        params['num_classes_sub'] = 61
    elif params['label_type_sub'] == 'phone48':
        params['num_classes_sub'] = 48
    elif params['label_type_sub'] == 'phone39':
        params['num_classes_sub'] = 39

    # Model setting
    model = load(model_type=params['model'])
    network = model(input_size=params['input_size'] * params['num_stack'],
                    num_unit=params['num_unit'],
                    num_layer_main=params['num_layer_main'],
                    num_layer_sub=params['num_layer_sub'],
                    num_classes_main=params['num_classes_main'],
                    num_classes_sub=params['num_classes_sub'],
                    main_task_weight=params['main_task_weight'],
                    parameter_init=params['weight_init'],
                    clip_grad=params['clip_grad'],
                    clip_activation=params['clip_activation'],
                    dropout_ratio_input=params['dropout_input'],
                    dropout_ratio_hidden=params['dropout_hidden'],
                    dropout_ratio_output=params['dropout_output'],
                    num_proj=params['num_proj'],
                    weight_decay=params['weight_decay'])

    network.model_name = params['model']
    network.model_name += '_' + str(params['num_unit'])
    network.model_name += '_main' + str(params['num_layer_main'])
    network.model_name += '_sub' + str(params['num_layer_sub'])
    network.model_name += '_' + params['optimizer']
    network.model_name += '_lr' + str(params['learning_rate'])
    if params['num_proj'] != 0:
        network.model_name += '_proj' + str(params['num_proj'])
    if params['dropout_input'] != 1:
        network.model_name += '_dropi' + str(params['dropout_input'])
    if params['dropout_hidden'] != 1:
        network.model_name += '_droph' + str(params['dropout_hidden'])
    if params['dropout_output'] != 1:
        network.model_name += '_dropo' + str(params['dropout_output'])
    if params['num_stack'] != 1:
        network.model_name += '_stack' + str(params['num_stack'])
    if params['weight_decay'] != 0:
        network.model_name += '_weightdecay' + str(params['weight_decay'])
    network.model_name += '_taskweight' + str(params['main_task_weight'])

    # Set save path
    network.model_dir = mkdir(model_save_path)
    network.model_dir = mkdir_join(network.model_dir, 'ctc')
    network.model_dir = mkdir_join(
        network.model_dir, 'char_' + params['label_type_sub'])
    network.model_dir = mkdir_join(network.model_dir, network.model_name)

    # Reset model directory
    if not isfile(join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('timit_multictc')

    # Save config file
    shutil.copyfile(config_path, join(network.model_dir, 'config.yml'))

    sys.stdout = open(join(network.model_dir, 'train.log'), 'w')
    do_train(network=network, params=params)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError
    main(config_path=args[1], model_save_path=args[2])
