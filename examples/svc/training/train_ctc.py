#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the CTC model (SVC corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, abspath
import sys
import time
import tensorflow as tf
from setproctitle import setproctitle
import yaml
import shutil

sys.path.append(abspath('../../../'))
from experiments.svc.data.load_dataset_ctc import Dataset
from experiments.svc.metrics.ctc import do_eval_fmeasure, do_eval_fmeasure_time
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss, plot_ler
from utils.directory import mkdir_join, mkdir
from utils.parameter import count_total_parameters
from models.ctc.ctc import CTC


def do_train(model, params):
    """Run training.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', label_type=params['label_type'],
        batch_size=params['batch_size'], max_epoch=params['num_epoch'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=True)
    dev_data = Dataset(
        data_type='dev', label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)
    test_data = Dataset(
        data_type='dev', label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        model.create_placeholders()
        learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

        # Add to the graph each operation (including model definition)
        loss_op, logits = model.compute_loss(
            model.inputs_pl_list[0],
            model.labels_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.keep_prob_pl_list[0])
        train_op = model.train(
            loss_op,
            optimizer=params['optimizer'],
            learning_rate=learning_rate_pl)
        decode_op = model.decoder(logits,
                                  model.inputs_seq_len_pl_list[0],
                                  beam_width=params['beam_width'])
        ler_op = model.compute_ler(decode_op, model.labels_pl_list[0])
        posteriors_op = model.posteriors(logits, blank_prior=1)

        # Define learning rate controller
        lr_controller = Controller(
            learning_rate_init=params['learning_rate'],
            decay_start_epoch=params['decay_start_epoch'],
            decay_rate=params['decay_rate'],
            decay_patient_epoch=params['decay_patient_epoch'],
            lower_better=False)

        # Build the summary tensor based on the TensorFlow collection of
        # summaries
        summary_train = tf.summary.merge(model.summaries_train)
        summary_dev = tf.summary.merge(model.summaries_dev)

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
        with tf.Session() as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                model.save_path, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Train model
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            fmean_dev_best = 0
            fmean_time_dev_best = 0
            learning_rate = float(params['learning_rate'])
            for step, (data, is_new_epoch) in enumerate(train_data):

                # Create feed dictionary for next mini batch (train)
                inputs, labels, inputs_seq_len, _ = data
                feed_dict_train = {
                    model.inputs_pl_list[0]: inputs[0],
                    model.labels_pl_list[0]: list2sparsetensor(
                        labels[0], padded_value=train_data.padded_value),
                    model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
                    model.keep_prob_pl_list[0]: 1 - float(params['dropout']),
                    learning_rate_pl: learning_rate
                }

                # Update parameters
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % params['print_step'] == 0:

                    # Create feed dictionary for next mini batch (dev)
                    (inputs, labels, inputs_seq_len, _), _ = dev_data.next()
                    feed_dict_dev = {
                        model.inputs_pl_list[0]: inputs[0],
                        model.labels_pl_list[0]: list2sparsetensor(
                            labels[0], padded_value=dev_data.padded_value),
                        model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
                        model.keep_prob_pl_list[0]: 1.0
                    }

                    # Compute loss
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_loss_train.append(loss_train)
                    csv_loss_dev.append(loss_dev)

                    # Change to evaluation mode
                    feed_dict_train[model.keep_prob_pl_list[0]] = 1.0

                    # Compute accuracy & update event files
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
                    print("Step %d (epoch: %.3f): loss = %.3f (%.3f) / ler = %.3f (%.3f) / lr = %.5f (%.3f min)" %
                          (step + 1, train_data.epoch_detail, loss_train, loss_dev, ler_train, ler_dev,
                           learning_rate, duration_step / 60))
                    sys.stdout.flush()
                    start_time_step = time.time()

                # Save checkpoint and evaluate model per epoch
                if is_new_epoch:
                    duration_epoch = time.time() - start_time_epoch
                    print('-----EPOCH:%d (%.3f min)-----' %
                          (train_data.epoch, duration_epoch / 60))

                    # Save fugure of loss & ler
                    plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                              save_path=model.save_path)
                    plot_ler(csv_ler_train, csv_ler_dev, csv_steps,
                             label_type=params['label_type'],
                             save_path=model.save_path)

                    if train_data.epoch >= params['eval_start_epoch']:
                        start_time_eval = time.time()
                        print('=== Dev Data Evaluation ===')
                        fmean_dev_epoch, df_acc = do_eval_fmeasure(
                            session=sess,
                            decode_op=decode_op,
                            model=model,
                            dataset=dev_data,
                            eval_batch_size=params['batch_size'])
                        print(df_acc)
                        print('  F-measure: %f %%' % (fmean_dev_epoch))

                        if fmean_dev_epoch > fmean_dev_best:
                            fmean_dev_best = fmean_dev_epoch
                            print('■■■ ↑Best Score (F-measure)↑ ■■■')

                            # Save model only when best accuracy is
                            # obtained (check point)
                            checkpoint_file = join(
                                model.save_path, 'model.ckpt')
                            save_path = saver.save(
                                sess, checkpoint_file, global_step=train_data.epoch)
                            print("Model saved in file: %s" % save_path)

                            print('=== Test Data Evaluation ===')
                            fmean_test_epoch, df_acc = do_eval_fmeasure(
                                session=sess,
                                decode_op=decode_op,
                                model=model,
                                dataset=test_data,
                                eval_batch_size=params['batch_size'])
                            print(df_acc)
                            print('  F-measure: %f %%' % (fmean_test_epoch))

                        # fmean_time_dev_epoch, df_acc = do_eval_fmeasure_time(
                        #     session=sess,
                        #     decode_op=decode_op,
                        #     posteriors_op=posteriors_op,
                        #     model=model,
                        #     dataset=dev_data,
                        #     eval_batch_size=params['batch_size'])
                        # print(df_acc)
                        # print('  Time F-measure: %f %%' %
                        #       (fmean_time_dev_epoch))

                        # if fmean_time_dev_best < fmean_time_dev_epoch:
                        #     fmean_time_dev_best = fmean_time_dev_epoch
                        #     print('■■■ ↑Best Score (Time F-measure)↑ ■■■')

                        # fmean_time_test_epoch, df_acc = do_eval_fmeasure_time(
                        #     session=sess,
                        #     decode_op=decode_op,
                        #     posteriors_op=posteriors_op,
                        #     model=model,
                        #     dataset=test_data,
                        #     eval_batch_size=params['batch_size'])
                        # print(df_acc)
                        # print('  Time F-measure: %f %%' %
                        #       (fmean_time_test_epoch))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=train_data.epoch,
                            value=fmean_dev_epoch)

                    start_time_epoch = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Training was finished correctly
            with open(join(model.save_path, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path, model_save_path):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['feature'] == 'fbank':
        input_size = 123
    elif params['feature'] == 'is13':
        input_size = 141

    if params['label_type'] in ['original', 'phone3']:
        params['num_classes'] = 3
    elif params['label_type'] == 'phone4':
        params['num_classes'] = 4
    elif params['label_type'] == 'phone43':
        params['num_classes'] = 43

    # Model setting
    model = CTC(encoder_type=params['encoder_type'],
                input_size=input_size * params['num_stack'],
                splice=params['splice'],
                num_units=params['num_units'],
                num_layers=params['num_layers'],
                num_classes=params['num_classes'],
                lstm_impl=params['lstm_impl'],
                use_peephole=params['use_peephole'],
                parameter_init=params['weight_init'],
                clip_grad_norm=params['clip_grad_norm'],
                clip_activation=params['clip_activation'],
                num_proj=params['num_proj'],
                weight_decay=params['weight_decay'])

    # Set process name
    setproctitle('tf_svc_' + model.name + '_' + params['label_type'])

    model.name += '_' + str(params['num_units'])
    model.name += '_' + str(params['num_layers'])
    model.name += '_' + params['optimizer']
    model.name += '_lr' + str(params['learning_rate'])
    if params['num_proj'] != 0:
        model.name += '_proj' + str(params['num_proj'])
    if params['dropout'] != 0:
        model.name += '_drop' + str(params['dropout'])
    if params['num_stack'] != 1:
        model.name += '_stack' + str(params['num_stack'])
    if params['weight_decay'] != 0:
        model.name += '_wd' + str(params['weight_decay'])

    # Set save path
    model.save_path = mkdir_join(
        model_save_path, 'ctc', params['label_type'], model.name)

    # Reset model directory
    model_index = 0
    new_model_path = model.save_path
    while True:
        if isfile(join(new_model_path, 'complete.txt')):
            # Training of the first model have been finished
            model_index += 1
            new_model_path = model.save_path + '_' + str(model_index)
        elif isfile(join(new_model_path, 'config.yml')):
            # Training of the first model have not been finished yet
            model_index += 1
            new_model_path = model.save_path + '_' + str(model_index)
        else:
            break
    model.save_path = mkdir(new_model_path)

    # Save config file
    shutil.copyfile(config_path, join(model.save_path, 'config.yml'))

    sys.stdout = open(join(model.save_path, 'train.log'), 'w')
    # TODO(hirofumi): change to logger
    do_train(model=model, params=params)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError('Length of args should be 3.')
    main(config_path=args[1], model_save_path=args[2])
