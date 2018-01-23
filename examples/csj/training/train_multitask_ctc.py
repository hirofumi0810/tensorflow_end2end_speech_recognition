#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the multi-task CTC model (CSJ corpus)."""

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
from experiments.csj.data.load_dataset_multitask_ctc import Dataset
from experiments.csj.metrics.ctc import do_eval_cer
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.training.learning_rate_controller.epoch import Controller

from utils.directory import mkdir, mkdir_join
from utils.parameter import count_total_parameters
from utils.csv import save_loss, save_ler
from models.ctc.load_model import load


def do_train(model, params):
    """Run training.
    Args:
        model: model to train
        params: A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(data_type='train',
                         label_type_main=params['label_type_main'],
                         label_type_sub=params['label_type_sub'],
                         train_data_size=params['train_data_size'],
                         batch_size=params['batch_size'],
                         num_stack=params['num_stack'],
                         num_skip=params['num_skip'],
                         sort_utt=True)
    dev_data_step = Dataset(data_type='dev',
                            label_type_main=params['label_type_main'],
                            label_type_sub=params['label_type_sub'],
                            train_data_size=params['train_data_size'],
                            batch_size=params['batch_size'],
                            num_stack=params['num_stack'],
                            num_skip=params['num_skip'],
                            sort_utt=False)
    dev_data_epoch = Dataset(data_type='dev',
                             label_type_main=params['label_type_main'],
                             label_type_sub=params['label_type_sub'],
                             train_data_size=params['train_data_size'],
                             batch_size=params['batch_size'],
                             num_stack=params['num_stack'],
                             num_skip=params['num_skip'],
                             sort_utt=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        model.create_placeholders(gpu_index=0)

        # Add to the graph each operation
        loss_op, logits_main, logits_sub = model.compute_loss(
            model.inputs_pl_list[0],
            model.labels_pl_list[0],
            model.labels_sub_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.keep_prob_input_pl_list[0],
            model.keep_prob_hidden_pl_list[0],
            model.keep_prob_output_pl_list[0])
        train_op = model.train(loss_op,
                               optimizer=params['optimizer'],
                               learning_rate=model.learning_rate_pl_list[0])
        decode_op_main, decode_op_sub = model.decoder(
            logits_main,
            logits_sub,
            model.inputs_seq_len_pl_list[0],
            decode_type='beam_search',
            beam_width=20)
        ler_op_main, ler_op_sub = model.compute_ler(
            decode_op_main, decode_op_sub,
            model.labels_pl_list[0], model.labels_sub_pl_list[0])

        # Define learning rate controller
        lr_controller = Controller(
            learning_rate_init=params['learning_rate'],
            decay_start_epoch=params['decay_start_epoch'],
            decay_rate=params['decay_rate'],
            decay_patient_epoch=1,
            lower_better=True)

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
        csv_ler_main_train, csv_ler_main_dev = [], []
        csv_ler_sub_train, csv_ler_sub_dev = [], []
        # Create a session for running operation on the graph
        with tf.Session() as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                model.save_path, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Make mini-batch generator
            mini_batch_train = train_data.next_batch()
            mini_batch_dev = dev_data_step.next_batch()

            # Train model
            iter_per_epoch = int(train_data.data_num / params['batch_size'])
            train_step = train_data.data_num / params['batch_size']
            if (train_step) != int(train_step):
                iter_per_epoch += 1
            max_steps = iter_per_epoch * params['num_epoch']
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            ler_main_dev_best = 1
            learning_rate = float(params['learning_rate'])
            for step in range(max_steps):

                # Create feed dictionary for next mini batch (train)
                inputs, labels_main, labels_sub, inputs_seq_len, _ = mini_batch_train.__next__()
                feed_dict_train = {
                    model.inputs_pl_list[0]: inputs,
                    model.labels_pl_list[0]: list2sparsetensor(labels_main, padded_value=-1),
                    model.labels_sub_pl_list[0]: list2sparsetensor(labels_sub, padded_value=-1),
                    model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                    model.keep_prob_input_pl_list[0]: model.dropout_ratio_input,
                    model.keep_prob_hidden_pl_list[0]: model.dropout_ratio_hidden,
                    model.keep_prob_output_pl_list[0]: model.dropout_ratio_output,
                    model.learning_rate_pl_list[0]: learning_rate
                }

                # Update parameters
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % 200 == 0:

                    # Create feed dictionary for next mini batch (dev)
                    inputs, labels_main, labels_sub, inputs_seq_len, _ = mini_batch_dev.__next__()
                    feed_dict_dev = {
                        model.inputs_pl_list[0]: inputs,
                        model.labels_pl_list[0]: list2sparsetensor(labels_main, padded_value=-1),
                        model.labels_sub_pl_list[0]: list2sparsetensor(labels_sub, padded_value=-1),
                        model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                        model.keep_prob_input_pl_list[0]: 1.0,
                        model.keep_prob_hidden_pl_list[0]: 1.0,
                        model.keep_prob_output_pl_list[0]: 1.0
                    }

                    # Compute loss
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_loss_train.append(loss_train)
                    csv_loss_dev.append(loss_dev)

                    # Change to evaluation mode
                    feed_dict_train[model.keep_prob_input_pl_list[0]] = 1.0
                    feed_dict_train[model.keep_prob_hidden_pl_list[0]] = 1.0
                    feed_dict_train[model.keep_prob_output_pl_list[0]] = 1.0

                    # Compute accuracy & update event file
                    ler_main_train, ler_sub_train, summary_str_train = sess.run(
                        [ler_op_main, ler_op_sub, summary_train],
                        feed_dict=feed_dict_train)
                    ler_main_dev, ler_sub_dev, summary_str_dev = sess.run(
                        [ler_op_main, ler_op_sub,  summary_dev],
                        feed_dict=feed_dict_dev)
                    csv_ler_main_train.append(ler_main_train)
                    csv_ler_main_dev.append(ler_main_dev)
                    csv_ler_sub_train.append(ler_sub_train)
                    csv_ler_sub_dev.append(ler_sub_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print('Step %d: loss = %.3f (%.3f) / ler_main = %.4f (%.4f) / ler_sub = %.4f (%.4f) (%.3f min)' %
                          (step + 1, loss_train, loss_dev, ler_main_train, ler_main_dev,
                           ler_sub_train, ler_sub_dev, duration_step / 60))
                    sys.stdout.flush()
                    start_time_step = time.time()

                # Save checkpoint and evaluate model per epoch
                if (step + 1) % iter_per_epoch == 0 or (step + 1) == max_steps:
                    duration_epoch = time.time() - start_time_epoch
                    epoch = (step + 1) // iter_per_epoch
                    print('-----EPOCH:%d (%.3f min)-----' %
                          (epoch, duration_epoch / 60))

                    # Save model (check point)
                    checkpoint_file = join(model.save_path, 'model.ckpt')
                    save_path = saver.save(
                        sess, checkpoint_file, global_step=epoch)
                    print("Model saved in file: %s" % save_path)

                    if epoch >= 5:
                        start_time_eval = time.time()
                        print('=== Dev Evaluation ===')
                        ler_main_dev_epoch = do_eval_cer(
                            session=sess,
                            decode_op=decode_op_main,
                            model=model,
                            dataset=dev_data_epoch,
                            label_type=params['label_type_main'],
                            eval_batch_size=params['batch_size'],
                            is_multitask=True,
                            is_main=True)
                        print('  CER (main): %f %%' %
                              (ler_main_dev_epoch * 100))

                        ler_sub_dev_epoch = do_eval_cer(
                            session=sess,
                            decode_op=decode_op_sub,
                            model=model,
                            dataset=dev_data_epoch,
                            label_type=params['label_type_sub'],
                            eval_batch_size=params['batch_size'],
                            is_multitask=True,
                            is_main=False)
                        print('  CER (sub): %f %%' %
                              (ler_sub_dev_epoch * 100))

                        if ler_main_dev_epoch < ler_main_dev_best:
                            ler_main_dev_best = ler_main_dev_epoch
                            print('■■■ ↑Best Score (CER main)↑ ■■■')

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=epoch,
                            value=ler_main_dev_epoch)

                    start_time_epoch = time.time()
                    start_time_step = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Save train & dev loss, ler
            save_loss(csv_steps, csv_loss_train, csv_loss_dev,
                      save_path=model.save_path)
            save_ler(csv_steps, csv_ler_main_train, csv_ler_sub_dev,
                     save_path=model.save_path)
            save_ler(csv_steps, csv_ler_sub_train, csv_ler_sub_dev,
                     save_path=model.save_path)

            # Training was finished correctly
            with open(join(model.save_path, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path, model_save_path):

    # Read a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank label
    if params['label_type_main'] == 'kanji':
        params['num_classes_main'] = 3386
    elif params['label_type_main'] == 'kana':
        params['num_classes_main'] = 147
    else:
        raise TypeError

    if params['label_type_sub'] == 'kana':
        params['num_classes_sub'] = 147
    elif params['label_type_sub'] == 'phone':
        params['num_classes_sub'] = 38
    else:
        TypeError

    # Model setting
    model = load(model_type=params['model'])
    model = model(batch_size=params['batch_size'],
                  input_size=params['input_size'],
                  splice=params['splice'],
                  num_stack=params['num_stack'],
                  num_units=params['num_units'],
                  num_layer_main=params['num_layer_main'],
                  num_layer_sub=params['num_layer_sub'],
                  #    bottleneck_dim=params['bottleneck_dim'],
                  num_classes_main=params['num_classes_main'],
                  num_classes_sub=params['num_classes_sub'],
                  main_task_weight=params['main_task_weight'],
                  parameter_init=params['weight_init'],
                  clip_grad_norm=params['clip_grad_norm'],
                  clip_activation=params['clip_activation'],
                  num_proj=params['num_proj'],
                  weight_decay=params['weight_decay'])

    model.model_name = params['model']
    model.model_name += '_' + str(params['num_units'])
    model.model_name += '_main' + str(params['num_layer_main'])
    model.model_name += '_sub' + str(params['num_layer_sub'])
    model.model_name += '_' + params['optimizer']
    model.model_name += '_lr' + str(params['learning_rate'])
    if params['bottleneck_dim'] != 0:
        model.model_name += '_bottoleneck' + str(params['bottleneck_dim'])
    if params['num_proj'] != 0:
        model.model_name += '_proj' + str(params['num_proj'])
    if params['num_stack'] != 1:
        model.model_name += '_stack' + str(params['num_stack'])
    if params['weight_decay'] != 0:
        model.model_name += '_weightdecay' + str(params['weight_decay'])
    model.model_name += '_taskweight' + str(params['main_task_weight'])
    if params['train_data_size'] == 'large':
        model.model_name += '_large'

    # Set save path
    model.save_path = mkdir(model_save_path)
    model.save_path = mkdir_join(model.save_path, 'ctc')
    model.save_path = mkdir_join(
        model.save_path,
        params['label_type_main'] + '_' + params['label_type_sub'])
    model.save_path = mkdir_join(model.save_path, model.model_name)

    # Reset model directory
    if not isfile(join(model.save_path, 'complete.txt')):
        tf.gfile.DeleteRecursively(model.save_path)
        tf.gfile.MakeDirs(model.save_path)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('csj_multictc_' + params['label_type_main'] + '_' +
                 params['label_type_sub'] + '_' + params['train_data_size'])

    # Save config file
    shutil.copyfile(config_path, join(model.save_path, 'config.yml'))

    sys.stdout = open(join(model.save_path, 'train.log'), 'w')
    do_train(model=model, params=params)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError
    main(config_path=args[1], model_save_path=args[2])
