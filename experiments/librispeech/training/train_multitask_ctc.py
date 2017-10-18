#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the multi-task CTC model with multiple GPUs (Librispeech corpus)."""

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
from experiments.librispeech.data.load_dataset_multitask_ctc import Dataset
from experiments.librispeech.metrics.ctc import do_eval_cer, do_eval_wer
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss, plot_ler
from utils.training.multi_gpu import average_gradients
from utils.directory import mkdir, mkdir_join
from utils.parameter import count_total_parameters
from models.ctc.multitask_ctc import MultitaskCTC


def do_train(model, params, gpu_indices):
    """Run multi-task CTC training.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
        gpu_indices (list): GPU indices
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', train_data_size=params['train_data_size'],
        label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=params['batch_size'], max_epoch=params['num_epoch'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, sort_stop_epoch=params['sort_stop_epoch'],
        num_gpu=len(gpu_indices))
    dev_clean_data = Dataset(
        data_type='dev_clean', train_data_size=params['train_data_size'],
        label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, num_gpu=len(gpu_indices))
    dev_other_data = Dataset(
        data_type='dev_other', train_data_size=params['train_data_size'],
        label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False, num_gpu=len(gpu_indices))
    test_clean_data = Dataset(
        data_type='test_clean', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    test_other_data = Dataset(
        data_type='test_other', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Create a variable to track the global step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Set optimizer
        learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = model._set_optimizer(
            params['optimizer'], learning_rate_pl)

        # Calculate the gradients for each model tower
        total_grads_and_vars, total_losses = [], []
        decode_ops_word, decode_ops_char = [], []
        wer_ops, cer_ops = [], []
        all_devices = ['/gpu:%d' % i_gpu for i_gpu in range(len(gpu_indices))]
        # NOTE: /cpu:0 is prepared for evaluation
        with tf.variable_scope(tf.get_variable_scope()):
            for i_gpu in range(len(all_devices)):
                with tf.device(all_devices[i_gpu]):
                    with tf.name_scope('tower_gpu%d' % i_gpu) as scope:

                        # Define placeholders in each tower
                        model.create_placeholders()

                        # Calculate the total loss for the current tower of the
                        # model. This function constructs the entire model but
                        # shares the variables across all towers.
                        tower_loss, tower_logits_word, tower_logits_char = model.compute_loss(
                            model.inputs_pl_list[i_gpu],
                            model.labels_pl_list[i_gpu],
                            model.labels_sub_pl_list[i_gpu],
                            model.inputs_seq_len_pl_list[i_gpu],
                            model.keep_prob_pl_list[i_gpu],
                            scope)
                        tower_loss = tf.expand_dims(tower_loss, axis=0)
                        total_losses.append(tower_loss)

                        # Reuse variables for the next tower
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this
                        # tower
                        tower_grads_and_vars = optimizer.compute_gradients(
                            tower_loss)

                        # Gradient clipping
                        tower_grads_and_vars = model._clip_gradients(
                            tower_grads_and_vars)

                        # TODO: Optionally add gradient noise

                        # Keep track of the gradients across all towers
                        total_grads_and_vars.append(tower_grads_and_vars)

                        # Add to the graph each operation per tower
                        decode_op_tower_word, decode_op_tower_char = model.decoder(
                            tower_logits_word, tower_logits_char,
                            model.inputs_seq_len_pl_list[i_gpu],
                            beam_width=params['beam_width'])
                        decode_ops_word.append(decode_op_tower_word)
                        decode_ops_char.append(decode_op_tower_char)
                        wer_op_tower, cer_op_tower = model.compute_ler(
                            decode_op_tower_word,
                            decode_op_tower_char,
                            model.labels_pl_list[i_gpu],
                            model.labels_sub_pl_list[i_gpu])
                        wer_op_tower = tf.expand_dims(wer_op_tower, axis=0)
                        wer_ops.append(wer_op_tower)
                        cer_op_tower = tf.expand_dims(cer_op_tower, axis=0)
                        cer_ops.append(cer_op_tower)

        # Aggregate losses, then calculate average loss
        total_losses = tf.concat(axis=0, values=total_losses)
        loss_op = tf.reduce_mean(total_losses, axis=0)
        wer_ops = tf.concat(axis=0, values=wer_ops)
        wer_op = tf.reduce_mean(wer_ops, axis=0)
        cer_ops = tf.concat(axis=0, values=cer_ops)
        cer_op = tf.reduce_mean(cer_ops, axis=0)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers
        average_grads_and_vars = average_gradients(total_grads_and_vars)

        # Apply the gradients to adjust the shared variables.
        train_op = optimizer.apply_gradients(average_grads_and_vars,
                                             global_step=global_step)

        # Define learning rate controller
        lr_controller = Controller(
            learning_rate_init=params['learning_rate'],
            decay_start_epoch=params['decay_start_epoch'],
            decay_rate=params['decay_rate'],
            decay_patient_epoch=params['decay_patient_epoch'],
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
        csv_wer_train, csv_wer_dev = [], []
        csv_cer_train, csv_cer_dev = [], []
        # Create a session for running operation on the graph
        # NOTE: Start running operations on the Graph. allow_soft_placement
        # must be set to True to build towers on GPU, as some of the ops do not
        # have GPU implementations.
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                              log_device_placement=False)) as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                model.save_path, sess.graph)

            # Initialize parameters
            sess.run(init_op)

            # Train model
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            wer_dev_best = 1
            learning_rate = float(params['learning_rate'])
            for step, (data, is_new_epoch) in enumerate(train_data):

                # Create feed dictionary for next mini batch (train)
                inputs, labels_word, labels_char, inputs_seq_len, _ = data
                feed_dict_train = {}
                for i_gpu in range(len(gpu_indices)):
                    feed_dict_train[model.inputs_pl_list[i_gpu]
                                    ] = inputs[i_gpu]
                    feed_dict_train[model.labels_pl_list[i_gpu]] = list2sparsetensor(
                        labels_word[i_gpu], padded_value=train_data.padded_value)
                    feed_dict_train[model.labels_sub_pl_list[i_gpu]] = list2sparsetensor(
                        labels_char[i_gpu], padded_value=train_data.padded_value)
                    feed_dict_train[model.inputs_seq_len_pl_list[i_gpu]
                                    ] = inputs_seq_len[i_gpu]
                    feed_dict_train[model.keep_prob_pl_list[i_gpu]
                                    ] = 1 - float(params['dropout'])
                feed_dict_train[learning_rate_pl] = learning_rate

                # Update parameters
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % int(params['print_step'] / len(gpu_indices)) == 0:

                    # Create feed dictionary for next mini batch (dev)
                    (inputs, labels_word, labels_char,
                     inputs_seq_len, _), _ = dev_other_data.next()
                    feed_dict_dev = {}
                    for i_gpu in range(len(gpu_indices)):
                        feed_dict_dev[model.inputs_pl_list[i_gpu]
                                      ] = inputs[i_gpu]
                        feed_dict_dev[model.labels_pl_list[i_gpu]] = list2sparsetensor(
                            labels_word[i_gpu], padded_value=dev_other_data.padded_value)
                        feed_dict_dev[model.labels_sub_pl_list[i_gpu]] = list2sparsetensor(
                            labels_char[i_gpu], padded_value=dev_other_data.padded_value)
                        feed_dict_dev[model.inputs_seq_len_pl_list[i_gpu]
                                      ] = inputs_seq_len[i_gpu]
                        feed_dict_dev[model.keep_prob_pl_list[i_gpu]] = 1.0

                    # Compute loss
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_loss_train.append(loss_train)
                    csv_loss_dev.append(loss_dev)

                    # Change to evaluation mode
                    for i_gpu in range(len(gpu_indices)):
                        feed_dict_train[model.keep_prob_pl_list[i_gpu]] = 1.0

                    # Compute accuracy & update event files
                    wer_train, cer_train, summary_str_train = sess.run(
                        [wer_op, cer_op, summary_train], feed_dict=feed_dict_train)
                    wer_dev, cer_dev, summary_str_dev = sess.run(
                        [wer_op, cer_op, summary_dev], feed_dict=feed_dict_dev)
                    csv_wer_train.append(wer_train)
                    csv_wer_dev.append(wer_dev)
                    csv_cer_train.append(cer_train)
                    csv_cer_dev.append(cer_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    duration_step = time.time() - start_time_step
                    print("Step %d (epoch: %.3f): loss = %.3f (%.3f) / wer = %.3f (%.3f) / cer = %.3f (%.3f) / lr = %.5f (%.3f min)" %
                          (step + 1, train_data.epoch_detail, loss_train, loss_dev, wer_train, wer_dev,
                           cer_train, cer_dev,
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
                    plot_ler(csv_wer_train, csv_wer_dev, csv_steps,
                             label_type=params['label_type_main'],
                             save_path=model.save_path)
                    plot_ler(csv_cer_train, csv_cer_dev, csv_steps,
                             label_type=params['label_type_sub'],
                             save_path=model.save_path)

                    if train_data.epoch >= params['eval_start_epoch']:
                        start_time_eval = time.time()
                        print('=== Dev Data Evaluation ===')
                        # dev-clean
                        wer_main_dev_clean_epoch = do_eval_wer(
                            session=sess,
                            decode_ops=decode_ops_word,
                            model=model,
                            dataset=dev_clean_data,
                            train_data_size=params['train_data_size'],
                            eval_batch_size=params['batch_size'],
                            is_multitask=True)
                        print('  WER (clean, main): %f %%' %
                              (wer_main_dev_clean_epoch * 100))
                        cer_sub_dev_clean_epoch, wer_sub_dev_clean_epoch = do_eval_cer(
                            session=sess,
                            decode_ops=decode_ops_char,
                            model=model,
                            dataset=dev_clean_data,
                            label_type=params['label_type_sub'],
                            eval_batch_size=params['batch_size'],
                            is_multitask=True)
                        print('  WER (clean, sub): %f %%' %
                              (wer_sub_dev_clean_epoch * 100))
                        print('  CER (clean, sub): %f %%' %
                              (cer_sub_dev_clean_epoch * 100))

                        # dev-other
                        wer_main_dev_other_epoch = do_eval_wer(
                            session=sess,
                            decode_ops=decode_ops_word,
                            model=model,
                            dataset=dev_other_data,
                            train_data_size=params['train_data_size'],
                            eval_batch_size=params['batch_size'],
                            is_multitask=True)
                        print('  WER (other, main): %f %%' %
                              (wer_main_dev_other_epoch * 100))
                        cer_sub_dev_other_epoch, wer_sub_dev_other_epoch = do_eval_cer(
                            session=sess,
                            decode_ops=decode_ops_char,
                            model=model,
                            dataset=dev_other_data,
                            label_type=params['label_type_sub'],
                            eval_batch_size=params['batch_size'],
                            is_multitask=True)
                        print('  WER (other, sub): %f %%' %
                              (wer_sub_dev_other_epoch * 100))
                        print('  CER (other, sub): %f %%' %
                              (cer_sub_dev_other_epoch * 100))

                        if wer_main_dev_other_epoch < wer_dev_best:
                            wer_dev_best = wer_main_dev_other_epoch
                            print('■■■ ↑Best Score (WER)↑ ■■■')

                            # Save model (check point)
                            checkpoint_file = join(
                                model.save_path, 'model.ckpt')
                            save_path = saver.save(
                                sess, checkpoint_file, global_step=train_data.epoch)
                            print("Model saved in file: %s" % save_path)

                            print('=== Test Data Evaluation ===')
                            # test-clean
                            wer_main_test_clean_epoch = do_eval_wer(
                                session=sess,
                                decode_ops=decode_ops_word,
                                model=model,
                                dataset=test_clean_data,
                                train_data_size=params['train_data_size'],
                                is_test=True,
                                eval_batch_size=params['batch_size'],
                                is_multitask=True)
                            print('  WER (clean, main): %f %%' %
                                  (wer_main_test_clean_epoch * 100))
                            cer_sub_test_clean_epoch, wer_sub_test_clean_epoch = do_eval_cer(
                                session=sess,
                                decode_ops=decode_ops_char,
                                model=model,
                                dataset=test_clean_data,
                                label_type=params['label_type_sub'],
                                is_test=True,
                                eval_batch_size=params['batch_size'],
                                is_multitask=True)
                            print('  WER (clean, sub): %f %%' %
                                  (wer_sub_test_clean_epoch * 100))
                            print('  CER (clean, sub): %f %%' %
                                  (cer_sub_test_clean_epoch * 100))

                            # test-other
                            wer_main_test_other_epoch = do_eval_wer(
                                session=sess,
                                decode_ops=decode_ops_word,
                                model=model,
                                dataset=test_other_data,
                                train_data_size=params['train_data_size'],
                                is_test=True,
                                eval_batch_size=params['batch_size'],
                                is_multitask=True)
                            print('  WER (other, main): %f %%' %
                                  (wer_main_test_other_epoch * 100))
                            cer_sub_test_other_epoch, wer_sub_test_other_epoch = do_eval_cer(
                                session=sess,
                                decode_ops=decode_ops_char,
                                model=model,
                                dataset=test_other_data,
                                label_type=params['label_type_sub'],
                                is_test=True,
                                eval_batch_size=params['batch_size'],
                                is_multitask=True)
                            print('  WER (other, sub): %f %%' %
                                  (wer_sub_test_other_epoch * 100))
                            print('  CER (other, sub): %f %%' %
                                  (cer_sub_test_other_epoch * 100))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=train_data.epoch,
                            value=wer_main_dev_other_epoch)

                    start_time_epoch = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Training was finished correctly
            with open(join(model.save_path, 'complete.txt'), 'w') as f:
                f.write('')


def main(config_path, model_save_path, gpu_indices):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type_main'] == 'word_freq10':
        if params['train_data_size'] == 'train100h':
            params['num_classes_main'] = 7213
        elif params['train_data_size'] == 'train460h':
            params['num_classes_main'] = 18641
        elif params['train_data_size'] == 'train960h':
            params['num_classes_main'] = 26642
    if params['label_type_sub'] == 'character':
        params['num_classes_sub'] = 28
    elif params['label_type_sub'] == 'character_capital_divide':
        if params['train_data_size'] == 'train100h':
            params['num_classes_sub'] = 72
        elif params['train_data_size'] == 'train460h':
            params['num_classes_sub'] = 77
        elif params['train_data_size'] == 'train960h':
            params['num_classes_sub'] = 77

    # Model setting
    model = MultitaskCTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        splice=params['splice'],
        num_units=params['num_units'],
        num_layers_main=params['num_layers_main'],
        num_layers_sub=params['num_layers_sub'],
        num_classes_main=params['num_classes_main'],
        num_classes_sub=params['num_classes_sub'],
        main_task_weight=params['main_task_weight'],
        lstm_impl=params['lstm_impl'],
        use_peephole=params['use_peephole'],
        parameter_init=params['weight_init'],
        clip_grad_norm=params['clip_grad_norm'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    # Set process name
    setproctitle(
        'libri_' + model.name + '_' + params['train_data_size'] + '_' +
        params['label_type_main'] + '_' + params['label_type_sub'])

    model.name += '_' + str(params['num_units'])
    model.name += '_main' + str(params['num_layers_main'])
    model.name += '_sub' + str(params['num_layers_sub'])
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
    model.name += '_main' + str(params['main_task_weight'])
    if len(gpu_indices) >= 2:
        model.name += '_gpu' + str(len(gpu_indices))

    # Set save path
    model.save_path = mkdir_join(
        model_save_path, 'ctc',
        params['label_type_main'] + '_' + params['label_type_sub'],
        params['train_data_size'], model.name)

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
    do_train(model=model, params=params, gpu_indices=gpu_indices)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3 and len(args) != 4:
        raise ValueError
    main(config_path=args[1], model_save_path=args[2],
         gpu_indices=list(map(int, args[3].split(','))))
