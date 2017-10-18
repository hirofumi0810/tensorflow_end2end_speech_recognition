#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the Attention-based model (TIMIT corpus)."""

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
from experiments.timit.data.load_dataset_attention import Dataset
from experiments.timit.metrics.attention import do_eval_per, do_eval_cer
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss, plot_ler
from utils.directory import mkdir_join, mkdir
from utils.parameter import count_total_parameters
from models.attention.attention_seq2seq import AttentionSeq2Seq


def do_train(model, params):
    """Run training. If target labels are phone, the model is evaluated by PER
    with 39 phones.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
    """
    map_file_path_train = '../metrics/mapping_files/' + \
        params['label_type'] + '.txt'
    map_file_path_eval = '../metrics/mapping_files/' + \
        params['label_type'] + '.txt'

    # Load dataset
    train_data = Dataset(
        data_type='train', label_type=params['label_type'],
        batch_size=params['batch_size'], map_file_path=map_file_path_train,
        max_epoch=params['num_epoch'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    # sort_stop_epoch=params['sort_stop_epoch']
    # TODO: check sort_utt
    dev_data = Dataset(
        data_type='dev', label_type=params['label_type'],
        batch_size=params['batch_size'], map_file_path=map_file_path_train,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    if params['label_type'] in ['character', 'character_capital_divide']:
        test_data = Dataset(
            data_type='test', label_type=params['label_type'],
            batch_size=1, map_file_path=map_file_path_eval,
            splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False)
    else:
        test_data = Dataset(
            data_type='test', label_type='phone39',
            batch_size=1, map_file_path=map_file_path_eval,
            splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        model.create_placeholders()
        learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

        # Add to the graph each operation (including model definition)
        loss_op, logits, decoder_outputs_train, decoder_outputs_infer = model.compute_loss(
            model.inputs_pl_list[0],
            model.labels_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.labels_seq_len_pl_list[0],
            model.keep_prob_encoder_pl_list[0],
            model.keep_prob_decoder_pl_list[0],
            model.keep_prob_embedding_pl_list[0])
        train_op = model.train(loss_op,
                               optimizer=params['optimizer'],
                               learning_rate=learning_rate_pl)
        _, decode_op_infer = model.decode(
            decoder_outputs_train,
            decoder_outputs_infer)
        ler_op = model.compute_ler(model.labels_st_true_pl,
                                   model.labels_st_pred_pl)

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

        # Count total param
        parameters_dict, total_parameters = count_total_parameters(
            tf.trainable_variables())
        for parameter_name in sorted(parameters_dict.keys()):
            print("%s %d" % (parameter_name, parameters_dict[parameter_name]))
        print("Total %d variables, %s M param" %
              (len(parameters_dict.keys()),
               "{:,}".format(total_parameters / 1000000)))

        csv_steps, csv_loss_train, csv_loss_dev = [], [], []
        csv_ler_train, csv_ler_dev = [], []
        # Create a session for running operation on the graph
        with tf.Session() as sess:

            # Instantiate a SummaryWriter to output summaries and the graph
            summary_writer = tf.summary.FileWriter(
                model.save_path, sess.graph)

            # Initialize param
            sess.run(init_op)

            # Train model
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            ler_dev_best = 1
            learning_rate = float(params['learning_rate'])
            for step, (data, is_new_epoch) in enumerate(train_data):

                # Create feed dictionary for next mini batch (train)
                inputs, labels_train, inputs_seq_len, labels_seq_len, _ = data
                feed_dict_train = {
                    model.inputs_pl_list[0]: inputs[0],
                    model.labels_pl_list[0]: labels_train[0],
                    model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
                    model.labels_seq_len_pl_list[0]: labels_seq_len[0],
                    model.keep_prob_encoder_pl_list[0]: 1 - float(params['dropout_encoder']),
                    model.keep_prob_decoder_pl_list[0]: 1 - float(params['dropout_decoder']),
                    model.keep_prob_embedding_pl_list[0]: 1 - float(params['dropout_embedding']),
                    learning_rate_pl: learning_rate
                }

                # Update parameters
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % params['print_step'] == 0:

                    # Create feed dictionary for next mini batch (dev)
                    (inputs, labels_dev, inputs_seq_len,
                     labels_seq_len, _), _ = dev_data.next()
                    feed_dict_dev = {
                        model.inputs_pl_list[0]: inputs[0],
                        model.labels_pl_list[0]: labels_dev[0],
                        model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
                        model.labels_seq_len_pl_list[0]: labels_seq_len[0],
                        model.keep_prob_encoder_pl_list[0]: 1.0,
                        model.keep_prob_decoder_pl_list[0]: 1.0,
                        model.keep_prob_embedding_pl_list[0]: 1.0
                    }

                    # Compute loss
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_loss_train.append(loss_train)
                    csv_loss_dev.append(loss_dev)

                    # Change to evaluation mode
                    feed_dict_train[model.keep_prob_encoder_pl_list[0]] = 1.0
                    feed_dict_train[model.keep_prob_decoder_pl_list[0]] = 1.0
                    feed_dict_train[model.keep_prob_embedding_pl_list[0]] = 1.0

                    # Predict class ids & update even files
                    predicted_ids_train, summary_str_train = sess.run(
                        [decode_op_infer, summary_train], feed_dict=feed_dict_train)
                    predicted_ids_dev, summary_str_dev = sess.run(
                        [decode_op_infer, summary_dev], feed_dict=feed_dict_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    # Convert to sparsetensor to compute LER
                    feed_dict_ler_train = {
                        model.labels_st_true_pl: list2sparsetensor(
                            labels_train[0], padded_value=train_data.padded_value),
                        model.labels_st_pred_pl: list2sparsetensor(
                            predicted_ids_train, padded_value=train_data.padded_value)
                    }
                    feed_dict_ler_dev = {
                        model.labels_st_true_pl: list2sparsetensor(
                            labels_dev[0], padded_value=dev_data.padded_value),
                        model.labels_st_pred_pl: list2sparsetensor(
                            predicted_ids_dev, padded_value=dev_data.padded_value)
                    }

                    # Compute accuracy
                    ler_train = sess.run(ler_op, feed_dict=feed_dict_ler_train)
                    ler_dev = sess.run(ler_op, feed_dict=feed_dict_ler_dev)
                    csv_ler_train.append(ler_train)
                    csv_ler_dev.append(ler_dev)

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

                    checkpoint_file = join(
                        model.save_path, 'model.ckpt')
                    save_path = saver.save(
                        sess, checkpoint_file, global_step=train_data.epoch)
                    print("Model saved in file: %s" % save_path)

                    # Save fugure of loss & ler
                    plot_loss(csv_loss_train, csv_loss_dev, csv_steps,
                              save_path=model.save_path)
                    plot_ler(csv_ler_train, csv_ler_dev, csv_steps,
                             label_type=params['label_type'],
                             save_path=model.save_path)

                    # if train_data.epoch >= params['eval_start_epoch']:
                    if train_data.epoch >= 5:
                        start_time_eval = time.time()
                        if 'char' in params['label_type']:
                            print('=== Dev Data Evaluation ===')
                            ler_dev_epoch, wer_dev_epoch = do_eval_cer(
                                session=sess,
                                decode_op=decode_op_infer,
                                model=model,
                                dataset=dev_data,
                                label_type=params['label_type'],
                                eval_batch_size=1)
                            print('  CER: %f %%' % (ler_dev_epoch * 100))
                            print('  WER: %f %%' % (wer_dev_epoch * 100))

                            if ler_dev_epoch < ler_dev_best:
                                ler_dev_best = ler_dev_epoch
                                print('■■■ ↑Best Score (CER)↑ ■■■')

                                # Save model only when best accuracy is
                                # obtained (check point)
                                checkpoint_file = join(
                                    model.save_path, 'model.ckpt')
                                save_path = saver.save(
                                    sess, checkpoint_file, global_step=train_data.epoch)
                                print("Model saved in file: %s" % save_path)

                                print('=== Test Data Evaluation ===')
                                ler_test, wer_test = do_eval_cer(
                                    session=sess,
                                    decode_op=decode_op_infer,
                                    model=model,
                                    dataset=test_data,
                                    label_type=params['label_type'],
                                    eval_batch_size=1)
                                print('  CER: %f %%' % (ler_test * 100))
                                print('  WER: %f %%' % (wer_test * 100))

                        else:
                            print('=== Dev Data Evaluation ===')
                            ler_dev_epoch = do_eval_per(
                                session=sess,
                                decode_op=decode_op_infer,
                                per_op=ler_op,
                                model=model,
                                dataset=dev_data,
                                label_type=params['label_type'],
                                eval_batch_size=1)
                            print('  PER: %f %%' % (ler_dev_epoch * 100))

                            if ler_dev_epoch < ler_dev_best:
                                ler_dev_best = ler_dev_epoch
                                print('■■■ ↑Best Score (PER)↑ ■■■')

                                # Save model only when best accuracy is
                                # obtained (check point)
                                # checkpoint_file = join(
                                #     model.save_path, 'model.ckpt')
                                # save_path = saver.save(
                                #     sess, checkpoint_file, global_step=train_data.epoch)
                                # print("Model saved in file: %s" % save_path)

                                print('=== Test Data Evaluation ===')
                                ler_test = do_eval_per(
                                    session=sess,
                                    decode_op=decode_op_infer,
                                    per_op=ler_op,
                                    model=model,
                                    dataset=test_data,
                                    label_type=params['label_type'],
                                    eval_batch_size=1)
                                print('  PER: %f %%' % (ler_test * 100))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=train_data.epoch,
                            value=ler_dev_epoch)

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

    # Except for a <SOS> and <EOS> class
    if params['label_type'] == 'phone61':
        params['num_classes'] = 61
    elif params['label_type'] == 'phone48':
        params['num_classes'] = 48
    elif params['label_type'] == 'phone39':
        params['num_classes'] = 39
    elif params['label_type'] == 'character':
        params['num_classes'] = 28
    elif params['label_type'] == 'character_capital_divide':
        params['num_classes'] = 72

    # Model setting
    model = AttentionSeq2Seq(
        input_size=params['input_size'],
        encoder_type=params['encoder_type'],
        encoder_num_units=params['encoder_num_units'],
        encoder_num_layers=params['encoder_num_layers'],
        encoder_num_proj=params['encoder_num_proj'],
        attention_type=params['attention_type'],
        attention_dim=params['attention_dim'],
        decoder_type=params['decoder_type'],
        decoder_num_units=params['decoder_num_units'],
        decoder_num_layers=params['decoder_num_layers'],
        embedding_dim=params['embedding_dim'],
        num_classes=params['num_classes'],
        sos_index=params['num_classes'],
        eos_index=params['num_classes'] + 1,
        max_decode_length=params['max_decode_length'],
        lstm_impl='LSTMBlockCell',
        use_peephole=params['use_peephole'],
        parameter_init=params['weight_init'],
        clip_grad_norm=params['clip_grad_norm'],
        clip_activation_encoder=params['clip_activation_encoder'],
        clip_activation_decoder=params['clip_activation_decoder'],
        weight_decay=params['weight_decay'],
        time_major=True,
        sharpening_factor=params['sharpening_factor'],
        logits_temperature=params['logits_temperature'])

    # Set process name
    setproctitle('tf_timit_' + model.name + '_' +
                 params['label_type'] + '_' + params['attention_type'])

    model.name += '_en' + str(params['encoder_num_units'])
    model.name += '_' + str(params['encoder_num_layers'])
    model.name += '_att' + str(params['attention_dim'])
    model.name += '_de' + str(params['decoder_num_units'])
    model.name += '_' + str(params['decoder_num_layers'])
    model.name += '_' + params['optimizer']
    model.name += '_lr' + str(params['learning_rate'])
    model.name += '_' + params['attention_type']
    if params['dropout_encoder'] != 0:
        model.name += '_dropen' + str(params['dropout_encoder'])
    if params['dropout_decoder'] != 0:
        model.name += '_dropde' + str(params['dropout_decoder'])
    if params['dropout_embedding'] != 0:
        model.name += '_dropem' + str(params['dropout_embedding'])
    if params['num_stack'] != 1:
        model.name += '_stack' + str(params['num_stack'])
    if params['weight_decay'] != 0:
        model.name += 'wd' + str(params['weight_decay'])
    if params['sharpening_factor'] != 1:
        model.name += '_sharp' + str(params['sharpening_factor'])
    if params['logits_temperature'] != 1:
        model.name += '_temp' + str(params['logits_temperature'])

    # Set save path
    model.save_path = mkdir_join(
        model_save_path, 'attention', params['label_type'], model.name)

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
