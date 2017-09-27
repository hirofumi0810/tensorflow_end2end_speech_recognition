#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the Joint CTC-Attention model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, isfile, isdir, abspath
import sys
import time
import tensorflow as tf
from setproctitle import setproctitle
import yaml
import shutil

sys.path.append(abspath('../../../'))
from experiments.timit.data.load_dataset_joint_ctc_attention import Dataset
from experiments.timit.metrics.attention import do_eval_per, do_eval_cer
from utils.io.labels.sparsetensor import list2sparsetensor
from utils.training.learning_rate_controller import Controller
from utils.training.plot import plot_loss, plot_ler
from utils.directory import mkdir_join, mkdir
from utils.parameter import count_total_parameters
from models.attention.joint_ctc_attention import JointCTCAttention


def do_train(model, params):
    """Run training. If target labels are phone, the model is evaluated by PER
    with 39 phones.
    Args:
        model: the model to train
        params (dict): A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', label_type=params['label_type'],
        batch_size=params['batch_size'], eos_index=params['eos_index'],
        max_epoch=params['num_epoch'], splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True)
    dev_data = Dataset(
        data_type='dev', label_type=params['label_type'],
        batch_size=params['batch_size'], eos_index=params['eos_index'],
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=False)
    if 'char' in params['label_type']:
        test_data = Dataset(
            data_type='test', label_type=params['label_type'],
            batch_size=1, eos_index=params['eos_index'],
            splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False)
    else:
        test_data = Dataset(
            data_type='test', label_type='phone39',
            batch_size=1, eos_index=params['eos_index'],
            splice=params['splice'],
            num_stack=params['num_stack'], num_skip=params['num_skip'],
            sort_utt=False)
    # TODO(hirofumi): add frame_stacking and splice

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        model.create_placeholders()
        learning_rate_pl = tf.placeholder(tf.float32, name='learning_rate')

        # Add to the graph each operation (including model definition)
        loss_op, att_logits, ctc_logits, decoder_outputs_train, decoder_outputs_infer = model.compute_loss(
            model.inputs_pl_list[0],
            model.att_labels_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.att_labels_seq_len_pl_list[0],
            model.ctc_labels_pl_list[0],
            model.keep_prob_input_pl_list[0],
            model.keep_prob_hidden_pl_list[0],
            model.keep_prob_output_pl_list[0])
        train_op = model.train(
            loss_op,
            optimizer=params['optimizer'],
            learning_rate=learning_rate_pl)
        _, decode_op_infer = model.decoder(
            decoder_outputs_train,
            decoder_outputs_infer,
            decode_type='greedy',
            beam_width=20)
        ler_op = model.compute_ler(
            model.att_labels_st_true_pl, model.att_labels_st_pred_pl)

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
                inputs, att_labels_train, ctc_labels, inputs_seq_len, att_labels_seq_len, _ = data
                feed_dict_train = {
                    model.inputs_pl_list[0]: inputs,
                    model.att_labels_pl_list[0]: att_labels_train,
                    model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                    model.att_labels_seq_len_pl_list[0]: att_labels_seq_len,
                    model.ctc_labels_pl_list[0]: list2sparsetensor(
                        ctc_labels, padded_value=train_data.ctc_padded_value),
                    model.keep_prob_input_pl_list[0]: params['dropout_input'],
                    model.keep_prob_hidden_pl_list[0]: params['dropout_hidden'],
                    model.keep_prob_output_pl_list[0]: params['dropout_output'],
                    learning_rate_pl: learning_rate
                }

                # Update param
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % params['print_step'] == 0:

                    # Create feed dictionary for next mini batch (dev)
                    (inputs, att_labels_dev, ctc_labels, inputs_seq_len,
                     att_labels_seq_len, _), _ = dev_data().next()
                    feed_dict_dev = {
                        model.inputs_pl_list[0]: inputs,
                        model.att_labels_pl_list[0]: att_labels_dev,
                        model.inputs_seq_len_pl_list[0]: inputs_seq_len,
                        model.att_labels_seq_len_pl_list[0]: att_labels_seq_len,
                        model.ctc_labels_pl_list[0]: list2sparsetensor(
                            ctc_labels, padded_value=dev_data.ctc_padded_value),
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

                    # Predict class ids & update event files
                    predicted_ids_train, summary_str_train = sess.run(
                        [decode_op_infer, summary_train], feed_dict=feed_dict_train)
                    predicted_ids_dev, summary_str_dev = sess.run(
                        [decode_op_infer, summary_dev], feed_dict=feed_dict_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    # Convert to sparsetensor to compute LER
                    feed_dict_ler_train = {
                        model.att_labels_true_st: list2sparsetensor(
                            att_labels_train, padded_value=params['eos_index']),
                        model.att_labels_st_pred_pl: list2sparsetensor(
                            predicted_ids_train, padded_value=params['eos_index'])
                    }
                    feed_dict_ler_dev = {
                        model.att_labels_true_st: list2sparsetensor(
                            att_labels_dev, padded_value=params['eos_index']),
                        model.att_labels_st_pred_pl: list2sparsetensor(
                            predicted_ids_dev, padded_value=params['eos_index'])
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
                    # sys.stdout.flush()
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
                        if 'char' in params['label_type']:
                            print('=== Dev Data Evaluation ===')
                            ler_dev_epoch = do_eval_cer(
                                session=sess,
                                decode_op=decode_op_infer,
                                model=model,
                                dataset=dev_data,
                                eval_batch_size=1)
                            print('  CER: %f %%' % (ler_dev_epoch * 100))

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
                                ler_test = do_eval_cer(
                                    session=sess,
                                    decode_op=decode_op_infer,
                                    model=model,
                                    dataset=test_data,
                                    eval_batch_size=1)
                                print('  CER: %f %%' %
                                      (ler_test * 100))

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
                                checkpoint_file = join(
                                    model.save_path, 'model.ckpt')
                                save_path = saver.save(
                                    sess, checkpoint_file, global_step=train_data.epoch)
                                print("Model saved in file: %s" % save_path)

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

    params['sos_index'] = 0
    params['eos_index'] = 1
    if params['label_type'] == 'phone61':
        params['att_num_classes'] = 63
        params['ctc_num_classes'] = 61
    elif params['label_type'] == 'phone48':
        params['att_num_classes'] = 50
        params['ctc_num_classes'] = 48
    elif params['label_type'] == 'phone39':
        params['att_num_classes'] = 41
        params['ctc_num_classes'] = 39
    elif params['label_type'] == 'character':
        params['att_num_classes'] = 30
        params['ctc_num_classes'] = 28

    # Model setting
    # AttentionModel = load(model_type=config['model_name'])
    model = JointCTCAttention(
        input_size=params['input_size'],
        encoder_num_unit=params['encoder_num_unit'],
        encoder_num_layer=params['encoder_num_layer'],
        attention_dim=params['attention_dim'],
        attention_type=params['attention_type'],
        decoder_num_unit=params['decoder_num_unit'],
        decoder_num_layer=params['decoder_num_layer'],
        embedding_dim=params['embedding_dim'],
        att_num_classes=params['att_num_classes'],
        ctc_num_classes=params['ctc_num_classes'],
        att_task_weight=params['att_task_weight'],
        sos_index=params['sos_index'],
        eos_index=params['eos_index'],
        max_decode_length=params['max_decode_length'],
        # attention_smoothing=params['attention_smoothing'],
        attention_weights_tempareture=params['attention_weights_tempareture'],
        logits_tempareture=params['logits_tempareture'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation_encoder=params['clip_activation_encoder'],
        clip_activation_decoder=params['clip_activation_decoder'],
        weight_decay=params['weight_decay'])

    # Set process name
    setproctitle('timit_' + model.name + '_' + params['label_type'])

    model.name = params['model']
    model.name += '_encoder' + str(params['encoder_num_unit'])
    model.name += '_' + str(params['encoder_num_layer'])
    model.name += '_attdim' + str(params['attention_dim'])
    model.name += '_decoder' + str(params['decoder_num_unit'])
    model.name += '_' + str(params['decoder_num_layer'])
    model.name += '_' + params['optimizer']
    model.name += '_lr' + str(params['learning_rate'])
    model.name += '_' + params['attention_type']
    # if bool(params['attention_smoothing']):
    #     model.name += '_smoothing'
    if params['attention_weights_tempareture'] != 1:
        model.name += '_sharpening' + \
            str(params['attention_weights_tempareture'])
    if params['weight_decay'] != 0:
        model.name += '_weightdecay' + str(params['weight_decay'])

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
            # tf.gfile.DeleteRecursively(new_model_path)
            # tf.gfile.MakeDirs(new_model_path)
            # break
            model_index += 1
            new_model_path = model.save_path + '_' + str(model_index)
        else:
            break
    model.save_path = mkdir(new_model_path)

    # Save config file
    shutil.copyfile(config_path, join(model.save_path, 'config.yml'))

    # sys.stdout = open(join(model.save_path, 'train.log'), 'w')
    # TODO(hirofumi): change to logger
    do_train(model=model, params=params)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError('Length of args should be 3.')
    main(config_path=args[1], model_save_path=args[2])
