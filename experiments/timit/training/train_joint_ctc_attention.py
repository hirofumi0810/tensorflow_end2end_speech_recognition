#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Train the Joint CTC-Attention model (TIMIT corpus)."""

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
from experiments.timit.data.load_dataset_joint_ctc_attention import Dataset
from experiments.timit.metrics.joint_ctc_attention import do_eval_per, do_eval_cer
from experiments.utils.data.sparsetensor import list2sparsetensor
from experiments.utils.training.learning_rate_controller import Controller

from experiments.utils.directory import mkdir, mkdir_join
from experiments.utils.parameter import count_total_parameters
from experiments.utils.csv import save_loss, save_ler
from models.attention.joint_ctc_attention import JointCTCAttention


def do_train(network, params):
    """Run training. If target labels are phone, the model is evaluated by PER
    with 39 phones.
    Args:
        network: network to train
        params: A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(
        data_type='train', label_type=params['label_type'],
        batch_size=params['batch_size'],
        eos_index=params['eos_index'], sort_utt=True)
    dev_data = Dataset(
        data_type='dev', label_type=params['label_type'],
        batch_size=params['batch_size'],
        eos_index=params['eos_index'], sort_utt=False)
    if params['label_type'] in ['character', 'character_capital_divide']:
        test_data = Dataset(
            data_type='test', label_type=params['label_type'], batch_size=1,
            eos_index=params['eos_index'], sort_utt=False)
    else:
        test_data = Dataset(
            data_type='test', label_type='phone39', batch_size=1,
            eos_index=params['eos_index'], sort_utt=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        network.create_placeholders(gpu_index=0)

        # Add to the graph each operation (including model definition)
        loss_op, att_logits, ctc_logits, decoder_outputs_train, decoder_outputs_infer = network.compute_loss(
            network.inputs_pl_list[0],
            network.att_labels_pl_list[0],
            network.inputs_seq_len_pl_list[0],
            network.att_labels_seq_len_pl_list[0],
            network.ctc_labels_pl_list[0],
            network.keep_prob_input_pl_list[0],
            network.keep_prob_hidden_pl_list[0],
            network.keep_prob_output_pl_list[0])
        train_op = network.train(
            loss_op,
            optimizer=params['optimizer'],
            learning_rate=network.learning_rate_pl_list[0])
        _, decode_op_infer = network.decoder(
            decoder_outputs_train,
            decoder_outputs_infer,
            decode_type='greedy',
            beam_width=20)
        ler_op = network.compute_ler(network.att_labels_st_true_pl,
                                     network.att_labels_st_pred_pl)

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
                network.model_dir, sess.graph)

            # Initialize param
            sess.run(init_op)

            # Train model
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            ler_dev_best = 1
            learning_rate = float(params['learning_rate'])
            epoch = 1
            for step, (data, next_epoch_flag) in enumerate(train_data()):

                # Create feed dictionary for next mini batch (train)
                inputs, att_labels_train, ctc_labels, inputs_seq_len, att_labels_seq_len, _ = data
                feed_dict_train = {
                    network.inputs_pl_list[0]: inputs,
                    network.att_labels_pl_list[0]: att_labels_train,
                    network.inputs_seq_len_pl_list[0]: inputs_seq_len,
                    network.att_labels_seq_len_pl_list[0]: att_labels_seq_len,
                    network.ctc_labels_pl_list[0]: list2sparsetensor(ctc_labels, padded_value=-1),
                    network.keep_prob_input_pl_list[0]: network.dropout_ratio_input,
                    network.keep_prob_hidden_pl_list[0]: network.dropout_ratio_hidden,
                    network.keep_prob_output_pl_list[0]: network.dropout_ratio_output,
                    network.learning_rate_pl_list[0]: learning_rate
                }

                # Update param
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % 10 == 0:

                    # Create feed dictionary for next mini batch (dev)
                    (inputs, att_labels_dev, ctc_labels, inputs_seq_len,
                     att_labels_seq_len, _), _ = dev_data().__next__()
                    feed_dict_dev = {
                        network.inputs_pl_list[0]: inputs,
                        network.att_labels_pl_list[0]: att_labels_dev,
                        network.inputs_seq_len_pl_list[0]: inputs_seq_len,
                        network.att_labels_seq_len_pl_list[0]: att_labels_seq_len,
                        network.ctc_labels_pl_list[0]: list2sparsetensor(ctc_labels, padded_value=-1),
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

                    # Predict class ids & update event files
                    predicted_ids_train, summary_str_train = sess.run(
                        [decode_op_infer, summary_train],
                        feed_dict=feed_dict_train)
                    predicted_ids_dev, summary_str_dev = sess.run(
                        [decode_op_infer, summary_dev],
                        feed_dict=feed_dict_dev)
                    summary_writer.add_summary(summary_str_train, step + 1)
                    summary_writer.add_summary(summary_str_dev, step + 1)
                    summary_writer.flush()

                    # Convert to sparsetensor to compute LER
                    feed_dict_ler_train = {
                        network.att_labels_true_st: list2sparsetensor(
                            att_labels_train,
                            padded_value=params['eos_index']),
                        network.att_labels_st_pred_pl: list2sparsetensor(
                            predicted_ids_train,
                            padded_value=params['eos_index'])
                    }
                    feed_dict_ler_dev = {
                        network.att_labels_true_st: list2sparsetensor(
                            att_labels_dev,
                            padded_value=params['eos_index']),
                        network.att_labels_st_pred_pl: list2sparsetensor(
                            predicted_ids_dev,
                            padded_value=params['eos_index'])
                    }

                    # Compute accuracy
                    ler_train = sess.run(
                        ler_op, feed_dict=feed_dict_ler_train)
                    ler_dev = sess.run(
                        ler_op, feed_dict=feed_dict_ler_dev)
                    csv_ler_train.append(ler_train)
                    csv_ler_dev.append(ler_dev)

                    duration_step = time.time() - start_time_step
                    print("Step %d: loss = %.3f (%.3f) / ler = %.4f (%.4f) / lr = %.5f (%.3f min)" %
                          (step + 1, loss_train, loss_dev, ler_train, ler_dev,
                           learning_rate, duration_step / 60))
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
                        if params['label_type'] in ['character', 'character_capital_divide']:
                            print('=== Dev Data Evaluation ===')
                            ler_dev_epoch = do_eval_cer(
                                session=sess,
                                decode_op=decode_op_infer,
                                network=network,
                                dataset=dev_data,
                                eval_batch_size=1)
                            print('  CER: %f %%' % (ler_dev_epoch * 100))

                            if ler_dev_epoch < ler_dev_best:
                                ler_dev_best = ler_dev_epoch
                                print('■■■ ↑Best Score (CER)↑ ■■■')

                                print('=== Test Data Evaluation ===')
                                ler_test = do_eval_cer(
                                    session=sess,
                                    decode_op=decode_op_infer,
                                    network=network,
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
                                network=network,
                                dataset=dev_data,
                                label_type=params['label_type'],
                                eos_index=params['eos_index'],
                                eval_batch_size=1)
                            print('  PER: %f %%' % (ler_dev_epoch * 100))

                            if ler_dev_epoch < ler_dev_best:
                                ler_dev_best = ler_dev_epoch
                                print('■■■ ↑Best Score (PER)↑ ■■■')

                                print('=== Test Data Evaluation ===')
                                ler_test = do_eval_per(
                                    session=sess,
                                    decode_op=decode_op_infer,
                                    per_op=ler_op,
                                    network=network,
                                    dataset=test_data,
                                    label_type=params['label_type'],
                                    eos_index=params['eos_index'],
                                    eval_batch_size=1)
                                print('  PER: %f %%' %
                                      (ler_test * 100))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

                        # Update learning rate
                        learning_rate = lr_controller.decay_lr(
                            learning_rate=learning_rate,
                            epoch=epoch,
                            value=ler_dev_epoch)

                        if epoch == params['num_epoch']:
                            break

                    epoch += 1
                    start_time_epoch = time.time()

                start_time_step = time.time()

            duration_train = time.time() - start_time_train
            print('Total time: %.3f hour' % (duration_train / 3600))

            # Save train & dev loss, ler
            save_loss(csv_steps, csv_loss_train, csv_loss_dev,
                      save_path=network.model_dir)
            save_ler(csv_steps, csv_ler_train, csv_loss_dev,
                     save_path=network.model_dir)

            # Training was finished correctly
            with open(join(network.model_dir, 'complete.txt'), 'w') as f:
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
    network = JointCTCAttention(
        batch_size=params['batch_size'],
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
        dropout_ratio_input=params['dropout_input'],
        dropout_ratio_hidden=params['dropout_hidden'],
        dropout_ratio_output=params['dropout_output'],
        weight_decay=params['weight_decay'])

    network.model_name = params['model']
    network.model_name += '_encoder' + str(params['encoder_num_unit'])
    network.model_name += '_' + str(params['encoder_num_layer'])
    network.model_name += '_attdim' + str(params['attention_dim'])
    network.model_name += '_decoder' + str(params['decoder_num_unit'])
    network.model_name += '_' + str(params['decoder_num_layer'])
    network.model_name += '_' + params['optimizer']
    network.model_name += '_lr' + str(params['learning_rate'])
    network.model_name += '_' + params['attention_type']
    # if bool(params['attention_smoothing']):
    #     network.model_name += '_smoothing'
    if params['attention_weights_tempareture'] != 1:
        network.model_name += '_sharpening' + \
            str(params['attention_weights_tempareture'])
    if params['weight_decay'] != 0:
        network.model_name += '_weightdecay' + str(params['weight_decay'])

    # Set save path
    network.model_dir = mkdir(model_save_path)
    network.model_dir = mkdir_join(network.model_dir, 'attention')
    network.model_dir = mkdir_join(network.model_dir, params['label_type'])
    network.model_dir = mkdir_join(network.model_dir, network.model_name)

    # Reset model directory
    if not isfile(join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('timit_jointctcatt_' + params['label_type'])

    # Save config file
    shutil.copyfile(config_path, join(network.model_dir, 'config.yml'))

    sys.stdout = open(join(network.model_dir, 'train.log'), 'w')
    do_train(network=network, params=params)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 3:
        raise ValueError
    main(config_path=args[1], model_save_path=args[2])
