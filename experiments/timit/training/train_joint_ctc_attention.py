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
from experiments.utils.sparsetensor import list2sparsetensor
from experiments.utils.directory import mkdir, mkdir_join
from experiments.utils.parameter import count_total_parameters
from experiments.utils.csv import save_loss, save_ler
from models.attention.joint_ctc_attention import JointCTCAttention


def do_train(network, param):
    """Run training. If target labels are phone, the model is evaluated by PER
    with 39 phones.
    Args:
        network: network to train
        param: A dictionary of parameters
    """
    # Load dataset
    train_data = Dataset(data_type='train', label_type=param['label_type'],
                         batch_size=param['batch_size'],
                         eos_index=param['eos_index'], is_sorted=True)
    dev_data = Dataset(data_type='dev', label_type=param['label_type'],
                       batch_size=param['batch_size'],
                       eos_index=param['eos_index'], is_sorted=False)
    if param['label_type'] == 'character':
        test_data = Dataset(data_type='test', label_type='character',
                            batch_size=1,
                            eos_index=param['eos_index'], is_sorted=False)
    else:
        test_data = Dataset(data_type='test', label_type='phone39',
                            batch_size=1,
                            eos_index=param['eos_index'], is_sorted=False)

    # Tell TensorFlow that the model will be built into the default graph
    with tf.Graph().as_default():

        # Define placeholders
        network.inputs = tf.placeholder(tf.float32,
                                        shape=[None, None, network.input_size],
                                        name='inputs')
        network.att_labels = tf.placeholder(tf.int32,
                                            shape=[None, None],
                                            name='att_labels')
        indices_pl = tf.placeholder(tf.int64, name='ctc_indices')
        values_pl = tf.placeholder(tf.int32, name='ctc_values')
        shape_pl = tf.placeholder(tf.int64, name='ctc_shape')
        network.ctc_labels = tf.SparseTensor(indices_pl,
                                             values_pl,
                                             shape_pl)

        # These are prepared for computing LER
        indices_true_pl = tf.placeholder(tf.int64, name='indices_true')
        values_true_pl = tf.placeholder(tf.int32, name='values_true')
        shape_true_pl = tf.placeholder(tf.int64, name='shape_true')
        network.att_labels_true_st = tf.SparseTensor(indices_true_pl,
                                                     values_true_pl,
                                                     shape_true_pl)
        indices_pred_pl = tf.placeholder(tf.int64, name='indices_pred')
        values_pred_pl = tf.placeholder(tf.int32, name='values_pred')
        shape_pred_pl = tf.placeholder(tf.int64, name='shape_pred')
        network.att_labels_pred_st = tf.SparseTensor(indices_pred_pl,
                                                     values_pred_pl,
                                                     shape_pred_pl)
        network.inputs_seq_len = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name='inputs_seq_len')
        network.att_labels_seq_len = tf.placeholder(tf.int32,
                                                    shape=[None],
                                                    name='att_labels_seq_len')
        network.keep_prob_input = tf.placeholder(tf.float32,
                                                 name='keep_prob_input')
        network.keep_prob_hidden = tf.placeholder(tf.float32,
                                                  name='keep_prob_hidden')

        # Add to the graph each operation (including model definition)
        loss_op, att_logits, ctc_logits, decoder_outputs_train, decoder_outputs_infer = network.compute_loss(
            network.inputs,
            network.att_labels,
            network.inputs_seq_len,
            network.att_labels_seq_len,
            network.ctc_labels,
            network.keep_prob_input,
            network.keep_prob_hidden)
        train_op = network.train(
            loss_op,
            optimizer=param['optimizer'],
            learning_rate_init=float(param['learning_rate']),
            is_scheduled=False)
        _, decode_op_infer = network.decoder(
            decoder_outputs_train,
            decoder_outputs_infer,
            decode_type='greedy',
            beam_width=20)
        ler_op = network.compute_ler(network.att_labels_true_st,
                                     network.att_labels_pred_st)

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

        # Make mini-batch generator
        mini_batch_train = train_data.next_batch()
        mini_batch_dev = dev_data.next_batch()

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
            iter_per_epoch = int(train_data.data_num / param['batch_size'])
            train_step = train_data.data_num / param['batch_size']
            if train_step != int(train_step):
                iter_per_epoch += 1
            max_steps = iter_per_epoch * param['num_epoch']
            start_time_train = time.time()
            start_time_epoch = time.time()
            start_time_step = time.time()
            error_best = 1
            for step in range(max_steps):

                # Create feed dictionary for next mini batch (train)
                inputs, att_labels_train, ctc_labels_st, inputs_seq_len, att_labels_seq_len, _ = mini_batch_train.__next__()
                feed_dict_train = {
                    network.inputs: inputs,
                    network.att_labels: att_labels_train,
                    network.inputs_seq_len: inputs_seq_len,
                    network.att_labels_seq_len: att_labels_seq_len,
                    network.ctc_labels: ctc_labels_st,
                    network.keep_prob_input: network.dropout_ratio_input,
                    network.keep_prob_hidden: network.dropout_ratio_hidden,
                    network.lr: float(param['learning_rate'])
                }

                # Create feed dictionary for next mini batch (dev)
                inputs, att_labels_dev, ctc_labels_st, inputs_seq_len, att_labels_seq_len, _ = mini_batch_dev.__next__()
                feed_dict_dev = {
                    network.inputs: inputs,
                    network.att_labels: att_labels_dev,
                    network.inputs_seq_len: inputs_seq_len,
                    network.att_labels_seq_len: att_labels_seq_len,
                    network.ctc_labels: ctc_labels_st,
                    network.keep_prob_input: network.dropout_ratio_input,
                    network.keep_prob_hidden: network.dropout_ratio_hidden
                }

                # Update param
                sess.run(train_op, feed_dict=feed_dict_train)

                if (step + 1) % 10 == 0:

                    # Compute loss
                    loss_train = sess.run(loss_op, feed_dict=feed_dict_train)
                    loss_dev = sess.run(loss_op, feed_dict=feed_dict_dev)
                    csv_steps.append(step)
                    csv_loss_train.append(loss_train)
                    csv_loss_dev.append(loss_dev)

                    # Change to evaluation mode
                    feed_dict_train[network.keep_prob_input] = 1.0
                    feed_dict_train[network.keep_prob_hidden] = 1.0
                    feed_dict_dev[network.keep_prob_input] = 1.0
                    feed_dict_dev[network.keep_prob_hidden] = 1.0

                    # Predict class ids &  update event file
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
                            padded_value=param['eos_index']),
                        network.att_labels_pred_st: list2sparsetensor(
                            predicted_ids_train,
                            padded_value=param['eos_index'])
                    }
                    feed_dict_ler_dev = {
                        network.att_labels_true_st: list2sparsetensor(
                            att_labels_dev,
                            padded_value=param['eos_index']),
                        network.att_labels_pred_st: list2sparsetensor(
                            predicted_ids_dev,
                            padded_value=param['eos_index'])
                    }

                    # Compute accuracy
                    ler_train = sess.run(
                        ler_op, feed_dict=feed_dict_ler_train)
                    ler_dev = sess.run(
                        ler_op, feed_dict=feed_dict_ler_dev)
                    csv_ler_train.append(ler_train)
                    csv_ler_dev.append(ler_dev)

                    duration_step = time.time() - start_time_step
                    print("Step %d: loss = %.3f (%.3f) / ler = %.4f (%.4f) (%.3f min)" %
                          (step + 1, loss_train, loss_dev, ler_train, ler_dev,
                           duration_step / 60))
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

                    if epoch >= 20:
                        start_time_eval = time.time()
                        if param['label_type'] == 'character':
                            print('=== Dev Data Evaluation ===')
                            cer_dev_epoch = do_eval_cer(
                                session=sess,
                                decode_op=decode_op_infer,
                                network=network,
                                dataset=dev_data,
                                eval_batch_size=1)
                            print('  CER: %f %%' % (cer_dev_epoch * 100))

                            if cer_dev_epoch < error_best:
                                error_best = cer_dev_epoch
                                print('■■■ ↑Best Score (CER)↑ ■■■')

                                print('=== Test Data Evaluation ===')
                                cer_test = do_eval_cer(
                                    session=sess,
                                    decode_op=decode_op_infer,
                                    network=network,
                                    dataset=test_data,
                                    eval_batch_size=1)
                                print('  CER: %f %%' %
                                      (cer_test * 100))

                        else:
                            print('=== Dev Data Evaluation ===')
                            per_dev_epoch = do_eval_per(
                                session=sess,
                                decode_op=decode_op_infer,
                                per_op=ler_op,
                                network=network,
                                dataset=dev_data,
                                label_type=param['label_type'],
                                eos_index=param['eos_index'],
                                eval_batch_size=1)
                            print('  PER: %f %%' % (per_dev_epoch * 100))

                            if per_dev_epoch < error_best:
                                error_best = per_dev_epoch
                                print('■■■ ↑Best Score (PER)↑ ■■■')

                                print('=== Test Data Evaluation ===')
                                per_test = do_eval_per(
                                    session=sess,
                                    decode_op=decode_op_infer,
                                    per_op=ler_op,
                                    network=network,
                                    dataset=test_data,
                                    label_type=param['label_type'],
                                    eos_index=param['eos_index'],
                                    eval_batch_size=1)
                                print('  PER: %f %%' %
                                      (per_test * 100))

                        duration_eval = time.time() - start_time_eval
                        print('Evaluation time: %.3f min' %
                              (duration_eval / 60))

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


def main(config_path):

    # Load a config file (.yml)
    with open(config_path, "r") as f:
        config = yaml.load(f)
        param = config['param']

    if param['label_type'] == 'phone61':
        param['att_num_classes'] = 63
        param['ctc_num_classes'] = 61
        param['sos_index'] = 0
        param['eos_index'] = 1
    elif param['label_type'] == 'phone48':
        param['att_num_classes'] = 50
        param['ctc_num_classes'] = 48
        param['sos_index'] = 0
        param['eos_index'] = 1
    elif param['label_type'] == 'phone39':
        param['att_num_classes'] = 41
        param['ctc_num_classes'] = 39
        param['sos_index'] = 0
        param['eos_index'] = 1
    elif param['label_type'] == 'character':
        param['att_num_classes'] = 35
        param['ctc_num_classes'] = 33
        param['sos_index'] = 1
        param['eos_index'] = 2

    # Model setting
    # AttentionModel = load(model_type=config['model_name'])
    network = JointCTCAttention(
        batch_size=param['batch_size'],
        input_size=param['input_size'],
        encoder_num_unit=param['encoder_num_unit'],
        encoder_num_layer=param['encoder_num_layer'],
        attention_dim=param['attention_dim'],
        attention_type=param['attention_type'],
        decoder_num_unit=param['decoder_num_unit'],
        decoder_num_layer=param['decoder_num_layer'],
        embedding_dim=param['embedding_dim'],
        att_num_classes=param['att_num_classes'],
        ctc_num_classes=param['ctc_num_classes'],
        att_task_weight=param['att_task_weight'],
        sos_index=param['sos_index'],
        eos_index=param['eos_index'],
        max_decode_length=param['max_decode_length'],
        # attention_smoothing=param['attention_smoothing'],
        attention_weights_tempareture=param['attention_weights_tempareture'],
        logits_tempareture=param['logits_tempareture'],
        parameter_init=param['weight_init'],
        clip_grad=param['clip_grad'],
        clip_activation_encoder=param['clip_activation_encoder'],
        clip_activation_decoder=param['clip_activation_decoder'],
        dropout_ratio_input=param['dropout_input'],
        dropout_ratio_hidden=param['dropout_hidden'],
        weight_decay=param['weight_decay'])

    network.model_name = param['model']
    network.model_name += '_encoder' + str(param['encoder_num_unit'])
    network.model_name += '_' + str(param['encoder_num_layer'])
    network.model_name += '_attdim' + str(param['attention_dim'])
    network.model_name += '_decoder' + str(param['decoder_num_unit'])
    network.model_name += '_' + str(param['decoder_num_layer'])
    network.model_name += '_' + param['optimizer']
    network.model_name += '_lr' + str(param['learning_rate'])
    network.model_name += '_' + param['attention_type']
    # if bool(param['attention_smoothing']):
    #     network.model_name += '_smoothing'
    if param['attention_weights_tempareture'] != 1:
        network.model_name += '_sharpening' + \
            str(param['attention_weights_tempareture'])
    if param['weight_decay'] != 0:
        network.model_name += '_weightdecay' + str(param['weight_decay'])

    # Set save path
    network.model_dir = mkdir('/n/sd8/inaguma/result/timit/')
    network.model_dir = mkdir_join(network.model_dir, 'attention')
    network.model_dir = mkdir_join(network.model_dir, param['label_type'])
    network.model_dir = mkdir_join(network.model_dir, network.model_name)

    # Reset model directory
    if not isfile(join(network.model_dir, 'complete.txt')):
        tf.gfile.DeleteRecursively(network.model_dir)
        tf.gfile.MakeDirs(network.model_dir)
    else:
        raise ValueError('File exists.')

    # Set process name
    setproctitle('timit_jointctcatt_' + param['label_type'])

    # Save config file
    shutil.copyfile(config_path, join(network.model_dir, 'config.yml'))

    sys.stdout = open(join(network.model_dir, 'train.log'), 'w')
    print(network.model_name)
    do_train(network=network, param=param)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError
    main(config_path=args[1])
