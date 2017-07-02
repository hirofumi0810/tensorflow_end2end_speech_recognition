#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained Attention model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../../../')
from experiments.timit.data.load_dataset_attention import Dataset
from experiments.timit.metrics.attention import do_eval_per, do_eval_cer
from models.attention import blstm_attention_seq2seq


def do_eval(network, param, epoch=None):
    """Evaluate the model.
    Args:
        network: model to restore
        param: A dictionary of parameters
        epoch: int the epoch to restore
    """
    # Load dataset
    if param['label_type'] == 'character':
        test_data = Dataset(data_type='test', label_type='character',
                            batch_size=1,
                            eos_index=param['eos_index'],
                            is_sorted=False, is_progressbar=True)
    else:
        test_data = Dataset(data_type='test', label_type='phone39',
                            batch_size=1,
                            eos_index=param['eos_index'],
                            is_sorted=False, is_progressbar=True)

    # Define placeholders
    network.inputs = tf.placeholder(tf.float32,
                                    shape=[None, None, network.input_size],
                                    name='input')
    network.labels = tf.placeholder(tf.int32,
                                    shape=[None, None],
                                    name='label')
    # These are prepared for computing LER
    indices_true_pl = tf.placeholder(tf.int64, name='indices_pred')
    values_true_pl = tf.placeholder(tf.int32, name='values_pred')
    shape_true_pl = tf.placeholder(tf.int64, name='shape_pred')
    network.labels_st_true = tf.SparseTensor(indices_true_pl,
                                             values_true_pl,
                                             shape_true_pl)
    indices_pred_pl = tf.placeholder(tf.int64, name='indices_pred')
    values_pred_pl = tf.placeholder(tf.int32, name='values_pred')
    shape_pred_pl = tf.placeholder(tf.int64, name='shape_pred')
    network.labels_st_pred = tf.SparseTensor(indices_pred_pl,
                                             values_pred_pl,
                                             shape_pred_pl)
    network.inputs_seq_len = tf.placeholder(tf.int32,
                                            shape=[None],
                                            name='inputs_seq_len')
    network.labels_seq_len = tf.placeholder(tf.int32,
                                            shape=[None],
                                            name='labels_seq_len')
    network.keep_prob_input = tf.placeholder(tf.float32,
                                             name='keep_prob_input')
    network.keep_prob_hidden = tf.placeholder(tf.float32,
                                              name='keep_prob_hidden')

    # Add to the graph each operation (including model definition)
    _, _, decoder_outputs_train, decoder_outputs_infer = network.compute_loss(
        network.inputs,
        network.labels,
        network.inputs_seq_len,
        network.labels_seq_len,
        network.keep_prob_input,
        network.keep_prob_hidden)
    _, decode_op_infer = network.decoder(
        decoder_outputs_train,
        decoder_outputs_infer,
        decode_type='greedy',
        beam_width=20)
    per_op = network.compute_ler(network.labels_st_true,
                                 network.labels_st_pred)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(network.model_dir)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch is not None:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        print('Test Data Evaluation:')
        if param['label_type'] == 'character':
            cer_test = do_eval_cer(
                session=sess,
                decode_op=decode_op_infer,
                network=network,
                dataset=test_data,
                is_progressbar=True)
            print('  CER: %f %%' % (cer_test * 100))
        else:
            per_test = do_eval_per(
                session=sess,
                decode_op=decode_op_infer,
                per_op=per_op,
                network=network,
                dataset=test_data,
                label_type=param['label_type'],
                eos_index=param['eos_index'],
                is_progressbar=True)
            print('  PER: %f %%' % (per_test * 100))


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        param = config['param']

    if param['label_type'] == 'phone61':
        param['num_classes'] = 63
        param['sos_index'] = 0
        param['eos_index'] = 1
    elif param['label_type'] == 'phone48':
        param['num_classes'] = 50
        param['sos_index'] = 0
        param['eos_index'] = 1
    elif param['label_type'] == 'phone39':
        param['num_classes'] = 41
        param['sos_index'] = 0
        param['eos_index'] = 1
    elif param['label_type'] == 'character':
        param['num_classes'] = 35
        param['sos_index'] = 1
        param['eos_index'] = 2

    # Model setting
    # AttentionModel = load(model_type=param['model'])
    network = blstm_attention_seq2seq.BLSTMAttetion(
        batch_size=1,
        input_size=param['input_size'],
        encoder_num_unit=param['encoder_num_unit'],
        encoder_num_layer=param['encoder_num_layer'],
        attention_dim=param['attention_dim'],
        attention_type=param['attention_type'],
        decoder_num_unit=param['decoder_num_unit'],
        decoder_num_layer=param['decoder_num_layer'],
        embedding_dim=param['embedding_dim'],
        num_classes=param['num_classes'],
        sos_index=param['sos_index'],
        eos_index=param['eos_index'],
        max_decode_length=param['max_decode_length'],
        attention_smoothing=param['attention_smoothing'],
        attention_weights_tempareture=param['attention_weights_tempareture'],
        logits_tempareture=param['logits_tempareture'],
        parameter_init=param['weight_init'],
        clip_grad=param['clip_grad'],
        clip_activation_encoder=param['clip_activation_encoder'],
        clip_activation_decoder=param['clip_activation_decoder'],
        dropout_ratio_input=param['dropout_input'],
        dropout_ratio_hidden=param['dropout_hidden'],
        weight_decay=param['weight_decay'])

    network.model_dir = model_path
    print(network.model_dir)
    do_eval(network=network, param=param, epoch=epoch)


if __name__ == '__main__':

    args = sys.argv
    if len(args) == 2:
        model_path = args[1]
        epoch = None
    elif len(args) == 3:
        model_path = args[1]
        epoch = args[2]
    else:
        raise ValueError(
            ("Set a path to saved model.\n"
             "Usase: python eval_attention.py path_to_saved_model (epoch)"))
    main(model_path=model_path, epoch=epoch)
