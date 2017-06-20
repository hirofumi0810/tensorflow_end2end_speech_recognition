#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate trained Attention-based model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from data.read_dataset_attention import DataSet
# from models.attention.load_model import load
from models.attention import blstm_attention_seq2seq
from metric.attention import do_eval_per, do_eval_cer


def do_eval(network, label_type, eos_index, epoch=None):
    """Evaluate the model.
    Args:
        network: model to restore
        label_type: phone39 or phone48 or phone61 or character
        epoch: epoch to restore
        eos_index: int, the index of <EOS> class. This is used for padding.
    """
    # Load dataset
    if label_type == 'character':
        test_data = DataSet(data_type='test', label_type='character',
                            batch_size=1,
                            eos_index=eos_index,
                            is_sorted=False, is_progressbar=True)
    else:
        test_data = DataSet(data_type='test', label_type='phone39',
                            batch_size=1,
                            eos_index=eos_index,
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
        if label_type == 'character':
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
                label_type=label_type,
                is_progressbar=True)
            print('  PER: %f %%' % (per_test * 100))


def main(model_path):

    epoch = None  # if None, restore the final epoch

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
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
        output_size = 33

    # Model setting
    # AttentionModel = load(model_type=config['model_name'])
    network = blstm_attention_seq2seq.BLSTMAttetion(
        batch_size=1,
        input_size=feature['input_size'],
        encoder_num_unit=param['encoder_num_unit'],
        encoder_num_layer=param['encoder_num_layer'],
        attention_dim=param['attention_dim'],
        decoder_num_unit=param['decoder_num_unit'],
        decoder_num_layer=param['decoder_num_layer'],
        embedding_dim=param['embedding_dim'],
        output_size=output_size,
        sos_index=output_size - 2,
        eos_index=output_size - 1,
        max_decode_length=param['max_decode_length'],
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
    do_eval(network=network,
            label_type=corpus['label_type'],
            eos_index=output_size - 1,
            epoch=epoch)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError(
            ("Set a path to saved model.\n"
             "Usase: python restore_ctc.py path_to_saved_model"))
    main(model_path=args[1])
