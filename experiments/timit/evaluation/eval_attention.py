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


def do_eval(network, params, epoch=None):
    """Evaluate the model.
    Args:
        network: model to restore
        params: A dictionary of parameters
        epoch: int the epoch to restore
    """
    # Load dataset
    if 'phone' in params['label_type']:
        test_data = Dataset(
            data_type='test', label_type='phone39', batch_size=1,
            eos_index=params['eos_index'], sort_utt=False, progressbar=True)
    else:
        test_data = Dataset(
            data_type='test', label_type=params['label_type'], batch_size=1,
            eos_index=params['eos_index'], sort_utt=False, progressbar=True)
    # TODO: add frame_stacking

    # Define placeholders
    network.create_placeholders()

    # Add to the graph each operation (including model definition)
    _, _, decoder_outputs_train, decoder_outputs_infer = network.compute_loss(
        network.inputs_pl_list[0],
        network.labels_pl_list[0],
        network.inputs_seq_len_pl_list[0],
        network.labels_seq_len_pl_list[0],
        network.keep_prob_input_pl_list[0],
        network.keep_prob_hidden_pl_list[0],
        network.keep_prob_output_pl_list[0])
    _, decode_op_infer = network.decoder(
        decoder_outputs_train,
        decoder_outputs_infer)
    per_op = network.compute_ler(network.labels_st_true_pl,
                                 network.labels_st_pred_pl)

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
        if params['label_type'] in ['character', 'character_capital_divide']:
            cer_test = do_eval_cer(
                session=sess,
                decode_op=decode_op_infer,
                network=network,
                dataset=test_data,
                label_type=params['label_type'],
                eval_batch_size=1,
                progressbar=True)
            print('  CER: %f %%' % (cer_test * 100))
        else:
            per_test = do_eval_per(
                session=sess,
                decode_op=decode_op_infer,
                per_op=per_op,
                network=network,
                dataset=test_data,
                label_type=params['label_type'],
                eos_index=params['eos_index'],
                eval_batch_size=1,
                progressbar=True)
            print('  PER: %f %%' % (per_test * 100))


def main(model_path, epoch):

    # Load config file
    with open(os.path.join(model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    params['sos_index'] = 0
    params['eos_index'] = 1
    if params['label_type'] == 'phone61':
        params['num_classes'] = 63
    elif params['label_type'] == 'phone48':
        params['num_classes'] = 50
    elif params['label_type'] == 'phone39':
        params['num_classes'] = 41
    elif params['label_type'] == 'character':
        params['num_classes'] = 30
    elif params['label_type'] == 'character_capital_divide':
        params['num_classes'] = 74

    # Model setting
    # AttentionModel = load(model_type=params['model'])
    network = blstm_attention_seq2seq.BLSTMAttetion(
        input_size=params['input_size'],
        encoder_num_unit=params['encoder_num_unit'],
        encoder_num_layer=params['encoder_num_layer'],
        attention_dim=params['attention_dim'],
        attention_type=params['attention_type'],
        decoder_num_unit=params['decoder_num_unit'],
        decoder_num_layer=params['decoder_num_layer'],
        embedding_dim=params['embedding_dim'],
        num_classes=params['num_classes'],
        sos_index=params['sos_index'],
        eos_index=params['eos_index'],
        max_decode_length=params['max_decode_length'],
        attention_smoothing=params['attention_smoothing'],
        attention_weights_tempareture=params['attention_weights_tempareture'],
        logits_tempareture=params['logits_tempareture'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation_encoder=params['clip_activation_encoder'],
        clip_activation_decoder=params['clip_activation_decoder'],
        weight_decay=params['weight_decay'],
        beam_width=20)

    network.model_dir = model_path
    do_eval(network=network, params=params, epoch=epoch)


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
