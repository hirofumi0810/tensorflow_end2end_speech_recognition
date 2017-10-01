#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained multi-task CTC model (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml
import argparse

sys.path.append(os.path.abspath('../../../'))
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.timit.metrics.ctc import do_eval_per, do_eval_cer
from models.ctc.multitask_ctc import Multitask_CTC

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--beam_width', type=int, default=20,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='the size of mini-batch when evaluation')


def do_eval(model, params, epoch, batch_size, beam_width):
    """Evaluate the model.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        batch_size (int): the size of mini-batch when evaluation
        beam_width (int): beam_width (int, optional): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
    """
    # Load dataset
    test_data = Dataset(
        data_type='test', label_type_main=params['label_type_main'],
        label_type_sub='phone39',
        batch_size=batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, progressbar=True)

    # Define placeholders
    model.create_placeholders()

    # Add to the graph each operation
    _, logits_main, logits_sub = model.compute_loss(
        model.inputs_pl_list[0],
        model.labels_pl_list[0],
        model.labels_sub_pl_list[0],
        model.inputs_seq_len_pl_list[0],
        model.keep_prob_input_pl_list[0],
        model.keep_prob_hidden_pl_list[0],
        model.keep_prob_output_pl_list[0])
    decode_op_main, decode_op_sub = model.decoder(
        logits_main, logits_sub,
        model.inputs_seq_len_pl_list[0],
        beam_width=beam_width)
    _, per_op = model.compute_ler(
        decode_op_main, decode_op_sub,
        model.labels_pl_list[0], model.labels_sub_pl_list[0])

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model.save_path)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch != -1:
                # Use the best model
                # NOTE: In the training stage, parameters are saved only when
                # accuracies are improved
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        print('=== Test Data Evaluation ===')
        cer_test, wer_test = do_eval_cer(
            session=sess,
            decode_op=decode_op_main,
            model=model,
            dataset=test_data,
            label_type=params['label_type_main'],
            eval_batch_size=1,
            progressbar=True,
            is_multitask=True)
        print('  WER: %f %%' % (wer_test * 100))
        print('  CER: %f %%' % (cer_test * 100))

        per_test = do_eval_per(
            session=sess,
            decode_op=decode_op_sub,
            per_op=per_op,
            model=model,
            dataset=test_data,
            label_type=params['label_type_sub'],
            eval_batch_size=1,
            progressbar=True,
            is_multitask=True)
        print('  PER: %f %%' % (per_test * 100))


def main():

    args = parser.parse_args()

    # Load config file
    with open(os.path.join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank label
    if params['label_type_main'] == 'character':
        params['num_classes_main'] = 28
    elif params['label_type_main'] == 'character_capital_divide':
        params['num_classes_main'] = 72

    if params['label_type_sub'] == 'phone61':
        params['num_classes_sub'] = 61
    elif params['label_type_sub'] == 'phone48':
        params['num_classes_sub'] = 48
    elif params['label_type_sub'] == 'phone39':
        params['num_classes_sub'] = 39

    # Model setting
    model = Multitask_CTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        num_units=params['num_units'],
        num_layers_main=params['num_layers_main'],
        num_layers_sub=params['num_layers_sub'],
        num_classes_main=params['num_classes_main'],
        num_classes_sub=params['num_classes_sub'],
        main_task_weight=params['main_task_weight'],
        lstm_impl=params['lstm_impl'],
        use_peephole=params['use_peephole'],
        parameter_init=params['weight_init'],
        clip_grad=params['clip_grad'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_eval(model=model, params=params,
            epoch=args.epoch, batch_size=args.batch_size,
            beam_width=args.beam_width)


if __name__ == '__main__':
    main()
