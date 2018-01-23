#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the trained multi-task CTC outputs (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import tensorflow as tf
import yaml
import argparse

sys.path.append(abspath('../../../'))
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from models.ctc.multitask_ctc import MultitaskCTC
from utils.io.labels.character import Idx2char
from utils.io.labels.phone import Idx2phone
from utils.io.labels.sparsetensor import sparsetensor2list
from utils.evaluation.edit_distance import wer_align

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--beam_width', type=int, default=20,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=str, default=1,
                    help='the size of mini-batch in evaluation')


def do_decode(model, params, epoch, beam_width, eval_batch_size):
    """Decode the Multi-task CTC outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        beam_width (int): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
    """
    # Load dataset
    test_data = Dataset(
        data_type='test', label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False, progressbar=True)

    # Define placeholders
    model.create_placeholders()

    # Add to the graph each operation (including model definition)
    _, logits_main, logits_sub = model.compute_loss(
        model.inputs_pl_list[0],
        model.labels_pl_list[0],
        model.labels_sub_pl_list[0],
        model.inputs_seq_len_pl_list[0],
        model.keep_prob_pl_list[0])
    decode_op_main, decode_op_sub = model.decoder(
        logits_main, logits_sub,
        model.inputs_seq_len_pl_list[0],
        beam_width=beam_width)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model.save_path)

        # If check point exists
        if ckpt:
            # Use last saved model
            model_path = ckpt.model_checkpoint_path
            if epoch != -1:
                model_path = model_path.split('/')[:-1]
                model_path = '/'.join(model_path) + '/model.ckpt-' + str(epoch)
            saver.restore(sess, model_path)
            print("Model restored: " + model_path)
        else:
            raise ValueError('There are not any checkpoints.')

        # Visualize
        decode(session=sess,
               decode_op_main=decode_op_main,
               decode_op_sub=decode_op_sub,
               model=model,
               dataset=test_data,
               label_type_main=params['label_type_main'],
               label_type_sub=params['label_type_sub'],
               is_test=True,
               save_path=None)
        #   save_path=model.save_path)


def decode(session, decode_op_main, decode_op_sub, model,
           dataset, label_type_main, label_type_sub,
           is_test=True, save_path=None):
    """Visualize label outputs of Multi-task CTC model.
    Args:
        session: session of training model
        decode_op_main: operation for decoding in the main task
        decode_op_sub: operation for decoding in the sub task
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type_main (string): character or character_capital_divide
        label_type_sub (string): phone39 or phone48 or phone61
        is_test (bool, optional):
        save_path (string, optional): path to save decoding results
    """
    idx2char = Idx2char(
        map_file_path='../metrics/mapping_files/' + label_type_main + '.txt')
    idx2phone = Idx2phone(
        map_file_path='../metrics/mapping_files/' + label_type_sub + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true_char, labels_true_phone, inputs_seq_len, input_names = data

        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_pl_list[0]: 1.0
        }

        batch_size = inputs[0].shape[0]
        labels_pred_char_st, labels_pred_phone_st = session.run(
            [decode_op_main, decode_op_sub], feed_dict=feed_dict)
        try:
            labels_pred_char = sparsetensor2list(
                labels_pred_char_st, batch_size=batch_size)
        except:
            # no output
            labels_pred_char = ['']
        try:
            labels_pred_phone = sparsetensor2list(
                labels_pred_char_st, batch_size=batch_size)
        except:
            # no output
            labels_pred_phone = ['']

        for i_batch in range(batch_size):
            print('----- wav: %s -----' % input_names[0][i_batch])

            if is_test:
                str_true_char = labels_true_char[0][i_batch][0].replace(
                    '_', ' ')
                str_true_phone = labels_true_phone[0][i_batch][0]
            else:
                str_true_char = idx2char(labels_true_char[0][i_batch])
                str_true_phone = idx2phone(labels_true_phone[0][i_batch])

            str_pred_char = idx2char(labels_pred_char[i_batch])
            str_pred_phone = idx2phone(labels_pred_phone[i_batch])

            print('Ref (char): %s' % str_true_char)
            print('Hyp (char): %s' % str_pred_char)
            print('Ref (phone): %s' % str_true_phone)
            print('Hyp (phone): %s' % str_pred_phone)

        if is_new_epoch:
            break


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
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
    model = MultitaskCTC(
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
        clip_grad_norm=params['clip_grad_norm'],
        clip_activation=params['clip_activation'],
        num_proj=params['num_proj'],
        weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_decode(model=model, params=params,
              epoch=args.epoch, beam_width=args.beam_width,
              eval_batch_size=args.eval_batch_size)


if __name__ == '__main__':
    main()
