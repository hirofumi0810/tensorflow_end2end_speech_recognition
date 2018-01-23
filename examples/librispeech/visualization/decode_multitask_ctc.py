#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the trained multi-task CTC outputs (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import tensorflow as tf
import yaml
import argparse

sys.path.append(abspath('../../../'))
from experiments.librispeech.data.load_dataset_multitask_ctc import Dataset
from models.ctc.multitask_ctc import MultitaskCTC
from utils.io.labels.character import Idx2char
from utils.io.labels.word import Idx2word
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
parser.add_argument('--eval_batch_size', type=int, default=-1,
                    help='the size of mini-batch when evaluation. ' +
                    'If you set -1, batch size is the same as that when training.')


def do_decode(model, params, epoch, beam_width, eval_batch_size):
    """Decode the CTC outputs.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        beam_width (int): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
    """
    # Load dataset
    test_clean_data = Dataset(
        data_type='test_clean', train_data_size=params['train_data_size'],
        label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)
    test_other_data = Dataset(
        data_type='test_other', train_data_size=params['train_data_size'],
        label_type_main=params['label_type_main'],
        label_type_sub=params['label_type_sub'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    with tf.name_scope('tower_gpu0'):
        # Define placeholders
        model.create_placeholders()

        # Add to the graph each operation (including model definition)
        _, logits_word, logits_char = model.compute_loss(
            model.inputs_pl_list[0],
            model.labels_pl_list[0],
            model.labels_sub_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.keep_prob_pl_list[0])
        decode_op_word, decode_op_char = model.decoder(
            logits_word, logits_char,
            model.inputs_seq_len_pl_list[0],
            beam_width=beam_width)

    # Create a saver for writing training checkpoints
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(model.save_path)

        # If check point exists
        if ckpt:
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
               decode_op_main=decode_op_word,
               decode_op_sub=decode_op_char,
               model=model,
               dataset=test_clean_data,
               label_type_main=params['label_type_main'],
               label_type_sub=params['label_type_sub'],
               train_data_size=params['train_data_size'],
               is_test=True,
               save_path=None)
        # save_path=model.save_path)

        decode(session=sess,
               decode_op_main=decode_op_word,
               decode_op_sub=decode_op_char,
               model=model,
               dataset=test_other_data,
               label_type_main=params['label_type_main'],
               label_type_sub=params['label_type_sub'],
               train_data_size=params['train_data_size'],
               is_test=True,
               save_path=None)
        # save_path=model.save_path)


def decode(session, decode_op_main, decode_op_sub, model,
           dataset, train_data_size, label_type_main,
           label_type_sub, is_test=True, save_path=None):
    """Visualize label outputs of Multi-task CTC model.
    Args:
        session: session of training model
        decode_op_main: operation for decoding in the main task
        decode_op_sub: operation for decoding in the sub task
        model: the model to evaluate
        dataset: An instance of a `Dataset` class
        label_type_main (string): word
        label_type_sub (string): character or character_capital_divide
        train_data_size (string, optional): train100h or train460h or
            train960h
        is_test (bool, optional): set to True when evaluating by the test set
        save_path (string, optional): path to save decoding results
    """
    idx2word = Idx2word(
        map_file_path='../metrics/mapping_files/word_' + train_data_size + '.txt')
    idx2char = Idx2char(
        map_file_path='../metrics/mapping_files/' + label_type_sub + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true_word, labels_true_char, inputs_seq_len, input_names = data
        feed_dict = {
            model.inputs_pl_list[0]: inputs[0],
            model.inputs_seq_len_pl_list[0]: inputs_seq_len[0],
            model.keep_prob_hidden_pl_list[0]: 1.0
        }

        # Decode
        batch_size = inputs[0].shape[0]
        labels_pred_st_word, labels_pred_st_char = session.run(
            [decode_op_main, decode_op_sub], feed_dict=feed_dict)
        try:
            labels_pred_word = sparsetensor2list(
                labels_pred_st_word, batch_size=batch_size)
        except IndexError:
            # no output
            labels_pred_word = ['']
        try:
            labels_pred_char = sparsetensor2list(
                labels_pred_st_char, batch_size=batch_size)
        except IndexError:
            # no output
            labels_pred_char = ['']

        # Visualize
        for i_batch in range(batch_size):
            print('----- wav: %s -----' % input_names[0][i_batch])
            if is_test:
                str_true_word = labels_true_word[0][i_batch][0]
                str_true_char = labels_true_char[0][i_batch][0]
            else:
                str_true_word = '_'.join(
                    idx2word(labels_true_word[0][i_batch]))
                str_true_char = idx2char(labels_true_char[0][i_batch])

            str_pred_word = '_'.join(idx2word(labels_pred_word[0]))
            str_pred_char = idx2char(labels_pred_char[0])

            print('Ref (word): %s' % str_true_word)
            print('Ref (char): %s' % str_true_char)
            print('Hyp (word): %s' % str_pred_word)
            print('Hyp (char): %s' % str_pred_char)

        if is_new_epoch:
            break


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
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
    model = MultitaskCTC(encoder_type=params['encoder_type'],
                         input_size=params['input_size']
                         splice=params['splice'],
                         num_stack=params['num_stack'],
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
