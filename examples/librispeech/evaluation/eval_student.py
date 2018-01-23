#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the student model (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tensorflow as tf
import yaml
import argparse

sys.path.append(os.path.abspath('../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from experiments.librispeech.metrics.ctc import do_eval_cer, do_eval_wer
from models.ctc.student_ctc import StudentCTC

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
parser.add_argument('--temperature', type=int, default=1,
                    help='temperature parameter')


def do_eval(model, params, epoch, beam_width, eval_batch_size, temperature):
    """Evaluate the model.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        beam_width (int): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
        temperature (int):
    """
    if 'temp1' in params['teacher_model_path']:
        teacher_train_temperature = 1
    elif 'temp2' in params['teacher_model_path']:
        teacher_train_temperature = 2

    print('=' * 40)
    print('  frame stack %d' % int(params['num_stack']))
    print('  splice %d' % int(params['splice']))
    print('  beam width: %d' % beam_width)
    print('  temperature (teacher, training): %d' % teacher_train_temperature)
    print('  temperature (teacher, inference): %d' %
          params['teacher_temperature'])
    print('  temperature (training): %d' % params['student_temperature'])
    print('  temperature (inference): %d' % temperature)
    print('=' * 40)

    # Load dataset
    test_clean_data = Dataset(
        data_type='test_clean', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)
    test_other_data = Dataset(
        data_type='test_other', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    with tf.name_scope('tower_gpu0') as scope:
        # Define placeholders
        model.create_placeholders_ctc()

        # Add to the graph each operation (including model definition)
        _, logits = model.compute_ctc_loss(
            model.inputs_pl_list[0],
            model.labels_pl_list[0],
            model.inputs_seq_len_pl_list[0],
            model.keep_prob_pl_list[0],
            scope,
            softmax_temperature=temperature,  # this is for training
            # is_training=False)
            is_training=True)
        logits /= temperature
        decode_op = model.decoder(logits,
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

        print('Test Data Evaluation:')
        cer_clean_test, wer_clean_test = do_eval_cer(
            session=sess,
            decode_ops=[decode_op],
            model=model,
            dataset=test_clean_data,
            label_type=params['label_type'],
            is_test=True,
            eval_batch_size=eval_batch_size,
            progressbar=True)
        print('  CER (clean): %f %%' % (cer_clean_test * 100))
        print('  WER (clean): %f %%' % (wer_clean_test * 100))

        cer_other_test, wer_other_test = do_eval_cer(
            session=sess,
            decode_ops=[decode_op],
            model=model,
            dataset=test_other_data,
            label_type=params['label_type'],
            is_test=True,
            eval_batch_size=eval_batch_size,
            progressbar=True)
        print('  CER (other): %f %%' % (cer_other_test * 100))
        print('  WER (other): %f %%' % (wer_other_test * 100))


def main():

    args = parser.parse_args()

    # Load config file
    with open(os.path.join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type'] == 'character':
        params['num_classes'] = 28

    if params['encoder_type'] == 'student_cnn_compact_xe':
        params['encoder_type'] = 'student_cnn_compact'
    elif params['encoder_type'] == 'student_cnn_xe':
        params['encoder_type'] = 'student_cnn'

    # Model setting
    model = StudentCTC(
        encoder_type=params['encoder_type'],
        input_size=params['input_size'] * params['num_stack'],
        splice=params['splice'],
        num_stack=params['num_stack'],
        num_classes=params['num_classes'],
        parameter_init=params['weight_init'],
        clip_grad_norm=params['clip_grad_norm'],
        weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_eval(model=model, params=params,
            epoch=args.epoch, beam_width=args.beam_width,
            eval_batch_size=args.eval_batch_size,
            temperature=args.temperature)


if __name__ == '__main__':
    main()
