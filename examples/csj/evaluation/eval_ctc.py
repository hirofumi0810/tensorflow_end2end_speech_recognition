#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the trained CTC model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import tensorflow as tf
import yaml
import argparse

sys.path.append(abspath('../../../'))
from experiments.csj.data.load_dataset_ctc import Dataset
from experiments.csj.metrics.ctc import do_eval_cer
from models.ctc.ctc import CTC

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--beam_width', type=int, default=20,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch when evaluation')


def do_eval(model, params, epoch, eval_batch_size, beam_width):
    """Evaluate the model.
    Args:
        model: the model to restore
        params (dict): A dictionary of parameters
        epoch (int): the epoch to restore
        eval_batch_size (int): the size of mini-batch when evaluation
        beam_width (int): beam_width (int, optional): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
    """
    # Load dataset
    eval1_data = Dataset(
        data_type='eval1', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)
    eval2_data = Dataset(
        data_type='eval2', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)
    eval3_data = Dataset(
        data_type='eval3', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=params['batch_size'] if eval_batch_size == -
        1 else eval_batch_size,
        splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        shuffle=False)

    with tf.name_scope('tower_gpu0'):
        # Define placeholders
        model.create_placeholders()

        # Add to the graph each operation (including model definition)
        _, logits = model.compute_loss(model.inputs_pl_list[0],
                                       model.labels_pl_list[0],
                                       model.inputs_seq_len_pl_list[0],
                                       model.keep_prob_pl_list[0])
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
        cer_eval1 = do_eval_cer(
            session=sess,
            decode_ops=[decode_op],
            model=model,
            dataset=eval1_data,
            label_type=params['label_type'],
            train_data_size=params['train_data_size'],
            is_test=True,
            # eval_batch_size=eval_batch_size,
            progressbar=True)
        print('  CER (eval1): %f %%' % (cer_eval1 * 100))

        cer_eval2 = do_eval_cer(
            session=sess,
            decode_ops=[decode_op],
            model=model,
            dataset=eval2_data,
            label_type=params['label_type'],
            train_data_size=params['train_data_size'],
            is_test=True,
            # eval_batch_size=eval_batch_size,
            progressbar=True)
        print('  CER (eval2): %f %%' % (cer_eval2 * 100))

        cer_eval3 = do_eval_cer(
            session=sess,
            decode_ops=[decode_op],
            model=model,
            dataset=eval3_data,
            label_type=params['label_type'],
            train_data_size=params['train_data_size'],
            is_test=True,
            # eval_batch_size=eval_batch_size,
            progressbar=True)
        print('  CER (eval3): %f %%' % (cer_eval3 * 100))

        cer_mean = (cer_eval1 + cer_eval2 + cer_eval3) / 3.
        print('  CER (mean): %f %%' % (cer_mean * 100))


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank label
    if params['label_type'] == 'kana':
        params['num_classes'] = 146
    elif params['label_type'] == 'kana_divide':
        params['num_classes'] = 147
    elif params['label_type'] == 'kanji':
        if params['train_data_size'] == 'train_subset':
            params['num_classes'] = 2981
        elif params['train_data_size'] == 'train_fullset':
            params['num_classes'] = 3385
    elif params['label_type'] == 'kanji_divide':
        if params['train_data_size'] == 'train_subset':
            params['num_classes'] = 2982
        elif params['train_data_size'] == 'train_fullset':
            params['num_classes'] = 3386
    else:
        raise TypeError

    # Modle setting
    model = CTC(encoder_type=params['encoder_type'],
                input_size=params['input_size'] * params['num_stack'],
                splice=params['splice'],
                num_units=params['num_units'],
                num_layers=params['num_layers'],
                num_classes=params['num_classes'],
                lstm_impl=params['lstm_impl'],
                use_peephole=params['use_peephole'],
                parameter_init=params['weight_init'],
                clip_grad_norm=params['clip_grad_norm'],
                clip_activation=params['clip_activation'],
                num_proj=params['num_proj'],
                weight_decay=params['weight_decay'])

    model.save_path = args.model_path
    do_eval(model=model, params=params,
            epoch=args.epoch, eval_batch_size=args.eval_batch_size,
            beam_width=args.beam_width)


if __name__ == '__main__':
    main()
