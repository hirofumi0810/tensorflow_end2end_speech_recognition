#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the ensemble of 8 CTC models (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from experiments.librispeech.evaluation.eval_ensemble4_ctc import do_eval_cer
from utils.io.labels.character import Idx2char, Char2idx
from utils.evaluation.edit_distance import compute_cer, compute_wer, wer_align
from models.ctc.decoders.beam_search_decoder import BeamSearchDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--result_save_path', type=str, default=None,
                    help='path to save results of ensemble')

parser.add_argument('--model1_path', type=str,
                    help='path to the 1st model to evaluate')
parser.add_argument('--model2_path', type=str,
                    help='path to the 2nd model to evaluate')
parser.add_argument('--model3_path', type=str,
                    help='path to the 3rd model to evaluate')
parser.add_argument('--model4_path', type=str,
                    help='path to the 4th model to evaluate')
parser.add_argument('--model5_path', type=str,
                    help='path to the 5th model to evaluate')
parser.add_argument('--model6_path', type=str,
                    help='path to the 6th model to evaluate')
parser.add_argument('--model7_path', type=str,
                    help='path to the 7th model to evaluate')
parser.add_argument('--model8_path', type=str,
                    help='path to the 8th model to evaluate')

parser.add_argument('--epoch_model1', type=int, default=-1,
                    help='the epoch of 1st model to restore')
parser.add_argument('--epoch_model2', type=int, default=-1,
                    help='the epoch of 2nd model to restore')
parser.add_argument('--epoch_model3', type=int, default=-1,
                    help='the epoch of 3rd model to restore')
parser.add_argument('--epoch_model4', type=int, default=-1,
                    help='the epoch of 4th model to restore')
parser.add_argument('--epoch_model5', type=int, default=-1,
                    help='the epoch of 5th model to restore')
parser.add_argument('--epoch_model6', type=int, default=-1,
                    help='the epoch of 6th model to restore')
parser.add_argument('--epoch_model7', type=int, default=-1,
                    help='the epoch of 7th model to restore')
parser.add_argument('--epoch_model8', type=int, default=-1,
                    help='the epoch of 8th model to restore')

parser.add_argument('--beam_width', type=int, default=20,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--temperature_infer', type=int, default=1,
                    help='temperature parameter in the inference stage')


def do_eval(save_paths, params, beam_width, temperature_infer,
            result_save_path):
    """Evaluate the model.
    Args:
        save_paths (list):
        params (dict): A dictionary of parameters
        epoch_list (list): list of the epoch to restore
        beam_width (int): beam width for beam search.
            1 disables beam search, which mean greedy decoding.
        eval_batch_size (int): the size of mini-batch when evaluation
        temperature_infer (int): temperature in the inference stage
        result_save_path (string, optional):
    """
    if 'temp1' in save_paths[0]:
        temperature_train = 1
    elif 'temp2' in save_paths[0]:
        temperature_train = 2
    else:
        raise ValueError

    if result_save_path is not None:
        sys.stdout = open(join(result_save_path,
                               '8models_traintemp' + str(temperature_train) +
                               '_inftemp' + str(temperature_infer) + '.log'), 'w')

    print('=' * 30)
    print('  frame stack %d' % int(params['num_stack']))
    print('  beam width: %d' % beam_width)
    print('  ensemble: %d' % len(save_paths))
    print('  temperature (training): %d' % temperature_train)
    print('  temperature (inference): %d' % temperature_infer)
    print('=' * 30)

    # Load dataset
    test_clean_data = Dataset(
        data_type='test_clean', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True)
    test_other_data = Dataset(
        data_type='test_other', train_data_size=params['train_data_size'],
        label_type=params['label_type'],
        batch_size=1, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True)

    print('Test Data Evaluation:')
    cer_clean_test, wer_clean_test = do_eval_cer(
        save_paths=save_paths,
        dataset=test_clean_data,
        data_type='test_clean',
        label_type=params['label_type'],
        num_classes=params['num_classes'] + 1,
        beam_width=beam_width,
        temperature_infer=temperature_infer,
        is_test=True,
        progressbar=True)
    print('  CER (clean): %f %%' % (cer_clean_test * 100))
    print('  WER (clean): %f %%' % (wer_clean_test * 100))

    cer_other_test, wer_other_test = do_eval_cer(
        save_paths=save_paths,
        dataset=test_other_data,
        data_type='test_other',
        label_type=params['label_type'],
        num_classes=params['num_classes'] + 1,
        beam_width=beam_width,
        temperature_infer=temperature_infer,
        is_test=True,
        progressbar=True)
    print('  CER (other): %f %%' % (cer_other_test * 100))
    print('  WER (other): %f %%' % (wer_other_test * 100))


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model1_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Except for a blank class
    if params['label_type'] == 'character':
        params['num_classes'] = 28
    else:
        raise TypeError

    save_paths = [args.model1_path, args.model2_path,
                  args.model3_path, args.model4_path,
                  args.model5_path, args.model6_path,
                  args.model7_path, args.model8_path]

    do_eval(save_paths=save_paths, params=params,
            beam_width=args.beam_width,
            temperature_infer=args.temperature_infer,
            result_save_path=args.result_save_path)


if __name__ == '__main__':

    args = sys.argv
    main()
