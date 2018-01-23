#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate the ensemble of 4 CTC models (Librispeech corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse
import re
from tqdm import tqdm
import numpy as np

sys.path.append(abspath('../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
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

parser.add_argument('--epoch_model1', type=int, default=-1,
                    help='the epoch of 1st model to restore')
parser.add_argument('--epoch_model2', type=int, default=-1,
                    help='the epoch of 2nd model to restore')
parser.add_argument('--epoch_model3', type=int, default=-1,
                    help='the epoch of 3rd model to restore')
parser.add_argument('--epoch_model4', type=int, default=-1,
                    help='the epoch of 4th model to restore')

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
                               '4models_traintemp' + str(temperature_train) +
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


def do_eval_cer(save_paths, dataset, data_type, label_type, num_classes,
                beam_width, temperature_infer,
                is_test=False, progressbar=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        save_paths (list):
        dataset: An instance of a `Dataset` class
        data_type (string):
        label_type (string): character
        num_classes (int):
        beam_width (int): the size of beam
        temperature (int): temperature in the inference stage
        is_test (bool, optional): set to True when evaluating by the test set
        progressbar (bool, optional): if True, visualize the progressbar
    Return:
        cer_mean (float): An average of CER
        wer_mean (float): An average of WER
    """
    if label_type == 'character':
        idx2char = Idx2char(
            map_file_path='../metrics/mapping_files/character.txt')
        char2idx = Char2idx(
            map_file_path='../metrics/mapping_files/character.txt')
    else:
        raise TypeError

    # Define decoder
    decoder = BeamSearchDecoder(space_index=char2idx(str_char='_')[0],
                                blank_index=num_classes - 1)

    ##################################################
    # Compute mean probabilities
    ##################################################
    if progressbar:
        pbar = tqdm(total=len(dataset))
    cer_mean, wer_mean = 0, 0
    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels_true, inputs_seq_len, input_names = data

        batch_size = inputs[0].shape[0]
        for i_batch in range(batch_size):
            probs_ensemble = None
            for i_model in range(len(save_paths)):

                # Load posteriors
                speaker = input_names[0][i_batch].split('-')[0]
                prob_save_path = join(
                    save_paths[i_model], 'temp' + str(temperature_infer),
                    data_type, 'probs_utt', speaker,
                    input_names[0][i_batch] + '.npy')
                probs_model_i = np.load(prob_save_path)
                # NOTE: probs_model_i: `[T, num_classes]`

                # Sum over probs
                if probs_ensemble is None:
                    probs_ensemble = probs_model_i
                else:
                    probs_ensemble += probs_model_i

            # Compute mean posteriors
            probs_ensemble /= len(save_paths)

            # Decode per utterance
            labels_pred, scores = decoder(
                probs=probs_ensemble[np.newaxis, :, :],
                seq_len=inputs_seq_len[0][i_batch: i_batch + 1],
                beam_width=beam_width)

            # Convert from list of index to string
            if is_test:
                str_true = labels_true[0][i_batch][0]
                # NOTE: transcript is seperated by space('_')
            else:
                str_true = idx2char(labels_true[0][i_batch],
                                    padded_value=dataset.padded_value)
            str_pred = idx2char(labels_pred[0])

            # Remove consecutive spaces
            str_pred = re.sub(r'[_]+', '_', str_pred)

            # Remove garbage labels
            str_true = re.sub(r'[\']+', '', str_true)
            str_pred = re.sub(r'[\']+', '', str_pred)

            # Compute WER
            wer_mean += compute_wer(ref=str_pred.split('_'),
                                    hyp=str_true.split('_'),
                                    normalize=True)
            # substitute, insert, delete = wer_align(
            #     ref=str_true.split('_'),
            #     hyp=str_pred.split('_'))
            # print('SUB: %d' % substitute)
            # print('INS: %d' % insert)
            # print('DEL: %d' % delete)

            # Remove spaces
            str_true = re.sub(r'[_]+', '', str_true)
            str_pred = re.sub(r'[_]+', '', str_pred)

            # Compute CER
            cer_mean += compute_cer(str_pred=str_pred,
                                    str_true=str_true,
                                    normalize=True)

            if progressbar:
                pbar.update(1)

        if is_new_epoch:
            break

    cer_mean /= (len(dataset))
    wer_mean /= (len(dataset))
    # TODO: Fix this

    return cer_mean, wer_mean


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
                  args.model3_path, args.model4_path]

    do_eval(save_paths=save_paths, params=params,
            beam_width=args.beam_width,
            temperature_infer=args.temperature_infer,
            result_save_path=args.result_save_path)


if __name__ == '__main__':

    args = sys.argv
    main()
