#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath('../../../../'))
from experiments.librispeech.data.load_dataset_ctc import Dataset
from utils.io.labels.character import idx2char
from utils.io.labels.word import idx2word
from utils.measure_time_func import measure_time


class TestLoadDatasetCTC(unittest.TestCase):

    def test(self):

        self.length_check = False

        # data_type
        self.check_loading(label_type='word', data_type='train_clean100')
        self.check_loading(label_type='word', data_type='dev_clean')
        self.check_loading(label_type='word', data_type='test_clean')
        self.check_loading(label_type='character', data_type='test_clean')

        # label_type
        self.check_loading(label_type='character_capital_divide')
        self.check_loading(label_type='word')

        # sort
        self.check_loading(label_type='character', sort_utt=True)
        self.check_loading(label_type='character', sort_utt=True,
                           sort_stop_epoch=2)
        self.check_loading(label_type='character', shuffle=True)

        # frame stacking
        self.check_loading(label_type='character', frame_stacking=True)

        # splicing
        self.check_loading(label_type='character', splice=11)

        # multi-GPU
        self.check_loading(label_type='character', num_gpu=8)

    @measure_time
    def check_loading(self, label_type, data_type='dev_clean',
                      shuffle=False,  sort_utt=False, sort_stop_epoch=None,
                      frame_stacking=False, splice=1, num_gpu=1):

        print('========================================')
        print('  label_type: %s' % label_type)
        print('  data_type: %s' % data_type)
        print('  shuffle: %s' % str(shuffle))
        print('  sort_utt: %s' % str(sort_utt))
        print('  sort_stop_epoch: %s' % str(sort_stop_epoch))
        print('  frame_stacking: %s' % str(frame_stacking))
        print('  splice: %d' % splice)
        print('  num_gpu: %d' % num_gpu)
        print('========================================')

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type=data_type, train_data_size='train_clean100',
            label_type=label_type,
            batch_size=64, max_epoch=1, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            shuffle=shuffle, sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True, num_gpu=num_gpu, is_gpu=False)

        print('=> Loading mini-batch...')
        if label_type == 'character':
            map_file_path = '../../metrics/mapping_files/ctc/character.txt'
        elif label_type == 'character_capital_divide':
            map_file_path = '../../metrics/mapping_files/ctc/character_capital.txt'
        elif label_type == 'word':
            map_file_path = '../../metrics/mapping_files/ctc/word_' + \
                dataset.train_data_size + '.txt'

        for data, is_new_epoch in dataset:
            inputs, labels, inputs_seq_len, input_names = data

            if not self.length_check:
                for i, l in zip(inputs[0], labels[0]):
                    if len(i) < len(l):
                        raise ValueError(
                            'input length must be longer than label length.')
                self.length_check = True

            if num_gpu > 1:
                for inputs_gpu in inputs:
                    print(inputs_gpu.shape)

            if label_type == 'word':
                if 'test' not in data_type:
                    str_true = ' '.join(idx2word(labels[0][0], map_file_path))
                else:
                    word_list = np.delete(labels[0][0], np.where(
                        labels[0][0] == None), axis=0)
                    str_true = ' '.join(word_list)
            else:
                str_true = ''.join(idx2char(labels[0][0], map_file_path))
            str_true = re.sub(r'_', ' ', str_true)
            print('----- %s (epoch: %.3f) -----' %
                  (input_names[0][0], dataset.epoch_detail))
            print(inputs[0].shape)
            print(str_true)

            if dataset.epoch_detail >= 0.05:
                break


if __name__ == '__main__':
    unittest.main()
