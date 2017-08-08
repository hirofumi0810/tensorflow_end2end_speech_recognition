#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest

sys.path.append('../../../../')
from experiments.librispeech.data.load_dataset_multitask_ctc import Dataset
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.word import num2word
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetMultitaskCTC(unittest.TestCase):

    def test(self):

        # label_type
        self.check_loading(label_type_sub='character')
        self.check_loading(label_type_sub='character_capital_divide')

        # sort
        self.check_loading(label_type_sub='character', sort_utt=True)

        # frame_stacking
        self.check_loading(label_type_sub='character', frame_stacking=True)

        # multi-GPU
        self.check_loading(label_type_sub='character', num_gpu=2)
        self.check_loading(label_type_sub='character', num_gpu=8)

    @measure_time
    def check_loading(self, label_type_sub, num_gpu=1, sort_utt=False,
                      sort_stop_epoch=None, frame_stacking=False):
        print('----- label_type_sub: %s, num_gpu: %d, sort_utt: %s, sort_stop_epoch: %s, frame_stacking: %s -----' %
              (label_type_sub, num_gpu, str(sort_utt), str(sort_stop_epoch), str(frame_stacking)))

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type='dev_clean', train_data_size='train_clean100',
            label_type_main='word', label_type_sub=label_type_sub,
            batch_size=64, num_stack=num_stack, num_skip=num_skip,
            sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True, num_gpu=num_gpu, is_gpu=False)

        print('=> Loading mini-batch...')
        if label_type_sub == 'character':
            map_file_path_char = '../../metrics/mapping_files/ctc/character2num.txt'
        elif label_type_sub == 'character_capital_divide':
            map_file_path_char = '../../metrics/mapping_files/ctc/character2num_capital.txt'
        map_file_path_word = '../../metrics/mapping_files/ctc/word2num_train_clean100.txt'

        for data, next_epoch_flag in dataset():
            inputs, labels_word, labels_char,  inputs_seq_len, input_names = data

            for inputs_gpu in inputs:
                print(inputs_gpu.shape)

            word_list = num2word(labels_word[0][0], map_file_path_word)
            str_true_word = ' '.join(word_list)
            str_true_char = num2char(labels_char[0][0], map_file_path_char)
            str_true_char = re.sub(r'_', ' ', str_true_char)
            print('----- %s -----' % input_names[0][0])
            print(str_true_word)
            print(str_true_char)

            if next_epoch_flag:
                break


if __name__ == '__main__':
    unittest.main()
