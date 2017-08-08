#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest

sys.path.append('../../../../')
from experiments.timit.data.load_dataset_multitask_ctc import Dataset
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetMultitaskCTC(unittest.TestCase):

    def test(self):

        # label_type
        self.check_loading(label_type_main='character')
        self.check_loading(label_type_main='character_capital_divide')

        # sort
        self.check_loading(label_type_main='character', sort_utt=True)
        self.check_loading(label_type_main='character', sort_utt=True,
                           sort_stop_epoch=2)

        # frame stacking
        self.check_loading(label_type_main='character', frame_stacking=True)

        # splicing
        self.check_loading(label_type_main='character', splice=11)

    @measure_time
    def check_loading(self, label_type_main, sort_utt=False,
                      sort_stop_epoch=None, frame_stacking=False, splice=1):
        print('----- label_type_main: %s, sort_utt: %s, sort_stop_epoch: %s, frame_stacking: %s, splice: %d -----' %
              (label_type_main, str(sort_utt), str(sort_stop_epoch), str(frame_stacking), splice))

        num_stack = 3 if frame_stacking else 1
        num_skip = 3 if frame_stacking else 1
        dataset = Dataset(
            data_type='dev',
            label_type_main=label_type_main, label_type_sub='phone61',
            batch_size=64, splice=splice,
            num_stack=num_stack, num_skip=num_skip,
            sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True)

        print('=> Loading mini-batch...')
        map_file_path_char = '../../metrics/mapping_files/ctc/' + \
            label_type_main + '_to_num.txt'
        map_file_path_phone = '../../metrics/mapping_files/ctc/phone61_to_num.txt'

        for data, next_epoch_flag in dataset():
            inputs, labels_char, labels_phone, inputs_seq_len, input_names = data

            str_true_char = num2char(labels_char[0], map_file_path_char)
            str_true_char = re.sub(r'_', ' ', str_true_char)
            str_true_phone = num2phone(
                labels_phone[0], map_file_path_phone)
            print('----- %s -----' % input_names[0])
            print(str_true_char)
            print(str_true_phone)

            if next_epoch_flag:
                break


if __name__ == '__main__':
    unittest.main()
