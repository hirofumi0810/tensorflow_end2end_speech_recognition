#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest

sys.path.append('../../../../')
from experiments.timit.data.load_dataset_ctc import Dataset
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetCTC(unittest.TestCase):

    def test(self):

        # label_type
        self.check_loading(label_type='character',
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='character_capital_divide',
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='phone61',
                           sort_utt=False, sorta_grad=False)

        # sort
        self.check_loading(label_type='phone61',
                           sort_utt=True, sorta_grad=False)
        self.check_loading(label_type='phone61',
                           sort_utt=False, sorta_grad=True)

    @measure_time
    def check_loading(self, label_type, sort_utt, sorta_grad):
        print('----- label_type: %s, sort_utt: %s, sorta_grad: %s -----' %
              (label_type, str(sort_utt), str(sorta_grad)))

        dataset = Dataset(
            data_type='dev', label_type=label_type,
            batch_size=64, num_stack=3, num_skip=3,
            sort_utt=sort_utt, sorta_grad=sorta_grad, progressbar=True)

        print('=> Loading mini-batch...')
        map_file_path = '../../metrics/mapping_files/ctc/' + label_type + '_to_num.txt'
        if label_type in ['character', 'character_capital_divide']:
            map_fn = num2char
        else:
            map_fn = num2phone

        for data, next_epoch_flag in dataset():
            inputs, labels, inputs_seq_len, input_names = data

            str_true = map_fn(labels[0], map_file_path)
            str_true = re.sub(r'_', ' ', str_true)
            print('----- %s -----' % input_names[0])
            print(str_true)

            if next_epoch_flag:
                break


if __name__ == '__main__':
    unittest.main()
