#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest
from tqdm import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.data.sparsetensor import list2sparsetensor
from read_dataset_multitask_ctc import DataSet


class TestReadDatasetMultitaskCTC(unittest.TestCase):

    def test(self):
        dataset = DataSet(data_type='train', label_type='phone61',
                          num_stack=3, num_skip=3,
                          is_sorted=True, is_progressbar=True)

        print('=> Reading mini-batch...')
        map_file_path_char = '../evaluation/mapping_files/ctc/char2num.txt'
        map_file_path_phone = '../evaluation/mapping_files/ctc/phone2num_61.txt'
        for i in tqdm(range(20000)):
            inputs, labels_char, labels_phone, seq_len, input_names = dataset.next_batch(
                batch_size=64)

            # str_true_char = num2char(labels_char[0], map_file_path_char)
            # str_true_char = re.sub(r'_', ' ', str_true_char)
            # str_true_phone = num2phone(labels_phone[0], map_file_path_phone)
            # print(str_true_char)
            # print(str_true_phone)
            # print('-----')


if __name__ == '__main__':
    unittest.main()
