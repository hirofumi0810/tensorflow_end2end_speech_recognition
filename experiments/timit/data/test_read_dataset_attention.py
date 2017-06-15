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
from read_dataset_attention import DataSet


class TestReadDatasetAttention(unittest.TestCase):
    def test(self):
        dataset = DataSet(data_type='dev', label_type='character', eos_index=32,
                          is_sorted=True, is_progressbar=True)

        print('=> Reading mini-batch...')
        map_file_path = '../metric/mapping_files/attention/char2num.txt'
        for i in tqdm(range(200)):
            inputs, labels, inputs_seq_len, labels_seq_len, input_names = dataset.next_batch(
                batch_size=64)

            str_true = num2char(labels[0], map_file_path)
            str_true = re.sub(r'_', ' ', str_true)
            print(str_true)


if __name__ == '__main__':
    unittest.main()
