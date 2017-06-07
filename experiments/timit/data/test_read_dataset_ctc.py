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
from utils.data.sparsetensor import list2sparsetensor
from read_dataset_ctc import DataSet


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):
        dataset = DataSet(data_type='dev', label_type='character',
                          num_stack=3, num_skip=3,
                          is_sorted=True, is_progressbar=True)

        print('=> Reading mini-batch...')
        map_file_path = '../evaluation/mapping_files/ctc/char2num.txt'
        for i in tqdm(range(20000)):
            inputs, labels, seq_len, input_names = dataset.next_batch(
                batch_size=64)
            indices, values, dense_shape = list2sparsetensor(labels)
            str_true = num2char(labels[0], map_file_path)
            str_true = re.sub(r'_', ' ', str_true)
            print(str_true)


if __name__ == '__main__':
    unittest.main()
