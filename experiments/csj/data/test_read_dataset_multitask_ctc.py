#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest
from tqdm import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from utils.data.sparsetensor import list2sparsetensor
from utils.labels.character import num2char
from read_dataset_multitask_ctc import DataSet


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):
        dataset = DataSet(data_type='eval1', train_data_size='large',
                          label_type_main='kanji',
                          label_type_second='character',
                          num_stack=3, num_skip=3,
                          is_sorted=True, is_progressbar=True)

        print('=> Reading mini-batch...')
        map_file_path = '../evaluation/mapping_files/ctc/char2num.txt'
        for i in tqdm(range(20000)):
            inputs, labels_main, labels_second, seq_len, input_names = dataset.next_batch(
                batch_size=64)
            print(labels_main[0])
            print(num2char(labels_second[0], map_file_path))


if __name__ == '__main__':
    unittest.main()
