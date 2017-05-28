#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import unittest
import tqdm

sys.path.append('../../')
sys.path.append('../../../')
from utils.data.sparsetensor import list2sparsetensor
from read_dataset_ctc import DataSet


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):
        dataset = DataSet(data_type='train', train_data_size='default',
                          label_type='character',
                          num_stack=3, num_skip=3,
                          is_sorted=True, is_progressbar=True)

        print('=> Reading mini-batch...')
        for i in tqdm(range(20000)):
            inputs, labels, seq_len, input_names = dataset.next_batch(
                batch_size=64)
            indices, values, shape = list2sparsetensor(labels)


if __name__ == '__main__':
    unittest.main()
