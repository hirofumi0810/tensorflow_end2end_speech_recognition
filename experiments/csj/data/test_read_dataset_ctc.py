#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest
import tensorflow as tf

sys.path.append('../../')
sys.path.append('../../../')
from read_dataset_ctc import DataSet
from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.sparsetensor import sparsetensor2list


class TestReadDatasetCTC(unittest.TestCase):

    def test(self):
        self.check_reading(label_type='kanji', num_gpu=1, is_sorted=True)
        self.check_reading(label_type='kanji', num_gpu=1, is_sorted=False)
        self.check_reading(label_type='kanji', num_gpu=2, is_sorted=True)
        self.check_reading(label_type='kanji', num_gpu=2, is_sorted=False)
        self.check_reading(label_type='character', num_gpu=1, is_sorted=True)
        self.check_reading(label_type='character', num_gpu=1, is_sorted=False)
        self.check_reading(label_type='character', num_gpu=2, is_sorted=True)
        self.check_reading(label_type='character', num_gpu=2, is_sorted=False)
        self.check_reading(label_type='phone', num_gpu=1, is_sorted=True)
        self.check_reading(label_type='phone', num_gpu=1, is_sorted=False)
        self.check_reading(label_type='phone', num_gpu=2, is_sorted=True)
        self.check_reading(label_type='phone', num_gpu=2, is_sorted=False)

        # For many GPUs
        self.check_reading(label_type='kanji', num_gpu=7, is_sorted=True)

    def check_reading(self, label_type, num_gpu, is_sorted):
        print('----- label_type: ' + label_type + ', num_gpu: ' +
              str(num_gpu) + ', is_sorted: ' + str(is_sorted) + ' -----')

        batch_size = 64
        dataset = DataSet(data_type='train', train_data_size='default',
                          label_type=label_type, batch_size=batch_size,
                          num_stack=3, num_skip=3,
                          is_sorted=is_sorted, is_progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Reading mini-batch...')
            if label_type == 'kanji':
                map_file_path = '../metric/mapping_files/ctc/kanji2num.txt'
                map_fn = num2char
            elif label_type == 'character':
                map_file_path = '../metric/mapping_files/ctc/char2num.txt'
                map_fn = num2char
            else:
                map_file_path = '../metric/mapping_files/ctc/phone2num.txt'
                map_fn = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                inputs, labels_st, inputs_seq_len, input_names = mini_batch.__next__()

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                        print(inputs_gpu.shape)
                    inputs = inputs[0]
                    labels_st = labels_st[0]

                labels = sparsetensor2list(
                    labels_st, batch_size=len(labels_st))
                str_true = map_fn(labels[0], map_file_path)
                str_true = re.sub(r'_', ' ', str_true)
                print(str_true)


if __name__ == '__main__':
    unittest.main()
