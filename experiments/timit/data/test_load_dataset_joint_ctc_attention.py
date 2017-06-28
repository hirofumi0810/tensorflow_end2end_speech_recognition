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
from load_dataset_joint_ctc_attention import Dataset
from utils.labels.character import num2char
from utils.labels.phone import num2phone
from utils.sparsetensor import sparsetensor2list


class TestLoadDatasetJointCTCAttention(unittest.TestCase):

    def test(self):
        self.check_loading(label_type='character', num_gpu=1, is_sorted=True)
        self.check_loading(label_type='character', num_gpu=1, is_sorted=False)
        self.check_loading(label_type='phone61', num_gpu=1, is_sorted=True)
        self.check_loading(label_type='phone61', num_gpu=1, is_sorted=False)

        # self.check_loading(label_type='character', num_gpu=2, is_sorted=True)
        # self.check_loading(label_type='character', num_gpu=2, is_sorted=False)
        # self.check_loading(label_type='phone61', num_gpu=2, is_sorted=True)
        # self.check_loading(label_type='phone61', num_gpu=2, is_sorted=False)

        # For many GPUs
        # self.check_loading(label_type='character', num_gpu=7, is_sorted=True)

    def check_loading(self, label_type, num_gpu, is_sorted):
        print('----- label_type: ' + label_type + ', num_gpu: ' +
              str(num_gpu) + ', is_sorted: ' + str(is_sorted) + ' -----')

        batch_size = 64
        eos_index = 2 if label_type == 'character' else 1
        dataset = Dataset(data_type='train', label_type=label_type,
                          batch_size=batch_size, eos_index=eos_index,
                          is_sorted=is_sorted, is_progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            if label_type == 'character':
                att_map_file_path = '../metrics/mapping_files/attention/char2num.txt'
                ctc_map_file_path = '../metrics/mapping_files/ctc/char2num.txt'
                map_fn = num2char
            else:
                att_map_file_path = '../metrics/mapping_files/attention/phone2num_' + \
                    label_type[5:7] + '.txt'
                ctc_map_file_path = '../metrics/mapping_files/ctc/phone2num_' + \
                    label_type[5:7] + '.txt'
                map_fn = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                return_tuple = mini_batch.__next__()
                inputs = return_tuple[0]
                att_labels = return_tuple[1]
                ctc_labels_st = return_tuple[2]
                att_labels_seq_len = return_tuple[4]

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                        print(inputs_gpu.shape)
                    inputs = inputs[0]
                    att_labels = att_labels[0]
                    ctc_labels_st = ctc_labels_st[0]
                    att_labels_seq_len = att_labels_seq_len[0]

                ctc_labels = sparsetensor2list(
                    ctc_labels_st, batch_size=len(inputs))

                if num_gpu == 1:
                    for inputs_i, labels_i in zip(inputs, ctc_labels):
                        if len(inputs_i) < len(labels_i):
                            print(len(inputs_i))
                            print(len(labels_i))
                            raise ValueError

                att_str_true = map_fn(
                    att_labels[0][0: att_labels_seq_len[0]], att_map_file_path)
                ctc_str_true = map_fn(ctc_labels[0], ctc_map_file_path)
                att_str_true = re.sub(r'_', ' ', att_str_true)
                ctc_str_true = re.sub(r'_', ' ', ctc_str_true)
                print(att_str_true)
                print(ctc_str_true)
                print('-----')


if __name__ == '__main__':
    unittest.main()
