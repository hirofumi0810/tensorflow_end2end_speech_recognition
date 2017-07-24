#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest
import tensorflow as tf

sys.path.append('../../../../')
from experiments.timit.data.load_dataset_joint_ctc_attention import Dataset
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.data.sparsetensor import sparsetensor2list
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetJointCTCAttention(unittest.TestCase):

    def test(self):

        # label_type
        self.check_loading(label_type='character', num_gpu=1, sort_utt=False)
        self.check_loading(label_type='phone61', num_gpu=1, sort_utt=False)

        # sort
        self.check_loading(label_type='phone61', num_gpu=2, sort_utt=False)

        # multi-GPU
        self.check_loading(label_type='phone61', num_gpu=2, sort_utt=True)
        self.check_loading(label_type='phone61', num_gpu=7, sort_utt=True)

    def check_loading(self, label_type, num_gpu, sort_utt):
        print('----- label_type: ' + label_type + ', num_gpu: ' +
              str(num_gpu) + ', sort_utt: ' + str(sort_utt) + ' -----')

        batch_size = 64
        eos_index = 2 if label_type == 'character' else 1
        dataset = Dataset(data_type='train', label_type=label_type,
                          batch_size=batch_size, eos_index=eos_index,
                          sort_utt=sort_utt, progressbar=True,
                          num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            ctc_map_file_path = '../../metrics/mapping_files/ctc/' + label_type + '_to_num.txt'
            att_map_file_path = '../../metrics/mapping_files/attention/' + \
                label_type + '_to_num.txt'
            if label_type == 'character':
                map_fn = num2char
            else:
                map_fn = num2phone

            mini_batch = dataset.next_batch(session=sess)

            iter_per_epoch = int(dataset.data_num /
                                 (batch_size * num_gpu)) + 1
            for i in range(iter_per_epoch + 1):
                return_tuple = mini_batch.__next__()
                inputs = return_tuple[0]
                att_labels = return_tuple[1]
                ctc_labels = return_tuple[2]
                att_labels_seq_len = return_tuple[4]

                if num_gpu > 1:
                    for inputs_gpu in inputs:
                    #     print(inputs_gpu.shape)
                    # inputs = inputs[0]
                    att_labels = att_labels[0]
                    ctc_labels = ctc_labels[0]
                    att_labels_seq_len = att_labels_seq_len[0]

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
