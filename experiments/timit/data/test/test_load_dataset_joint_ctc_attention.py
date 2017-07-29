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
        self.check_loading(label_type='character', num_gpu=1,
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='character_capital_divide', num_gpu=1,
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='phone61', num_gpu=1,
                           sort_utt=False, sorta_grad=False)

        # sort
        self.check_loading(label_type='phone61', num_gpu=1,
                           sort_utt=True, sorta_grad=False)
        self.check_loading(label_type='phone61', num_gpu=1,
                           sort_utt=False, sorta_grad=True)

        # multi-GPU
        self.check_loading(label_type='phone61', num_gpu=2,
                           sort_utt=False, sorta_grad=False)
        self.check_loading(label_type='phone61', num_gpu=7,
                           sort_utt=False, sorta_grad=False)

    @measure_time
    def check_loading(self, label_type, num_gpu, sort_utt, sorta_grad):
        print('----- label_type: %s, num_gpu: %d, sort_utt: %s, sorta_grad: %s -----' %
              (label_type, num_gpu, str(sort_utt), str(sorta_grad)))

        dataset = Dataset(data_type='dev', label_type=label_type,
                          batch_size=64, eos_index=1,
                          sort_utt=sort_utt, sorta_grad=sorta_grad,
                          progressbar=True, num_gpu=num_gpu)

        tf.reset_default_graph()
        with tf.Session().as_default() as sess:
            print('=> Loading mini-batch...')
            if label_type == 'character_capital_divide':
                ctc_map_file_path = '../../metrics/mapping_files/ctc/character_to_num_capital.txt'
                att_map_file_path = '../../metrics/mapping_files/attention/character_to_num_capital.txt'
            else:
                ctc_map_file_path = '../../metrics/mapping_files/ctc/' + label_type + '_to_num.txt'
                att_map_file_path = '../../metrics/mapping_files/attention/' + \
                    label_type + '_to_num.txt'
            if label_type in ['character', 'character_capital_divide']:
                map_fn = num2char
            else:
                map_fn = num2phone

            for data, next_epoch_flag in dataset(session=sess):
                inputs, att_labels, ctc_labels, _, att_labels_seq_len, _ = data
                if num_gpu > 1:
                    # for inputs_gpu in inputs:
                    #     print(inputs_gpu.shape)
                    inputs = inputs[0]
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

                if next_epoch_flag:
                    break


if __name__ == '__main__':
    unittest.main()
