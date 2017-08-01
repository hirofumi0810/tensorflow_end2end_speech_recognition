#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import unittest

sys.path.append('../../../../')
from experiments.timit.data.load_dataset_joint_ctc_attention import Dataset
from experiments.utils.data.labels.character import num2char
from experiments.utils.data.labels.phone import num2phone
from experiments.utils.measure_time_func import measure_time


class TestLoadDatasetJointCTCAttention(unittest.TestCase):

    def test(self):

        # label_type
        self.check_loading(label_type='character', sort_utt=False)
        self.check_loading(label_type='character_capital_divide',
                           sort_utt=False)
        self.check_loading(label_type='phone61', sort_utt=False)

        # sort
        self.check_loading(label_type='phone61', sort_utt=True)
        self.check_loading(label_type='phone61', sort_utt=True,
                           sort_stop_epoch=2)

    @measure_time
    def check_loading(self, label_type, sort_utt, sort_stop_epoch=None):
        print('----- label_type: %s, sort_utt: %s, sort_stop_epoch: %s -----' %
              (label_type, str(sort_utt), str(sort_stop_epoch)))

        dataset = Dataset(
            data_type='dev', label_type=label_type,
            batch_size=64, eos_index=1,
            sort_utt=sort_utt, sort_stop_epoch=sort_stop_epoch,
            progressbar=True)

        print('=> Loading mini-batch...')
        ctc_map_file_path = '../../metrics/mapping_files/ctc/' + label_type + '_to_num.txt'
        att_map_file_path = '../../metrics/mapping_files/attention/' + \
            label_type + '_to_num.txt'
        if label_type in ['character', 'character_capital_divide']:
            map_fn = num2char
        else:
            map_fn = num2phone

        for data, next_epoch_flag in dataset():
            inputs, att_labels, ctc_labels, inputs_seq_len, att_labels_seq_len, input_names = data

            att_str_true = map_fn(
                att_labels[0][0: att_labels_seq_len[0]], att_map_file_path)
            ctc_str_true = map_fn(ctc_labels[0], ctc_map_file_path)
            att_str_true = re.sub(r'_', ' ', att_str_true)
            ctc_str_true = re.sub(r'_', ' ', ctc_str_true)
            print('----- %s -----' % input_names[0])
            print(att_str_true)
            print(ctc_str_true)

            if next_epoch_flag:
                break


if __name__ == '__main__':
    unittest.main()
