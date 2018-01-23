#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import codecs
from glob import glob
import numpy as np
import Levenshtein

sys.path.append('../../../')
from experiments.utils.labels.character import kana2num, num2char
from experiments.csj.data.load_dataset_ctc import Dataset
from experiments.utils.progressbar import wrap_iterator


def main():

    # Make mapping dictionary from kana to phone
    phone2kana_dict = {}
    with open('../metrics/mapping_files/kana2phone.txt', 'r') as f:
        for line in f:
            line = line.strip().split('+')
            kana, phone_seq = line
            phone = re.sub(' ', '', phone_seq)
            if phone in phone2kana_dict.keys():
                continue
            phone2kana_dict[phone] = kana
            phone2kana_dict[phone + ':'] = kana + 'ー'

    # Julius Results
    for data_type in ['eval1', 'eval2', 'eval3']:
        results_paths = [path for path in glob(
            '/home/lab5/inaguma/asru2017/csj_results_0710_kana/' + data_type + '/*.kana')]
        result_dict = {}

        for path in results_paths:
            with codecs.open(path, 'r', 'euc_jp') as f:
                file_name = ''
                output = ''
                for line in f:
                    line = line.strip()

                    if 'wav' in line:
                        file_name = line.split(': ')[-1]
                        file_name = '_'.join(line.split('/')[-2:])
                        file_name = re.sub('.wav', '', file_name)

                    else:
                        output = line
                        output = re.sub('sp', '', output)
                        result_dict[file_name] = output

        label_type = 'kana'
        dataset = Dataset(data_type=data_type,
                          label_type=label_type,
                          batch_size=1,
                          train_data_size='large',
                          is_sorted=False,
                          is_progressbar=True,
                          is_gpu=False)

        num_examples = dataset.data_num
        cer_sum = 0

        mini_batch = dataset.next_batch(batch_size=1)

        def map_fn(phone): return phone2kana_dict[phone]

        for _ in wrap_iterator(range(num_examples), False):
            # Create feed dictionary for next mini batch
            _, labels_true, _, input_names = mini_batch.__next__()

            if input_names[0] not in result_dict.keys():
                continue

            output = result_dict[input_names[0]].split(' ')
            while '' in output:
                output.remove('')

            str_pred = ''.join(list(map(map_fn, output)))

            if input_names[0] in ['A03M0106_0057', 'A03M0016_0014']:
                print(str_pred)
                print(labels_true[0])
                print('-----')

            # Remove silence(_) & noise(NZ) labels
            str_true = re.sub(r'[_NZー・]+', "", labels_true[0])
            str_pred = re.sub(r'[_NZー・]+', "", str_pred)

            # Compute edit distance
            cer_each = Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))

            cer_sum += cer_each

        print('CER (' + data_type + '): %f' % (cer_sum / dataset.data_num))


if __name__ == '__main__':
    main()
