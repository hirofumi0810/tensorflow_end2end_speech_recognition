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
import pandas as pd

sys.path.append('../../../')
from utils.io.labels.character import Char2idx
from experiments.csj.data.load_dataset_ctc_ss import Dataset


def main():

    # Julius Results
    mean_f_f = 0
    mean_f_d = 0
    for data_type in ['eval1', 'eval2', 'eval3']:
        results_paths = [path for path in glob(
            '/home/lab5/inaguma/asru2017/csj_results_0710/' + data_type + '/*.log')]
        result_dict = {}

        for path in results_paths:
            with codecs.open(path, 'r', 'euc_jp') as f:
                start_flag = False
                file_name = ''
                output, output_pos = '', ''
                for line in f:
                    line = line.strip()
                    if line == '----------------------- System Information end -----------------------':
                        start_flag = True
                    if start_flag:
                        if 'input MFCC file' in line:
                            file_name = line.split(': ')[-1]
                            if data_type != 'dialog':
                                file_name = '_'.join(file_name.split('/')[-2:])
                            else:
                                file_name = os.path.basename(file_name)
                            file_name = re.sub('.wav', '', file_name)
                            file_name = re.sub('.htk', '', file_name)

                        if 'sentence1' in line:
                            output = line.split(': ')[-1]
                            output = re.sub('<s>', '', output)
                            output = re.sub('</s>', '', output)
                            output = re.sub('<sp>', '', output)
                            output = re.sub(r'[\sー]', '', output)

                        if 'wseq1' in line:
                            output_pos = line.split(': ')[-1]
                            output_pos = re.sub('<s>', '', output_pos)
                            output_pos = re.sub('</s>', '', output_pos)
                            output_pos = re.sub('<sp>', '', output_pos)
                            output_pos = re.sub('感動詞', 'f', output_pos)
                            output_pos = re.sub('言いよどみ', 'd', output_pos)
                            result_dict[file_name] = [output, output_pos[1:]]

        label_type = 'kana'
        dataset = Dataset(data_type=data_type,
                          label_type=label_type,
                          ss_type='insert_left',
                          batch_size=1,
                          max_epoch=1,
                          train_data_size='train_subset',
                          shuffle=False)

        tp_f, fp_f, fn_f = 0., 0., 0.
        tp_d, fp_d, fn_d = 0., 0., 0.

        for data, is_new_epoch in dataset:

            # Create feed dictionary for next mini batch
            inputs, labels_true, inputs_seq_len, input_names = data

            if input_names[0][0] not in result_dict.keys():
                continue

            output, output_pos = result_dict[input_names[0][0]]

            detected_f_num = output_pos.count('f')
            detected_d_num = output_pos.count('d')

            # if detected_d_num != 0:
            #     print(output_pos)
            #     print(output)
            #     str_true = labels_true[0][0][0]
            #     print(str_true)
            #     print('-----')

            true_f_num = np.sum(labels_true[0][0][0].count('f'))
            true_d_num = np.sum(labels_true[0][0][0].count('d'))

            # Filler
            if detected_f_num <= true_f_num:
                tp_f += detected_f_num
                fn_f += true_f_num - detected_f_num
            else:
                tp_f += true_f_num
                fp_f += detected_f_num - true_f_num

            # Disfluency
            if detected_d_num <= true_d_num:
                tp_d += detected_d_num
                fn_d += true_d_num - detected_d_num
            else:
                tp_d += true_d_num
                fp_d += detected_d_num - true_d_num

        r_f = tp_f / (tp_f + fn_f) if (tp_f + fn_f) != 0 else 0
        p_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) != 0 else 0
        f_f = 2 * r_f * p_f / (r_f + p_f) if (r_f + p_f) != 0 else 0
        if data_type != 'eval3':
            mean_f_f += f_f

        r_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) != 0 else 0
        p_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) != 0 else 0
        f_d = 2 * r_d * p_d / (r_d + p_d) if (r_d + p_d) != 0 else 0
        if data_type != 'eval3':
            mean_f_d += f_d

        acc_f = [p_f, r_f, f_f]
        acc_d = [p_d, r_d, f_d]

        df_acc = pd.DataFrame({'Filler': acc_f, 'Disfluency': acc_d},
                              columns=['Filler', 'Disfluency'],
                              index=['Precision', 'Recall', 'F-measure'])
        print(df_acc)

    print('Mean F-measure (filler): %f' % (mean_f_f / 2.))
    print('Mean F-measure (disfluency): %f' % (mean_f_d / 2.))


if __name__ == '__main__':
    main()
