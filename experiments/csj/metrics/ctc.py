#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Define evaluation method for the CTC model (CSJ corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import Levenshtein

from experiments.utils.data.labels.character import num2char
from experiments.utils.data.sparsetensor import sparsetensor2list
from experiments.utils.progressbar import wrap_iterator


def do_eval_cer(session, decode_op, network, dataset, label_type, is_test=None,
                eval_batch_size=None, progressbar=False,
                is_multitask=False, is_main=False):
    """Evaluate trained model by Character Error Rate.
    Args:
        session: session of training model
        decode_op: operation for decoding
        network: network to evaluate
        dataset: An instance of `Dataset` class
        label_type: string, kanji or kana or phone
        is_test: bool, set to True when evaluating by the test set
        eval_batch_size: int, the batch size when evaluating the model
        progressbar: if True, visualize progressbar
        is_multitask: if True, evaluate the multitask model
        is_main: if True, evaluate the main task
    Return:
        cer_mean: An average of CER
    """
    if eval_batch_size is None:
        batch_size = dataset.batch_size
    else:
        batch_size = eval_batch_size

    num_examples = dataset.data_num
    iteration = int(num_examples / batch_size)
    if (num_examples / batch_size) != int(num_examples / batch_size):
        iteration += 1
    cer_sum = 0

    # Make data generator
    mini_batch = dataset.next_batch(batch_size=batch_size)

    if label_type == 'kanji':
        map_file_path = '../metrics/mapping_files/ctc/kanji2num.txt'
    elif label_type == 'kana':
        map_file_path = '../metrics/mapping_files/ctc/kana2num.txt'
    elif label_type == 'phone':
        map_file_path == '../metrics/mapping_files/ctc/phone2num.txt'

    for step in wrap_iterator(range(iteration), progressbar):
        # Create feed dictionary for next mini batch
        if not is_multitask:
            inputs, labels_true, inputs_seq_len, _ = mini_batch.__next__()
        else:
            if is_main:
                inputs, labels_true, _, inputs_seq_len, _ = mini_batch.__next__()
            else:
                inputs, _, labels_true, inputs_seq_len, _ = mini_batch.__next__()

        feed_dict = {
            network.inputs: inputs,
            network.inputs_seq_len: inputs_seq_len,
            network.keep_prob_input: 1.0,
            network.keep_prob_hidden: 1.0
        }

        batch_size_each = len(inputs_seq_len)

        labels_pred_st = session.run(decode_op, feed_dict=feed_dict)
        labels_pred = sparsetensor2list(labels_pred_st, batch_size_each)

        for i_batch in range(batch_size_each):
            # Convert from list to string
            if label_type != 'phone' and is_test:
                str_true = ''.join(labels_true[i_batch])
                # NOTE: 漢字とかなの場合はテストデータのラベルはそのまま保存してある
            else:
                str_true = num2char(labels_true[i_batch], map_file_path)
            str_pred = num2char(labels_pred[i_batch], map_file_path)

            # Remove silence(_) & noise(NZ) labels
            str_true = re.sub(r'[_NZー・]+', "", str_true)
            str_pred = re.sub(r'[_NZー・]+', "", str_pred)

            # Compute edit distance
            cer_each = Levenshtein.distance(
                str_pred, str_true) / len(list(str_true))

            cer_sum += cer_each

    cer_mean = cer_sum / dataset.data_num

    return cer_mean
