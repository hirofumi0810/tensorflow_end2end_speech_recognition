#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for laoding dataset for the Jont CTC-Attention model.
   You can use the multi-GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import random
import numpy as np
import tensorflow as tf


class DatasetBase(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def next_batch(self, batch_size=None, session=None):
        """Make mini-batch.
        Args:
            batch_size: int, the size of mini-batch
            session:
        Returns:
            inputs: list of input data, size `[batch_size]`
            att_labels: list of target labels, size `[batch_size]`
            ctc_labels_st: list of SparseTensor of taret labels
            inputs_seq_len: list of length of inputs of size `[batch_size]`
            att_labels_seq_len: list of length of target labels of size
                `[batch_size]`
            input_names: list of file name of input data of size `[batch_size]`

            If num_gpu > 1, each return is divide into list of size `[num_gpu]`.
        """
        if session is None and self.num_gpu != 1:
            raise ValueError('Set session when using multiple GPUs.')

        if batch_size is None:
            batch_size = self.batch_size

        next_epoch_flag = False
        ctc_padded_value = -1

        while True:
            # Sort all uttrances in each epoch
            if self.sort_utt or self.sorta_grad:
                if len(self.rest) > batch_size:
                    data_indices = list(self.rest)[:batch_size]
                    self.rest -= set(data_indices)
                else:
                    data_indices = list(self.rest)
                    self.rest = set(range(0, self.data_num, 1))
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')
                    if self.sorta_grad:
                        self.sorta_grad = False

                # Shuffle selected mini-batch
                if not self.sorta_grad:
                    random.shuffle(data_indices)

            else:
                if len(self.rest) > batch_size:
                    # Randomly sample mini-batch
                    data_indices = random.sample(list(self.rest), batch_size)
                    self.rest -= set(data_indices)
                else:
                    data_indices = list(self.rest)
                    self.rest = set(range(0, self.data_num, 1))
                    next_epoch_flag = True
                    if self.data_type == 'train':
                        print('---Next epoch---')

                    # Shuffle selected mini-batch
                    random.shuffle(data_indices)

            # Compute max frame num in mini-batch
            max_frame_num = max(map(lambda x: x.shape[0],
                                    self.input_list[data_indices]))

            # Compute max target label length in mini-batch
            att_max_seq_len = max(
                map(len, self.att_label_list[data_indices]))
            ctc_max_seq_len = max(
                map(len, self.ctc_label_list[data_indices]))

            # Initialization
            inputs = np.zeros(
                (len(data_indices), max_frame_num, self.input_size),
                dtype=np.int32)
            # Padding with <EOS>
            att_labels = np.array([[self.eos_index] * att_max_seq_len]
                                  * len(data_indices), dtype=np.int32)
            ctc_labels = np.array([[ctc_padded_value] * ctc_max_seq_len]
                                  * len(data_indices), dtype=np.int32)
            inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
            att_labels_seq_len = np.zeros(
                (len(data_indices),), dtype=np.int32)
            input_names = list(
                map(lambda path: basename(path).split('.')[0],
                    np.take(self.input_paths, data_indices, axis=0)))

            # Set values of each data in mini-batch
            for i_batch, x in enumerate(data_indices):
                data_i = self.input_list[x]
                frame_num = data_i.shape[0]
                inputs[i_batch, :frame_num, :] = data_i
                att_labels[i_batch, :len(self.att_label_list[x])
                           ] = self.att_label_list[x]
                ctc_labels[i_batch, :len(
                    self.ctc_label_list[x])] = self.ctc_label_list[x]
                inputs_seq_len[i_batch] = frame_num
                att_labels_seq_len[i_batch] = len(self.att_label_list[x])

            ##########
            # GPU
            ##########
            if self.num_gpu > 1:
                divide_num = self.num_gpu
                if next_epoch_flag:
                    for i in range(self.num_gpu, 0, -1):
                        if len(self.rest) % i == 0:
                            divide_num = i
                            break
                    next_epoch_flag = False

                # Now we split the mini-batch data by num_gpu
                inputs = tf.split(inputs, divide_num, axis=0)
                att_labels = tf.split(att_labels, divide_num, axis=0)
                ctc_labels = tf.split(ctc_labels, divide_num, axis=0)
                inputs_seq_len = tf.split(inputs_seq_len, divide_num, axis=0)
                att_labels_seq_len = tf.split(
                    att_labels_seq_len, divide_num, axis=0)
                input_names = tf.split(input_names, divide_num, axis=0)

                # Convert from SparseTensor to numpy.ndarray
                inputs = list(map(session.run, inputs))
                att_labels = list(map(session.run, att_labels))
                ctc_labels = list(map(session.run, ctc_labels))
                inputs_seq_len = list(map(session.run, inputs_seq_len))
                att_labels_seq_len = list(map(session.run, att_labels_seq_len))
                input_names = list(map(session.run, input_names))

            yield (inputs, att_labels, ctc_labels, inputs_seq_len,
                   att_labels_seq_len, input_names)
