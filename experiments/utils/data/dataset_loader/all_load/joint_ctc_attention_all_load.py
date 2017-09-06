#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for laoding dataset for the Jont CTC-Attention model.
   In this class, all data will be loaded at once.
   You can use only the single GPU version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import basename
import random
import numpy as np

from experiments.utils.data.inputs.splicing import do_splice


class DatasetBase(object):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        return (self.input_list[index], self.label_list[index])

    def next(self):
        # For python2
        return self.__next__()

    def reset(self):
        """Reset data counter. This is useful when you'd like to evaluate
        overall data during training.
        """
        self.rest = set(range(0, len(self), 1))

    def __next__(self, batch_size=None):
        """Generate each mini-batch.
        Args:
            batch_size (int, option): the size of mini-batch
        Returns:
            A tuple of `(inputs, labels, inputs_seq_len, labels_seq_len, input_names)`
                inputs: list of input data of size `[B, T, input_dim]`
                att_labels: list of target labels for Attention, of size `[B, T]`
                ctc_labels: list of target labels for CTC, of size `[B, T]`
                inputs_seq_len: list of length of inputs of size `[B]`
                att_labels_seq_len: list of length of target labels for Attention, of size `[B]`
                input_names: list of file name of input data of size `[B]`
            next_epoch_flag: If true, one epoch is finished
        """
        if batch_size is None:
            batch_size = self.batch_size

        next_epoch_flag = False
        self.ctc_padded_value = -1
        self.att_padded_value = self.eos_index

        while True:
            if next_epoch_flag:
                next_epoch_flag = False

            # Sort all uttrances in each epoch
            if self.sort_utt:
                if len(self.rest) > batch_size:
                    data_indices = list(self.rest)[:batch_size]
                    self.rest -= set(data_indices)
                else:
                    # Last mini-batch
                    data_indices = list(self.rest)
                    self.rest = set(range(0, len(self), 1))
                    next_epoch_flag = True
                    if self.is_training:
                        print('---Next epoch---')
                    self.epoch += 1
                    if self.epoch == self.sort_stop_epoch:
                        self.sort_utt = False

                # Shuffle selected mini-batch
                random.shuffle(data_indices)

            else:
                if len(self.rest) > batch_size:
                    # Randomly sample mini-batch
                    data_indices = random.sample(list(self.rest), batch_size)
                    self.rest -= set(data_indices)
                else:
                    # Last mini-batch
                    data_indices = list(self.rest)
                    self.rest = set(range(0, len(self), 1))
                    next_epoch_flag = True
                    if self.is_training:
                        print('---Next epoch---')

                    # Shuffle selected mini-batch
                    random.shuffle(data_indices)

            # Compute max frame num in mini-batch
            max_frame_num = max(map(lambda x: x.shape[0],
                                    self.input_list[data_indices]))

            # Compute max target label length in mini-batch
            att_max_seq_len = max(map(len,
                                      self.att_label_list[data_indices]))
            ctc_max_seq_len = max(map(len,
                                      self.ctc_label_list[data_indices]))

            # Initialization
            inputs = np.zeros(
                (len(data_indices), max_frame_num,
                 self.input_list[0].shape[-1] * self.splice),
                dtype=np.int32)
            att_labels = np.array([[self.att_padded_value] * att_max_seq_len]
                                  * len(data_indices), dtype=np.int32)
            ctc_labels = np.array([[self.ctc_padded_value] * ctc_max_seq_len]
                                  * len(data_indices), dtype=np.int32)
            inputs_seq_len = np.zeros((len(data_indices),), dtype=np.int32)
            att_labels_seq_len = np.zeros(
                (len(data_indices),), dtype=np.int32)
            input_names = np.array(list(
                map(lambda path: basename(path).split('.')[0],
                    np.take(self.input_paths, data_indices, axis=0))))

            # Set values of each data in mini-batch
            for i_batch, x in enumerate(data_indices):
                data_i = self.input_list[x]
                frame_num, input_size = data_i.shape

                # Splicing
                data_i = data_i.reshape(1, frame_num, input_size)
                data_i = do_splice(data_i,
                                   splice=self.splice,
                                   batch_size=1).reshape(frame_num, -1)

                inputs[i_batch, :frame_num, :] = data_i
                att_labels[i_batch, :len(self.att_label_list[x])
                           ] = self.att_label_list[x]
                ctc_labels[i_batch, :len(
                    self.ctc_label_list[x])] = self.ctc_label_list[x]
                inputs_seq_len[i_batch] = frame_num
                att_labels_seq_len[i_batch] = len(self.att_label_list[x])

            return (inputs, att_labels, ctc_labels, inputs_seq_len,
                    att_labels_seq_len, input_names), next_epoch_flag
