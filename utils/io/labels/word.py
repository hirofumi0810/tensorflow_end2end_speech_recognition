#! /usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def idx2word(index_list, map_file_path, padded_value=-1):
    """Convert from index to word.
    Args:
        index_list (np.ndarray): list of word indices. batch_size == 1 is
            expected.
        map_file_path (string): path to the mapping file
        padded_value (int): the value used for padding
    Returns:
        word_list: list of words
    """
    # Read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    # Remove padded values
    assert type(index_list) == np.ndarray, 'index_list should be np.ndarray.'
    index_list = np.delete(index_list, np.where(index_list == -1), axis=0)

    # Convert from indices to the corresponding words
    word_list = list(map(lambda x: map_dict[x], index_list))

    return word_list
