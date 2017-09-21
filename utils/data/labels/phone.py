#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def phone2idx(phone_list, map_file_path):
    """Convert from phone to index.
    Args:
        phone_list (list): phones whose element is string
        map_file_path: path to the mapping file
    Returns:
        phone_list (list): phone indices
    """
    # Read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split('  ')
            map_dict[str(line[0])] = int(line[1])

    # Convert from phone to index
    for i in range(len(phone_list)):
        phone_list[i] = map_dict[phone_list[i]]

    return np.array(phone_list)


def idx2phone(index_list, map_file_path, padded_value=-1):
    """Convert from index to phone.
    Args:
        index_list (list): phone indices
        map_file_path (string): path to the mapping file
        padded_value (int): the value used for padding
    Returns:
        str_phone (string): a sequence of phones
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

    # Convert from indices to the corresponding phones
    phone_list = list(map(lambda x: map_dict[x], index_list))
    str_phone = ' '.join(phone_list)

    return str_phone
