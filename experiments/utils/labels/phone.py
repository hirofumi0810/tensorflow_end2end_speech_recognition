#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def phone2num(phone_list, map_file_path):
    """Convert from phone to number.
    Args:
        phone_list: list of phones (string)
        map_file_path: path to the mapping file
    Returns:
        phone_list: list of phone indices (int)
    """
    # Read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split('  ')
            map_dict[str(line[0])] = int(line[1])

    # Convert from phone to number
    for i in range(len(phone_list)):
        phone_list[i] = map_dict[phone_list[i]]

    return np.array(phone_list)


def num2phone(num_list, map_file_path, padded_value=-1):
    """Convert from number to phone.
    Args:
        num_list: list of phone indices
        map_file_path: path to the mapping file
        padded_value: int, the value used for padding
    Returns:
        str_phone: string of phones
    """
    # Read mapping file
    map_dict = {}
    with open(map_file_path, 'r') as f:
        for line in f:
            line = line.strip().split()
            map_dict[int(line[1])] = line[0]

    # Remove padded values
    assert type(num_list) == np.ndarray, 'num_list should be np.ndarray.'
    num_list = np.delete(num_list, np.where(num_list == -1), axis=0)

    # Convert from indices to the corresponding phones
    phone_list = list(map(lambda x: map_dict[x], num_list))
    str_phone = ' '.join(phone_list)

    return str_phone
