#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for phone mapping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def map_to_39phone(phone_list, label_type, map_file_path):
    """Map from 61 or 48 phones to 39 phones.
    Args:
        phone_list: list of phones (string)
        label_type: phone48 or phone61
        map_file_path: path to the mapping file
    Returns:
        phone_list: list of 39 phones (string)
    """
    if label_type == 'phone39':
        return phone_list

    # Read a mapping file
    map_dict = {}
    with open(map_file_path) as f:
        for line in f:
            line = line.strip().split()
            if label_type == 'phone61':
                if line[1] != 'nan':
                    map_dict[line[0]] = line[2]
                else:
                    map_dict[line[0]] = ''
            elif label_type == 'phone48':
                if line[1] != 'nan':
                    map_dict[line[1]] = line[2]

    # Map to 39 phones
    for i in range(len(phone_list)):
        phone_list[i] = map_dict[phone_list[i]]

    # Ignore q (only if 61 phones)
    while '' in phone_list:
        phone_list.remove('')

    return phone_list
