#! /usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm


def wrap_iterator(iterator, is_progressbar):
    if is_progressbar:
        iterator = tqdm(iterator)
    return iterator
