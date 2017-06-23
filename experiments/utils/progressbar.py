#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm


def wrap_iterator(iterator, is_progressbar):
    if is_progressbar:
        iterator = tqdm(iterator)
    return iterator
