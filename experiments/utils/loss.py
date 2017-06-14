#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


def save_loss(steps, train_loss, dev_loss, save_path):
    loss_graph = np.column_stack((steps, train_loss, dev_loss))
    np.savetxt(os.path.join(save_path, "loss.csv"), loss_graph, delimiter=",")
