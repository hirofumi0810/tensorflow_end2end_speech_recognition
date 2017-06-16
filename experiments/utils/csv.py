#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


def save_loss(steps, loss_train, loss_dev, save_path):
    loss_graph = np.column_stack((steps, loss_train, loss_dev))
    np.savetxt(os.path.join(save_path, "loss.csv"), loss_graph, delimiter=",")


def save_ler(steps, ler_train, ler_dev, save_path):
    loss_graph = np.column_stack((steps, ler_train, ler_dev))
    np.savetxt(os.path.join(save_path, "ler.csv"), loss_graph, delimiter=",")
