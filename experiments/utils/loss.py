#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def plot_loss(steps, train_loss, dev_loss, save_path):
    """Plot losses.
    Args:
        steps:
        train_loss:
        dev_loss:
        save_path:
    """
    plt.plot(steps, train_loss, "b", label="Training Loss")
    plt.plot(steps, dev_loss, "r", label="Dev Loss")
    # plt.vlines(steps[-1], 0, train_loss[-1], color='0.75')
    # plt.hlines(train_loss[-1], 0, steps[-1], color='0.75')
    plt.legend(loc="upper right", fontsize=12)
    plt.show()
    plt.savefig(os.path.join(save_path, "loss.png"), dvi=500)


def save_loss(steps, train_loss, dev_loss, save_path):
    loss_graph = np.column_stack((steps, train_loss, dev_loss))
    np.savetxt(os.path.join(save_path, "loss.csv"), loss_graph, delimiter=",")
