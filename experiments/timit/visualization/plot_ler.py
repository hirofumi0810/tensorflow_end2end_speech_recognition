#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

blue = '#4682B4'
orange = '#D2691E'


def main(model_path):
    # Load train & dev ler
    data = np.loadtxt(os.path.join(model_path, "ler.csv"),
                      delimiter=",")

    # Plot per 100 steps
    steps = data[:, 0] * 100
    train_loss = data[:, 1]
    dev_loss = data[:, 2]

    # Plot
    plt.plot(steps, train_loss, blue, label="Training LER")
    plt.plot(steps, dev_loss, orange, label="Dev LER")
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('LER', fontsize=12)
    plt.legend(loc="upper right", fontsize=12)
    plt.savefig(os.path.join(model_path, "ler.png"), dvi=500)
    plt.show()


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError(("Set a path to saved model.\n"
                          "Usase: python plot_loss.py path_to_saved_model"))
    main(model_path=args[1])
