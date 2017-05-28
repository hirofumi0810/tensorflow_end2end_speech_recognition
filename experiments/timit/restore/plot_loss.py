#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

sys.path.append('../../')
from utils.loss import plot_loss


def main(model_path):
    # load train & dev loss
    data = np.loadtxt(os.path.join(model_path, "loss.csv"),
                      delimiter=",")
    steps = data[:, 0]
    train_loss = data[:, 1]
    dev_loss = data[:, 2]
    plot_loss(steps, train_loss, dev_loss, model_path)


if __name__ == '__main__':

    args = sys.argv
    if len(args) != 2:
        raise ValueError(("Set a path to saved model.\n"
                          "Usase: python plot_loss.py path_to_saved_model"))

    main(model_path=args[1])
